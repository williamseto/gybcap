"""Data source protocol and MySQL implementation."""

import datetime
import zoneinfo
from typing import Protocol

import pandas as pd
from sqlalchemy import create_engine

from strategies.realtime.config import DatabaseConfig
from strategies.realtime.orderflow_columns import normalize_orderflow_columns


# ---------------------------------------------------------------------------
# Trading day boundary helpers
# ---------------------------------------------------------------------------

def get_trading_day_start_ts(now_ts: int) -> int:
    """
    Return UTC timestamp of the current trading day's start.

    A trading day starts at 15:00 America/Los_Angeles.  If *now_ts* is
    before 15:00 LA time, the start belongs to the previous calendar day.
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt_local = datetime.datetime.fromtimestamp(now_ts, la)

    if dt_local.hour >= 15:
        day_start_local = dt_local.replace(hour=15, minute=0, second=0, microsecond=0)
    else:
        prev = dt_local.date() - datetime.timedelta(days=1)
        day_start_local = datetime.datetime(
            prev.year, prev.month, prev.day, 15, 0, 0, tzinfo=la,
        )

    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())


def get_prev_trading_day_start_ts(now_ts: int) -> int:
    """
    Return UTC timestamp of the *previous* trading day's start.

    Handles weekends by skipping back past Friday's 15:00.
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    dt_local = datetime.datetime.fromtimestamp(now_ts, la)

    # Resolve current trading day start date first.
    if dt_local.hour >= 15:
        current_start_date = dt_local.date()
    else:
        current_start_date = dt_local.date() - datetime.timedelta(days=1)

    # Previous trading day start is one business day before the current start.
    prev = current_start_date - datetime.timedelta(days=1)
    while prev.weekday() >= 5:  # Sat/Sun
        prev -= datetime.timedelta(days=1)

    day_start_local = datetime.datetime(
        prev.year, prev.month, prev.day, 15, 0, 0, tzinfo=la,
    )
    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())


def get_session_window_for_trading_day(trading_day: str) -> tuple[int, int]:
    """Return epoch-second [start, end] bounds for a trading_day label.

    `trading_day` follows project convention: session rolls at 15:00
    America/Los_Angeles.
    """
    la = zoneinfo.ZoneInfo("America/Los_Angeles")
    day = datetime.date.fromisoformat(trading_day)
    prev = day - datetime.timedelta(days=1)
    start_local = datetime.datetime(
        prev.year, prev.month, prev.day, 15, 0, 0, tzinfo=la,
    )
    end_local = start_local + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
    return (
        int(start_local.astimezone(datetime.timezone.utc).timestamp()),
        int(end_local.astimezone(datetime.timezone.utc).timestamp()),
    )


# ---------------------------------------------------------------------------
# DataSource protocol
# ---------------------------------------------------------------------------

class DataSource(Protocol):
    """Protocol for fetching tick data."""

    def fetch_range(self, start_ts: int, end_ts: int) -> pd.DataFrame: ...
    def fetch_since(self, since_ts: int, end_ts: int) -> pd.DataFrame: ...
    def fetch_prev_day(self, now_ts: int) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# MySQL implementation
# ---------------------------------------------------------------------------

class MySQLSource:
    """Fetch tick data from a MySQL ``price_data`` table."""

    def __init__(self, config: DatabaseConfig):
        self._config = config
        self._engine = create_engine(config.connection_string)

    def fetch_range(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM {self._config.table} "
            "WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self._engine,
            params=(self._config.symbol, start_ts, end_ts),
        )
        return normalize_orderflow_columns(df, copy=False)

    def fetch_since(self, since_ts: int, end_ts: int) -> pd.DataFrame:
        df = pd.read_sql_query(
            f"SELECT * FROM {self._config.table} "
            "WHERE symbol = %s AND timestamp > %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self._engine,
            params=(self._config.symbol, since_ts, end_ts),
        )
        return normalize_orderflow_columns(df, copy=False)

    def fetch_range_bounds(self, start_ts: int, end_ts: int) -> tuple[int, int]:
        """Return (min_ts, max_ts) present in [start_ts, end_ts] for configured symbol."""
        df = pd.read_sql_query(
            f"SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts "
            f"FROM {self._config.table} "
            "WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s",
            self._engine,
            params=(self._config.symbol, start_ts, end_ts),
        )
        if df.empty:
            return (0, 0)
        min_ts = df.iloc[0].get("min_ts")
        max_ts = df.iloc[0].get("max_ts")
        if pd.isna(min_ts) or pd.isna(max_ts):
            return (0, 0)
        return (int(min_ts), int(max_ts))

    def fetch_prev_day(self, now_ts: int) -> pd.DataFrame:
        prev_start = get_prev_trading_day_start_ts(now_ts)
        prev_end = prev_start + 22 * 3600
        return self.fetch_range(prev_start, prev_end)
