"""Data source protocol and MySQL implementation."""

import datetime
import zoneinfo
import time
from typing import Protocol

import pandas as pd
from sqlalchemy import create_engine

from strategies.realtime.config import DatabaseConfig


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

    day_delta = 1 if dt_local.hour >= 15 else 2

    # Skip weekends (past Thursday 15:00 boundary)
    weekday_diff = dt_local.weekday() - 4
    if weekday_diff >= 0:
        day_delta += weekday_diff

    prev = dt_local.date() - datetime.timedelta(days=day_delta)
    day_start_local = datetime.datetime(
        prev.year, prev.month, prev.day, 15, 0, 0, tzinfo=la,
    )
    return int(day_start_local.astimezone(datetime.timezone.utc).timestamp())


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
        return pd.read_sql_query(
            f"SELECT * FROM {self._config.table} "
            "WHERE symbol = %s AND timestamp >= %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self._engine,
            params=(self._config.symbol, start_ts, end_ts),
        )

    def fetch_since(self, since_ts: int, end_ts: int) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT * FROM {self._config.table} "
            "WHERE symbol = %s AND timestamp > %s AND timestamp <= %s "
            "ORDER BY timestamp ASC",
            self._engine,
            params=(self._config.symbol, since_ts, end_ts),
        )

    def fetch_prev_day(self, now_ts: int) -> pd.DataFrame:
        prev_start = get_prev_trading_day_start_ts(now_ts)
        prev_end = prev_start + 22 * 3600
        return self.fetch_range(prev_start, prev_end)
