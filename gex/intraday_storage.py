"""Date-partitioned parquet storage for intraday option chain snapshots.

Layout::

    data/intraday_options/
        date=2026-04-24/
            093000.parquet
            094500.parquet
            ...

Each file is one snapshot (~100K rows compressed to ~1-2 MB).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from gex.intraday_collector import SnapshotResult

DEFAULT_ROOT = "data/intraday_options"


class IntradayStorage:
    """Write and read date-partitioned intraday option snapshots."""

    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = Path(root)

    # ── Write ─────────────────────────────────────────────────────

    def save_snapshot(self, result: SnapshotResult) -> Path:
        """Persist a single snapshot to disk.

        Returns the path of the written parquet file.
        """
        ts = result.timestamp
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H%M%S")

        partition_dir = self.root / f"date={date_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        df = result.chain.copy()
        df["underlying_price"] = result.underlying_price

        out_path = partition_dir / f"{time_str}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow")
        return out_path

    # ── Read ──────────────────────────────────────────────────────

    def load_date(self, date: str) -> pd.DataFrame:
        """Load all snapshots for a single date.

        Args:
            date: ``YYYY-MM-DD`` string.

        Returns:
            Concatenated DataFrame of all snapshots for the date, sorted by
            timestamp.  Empty DataFrame if no data found.
        """
        partition_dir = self.root / f"date={date}"
        if not partition_dir.exists():
            return pd.DataFrame()

        dfs = []
        for f in sorted(partition_dir.glob("*.parquet")):
            dfs.append(pd.read_parquet(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    def load_range(
        self,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Load all snapshots across a date range (inclusive).

        Args:
            start: ``YYYY-MM-DD`` start date.
            end: ``YYYY-MM-DD`` end date.

        Returns:
            Concatenated DataFrame sorted by timestamp.
        """
        dates = pd.date_range(start, end, freq="B")  # business days
        dfs = []
        for d in dates:
            day_df = self.load_date(d.strftime("%Y-%m-%d"))
            if not day_df.empty:
                dfs.append(day_df)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    def available_dates(self) -> list[str]:
        """List all dates that have stored data."""
        if not self.root.exists():
            return []
        dates = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and d.name.startswith("date="):
                dates.append(d.name.split("=")[1])
        return dates
