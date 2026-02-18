"""Multi-schema CSV loader for minute-level instrument data."""
import pandas as pd
import numpy as np
from pathlib import Path

from strategies.swing.config import InstrumentConfig, INSTRUMENTS


class InstrumentLoader:
    """Loads minute-bar data from various CSV schemas into a standard format.

    Output columns: open, high, low, close, volume, trading_day
    Index: DatetimeIndex in US/Eastern
    """

    def load(self, config: InstrumentConfig) -> pd.DataFrame:
        parsers = {
            "combined": self._parse_combined,
            "mnt_standard": self._parse_mnt_standard,
        }
        parser = parsers.get(config.schema)
        if parser is None:
            raise ValueError(f"Unknown schema: {config.schema}")

        df = parser(config.path)
        df = self._assign_trading_day(df, config.session_start_hour)
        return df

    def _parse_combined(self, path: Path) -> pd.DataFrame:
        """Parse es_min_combined.csv: datetime,open,high,low,close,volume."""
        df = pd.read_csv(path, parse_dates=["datetime"])
        df = df.set_index("datetime")
        df.index = df.index.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
        df = df[df.index.notna()]
        return df[["open", "high", "low", "close", "volume"]]

    def _parse_mnt_standard(self, path: Path) -> pd.DataFrame:
        """Parse NQ/ZN style: BOM header, DateTime with AM/PM, trailing comma."""
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip().str.rstrip(",")
        # Drop empty trailing column from trailing comma
        df = df.loc[:, ~df.columns.duplicated()]
        if "" in df.columns:
            df = df.drop(columns=[""])

        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime")
        df.index = df.index.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
        df = df[df.index.notna()]
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        return df[["open", "high", "low", "close", "volume"]]

    def _assign_trading_day(
        self, df: pd.DataFrame, session_start_hour: int = 18
    ) -> pd.DataFrame:
        """Assign trading_day: bars at/after 6PM ET belong to next calendar day's session."""
        dates = df.index.normalize()
        hours = df.index.hour
        # Bars at or after session_start_hour belong to next day
        offset = pd.to_timedelta((hours >= session_start_hour).astype(int), unit="D")
        df["trading_day"] = (dates + offset).date
        return df


def load_instruments(
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load multiple instruments, return dict of symbol → minute DataFrame."""
    if symbols is None:
        symbols = list(INSTRUMENTS.keys())

    loader = InstrumentLoader()
    result = {}
    for sym in symbols:
        cfg = INSTRUMENTS[sym]
        if not cfg.path.exists():
            print(f"  WARNING: {sym} data not found at {cfg.path}, skipping")
            continue
        print(f"  Loading {sym} from {cfg.path}...")
        df = loader.load(cfg)
        print(f"    {len(df):,} bars, {df.index.min()} – {df.index.max()}")
        result[sym] = df
    return result
