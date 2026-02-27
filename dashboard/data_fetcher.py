"""yfinance daily top-up + external CSV patching for dashboard data pipeline."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
ROOT = Path(__file__).resolve().parent.parent

# yfinance symbol → cache parquet filename
YF_SYMBOLS = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "ZN": "ZN=F",
}

# yfinance symbol → raw_data CSV path (external CSVs read by external_daily.py)
YF_EXTERNAL = {
    "^VIX":     ROOT / "raw_data" / "VIX_History.csv",
    "^VIX3M":   ROOT / "raw_data" / "VIX3M_daily.csv",
    "DX-Y.NYB": ROOT / "raw_data" / "DXY_daily.csv",
    "^TNX":     ROOT / "raw_data" / "TNX_daily.csv",
    "^IRX":     ROOT / "raw_data" / "IRX_daily.csv",
    "^SKEW":    ROOT / "raw_data" / "SKEW_daily.csv",
    "HYG":      ROOT / "raw_data" / "HYG_daily.csv",
    "LQD":      ROOT / "raw_data" / "LQD_daily.csv",
}

# Columns expected in the parquet cache
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class DailyDataFetcher:
    """Manages daily OHLCV caches for ES/NQ/ZN using yfinance top-up.

    Two modes:
      csv_plus_yfinance (default): bootstraps from local minute CSVs,
        then tops up from yfinance for new bars.
      yfinance_only: fetches full history from yfinance (cloud deploy).
    """

    def __init__(self, data_mode: str = "csv_plus_yfinance"):
        self.data_mode = data_mode
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch_and_update(self, symbol: str) -> pd.DataFrame:
        """Return up-to-date daily OHLCV for symbol (ES/NQ/ZN).

        Loads parquet cache → finds last date → fetches new bars from yfinance
        → deduplicates → saves → returns combined DataFrame.

        VP columns (vp_poc_rel etc.) are NaN for yfinance-sourced rows.
        """
        cache_path = CACHE_DIR / f"daily_cache_{symbol}.parquet"

        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
            last_date = cached.index.max().date()
            today = date.today()
            if last_date >= today:
                logger.info("%s cache is current (%s)", symbol, last_date)
                return cached
            fetch_start = last_date + timedelta(days=1)
        else:
            cached = None
            # Bootstrap: try CSV first (csv_plus_yfinance mode), else full yfinance
            if self.data_mode == "csv_plus_yfinance":
                bootstrapped = self._bootstrap_from_csv(symbol)
                if bootstrapped is not None:
                    cached = bootstrapped
                    last_date = cached.index.max().date()
                    fetch_start = last_date + timedelta(days=1)
                else:
                    fetch_start = None  # full history
            else:
                fetch_start = None  # full history from yfinance

        # Fetch new bars from yfinance
        yf_df = self._fetch_yfinance(symbol, fetch_start)

        if yf_df is not None and len(yf_df) > 0:
            if cached is not None:
                combined = pd.concat([cached, yf_df])
                # Keep yfinance rows on overlap so corrected bars replace stale cache rows.
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = yf_df
            combined.to_parquet(cache_path)
            logger.info("%s: updated cache → %d days through %s",
                        symbol, len(combined), combined.index.max().date())
            return combined
        elif cached is not None:
            return cached
        else:
            raise RuntimeError(f"No data available for {symbol}")

    def topup_external_csvs(self) -> None:
        """Append today's yfinance close to each external raw_data/*.csv.

        This lets external_daily.py read updated data without code changes.
        """
        today = date.today()
        for yf_sym, csv_path in YF_EXTERNAL.items():
            try:
                self._topup_one_external(yf_sym, csv_path, today)
            except Exception as e:
                logger.warning("Failed to top-up %s → %s: %s", yf_sym, csv_path.name, e)

    # ------------------------------------------------------------------ #
    # Bootstrap from local minute CSV
    # ------------------------------------------------------------------ #

    def _bootstrap_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load daily bars from the local minute CSV via InstrumentLoader/DailyAggregator."""
        try:
            from strategies.swing.config import INSTRUMENTS
            from strategies.swing.data_loader import InstrumentLoader
            from strategies.swing.daily_aggregator import DailyAggregator

            cfg = INSTRUMENTS.get(symbol)
            if cfg is None or not cfg.path.exists():
                logger.info("%s: no local CSV found at %s", symbol, cfg.path if cfg else "?")
                return None

            logger.info("%s: bootstrapping from %s...", symbol, cfg.path)
            loader = InstrumentLoader()
            minute_df = loader.load(cfg)
            aggregator = DailyAggregator()
            compute_vp = (symbol == "ES" and "volume" in minute_df.columns)
            daily = aggregator.aggregate(minute_df, compute_vp=compute_vp)
            logger.info("%s: bootstrapped %d trading days", symbol, len(daily))
            return daily

        except Exception as e:
            logger.warning("%s bootstrap failed: %s", symbol, e)
            return None

    # ------------------------------------------------------------------ #
    # yfinance fetch
    # ------------------------------------------------------------------ #

    def _fetch_yfinance(
        self,
        symbol: str,
        start: Optional[date],
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV from yfinance, return normalized DataFrame."""
        yf_sym = YF_SYMBOLS.get(symbol, symbol)
        start_str = start.isoformat() if start else "2005-01-01"
        logger.info("Fetching %s (%s) from yfinance since %s...", symbol, yf_sym, start_str)

        raw = self._download_yfinance(yf_sym, start_str)

        if raw is None or len(raw) == 0:
            logger.warning("yfinance returned empty data for %s", yf_sym)
            return None

        # Flatten multi-level columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = pd.DataFrame(index=raw.index)
        col_map = {c.lower(): c for c in raw.columns}
        for col in OHLCV_COLS:
            src = col_map.get(col)
            if src is not None:
                df[col] = raw[src]
            else:
                df[col] = np.nan

        df.index = pd.to_datetime(df.index).normalize()
        df = df.dropna(subset=["close"])
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------ #
    # External CSV top-up
    # ------------------------------------------------------------------ #

    def _topup_one_external(
        self,
        yf_sym: str,
        csv_path: Path,
        today: date,
    ) -> None:
        """Fetch today's bar for yf_sym and append to csv_path if needed."""
        existing = None
        fetch_start: Optional[date] = None

        # Check whether CSV is current; otherwise fetch from the first missing date.
        if csv_path.exists():
            existing = self._read_external_csv(csv_path)
            if existing is not None and len(existing) > 0:
                last = existing.index.max()
                if hasattr(last, "date"):
                    last = last.date()
                if last >= today:
                    return  # already current
                fetch_start = last + timedelta(days=1)
            else:
                logger.warning("Could not parse existing %s; rebuilding from yfinance", csv_path.name)
                fetch_start = None

        start_str = fetch_start.isoformat() if fetch_start else "2005-01-01"
        logger.info("Top-up %s from yfinance since %s", csv_path.name, start_str)
        raw = self._download_yfinance(yf_sym, start_str)

        if raw is None or len(raw) == 0:
            return

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Normalize to standard columns
        col_map = {c.lower(): c for c in raw.columns}
        new_rows = pd.DataFrame(index=raw.index)
        for col in ["open", "high", "low", "close"]:
            src = col_map.get(col)
            new_rows[col] = raw[src] if src else np.nan
        new_rows = new_rows.dropna(subset=["close"])
        new_rows.index = pd.to_datetime(new_rows.index).normalize()

        if csv_path.exists():
            existing = self._read_external_csv(csv_path)
            if existing is not None:
                # Combine and dedup
                combined = pd.concat([existing, new_rows])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = new_rows
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            combined = new_rows

        # Write in the format that external_daily.py can read (date, open, high, low, close)
        out = combined.reset_index()
        out.columns = [c.lower() for c in out.columns]
        # Rename index column (could be 'datetime' or 'date' or 'Datetime')
        if "datetime" in out.columns:
            out = out.rename(columns={"datetime": "date"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "date"})

        # VIX uses MM/DD/YYYY format — detect from existing file
        if csv_path.name == "VIX_History.csv":
            out["DATE"] = out["date"].dt.strftime("%m/%d/%Y")
            out = out.rename(columns={
                "open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE"
            })
            out[["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]].to_csv(csv_path, index=False)
        else:
            out[["date", "open", "high", "low", "close"]].to_csv(csv_path, index=False)

        logger.info("Updated %s through %s", csv_path.name, combined.index.max().date())

    def _read_external_csv(self, path: Path) -> Optional[pd.DataFrame]:
        """Read an external CSV in either VIX format (DATE) or standard (date)."""
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]

            date_col = next((c for c in df.columns if c.lower() == "date"), None)
            if date_col is None:
                return None

            df[date_col] = pd.to_datetime(df[date_col], dayfirst=False, errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
            df.columns = [c.lower() for c in df.columns]

            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "close" in df.columns:
                df = df.dropna(subset=["close"])
            return df
        except Exception:
            return None

    def _download_yfinance(self, yf_symbol: str, start_str: str) -> Optional[pd.DataFrame]:
        """Thin wrapper around yfinance download to keep fetch logic testable."""
        import yfinance as yf

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return yf.download(
                    yf_symbol,
                    start=start_str,
                    progress=False,
                    auto_adjust=True,
                )
        except Exception as e:
            logger.warning("yfinance download failed for %s: %s", yf_symbol, e)
            return None
