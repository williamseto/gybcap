"""DataSource backed by a CSV file of 1-min bars.

Serves two roles:
- Playback: feeds 1-min bars as synthetic ticks through the engine
- Historical fallback: provides multi-day lookback for feature warm-up

Each 1-min bar is emitted as a single tick at the bar's timestamp with
price=close, volume=volume.  This is the minimal representation that
reconstructs the same 1-min bars through BarAggregator.
"""

import numpy as np
import pandas as pd

from strategies.realtime.data_source import (
    get_prev_trading_day_start_ts,
)
from strategies.realtime.orderflow_columns import normalize_orderflow_columns


class CSVDataSource:
    """DataSource backed by a CSV file of 1-min bars."""

    def __init__(self, csv_path: str, upsample: bool = False):
        self._df = self._load_csv(csv_path)
        self._upsample = upsample

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Build datetime column
        if 'Date' in df.columns and 'Time' in df.columns:
            df['dt'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
            )
        elif 'dt' not in df.columns:
            raise ValueError("CSV must have 'Date'+'Time' or 'dt' column")

        df.columns = df.columns.str.lower()

        # Localize to LA timezone and compute unix timestamp
        if df['dt'].dt.tz is None:
            df['dt'] = df['dt'].dt.tz_localize('America/Los_Angeles')
        df['timestamp'] = df['dt'].astype(np.int64) // 10**9

        df = normalize_orderflow_columns(df, copy=False)

        return df

    # ------------------------------------------------------------------
    # DataSource protocol
    # ------------------------------------------------------------------

    def fetch_range(self, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Return synthetic ticks for bars within [start_ts, end_ts]."""
        mask = (self._df['timestamp'] >= start_ts) & (self._df['timestamp'] <= end_ts)
        return self._bars_to_ticks(self._df.loc[mask], upsample=self._upsample)

    def fetch_since(self, since_ts: int, end_ts: int) -> pd.DataFrame:
        """Return synthetic ticks for bars within (since_ts, end_ts]."""
        mask = (self._df['timestamp'] > since_ts) & (self._df['timestamp'] <= end_ts)
        return self._bars_to_ticks(self._df.loc[mask], upsample=self._upsample)

    def fetch_prev_day(self, now_ts: int) -> pd.DataFrame:
        """Return synthetic ticks for the previous trading day."""
        prev_start = get_prev_trading_day_start_ts(now_ts)
        prev_end = prev_start + 22 * 3600
        return self.fetch_range(prev_start, prev_end)

    # ------------------------------------------------------------------
    # Extended: multi-day historical lookback
    # ------------------------------------------------------------------

    def fetch_history_bars(self, end_ts: int, n_days: int = 60) -> pd.DataFrame:
        """Return raw 1-min bars for the last n_days trading days before end_ts.

        Returns a DataFrame with columns matching what feature providers expect:
        open, high, low, close, volume, trading_day, dt (datetime-indexed).
        """
        mask = self._df['timestamp'] <= end_ts
        subset = self._df.loc[mask].copy()

        if 'trading_day' not in subset.columns:
            return subset.tail(n_days * 1380)  # ~23h × 60min fallback

        days = sorted(subset['trading_day'].unique())
        if len(days) > n_days:
            days = days[-n_days:]
        return subset[subset['trading_day'].isin(days)].copy()

    def get_trading_days(self) -> list:
        """Return sorted list of unique trading days."""
        if 'trading_day' in self._df.columns:
            return sorted(self._df['trading_day'].unique())
        return []

    def get_day_bars(self, trading_day: str) -> pd.DataFrame:
        """Return raw 1-min bars for a specific trading day."""
        if 'trading_day' not in self._df.columns:
            return pd.DataFrame()
        return self._df[self._df['trading_day'] == trading_day].copy()

    def get_day_timestamp_range(self, trading_day: str) -> tuple:
        """Return (start_ts, end_ts) for a trading day."""
        day_df = self.get_day_bars(trading_day)
        if day_df.empty:
            return (0, 0)
        return (int(day_df['timestamp'].min()), int(day_df['timestamp'].max()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_ticks(bars: pd.DataFrame, upsample: bool = False) -> pd.DataFrame:
        """Convert 1-min bars to synthetic tick format for BarAggregator."""
        if bars.empty:
            return pd.DataFrame(
                columns=["timestamp", "price", "volume", "bidvolume", "askvolume"]
            )

        bars = normalize_orderflow_columns(bars, copy=True)

        if not upsample:
            ticks = pd.DataFrame({
                'timestamp': bars['timestamp'].values,
                'price': bars['close'].values,
                'volume': bars['volume'].values,
                'bidvolume': bars.get('bidvolume', pd.Series(0.0, index=bars.index)).values,
                'askvolume': bars.get('askvolume', pd.Series(0.0, index=bars.index)).values,
            })
            return ticks.reset_index(drop=True)

        # Upsample: 4 ticks per bar (O, H, L, C) at :00, :15, :30, :45
        timestamps = bars['timestamp'].values
        opens = bars['open'].values
        highs = bars['high'].values
        lows = bars['low'].values
        closes = bars['close'].values
        volumes = bars['volume'].values
        bid_col = bars.get('bidvolume', pd.Series(0.0, index=bars.index)).values
        ask_col = bars.get('askvolume', pd.Series(0.0, index=bars.index)).values

        n = len(bars)
        tick_ts = np.empty(n * 4, dtype=np.int64)
        tick_price = np.empty(n * 4, dtype=np.float64)
        tick_vol = np.empty(n * 4, dtype=np.float64)
        tick_bid = np.empty(n * 4, dtype=np.float64)
        tick_ask = np.empty(n * 4, dtype=np.float64)

        vol_quarter = volumes / 4.0
        bid_quarter = bid_col / 4.0
        ask_quarter = ask_col / 4.0

        for i in range(n):
            base = i * 4
            ts = timestamps[i]
            tick_ts[base] = ts
            tick_ts[base + 1] = ts + 15
            tick_ts[base + 2] = ts + 30
            tick_ts[base + 3] = ts + 45

            tick_price[base] = opens[i]
            tick_price[base + 1] = highs[i]
            tick_price[base + 2] = lows[i]
            tick_price[base + 3] = closes[i]

            tick_vol[base:base + 4] = vol_quarter[i]
            tick_bid[base:base + 4] = bid_quarter[i]
            tick_ask[base:base + 4] = ask_quarter[i]

        out = pd.DataFrame({
            'timestamp': tick_ts,
            'price': tick_price,
            'volume': tick_vol,
            'bidvolume': tick_bid,
            'askvolume': tick_ask,
        })
        return out
