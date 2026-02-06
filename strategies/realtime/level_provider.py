"""Real-time price level computation.

Migrated from util/strategy_util.py -- this is the single-day real-time
version that computes VWAP, overnight hi/lo, IB hi/lo, RTH hi/lo, RSI,
ADX, OFI, and z-scores from live 1-minute bars.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from strategies.realtime.data_source import DataSource

# Feature columns produced by compute_bar_stats
FEAT_COLS = [
    'close_z20', 'ovn_lo_z', 'ovn_hi_z', 'ib_lo_z', 'ib_hi_z',
    'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z',
]


class DayPriceLevelProvider:
    """Compute intraday price levels for a single trading day."""

    def __init__(self, data_source: DataSource):
        self._data_source = data_source
        self.prev_day_levels: Dict[str, float] = {}
        self.feat_cols: List[str] = list(FEAT_COLS)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, now_ts: int) -> None:
        """Fetch previous-day data and compute prev_high / prev_low / prev_mid."""
        prev_day_df = self._data_source.fetch_prev_day(now_ts)

        if prev_day_df.empty:
            self.prev_day_levels = {'prev_high': 0.0, 'prev_low': 0.0, 'prev_mid': 0.0}
            return

        self.prev_day_levels = {
            'prev_high': float(prev_day_df['price'].max()),
            'prev_low': float(prev_day_df['price'].min()),
            'prev_mid': float(
                (prev_day_df['price'].max() + prev_day_df['price'].min()) / 2
            ),
        }

    # ------------------------------------------------------------------
    # Per-bar feature computation (operates on 1-min bars in-place)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_bar_stats(group: pd.DataFrame) -> pd.DataFrame:
        """Compute intraday bar statistics on 1-min bars (datetime-indexed, LA tz)."""

        rth_open_hr = 6
        rth_close_hr = 13

        def _is_ovn(idx: pd.DatetimeIndex) -> pd.Series:
            h = idx.hour
            m = idx.minute
            ovn = (
                (h < rth_open_hr) | (h > rth_close_hr)
                | ((h == rth_open_hr) & (m < 30))
            )
            return pd.Series(ovn.astype(int), index=idx)

        group['ovn'] = _is_ovn(group.index)

        # RSI
        if HAS_TALIB:
            group['rsi'] = talib.RSI(group['close'], timeperiod=14)
        else:
            group['rsi'] = 50.0

        # Overnight hi/lo
        ovn_hi = group.loc[group['ovn'] == 1, 'high'].max()
        ovn_lo = group.loc[group['ovn'] == 1, 'low'].min()
        group['ovn_hi'] = ovn_hi
        group['ovn_lo'] = ovn_lo

        # VWAP
        avg_price = (group['open'] + group['high'] + group['low'] + group['close']) / 4
        cum_vol = group['volume'].cumsum()
        vwap = avg_price.mul(group['volume']).cumsum().div(cum_vol)
        vwap_std = (
            avg_price.sub(vwap).pow(2).cumsum()
            .div(np.arange(1, len(vwap) + 1))
            .apply(np.sqrt)
            .clip(lower=1e-6)
        )
        group['vwap'] = vwap
        group['vwap_z'] = (group['close'] - vwap) / vwap_std

        # SMA20 z-score
        if HAS_TALIB:
            sma20 = talib.SMA(group['close'], timeperiod=20)
        else:
            sma20 = group['close'].rolling(window=20).mean()
        std20 = group['close'].rolling(window=20).std().fillna(1e-6)
        group['close_z20'] = (group['close'] - sma20) / std20

        # Overnight z-scores
        group['ovn_lo_z'] = (group['close'] - ovn_lo) / vwap_std
        group['ovn_hi_z'] = (ovn_hi - group['close']) / vwap_std

        # Initial balance
        ib_df = group.between_time('6:30', '7:30')
        ib_lo = ib_df['low'].min() if not ib_df.empty else np.nan
        ib_hi = ib_df['high'].max() if not ib_df.empty else np.nan

        ib_lo_z = (group['close'] - ib_lo) / vwap_std
        ib_hi_z = (ib_hi - group['close']) / vwap_std

        group['ib_lo_z'] = 0.0
        group['ib_hi_z'] = 0.0
        group['ib_lo'] = 0.0
        group['ib_hi'] = 0.0

        rth_after_ib_idx = group.between_time('7:30', '12:59').index
        group.loc[rth_after_ib_idx, 'ib_lo_z'] = ib_lo_z.loc[rth_after_ib_idx]
        group.loc[rth_after_ib_idx, 'ib_hi_z'] = ib_hi_z.loc[rth_after_ib_idx]
        group.loc[rth_after_ib_idx, 'ib_lo'] = ib_lo
        group.loc[rth_after_ib_idx, 'ib_hi'] = ib_hi

        # RTH hi/lo (cumulative after IB)
        group['rth_lo'] = 0.0
        group['rth_hi'] = 0.0
        group.loc[rth_after_ib_idx, 'rth_lo'] = group.loc[rth_after_ib_idx, 'low'].cummin()
        group.loc[rth_after_ib_idx, 'rth_hi'] = group.loc[rth_after_ib_idx, 'high'].cummax()

        # Volume z-score
        vol_mean = group['volume'].rolling(window=20, center=True, min_periods=1).mean()
        vol_std = group['volume'].rolling(window=20, center=True, min_periods=1).std().add(1e-6)
        group['vol_z'] = (group['volume'] - vol_mean) / vol_std

        # ADX
        if HAS_TALIB:
            group['adx'] = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)
        else:
            group['adx'] = 0.0

        # Order flow imbalance z-score
        if 'buys' in group.columns and 'sells' in group.columns:
            ofi_pct = (group['buys'] - group['sells']) / (group['buys'] + group['sells'])
            window = 60
            ofi_pct_series = ofi_pct.fillna(0)
            ofi_z = (
                ofi_pct_series - ofi_pct_series.rolling(window).mean()
            ) / ofi_pct_series.rolling(window).std()
            group['ofi_z'] = ofi_z
        else:
            group['ofi_z'] = 0.0

        return group

    # ------------------------------------------------------------------
    # Attach levels to resampled bars
    # ------------------------------------------------------------------

    def attach_levels_to_bars(
        self,
        ohlcv_1m: pd.DataFrame,
        t_samp: str = '5Min',
    ) -> pd.DataFrame:
        """
        Resample 1-min bars to *t_samp* and attach price-level features.

        Args:
            ohlcv_1m: 1-minute bars (datetime-indexed, LA tz) with OHLCV + buys/sells.
            t_samp: Resample period (e.g. '5Min', '15Min').

        Returns:
            Resampled bars with prev-day levels + intraday features.
        """
        ohlcv_feat = self.compute_bar_stats(ohlcv_1m.copy())

        bars = ohlcv_1m.resample(t_samp).agg({
            'price': 'last',
            'buys': 'sum',
            'sells': 'sum',
            'volume': 'sum',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })

        # Attach previous-day levels
        for col, val in self.prev_day_levels.items():
            bars[col] = val

        # Merge intraday features from 1-min
        feat_merge_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'ovn', 'ib_lo', 'ib_hi',
                           'rth_lo', 'rth_hi'] + self.feat_cols
        feat_merge_cols = [c for c in feat_merge_cols if c in ohlcv_feat.columns]

        feat_1m = ohlcv_feat[feat_merge_cols]

        bars = pd.merge(bars, feat_1m, left_index=True, right_index=True, how='left')

        return bars
