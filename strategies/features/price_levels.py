"""
Price level feature provider.

Computes previous day levels and intraday features like VWAP, RSI,
overnight high/low, initial balance, etc.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


@FeatureRegistry.register('price_levels')
class PriceLevelProvider(BaseFeatureProvider):
    """
    Computes price level features from OHLCV data.

    Features include:
    - Previous day high/low/mid
    - VWAP and VWAP z-score
    - Overnight high/low
    - Initial balance high/low
    - RSI, ADX, volume z-score
    - Order flow imbalance z-score
    """

    LEVEL_COLS = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']

    FEATURE_COLS = [
        'close_z20', 'ovn_lo_z', 'ovn_hi_z', 'ib_lo_z', 'ib_hi_z',
        'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z'
    ]

    def __init__(self, include_gamma: bool = True):
        """
        Initialize provider.

        Args:
            include_gamma: Whether to include gamma score feature
        """
        super().__init__()
        self.include_gamma = include_gamma
        self._ohlcv_feat: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "price_levels"

    @property
    def feature_names(self) -> List[str]:
        cols = self.FEATURE_COLS.copy()
        if self.include_gamma:
            cols.append('nearby_gamma_score')
        return cols

    @property
    def level_cols(self) -> List[str]:
        return self.LEVEL_COLS

    def prev_day_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute previous day high/low/mid levels."""
        daily = ohlcv.groupby('trading_day').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        daily = daily[['high', 'low']].rename(columns={
            'high': 'day_high',
            'low': 'day_low'
        })
        prev = daily.shift(1).rename(columns={
            'day_high': 'prev_high',
            'day_low': 'prev_low'
        })
        prev['prev_mid'] = (prev['prev_high'] + prev['prev_low']) / 2.0
        return prev[['prev_high', 'prev_low', 'prev_mid']]

    def _compute_bar_stats(self, group: pd.DataFrame) -> pd.DataFrame:
        """Compute intraday bar statistics for a single day."""
        # RSI
        if HAS_TALIB:
            group["rsi"] = talib.RSI(group["close"], timeperiod=14)
        else:
            group["rsi"] = 50.0  # Placeholder

        # Overnight high/low
        ovn_hi = group[group['ovn'] == 1]['high'].max() if 'ovn' in group.columns else np.nan
        ovn_lo = group[group['ovn'] == 1]['low'].min() if 'ovn' in group.columns else np.nan
        group['ovn_hi'] = ovn_hi
        group['ovn_lo'] = ovn_lo

        # VWAP
        avg_price = (group["open"] + group["high"] + group["low"] + group["close"]) / 4
        vwap = avg_price.mul(group["volume"]).cumsum().div(group["volume"].cumsum())
        vwap_std = avg_price.sub(vwap).pow(2).cumsum().div(
            np.arange(1, len(vwap) + 1)
        ).apply(np.sqrt).clip(lower=1e-6)

        group['vwap'] = vwap
        group['vwap_z'] = (group['close'] - vwap) / vwap_std

        # SMA20 z-score
        if HAS_TALIB:
            sma20 = talib.SMA(group["close"], timeperiod=20)
        else:
            sma20 = group['close'].rolling(window=20).mean()
        std20 = group['close'].rolling(window=20).std().fillna(0.000001)
        group['close_z20'] = (group['close'] - sma20) / std20

        # Overnight z-scores
        group['ovn_lo_z'] = (group['close'] - ovn_lo) / vwap_std
        group['ovn_hi_z'] = (ovn_hi - group['close']) / vwap_std

        # Initial balance and RTH levels
        if 'dt' in group.columns:
            time_df = group.set_index("dt")
        elif isinstance(group.index, pd.DatetimeIndex):
            time_df = group
        else:
            # Fallback: create dummy time-based features
            group['ib_lo_z'] = 0.0
            group['ib_hi_z'] = 0.0
            group['ib_lo'] = 0.0
            group['ib_hi'] = 0.0
            group['rth_lo'] = 0.0
            group['rth_hi'] = 0.0
            group['vol_z'] = 0.0
            group['adx'] = 0.0
            group['ofi_z'] = 0.0
            return group

        try:
            # Use between_time() on datetime-indexed df (matches realtime version)
            ib_df = time_df.between_time('6:30', '7:30')
            ib_lo = ib_df['low'].min() if not ib_df.empty else np.nan
            ib_hi = ib_df['high'].max() if not ib_df.empty else np.nan

            # Compute z-scores on datetime-indexed df
            vwap_std_dt = vwap_std.values if not isinstance(vwap_std, np.ndarray) else vwap_std
            vwap_std_series = pd.Series(vwap_std_dt, index=time_df.index)
            ib_lo_z = (time_df['close'] - ib_lo) / vwap_std_series
            ib_hi_z = (ib_hi - time_df['close']) / vwap_std_series

            time_df['ib_lo_z'] = 0.0
            time_df['ib_hi_z'] = 0.0
            time_df['ib_lo'] = 0.0
            time_df['ib_hi'] = 0.0

            rth_after_ib_idx = time_df.between_time('7:30', '12:59').index
            time_df.loc[rth_after_ib_idx, 'ib_lo_z'] = ib_lo_z.loc[rth_after_ib_idx]
            time_df.loc[rth_after_ib_idx, 'ib_hi_z'] = ib_hi_z.loc[rth_after_ib_idx]
            time_df.loc[rth_after_ib_idx, 'ib_lo'] = ib_lo
            time_df.loc[rth_after_ib_idx, 'ib_hi'] = ib_hi

            # RTH high/low (cumulative after IB)
            time_df['rth_lo'] = 0.0
            time_df['rth_hi'] = 0.0
            time_df.loc[rth_after_ib_idx, 'rth_lo'] = time_df.loc[rth_after_ib_idx, 'low'].cummin()
            time_df.loc[rth_after_ib_idx, 'rth_hi'] = time_df.loc[rth_after_ib_idx, 'high'].cummax()

            # Map results back to original integer-indexed group
            for col in ['ib_lo_z', 'ib_hi_z', 'ib_lo', 'ib_hi', 'rth_lo', 'rth_hi']:
                group[col] = time_df[col].values
        except Exception:
            group['ib_lo_z'] = 0.0
            group['ib_hi_z'] = 0.0
            group['ib_lo'] = 0.0
            group['ib_hi'] = 0.0
            group['rth_lo'] = 0.0
            group['rth_hi'] = 0.0

        # Volume z-score
        vol_mean = group['volume'].rolling(window=20, center=True, min_periods=1).mean()
        vol_std = group['volume'].rolling(window=20, center=True, min_periods=1).std().add(1e-6)
        group['vol_z'] = (group['volume'] - vol_mean) / vol_std

        # ADX
        if HAS_TALIB:
            group['adx'] = talib.ADX(
                group['high'], group['low'], group['close'], timeperiod=14
            )
        else:
            group['adx'] = 0.0

        # Order flow imbalance z-score
        if 'askvolume' in group.columns and 'bidvolume' in group.columns:
            ofi_pct = (group['askvolume'] - group['bidvolume']) / (
                group['askvolume'] + group['bidvolume']
            )
            window = 60
            ofi_pct_series = ofi_pct.fillna(0)
            ofi_z = (
                ofi_pct_series - ofi_pct_series.rolling(window).mean()
            ) / ofi_pct_series.rolling(window).std()
            group['ofi_z'] = ofi_z
        else:
            group['ofi_z'] = 0.0

        return group

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compute all price level features."""
        # Compute bar stats per day
        self._ohlcv_feat = ohlcv.groupby('trading_day').apply(
            self._compute_bar_stats, include_groups=False
        ).reset_index()

        return self._ohlcv_feat

    def attach_levels_to_bars(self, bars: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Attach computed features to HTF bars.

        Args:
            bars: Higher timeframe bars (e.g., 5min, 15min)
            ohlcv: Original 1-min OHLCV data

        Returns:
            bars with features attached
        """
        # Compute features if not already done
        if self._ohlcv_feat is None:
            self._compute_impl(ohlcv)

        levels = self.prev_day_levels(ohlcv)

        # Map previous day levels
        bars['prev_high'] = bars['trading_day'].map(levels['prev_high'])
        bars['prev_low'] = bars['trading_day'].map(levels['prev_low'])
        bars['prev_mid'] = bars['trading_day'].map(levels['prev_mid'])

        # Feature columns to merge
        feat_cols = self.feature_names.copy()
        if 'nearby_gamma_score' not in self._ohlcv_feat.columns and 'nearby_gamma_score' in feat_cols:
            feat_cols.remove('nearby_gamma_score')

        merge_cols = ['dt', 'vwap', 'ovn_lo', 'ovn_hi', 'ovn', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi'] + feat_cols

        # Only include columns that exist
        merge_cols = [c for c in merge_cols if c in self._ohlcv_feat.columns]

        feat_1m = self._ohlcv_feat[merge_cols]

        bars = pd.merge(
            bars,
            feat_1m,
            left_on=['dt'],
            right_on=['dt'],
            how='left'
        )

        return bars
