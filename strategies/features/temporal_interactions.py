"""
Temporal interaction feature provider.

Creates interaction features that combine multiple base features to capture
reversal dynamics. Individual features may have weak correlation with reversal
success, but interactions can capture non-linear patterns.

Features include:
- Wick-delta interaction (rejection quality * order flow)
- Volume exhaustion ratios (volume dynamics around rejection)
- Delta exhaustion (cumulative delta divergence from price)
- Consecutive rejection patterns (multi-bar rejection counts)
- Time-of-day volatility interactions
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


@FeatureRegistry.register('temporal_interactions')
class TemporalInteractionProvider(BaseFeatureProvider):
    """
    Computes temporal interaction features for reversal prediction.

    These features combine multiple signals to capture reversal dynamics
    that may not be apparent from individual features.
    """

    FEATURE_COLS = [
        # Wick-volume interactions
        'wick_delta_interaction',      # wick_to_body_ratio * delta_flip indicator
        'wick_volume_interaction',     # wick_ratio * volume_z
        'vol_exhaustion_ratio',        # vol_at_rejection / vol_into_level

        # Delta/flow features
        'delta_exhaustion',            # cumulative delta divergence from price
        'delta_momentum_3bar',         # rolling 3-bar delta momentum
        'delta_reversal_signal',       # delta changing direction

        # Multi-bar rejection patterns
        'consecutive_rejections_3',    # rejection bars in last 3
        'consecutive_rejections_5',    # rejection bars in last 5
        'rejection_momentum',          # acceleration of rejection strength
        'max_wick_ratio_3bar',         # max wick ratio in 3-bar window

        # Time-volatility interactions
        'tod_volatility_interaction',  # time_of_day * intraday_vol
        'session_vol_ratio',           # current vol / session avg vol

        # Zone-aware features
        'bars_to_nearest_level',       # distance to key S/R
        'level_test_sequence',         # pattern of touches in last N bars
        'mae_estimate',                # estimated MAE based on recent volatility
    ]

    def __init__(
        self,
        wick_threshold: float = 0.5,
        vol_lookback: int = 20,
        level_cols: Optional[List[str]] = None
    ):
        """
        Initialize provider.

        Args:
            wick_threshold: Minimum wick ratio to count as rejection
            vol_lookback: Bars for volume rolling calculations
            level_cols: Price level columns for distance calculations
        """
        super().__init__()
        self.wick_threshold = wick_threshold
        self.vol_lookback = vol_lookback
        self.level_cols = level_cols or ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi']

    @property
    def name(self) -> str:
        return "temporal_interactions"

    @property
    def feature_names(self) -> List[str]:
        return self.FEATURE_COLS

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute temporal interaction features for all bars (vectorized).

        Args:
            ohlcv: DataFrame with OHLCV data and optionally volume features
            context: Optional context with pre-computed features

        Returns:
            DataFrame with interaction features added
        """
        result = ohlcv.copy()
        n = len(ohlcv)

        # Extract base arrays
        open_arr = ohlcv['open'].values.astype(np.float64)
        high_arr = ohlcv['high'].values.astype(np.float64)
        low_arr = ohlcv['low'].values.astype(np.float64)
        close_arr = ohlcv['close'].values.astype(np.float64)
        volume_arr = ohlcv['volume'].values.astype(np.float64)

        # --- Compute base components ---

        # Bar range and body
        bar_range = np.maximum(high_arr - low_arr, 0.01)
        body = np.maximum(np.abs(close_arr - open_arr), 0.01)

        # Direction
        is_bull = close_arr >= open_arr

        # Wick calculation (same as reversion_quality)
        bull_wick = np.where(is_bull, open_arr - low_arr, close_arr - low_arr)
        bear_wick = np.where(~is_bull, high_arr - open_arr, high_arr - close_arr)
        wick = np.maximum(np.where(is_bull, bull_wick, bear_wick), 0.0)
        wick_to_body_ratio = wick / body
        wick_pct = wick / bar_range

        # Volume z-score
        vol_series = pd.Series(volume_arr)
        vol_mean = vol_series.rolling(window=self.vol_lookback, min_periods=1).mean()
        vol_std = vol_series.rolling(window=self.vol_lookback, min_periods=1).std().fillna(1)
        vol_std = vol_std.replace(0, 1)
        volume_z = ((volume_arr - vol_mean.values) / vol_std.values)

        # --- Wick-volume interactions ---

        # Wick-delta interaction (use volume as proxy for delta if delta not available)
        if 'delta' in ohlcv.columns:
            delta = ohlcv['delta'].values.astype(np.float64)
        else:
            # Approximate delta using close position
            close_position = (close_arr - low_arr) / bar_range
            delta = (close_position - 0.5) * volume_arr  # positive = buy pressure

        # Delta flip: did delta change sign?
        delta_series = pd.Series(delta)
        prev_delta = delta_series.shift(1).fillna(0).values
        delta_flip = ((delta > 0) & (prev_delta < 0)) | ((delta < 0) & (prev_delta > 0))
        result['wick_delta_interaction'] = wick_to_body_ratio * delta_flip.astype(float)

        # Wick-volume interaction
        result['wick_volume_interaction'] = wick_to_body_ratio * np.clip(volume_z, -3, 3)

        # Volume exhaustion ratio (vol at bar / vol into bar)
        vol_into = vol_series.rolling(window=3, min_periods=1).mean().shift(1).fillna(volume_arr[0])
        vol_exhaustion = volume_arr / np.maximum(vol_into.values, 1)
        result['vol_exhaustion_ratio'] = np.clip(vol_exhaustion, 0, 5)

        # --- Delta/flow features ---

        # Delta exhaustion: cumulative delta vs price trend divergence
        cum_delta = delta_series.rolling(window=10, min_periods=1).sum()
        price_trend = pd.Series(close_arr).diff(10).fillna(0)

        # Normalize both
        cum_delta_z = (cum_delta - cum_delta.rolling(50, min_periods=1).mean()) / \
                      cum_delta.rolling(50, min_periods=1).std().fillna(1).replace(0, 1)
        price_trend_z = (price_trend - price_trend.rolling(50, min_periods=1).mean()) / \
                        price_trend.rolling(50, min_periods=1).std().fillna(1).replace(0, 1)

        # Divergence: when delta and price trend disagree
        result['delta_exhaustion'] = np.clip(
            (cum_delta_z.values - price_trend_z.values * np.sign(cum_delta.values)),
            -5, 5
        )

        # Delta momentum (3-bar)
        delta_3bar = delta_series.rolling(window=3, min_periods=1).sum()
        delta_3bar_prev = delta_3bar.shift(1).fillna(0)
        result['delta_momentum_3bar'] = np.clip(
            (delta_3bar.values - delta_3bar_prev.values) / np.maximum(np.abs(delta_3bar_prev.values), 1),
            -5, 5
        )

        # Delta reversal signal
        result['delta_reversal_signal'] = delta_flip.astype(float)

        # --- Multi-bar rejection patterns ---

        # Count rejection bars in window
        is_rejection = wick_to_body_ratio > self.wick_threshold
        rejection_series = pd.Series(is_rejection.astype(float))

        result['consecutive_rejections_3'] = rejection_series.rolling(
            window=3, min_periods=1
        ).sum().values

        result['consecutive_rejections_5'] = rejection_series.rolling(
            window=5, min_periods=1
        ).sum().values

        # Rejection momentum (are rejections getting stronger?)
        wick_ratio_series = pd.Series(wick_to_body_ratio)
        wick_3bar = wick_ratio_series.rolling(window=3, min_periods=1).mean()
        wick_3bar_prev = wick_3bar.shift(3).fillna(0)
        result['rejection_momentum'] = np.clip(
            wick_3bar.values - wick_3bar_prev.values,
            -3, 3
        )

        # Max wick ratio in 3-bar window
        result['max_wick_ratio_3bar'] = wick_ratio_series.rolling(
            window=3, min_periods=1
        ).max().values

        # --- Time-volatility interactions ---

        # Bars since RTH open (time of day proxy)
        if 'trading_day' in ohlcv.columns and 'ovn' in ohlcv.columns:
            bars_since_open = np.zeros(n, dtype=np.float64)
            for day, day_df in ohlcv.groupby('trading_day'):
                day_mask = ohlcv['trading_day'] == day
                rth_mask = day_mask & (ohlcv['ovn'] == 0)
                if rth_mask.any():
                    rth_count = rth_mask.cumsum()
                    bars_since_open[day_mask.values] = np.where(
                        ohlcv.loc[day_mask, 'ovn'].values == 0,
                        rth_count[day_mask].values - rth_count[rth_mask].values.min(),
                        0
                    )
            tod = bars_since_open / 390  # Normalize to [0, 1] assuming 390 min RTH
        else:
            tod = np.linspace(0, 1, n)

        # Intraday volatility (rolling ATR-like)
        atr = pd.Series(bar_range).rolling(window=14, min_periods=1).mean().values
        atr_z = (atr - np.mean(atr)) / (np.std(atr) + 1e-6)

        result['tod_volatility_interaction'] = tod * np.clip(atr_z, -3, 3)

        # Session volume ratio (current vol / session avg vol so far)
        if 'trading_day' in ohlcv.columns:
            session_vol_ratio = np.ones(n)
            for day, day_df in ohlcv.groupby('trading_day'):
                day_mask = ohlcv['trading_day'] == day
                day_vol = volume_arr[day_mask.values]
                cumsum = np.cumsum(day_vol)
                cumcount = np.arange(1, len(day_vol) + 1)
                cumavg = cumsum / cumcount
                session_vol_ratio[day_mask.values] = day_vol / np.maximum(cumavg, 1)
            result['session_vol_ratio'] = np.clip(session_vol_ratio, 0, 5)
        else:
            result['session_vol_ratio'] = np.clip(volume_arr / vol_mean.values, 0, 5)

        # --- Zone-aware features ---

        # Distance to nearest level
        available_levels = [c for c in self.level_cols if c in ohlcv.columns]
        if available_levels:
            level_distances = []
            for level_col in available_levels:
                level_vals = ohlcv[level_col].values.astype(np.float64)
                dist = np.abs(close_arr - level_vals) / close_arr
                level_distances.append(dist)
            min_dist = np.min(level_distances, axis=0)
            result['bars_to_nearest_level'] = min_dist * 10000  # Convert to basis points
        else:
            result['bars_to_nearest_level'] = 0.0

        # Level test sequence (simplified: count how many bars touched a level recently)
        if available_levels:
            touch_count = np.zeros(n)
            for level_col in available_levels:
                level_vals = ohlcv[level_col].values.astype(np.float64)
                threshold = level_vals * 0.001  # 0.1% threshold
                touched = (
                    ((low_arr >= level_vals - threshold) & (low_arr <= level_vals + threshold)) |
                    ((high_arr >= level_vals - threshold) & (high_arr <= level_vals + threshold))
                )
                touch_series = pd.Series(touched.astype(float))
                touch_count += touch_series.rolling(window=10, min_periods=1).sum().values
            result['level_test_sequence'] = touch_count
        else:
            result['level_test_sequence'] = 0.0

        # MAE estimate (based on recent volatility)
        # Estimate: how much could price move against us in next few bars
        rolling_range_max = pd.Series(bar_range).rolling(window=5, min_periods=1).max()
        result['mae_estimate'] = (rolling_range_max.values / close_arr) * 100  # as percentage

        return result


def compute_interaction_features(
    ohlcv: pd.DataFrame,
    level_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function to compute all temporal interaction features.

    Args:
        ohlcv: DataFrame with OHLCV data
        level_cols: Optional list of price level column names

    Returns:
        DataFrame with interaction features added
    """
    provider = TemporalInteractionProvider(level_cols=level_cols)
    return provider.compute(ohlcv)
