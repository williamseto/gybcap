"""
Reversion quality feature provider.

Computes features that describe the quality of a rejection/reversion bar,
focusing on "how" the rejection happened rather than "where" price is.

These features capture:
- Rejection bar shape (wick vs body ratio, close position)
- Level penetration depth
- Touch count and time of day context
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


@FeatureRegistry.register('reversion_quality')
class ReversionQualityProvider(BaseFeatureProvider):
    """
    Computes rejection/reversion quality features from OHLC data.

    These features describe the characteristics of the rejection bar itself,
    which may be more predictive of success than price-level z-scores.
    """

    FEATURE_COLS = [
        'wick_to_body_ratio',      # Larger = stronger rejection
        'rejection_wick_pct',      # Wick as % of total range
        'close_position',          # Where close is within bar (0=low, 1=high)
        'rejection_penetration',   # How far past level before rejecting
        'bar_range_z',             # Range vs recent average (high = conviction)
        'level_touch_count',       # First touch vs repeated tests
        'bars_since_rth_open',     # Time of day effect
        'bar_momentum',            # Close - open direction strength
        # Multi-bar rejection patterns (Phase 1 additions)
        'rejection_bar_count_3',   # Rejection bars in last 3
        'max_wick_ratio_3bar',     # Max wick ratio in 3-bar window
        'bar_momentum_reversal',   # Did momentum change direction?
        # Approach dynamics features
        'vol_trend_into_level',    # Slope of volume over 10-bar approach
        'bar_size_trend_into_level',  # Slope of bar ranges over 10-bar approach
        'consecutive_same_dir',    # Consecutive bars moving toward level
        'approach_cum_delta_z',    # Z-scored cumulative delta over approach
    ]

    def __init__(
        self,
        level_cols: Optional[List[str]] = None,
        lookback_touches: int = 50,
        touch_threshold_pct: float = 0.001
    ):
        """
        Initialize provider.

        Args:
            level_cols: Price level columns to check for touches
            lookback_touches: Bars to look back for touch counting
            touch_threshold_pct: Threshold for considering a level "touched"
        """
        super().__init__()
        self.level_cols = level_cols or ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi']
        self.lookback_touches = lookback_touches
        self.touch_threshold_pct = touch_threshold_pct

    @property
    def name(self) -> str:
        return "reversion_quality"

    @property
    def feature_names(self) -> List[str]:
        return self.FEATURE_COLS

    def compute_bar_quality_features(
        self,
        bar: pd.Series,
        level: float,
        direction: str,  # 'bull' or 'bear'
        touch_count: int = 1,
        bars_since_open: int = 0,
        recent_ranges: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute quality features for a single rejection bar.

        Args:
            bar: Series with open, high, low, close
            level: Price level that was tested
            direction: 'bull' (rejection at low) or 'bear' (rejection at high)
            touch_count: Number of times level was touched before
            bars_since_open: Bars since RTH open
            recent_ranges: Array of recent bar ranges for z-score

        Returns:
            Dictionary of feature values
        """
        features = {}

        o = float(bar['open'])
        h = float(bar['high'])
        l = float(bar['low'])
        c = float(bar['close'])

        bar_range = h - l
        body = abs(c - o)

        # Avoid division by zero
        bar_range = max(bar_range, 0.01)
        body = max(body, 0.01)

        # --- Wick Features ---
        if direction == 'bull':
            # Bull rejection: wick is the lower tail (rejection of breakdown)
            wick = o - l if c >= o else c - l
        else:
            # Bear rejection: wick is the upper tail (rejection of breakout)
            wick = h - o if c <= o else h - c

        wick = max(wick, 0.0)

        features['wick_to_body_ratio'] = wick / body
        features['rejection_wick_pct'] = wick / bar_range

        # --- Close Position ---
        # 0 = close at low, 1 = close at high
        features['close_position'] = (c - l) / bar_range

        # --- Rejection Penetration ---
        # How far price went past the level before reversing
        if direction == 'bull':
            penetration = max(level - l, 0)  # How far below level
        else:
            penetration = max(h - level, 0)  # How far above level

        features['rejection_penetration'] = penetration / max(level * 0.001, 0.01)  # Normalize

        # --- Bar Range Z-Score ---
        if recent_ranges is not None and len(recent_ranges) > 1:
            mean_range = recent_ranges.mean()
            std_range = recent_ranges.std()
            features['bar_range_z'] = (bar_range - mean_range) / max(std_range, 0.01)
        else:
            features['bar_range_z'] = 0.0

        # --- Touch Count and Time ---
        features['level_touch_count'] = float(touch_count)
        features['bars_since_rth_open'] = float(bars_since_open)

        # --- Bar Momentum ---
        # Positive = bullish close, Negative = bearish close
        # Normalized by range
        features['bar_momentum'] = (c - o) / bar_range

        return features

    def _count_level_touches(
        self,
        bars: pd.DataFrame,
        bar_idx: int,
        level: float
    ) -> int:
        """Count how many times a level was touched in lookback window."""
        start_idx = max(0, bar_idx - self.lookback_touches)
        lookback = bars.iloc[start_idx:bar_idx]

        if lookback.empty:
            return 0

        threshold = level * self.touch_threshold_pct

        # Count bars where price came within threshold of level
        touched_low = (lookback['low'] <= level + threshold) & (lookback['low'] >= level - threshold)
        touched_high = (lookback['high'] >= level - threshold) & (lookback['high'] <= level + threshold)

        return int((touched_low | touched_high).sum())

    def _get_rth_bar_index(self, bars: pd.DataFrame, bar_idx: int) -> int:
        """Get number of bars since RTH open for this bar."""
        if 'ovn' not in bars.columns:
            return bar_idx

        day = bars.iloc[bar_idx].get('trading_day')
        if day is None:
            return bar_idx

        day_bars = bars[bars['trading_day'] == day]
        rth_bars = day_bars[day_bars['ovn'] == 0]

        if rth_bars.empty:
            return 0

        # Find position within RTH bars
        current_ts = bars.iloc[bar_idx].name if isinstance(bars.index, pd.DatetimeIndex) else bar_idx
        try:
            rth_idx = rth_bars.index.get_loc(current_ts)
            return int(rth_idx)
        except KeyError:
            return 0

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute reversion quality features for all bars (vectorized).

        Note: Features are most meaningful when computed at specific trigger bars.
        This implementation computes generic bar-quality features using VWAP as
        the reference level. For trade-specific features, use compute_trade_features().
        """
        result = ohlcv.copy()

        # Extract arrays for vectorized operations
        open_arr = ohlcv['open'].values.astype(np.float64)
        high_arr = ohlcv['high'].values.astype(np.float64)
        low_arr = ohlcv['low'].values.astype(np.float64)
        close_arr = ohlcv['close'].values.astype(np.float64)

        # Bar range with minimum to avoid division by zero
        bar_range = np.maximum(high_arr - low_arr, 0.01)
        body = np.maximum(np.abs(close_arr - open_arr), 0.01)

        # Direction: True = bull, False = bear
        is_bull = close_arr >= open_arr

        # Wick calculation (vectorized)
        # Bull: wick = (o - l) if c >= o else (c - l)
        # Bear: wick = (h - o) if c <= o else (h - c)
        bull_wick = np.where(is_bull, open_arr - low_arr, close_arr - low_arr)
        bear_wick = np.where(~is_bull, high_arr - open_arr, high_arr - close_arr)
        wick = np.where(is_bull, bull_wick, bear_wick)
        wick = np.maximum(wick, 0.0)

        # Wick features
        result['wick_to_body_ratio'] = wick / body
        result['rejection_wick_pct'] = wick / bar_range

        # Close position: (close - low) / range
        result['close_position'] = (close_arr - low_arr) / bar_range

        # Bar momentum: (close - open) / range
        result['bar_momentum'] = (close_arr - open_arr) / bar_range

        # Bar range z-score (rolling 20-bar window)
        range_series = pd.Series(bar_range)
        rolling_mean = range_series.rolling(window=20, min_periods=1).mean()
        rolling_std = range_series.rolling(window=20, min_periods=1).std().fillna(0.01)
        rolling_std = rolling_std.replace(0, 0.01)
        result['bar_range_z'] = (bar_range - rolling_mean.values) / rolling_std.values

        # Reference level for penetration (VWAP if available, else midpoint)
        if 'vwap' in ohlcv.columns:
            level = ohlcv['vwap'].fillna(pd.Series((high_arr + low_arr) / 2, index=ohlcv.index)).values
        else:
            level = (high_arr + low_arr) / 2

        # Rejection penetration (vectorized)
        # Bull: max(level - low, 0), Bear: max(high - level, 0)
        bull_penetration = np.maximum(level - low_arr, 0)
        bear_penetration = np.maximum(high_arr - level, 0)
        penetration = np.where(is_bull, bull_penetration, bear_penetration)
        result['rejection_penetration'] = penetration / np.maximum(level * 0.001, 0.01)

        # Bars since RTH open (vectorized per day)
        if 'trading_day' in ohlcv.columns and 'ovn' in ohlcv.columns:
            bars_since_open = np.zeros(len(ohlcv), dtype=np.float64)
            for day, day_df in ohlcv.groupby('trading_day'):
                day_mask = ohlcv['trading_day'] == day
                rth_mask = day_mask & (ohlcv['ovn'] == 0)
                rth_count = rth_mask.cumsum()
                # Only count within RTH bars
                bars_since_open[day_mask.values] = np.where(
                    ohlcv.loc[day_mask, 'ovn'].values == 0,
                    rth_count[day_mask].values - rth_count[rth_mask].values.min() if rth_mask.any() else 0,
                    0
                )
            result['bars_since_rth_open'] = bars_since_open
        else:
            result['bars_since_rth_open'] = np.arange(len(ohlcv), dtype=np.float64)

        # Level touch count (simplified: use rolling window count of touches)
        # This is approximated since exact touch counting is expensive
        # Count how many times low or high came within threshold of VWAP
        threshold = np.abs(level) * self.touch_threshold_pct
        touched = (
            ((low_arr >= level - threshold) & (low_arr <= level + threshold)) |
            ((high_arr >= level - threshold) & (high_arr <= level + threshold))
        )
        touch_series = pd.Series(touched.astype(float))
        result['level_touch_count'] = touch_series.rolling(
            window=self.lookback_touches, min_periods=1
        ).sum().values

        # --- Multi-bar rejection pattern features (Phase 1 additions) ---

        # Count rejection bars in last 3 (wick_to_body_ratio > 0.5)
        wick_to_body_series = pd.Series(result['wick_to_body_ratio'].values)
        is_rejection_bar = (wick_to_body_series > 0.5).astype(float)
        result['rejection_bar_count_3'] = is_rejection_bar.rolling(
            window=3, min_periods=1
        ).sum().values

        # Max wick ratio in 3-bar window
        result['max_wick_ratio_3bar'] = wick_to_body_series.rolling(
            window=3, min_periods=1
        ).max().values

        # Bar momentum reversal (did momentum change direction?)
        # Compare current bar momentum sign to previous bar
        bar_momentum = result['bar_momentum'].values
        momentum_series = pd.Series(bar_momentum)
        prev_momentum = momentum_series.shift(1).fillna(0).values
        momentum_reversal = (
            ((bar_momentum > 0) & (prev_momentum < 0)) |
            ((bar_momentum < 0) & (prev_momentum > 0))
        )
        result['bar_momentum_reversal'] = momentum_reversal.astype(float)

        # --- Approach dynamics features ---

        # Volume trend into level (slope of volume over 10-bar lookback)
        vol_series = pd.Series(ohlcv['volume'].values.astype(np.float64))
        x_10 = np.arange(10, dtype=np.float64)
        x_10_mean = x_10.mean()
        x_10_var = ((x_10 - x_10_mean) ** 2).sum()

        def _rolling_slope(series: pd.Series, window: int = 10) -> np.ndarray:
            """Compute rolling OLS slope (normalized)."""
            n = len(series)
            slopes = np.zeros(n, dtype=np.float64)
            vals = series.values
            x = np.arange(window, dtype=np.float64)
            x_m = x.mean()
            x_v = ((x - x_m) ** 2).sum()
            if x_v == 0:
                return slopes
            for t in range(window, n):
                y = vals[t - window:t]
                y_m = y.mean()
                if y_m == 0:
                    continue
                cov = ((x - x_m) * (y - y_m)).sum()
                slopes[t] = cov / x_v / max(abs(y_m), 1e-6)
            return slopes

        result['vol_trend_into_level'] = np.clip(_rolling_slope(vol_series, 10), -3, 3)

        # Bar size trend (slope of bar ranges over 10-bar approach)
        range_series = pd.Series(bar_range)
        result['bar_size_trend_into_level'] = np.clip(_rolling_slope(range_series, 10), -3, 3)

        # Consecutive same-direction bars (count of bars moving same direction)
        close_diff = np.diff(close_arr, prepend=close_arr[0])
        is_up = close_diff > 0
        is_down = close_diff < 0

        consec = np.zeros(len(close_arr), dtype=np.float64)
        for t in range(1, len(close_arr)):
            if is_up[t] and is_up[t - 1]:
                consec[t] = consec[t - 1] + 1
            elif is_down[t] and is_down[t - 1]:
                consec[t] = consec[t - 1] + 1
            else:
                consec[t] = 0
        result['consecutive_same_dir'] = consec

        # Approach cumulative delta z-score (order flow direction over 10 bars)
        if 'bidvolume' in ohlcv.columns and 'askvolume' in ohlcv.columns:
            bid_vol = ohlcv['bidvolume'].fillna(0).values.astype(np.float64)
            ask_vol = ohlcv['askvolume'].fillna(0).values.astype(np.float64)
            delta = bid_vol - ask_vol
        else:
            # Approximate delta from close position
            close_pos = (close_arr - low_arr) / bar_range
            delta = (close_pos - 0.5) * ohlcv['volume'].values.astype(np.float64)

        delta_series = pd.Series(delta)
        cum_delta_10 = delta_series.rolling(window=10, min_periods=1).sum()
        cum_delta_mean = cum_delta_10.rolling(window=50, min_periods=1).mean()
        cum_delta_std = cum_delta_10.rolling(window=50, min_periods=1).std().replace(0, 1)
        result['approach_cum_delta_z'] = np.clip(
            ((cum_delta_10 - cum_delta_mean) / cum_delta_std).fillna(0).values, -5, 5
        )

        return result

    def compute_trade_features(
        self,
        ohlcv: pd.DataFrame,
        trades_df: pd.DataFrame,
        level_col: str = 'level_price',
        direction_col: str = 'direction'
    ) -> pd.DataFrame:
        """
        Compute quality features specifically for trade entries.

        Args:
            ohlcv: Full OHLCV DataFrame
            trades_df: DataFrame with trade info
            level_col: Column containing the tested level
            direction_col: Column containing direction ('bull'/'bear')

        Returns:
            trades_df with quality features added
        """
        result = trades_df.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        # Compute rolling range statistics
        bar_ranges = (ohlcv['high'] - ohlcv['low']).values

        # Index OHLCV for fast lookup
        if 'dt' in ohlcv.columns:
            ohlcv_indexed = ohlcv.set_index('dt')
        else:
            ohlcv_indexed = ohlcv

        for idx, trade_row in trades_df.iterrows():
            # Get timestamp
            if 'entry_ts' in trade_row:
                ts = trade_row['entry_ts']
            elif 'dt' in trade_row:
                ts = trade_row['dt']
            else:
                continue

            try:
                bar = ohlcv_indexed.loc[ts]
            except KeyError:
                continue

            # Get level and direction
            level = trade_row.get(level_col, 0.0)
            direction = trade_row.get(direction_col, 'bull')

            if pd.isna(level) or level == 0:
                continue

            # Standardize direction
            if isinstance(direction, str):
                direction = direction.lower()
                if 'bull' in direction:
                    direction = 'bull'
                else:
                    direction = 'bear'

            # Get bar index in original ohlcv
            try:
                bar_idx = ohlcv_indexed.index.get_loc(ts)
            except (KeyError, TypeError):
                bar_idx = 0

            # Recent ranges
            start_idx = max(0, bar_idx - 20)
            recent_ranges = bar_ranges[start_idx:bar_idx] if bar_idx > 0 else None

            # Count touches
            touch_count = self._count_level_touches(ohlcv_indexed.reset_index(), bar_idx, level)

            # Bars since RTH open
            bars_since_open = self._get_rth_bar_index(ohlcv_indexed.reset_index(), bar_idx)

            # Compute features
            features = self.compute_bar_quality_features(
                bar, level, direction,
                touch_count=touch_count,
                bars_since_open=bars_since_open,
                recent_ranges=recent_ranges
            )

            for feat_name, feat_val in features.items():
                result.loc[idx, feat_name] = feat_val

        return result


def compute_rejection_strength(
    bar: pd.Series,
    level: float,
    direction: str
) -> float:
    """
    Compute a single "rejection strength" score combining multiple features.

    This can be used as a quality gate filter.

    Args:
        bar: OHLC bar series
        level: Price level tested
        direction: 'bull' or 'bear'

    Returns:
        Rejection strength score (higher = stronger rejection)
    """
    o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
    bar_range = max(h - l, 0.01)
    body = max(abs(c - o), 0.01)

    if direction == 'bull':
        wick = o - l if c >= o else c - l
        penetration = max(level - l, 0)
        # Good bull rejection: close near high, long lower wick
        close_strength = (c - l) / bar_range
    else:
        wick = h - o if c <= o else h - c
        penetration = max(h - level, 0)
        # Good bear rejection: close near low, long upper wick
        close_strength = (h - c) / bar_range

    wick = max(wick, 0)
    wick_ratio = wick / body
    wick_pct = wick / bar_range

    # Combine into single score
    # Higher wick ratio, better close position, some penetration = stronger
    penetration_score = min(penetration / (level * 0.002), 2.0)  # Cap at 2

    score = (
        0.4 * min(wick_ratio, 3.0) / 3.0 +  # Normalize wick ratio
        0.3 * close_strength +
        0.2 * wick_pct +
        0.1 * penetration_score
    )

    return float(score)
