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
        Compute reversion quality features for all bars.

        Note: Features are most meaningful when computed at specific trigger bars.
        This implementation computes generic bar-quality features using VWAP as
        the reference level. For trade-specific features, use compute_trade_features().
        """
        result = ohlcv.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        # Compute rolling range statistics
        bar_ranges = (ohlcv['high'] - ohlcv['low']).values

        for i in range(len(ohlcv)):
            bar = ohlcv.iloc[i]

            # Get reference level (use VWAP if available)
            if 'vwap' in ohlcv.columns and not pd.isna(bar.get('vwap')):
                level = bar['vwap']
            else:
                level = (bar['high'] + bar['low']) / 2

            # Determine direction based on close vs open
            direction = 'bull' if bar['close'] >= bar['open'] else 'bear'

            # Recent ranges for z-score
            start_idx = max(0, i - 20)
            recent_ranges = bar_ranges[start_idx:i] if i > 0 else None

            # Count touches
            touch_count = self._count_level_touches(ohlcv, i, level)

            # Bars since RTH open
            bars_since_open = self._get_rth_bar_index(ohlcv, i)

            # Compute features
            features = self.compute_bar_quality_features(
                bar, level, direction,
                touch_count=touch_count,
                bars_since_open=bars_since_open,
                recent_ranges=recent_ranges
            )

            for feat_name, feat_val in features.items():
                result.iloc[i, result.columns.get_loc(feat_name)] = feat_val

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
