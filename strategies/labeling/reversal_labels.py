"""
Strict reversal labeling for end-to-end prediction.

This module provides labeling logic for identifying "strong reversals"
directly from price data, without heuristic detection rules.

Key differences from heuristic approach:
- Labels are computed using FUTURE data (we know what happened)
- Model only sees PAST data when predicting
- Strict criteria: minimum move, sustained reversal, validated bounce

Labeling criteria:
1. Find local extrema using rolling slope analysis
2. Require minimum subsequent move (e.g., 0.2% = ~10 ES points)
3. Validate reversal sustains (doesn't immediately fail back)
4. Label the bar where reversal BEGINS, not after the fact
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import linregress


class ReversalLabel(IntEnum):
    """Reversal direction labels."""
    NONE = 0
    BULL = 1   # Price was falling, reverses up
    BEAR = -1  # Price was rising, reverses down


@dataclass
class ReversalConfig:
    """Configuration for reversal labeling."""
    min_move_pct: float = 0.002       # Minimum move to qualify (0.2%)
    slope_window: int = 20            # Bars for slope calculation
    min_slope: float = 1e-4           # Minimum slope to detect trend
    validation_bars: int = 15         # Bars to validate reversal holds
    bounce_threshold_pct: float = 0.5 # Max bounce as fraction of move
    use_close: bool = True            # Use close vs high/low for detection


def _compute_rolling_slope(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling linear regression slope.

    Args:
        prices: Array of prices
        window: Lookback window for slope calculation

    Returns:
        Array of slope values (same length as prices, NaN-padded)
    """
    n = len(prices)
    slopes = np.full(n, np.nan)

    for i in range(window - 1, n):
        start = i - window + 1
        x = np.arange(window)
        y = prices[start:i + 1]
        if len(y) == window:
            slope, _, _, _, _ = linregress(x, y)
            slopes[i] = slope

    return slopes


def _find_local_extrema(
    prices: np.ndarray,
    slopes: np.ndarray,
    min_slope: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find potential reversal points based on slope changes.

    Returns:
        local_max_mask: Boolean array of local maxima (bearish reversal starts)
        local_min_mask: Boolean array of local minima (bullish reversal starts)
    """
    n = len(prices)
    local_max_mask = np.zeros(n, dtype=bool)
    local_min_mask = np.zeros(n, dtype=bool)

    for i in range(1, n - 1):
        if np.isnan(slopes[i]) or np.isnan(slopes[i - 1]):
            continue

        # Detect slope sign change with minimum magnitude
        prev_slope = slopes[i - 1]
        curr_slope = slopes[i]

        # Local maximum: slope was positive (up), now turning negative
        if prev_slope > min_slope and curr_slope < -min_slope:
            local_max_mask[i] = True

        # Local minimum: slope was negative (down), now turning positive
        if prev_slope < -min_slope and curr_slope > min_slope:
            local_min_mask[i] = True

    return local_max_mask, local_min_mask


def _validate_reversal(
    prices: np.ndarray,
    start_idx: int,
    direction: int,  # 1 for bull, -1 for bear
    min_move_pct: float,
    validation_bars: int,
    bounce_threshold_pct: float
) -> Tuple[bool, float, int]:
    """
    Validate that a reversal meets minimum criteria.

    Args:
        prices: Price array
        start_idx: Index where reversal starts
        direction: 1 for bullish (up), -1 for bearish (down)
        min_move_pct: Minimum % move required
        validation_bars: Bars to check for sustained move
        bounce_threshold_pct: Max bounce as fraction of min_move

    Returns:
        is_valid: Whether reversal meets criteria
        magnitude: Actual move magnitude (% of start price)
        duration: Bars until move completed
    """
    n = len(prices)
    end_idx = min(start_idx + validation_bars, n)
    future_prices = prices[start_idx:end_idx]

    if len(future_prices) < 3:
        return False, 0.0, 0

    start_price = prices[start_idx]

    # Calculate move in expected direction
    if direction == 1:  # Bullish: expect price to go up
        move_prices = future_prices - start_price
        best_move_idx = np.argmax(move_prices)
        best_move = move_prices[best_move_idx]
    else:  # Bearish: expect price to go down
        move_prices = start_price - future_prices
        best_move_idx = np.argmax(move_prices)
        best_move = move_prices[best_move_idx]

    move_pct = best_move / start_price

    # Check minimum move
    if move_pct < min_move_pct:
        return False, move_pct, best_move_idx

    # Check for disqualifying bounce AFTER the move
    # (reversal that immediately fails back)
    if best_move_idx < len(future_prices) - 1:
        post_move_prices = future_prices[best_move_idx:]

        if direction == 1:  # Bullish: check for drop back
            bounce_down = future_prices[best_move_idx] - np.min(post_move_prices)
            bounce_pct = bounce_down / start_price
            max_bounce = bounce_threshold_pct * min_move_pct
            if bounce_pct > max_bounce:
                return False, move_pct, best_move_idx
        else:  # Bearish: check for bounce back up
            bounce_up = np.max(post_move_prices) - future_prices[best_move_idx]
            bounce_pct = bounce_up / start_price
            max_bounce = bounce_threshold_pct * min_move_pct
            if bounce_pct > max_bounce:
                return False, move_pct, best_move_idx

    return True, move_pct, best_move_idx


def label_strong_reversals(
    ohlcv: pd.DataFrame,
    min_move_pct: float = 0.002,
    slope_window: int = 20,
    min_slope: float = 1e-4,
    validation_bars: int = 15,
    bounce_threshold_pct: float = 0.5
) -> pd.DataFrame:
    """
    Label bars where strong reversals BEGIN.

    This function identifies reversal points using strict criteria:
    1. Prior trend exists (slope magnitude > threshold)
    2. Move in opposite direction exceeds min_move_pct
    3. Reversal sustains without excessive bounce-back

    Args:
        ohlcv: DataFrame with trading_day, open, high, low, close, volume
        min_move_pct: Minimum % move for valid reversal (default 0.2%)
        slope_window: Bars for slope calculation
        min_slope: Minimum slope magnitude to detect trend
        validation_bars: Bars to check for sustained move
        bounce_threshold_pct: Max bounce as fraction of move

    Returns:
        DataFrame with added columns:
        - reversal_label: 1 (bull), -1 (bear), 0 (none)
        - reversal_magnitude: Actual move size as %
        - reversal_duration: Bars until move peak
        - prior_trend: Slope before reversal
    """
    result = ohlcv.copy()

    # Initialize label columns
    result['reversal_label'] = 0
    result['reversal_magnitude'] = 0.0
    result['reversal_duration'] = 0
    result['prior_trend'] = 0.0

    # Process by day to respect boundaries
    for day, day_df in ohlcv.groupby('trading_day'):
        day_indices = day_df.index.tolist()
        prices = day_df['close'].values.astype(np.float64)
        n = len(prices)

        if n < slope_window + validation_bars:
            continue

        # Compute rolling slope
        slopes = _compute_rolling_slope(prices, slope_window)

        # Find potential reversal points
        local_max_mask, local_min_mask = _find_local_extrema(
            prices, slopes, min_slope
        )

        # Validate each potential reversal
        for i in range(n):
            idx = day_indices[i]

            # Check for bullish reversal (local min = price was falling)
            if local_min_mask[i]:
                is_valid, magnitude, duration = _validate_reversal(
                    prices, i,
                    direction=1,  # Bullish
                    min_move_pct=min_move_pct,
                    validation_bars=validation_bars,
                    bounce_threshold_pct=bounce_threshold_pct
                )

                if is_valid:
                    result.loc[idx, 'reversal_label'] = ReversalLabel.BULL
                    result.loc[idx, 'reversal_magnitude'] = magnitude
                    result.loc[idx, 'reversal_duration'] = duration
                    result.loc[idx, 'prior_trend'] = slopes[i - 1] if i > 0 else 0

            # Check for bearish reversal (local max = price was rising)
            elif local_max_mask[i]:
                is_valid, magnitude, duration = _validate_reversal(
                    prices, i,
                    direction=-1,  # Bearish
                    min_move_pct=min_move_pct,
                    validation_bars=validation_bars,
                    bounce_threshold_pct=bounce_threshold_pct
                )

                if is_valid:
                    result.loc[idx, 'reversal_label'] = ReversalLabel.BEAR
                    result.loc[idx, 'reversal_magnitude'] = magnitude
                    result.loc[idx, 'reversal_duration'] = duration
                    result.loc[idx, 'prior_trend'] = slopes[i - 1] if i > 0 else 0

    return result


class ReversalLabeler:
    """
    Class-based interface for reversal labeling with configuration.

    Allows caching of labeled data and incremental processing.
    """

    def __init__(
        self,
        min_move_pct: float = 0.002,
        slope_window: int = 20,
        min_slope: float = 1e-4,
        validation_bars: int = 15,
        bounce_threshold_pct: float = 0.5
    ):
        """Initialize labeler with configuration."""
        self.config = ReversalConfig(
            min_move_pct=min_move_pct,
            slope_window=slope_window,
            min_slope=min_slope,
            validation_bars=validation_bars,
            bounce_threshold_pct=bounce_threshold_pct
        )

        self._labeled_data: Optional[pd.DataFrame] = None
        self._label_stats: Dict[str, Any] = {}

    def fit(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Label reversals in the dataset.

        Args:
            ohlcv: DataFrame with OHLCV data

        Returns:
            DataFrame with reversal labels added
        """
        self._labeled_data = label_strong_reversals(
            ohlcv,
            min_move_pct=self.config.min_move_pct,
            slope_window=self.config.slope_window,
            min_slope=self.config.min_slope,
            validation_bars=self.config.validation_bars,
            bounce_threshold_pct=self.config.bounce_threshold_pct
        )

        # Compute statistics
        labels = self._labeled_data['reversal_label']
        self._label_stats = {
            'total_bars': len(labels),
            'n_bull_reversals': (labels == ReversalLabel.BULL).sum(),
            'n_bear_reversals': (labels == ReversalLabel.BEAR).sum(),
            'n_no_reversal': (labels == ReversalLabel.NONE).sum(),
            'reversal_rate': (labels != ReversalLabel.NONE).mean(),
            'avg_magnitude': self._labeled_data.loc[
                labels != ReversalLabel.NONE, 'reversal_magnitude'
            ].mean(),
            'avg_duration': self._labeled_data.loc[
                labels != ReversalLabel.NONE, 'reversal_duration'
            ].mean(),
        }

        # Per-day stats
        daily_stats = self._labeled_data.groupby('trading_day').agg({
            'reversal_label': lambda x: (x != ReversalLabel.NONE).sum()
        }).rename(columns={'reversal_label': 'reversals_per_day'})

        self._label_stats['avg_reversals_per_day'] = daily_stats['reversals_per_day'].mean()
        self._label_stats['std_reversals_per_day'] = daily_stats['reversals_per_day'].std()

        return self._labeled_data

    def get_stats(self) -> Dict[str, Any]:
        """Get labeling statistics."""
        return self._label_stats

    def print_summary(self) -> None:
        """Print labeling summary."""
        if not self._label_stats:
            print("No data labeled yet. Call fit() first.")
            return

        s = self._label_stats
        print("\n" + "=" * 50)
        print("REVERSAL LABELING SUMMARY")
        print("=" * 50)
        print(f"Total bars: {s['total_bars']:,}")
        print(f"Bull reversals: {s['n_bull_reversals']:,}")
        print(f"Bear reversals: {s['n_bear_reversals']:,}")
        print(f"Reversal rate: {s['reversal_rate']:.2%}")
        print(f"Avg reversals/day: {s['avg_reversals_per_day']:.1f} "
              f"(std: {s['std_reversals_per_day']:.1f})")
        print(f"Avg magnitude: {s['avg_magnitude']:.3%}")
        print(f"Avg duration: {s['avg_duration']:.1f} bars")
        print("=" * 50)

    def get_reversal_bars(self) -> pd.DataFrame:
        """Get only bars labeled as reversals."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")

        return self._labeled_data[
            self._labeled_data['reversal_label'] != ReversalLabel.NONE
        ]

    def get_binary_labels(self) -> np.ndarray:
        """Get binary labels (1 = any reversal, 0 = none)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")

        return (self._labeled_data['reversal_label'] != ReversalLabel.NONE).astype(int).values

    def get_directional_labels(self) -> np.ndarray:
        """Get directional labels (-1, 0, 1)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")

        return self._labeled_data['reversal_label'].values

    def get_multiclass_labels(self) -> np.ndarray:
        """Get multiclass labels (0=none, 1=bull, 2=bear)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")

        labels = self._labeled_data['reversal_label'].values.copy()
        # Map: -1 (bear) -> 2, 0 (none) -> 0, 1 (bull) -> 1
        labels[labels == -1] = 2
        return labels.astype(int)
