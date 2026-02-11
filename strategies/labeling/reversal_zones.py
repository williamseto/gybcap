"""
Trade-viable zone labeling for reversal prediction.

This module provides a trading-centric approach to labeling reversal zones.
Instead of labeling only the exact reversal bar (which is too late to trade),
or arbitrary windows (which ignore price action), we label bars where
entering a trade would actually be profitable.

Key insight: A bar is a valid entry if entering a trade there wouldn't get
stopped out before the reversal completes.

For each reversal at bar i with magnitude M:
- Walk backwards from bar i
- For each bar j < i:
  - Compute max adverse excursion (MAE) if entered at bar j
  - If MAE < stop_loss_pct, bar j is a valid entry
  - Stop when MAE exceeds stop_loss

This naturally accounts for volatility before the reversal and aligns
the ML objective with the trading objective.

Expected label distribution:
- Current exact-bar: ~0.11% positive rate
- Trade-viable zones: ~0.3-0.5% positive rate (3-4 valid entries per reversal)
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
import pandas as pd
import numpy as np
from scipy.stats import linregress


# Canonical list of tracked price levels — import this everywhere
TRACKED_LEVELS = [
    'vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi',
    'prev_high', 'prev_low', 'ib_lo', 'ib_hi',
]


class ZoneLabel(IntEnum):
    """Zone label values."""
    NO_ZONE = 0
    BULL_ZONE = 1   # Valid entry for bullish reversal
    BEAR_ZONE = -1  # Valid entry for bearish reversal


@dataclass
class ZoneConfig:
    """Configuration for trade-viable zone labeling."""
    stop_loss_pct: float = 0.0015      # 0.15% stop (3 ES points at ~5000)
    min_reward_risk: float = 1.5       # Only label if R:R > 1.5
    max_lookback_bars: int = 10        # Max bars before reversal to check
    min_move_pct: float = 0.002        # Minimum reversal magnitude (0.2%)
    slope_window: int = 20             # Bars for slope calculation
    min_slope: float = 1e-4            # Minimum slope to detect trend
    validation_bars: int = 15          # Bars to validate reversal holds
    bounce_threshold_pct: float = 0.5  # Max bounce as fraction of move


@dataclass
class ZoneStats:
    """Statistics for zone labeling results."""
    total_bars: int
    n_bull_zones: int
    n_bear_zones: int
    n_no_zone: int
    n_reversals: int
    avg_zone_size: float
    avg_mae_in_zone: float
    avg_reward: float
    positive_rate: float
    avg_zones_per_day: float


def _compute_rolling_slope(prices: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling linear regression slope."""
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


def _find_reversal_candidates(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    slopes: np.ndarray,
    min_slope: float,
    min_move_pct: float,
    validation_bars: int,
    bounce_threshold_pct: float
) -> List[Dict[str, Any]]:
    """
    Find validated reversal points with magnitude.

    Returns list of dicts with:
    - index: bar index where reversal starts
    - direction: 1 (bull) or -1 (bear)
    - magnitude: achieved move as fraction
    - duration: bars until peak move
    """
    n = len(prices)
    reversals = []

    for i in range(1, n - 1):
        if np.isnan(slopes[i]) or np.isnan(slopes[i - 1]):
            continue

        prev_slope = slopes[i - 1]
        curr_slope = slopes[i]

        # Check for bullish reversal (slope was negative, now positive)
        if prev_slope < -min_slope and curr_slope > min_slope:
            # Validate the move
            is_valid, magnitude, duration = _validate_reversal(
                prices, highs, lows, i,
                direction=1,
                min_move_pct=min_move_pct,
                validation_bars=validation_bars,
                bounce_threshold_pct=bounce_threshold_pct
            )
            if is_valid:
                reversals.append({
                    'index': i,
                    'direction': 1,
                    'magnitude': magnitude,
                    'duration': duration,
                    'entry_price': prices[i]
                })

        # Check for bearish reversal (slope was positive, now negative)
        elif prev_slope > min_slope and curr_slope < -min_slope:
            is_valid, magnitude, duration = _validate_reversal(
                prices, highs, lows, i,
                direction=-1,
                min_move_pct=min_move_pct,
                validation_bars=validation_bars,
                bounce_threshold_pct=bounce_threshold_pct
            )
            if is_valid:
                reversals.append({
                    'index': i,
                    'direction': -1,
                    'magnitude': magnitude,
                    'duration': duration,
                    'entry_price': prices[i]
                })

    return reversals


def _validate_reversal(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    direction: int,
    min_move_pct: float,
    validation_bars: int,
    bounce_threshold_pct: float
) -> Tuple[bool, float, int]:
    """Validate reversal meets minimum criteria."""
    n = len(prices)
    end_idx = min(start_idx + validation_bars, n)

    if end_idx - start_idx < 3:
        return False, 0.0, 0

    start_price = prices[start_idx]

    if direction == 1:  # Bullish: expect price to go up
        future_highs = highs[start_idx:end_idx]
        best_move = (future_highs.max() - start_price) / start_price
        best_move_idx = int(np.argmax(future_highs))
    else:  # Bearish: expect price to go down
        future_lows = lows[start_idx:end_idx]
        best_move = (start_price - future_lows.min()) / start_price
        best_move_idx = int(np.argmin(future_lows))

    if best_move < min_move_pct:
        return False, best_move, best_move_idx

    # Check for disqualifying bounce after the move
    if best_move_idx < end_idx - start_idx - 1:
        if direction == 1:
            post_move_lows = lows[start_idx + best_move_idx:end_idx]
            peak_price = highs[start_idx + best_move_idx]
            bounce = (peak_price - post_move_lows.min()) / start_price
            if bounce > bounce_threshold_pct * min_move_pct:
                return False, best_move, best_move_idx
        else:
            post_move_highs = highs[start_idx + best_move_idx:end_idx]
            trough_price = lows[start_idx + best_move_idx]
            bounce = (post_move_highs.max() - trough_price) / start_price
            if bounce > bounce_threshold_pct * min_move_pct:
                return False, best_move, best_move_idx

    return True, best_move, best_move_idx


def _compute_mae_for_entry(
    entry_idx: int,
    reversal_idx: int,
    direction: int,
    entry_price: float,
    highs: np.ndarray,
    lows: np.ndarray
) -> float:
    """
    Compute max adverse excursion if entering at entry_idx.

    For bull reversal: MAE = how far price dropped after entry
    For bear reversal: MAE = how far price rose after entry
    """
    if direction == 1:  # Bull reversal - went long
        # MAE is how far price dropped after entry before reversal
        min_low = lows[entry_idx:reversal_idx + 1].min()
        mae = (entry_price - min_low) / entry_price
    else:  # Bear reversal - went short
        # MAE is how far price rose after entry before reversal
        max_high = highs[entry_idx:reversal_idx + 1].max()
        mae = (max_high - entry_price) / entry_price

    return max(0.0, mae)


def compute_trade_viable_zones(
    ohlcv: pd.DataFrame,
    stop_loss_pct: float = 0.0015,
    min_reward_risk: float = 1.5,
    max_lookback_bars: int = 10,
    min_move_pct: float = 0.002,
    slope_window: int = 20,
    min_slope: float = 1e-4,
    validation_bars: int = 15,
    bounce_threshold_pct: float = 0.5
) -> pd.DataFrame:
    """
    Label bars where entering a reversal trade would be profitable.

    For each reversal, walks backwards to find valid entry bars where
    the max adverse excursion (MAE) would not exceed the stop loss.

    Args:
        ohlcv: DataFrame with trading_day, open, high, low, close, volume
        stop_loss_pct: Maximum allowed MAE (default 0.15%)
        min_reward_risk: Minimum reward:risk ratio required
        max_lookback_bars: Max bars before reversal to check for entries
        min_move_pct: Minimum reversal magnitude to consider
        slope_window: Bars for trend slope calculation
        min_slope: Minimum slope to detect trend
        validation_bars: Bars to validate reversal sustains
        bounce_threshold_pct: Max bounce as fraction of move

    Returns:
        DataFrame with columns:
        - zone_label: 0 (no zone), 1 (bull zone), -1 (bear zone)
        - max_adverse_excursion: MAE if entered at this bar
        - potential_reward: Expected reward based on reversal magnitude
        - bars_to_reversal: Distance to actual reversal bar
        - reward_risk_ratio: potential_reward / stop_loss
        - reversal_magnitude: The magnitude of the associated reversal
    """
    result = ohlcv.copy()

    # Initialize zone columns
    result['zone_label'] = ZoneLabel.NO_ZONE
    result['max_adverse_excursion'] = 0.0
    result['potential_reward'] = 0.0
    result['bars_to_reversal'] = 0
    result['reward_risk_ratio'] = 0.0
    result['reversal_magnitude'] = 0.0

    # Process by day to respect boundaries
    for day, day_df in ohlcv.groupby('trading_day'):
        day_indices = day_df.index.tolist()
        prices = day_df['close'].values.astype(np.float64)
        highs = day_df['high'].values.astype(np.float64)
        lows = day_df['low'].values.astype(np.float64)
        n = len(prices)

        if n < slope_window + validation_bars:
            continue

        # Compute rolling slope for reversal detection
        slopes = _compute_rolling_slope(prices, slope_window)

        # Find validated reversals
        reversals = _find_reversal_candidates(
            prices, highs, lows, slopes,
            min_slope=min_slope,
            min_move_pct=min_move_pct,
            validation_bars=validation_bars,
            bounce_threshold_pct=bounce_threshold_pct
        )

        # For each reversal, find valid entry bars
        for rev in reversals:
            rev_idx = rev['index']
            direction = rev['direction']
            magnitude = rev['magnitude']
            entry_price_at_rev = rev['entry_price']

            # Check reward:risk at the reversal bar itself
            rr_at_rev = magnitude / stop_loss_pct if stop_loss_pct > 0 else 0
            if rr_at_rev < min_reward_risk:
                continue  # Skip this reversal entirely

            # Walk backwards to find valid entry bars
            for lookback in range(0, min(max_lookback_bars + 1, rev_idx + 1)):
                entry_bar_local = rev_idx - lookback
                if entry_bar_local < 0:
                    break

                entry_idx = day_indices[entry_bar_local]
                entry_price = prices[entry_bar_local]

                # Compute MAE for this entry
                mae = _compute_mae_for_entry(
                    entry_bar_local, rev_idx, direction,
                    entry_price, highs, lows
                )

                if mae > stop_loss_pct:
                    break  # Stop looking - would have been stopped out

                # Compute potential reward from this entry point
                if direction == 1:  # Bull reversal
                    # Reward is from entry to the reversal's peak
                    future_highs = highs[rev_idx:min(rev_idx + validation_bars, n)]
                    if len(future_highs) > 0:
                        reward = (future_highs.max() - entry_price) / entry_price
                    else:
                        reward = magnitude
                else:  # Bear reversal
                    future_lows = lows[rev_idx:min(rev_idx + validation_bars, n)]
                    if len(future_lows) > 0:
                        reward = (entry_price - future_lows.min()) / entry_price
                    else:
                        reward = magnitude

                # Check reward:risk ratio
                rr_ratio = reward / stop_loss_pct if stop_loss_pct > 0 else 0
                if rr_ratio < min_reward_risk:
                    continue  # This entry doesn't have good enough R:R

                # Valid entry - label it
                result.loc[entry_idx, 'zone_label'] = direction
                result.loc[entry_idx, 'max_adverse_excursion'] = mae
                result.loc[entry_idx, 'potential_reward'] = reward
                result.loc[entry_idx, 'bars_to_reversal'] = lookback
                result.loc[entry_idx, 'reward_risk_ratio'] = rr_ratio
                result.loc[entry_idx, 'reversal_magnitude'] = magnitude

    return result


class TradeViableZoneLabeler:
    """
    Label bars where entering a trade would be profitable.

    For each reversal at bar i with magnitude M:
    - Walk backwards from bar i
    - For each bar j < i:
      - Compute max adverse excursion (MAE) if entered at bar j
      - If MAE < stop_loss_pct, bar j is a valid entry
      - Stop when MAE exceeds stop_loss

    This is trading-centric:
    - Only labels bars where a trade would work
    - Naturally accounts for volatility before reversal
    - Aligns ML objective with trading objective
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.0015,
        min_reward_risk: float = 1.5,
        max_lookback_bars: int = 10,
        min_move_pct: float = 0.002,
        slope_window: int = 20,
        min_slope: float = 1e-4,
        validation_bars: int = 15,
        bounce_threshold_pct: float = 0.5
    ):
        """
        Initialize the zone labeler.

        Args:
            stop_loss_pct: Maximum allowed MAE (0.15% = 3 ES points)
            min_reward_risk: Minimum reward:risk ratio (1.5 = risk $1 to make $1.50)
            max_lookback_bars: How far back to look for valid entries
            min_move_pct: Minimum reversal magnitude to consider
            slope_window: Bars for slope calculation
            min_slope: Minimum slope to detect trend
            validation_bars: Bars to validate reversal sustains
            bounce_threshold_pct: Max bounce as fraction of move
        """
        self.config = ZoneConfig(
            stop_loss_pct=stop_loss_pct,
            min_reward_risk=min_reward_risk,
            max_lookback_bars=max_lookback_bars,
            min_move_pct=min_move_pct,
            slope_window=slope_window,
            min_slope=min_slope,
            validation_bars=validation_bars,
            bounce_threshold_pct=bounce_threshold_pct
        )

        self._labeled_data: Optional[pd.DataFrame] = None
        self._stats: Optional[ZoneStats] = None

    def fit(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trade-viable zone labels for the dataset.

        Args:
            ohlcv: DataFrame with OHLCV data and trading_day

        Returns:
            DataFrame with zone labels added
        """
        self._labeled_data = compute_trade_viable_zones(
            ohlcv,
            stop_loss_pct=self.config.stop_loss_pct,
            min_reward_risk=self.config.min_reward_risk,
            max_lookback_bars=self.config.max_lookback_bars,
            min_move_pct=self.config.min_move_pct,
            slope_window=self.config.slope_window,
            min_slope=self.config.min_slope,
            validation_bars=self.config.validation_bars,
            bounce_threshold_pct=self.config.bounce_threshold_pct
        )

        # Compute statistics
        self._compute_stats()

        return self._labeled_data

    def _compute_stats(self) -> None:
        """Compute zone labeling statistics."""
        if self._labeled_data is None:
            return

        labels = self._labeled_data['zone_label']
        n_bull = (labels == ZoneLabel.BULL_ZONE).sum()
        n_bear = (labels == ZoneLabel.BEAR_ZONE).sum()
        n_none = (labels == ZoneLabel.NO_ZONE).sum()
        total = len(labels)

        # Zone samples
        zone_mask = labels != ZoneLabel.NO_ZONE
        zone_data = self._labeled_data[zone_mask]

        # Count unique reversals (bars_to_reversal == 0 marks the reversal bar itself)
        reversal_bars = zone_data[zone_data['bars_to_reversal'] == 0]
        n_reversals = len(reversal_bars)

        # Average zone size (entries per reversal)
        if n_reversals > 0:
            avg_zone_size = len(zone_data) / n_reversals
        else:
            avg_zone_size = 0.0

        # Average MAE and reward in zones
        avg_mae = zone_data['max_adverse_excursion'].mean() if len(zone_data) > 0 else 0.0
        avg_reward = zone_data['potential_reward'].mean() if len(zone_data) > 0 else 0.0

        # Positive rate
        positive_rate = (n_bull + n_bear) / total if total > 0 else 0.0

        # Average zones per day
        daily_zones = self._labeled_data.groupby('trading_day').apply(
            lambda x: (x['zone_label'] != ZoneLabel.NO_ZONE).sum()
        )
        avg_zones_per_day = daily_zones.mean() if len(daily_zones) > 0 else 0.0

        self._stats = ZoneStats(
            total_bars=total,
            n_bull_zones=int(n_bull),
            n_bear_zones=int(n_bear),
            n_no_zone=int(n_none),
            n_reversals=n_reversals,
            avg_zone_size=avg_zone_size,
            avg_mae_in_zone=avg_mae,
            avg_reward=avg_reward,
            positive_rate=positive_rate,
            avg_zones_per_day=avg_zones_per_day
        )

    def get_stats(self) -> ZoneStats:
        """Get zone labeling statistics."""
        if self._stats is None:
            raise ValueError("Must call fit() first")
        return self._stats

    def print_summary(self) -> None:
        """Print zone labeling summary."""
        if self._stats is None:
            print("No data labeled yet. Call fit() first.")
            return

        s = self._stats
        print("\n" + "=" * 60)
        print("TRADE-VIABLE ZONE LABELING SUMMARY")
        print("=" * 60)
        print(f"Total bars:           {s.total_bars:,}")
        print(f"Bull zone bars:       {s.n_bull_zones:,}")
        print(f"Bear zone bars:       {s.n_bear_zones:,}")
        print(f"No zone bars:         {s.n_no_zone:,}")
        print("-" * 60)
        print(f"Unique reversals:     {s.n_reversals:,}")
        print(f"Avg entries/reversal: {s.avg_zone_size:.1f}")
        print(f"Positive rate:        {s.positive_rate:.2%}")
        print(f"Avg zones/day:        {s.avg_zones_per_day:.1f}")
        print("-" * 60)
        print(f"Avg MAE in zones:     {s.avg_mae_in_zone:.4%}")
        print(f"Avg potential reward: {s.avg_reward:.4%}")
        print(f"Stop loss setting:    {self.config.stop_loss_pct:.4%}")
        print(f"Min R:R ratio:        {self.config.min_reward_risk:.1f}")
        print("=" * 60)

    def get_zone_bars(self) -> pd.DataFrame:
        """Get only bars labeled as part of a zone."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")
        return self._labeled_data[
            self._labeled_data['zone_label'] != ZoneLabel.NO_ZONE
        ]

    def get_binary_labels(self) -> np.ndarray:
        """Get binary labels (1 = in any zone, 0 = none)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")
        return (self._labeled_data['zone_label'] != ZoneLabel.NO_ZONE).astype(int).values

    def get_directional_labels(self) -> np.ndarray:
        """Get directional labels (-1, 0, 1)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")
        return self._labeled_data['zone_label'].values

    def get_multiclass_labels(self) -> np.ndarray:
        """Get multiclass labels (0=none, 1=bull, 2=bear)."""
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")
        labels = self._labeled_data['zone_label'].values.copy()
        # Map: -1 (bear) -> 2, 0 (none) -> 0, 1 (bull) -> 1
        labels[labels == -1] = 2
        return labels.astype(int)

    def get_zone_features(self) -> pd.DataFrame:
        """
        Get zone-specific features for ML.

        Returns:
            DataFrame with zone-aware features:
            - bars_to_reversal: Distance to reversal (0 at reversal bar)
            - max_adverse_excursion: MAE if entered at this bar
            - potential_reward: Expected reward
            - reward_risk_ratio: R:R ratio
        """
        if self._labeled_data is None:
            raise ValueError("Must call fit() first")

        return self._labeled_data[[
            'bars_to_reversal',
            'max_adverse_excursion',
            'potential_reward',
            'reward_risk_ratio',
            'reversal_magnitude'
        ]].copy()


class LevelAnchoredZoneLabeler:
    """
    Find reversal zones anchored to tracked price levels.

    Pipeline:
    1. Find all reversals via slope-based detection (existing logic)
    2. Filter: keep only reversals where price is within proximity of a tracked level
    3. Walk backwards from reversal to find zone bars (MAE < stop)
    4. Label zone bars with distance-weighted probability

    This produces training data for the V3 causal zone model.
    """

    def __init__(
        self,
        level_proximity_pts: float = 3.0,
        zone_config: Optional[ZoneConfig] = None,
        decay_alpha: float = 0.5,
    ):
        """
        Args:
            level_proximity_pts: Max distance (ES points) from a level for a
                reversal to be considered level-anchored.
            zone_config: Configuration for zone detection. Uses defaults if None.
            decay_alpha: Exponential decay rate for zone probability.
                P = exp(-alpha * bars_to_reversal).
        """
        self.level_proximity_pts = level_proximity_pts
        self.config = zone_config or ZoneConfig()
        self.decay_alpha = decay_alpha

        self._labeled_data: Optional[pd.DataFrame] = None
        self._n_reversals_total: int = 0
        self._n_reversals_anchored: int = 0

    def fit(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Label bars near levels that are in pre-reversal zones.

        Expects level columns (vwap, ovn_lo, etc.) already present in ohlcv,
        plus prev_high and prev_low computed from trading_day groupby.

        Args:
            ohlcv: DataFrame with OHLCV, trading_day, and level columns.

        Returns:
            DataFrame with zone columns added.
        """
        result = ohlcv.copy()

        # Initialize output columns
        result['zone_label'] = ZoneLabel.NO_ZONE
        result['zone_probability'] = 0.0
        result['nearest_level'] = ''
        result['level_distance'] = np.nan
        result['bars_to_reversal'] = 0

        # Compute prev_high/prev_low if not present
        if 'prev_high' not in result.columns:
            daily = ohlcv.groupby('trading_day').agg(
                day_high=('high', 'max'),
                day_low=('low', 'min'),
            )
            daily['prev_high'] = daily['day_high'].shift(1)
            daily['prev_low'] = daily['day_low'].shift(1)
            prev_map_hi = daily['prev_high'].to_dict()
            prev_map_lo = daily['prev_low'].to_dict()
            result['prev_high'] = result['trading_day'].map(prev_map_hi)
            result['prev_low'] = result['trading_day'].map(prev_map_lo)

        cfg = self.config
        self._n_reversals_total = 0
        self._n_reversals_anchored = 0

        for day, day_df in result.groupby('trading_day'):
            day_indices = day_df.index.tolist()
            prices = day_df['close'].values.astype(np.float64)
            highs = day_df['high'].values.astype(np.float64)
            lows = day_df['low'].values.astype(np.float64)
            n = len(prices)

            if n < cfg.slope_window + cfg.validation_bars:
                continue

            # Compute rolling slope
            slopes = _compute_rolling_slope(prices, cfg.slope_window)

            # Find validated reversals
            reversals = _find_reversal_candidates(
                prices, highs, lows, slopes,
                min_slope=cfg.min_slope,
                min_move_pct=cfg.min_move_pct,
                validation_bars=cfg.validation_bars,
                bounce_threshold_pct=cfg.bounce_threshold_pct,
            )

            self._n_reversals_total += len(reversals)

            # Collect level values for this day
            level_values = {}
            for lvl_name in TRACKED_LEVELS:
                if lvl_name in day_df.columns:
                    vals = day_df[lvl_name].values
                    # Use median non-NaN value (levels are constant per day)
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0 and valid[0] != 0:
                        level_values[lvl_name] = float(np.median(valid))

            if not level_values:
                continue

            for rev in reversals:
                rev_idx_local = rev['index']
                direction = rev['direction']
                magnitude = rev['magnitude']
                rev_price = prices[rev_idx_local]

                # Check reward:risk at reversal bar
                rr = magnitude / cfg.stop_loss_pct if cfg.stop_loss_pct > 0 else 0
                if rr < cfg.min_reward_risk:
                    continue

                # Find nearest level
                nearest_name = None
                nearest_dist = np.inf
                for lvl_name, lvl_price in level_values.items():
                    dist = abs(rev_price - lvl_price)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_name = lvl_name

                if nearest_dist > self.level_proximity_pts:
                    continue  # Not near any level

                self._n_reversals_anchored += 1

                # Walk backwards to find valid zone bars
                for lookback in range(0, min(cfg.max_lookback_bars + 1, rev_idx_local + 1)):
                    entry_local = rev_idx_local - lookback
                    if entry_local < 0:
                        break

                    entry_idx = day_indices[entry_local]
                    entry_price = prices[entry_local]

                    mae = _compute_mae_for_entry(
                        entry_local, rev_idx_local, direction,
                        entry_price, highs, lows,
                    )

                    if mae > cfg.stop_loss_pct:
                        break

                    # Compute reward
                    end_local = min(rev_idx_local + cfg.validation_bars, n)
                    if direction == 1:
                        future_highs = highs[rev_idx_local:end_local]
                        reward = (future_highs.max() - entry_price) / entry_price if len(future_highs) > 0 else magnitude
                    else:
                        future_lows = lows[rev_idx_local:end_local]
                        reward = (entry_price - future_lows.min()) / entry_price if len(future_lows) > 0 else magnitude

                    rr_ratio = reward / cfg.stop_loss_pct if cfg.stop_loss_pct > 0 else 0
                    if rr_ratio < cfg.min_reward_risk:
                        continue

                    # Zone probability: decays with distance from reversal
                    zone_prob = np.exp(-self.decay_alpha * lookback)

                    # Label this bar
                    result.loc[entry_idx, 'zone_label'] = direction
                    result.loc[entry_idx, 'zone_probability'] = zone_prob
                    result.loc[entry_idx, 'nearest_level'] = nearest_name
                    result.loc[entry_idx, 'level_distance'] = nearest_dist
                    result.loc[entry_idx, 'bars_to_reversal'] = lookback

        self._labeled_data = result
        return result

    def print_summary(self) -> None:
        """Print labeling summary."""
        if self._labeled_data is None:
            print("No data labeled yet. Call fit() first.")
            return

        labels = self._labeled_data['zone_label']
        n_bull = int((labels == ZoneLabel.BULL_ZONE).sum())
        n_bear = int((labels == ZoneLabel.BEAR_ZONE).sum())
        total = len(labels)
        zone_mask = labels != ZoneLabel.NO_ZONE

        n_days = self._labeled_data['trading_day'].nunique()
        zone_bars = self._labeled_data[zone_mask]

        print("\n" + "=" * 60)
        print("LEVEL-ANCHORED ZONE LABELING SUMMARY")
        print("=" * 60)
        print(f"Total bars:              {total:,}")
        print(f"Trading days:            {n_days}")
        print(f"Total reversals found:   {self._n_reversals_total}")
        print(f"Level-anchored reversals:{self._n_reversals_anchored}")
        print(f"  ({self._n_reversals_anchored / max(self._n_reversals_total, 1):.0%} "
              f"near a tracked level)")
        print(f"Bull zone bars:          {n_bull}")
        print(f"Bear zone bars:          {n_bear}")
        print(f"Positive rate:           {(n_bull + n_bear) / total:.4%}")
        if len(zone_bars) > 0:
            print(f"Avg zone probability:    {zone_bars['zone_probability'].mean():.3f}")
            print(f"Avg bars to reversal:    {zone_bars['bars_to_reversal'].mean():.1f}")
            # Level breakdown
            level_counts = zone_bars['nearest_level'].value_counts()
            print("\nZone bars by level:")
            for lvl, cnt in level_counts.items():
                print(f"  {lvl:15s}: {cnt:5d}")
        print("=" * 60)


class ReversalBreakoutLabeler:
    """
    Label near-level bars with P(reversal) based on structural outcome.

    For each bar within proximity of a tracked level:
    1. Determine side (above/below level)
    2. Look forward: does price reverse from or break through the level?
    3. Assign P(reversal) based on outcome

    Reversal = level holds, price bounces away from level
    Breakout = price pushes through level and continues

    Key differences from LevelAnchoredZoneLabeler:
    - Labels structural events, not trade viability
    - P(reversal) is NOT conditioned on stop/target parameters
    - Includes breakout bars as explicit negative class
    - Evaluates on ALL near-level bars (real base rate)
    """

    def __init__(
        self,
        proximity_pts: float = 5.0,
        forward_window: int = 45,
        reversal_threshold_pts: float = 6.0,
        breakout_threshold_pts: float = 4.0,
        decay_alpha: float = 0.3,
    ):
        """
        Args:
            proximity_pts: Max distance from level to be "near-level".
            forward_window: Bars to look forward for outcome.
            reversal_threshold_pts: Min move in reversal direction (away
                from level) to classify as reversal.
            breakout_threshold_pts: Min move past the level to classify
                as breakout.
            decay_alpha: P(reversal) = exp(-alpha * bars_to_event).
        """
        self.proximity_pts = proximity_pts
        self.forward_window = forward_window
        self.reversal_threshold_pts = reversal_threshold_pts
        self.breakout_threshold_pts = breakout_threshold_pts
        self.decay_alpha = decay_alpha

        self._labeled_data: Optional[pd.DataFrame] = None
        self._stats: Dict[str, Any] = {}

    def fit(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Label all near-level bars with structural outcome.

        For bars below the level (level = resistance above):
          - Reversal = price drops from entry by threshold (rejection)
          - Breakout = price rises past level by threshold

        For bars above the level (level = support below):
          - Reversal = price rises from entry by threshold (bounce)
          - Breakout = price drops past level by threshold

        Returns DataFrame with columns:
          near_level, p_reversal, outcome, nearest_level,
          level_distance, side, bars_to_event, trade_direction
        """
        result = ohlcv.copy()

        n_total = len(result)
        near_level = np.zeros(n_total, dtype=bool)
        p_reversal = np.full(n_total, np.nan)
        outcome_arr = np.empty(n_total, dtype=object)
        outcome_arr[:] = ''
        nearest_level_arr = np.empty(n_total, dtype=object)
        nearest_level_arr[:] = ''
        level_distance_arr = np.full(n_total, np.nan)
        side_arr = np.zeros(n_total, dtype=np.int8)
        bars_to_event_arr = np.zeros(n_total, dtype=np.int32)
        trade_dir_arr = np.zeros(n_total, dtype=np.int8)

        n_reversal = 0
        n_breakout = 0
        n_inconclusive = 0

        # Compute prev_high/prev_low if absent
        if 'prev_high' not in result.columns:
            daily = ohlcv.groupby('trading_day').agg(
                day_high=('high', 'max'), day_low=('low', 'min'),
            )
            daily['prev_high'] = daily['day_high'].shift(1)
            daily['prev_low'] = daily['day_low'].shift(1)
            result['prev_high'] = result['trading_day'].map(
                daily['prev_high'].to_dict()
            )
            result['prev_low'] = result['trading_day'].map(
                daily['prev_low'].to_dict()
            )

        close_arr = result['close'].values.astype(np.float64)
        high_arr = result['high'].values.astype(np.float64)
        low_arr = result['low'].values.astype(np.float64)

        for day, day_df in result.groupby('trading_day'):
            day_indices = np.where(
                (result['trading_day'] == day).values
            )[0]
            n_day = len(day_indices)
            if n_day < 20:
                continue

            # Collect level values for this day
            level_values = {}
            for lvl_name in TRACKED_LEVELS:
                if lvl_name in day_df.columns:
                    vals = day_df[lvl_name].values
                    valid = vals[~np.isnan(vals)]
                    if len(valid) > 0 and valid[0] != 0:
                        level_values[lvl_name] = float(np.median(valid))

            if not level_values:
                continue

            for local_i in range(n_day):
                gi = day_indices[local_i]
                c = close_arr[gi]

                # Find nearest level
                best_name = None
                best_dist = np.inf
                best_price = 0.0
                for lvl_name, lvl_price in level_values.items():
                    d = abs(c - lvl_price)
                    if d < best_dist:
                        best_dist = d
                        best_name = lvl_name
                        best_price = lvl_price

                if best_dist > self.proximity_pts:
                    continue

                near_level[gi] = True
                nearest_level_arr[gi] = best_name
                level_distance_arr[gi] = best_dist

                # Side: 1 = above level (support), -1 = below level (resistance)
                s = 1 if c >= best_price else -1
                side_arr[gi] = s
                # Trade direction for reversal: same as side
                # Above support → reversal = bounce UP → long
                # Below resistance → reversal = drop DOWN → short
                trade_dir_arr[gi] = s

                # Forward window (within day)
                fwd_end = min(local_i + 1 + self.forward_window, n_day)
                fwd_gi = day_indices[local_i + 1:fwd_end]

                if len(fwd_gi) == 0:
                    outcome_arr[gi] = 'inconclusive'
                    p_reversal[gi] = 0.15
                    n_inconclusive += 1
                    continue

                fwd_high = high_arr[fwd_gi]
                fwd_low = low_arr[fwd_gi]
                entry = c

                if s >= 1:
                    # Above level → reversal=UP, breakout=DOWN through level
                    rev_hits = np.where(
                        fwd_high >= entry + self.reversal_threshold_pts
                    )[0]
                    brk_hits = np.where(
                        fwd_low <= best_price - self.breakout_threshold_pts
                    )[0]
                else:
                    # Below level → reversal=DOWN, breakout=UP through level
                    rev_hits = np.where(
                        fwd_low <= entry - self.reversal_threshold_pts
                    )[0]
                    brk_hits = np.where(
                        fwd_high >= best_price + self.breakout_threshold_pts
                    )[0]

                first_rev = (rev_hits[0] + 1) if len(rev_hits) > 0 else 9999
                first_brk = (brk_hits[0] + 1) if len(brk_hits) > 0 else 9999

                if first_rev < first_brk and first_rev < 9999:
                    outcome_arr[gi] = 'reversal'
                    bars_to_event_arr[gi] = first_rev
                    p_reversal[gi] = np.exp(
                        -self.decay_alpha * first_rev
                    )
                    n_reversal += 1
                elif first_brk < first_rev and first_brk < 9999:
                    outcome_arr[gi] = 'breakout'
                    bars_to_event_arr[gi] = first_brk
                    p_reversal[gi] = 0.0
                    n_breakout += 1
                else:
                    outcome_arr[gi] = 'inconclusive'
                    p_reversal[gi] = 0.15
                    n_inconclusive += 1

        result['near_level'] = near_level
        result['p_reversal'] = p_reversal
        result['outcome'] = outcome_arr
        result['nearest_level'] = nearest_level_arr
        result['level_distance'] = level_distance_arr
        result['side'] = side_arr
        result['bars_to_event'] = bars_to_event_arr
        result['trade_direction'] = trade_dir_arr

        self._labeled_data = result
        self._stats = {
            'n_near_level': int(near_level.sum()),
            'n_reversal': n_reversal,
            'n_breakout': n_breakout,
            'n_inconclusive': n_inconclusive,
        }
        return result

    def print_summary(self) -> None:
        s = self._stats
        total = s.get('n_near_level', 0)
        n_rev = s.get('n_reversal', 0)
        n_brk = s.get('n_breakout', 0)
        n_inc = s.get('n_inconclusive', 0)

        print(f"\n{'='*60}")
        print("REVERSAL / BREAKOUT LABELING SUMMARY")
        print(f"{'='*60}")
        print(f"Total bars:        {len(self._labeled_data):,}")
        print(f"Near-level bars:   {total:,}")
        print(f"  Reversals:       {n_rev:,} ({n_rev/max(total,1):.1%})")
        print(f"  Breakouts:       {n_brk:,} ({n_brk/max(total,1):.1%})")
        print(f"  Inconclusive:    {n_inc:,} ({n_inc/max(total,1):.1%})")
        print(f"  Rev / Brk ratio: {n_rev/max(n_brk,1):.2f}")

        if self._labeled_data is not None:
            nl = self._labeled_data[self._labeled_data['near_level']]
            rev = nl[nl['outcome'] == 'reversal']
            brk = nl[nl['outcome'] == 'breakout']

            if len(rev) > 0:
                print(f"\nReversals:")
                print(f"  Avg P(reversal):     {rev['p_reversal'].mean():.3f}")
                print(f"  Avg bars to event:   {rev['bars_to_event'].mean():.1f}")
                lc = rev['nearest_level'].value_counts()
                print(f"  By level:")
                for lvl, cnt in lc.head(7).items():
                    print(f"    {lvl:15s}: {cnt:5d}")

            if len(brk) > 0:
                print(f"\nBreakouts:")
                print(f"  Avg bars to event:   {brk['bars_to_event'].mean():.1f}")
                lc = brk['nearest_level'].value_counts()
                print(f"  By level:")
                for lvl, cnt in lc.head(7).items():
                    print(f"    {lvl:15s}: {cnt:5d}")
        print(f"{'='*60}")


def analyze_zone_distribution(
    labeled_data: pd.DataFrame,
    max_lookback: int = 10
) -> pd.DataFrame:
    """
    Analyze the distribution of zone entries by distance to reversal.

    Args:
        labeled_data: Output from TradeViableZoneLabeler.fit()
        max_lookback: Maximum lookback to analyze

    Returns:
        DataFrame with distribution stats per lookback distance
    """
    zone_data = labeled_data[labeled_data['zone_label'] != ZoneLabel.NO_ZONE]

    results = []
    for lookback in range(max_lookback + 1):
        subset = zone_data[zone_data['bars_to_reversal'] == lookback]
        if len(subset) > 0:
            results.append({
                'bars_to_reversal': lookback,
                'count': len(subset),
                'avg_mae': subset['max_adverse_excursion'].mean(),
                'avg_reward': subset['potential_reward'].mean(),
                'avg_rr': subset['reward_risk_ratio'].mean(),
                'pct_bull': (subset['zone_label'] == ZoneLabel.BULL_ZONE).mean(),
                'pct_bear': (subset['zone_label'] == ZoneLabel.BEAR_ZONE).mean()
            })

    return pd.DataFrame(results)
