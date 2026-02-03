"""
Feature normalization pipeline for neural network training.

Handles proper normalization for different feature types:
- Price sequences: Per-window z-score or returns
- Volume sequences: Per-day z-score
- Higher TF indicators: Global z-score (rolling)
- Bounded indicators (RSI, %B): Already bounded, scale to [0, 1]
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np


class NormalizationPipeline:
    """
    Consistent normalization for NN training.

    Handles different feature types appropriately:
    - Sequences: Per-sequence normalization
    - Scalars: Rolling z-score or clipping
    - Bounded: Scale to [0, 1]
    """

    # Features that are already bounded
    BOUNDED_FEATURES = [
        'daily_rsi_14', 'weekly_rsi_14',
        'daily_bb_pct_b', 'weekly_bb_pct_b', 'intraday_bb_pct_b',
        'va_contains_level', 'gap_filled', 'delta_flip_at_level',
    ]

    # Features that should use z-score normalization
    ZSCORE_FEATURES = [
        'daily_close_vs_sma20', 'daily_close_vs_sma50',
        'weekly_close_vs_sma10',
        'daily_bb_upper_dist', 'daily_bb_lower_dist', 'daily_bb_width',
        'weekly_bb_width',
        'intraday_bb_upper_dist', 'intraday_bb_lower_dist', 'intraday_bb_width_z',
        'gap_pct', 'gap_vs_prior_range',
        'daily_trend_5d', 'daily_trend_20d',
        'vol_concentration_z', 'delta_at_level_z',
    ]

    # Features that should be log-transformed before z-score
    LOG_FEATURES = [
        'daily_atr_14', 'prior_day_range', 'volume',
        'vol_at_level', 'vol_into_level', 'vol_at_rejection_bar',
    ]

    def __init__(
        self,
        clip_outliers: bool = True,
        clip_std: float = 3.0,
        rolling_window: int = 60,  # Days for rolling stats
    ):
        """
        Initialize normalization pipeline.

        Args:
            clip_outliers: Whether to clip extreme values
            clip_std: Number of std deviations for clipping
            rolling_window: Window for rolling statistics (in days)
        """
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        self.rolling_window = rolling_window

        # Fitted statistics
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._fitted = False

    def fit(self, features_df: pd.DataFrame, day_col: str = 'trading_day') -> 'NormalizationPipeline':
        """
        Fit normalization statistics on training data.

        Uses expanding window to avoid look-ahead bias.

        Args:
            features_df: DataFrame with features
            day_col: Column containing trading day
        """
        # Compute global statistics for each feature
        for col in features_df.columns:
            if col == day_col or col.startswith('_'):
                continue

            values = features_df[col].values
            valid = ~np.isnan(values)

            if valid.sum() < 10:
                continue

            self._feature_stats[col] = {
                'mean': np.mean(values[valid]),
                'std': np.std(values[valid]),
                'min': np.min(values[valid]),
                'max': np.max(values[valid]),
                'median': np.median(values[valid]),
            }

        self._fitted = True
        return self

    def transform(
        self,
        features_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform features using fitted statistics.

        Args:
            features_df: DataFrame with features
            feature_cols: Specific columns to transform (None = all)

        Returns:
            Transformed DataFrame
        """
        result = features_df.copy()

        if feature_cols is None:
            feature_cols = [c for c in features_df.columns
                          if c in self._feature_stats]

        for col in feature_cols:
            if col not in self._feature_stats:
                continue

            values = result[col].values.copy()
            stats = self._feature_stats[col]

            # Handle bounded features
            if col in self.BOUNDED_FEATURES:
                result[col] = self._normalize_bounded(values, col)

            # Handle log features
            elif col in self.LOG_FEATURES:
                result[col] = self._normalize_log(values, stats)

            # Handle z-score features
            else:
                result[col] = self._normalize_zscore(values, stats)

        return result

    def fit_transform(
        self,
        features_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        day_col: str = 'trading_day'
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(features_df, day_col)
        return self.transform(features_df, feature_cols)

    def _normalize_bounded(self, values: np.ndarray, col: str) -> np.ndarray:
        """Normalize bounded features to [0, 1]."""
        result = values.copy()

        if 'rsi' in col.lower():
            # RSI is 0-100, scale to 0-1
            result = result / 100.0
        elif 'pct_b' in col.lower():
            # %B is typically 0-1 but can exceed, clip to reasonable range
            result = np.clip(result, -0.5, 1.5)
            result = (result + 0.5) / 2.0  # Scale to [0, 1]
        else:
            # Binary features, leave as is
            pass

        return result

    def _normalize_log(self, values: np.ndarray, stats: Dict) -> np.ndarray:
        """Log-transform then z-score normalize."""
        result = values.copy()

        # Handle zeros and negatives
        result = np.where(result > 0, result, 1e-6)
        result = np.log(result)

        # Z-score of log values
        mean = np.log(max(stats['mean'], 1e-6))
        std = max(stats['std'] / max(stats['mean'], 1e-6), 1e-6)  # Approx std of log

        result = (result - mean) / std

        # Clip outliers
        if self.clip_outliers:
            result = np.clip(result, -self.clip_std, self.clip_std)

        return result

    def _normalize_zscore(self, values: np.ndarray, stats: Dict) -> np.ndarray:
        """Standard z-score normalization."""
        result = values.copy()

        mean = stats['mean']
        std = max(stats['std'], 1e-6)

        result = (result - mean) / std

        # Clip outliers
        if self.clip_outliers:
            result = np.clip(result, -self.clip_std, self.clip_std)

        # Fill NaN with 0 (neutral after z-score)
        result = np.where(np.isnan(result), 0, result)

        return result

    # ==================== Sequence Normalization ====================

    def normalize_price_window(
        self,
        prices: np.ndarray,
        method: str = 'returns'
    ) -> np.ndarray:
        """
        Normalize price sequence to preserve shape, remove absolute level.

        Args:
            prices: Price array (1D)
            method: 'returns' or 'zscore'

        Returns:
            Normalized prices
        """
        if len(prices) == 0:
            return prices

        if method == 'returns':
            # (price - price[0]) / price[0]
            base = prices[0]
            if base == 0:
                return np.zeros_like(prices)
            return (prices - base) / base

        elif method == 'zscore':
            mean = np.mean(prices)
            std = np.std(prices)
            if std < 1e-6:
                return np.zeros_like(prices)
            return (prices - mean) / std

        else:
            raise ValueError(f"Unknown method: {method}")

    def normalize_volume(
        self,
        volume: np.ndarray,
        day_volume: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize volume relative to day's activity or local window.

        Args:
            volume: Volume array
            day_volume: Optional total day volume for context

        Returns:
            Normalized volume
        """
        if len(volume) == 0:
            return volume

        # Use window statistics if no day_volume
        if day_volume is None:
            mean = np.mean(volume)
            std = np.std(volume)
        else:
            mean = day_volume / len(volume)  # Average per-bar
            std = np.std(volume)

        if std < 1e-6:
            return np.zeros_like(volume)

        return (volume - mean) / std

    def normalize_for_xgboost(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for XGBoost.

        XGBoost is scale-invariant, but extreme values can still hurt.
        Clip outliers and fill NaN.
        """
        result = features_df.copy()

        for col in result.columns:
            if col.startswith('_') or result[col].dtype not in [np.float64, np.float32]:
                continue

            values = result[col].values

            # Clip extreme outliers
            if self.clip_outliers and col not in self.BOUNDED_FEATURES:
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 1e-6:
                    low = mean - self.clip_std * std
                    high = mean + self.clip_std * std
                    values = np.clip(values, low, high)

            # Fill NaN with 0
            values = np.where(np.isnan(values), 0, values)
            result[col] = values

        return result

    def normalize_for_tcn(
        self,
        sequences: np.ndarray,
        per_sequence: bool = True
    ) -> np.ndarray:
        """
        Normalize sequences for TCN input.

        Args:
            sequences: Array of shape (batch, seq_len, channels)
            per_sequence: If True, normalize each sequence independently

        Returns:
            Normalized sequences
        """
        result = sequences.copy()
        batch, seq_len, channels = result.shape

        if per_sequence:
            # Normalize each sequence independently
            for b in range(batch):
                for c in range(channels):
                    seq = result[b, :, c]
                    mean = np.mean(seq)
                    std = np.std(seq)
                    if std > 1e-6:
                        result[b, :, c] = (seq - mean) / std
                    else:
                        result[b, :, c] = 0
        else:
            # Global normalization per channel
            for c in range(channels):
                all_values = result[:, :, c].flatten()
                mean = np.mean(all_values)
                std = np.std(all_values)
                if std > 1e-6:
                    result[:, :, c] = (result[:, :, c] - mean) / std
                else:
                    result[:, :, c] = 0

        return result

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get fitted statistics."""
        return self._feature_stats.copy()

    def save_stats(self, path: str) -> None:
        """Save fitted statistics to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self._feature_stats, f, indent=2)

    def load_stats(self, path: str) -> None:
        """Load statistics from file."""
        import json
        with open(path, 'r') as f:
            self._feature_stats = json.load(f)
        self._fitted = True


def create_normalization_pipeline(
    feature_groups: Optional[List[str]] = None
) -> NormalizationPipeline:
    """
    Factory function to create a normalization pipeline.

    Args:
        feature_groups: List of feature group names to include
                       ('htf', 'volume', 'quality', 'all')

    Returns:
        Configured NormalizationPipeline
    """
    return NormalizationPipeline(clip_outliers=True, clip_std=3.0)
