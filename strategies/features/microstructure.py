"""
Microstructure sequence feature provider for TCN/NN models.

Provides raw sequence data in formats suitable for neural network input:
- OHLCV sequences (normalized)
- Cumulative delta curves
- Volume at price bins
- Order flow imbalance

Unlike hand-crafted features, these sequences let the NN learn patterns directly.
"""

from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


def _normalize_price_sequence(prices: np.ndarray) -> np.ndarray:
    """
    Normalize price sequence to preserve shape, remove absolute level.

    Uses returns-based normalization: (price[t] - price[0]) / price[0]
    This preserves the shape while making sequences comparable.
    """
    if len(prices) == 0:
        return prices
    base = prices[0]
    if base == 0:
        return np.zeros_like(prices)
    return (prices - base) / base


def _normalize_volume_sequence(volumes: np.ndarray) -> np.ndarray:
    """
    Normalize volume sequence using z-score.

    Handles varying market activity levels.
    """
    if len(volumes) == 0:
        return volumes
    mean_vol = np.mean(volumes)
    std_vol = np.std(volumes)
    if std_vol < 1e-6:
        return np.zeros_like(volumes)
    return (volumes - mean_vol) / std_vol


def _compute_cumulative_delta(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray
) -> np.ndarray:
    """Compute cumulative delta (bid - ask) over time."""
    delta = bid_volumes - ask_volumes
    return np.cumsum(delta)


def _build_volume_at_price_bins(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = 10,
    lookback: int = 20
) -> np.ndarray:
    """
    Build mini volume profile bins around current price.

    Returns:
        Array of shape (len(prices), n_bins) with volume distribution
    """
    T = len(prices)
    result = np.zeros((T, n_bins), dtype=np.float64)

    for t in range(lookback, T):
        # Get lookback window
        start = t - lookback
        window_prices = prices[start:t]
        window_volumes = volumes[start:t]

        if len(window_prices) == 0:
            continue

        # Create bins around current price
        current = prices[t]
        price_range = window_prices.max() - window_prices.min()
        if price_range < 0.01:
            price_range = 1.0

        # Bins centered on current price, spanning recent range
        bin_edges = np.linspace(
            current - price_range / 2,
            current + price_range / 2,
            n_bins + 1
        )

        # Distribute volume into bins
        for i in range(len(window_prices)):
            p = window_prices[i]
            v = window_volumes[i]
            bin_idx = np.searchsorted(bin_edges[:-1], p) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            result[t, bin_idx] += v

        # Normalize to sum to 1 (or 0 if no volume)
        total = result[t].sum()
        if total > 0:
            result[t] /= total

    return result


@FeatureRegistry.register('microstructure')
class MicrostructureSequenceProvider(BaseFeatureProvider):
    """
    Provides raw microstructure sequences for neural network input.

    Instead of hand-crafted features, this provider generates normalized
    sequence data that TCN/LSTM models can learn patterns from directly.

    Output format for each bar:
    - OHLCV sequence (lookback_bars x 5 channels)
    - Cumulative delta curve (lookback_bars x 1 channel)
    - Volume at price bins (lookback_bars x n_bins channels)
    """

    def __init__(
        self,
        lookback_bars: int = 60,
        n_vap_bins: int = 10,
        include_bidask: bool = True
    ):
        """
        Initialize provider.

        Args:
            lookback_bars: Number of bars to look back for sequences
            n_vap_bins: Number of volume-at-price bins
            include_bidask: Whether to include bid/ask data (if available)
        """
        super().__init__()
        self.lookback_bars = lookback_bars
        self.n_vap_bins = n_vap_bins
        self.include_bidask = include_bidask

        # Computed sequences stored here
        self._ohlcv_sequences: Optional[np.ndarray] = None
        self._delta_sequences: Optional[np.ndarray] = None
        self._vap_sequences: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "microstructure"

    @property
    def feature_names(self) -> List[str]:
        """
        Feature names are placeholders - actual data is in sequences.

        These scalar features summarize the sequence state.
        """
        return [
            'seq_price_trend',        # Slope of price over lookback
            'seq_volume_trend',       # Slope of volume over lookback
            'seq_delta_final',        # Final cumulative delta value
            'seq_delta_trend',        # Slope of cumulative delta
            'seq_vap_concentration',  # How concentrated volume is in bins
        ]

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute microstructure sequences and summary features.

        The main output is stored in self._ohlcv_sequences etc.
        The DataFrame gets scalar summary features.
        """
        result = ohlcv.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        T = len(ohlcv)
        lookback = self.lookback_bars

        # Extract arrays
        opens = ohlcv['open'].values.astype(np.float64)
        highs = ohlcv['high'].values.astype(np.float64)
        lows = ohlcv['low'].values.astype(np.float64)
        closes = ohlcv['close'].values.astype(np.float64)
        volumes = ohlcv['volume'].values.astype(np.float64)

        # Check for bid/ask data
        has_bidask = (
            self.include_bidask and
            'bidvolume' in ohlcv.columns and
            'askvolume' in ohlcv.columns
        )

        if has_bidask:
            bid_vols = ohlcv['bidvolume'].fillna(0).values.astype(np.float64)
            ask_vols = ohlcv['askvolume'].fillna(0).values.astype(np.float64)
        else:
            bid_vols = np.zeros(T)
            ask_vols = np.zeros(T)

        # Initialize sequence arrays
        # Shape: (T, lookback, channels)
        self._ohlcv_sequences = np.zeros((T, lookback, 5), dtype=np.float32)
        self._delta_sequences = np.zeros((T, lookback), dtype=np.float32)
        self._vap_sequences = np.zeros((T, lookback, self.n_vap_bins), dtype=np.float32)

        # Build VAP matrix for efficiency
        vap_matrix = _build_volume_at_price_bins(
            closes, volumes, self.n_vap_bins, lookback
        )

        # Compute per-day to respect day boundaries
        for day, day_df in ohlcv.groupby('trading_day'):
            day_indices = day_df.index.tolist()
            day_start = ohlcv.index.get_loc(day_indices[0])

            for i, idx in enumerate(day_indices):
                t = ohlcv.index.get_loc(idx)

                # Check if we have enough lookback within this day
                day_lookback = min(i, lookback)
                if day_lookback < 5:  # Need at least 5 bars
                    continue

                start_t = t - day_lookback
                end_t = t

                # Extract OHLCV window
                o_win = opens[start_t:end_t]
                h_win = highs[start_t:end_t]
                l_win = lows[start_t:end_t]
                c_win = closes[start_t:end_t]
                v_win = volumes[start_t:end_t]

                # Normalize
                o_norm = _normalize_price_sequence(o_win)
                h_norm = _normalize_price_sequence(h_win)
                l_norm = _normalize_price_sequence(l_win)
                c_norm = _normalize_price_sequence(c_win)
                v_norm = _normalize_volume_sequence(v_win)

                # Pad if needed
                pad_len = lookback - day_lookback
                if pad_len > 0:
                    o_norm = np.pad(o_norm, (pad_len, 0), mode='constant')
                    h_norm = np.pad(h_norm, (pad_len, 0), mode='constant')
                    l_norm = np.pad(l_norm, (pad_len, 0), mode='constant')
                    c_norm = np.pad(c_norm, (pad_len, 0), mode='constant')
                    v_norm = np.pad(v_norm, (pad_len, 0), mode='constant')

                self._ohlcv_sequences[t, :, 0] = o_norm
                self._ohlcv_sequences[t, :, 1] = h_norm
                self._ohlcv_sequences[t, :, 2] = l_norm
                self._ohlcv_sequences[t, :, 3] = c_norm
                self._ohlcv_sequences[t, :, 4] = v_norm

                # Cumulative delta
                if has_bidask:
                    delta_win = _compute_cumulative_delta(
                        bid_vols[start_t:end_t],
                        ask_vols[start_t:end_t]
                    )
                    # Normalize
                    delta_norm = _normalize_volume_sequence(delta_win)
                    if pad_len > 0:
                        delta_norm = np.pad(delta_norm, (pad_len, 0), mode='constant')
                    self._delta_sequences[t, :] = delta_norm

                # VAP bins
                vap_win = vap_matrix[start_t:end_t]
                if pad_len > 0:
                    vap_win = np.pad(vap_win, ((pad_len, 0), (0, 0)), mode='constant')
                self._vap_sequences[t, :, :] = vap_win

                # Compute scalar summary features
                # Price trend (slope of close)
                if len(c_win) > 1:
                    x = np.arange(len(c_win))
                    price_slope = np.polyfit(x, c_win, 1)[0]
                    result.loc[idx, 'seq_price_trend'] = price_slope / (c_win[0] + 1e-6)
                else:
                    result.loc[idx, 'seq_price_trend'] = 0.0

                # Volume trend
                if len(v_win) > 1:
                    x = np.arange(len(v_win))
                    vol_slope = np.polyfit(x, v_win, 1)[0]
                    result.loc[idx, 'seq_volume_trend'] = vol_slope / (np.mean(v_win) + 1e-6)
                else:
                    result.loc[idx, 'seq_volume_trend'] = 0.0

                # Delta summary
                if has_bidask:
                    delta_win = _compute_cumulative_delta(
                        bid_vols[start_t:end_t],
                        ask_vols[start_t:end_t]
                    )
                    result.loc[idx, 'seq_delta_final'] = delta_win[-1] if len(delta_win) > 0 else 0.0

                    if len(delta_win) > 1:
                        x = np.arange(len(delta_win))
                        delta_slope = np.polyfit(x, delta_win, 1)[0]
                        result.loc[idx, 'seq_delta_trend'] = delta_slope
                    else:
                        result.loc[idx, 'seq_delta_trend'] = 0.0

                # VAP concentration (entropy-based)
                vap_current = vap_matrix[t]
                if vap_current.sum() > 0:
                    vap_norm = vap_current / vap_current.sum()
                    vap_norm = np.clip(vap_norm, 1e-10, 1.0)
                    entropy = -np.sum(vap_norm * np.log(vap_norm))
                    max_entropy = np.log(self.n_vap_bins)
                    # Concentration = 1 - normalized_entropy (1 = very concentrated)
                    result.loc[idx, 'seq_vap_concentration'] = 1.0 - entropy / max_entropy
                else:
                    result.loc[idx, 'seq_vap_concentration'] = 0.0

        return result

    def get_ohlcv_sequence(self, bar_idx: int) -> np.ndarray:
        """
        Get OHLCV sequence for a specific bar.

        Returns:
            Array of shape (lookback_bars, 5) with normalized OHLCV
        """
        if self._ohlcv_sequences is None:
            raise ValueError("Must call compute() first")
        return self._ohlcv_sequences[bar_idx]

    def get_delta_sequence(self, bar_idx: int) -> np.ndarray:
        """
        Get cumulative delta sequence for a specific bar.

        Returns:
            Array of shape (lookback_bars,) with normalized cumulative delta
        """
        if self._delta_sequences is None:
            raise ValueError("Must call compute() first")
        return self._delta_sequences[bar_idx]

    def get_vap_sequence(self, bar_idx: int) -> np.ndarray:
        """
        Get volume-at-price sequence for a specific bar.

        Returns:
            Array of shape (lookback_bars, n_vap_bins) with VAP distribution
        """
        if self._vap_sequences is None:
            raise ValueError("Must call compute() first")
        return self._vap_sequences[bar_idx]

    def get_combined_sequence(self, bar_idx: int) -> np.ndarray:
        """
        Get all sequences combined for TCN input.

        Returns:
            Array of shape (lookback_bars, n_channels) where
            n_channels = 5 (OHLCV) + 1 (delta) + n_vap_bins
        """
        ohlcv = self.get_ohlcv_sequence(bar_idx)
        delta = self.get_delta_sequence(bar_idx)
        vap = self.get_vap_sequence(bar_idx)

        # Stack: (lookback, 5) + (lookback, 1) + (lookback, n_bins)
        return np.concatenate([
            ohlcv,
            delta[:, np.newaxis],
            vap
        ], axis=1)

    def get_all_combined_sequences(self) -> np.ndarray:
        """
        Get all combined sequences for batch processing.

        Returns:
            Array of shape (T, lookback_bars, n_channels)
        """
        if self._ohlcv_sequences is None:
            raise ValueError("Must call compute() first")

        return np.concatenate([
            self._ohlcv_sequences,
            self._delta_sequences[:, :, np.newaxis],
            self._vap_sequences
        ], axis=2)

    @property
    def n_input_channels(self) -> int:
        """Number of input channels for TCN."""
        return 5 + 1 + self.n_vap_bins  # OHLCV + delta + VAP bins
