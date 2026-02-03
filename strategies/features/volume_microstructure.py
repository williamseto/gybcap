"""
Volume microstructure feature provider.

Computes volume-based features that describe "how" price interacted with levels,
not just "where" price is relative to levels. These features are hypothesized
to be more predictive of reversion success than price-based z-scores.

Features include:
- Mini volume profiles around key levels
- Bid-ask delta and imbalance at levels
- Volume dynamics leading to rejection
- Volume profile shape metrics
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


def _build_cumulative_vbp(
    prices: np.ndarray,
    volumes: np.ndarray,
    bin_size: float = 0.5,
    kernel: tuple = (0.2, 0.6, 0.2)
) -> tuple:
    """
    Build cumulative volume-by-price matrix.

    Args:
        prices: Array of close prices
        volumes: Array of volumes
        bin_size: Price bin width
        kernel: Smoothing kernel for volume distribution

    Returns:
        bin_centers: Array of price bin centers
        cum_vbp: Cumulative VBP matrix (T x n_bins)
    """
    min_p = prices.min() - 5
    max_p = prices.max() + 5
    bins = np.arange(np.floor(min_p), np.ceil(max_p) + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    n_bins = len(bin_centers)
    T = len(prices)

    per_bar_vbp = np.zeros((T, n_bins), dtype=np.float64)

    # Find nearest bin for each price
    idxs = np.searchsorted(bins, prices, side='right') - 1
    idxs = np.clip(idxs, 0, n_bins - 1)

    # Distribute volume using kernel
    offsets = [-1, 0, 1]
    for offset, weight in zip(offsets, kernel):
        idxs_off = np.clip(idxs + offset, 0, n_bins - 1)
        per_bar_vbp[np.arange(T), idxs_off] += volumes * weight

    # Cumulative sum
    cum_vbp = np.cumsum(per_bar_vbp, axis=0)

    return bin_centers, cum_vbp


def _build_bidask_vbp(
    prices: np.ndarray,
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    bin_size: float = 0.5
) -> tuple:
    """
    Build separate bid and ask volume-by-price matrices.

    Returns:
        bin_centers, cum_bid_vbp, cum_ask_vbp
    """
    min_p = prices.min() - 5
    max_p = prices.max() + 5
    bins = np.arange(np.floor(min_p), np.ceil(max_p) + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    n_bins = len(bin_centers)
    T = len(prices)

    bid_vbp = np.zeros((T, n_bins), dtype=np.float64)
    ask_vbp = np.zeros((T, n_bins), dtype=np.float64)

    idxs = np.searchsorted(bins, prices, side='right') - 1
    idxs = np.clip(idxs, 0, n_bins - 1)

    for t in range(T):
        bid_vbp[t, idxs[t]] = bid_volumes[t]
        ask_vbp[t, idxs[t]] = ask_volumes[t]

    cum_bid_vbp = np.cumsum(bid_vbp, axis=0)
    cum_ask_vbp = np.cumsum(ask_vbp, axis=0)

    return bin_centers, cum_bid_vbp, cum_ask_vbp


def _find_level_bin_range(bin_centers: np.ndarray, level: float, n_ticks: int = 2) -> tuple:
    """
    Find bin indices within n_ticks (ES points) of a level.

    Args:
        bin_centers: Array of VBP bin centers
        level: Price level
        n_ticks: Number of ES points (1 tick = 0.25, but we use points)

    Returns:
        low_idx, high_idx: Slice indices for bins near level
    """
    tick_size = 0.25 * n_ticks * 4  # n_ticks ES points
    low_idx = np.searchsorted(bin_centers, level - tick_size)
    high_idx = np.searchsorted(bin_centers, level + tick_size)
    return int(low_idx), int(min(high_idx + 1, len(bin_centers)))


def _compute_va70(vbp: np.ndarray, bin_centers: np.ndarray) -> tuple:
    """Compute 70% Value Area from VBP."""
    total = vbp.sum()
    if total <= 0:
        return float(bin_centers[0]), float(bin_centers[-1])

    poc_idx = int(np.argmax(vbp))
    cum = vbp[poc_idx]
    low = poc_idx
    high = poc_idx
    target = 0.7 * total

    while cum < target:
        left = vbp[low - 1] if low > 0 else -1
        right = vbp[high + 1] if high < len(vbp) - 1 else -1
        if left >= right:
            low -= 1
            cum += vbp[low]
        else:
            high += 1
            cum += vbp[high]
        if low == 0 and high == len(vbp) - 1:
            break

    return float(bin_centers[low]), float(bin_centers[high])


def _count_peaks(vbp: np.ndarray, min_prominence: float = 0.05) -> int:
    """Count local maxima in VBP."""
    if len(vbp) < 3:
        return 0

    total = vbp.sum()
    if total <= 0:
        return 0

    # Normalize
    vbp_norm = vbp / total

    # Find local maxima
    left = vbp_norm[:-2]
    center = vbp_norm[1:-1]
    right = vbp_norm[2:]

    is_peak = (center > left) & (center > right) & (center > min_prominence)
    return int(is_peak.sum())


@FeatureRegistry.register('volume_microstructure')
class VolumeMicrostructureProvider(BaseFeatureProvider):
    """
    Computes volume-based features around price levels.

    Features are computed at the bar level and can be attached to trades
    based on the trigger bar.
    """

    # Mini volume profile features
    MINI_PROFILE_FEATURES = [
        'vol_at_level',           # Total volume within +-2 ticks of level
        'vol_at_level_pct',       # As % of day's total volume
        'vol_concentration_z',    # Z-score of concentration vs other zones
        'vol_profile_entropy',    # Entropy around level (low = concentrated)
    ]

    # Bid-ask microstructure features
    BIDASK_FEATURES = [
        'bid_vol_at_level',       # Bid volume within +-2 ticks
        'ask_vol_at_level',       # Ask volume within +-2 ticks
        'delta_at_level',         # (bid - ask) cumulative near level
        'delta_at_level_z',       # Normalized delta
        'bid_ask_ratio_at_level', # bid_vol / ask_vol
    ]

    # Volume dynamics features
    DYNAMICS_FEATURES = [
        'vol_into_level',         # Volume in bars approaching level
        'vol_at_rejection_bar',   # Volume on the rejection bar
        'vol_ratio_rejection',    # rejection_bar_vol / avg_vol
        'delta_into_level',       # Cumulative delta approaching
        'delta_flip_at_level',    # Did delta flip sign? (absorption)
    ]

    # Profile shape features
    SHAPE_FEATURES = [
        'poc_distance',           # Distance from POC to level
        'va_contains_level',      # Is level inside 70% Value Area?
        'profile_skew_at_level',  # Skewness of volume around level
        'n_peaks_near_level',     # Multiple peaks = potential bounces
    ]

    def __init__(
        self,
        bin_size: float = 0.5,
        level_radius_ticks: int = 2,
        lookback_bars: int = 5,
        include_bidask: bool = True
    ):
        """
        Initialize provider.

        Args:
            bin_size: VBP bin size in price points
            level_radius_ticks: Number of ES points for "near level" zone
            lookback_bars: Bars to look back for "into level" dynamics
            include_bidask: Whether to compute bid/ask features (requires bidvolume/askvolume columns)
        """
        super().__init__()
        self.bin_size = bin_size
        self.level_radius_ticks = level_radius_ticks
        self.lookback_bars = lookback_bars
        self.include_bidask = include_bidask

        # Cache for day-level VBP matrices
        self._day_vbp_cache: Dict[str, Dict] = {}

    @property
    def name(self) -> str:
        return "volume_microstructure"

    @property
    def feature_names(self) -> List[str]:
        features = (
            self.MINI_PROFILE_FEATURES +
            self.DYNAMICS_FEATURES +
            self.SHAPE_FEATURES
        )
        if self.include_bidask:
            features += self.BIDASK_FEATURES
        return features

    def _compute_day_vbp(self, day_df: pd.DataFrame) -> Dict:
        """Compute VBP matrices for a single day."""
        prices = day_df['close'].values.astype(np.float64)
        volumes = day_df['volume'].values.astype(np.float64)

        bin_centers, cum_vbp = _build_cumulative_vbp(
            prices, volumes, self.bin_size
        )

        result = {
            'bin_centers': bin_centers,
            'cum_vbp': cum_vbp,
            'prices': prices,
            'volumes': volumes,
        }

        # Build bid/ask VBP if available
        if self.include_bidask and 'bidvolume' in day_df.columns and 'askvolume' in day_df.columns:
            bid_vols = day_df['bidvolume'].fillna(0).values.astype(np.float64)
            ask_vols = day_df['askvolume'].fillna(0).values.astype(np.float64)

            _, cum_bid_vbp, cum_ask_vbp = _build_bidask_vbp(
                prices, bid_vols, ask_vols, self.bin_size
            )

            result['cum_bid_vbp'] = cum_bid_vbp
            result['cum_ask_vbp'] = cum_ask_vbp
            result['bid_volumes'] = bid_vols
            result['ask_volumes'] = ask_vols

        return result

    def compute_features_at_bar(
        self,
        day_data: Dict,
        bar_idx: int,
        level: float
    ) -> Dict[str, float]:
        """
        Compute all volume microstructure features for a specific bar and level.

        Args:
            day_data: Dictionary with VBP matrices from _compute_day_vbp
            bar_idx: Index within the day (0-based)
            level: Price level that was tested/rejected

        Returns:
            Dictionary of feature values
        """
        features = {}

        bin_centers = day_data['bin_centers']
        cum_vbp = day_data['cum_vbp']
        prices = day_data['prices']
        volumes = day_data['volumes']

        # Get VBP at current bar
        vbp_at_bar = cum_vbp[bar_idx]
        total_vol = vbp_at_bar.sum()

        # Find bins near level
        lo_idx, hi_idx = _find_level_bin_range(
            bin_centers, level, self.level_radius_ticks
        )

        # --- Mini Profile Features ---
        vol_at_level = vbp_at_bar[lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0
        features['vol_at_level'] = vol_at_level
        features['vol_at_level_pct'] = vol_at_level / max(total_vol, 1e-6)

        # Concentration z-score: how much volume at level vs average bin
        n_bins = len(vbp_at_bar)
        avg_vol_per_bin = total_vol / max(n_bins, 1)
        std_vol_per_bin = vbp_at_bar.std() if n_bins > 1 else 1.0
        vol_per_bin_at_level = vol_at_level / max(hi_idx - lo_idx, 1)
        features['vol_concentration_z'] = (
            (vol_per_bin_at_level - avg_vol_per_bin) / max(std_vol_per_bin, 1e-6)
        )

        # Entropy of volume distribution near level
        if hi_idx > lo_idx and vol_at_level > 0:
            local_vbp = vbp_at_bar[lo_idx:hi_idx]
            local_vbp_norm = local_vbp / local_vbp.sum()
            # Avoid log(0)
            local_vbp_norm = np.clip(local_vbp_norm, 1e-10, 1.0)
            features['vol_profile_entropy'] = float(scipy_entropy(local_vbp_norm))
        else:
            features['vol_profile_entropy'] = 0.0

        # --- Volume Dynamics Features ---
        start_idx = max(0, bar_idx - self.lookback_bars)
        vol_into_level = volumes[start_idx:bar_idx].sum() if bar_idx > start_idx else 0.0
        features['vol_into_level'] = vol_into_level

        vol_at_rejection_bar = volumes[bar_idx]
        features['vol_at_rejection_bar'] = vol_at_rejection_bar

        avg_vol = volumes[:bar_idx + 1].mean() if bar_idx >= 0 else 1.0
        features['vol_ratio_rejection'] = vol_at_rejection_bar / max(avg_vol, 1e-6)

        # --- Profile Shape Features ---
        poc_idx = int(np.argmax(vbp_at_bar))
        poc_price = bin_centers[poc_idx] if len(bin_centers) > 0 else level
        features['poc_distance'] = abs(poc_price - level)

        va_lo, va_hi = _compute_va70(vbp_at_bar, bin_centers)
        features['va_contains_level'] = float(va_lo <= level <= va_hi)

        # Skewness of volume around level
        if hi_idx > lo_idx + 2:
            local_vbp = vbp_at_bar[lo_idx:hi_idx]
            local_bins = bin_centers[lo_idx:hi_idx]
            if local_vbp.sum() > 0:
                mean_price = np.average(local_bins, weights=local_vbp)
                std_price = np.sqrt(np.average((local_bins - mean_price)**2, weights=local_vbp))
                if std_price > 0:
                    skew = np.average(((local_bins - mean_price) / std_price)**3, weights=local_vbp)
                    features['profile_skew_at_level'] = float(skew)
                else:
                    features['profile_skew_at_level'] = 0.0
            else:
                features['profile_skew_at_level'] = 0.0
        else:
            features['profile_skew_at_level'] = 0.0

        features['n_peaks_near_level'] = _count_peaks(vbp_at_bar[lo_idx:hi_idx])

        # --- Bid/Ask Features ---
        if self.include_bidask and 'cum_bid_vbp' in day_data:
            cum_bid = day_data['cum_bid_vbp'][bar_idx]
            cum_ask = day_data['cum_ask_vbp'][bar_idx]
            bid_vols = day_data['bid_volumes']
            ask_vols = day_data['ask_volumes']

            bid_at_level = cum_bid[lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0
            ask_at_level = cum_ask[lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0

            features['bid_vol_at_level'] = bid_at_level
            features['ask_vol_at_level'] = ask_at_level
            features['delta_at_level'] = bid_at_level - ask_at_level

            total_at_level = bid_at_level + ask_at_level
            if total_at_level > 0:
                features['delta_at_level_z'] = features['delta_at_level'] / total_at_level
            else:
                features['delta_at_level_z'] = 0.0

            features['bid_ask_ratio_at_level'] = (
                bid_at_level / max(ask_at_level, 1e-6)
            )

            # Delta into level and flip detection
            delta_into = (
                bid_vols[start_idx:bar_idx].sum() - ask_vols[start_idx:bar_idx].sum()
                if bar_idx > start_idx else 0.0
            )
            features['delta_into_level'] = delta_into

            # Delta flip: did sign change between approach and rejection bar?
            delta_at_bar = bid_vols[bar_idx] - ask_vols[bar_idx]
            delta_flipped = (delta_into * delta_at_bar) < 0  # Opposite signs
            features['delta_flip_at_level'] = float(delta_flipped)
        else:
            # Fill with defaults if no bid/ask data
            for feat in self.BIDASK_FEATURES:
                features[feat] = 0.0
            features['delta_into_level'] = 0.0
            features['delta_flip_at_level'] = 0.0

        return features

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute volume microstructure features for all bars.

        Note: Features are computed per-bar but are most meaningful when
        used at specific trigger/rejection bars with a known level.
        For general computation, we compute features relative to VWAP.

        For trade-specific features, use compute_features_at_bar() directly.
        """
        result = ohlcv.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        # Process by day
        for day, day_df in ohlcv.groupby('trading_day'):
            day_idx = day_df.index
            day_data = self._compute_day_vbp(day_df)

            # Cache for later use
            self._day_vbp_cache[day] = day_data

            # Compute features using VWAP as reference level
            if 'vwap' in day_df.columns:
                vwap_vals = day_df['vwap'].values
            else:
                # Compute simple VWAP
                avg_price = (day_df['open'] + day_df['high'] + day_df['low'] + day_df['close']) / 4
                cum_pv = (avg_price * day_df['volume']).cumsum()
                cum_vol = day_df['volume'].cumsum()
                vwap_vals = (cum_pv / cum_vol.clip(lower=1)).values

            for i, (idx, row) in enumerate(day_df.iterrows()):
                level = vwap_vals[i]
                features = self.compute_features_at_bar(day_data, i, level)

                for feat_name, feat_val in features.items():
                    result.loc[idx, feat_name] = feat_val

        return result

    def compute_trade_features(
        self,
        ohlcv: pd.DataFrame,
        trades_df: pd.DataFrame,
        level_col: str = 'level_price'
    ) -> pd.DataFrame:
        """
        Compute volume features specifically for trade entries.

        Args:
            ohlcv: Full OHLCV DataFrame with trading_day column
            trades_df: DataFrame with trade info including bar timestamps
            level_col: Column in trades_df containing the level that was tested

        Returns:
            trades_df with volume features added
        """
        result = trades_df.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        # Build day cache if not already built
        if not self._day_vbp_cache:
            for day, day_df in ohlcv.groupby('trading_day'):
                self._day_vbp_cache[day] = self._compute_day_vbp(day_df)

        # Get day mapping from ohlcv
        if 'dt' in ohlcv.columns:
            dt_to_day = ohlcv.set_index('dt')['trading_day'].to_dict()
        else:
            dt_to_day = ohlcv['trading_day'].to_dict()

        for idx, trade_row in trades_df.iterrows():
            # Get trading day for this trade
            if 'entry_ts' in trade_row:
                ts = trade_row['entry_ts']
            elif 'dt' in trade_row:
                ts = trade_row['dt']
            else:
                continue

            day = dt_to_day.get(ts)
            if day is None or day not in self._day_vbp_cache:
                continue

            day_data = self._day_vbp_cache[day]

            # Find bar index within day
            day_ohlcv = ohlcv[ohlcv['trading_day'] == day]
            if 'dt' in day_ohlcv.columns:
                day_ohlcv = day_ohlcv.set_index('dt')

            try:
                bar_idx = day_ohlcv.index.get_loc(ts)
            except KeyError:
                continue

            # Get level
            level = trade_row.get(level_col, 0.0)
            if pd.isna(level) or level == 0:
                continue

            # Compute features
            features = self.compute_features_at_bar(day_data, bar_idx, level)

            for feat_name, feat_val in features.items():
                result.loc[idx, feat_name] = feat_val

        return result
