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
from scipy.signal import find_peaks

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
    Build separate bid and ask volume-by-price matrices (vectorized).

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

    # Vectorized assignment using advanced indexing
    time_indices = np.arange(T)
    bid_vbp[time_indices, idxs] = bid_volumes
    ask_vbp[time_indices, idxs] = ask_volumes

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

    # Composite VP features (multi-day volume profile)
    COMPOSITE_VP_FEATURES = [
        'composite_vp_density_3d',  # Volume density at level on 3-day composite VP
        'composite_vp_density_5d',  # Same for 5-day composite VP
        'level_is_hvn_3d',          # Binary: level within top 30% volume bins on 3d VP
        'level_is_lvn_3d',          # Binary: level within bottom 20% volume bins on 3d VP
        'nearest_lvn_dist',         # Distance (pts) to nearest LVN on session VP
        'nearest_hvn_dist',         # Distance (pts) to nearest HVN on session VP
        'vp_valley_depth',          # Depth of valley between price and level
        'composite_poc_dist',       # Distance from 3-day POC to level (pts)
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
            self.SHAPE_FEATURES +
            self.COMPOSITE_VP_FEATURES
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

    def _build_composite_vbp(
        self,
        day_keys: List,
        level: float,
    ) -> Optional[Dict]:
        """
        Build composite VBP from multiple days' cached data.

        Returns dict with 'vbp' (summed profile), 'bin_centers', 'poc_price',
        or None if insufficient data.
        """
        available = [d for d in day_keys if d in self._day_vbp_cache]
        if not available:
            return None

        # Use first day's bin_centers as reference (similar price range)
        ref = self._day_vbp_cache[available[0]]
        ref_bins = ref['bin_centers']

        # Sum final-bar VBP from each day, aligned to common bins
        composite = np.zeros(len(ref_bins), dtype=np.float64)
        for d in available:
            data = self._day_vbp_cache[d]
            day_vbp = data['cum_vbp'][-1]  # Final cumulative = total day VBP
            day_bins = data['bin_centers']

            # Align: interpolate to reference bins
            if len(day_bins) == len(ref_bins) and np.allclose(day_bins, ref_bins):
                composite += day_vbp
            else:
                aligned = np.interp(ref_bins, day_bins, day_vbp, left=0, right=0)
                composite += aligned

        if composite.sum() <= 0:
            return None

        poc_idx = int(np.argmax(composite))
        return {
            'vbp': composite,
            'bin_centers': ref_bins,
            'poc_price': float(ref_bins[poc_idx]),
        }

    @staticmethod
    def _find_hvn_lvn(
        vbp: np.ndarray,
        bin_centers: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Find High Volume Nodes and Low Volume Nodes in a profile.

        HVN: peaks in smoothed profile above 70th percentile.
        LVN: valleys below 20th percentile.

        Returns dict with 'hvn_prices', 'lvn_prices', 'hvn_mask', 'lvn_mask'.
        """
        result = {'hvn_prices': [], 'lvn_prices': [], 'hvn_mask': np.zeros(len(vbp), dtype=bool),
                  'lvn_mask': np.zeros(len(vbp), dtype=bool)}

        if vbp.sum() <= 0 or len(vbp) < 5:
            return result

        # Smooth
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smoothed = np.convolve(vbp, kernel, mode='same')

        # HVN: peaks above 70th percentile
        p70 = np.percentile(smoothed[smoothed > 0], 70) if (smoothed > 0).any() else 0
        peaks, _ = find_peaks(smoothed, height=p70, distance=max(3, len(vbp) // 20))
        result['hvn_prices'] = bin_centers[peaks].tolist() if len(peaks) > 0 else []
        result['hvn_mask'][peaks] = True

        # LVN: valleys (inverted peaks) below 20th percentile
        p20 = np.percentile(smoothed[smoothed > 0], 20) if (smoothed > 0).any() else 0
        inverted = smoothed.max() - smoothed
        valleys, _ = find_peaks(inverted, distance=max(3, len(vbp) // 20))
        # Keep only valleys where original volume is below p20
        lvn_valleys = valleys[smoothed[valleys] <= p20]
        result['lvn_prices'] = bin_centers[lvn_valleys].tolist() if len(lvn_valleys) > 0 else []
        result['lvn_mask'][lvn_valleys] = True

        return result

    def _compute_composite_features_for_day(
        self,
        day: str,
        sorted_days: List,
        day_df: pd.DataFrame,
        levels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute composite VP features for one day."""
        T = len(day_df)
        features = {f: np.zeros(T, dtype=np.float64) for f in self.COMPOSITE_VP_FEATURES}

        day_idx_in_list = sorted_days.index(day) if day in sorted_days else -1
        if day_idx_in_list < 0:
            return features

        # Build 3-day and 5-day composite VPs from prior days
        prior_3d = sorted_days[max(0, day_idx_in_list - 3):day_idx_in_list]
        prior_5d = sorted_days[max(0, day_idx_in_list - 5):day_idx_in_list]

        comp_3d = self._build_composite_vbp(prior_3d, 0)
        comp_5d = self._build_composite_vbp(prior_5d, 0)

        # Session VP: use current day's final VBP
        if day in self._day_vbp_cache:
            session_data = self._day_vbp_cache[day]
            session_vbp = session_data['cum_vbp'][-1]
            session_bins = session_data['bin_centers']
            session_hvn_lvn = self._find_hvn_lvn(session_vbp, session_bins)
        else:
            session_hvn_lvn = None

        for t in range(T):
            level = levels[t]
            if np.isnan(level) or level == 0:
                continue

            # Composite 3-day features
            if comp_3d is not None:
                bins_3d = comp_3d['bin_centers']
                vbp_3d = comp_3d['vbp']
                total_3d = vbp_3d.sum()

                # Density at level
                bin_idx = np.searchsorted(bins_3d, level)
                bin_idx = min(max(bin_idx, 0), len(vbp_3d) - 1)
                features['composite_vp_density_3d'][t] = vbp_3d[bin_idx] / max(total_3d, 1e-6)

                # HVN/LVN classification
                p70 = np.percentile(vbp_3d[vbp_3d > 0], 70) if (vbp_3d > 0).any() else 0
                p20 = np.percentile(vbp_3d[vbp_3d > 0], 20) if (vbp_3d > 0).any() else 0
                features['level_is_hvn_3d'][t] = float(vbp_3d[bin_idx] >= p70)
                features['level_is_lvn_3d'][t] = float(vbp_3d[bin_idx] <= p20)

                # POC distance
                features['composite_poc_dist'][t] = abs(comp_3d['poc_price'] - level)

            # Composite 5-day density
            if comp_5d is not None:
                bins_5d = comp_5d['bin_centers']
                vbp_5d = comp_5d['vbp']
                total_5d = vbp_5d.sum()
                bin_idx_5d = np.searchsorted(bins_5d, level)
                bin_idx_5d = min(max(bin_idx_5d, 0), len(vbp_5d) - 1)
                features['composite_vp_density_5d'][t] = vbp_5d[bin_idx_5d] / max(total_5d, 1e-6)

            # Session LVN/HVN distances
            if session_hvn_lvn is not None:
                hvn_prices = session_hvn_lvn['hvn_prices']
                lvn_prices = session_hvn_lvn['lvn_prices']

                if hvn_prices:
                    dists = [abs(level - p) for p in hvn_prices]
                    features['nearest_hvn_dist'][t] = min(dists)

                if lvn_prices:
                    dists = [abs(level - p) for p in lvn_prices]
                    features['nearest_lvn_dist'][t] = min(dists)

            # VP valley depth: min/max volume ratio between price and level
            if day in self._day_vbp_cache:
                session_data = self._day_vbp_cache[day]
                cum_vbp_t = session_data['cum_vbp'][min(t, len(session_data['cum_vbp']) - 1)]
                s_bins = session_data['bin_centers']
                price = day_df['close'].iloc[t]
                lo_p = min(price, level)
                hi_p = max(price, level)
                lo_bin = np.searchsorted(s_bins, lo_p)
                hi_bin = np.searchsorted(s_bins, hi_p)
                lo_bin = min(max(lo_bin, 0), len(s_bins) - 1)
                hi_bin = min(max(hi_bin, 0), len(s_bins) - 1)
                if hi_bin > lo_bin:
                    slice_vol = cum_vbp_t[lo_bin:hi_bin + 1]
                    max_vol = slice_vol.max()
                    min_vol = slice_vol.min()
                    features['vp_valley_depth'][t] = min_vol / max(max_vol, 1e-6)

        return features

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute volume microstructure features for all bars (vectorized per day).

        Note: Features are computed per-bar but are most meaningful when
        used at specific trigger/rejection bars with a known level.
        For general computation, we compute features relative to VWAP.

        For trade-specific features, use compute_features_at_bar() directly.
        """
        result = ohlcv.copy()

        # Initialize feature columns
        for feat in self.feature_names:
            result[feat] = 0.0

        # First pass: build VBP caches for all days
        sorted_days = sorted(ohlcv['trading_day'].unique())
        for day in sorted_days:
            day_df = ohlcv[ohlcv['trading_day'] == day]
            if day not in self._day_vbp_cache:
                self._day_vbp_cache[day] = self._compute_day_vbp(day_df)

        # Second pass: compute features per day
        for day in sorted_days:
            day_df = ohlcv[ohlcv['trading_day'] == day]
            day_idx = day_df.index
            day_data = self._day_vbp_cache[day]

            # Compute features using VWAP as reference level
            if 'vwap' in day_df.columns:
                vwap_vals = day_df['vwap'].values
            else:
                # Compute simple VWAP
                avg_price = (day_df['open'] + day_df['high'] + day_df['low'] + day_df['close']) / 4
                cum_pv = (avg_price * day_df['volume']).cumsum()
                cum_vol = day_df['volume'].cumsum()
                vwap_vals = (cum_pv / cum_vol.clip(lower=1)).values

            # Vectorized feature computation for the day
            day_features = self._compute_day_features_vectorized(day_data, vwap_vals)

            # Composite VP features
            composite_features = self._compute_composite_features_for_day(
                day, sorted_days, day_df, vwap_vals
            )
            day_features.update(composite_features)

            # Assign features to result
            for feat_name, feat_vals in day_features.items():
                result.loc[day_idx, feat_name] = feat_vals

        return result

    def _compute_day_features_vectorized(
        self,
        day_data: Dict,
        levels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute all volume features for a day in a vectorized manner.

        Args:
            day_data: VBP data from _compute_day_vbp
            levels: Array of reference levels (e.g., VWAP) for each bar

        Returns:
            Dictionary of feature arrays
        """
        bin_centers = day_data['bin_centers']
        cum_vbp = day_data['cum_vbp']
        volumes = day_data['volumes']
        T = len(levels)
        n_bins = len(bin_centers)

        features = {feat: np.zeros(T, dtype=np.float64) for feat in self.feature_names}

        # Pre-compute bin indices for all levels
        bins = np.concatenate([[bin_centers[0] - self.bin_size], bin_centers + self.bin_size / 2])
        tick_size = 0.25 * self.level_radius_ticks * 4
        lo_indices = np.clip(np.searchsorted(bin_centers, levels - tick_size), 0, n_bins - 1)
        hi_indices = np.clip(np.searchsorted(bin_centers, levels + tick_size) + 1, 0, n_bins)

        # Cumulative volume at each bar
        total_vol = cum_vbp.sum(axis=1)
        total_vol = np.maximum(total_vol, 1e-6)

        # Average and std volume per bin at each bar
        avg_vol_per_bin = total_vol / n_bins
        std_vol_per_bin = np.std(cum_vbp, axis=1)
        std_vol_per_bin = np.maximum(std_vol_per_bin, 1e-6)

        # POC (point of control) index at each bar
        poc_indices = np.argmax(cum_vbp, axis=1)
        poc_prices = bin_centers[poc_indices]

        # Cumulative volume for rolling average
        cum_vol = np.cumsum(volumes)
        bar_counts = np.arange(1, T + 1)
        avg_vol = cum_vol / bar_counts

        # Compute features for each bar (still need loop but optimized)
        for t in range(T):
            lo_idx = lo_indices[t]
            hi_idx = hi_indices[t]
            vbp_at_bar = cum_vbp[t]

            # Volume at level
            vol_at_level = vbp_at_bar[lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0
            features['vol_at_level'][t] = vol_at_level
            features['vol_at_level_pct'][t] = vol_at_level / total_vol[t]

            # Concentration z-score
            n_level_bins = max(hi_idx - lo_idx, 1)
            vol_per_bin_at_level = vol_at_level / n_level_bins
            features['vol_concentration_z'][t] = (
                (vol_per_bin_at_level - avg_vol_per_bin[t]) / std_vol_per_bin[t]
            )

            # Entropy (simplified - just use concentration as proxy)
            if vol_at_level > 0 and hi_idx > lo_idx:
                local_vbp = vbp_at_bar[lo_idx:hi_idx]
                local_sum = local_vbp.sum()
                if local_sum > 0:
                    local_vbp_norm = np.clip(local_vbp / local_sum, 1e-10, 1.0)
                    features['vol_profile_entropy'][t] = float(scipy_entropy(local_vbp_norm))

            # Volume dynamics
            start_idx = max(0, t - self.lookback_bars)
            features['vol_into_level'][t] = volumes[start_idx:t].sum() if t > start_idx else 0.0
            features['vol_at_rejection_bar'][t] = volumes[t]
            features['vol_ratio_rejection'][t] = volumes[t] / avg_vol[t]

            # Profile shape
            features['poc_distance'][t] = abs(poc_prices[t] - levels[t])
            features['n_peaks_near_level'][t] = _count_peaks(vbp_at_bar[lo_idx:hi_idx])

            # Profile skew (simplified)
            if hi_idx > lo_idx + 2:
                local_vbp = vbp_at_bar[lo_idx:hi_idx]
                local_sum = local_vbp.sum()
                if local_sum > 0:
                    local_bins = bin_centers[lo_idx:hi_idx]
                    mean_price = np.average(local_bins, weights=local_vbp)
                    std_price = np.sqrt(np.average((local_bins - mean_price)**2, weights=local_vbp))
                    if std_price > 0:
                        features['profile_skew_at_level'][t] = float(
                            np.average(((local_bins - mean_price) / std_price)**3, weights=local_vbp)
                        )

        # VA contains level (compute once for all bars)
        for t in range(T):
            vbp_at_bar = cum_vbp[t]
            va_lo, va_hi = _compute_va70(vbp_at_bar, bin_centers)
            features['va_contains_level'][t] = float(va_lo <= levels[t] <= va_hi)

        # Bid/ask features if available
        if self.include_bidask and 'cum_bid_vbp' in day_data:
            cum_bid = day_data['cum_bid_vbp']
            cum_ask = day_data['cum_ask_vbp']
            bid_vols = day_data['bid_volumes']
            ask_vols = day_data['ask_volumes']

            for t in range(T):
                lo_idx = lo_indices[t]
                hi_idx = hi_indices[t]

                bid_at_level = cum_bid[t, lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0
                ask_at_level = cum_ask[t, lo_idx:hi_idx].sum() if hi_idx > lo_idx else 0.0

                features['bid_vol_at_level'][t] = bid_at_level
                features['ask_vol_at_level'][t] = ask_at_level
                features['delta_at_level'][t] = bid_at_level - ask_at_level

                total_at_level = bid_at_level + ask_at_level
                features['delta_at_level_z'][t] = (
                    features['delta_at_level'][t] / total_at_level if total_at_level > 0 else 0.0
                )
                features['bid_ask_ratio_at_level'][t] = bid_at_level / max(ask_at_level, 1e-6)

                # Delta dynamics
                start_idx = max(0, t - self.lookback_bars)
                delta_into = (
                    bid_vols[start_idx:t].sum() - ask_vols[start_idx:t].sum()
                    if t > start_idx else 0.0
                )
                features['delta_into_level'][t] = delta_into

                delta_at_bar = bid_vols[t] - ask_vols[t]
                features['delta_flip_at_level'][t] = float((delta_into * delta_at_bar) < 0)

        return features

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
