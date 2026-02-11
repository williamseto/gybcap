"""
Zone feature extraction for V3 causal zone prediction.

Extracts two types of features at every near-level bar:
1. Scalar features (~85) for XGBoost — aggregated from existing providers
2. Multi-scale VP heatmaps + 1-min sequence for the neural V3 model

All features are backward-looking (causal) relative to the bar being evaluated.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
import pickle


@dataclass
class ZoneFeatureConfig:
    """Configuration for zone feature extraction."""
    # VP heatmap scales
    micro_window: int = 30      # bars (30 min)
    micro_price_bins: int = 20  # ±5pt around level
    micro_price_range: float = 5.0  # points each side

    meso_window: int = 300      # bars (5 hours)
    meso_time_bins: int = 10    # 30 bars per bin
    meso_price_bins: int = 20   # ±15pt around level
    meso_price_range: float = 15.0

    macro_time_bins: int = 15   # 30-min buckets
    macro_price_bins: int = 20  # full range
    macro_bars_per_bin: int = 30

    # Sequence params
    seq_window: int = 60        # bars (1 hour)
    seq_channels: int = 6       # close, vol_z, tick_delta, range, close_pos, cum_delta

    # Negative sampling
    near_level_pts: float = 5.0     # bars within this distance of a level are "near-level"
    neg_subsample_ratio: float = 3.0  # negatives = ratio * positives


class ZoneFeatureExtractor:
    """
    Extract scalar features and VP heatmaps for zone prediction.

    Usage:
        extractor = ZoneFeatureExtractor()
        scalars, heatmaps = extractor.extract(ohlcv, zone_labels)
    """

    def __init__(self, config: Optional[ZoneFeatureConfig] = None):
        self.config = config or ZoneFeatureConfig()

    def extract_scalars(
        self,
        ohlcv: pd.DataFrame,
        zone_labels: pd.DataFrame,
        include_htf: bool = True,
        include_volume: bool = True,
        include_quality: bool = True,
        include_temporal: bool = True,
        include_price_level: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract scalar features for all zone bars + subsampled negatives.

        Args:
            ohlcv: OHLCV DataFrame with level columns and trading_day.
            zone_labels: Output from LevelAnchoredZoneLabeler.fit().
            include_htf: Include higher-timeframe features.
            include_volume: Include volume microstructure features.
            include_quality: Include reversion quality features.
            include_temporal: Include temporal interaction features.
            include_price_level: Include price level features.

        Returns:
            (samples_df, feature_cols): DataFrame of selected bars with features,
                and list of feature column names.
        """
        # Select sample bars: all zone bars + subsampled near-level negatives
        samples_df = self._select_sample_bars(ohlcv, zone_labels)
        print(f"Selected {len(samples_df)} sample bars "
              f"({(samples_df['zone_label'] != 0).sum()} positive, "
              f"{(samples_df['zone_label'] == 0).sum()} negative)")

        feature_cols = []

        # Compute features on full dataset, then slice to samples
        if include_htf:
            print("  Computing higher timeframe features...")
            from strategies.features.higher_timeframe import HigherTimeframeProvider
            htf = HigherTimeframeProvider()
            htf_df = htf._compute_impl(ohlcv)
            for col in htf.feature_names:
                if col in htf_df.columns:
                    ohlcv[col] = htf_df[col].values
                    feature_cols.append(col)

        if include_volume:
            print("  Computing volume microstructure features...")
            from strategies.features.volume_microstructure import VolumeMicrostructureProvider
            has_bidask = 'bidvolume' in ohlcv.columns
            vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
            vol_df = vmp._compute_impl(ohlcv)
            for col in vmp.feature_names:
                if col in vol_df.columns:
                    ohlcv[col] = vol_df[col].values
                    feature_cols.append(col)

        if include_quality:
            print("  Computing reversion quality features...")
            from strategies.features.reversion_quality import ReversionQualityProvider
            rqp = ReversionQualityProvider()
            qual_df = rqp._compute_impl(ohlcv)
            for col in rqp.feature_names:
                if col in qual_df.columns:
                    ohlcv[col] = qual_df[col].values
                    feature_cols.append(col)

        if include_temporal:
            print("  Computing temporal interaction features...")
            from strategies.features.temporal_interactions import TemporalInteractionProvider
            tip = TemporalInteractionProvider()
            temp_df = tip._compute_impl(ohlcv)
            for col in tip.feature_names:
                if col in temp_df.columns:
                    ohlcv[col] = temp_df[col].values
                    feature_cols.append(col)

        if include_price_level:
            print("  Computing price level features...")
            from strategies.features.price_levels import PriceLevelProvider
            plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
            pl_df = plp._compute_impl(ohlcv)
            for col in plp.feature_names:
                if col in pl_df.columns:
                    ohlcv[col] = pl_df[col].values
                    feature_cols.append(col)

        # Add level-distance feature: distance to nearest level for ALL bars
        # (computed fresh, not from zone_labels which only has it for zone bars)
        from strategies.labeling.reversal_zones import LevelAnchoredZoneLabeler
        close_vals = ohlcv['close'].values
        min_dist = np.full(len(ohlcv), np.inf)
        for lvl_name in LevelAnchoredZoneLabeler.TRACKED_LEVELS:
            if lvl_name in ohlcv.columns:
                lvl_vals = ohlcv[lvl_name].values
                dist = np.abs(close_vals - lvl_vals)
                valid = ~np.isnan(dist)
                min_dist[valid] = np.minimum(min_dist[valid], dist[valid])
        min_dist[np.isinf(min_dist)] = np.nan
        ohlcv['level_distance_norm'] = min_dist / np.maximum(close_vals, 1) * 1000
        feature_cols.append('level_distance_norm')

        # --- Level confluence features ---
        print("  Computing level confluence features...")
        level_strength = {
            'vwap': 3, 'ovn_lo': 2, 'ovn_hi': 2,
            'rth_lo': 2, 'rth_hi': 2, 'prev_high': 1, 'prev_low': 1,
        }
        tracked = [l for l in LevelAnchoredZoneLabeler.TRACKED_LEVELS if l in ohlcv.columns]

        n_within_3pt = np.zeros(len(ohlcv), dtype=np.float64)
        n_within_5pt = np.zeros(len(ohlcv), dtype=np.float64)
        cluster_score = np.zeros(len(ohlcv), dtype=np.float64)
        type_strength = np.zeros(len(ohlcv), dtype=np.float64)

        if tracked:
            # For each bar, find nearest level, then count other levels nearby
            all_level_vals = {}
            for lvl_name in tracked:
                all_level_vals[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

            # Find nearest level per bar
            nearest_level_name = np.empty(len(ohlcv), dtype=object)
            nearest_level_price = np.full(len(ohlcv), np.nan)
            for i in range(len(ohlcv)):
                best_dist = np.inf
                for lvl_name in tracked:
                    d = abs(close_vals[i] - all_level_vals[lvl_name][i])
                    if not np.isnan(d) and d < best_dist:
                        best_dist = d
                        nearest_level_name[i] = lvl_name
                        nearest_level_price[i] = all_level_vals[lvl_name][i]

            # Compute confluence features
            for i in range(len(ohlcv)):
                ref_price = nearest_level_price[i]
                ref_name = nearest_level_name[i]
                if np.isnan(ref_price):
                    continue
                type_strength[i] = level_strength.get(ref_name, 1)
                for lvl_name in tracked:
                    if lvl_name == ref_name:
                        continue
                    d = abs(all_level_vals[lvl_name][i] - ref_price)
                    if np.isnan(d):
                        continue
                    if d <= 3.0:
                        n_within_3pt[i] += 1
                    if d <= 5.0:
                        n_within_5pt[i] += 1
                    cluster_score[i] += np.exp(-d / 2.0)

        ohlcv['n_levels_within_3pt'] = n_within_3pt
        ohlcv['n_levels_within_5pt'] = n_within_5pt
        ohlcv['level_cluster_score'] = cluster_score
        ohlcv['level_type_strength'] = type_strength
        feature_cols.extend(['n_levels_within_3pt', 'n_levels_within_5pt',
                            'level_cluster_score', 'level_type_strength'])

        # Slice to sample bars
        sample_indices = samples_df.index
        result = ohlcv.loc[sample_indices].copy()

        # Attach zone labels
        for col in ['zone_label', 'zone_probability', 'nearest_level',
                     'level_distance', 'bars_to_reversal']:
            result[col] = zone_labels.loc[sample_indices, col].values

        # Remove duplicates in feature_cols
        feature_cols = list(dict.fromkeys(feature_cols))

        print(f"  Total scalar features: {len(feature_cols)}")
        return result, feature_cols

    def extract_heatmaps(
        self,
        ohlcv: pd.DataFrame,
        zone_labels: pd.DataFrame,
        sample_indices: np.ndarray,
        cache_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract multi-scale VP heatmaps + sequences for sample bars.

        Args:
            ohlcv: Full OHLCV DataFrame with level columns.
            zone_labels: Output from LevelAnchoredZoneLabeler.
            sample_indices: Indices into ohlcv for which to extract heatmaps.
            cache_dir: Optional directory to cache/load results.

        Returns:
            Dict with keys: 'micro_vp', 'meso_vp', 'macro_vp', 'sequence'
            Each is np.ndarray indexed by position in sample_indices.
        """
        if cache_dir and os.path.exists(os.path.join(cache_dir, 'heatmaps.pkl')):
            print("  Loading cached heatmaps...")
            with open(os.path.join(cache_dir, 'heatmaps.pkl'), 'rb') as f:
                return pickle.load(f)

        cfg = self.config
        N = len(sample_indices)

        micro_vps = np.zeros((N, 1, cfg.micro_price_bins, cfg.micro_window), dtype=np.float32)
        meso_vps = np.zeros((N, 1, cfg.meso_price_bins, cfg.meso_time_bins), dtype=np.float32)
        macro_vps = np.zeros((N, 1, cfg.macro_price_bins, cfg.macro_time_bins), dtype=np.float32)
        sequences = np.zeros((N, cfg.seq_window, cfg.seq_channels), dtype=np.float32)

        # Pre-extract arrays
        close_arr = ohlcv['close'].values.astype(np.float64)
        high_arr = ohlcv['high'].values.astype(np.float64)
        low_arr = ohlcv['low'].values.astype(np.float64)
        open_arr = ohlcv['open'].values.astype(np.float64)
        vol_arr = ohlcv['volume'].values.astype(np.float64)

        # Get trading_day boundaries for macro VP
        day_starts = {}
        for day, grp in ohlcv.groupby('trading_day'):
            day_starts[day] = grp.index[0]

        trading_days = ohlcv['trading_day'].values

        print(f"  Extracting heatmaps for {N} bars...")
        for i, idx in enumerate(sample_indices):
            if i % 1000 == 0 and i > 0:
                print(f"    {i}/{N}...")

            level_name = zone_labels.loc[idx, 'nearest_level']
            if level_name and level_name in ohlcv.columns:
                level_price = float(ohlcv.loc[idx, level_name])
            else:
                level_price = close_arr[idx]

            if np.isnan(level_price) or level_price == 0:
                level_price = close_arr[idx]

            # --- Micro VP: (1, 20, 30) ---
            micro_start = max(0, idx - cfg.micro_window)
            micro_end = idx
            if micro_end > micro_start:
                micro_vps[i] = self._build_vp_heatmap(
                    close_arr[micro_start:micro_end],
                    vol_arr[micro_start:micro_end],
                    level_price, cfg.micro_price_range,
                    cfg.micro_price_bins, cfg.micro_window,
                )

            # --- Meso VP: (1, 20, 10) ---
            meso_start = max(0, idx - cfg.meso_window)
            meso_end = idx
            if meso_end > meso_start:
                meso_vps[i] = self._build_vp_heatmap_binned(
                    close_arr[meso_start:meso_end],
                    vol_arr[meso_start:meso_end],
                    level_price, cfg.meso_price_range,
                    cfg.meso_price_bins, cfg.meso_time_bins,
                )

            # --- Macro VP: (1, 20, 15) — from session start to bar ---
            day = trading_days[idx]
            day_start = day_starts.get(day, idx)
            if idx > day_start:
                macro_vps[i] = self._build_vp_heatmap_binned(
                    close_arr[day_start:idx],
                    vol_arr[day_start:idx],
                    level_price, None,  # full range
                    cfg.macro_price_bins, cfg.macro_time_bins,
                )

            # --- Sequence: (60, 6) ---
            seq_start = max(0, idx - cfg.seq_window)
            seq_end = idx
            seq_len = seq_end - seq_start
            if seq_len > 0:
                seq = self._build_sequence(
                    open_arr[seq_start:seq_end],
                    high_arr[seq_start:seq_end],
                    low_arr[seq_start:seq_end],
                    close_arr[seq_start:seq_end],
                    vol_arr[seq_start:seq_end],
                    level_price,
                )
                # Right-align into fixed-width array
                sequences[i, cfg.seq_window - seq_len:, :] = seq

        result = {
            'micro_vp': micro_vps,
            'meso_vp': meso_vps,
            'macro_vp': macro_vps,
            'sequence': sequences,
        }

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, 'heatmaps.pkl'), 'wb') as f:
                pickle.dump(result, f)
            print(f"  Cached heatmaps to {cache_dir}/heatmaps.pkl")

        return result

    def _select_sample_bars(
        self,
        ohlcv: pd.DataFrame,
        zone_labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select zone bars + subsampled near-level negatives.

        Returns a DataFrame indexed like ohlcv with zone_label column.
        """
        cfg = self.config

        # Positive bars: any bar in a zone
        pos_mask = zone_labels['zone_label'] != 0
        pos_indices = zone_labels.index[pos_mask].tolist()

        # Near-level bars: bars close to any tracked level but NOT in a zone
        from strategies.labeling.reversal_zones import LevelAnchoredZoneLabeler
        near_level_mask = np.zeros(len(ohlcv), dtype=bool)

        for lvl_name in LevelAnchoredZoneLabeler.TRACKED_LEVELS:
            if lvl_name in ohlcv.columns:
                lvl_vals = ohlcv[lvl_name].values
                close_vals = ohlcv['close'].values
                dist = np.abs(close_vals - lvl_vals)
                near_level_mask |= (dist <= cfg.near_level_pts)

        # Negative bars: near level but not in a zone
        neg_mask = near_level_mask & ~pos_mask.values
        neg_indices = ohlcv.index[neg_mask].tolist()

        # Subsample negatives
        n_pos = len(pos_indices)
        n_neg_target = int(n_pos * cfg.neg_subsample_ratio)
        if len(neg_indices) > n_neg_target:
            rng = np.random.RandomState(42)
            neg_indices = list(rng.choice(neg_indices, n_neg_target, replace=False))

        all_indices = sorted(set(pos_indices + neg_indices))
        samples = ohlcv.loc[all_indices].copy()
        samples['zone_label'] = zone_labels.loc[all_indices, 'zone_label'].values

        return samples

    def _build_vp_heatmap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        center_price: float,
        price_range: float,
        n_price_bins: int,
        n_time_bins: int,
    ) -> np.ndarray:
        """
        Build VP heatmap with 1:1 time bins (one bar per column).
        Returns shape (1, n_price_bins, n_time_bins).
        """
        heatmap = np.zeros((1, n_price_bins, n_time_bins), dtype=np.float32)
        n_bars = len(prices)

        price_lo = center_price - price_range
        price_hi = center_price + price_range
        bin_width = (price_hi - price_lo) / n_price_bins

        if bin_width <= 0 or n_bars == 0:
            return heatmap

        for t in range(min(n_bars, n_time_bins)):
            # Right-align: most recent bar at rightmost column
            col = n_time_bins - n_bars + t
            if col < 0:
                continue
            p = prices[t]
            v = volumes[t]
            row = int((p - price_lo) / bin_width)
            row = max(0, min(n_price_bins - 1, row))
            heatmap[0, row, col] += v

        # Normalize
        total = heatmap.sum()
        if total > 0:
            heatmap /= total

        return heatmap

    def _build_vp_heatmap_binned(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        center_price: float,
        price_range: Optional[float],
        n_price_bins: int,
        n_time_bins: int,
    ) -> np.ndarray:
        """
        Build VP heatmap with aggregated time bins.
        Returns shape (1, n_price_bins, n_time_bins).
        """
        heatmap = np.zeros((1, n_price_bins, n_time_bins), dtype=np.float32)
        n_bars = len(prices)

        if n_bars == 0:
            return heatmap

        # Auto-range: use actual price range
        if price_range is None:
            margin = 2.0
            price_lo = prices.min() - margin
            price_hi = prices.max() + margin
        else:
            price_lo = center_price - price_range
            price_hi = center_price + price_range

        bin_width = (price_hi - price_lo) / n_price_bins
        if bin_width <= 0:
            return heatmap

        # Compute bars per time bin
        bars_per_bin = max(1, n_bars // n_time_bins)

        for t in range(n_bars):
            time_bin = min(t // bars_per_bin, n_time_bins - 1)
            p = prices[t]
            v = volumes[t]
            row = int((p - price_lo) / bin_width)
            row = max(0, min(n_price_bins - 1, row))
            heatmap[0, row, time_bin] += v

        # Normalize
        total = heatmap.sum()
        if total > 0:
            heatmap /= total

        return heatmap

    def _build_sequence(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        level_price: float,
    ) -> np.ndarray:
        """
        Build 6-channel sequence for TCN input.

        Channels:
            0: normalized_close — (close - level) / level * 1000
            1: volume_z — z-scored volume
            2: tick_delta — sign(close - open) * volume, normalized
            3: range — (high - low) / level * 1000
            4: close_position — (close - low) / (high - low)
            5: cumulative_delta — running sum of tick_delta

        Returns shape (T, 6).
        """
        T = len(closes)
        seq = np.zeros((T, 6), dtype=np.float32)

        # Channel 0: normalized close
        if level_price > 0:
            seq[:, 0] = (closes - level_price) / level_price * 1000
        else:
            seq[:, 0] = 0

        # Channel 1: volume z-score
        vol_mean = volumes.mean()
        vol_std = volumes.std()
        if vol_std > 0:
            seq[:, 1] = (volumes - vol_mean) / vol_std
        else:
            seq[:, 1] = 0

        # Channel 2: tick delta (signed volume)
        signs = np.sign(closes - opens)
        tick_delta = signs * volumes
        td_std = tick_delta.std()
        if td_std > 0:
            seq[:, 2] = tick_delta / td_std
        else:
            seq[:, 2] = 0

        # Channel 3: range
        if level_price > 0:
            seq[:, 3] = (highs - lows) / level_price * 1000
        else:
            seq[:, 3] = 0

        # Channel 4: close position in bar
        bar_range = highs - lows
        valid = bar_range > 0
        seq[valid, 4] = (closes[valid] - lows[valid]) / bar_range[valid]
        seq[~valid, 4] = 0.5

        # Channel 5: cumulative delta
        cum_delta = np.cumsum(tick_delta)
        cd_std = cum_delta.std()
        if cd_std > 0:
            seq[:, 5] = cum_delta / cd_std
        else:
            seq[:, 5] = 0

        return seq
