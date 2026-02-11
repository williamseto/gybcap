#!/usr/bin/env python
"""
Signal Detection for Reversal vs Breakout at Price Levels.

Systematically tests whether ANY features (old or new) can distinguish
reversal from breakout outcomes at tracked price levels.

Tests:
  1. Univariate: Cohen's d, KS test, Mutual Information, conditional reversal rates
  2. Conditional: BB extreme + level, range exhaustion, VP structure, level confluence
  3. Per-level-type: separate analysis per level (vwap, ovn, prev, etc.)
  4. Temporal regime: volatility terciles, time-of-day

Decision criteria (GO/NO-GO):
  - ANY feature with |Cohen's d| >= 0.2
  - ANY conditional reversal rate difference >= 5pp
  - ANY mutual information > 0.01 bits
  - ANY per-level-type d >= 0.3
  - ANY regime-conditional d >= 0.3

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/signal_detection.py
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURE_DIR = "sandbox/figures/signal_detection"
DATA_PATH = "raw_data/es_min_3y_clean_td_gamma.csv"

from strategies.labeling.reversal_zones import TRACKED_LEVELS


# ── Data loading ──────────────────────────────────────────────────────────

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv):
    from strategies.features.price_levels import PriceLevelProvider
    print("\nComputing price levels...")
    plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
    feat_df = plp._compute_impl(ohlcv)

    level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']
    if 'dt' in feat_df.columns:
        feat_df = feat_df.set_index('dt')
    ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv
    for col in level_cols:
        if col in feat_df.columns:
            ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

    levels = plp.prev_day_levels(ohlcv)
    ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
    ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])
    return ohlcv


def compute_all_features(ohlcv):
    """Compute all feature providers on the data."""
    feature_cols = []

    print("\nComputing higher timeframe features...")
    from strategies.features.higher_timeframe import HigherTimeframeProvider
    htf = HigherTimeframeProvider()
    htf_df = htf._compute_impl(ohlcv)
    for col in htf.feature_names:
        if col in htf_df.columns:
            ohlcv[col] = htf_df[col].values
            feature_cols.append(col)

    print("Computing volume microstructure features...")
    from strategies.features.volume_microstructure import VolumeMicrostructureProvider
    has_bidask = 'bidvolume' in ohlcv.columns
    vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
    vol_df = vmp._compute_impl(ohlcv)
    for col in vmp.feature_names:
        if col in vol_df.columns:
            ohlcv[col] = vol_df[col].values
            feature_cols.append(col)

    print("Computing reversion quality features...")
    from strategies.features.reversion_quality import ReversionQualityProvider
    rqp = ReversionQualityProvider()
    qual_df = rqp._compute_impl(ohlcv)
    for col in rqp.feature_names:
        if col in qual_df.columns:
            ohlcv[col] = qual_df[col].values
            feature_cols.append(col)

    print("Computing temporal interaction features...")
    from strategies.features.temporal_interactions import TemporalInteractionProvider
    tip = TemporalInteractionProvider()
    temp_df = tip._compute_impl(ohlcv)
    for col in tip.feature_names:
        if col in temp_df.columns:
            ohlcv[col] = temp_df[col].values
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"  Total features: {len(feature_cols)}")
    return ohlcv, feature_cols


def label_reversals_breakouts(ohlcv, proximity_pts=5.0, forward_bars=45,
                               reversal_pts=6.0, breakout_pts=6.0):
    """
    Label near-level bars as reversal, breakout, or inconclusive.

    A bar near a level is:
    - REVERSAL if price bounces away from level by reversal_pts within forward_bars
    - BREAKOUT if price pushes through level by breakout_pts within forward_bars
    - INCONCLUSIVE otherwise
    """
    print(f"\nLabeling reversals/breakouts (proximity={proximity_pts}pt, "
          f"reversal={reversal_pts}pt, breakout={breakout_pts}pt)...")

    close_arr = ohlcv['close'].values
    high_arr = ohlcv['high'].values
    low_arr = ohlcv['low'].values
    n = len(ohlcv)

    labels = np.full(n, -1, dtype=np.int32)  # -1 = not near level
    nearest_level_name = np.empty(n, dtype=object)
    nearest_level_price = np.full(n, np.nan)

    # Find nearest level for each bar
    for lvl_name in TRACKED_LEVELS:
        if lvl_name not in ohlcv.columns:
            continue
        lvl_vals = ohlcv[lvl_name].values
        for i in range(n):
            d = abs(close_arr[i] - lvl_vals[i])
            if np.isnan(d):
                continue
            if d <= proximity_pts:
                prev_dist = abs(close_arr[i] - nearest_level_price[i]) if not np.isnan(nearest_level_price[i]) else np.inf
                if d < prev_dist:
                    nearest_level_name[i] = lvl_name
                    nearest_level_price[i] = lvl_vals[i]

    near_level_mask = ~np.isnan(nearest_level_price)
    n_near = near_level_mask.sum()
    print(f"  {n_near:,} near-level bars out of {n:,} ({100*n_near/n:.1f}%)")

    # Label each near-level bar
    for i in range(n):
        if not near_level_mask[i]:
            continue

        level = nearest_level_price[i]
        price = close_arr[i]
        is_above = price >= level  # Price above level → level is support

        end = min(i + forward_bars, n)
        future_highs = high_arr[i+1:end]
        future_lows = low_arr[i+1:end]

        if len(future_highs) == 0:
            labels[i] = 2  # inconclusive
            continue

        if is_above:
            # Level is support. Reversal = price bounces up. Breakout = price breaks below.
            max_up = future_highs.max() - price
            max_down = price - future_lows.min()

            if max_up >= reversal_pts and (max_down < breakout_pts or
                np.argmax(future_highs >= price + reversal_pts) < np.argmax(future_lows <= level - breakout_pts + level - price) if (future_lows <= level - breakout_pts + level - price).any() else True):
                labels[i] = 1  # reversal
            elif max_down >= breakout_pts:
                labels[i] = 0  # breakout
            else:
                labels[i] = 2  # inconclusive
        else:
            # Level is resistance. Reversal = price bounces down. Breakout = price breaks above.
            max_down = price - future_lows.min()
            max_up = future_highs.max() - price

            if max_down >= reversal_pts and (max_up < breakout_pts or
                np.argmax(future_lows <= price - reversal_pts) < np.argmax(future_highs >= level + breakout_pts - (level - price)) if (future_highs >= level + breakout_pts - (level - price)).any() else True):
                labels[i] = 1  # reversal
            elif max_up >= breakout_pts:
                labels[i] = 0  # breakout
            else:
                labels[i] = 2  # inconclusive

    ohlcv['outcome'] = labels
    ohlcv['nearest_level_name'] = nearest_level_name
    ohlcv['nearest_level_price'] = nearest_level_price

    n_rev = (labels == 1).sum()
    n_bo = (labels == 0).sum()
    n_inc = (labels == 2).sum()
    print(f"  Reversal: {n_rev:,} ({100*n_rev/max(n_near,1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100*n_bo/max(n_near,1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100*n_inc/max(n_near,1):.1f}%)")

    return ohlcv


# ── Level confluence features (computed inline) ───────────────────────────

def compute_confluence_features(ohlcv):
    """Compute level confluence features directly on ohlcv."""
    print("\nComputing level confluence features...")
    close_vals = ohlcv['close'].values
    n = len(ohlcv)

    level_strength = {
        'vwap': 3, 'ovn_lo': 2, 'ovn_hi': 2,
        'rth_lo': 2, 'rth_hi': 2, 'prev_high': 1, 'prev_low': 1,
    }
    tracked = [l for l in TRACKED_LEVELS if l in ohlcv.columns]

    n_within_3pt = np.zeros(n, dtype=np.float64)
    n_within_5pt = np.zeros(n, dtype=np.float64)
    cluster_score = np.zeros(n, dtype=np.float64)
    type_strength = np.zeros(n, dtype=np.float64)

    nearest_price = ohlcv['nearest_level_price'].values
    nearest_name = ohlcv['nearest_level_name'].values

    all_level_vals = {}
    for lvl_name in tracked:
        all_level_vals[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

    for i in range(n):
        ref_price = nearest_price[i]
        ref_name = nearest_name[i]
        if np.isnan(ref_price) or ref_name is None:
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

    return ohlcv, ['n_levels_within_3pt', 'n_levels_within_5pt',
                   'level_cluster_score', 'level_type_strength']


# ── Statistical tests ─────────────────────────────────────────────────────

def cohens_d(group_a, group_b):
    """Compute Cohen's d effect size."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return 0.0
    ma, mb = group_a.mean(), group_b.mean()
    sa, sb = group_a.std(ddof=1), group_b.std(ddof=1)
    pooled_std = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (ma - mb) / pooled_std


def ks_test(group_a, group_b):
    """KS test between two distributions."""
    if len(group_a) < 5 or len(group_b) < 5:
        return 1.0
    _, p = sp_stats.ks_2samp(group_a, group_b)
    return p


def mutual_info_binned(feature, binary_label, n_bins=10):
    """Compute MI between continuous feature and binary label using binning."""
    if len(feature) < 20:
        return 0.0
    # Remove NaN/inf
    mask = np.isfinite(feature) & np.isfinite(binary_label.astype(float))
    feature = feature[mask]
    binary_label = binary_label[mask]
    if len(feature) < 20:
        return 0.0
    # Bin the feature
    try:
        binned = pd.qcut(pd.Series(feature), q=n_bins, labels=False, duplicates='drop').values
        # Remove any NaN bins
        valid = ~np.isnan(binned)
        binned = binned[valid].astype(int)
        binary_label = binary_label[valid].astype(int)
        if len(binned) < 20:
            return 0.0
        return mutual_info_score(binned, binary_label)
    except (ValueError, TypeError):
        return 0.0


def conditional_reversal_rate(feature, outcome, n_bins=5):
    """
    Compute reversal rate per feature quintile.
    Returns (bin_labels, reversal_rates, counts).
    """
    mask = np.isfinite(feature) & (outcome >= 0) & (outcome <= 1)
    feature = feature[mask]
    outcome = outcome[mask]
    if len(feature) < 50:
        return [], [], []

    try:
        bins = pd.qcut(pd.Series(feature), q=n_bins, labels=False, duplicates='drop').values
    except ValueError:
        return [], [], []

    unique_bins = sorted(np.unique(bins[~np.isnan(bins)]).astype(int))
    rates = []
    counts = []
    labels = []
    for b in unique_bins:
        mask_b = bins == b
        n = mask_b.sum()
        if n < 10:
            continue
        rev_rate = outcome[mask_b].mean()
        rates.append(rev_rate)
        counts.append(n)
        labels.append(f"Q{b+1}")

    return labels, rates, counts


# ── Phase 2A: Univariate Analysis ────────────────────────────────────────

def run_univariate_analysis(df, feature_cols, outcome_col='outcome'):
    """
    For each feature, compute Cohen's d, KS test, MI between
    reversal (1) and breakout (0) distributions.
    """
    print("\n" + "="*80)
    print("PHASE 2A: UNIVARIATE ANALYSIS")
    print("="*80)

    # Filter to reversal (1) and breakout (0) only
    mask = df[outcome_col].isin([0, 1])
    df_rb = df[mask].copy()
    rev_mask = df_rb[outcome_col] == 1
    bo_mask = df_rb[outcome_col] == 0

    n_rev = rev_mask.sum()
    n_bo = bo_mask.sum()
    print(f"\nSample: {n_rev:,} reversals, {n_bo:,} breakouts")

    results = []
    n_features = len(feature_cols)
    bonferroni = 0.001 / n_features  # Bonferroni-corrected threshold

    for feat in feature_cols:
        if feat not in df_rb.columns:
            continue
        vals = df_rb[feat].values.astype(np.float64)
        rev_vals = vals[rev_mask.values]
        bo_vals = vals[bo_mask.values]

        # Remove non-finite values
        rev_vals = rev_vals[np.isfinite(rev_vals)]
        bo_vals = bo_vals[np.isfinite(bo_vals)]

        d = cohens_d(rev_vals, bo_vals)
        p = ks_test(rev_vals, bo_vals)
        mi = mutual_info_binned(vals, df_rb[outcome_col].values)

        # Conditional reversal rate
        labels, rates, counts = conditional_reversal_rate(
            vals, df_rb[outcome_col].values
        )
        rate_range = max(rates) - min(rates) if rates else 0.0

        results.append({
            'feature': feat,
            'cohens_d': d,
            'abs_d': abs(d),
            'ks_p': p,
            'ks_sig': p < bonferroni,
            'mi': mi,
            'rate_range': rate_range,
            'rev_mean': np.mean(rev_vals) if len(rev_vals) > 0 else 0,
            'bo_mean': np.mean(bo_vals) if len(bo_vals) > 0 else 0,
        })

    results_df = pd.DataFrame(results).sort_values('abs_d', ascending=False)

    # Print top features
    print(f"\n{'Feature':<35} {'Cohen d':>8} {'KS p':>10} {'MI':>8} {'Rate Δ':>8}")
    print("-" * 75)

    signal_found = False
    for _, row in results_df.head(30).iterrows():
        marker = ""
        if abs(row['cohens_d']) >= 0.2:
            marker = " *** SIGNAL ***"
            signal_found = True
        elif abs(row['cohens_d']) >= 0.1:
            marker = " *"
        print(f"{row['feature']:<35} {row['cohens_d']:>8.4f} {row['ks_p']:>10.2e} "
              f"{row['mi']:>8.4f} {row['rate_range']:>8.4f}{marker}")

    # Summary
    n_above_02 = (results_df['abs_d'] >= 0.2).sum()
    n_above_01 = (results_df['abs_d'] >= 0.1).sum()
    n_ks_sig = results_df['ks_sig'].sum()
    n_mi_above = (results_df['mi'] > 0.01).sum()

    print(f"\n  Features with |d| >= 0.2: {n_above_02}")
    print(f"  Features with |d| >= 0.1: {n_above_01}")
    print(f"  Features with KS p < {bonferroni:.2e} (Bonferroni): {n_ks_sig}")
    print(f"  Features with MI > 0.01: {n_mi_above}")

    # NEW features specifically
    new_features = [
        'composite_vp_density_3d', 'composite_vp_density_5d', 'level_is_hvn_3d',
        'level_is_lvn_3d', 'nearest_lvn_dist', 'nearest_hvn_dist', 'vp_valley_depth',
        'composite_poc_dist', 'range_pct_of_atr', 'range_pct_of_prior_range',
        'distance_from_open_atr', 'session_range_percentile', 'remaining_atr_pct',
        'prior_va_width_actual', 'bb_side_alignment', 'trend_exhaustion_score',
        'weekly_trend_alignment', 'approach_velocity_10bar', 'approach_deceleration',
        'bars_since_session_extreme', 'vol_trend_into_level', 'bar_size_trend_into_level',
        'consecutive_same_dir', 'approach_cum_delta_z', 'n_levels_within_3pt',
        'n_levels_within_5pt', 'level_cluster_score', 'level_type_strength',
    ]
    new_in_results = results_df[results_df['feature'].isin(new_features)]
    if not new_in_results.empty:
        print(f"\n  NEW features specifically:")
        for _, row in new_in_results.head(15).iterrows():
            marker = " ***" if abs(row['cohens_d']) >= 0.2 else ""
            print(f"    {row['feature']:<35} d={row['cohens_d']:>8.4f}{marker}")

    return results_df, signal_found


# ── Phase 2B: Conditional Analysis ───────────────────────────────────────

def run_conditional_analysis(df, outcome_col='outcome'):
    """
    Test pairwise interactions. A feature might show no marginal signal
    but have signal conditional on another feature.
    """
    print("\n" + "="*80)
    print("PHASE 2B: CONDITIONAL ANALYSIS")
    print("="*80)

    mask = df[outcome_col].isin([0, 1])
    df_rb = df[mask].copy()
    rev_rate_all = (df_rb[outcome_col] == 1).mean()
    print(f"\nBase reversal rate: {rev_rate_all:.4f}")

    signal_found = False
    conditions = []

    # 1. BB extreme + level side
    if 'daily_bb_pct_b' in df_rb.columns:
        for bb_low, bb_high, label in [(0, 0.1, "BB<0.1 (oversold)"),
                                        (0.9, 1.1, "BB>0.9 (overbought)"),
                                        (0.3, 0.7, "BB 0.3-0.7 (middle)")]:
            cond = (df_rb['daily_bb_pct_b'] >= bb_low) & (df_rb['daily_bb_pct_b'] <= bb_high)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 2. Range exhaustion + level
    if 'range_pct_of_atr' in df_rb.columns:
        for lo, hi, label in [(0, 0.5, "Range<0.5×ATR"),
                               (0.5, 1.0, "Range 0.5-1.0×ATR"),
                               (1.0, 99, "Range>1.0×ATR (extended)")]:
            cond = (df_rb['range_pct_of_atr'] >= lo) & (df_rb['range_pct_of_atr'] < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 3. VP structure + level
    if 'level_is_hvn_3d' in df_rb.columns:
        for col, val, label in [('level_is_hvn_3d', 1, "Level is HVN (3d)"),
                                 ('level_is_hvn_3d', 0, "Level is NOT HVN"),
                                 ('level_is_lvn_3d', 1, "Level is LVN (3d)"),
                                 ('level_is_lvn_3d', 0, "Level is NOT LVN")]:
            if col not in df_rb.columns:
                continue
            cond = df_rb[col] == val
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 4. Level confluence + outcome
    if 'n_levels_within_3pt' in df_rb.columns:
        for lo, hi, label in [(0, 0.5, "0 other levels within 3pt"),
                               (1, 1.5, "1 level within 3pt"),
                               (2, 99, "2+ levels within 3pt (confluence)")]:
            cond = (df_rb['n_levels_within_3pt'] >= lo) & (df_rb['n_levels_within_3pt'] < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 5. Approach dynamics
    if 'approach_velocity_10bar' in df_rb.columns:
        vel = df_rb['approach_velocity_10bar']
        for lo, hi, label in [(-99, -0.3, "Fast approach down"),
                               (-0.3, 0.3, "Slow approach"),
                               (0.3, 99, "Fast approach up")]:
            cond = (vel >= lo) & (vel < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 6. Trend exhaustion
    if 'trend_exhaustion_score' in df_rb.columns:
        score = df_rb['trend_exhaustion_score']
        for lo, hi, label in [(-1, -0.3, "Exhaustion (bearish)"),
                               (-0.3, 0.3, "No exhaustion"),
                               (0.3, 1, "Exhaustion (bullish)")]:
            cond = (score >= lo) & (score < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # 7. Bars since session extreme
    if 'bars_since_session_extreme' in df_rb.columns:
        bse = df_rb['bars_since_session_extreme']
        for lo, hi, label in [(0, 5, "Recent extreme (0-5 bars)"),
                               (5, 30, "Moderate (5-30 bars)"),
                               (30, 9999, "Stale extreme (30+ bars)")]:
            cond = (bse >= lo) & (bse < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            n = cond.sum()
            conditions.append((label, rate, n))

    # Print results
    print(f"\n{'Condition':<40} {'Rev Rate':>10} {'N':>8} {'Δ from base':>12}")
    print("-" * 72)
    for label, rate, n in conditions:
        delta = rate - rev_rate_all
        marker = ""
        if abs(delta) >= 0.05:
            marker = " *** SIGNAL ***"
            signal_found = True
        elif abs(delta) >= 0.03:
            marker = " *"
        print(f"{label:<40} {rate:>10.4f} {n:>8,} {delta:>+12.4f}{marker}")

    # Check max delta
    if conditions:
        max_delta = max(abs(r - rev_rate_all) for _, r, _ in conditions)
        print(f"\n  Max reversal rate delta: {max_delta:.4f}")
        print(f"  Threshold: 0.05 (5pp)")
        print(f"  Signal found: {'YES' if signal_found else 'NO'}")

    return conditions, signal_found


# ── Phase 2C: Per-Level-Type Analysis ────────────────────────────────────

def run_per_level_analysis(df, feature_cols, outcome_col='outcome'):
    """
    Compute signal metrics separately per level type.
    """
    print("\n" + "="*80)
    print("PHASE 2C: PER-LEVEL-TYPE ANALYSIS")
    print("="*80)

    mask = df[outcome_col].isin([0, 1])
    df_rb = df[mask].copy()

    signal_found = False
    level_results = {}

    for level_name in TRACKED_LEVELS:
        level_mask = df_rb['nearest_level_name'] == level_name
        n_level = level_mask.sum()
        if n_level < 100:
            continue

        rev_mask = (df_rb[outcome_col] == 1) & level_mask
        bo_mask = (df_rb[outcome_col] == 0) & level_mask
        n_rev = rev_mask.sum()
        n_bo = bo_mask.sum()
        rev_rate = n_rev / max(n_rev + n_bo, 1)

        print(f"\n  {level_name}: {n_level:,} bars ({n_rev:,} rev, {n_bo:,} bo, rate={rev_rate:.3f})")

        # Find best features for this level
        best_d = 0
        best_feat = ""
        for feat in feature_cols:
            if feat not in df_rb.columns:
                continue
            rev_vals = df_rb.loc[rev_mask, feat].values.astype(np.float64)
            bo_vals = df_rb.loc[bo_mask, feat].values.astype(np.float64)
            rev_vals = rev_vals[np.isfinite(rev_vals)]
            bo_vals = bo_vals[np.isfinite(bo_vals)]
            d = cohens_d(rev_vals, bo_vals)
            if abs(d) > abs(best_d):
                best_d = d
                best_feat = feat

        marker = ""
        if abs(best_d) >= 0.3:
            marker = " *** SIGNAL ***"
            signal_found = True
        elif abs(best_d) >= 0.2:
            marker = " *"

        print(f"    Best feature: {best_feat} (d={best_d:.4f}){marker}")
        level_results[level_name] = {'best_feat': best_feat, 'best_d': best_d,
                                      'n': n_level, 'rev_rate': rev_rate}

    print(f"\n  Signal found (|d| >= 0.3 for any level): {'YES' if signal_found else 'NO'}")
    return level_results, signal_found


# ── Phase 2D: Temporal Regime Analysis ───────────────────────────────────

def run_regime_analysis(df, feature_cols, outcome_col='outcome'):
    """
    Test if features are more predictive in certain market regimes.
    """
    print("\n" + "="*80)
    print("PHASE 2D: TEMPORAL REGIME ANALYSIS")
    print("="*80)

    mask = df[outcome_col].isin([0, 1])
    df_rb = df[mask].copy()

    signal_found = False
    regime_results = {}

    # 1. Volatility regime (daily BB width terciles)
    if 'daily_bb_width' in df_rb.columns:
        print("\n  Volatility regimes (daily BB width terciles):")
        try:
            df_rb['vol_regime'] = pd.qcut(
                df_rb['daily_bb_width'], q=3, labels=['low_vol', 'mid_vol', 'high_vol'],
                duplicates='drop'
            )
            for regime in ['low_vol', 'mid_vol', 'high_vol']:
                regime_mask = df_rb['vol_regime'] == regime
                if regime_mask.sum() < 200:
                    continue

                rev_mask = (df_rb[outcome_col] == 1) & regime_mask
                bo_mask = (df_rb[outcome_col] == 0) & regime_mask

                best_d = 0
                best_feat = ""
                for feat in feature_cols[:30]:  # Check top 30 to save time
                    if feat not in df_rb.columns:
                        continue
                    rev_vals = df_rb.loc[rev_mask, feat].values.astype(np.float64)
                    bo_vals = df_rb.loc[bo_mask, feat].values.astype(np.float64)
                    rev_vals = rev_vals[np.isfinite(rev_vals)]
                    bo_vals = bo_vals[np.isfinite(bo_vals)]
                    d = cohens_d(rev_vals, bo_vals)
                    if abs(d) > abs(best_d):
                        best_d = d
                        best_feat = feat

                marker = ""
                if abs(best_d) >= 0.3:
                    marker = " *** SIGNAL ***"
                    signal_found = True

                rev_rate = rev_mask.sum() / max(rev_mask.sum() + bo_mask.sum(), 1)
                print(f"    {regime}: {regime_mask.sum():,} bars, rev_rate={rev_rate:.3f}, "
                      f"best d={best_d:.4f} ({best_feat}){marker}")
                regime_results[regime] = {'best_d': best_d, 'best_feat': best_feat,
                                           'rev_rate': rev_rate}
        except Exception as e:
            print(f"    Skipped: {e}")

    # 2. Time of day regime
    if 'bars_since_rth_open' in df_rb.columns:
        print("\n  Time-of-day regimes:")
        bso = df_rb['bars_since_rth_open']
        for lo, hi, label in [(0, 60, "First hour"),
                               (60, 240, "Mid-day"),
                               (240, 999, "Last hour")]:
            regime_mask = (bso >= lo) & (bso < hi)
            if regime_mask.sum() < 200:
                continue

            rev_mask = (df_rb[outcome_col] == 1) & regime_mask
            bo_mask = (df_rb[outcome_col] == 0) & regime_mask

            best_d = 0
            best_feat = ""
            for feat in feature_cols[:30]:
                if feat not in df_rb.columns:
                    continue
                rev_vals = df_rb.loc[rev_mask, feat].values.astype(np.float64)
                bo_vals = df_rb.loc[bo_mask, feat].values.astype(np.float64)
                rev_vals = rev_vals[np.isfinite(rev_vals)]
                bo_vals = bo_vals[np.isfinite(bo_vals)]
                d = cohens_d(rev_vals, bo_vals)
                if abs(d) > abs(best_d):
                    best_d = d
                    best_feat = feat

            marker = ""
            if abs(best_d) >= 0.3:
                marker = " *** SIGNAL ***"
                signal_found = True

            rev_rate = rev_mask.sum() / max(rev_mask.sum() + bo_mask.sum(), 1)
            print(f"    {label}: {regime_mask.sum():,} bars, rev_rate={rev_rate:.3f}, "
                  f"best d={best_d:.4f} ({best_feat}){marker}")

    print(f"\n  Signal found (|d| >= 0.3 in any regime): {'YES' if signal_found else 'NO'}")
    return regime_results, signal_found


# ── Plotting ──────────────────────────────────────────────────────────────

def save_plots(results_df, conditions, df, outcome_col='outcome'):
    """Save analysis figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 1. Cohen's d bar chart (top 30)
    fig, ax = plt.subplots(figsize=(12, 8))
    top30 = results_df.head(30)
    colors = ['#e74c3c' if abs(d) >= 0.2 else '#f39c12' if abs(d) >= 0.1 else '#3498db'
              for d in top30['cohens_d']]
    ax.barh(range(len(top30)), top30['abs_d'].values, color=colors)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['feature'].values, fontsize=8)
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Signal threshold (0.2)')
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Weak signal (0.1)')
    ax.set_xlabel('|Cohen\'s d|')
    ax.set_title('Feature Effect Sizes: Reversal vs Breakout')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'cohens_d_top30.png'), dpi=150)
    plt.close()

    # 2. Conditional reversal rates
    if conditions:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels_list = [c[0] for c in conditions]
        rates = [c[1] for c in conditions]
        mask_rb = df[outcome_col].isin([0, 1])
        base_rate = (df.loc[mask_rb, outcome_col] == 1).mean()
        colors = ['#e74c3c' if abs(r - base_rate) >= 0.05 else '#3498db' for r in rates]
        ax.bar(range(len(labels_list)), rates, color=colors)
        ax.axhline(y=base_rate, color='black', linestyle='--', alpha=0.7, label=f'Base rate ({base_rate:.3f})')
        ax.set_xticks(range(len(labels_list)))
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Reversal Rate')
        ax.set_title('Conditional Reversal Rates')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'conditional_reversal_rates.png'), dpi=150)
        plt.close()

    # 3. MI vs Cohen's d scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['abs_d'], results_df['mi'], alpha=0.5, s=20)
    for _, row in results_df.head(10).iterrows():
        ax.annotate(row['feature'], (row['abs_d'], row['mi']), fontsize=6)
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('|Cohen\'s d|')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('MI vs Effect Size')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'mi_vs_cohens_d.png'), dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Signal detection for reversal vs breakout")
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()

    t0 = time.time()

    # Load and prepare data
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)
    ohlcv, feature_cols = compute_all_features(ohlcv)
    ohlcv = label_reversals_breakouts(ohlcv)

    # Compute confluence features (need nearest_level info from labeling)
    ohlcv, confluence_cols = compute_confluence_features(ohlcv)
    feature_cols.extend(confluence_cols)
    feature_cols = list(dict.fromkeys(feature_cols))

    # Run all analyses
    results_df, signal_2a = run_univariate_analysis(ohlcv, feature_cols)
    conditions, signal_2b = run_conditional_analysis(ohlcv)
    level_results, signal_2c = run_per_level_analysis(ohlcv, feature_cols)
    regime_results, signal_2d = run_regime_analysis(ohlcv, feature_cols)

    # Save plots
    if not args.skip_plots:
        save_plots(results_df, conditions, ohlcv)

    # ── GO/NO-GO Verdict ──
    print("\n" + "="*80)
    print("GO/NO-GO VERDICT")
    print("="*80)

    any_signal = signal_2a or signal_2b or signal_2c or signal_2d

    tests = [
        ("2A: Univariate (|d| >= 0.2)", signal_2a),
        ("2B: Conditional (rate Δ >= 5pp)", signal_2b),
        ("2C: Per-level-type (|d| >= 0.3)", signal_2c),
        ("2D: Regime-conditional (|d| >= 0.3)", signal_2d),
    ]

    for name, passed in tests:
        status = "SIGNAL" if passed else "no signal"
        print(f"  {name}: {status}")

    print(f"\n  Overall verdict: {'GO — proceed to Phase 3' if any_signal else 'NO-GO — no actionable signal found'}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")

    if not any_signal:
        print("\n  Recommended next steps:")
        print("    1. Try different timeframe (5-min, 15-min bars)")
        print("    2. Try different question (level quality scoring)")
        print("    3. Try different data (order book, tick-by-tick)")

    return any_signal


if __name__ == '__main__':
    main()
