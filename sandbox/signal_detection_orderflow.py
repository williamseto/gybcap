#!/usr/bin/env python
"""
Phase 2: Signal Detection for Order Flow Features.

Tests whether per-second order flow features (aggregated to 1-min) carry
discriminative signal for reversal vs breakout outcomes at price levels.

Follows the exact pattern of sandbox/signal_detection.py.

Analyses:
  2A: Univariate — Cohen's d, KS test, MI for each ~32 order flow features
  2B: Conditional — Order flow + existing feature interactions
  2C: Per-level-type — Signal split by level type (ovn_hi, ovn_lo, vwap, etc.)
  2D: Temporal regime — Signal by time of day, volatility
  2E: Redundancy check — Correlation with existing 107 features

GO/NO-GO Criteria:
  - ANY feature |d| >= 0.2 across all levels → GO
  - ANY feature |d| >= 0.3 at a specific level type → GO
  - ANY conditional reversal rate delta >= 5pp → GO
  - None of the above → NO-GO (stop)

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/signal_detection_orderflow.py
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

FIGURE_DIR = "sandbox/figures/signal_detection_orderflow"
DATA_PATH = "raw_data/es_min_3y_clean_td_gamma.csv"

from strategies.labeling.reversal_zones import TRACKED_LEVELS
from strategies.data.sec_features import (
    load_cached_features,
    ORDERFLOW_FEATURE_COLS,
)


# ── Data Loading ─────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading 1-min data from {path}...")
    df = pd.read_csv(path)
    if "Date" in df.columns and "Time" in df.columns:
        df["dt"] = pd.to_datetime(
            df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M:%S"
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from strategies.features.price_levels import PriceLevelProvider

    print("\nComputing price levels...")
    plp = PriceLevelProvider(include_gamma="gamma_score" in ohlcv.columns)
    feat_df = plp._compute_impl(ohlcv)

    level_cols = ["vwap", "ovn_lo", "ovn_hi", "rth_lo", "rth_hi", "ib_lo", "ib_hi"]
    if "dt" in feat_df.columns:
        feat_df = feat_df.set_index("dt")
    ohlcv_dt = ohlcv.set_index("dt") if "dt" in ohlcv.columns else ohlcv
    for col in level_cols:
        if col in feat_df.columns:
            ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

    levels = plp.prev_day_levels(ohlcv)
    ohlcv["prev_high"] = ohlcv["trading_day"].map(levels["prev_high"])
    ohlcv["prev_low"] = ohlcv["trading_day"].map(levels["prev_low"])
    return ohlcv


def compute_all_features(ohlcv: pd.DataFrame):
    """Compute existing feature providers (for redundancy check)."""
    feature_cols = []

    print("\nComputing existing features (for redundancy check)...")
    from strategies.features.higher_timeframe import HigherTimeframeProvider

    htf = HigherTimeframeProvider()
    htf_df = htf._compute_impl(ohlcv)
    for col in htf.feature_names:
        if col in htf_df.columns:
            ohlcv[col] = htf_df[col].values
            feature_cols.append(col)

    from strategies.features.volume_microstructure import VolumeMicrostructureProvider

    has_bidask = "bidvolume" in ohlcv.columns
    vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
    vol_df = vmp._compute_impl(ohlcv)
    for col in vmp.feature_names:
        if col in vol_df.columns:
            ohlcv[col] = vol_df[col].values
            feature_cols.append(col)

    from strategies.features.reversion_quality import ReversionQualityProvider

    rqp = ReversionQualityProvider()
    qual_df = rqp._compute_impl(ohlcv)
    for col in rqp.feature_names:
        if col in qual_df.columns:
            ohlcv[col] = qual_df[col].values
            feature_cols.append(col)

    from strategies.features.temporal_interactions import TemporalInteractionProvider

    tip = TemporalInteractionProvider()
    temp_df = tip._compute_impl(ohlcv)
    for col in tip.feature_names:
        if col in temp_df.columns:
            ohlcv[col] = temp_df[col].values
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"  {len(feature_cols)} existing features computed")
    return ohlcv, feature_cols


def label_reversals_breakouts(
    ohlcv: pd.DataFrame,
    proximity_pts: float = 5.0,
    forward_bars: int = 45,
    reversal_pts: float = 6.0,
    breakout_pts: float = 4.0,
) -> pd.DataFrame:
    """Label near-level bars as reversal (1), breakout (0), or inconclusive (2)."""
    print(
        f"\nLabeling reversals/breakouts (proximity={proximity_pts}pt, "
        f"reversal={reversal_pts}pt, breakout={breakout_pts}pt)..."
    )

    close_arr = ohlcv["close"].values.astype(np.float64)
    high_arr = ohlcv["high"].values.astype(np.float64)
    low_arr = ohlcv["low"].values.astype(np.float64)
    n = len(ohlcv)

    labels = np.full(n, -1, dtype=np.int32)
    nearest_level_name = np.empty(n, dtype=object)
    nearest_level_price = np.full(n, np.nan)
    side_arr = np.zeros(n, dtype=np.int8)

    level_arrs = {}
    for lvl_name in TRACKED_LEVELS:
        if lvl_name in ohlcv.columns:
            level_arrs[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

    for lvl_name, lvl_vals in level_arrs.items():
        for i in range(n):
            d = abs(close_arr[i] - lvl_vals[i])
            if np.isnan(d):
                continue
            if d <= proximity_pts:
                prev_dist = (
                    abs(close_arr[i] - nearest_level_price[i])
                    if not np.isnan(nearest_level_price[i])
                    else np.inf
                )
                if d < prev_dist:
                    nearest_level_name[i] = lvl_name
                    nearest_level_price[i] = lvl_vals[i]

    near_level_mask = ~np.isnan(nearest_level_price)
    n_near = near_level_mask.sum()
    print(f"  {n_near:,} near-level bars out of {n:,} ({100 * n_near / n:.1f}%)")

    for i in range(n):
        if not near_level_mask[i]:
            continue

        level = nearest_level_price[i]
        price = close_arr[i]
        is_above = price >= level
        side_arr[i] = 1 if is_above else -1

        end = min(i + 1 + forward_bars, n)
        future_highs = high_arr[i + 1 : end]
        future_lows = low_arr[i + 1 : end]

        if len(future_highs) == 0:
            labels[i] = 2
            continue

        if is_above:
            rev_hits = np.where(future_highs >= price + reversal_pts)[0]
            brk_hits = np.where(future_lows <= level - breakout_pts)[0]
        else:
            rev_hits = np.where(future_lows <= price - reversal_pts)[0]
            brk_hits = np.where(future_highs >= level + breakout_pts)[0]

        first_rev = rev_hits[0] if len(rev_hits) > 0 else 9999
        first_brk = brk_hits[0] if len(brk_hits) > 0 else 9999

        if first_rev < first_brk and first_rev < 9999:
            labels[i] = 1
        elif first_brk < first_rev and first_brk < 9999:
            labels[i] = 0
        else:
            labels[i] = 2

    ohlcv["outcome"] = labels
    ohlcv["nearest_level_name"] = nearest_level_name
    ohlcv["nearest_level_price"] = nearest_level_price
    ohlcv["side"] = side_arr

    n_rev = (labels == 1).sum()
    n_bo = (labels == 0).sum()
    n_inc = (labels == 2).sum()
    print(f"  Reversal: {n_rev:,} ({100 * n_rev / max(n_near, 1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100 * n_bo / max(n_near, 1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100 * n_inc / max(n_near, 1):.1f}%)")

    return ohlcv


# ── Merge Order Flow Features ─────────────────────────────────────────

def merge_orderflow_features(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Load cached order flow features and merge with 1-min data on timestamp."""
    print("\nLoading cached order flow features...")
    of_df = load_cached_features()

    # Align: of_df has 'minute' column, ohlcv has 'dt' column
    of_df = of_df.set_index("minute")
    ohlcv_dt = ohlcv["dt"]

    # Left join: keep all 1-min bars, add OF features where available
    of_cols = [c for c in ORDERFLOW_FEATURE_COLS if c in of_df.columns]

    print(f"  Order flow feature columns: {len(of_cols)}")
    print(f"  Order flow minutes available: {len(of_df):,}")

    # Create a lookup from the OF features
    for col in of_cols:
        ohlcv[col] = of_df[col].reindex(ohlcv_dt.values).values

    # Count matches
    n_matched = ohlcv[of_cols[0]].notna().sum()
    print(f"  Matched: {n_matched:,} / {len(ohlcv):,} bars ({100 * n_matched / len(ohlcv):.1f}%)")

    return ohlcv, of_cols


# ── Statistical Tests ─────────────────────────────────────────────────

def cohens_d(group_a, group_b):
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
    if len(group_a) < 5 or len(group_b) < 5:
        return 1.0
    _, p = sp_stats.ks_2samp(group_a, group_b)
    return p


def mutual_info_binned(feature, binary_label, n_bins=10):
    if len(feature) < 20:
        return 0.0
    mask = np.isfinite(feature) & np.isfinite(binary_label.astype(float))
    feature = feature[mask]
    binary_label = binary_label[mask]
    if len(feature) < 20:
        return 0.0
    try:
        binned = pd.qcut(
            pd.Series(feature), q=n_bins, labels=False, duplicates="drop"
        ).values
        valid = ~np.isnan(binned)
        binned = binned[valid].astype(int)
        binary_label = binary_label[valid].astype(int)
        if len(binned) < 20:
            return 0.0
        return mutual_info_score(binned, binary_label)
    except (ValueError, TypeError):
        return 0.0


def conditional_reversal_rate(feature, outcome, n_bins=5):
    mask = np.isfinite(feature) & (outcome >= 0) & (outcome <= 1)
    feature = feature[mask]
    outcome = outcome[mask]
    if len(feature) < 50:
        return [], [], []
    try:
        bins = pd.qcut(
            pd.Series(feature), q=n_bins, labels=False, duplicates="drop"
        ).values
    except ValueError:
        return [], [], []

    unique_bins = sorted(np.unique(bins[~np.isnan(bins)]).astype(int))
    rates, counts, labels = [], [], []
    for b in unique_bins:
        mask_b = bins == b
        n = mask_b.sum()
        if n < 10:
            continue
        rates.append(outcome[mask_b].mean())
        counts.append(n)
        labels.append(f"Q{b + 1}")
    return labels, rates, counts


# ── Phase 2A: Univariate Analysis ─────────────────────────────────────

def run_univariate_analysis(df, feature_cols, outcome_col="outcome"):
    print("\n" + "=" * 80)
    print("PHASE 2A: UNIVARIATE ANALYSIS — ORDER FLOW FEATURES")
    print("=" * 80)

    mask = df[outcome_col].isin([0, 1])
    # Also require order flow features to be non-null
    of_available = df[feature_cols[0]].notna() if feature_cols else pd.Series(True, index=df.index)
    mask = mask & of_available

    df_rb = df[mask].copy()
    rev_mask = df_rb[outcome_col] == 1
    bo_mask = df_rb[outcome_col] == 0

    n_rev = rev_mask.sum()
    n_bo = bo_mask.sum()
    print(f"\nSample: {n_rev:,} reversals, {n_bo:,} breakouts (with OF data)")

    results = []
    n_features = len(feature_cols)
    bonferroni = 0.001 / max(n_features, 1)

    for feat in feature_cols:
        if feat not in df_rb.columns:
            continue
        vals = df_rb[feat].values.astype(np.float64)
        rev_vals = vals[rev_mask.values]
        bo_vals = vals[bo_mask.values]

        rev_vals = rev_vals[np.isfinite(rev_vals)]
        bo_vals = bo_vals[np.isfinite(bo_vals)]

        d = cohens_d(rev_vals, bo_vals)
        p = ks_test(rev_vals, bo_vals)
        mi = mutual_info_binned(vals, df_rb[outcome_col].values)

        labels, rates, counts = conditional_reversal_rate(
            vals, df_rb[outcome_col].values
        )
        rate_range = max(rates) - min(rates) if rates else 0.0

        results.append({
            "feature": feat,
            "cohens_d": d,
            "abs_d": abs(d),
            "ks_p": p,
            "ks_sig": p < bonferroni,
            "mi": mi,
            "rate_range": rate_range,
            "rev_mean": np.mean(rev_vals) if len(rev_vals) > 0 else 0,
            "bo_mean": np.mean(bo_vals) if len(bo_vals) > 0 else 0,
        })

    results_df = pd.DataFrame(results).sort_values("abs_d", ascending=False)

    print(f"\n{'Feature':<35} {'Cohen d':>8} {'KS p':>10} {'MI':>8} {'Rate Δ':>8}")
    print("-" * 75)

    signal_found = False
    for _, row in results_df.iterrows():
        marker = ""
        if abs(row["cohens_d"]) >= 0.2:
            marker = " *** SIGNAL ***"
            signal_found = True
        elif abs(row["cohens_d"]) >= 0.1:
            marker = " *"
        print(
            f"{row['feature']:<35} {row['cohens_d']:>8.4f} {row['ks_p']:>10.2e} "
            f"{row['mi']:>8.4f} {row['rate_range']:>8.4f}{marker}"
        )

    n_above_02 = (results_df["abs_d"] >= 0.2).sum()
    n_above_01 = (results_df["abs_d"] >= 0.1).sum()
    n_ks_sig = results_df["ks_sig"].sum()
    n_mi_above = (results_df["mi"] > 0.01).sum()

    print(f"\n  Features with |d| >= 0.2: {n_above_02}")
    print(f"  Features with |d| >= 0.1: {n_above_01}")
    print(f"  Features with KS p < {bonferroni:.2e} (Bonferroni): {n_ks_sig}")
    print(f"  Features with MI > 0.01: {n_mi_above}")

    return results_df, signal_found


# ── Phase 2B: Conditional Analysis ─────────────────────────────────────

def run_conditional_analysis(df, of_cols, outcome_col="outcome"):
    print("\n" + "=" * 80)
    print("PHASE 2B: CONDITIONAL ANALYSIS — ORDER FLOW INTERACTIONS")
    print("=" * 80)

    mask = df[outcome_col].isin([0, 1]) & df[of_cols[0]].notna()
    df_rb = df[mask].copy()
    rev_rate_all = (df_rb[outcome_col] == 1).mean()
    print(f"\nBase reversal rate: {rev_rate_all:.4f}")

    signal_found = False
    conditions = []

    # 1. High absorption + near level
    if "delta_absorption_score" in df_rb.columns:
        absorption = df_rb["delta_absorption_score"]
        for lo, hi, label in [
            (0, 5, "Low absorption (<5)"),
            (5, 15, "Medium absorption (5-15)"),
            (15, 999999, "High absorption (>15)"),
        ]:
            cond = (absorption >= lo) & (absorption < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 2. Delta imbalance extremes
    if "fp_delta_imbalance" in df_rb.columns:
        di = df_rb["fp_delta_imbalance"]
        for lo, hi, label in [
            (-1, -0.3, "Strong sell delta (<-0.3)"),
            (-0.3, -0.1, "Moderate sell delta"),
            (-0.1, 0.1, "Balanced delta"),
            (0.1, 0.3, "Moderate buy delta"),
            (0.3, 1, "Strong buy delta (>0.3)"),
        ]:
            cond = (di >= lo) & (di < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 3. Volume taper + reversal
    if "vol_taper_score" in df_rb.columns:
        taper = df_rb["vol_taper_score"]
        for lo, hi, label in [
            (0, 0.5, "Volume tapering off (<0.5)"),
            (0.5, 1.5, "Stable volume"),
            (1.5, 999, "Volume increasing (>1.5)"),
        ]:
            cond = (taper >= lo) & (taper < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 4. Delta flip within bar
    if "delta_flip_within_bar" in df_rb.columns:
        for val, label in [(0, "No delta flip"), (1, "Delta flipped within bar")]:
            cond = df_rb["delta_flip_within_bar"] == val
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 5. Volume climax + exhaustion
    if "approach_vol_climax_5bar" in df_rb.columns:
        for val, label in [(0, "No vol climax in 5 bars"), (1, "Vol climax in last 5 bars")]:
            cond = df_rb["approach_vol_climax_5bar"] == val
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 6. Approach exhaustion composite
    if "approach_exhaustion" in df_rb.columns:
        exh = df_rb["approach_exhaustion"]
        for lo, hi, label in [
            (0, 0.33, "No exhaustion (0)"),
            (0.33, 0.67, "Partial exhaustion"),
            (0.67, 1.01, "Full exhaustion (1.0)"),
        ]:
            cond = (exh >= lo) & (exh < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 7. Delta divergence (price up + delta down or vice versa)
    if "delta_divergence" in df_rb.columns:
        for val, label in [(-1, "Delta diverges from price"), (1, "Delta confirms price")]:
            cond = df_rb["delta_divergence"] == val
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # 8. POC position (where most volume traded in the bar)
    if "fp_poc_position" in df_rb.columns:
        poc = df_rb["fp_poc_position"]
        for lo, hi, label in [
            (0, 0.3, "POC at bar low (rejection from below)"),
            (0.3, 0.7, "POC mid-bar"),
            (0.7, 1.01, "POC at bar high (rejection from above)"),
        ]:
            cond = (poc >= lo) & (poc < hi)
            if cond.sum() < 50:
                continue
            rate = (df_rb.loc[cond, outcome_col] == 1).mean()
            conditions.append((label, rate, cond.sum()))

    # Print results
    print(
        f"\n{'Condition':<45} {'Rev Rate':>10} {'N':>8} {'Δ from base':>12}"
    )
    print("-" * 78)
    for label, rate, n in conditions:
        delta = rate - rev_rate_all
        marker = ""
        if abs(delta) >= 0.05:
            marker = " *** SIGNAL ***"
            signal_found = True
        elif abs(delta) >= 0.03:
            marker = " *"
        print(f"{label:<45} {rate:>10.4f} {n:>8,} {delta:>+12.4f}{marker}")

    if conditions:
        max_delta = max(abs(r - rev_rate_all) for _, r, _ in conditions)
        print(f"\n  Max reversal rate delta: {max_delta:.4f}")
        print(f"  Threshold: 0.05 (5pp)")
        print(f"  Signal found: {'YES' if signal_found else 'NO'}")

    return conditions, signal_found


# ── Phase 2C: Per-Level-Type Analysis ──────────────────────────────────

def run_per_level_analysis(df, of_cols, outcome_col="outcome"):
    print("\n" + "=" * 80)
    print("PHASE 2C: PER-LEVEL-TYPE ANALYSIS — ORDER FLOW")
    print("=" * 80)

    mask = df[outcome_col].isin([0, 1]) & df[of_cols[0]].notna()
    df_rb = df[mask].copy()

    signal_found = False
    level_results = {}

    for level_name in TRACKED_LEVELS:
        level_mask = df_rb["nearest_level_name"] == level_name
        n_level = level_mask.sum()
        if n_level < 100:
            continue

        rev_mask = (df_rb[outcome_col] == 1) & level_mask
        bo_mask = (df_rb[outcome_col] == 0) & level_mask
        n_rev = rev_mask.sum()
        n_bo = bo_mask.sum()
        rev_rate = n_rev / max(n_rev + n_bo, 1)

        print(
            f"\n  {level_name}: {n_level:,} bars "
            f"({n_rev:,} rev, {n_bo:,} bo, rate={rev_rate:.3f})"
        )

        # Find best order flow features for this level
        top_features = []
        for feat in of_cols:
            if feat not in df_rb.columns:
                continue
            rev_vals = df_rb.loc[rev_mask, feat].values.astype(np.float64)
            bo_vals = df_rb.loc[bo_mask, feat].values.astype(np.float64)
            rev_vals = rev_vals[np.isfinite(rev_vals)]
            bo_vals = bo_vals[np.isfinite(bo_vals)]
            d = cohens_d(rev_vals, bo_vals)
            top_features.append((feat, d))

        top_features.sort(key=lambda x: abs(x[1]), reverse=True)

        # Print top 5
        for feat, d in top_features[:5]:
            marker = ""
            if abs(d) >= 0.3:
                marker = " *** SIGNAL ***"
                signal_found = True
            elif abs(d) >= 0.2:
                marker = " *"
            print(f"    {feat:<35} d={d:>8.4f}{marker}")

        best_feat, best_d = top_features[0] if top_features else ("", 0.0)
        level_results[level_name] = {
            "best_feat": best_feat,
            "best_d": best_d,
            "n": n_level,
            "rev_rate": rev_rate,
            "top5": top_features[:5],
        }

    print(
        f"\n  Signal found (|d| >= 0.3 for any level): "
        f"{'YES' if signal_found else 'NO'}"
    )
    return level_results, signal_found


# ── Phase 2D: Temporal Regime Analysis ──────────────────────────────────

def run_regime_analysis(df, of_cols, outcome_col="outcome"):
    print("\n" + "=" * 80)
    print("PHASE 2D: TEMPORAL REGIME ANALYSIS — ORDER FLOW")
    print("=" * 80)

    mask = df[outcome_col].isin([0, 1]) & df[of_cols[0]].notna()
    df_rb = df[mask].copy()

    signal_found = False

    # 1. Volatility regime (ATR terciles)
    if "daily_bb_width" in df_rb.columns:
        print("\n  Volatility regimes (daily BB width terciles):")
        try:
            df_rb["vol_regime"] = pd.qcut(
                df_rb["daily_bb_width"],
                q=3,
                labels=["low_vol", "mid_vol", "high_vol"],
                duplicates="drop",
            )
            for regime in ["low_vol", "mid_vol", "high_vol"]:
                regime_mask = df_rb["vol_regime"] == regime
                if regime_mask.sum() < 200:
                    continue

                rev_mask = (df_rb[outcome_col] == 1) & regime_mask
                bo_mask = (df_rb[outcome_col] == 0) & regime_mask

                best_d, best_feat = 0.0, ""
                for feat in of_cols:
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
                print(
                    f"    {regime}: {regime_mask.sum():,} bars, "
                    f"rev_rate={rev_rate:.3f}, "
                    f"best d={best_d:.4f} ({best_feat}){marker}"
                )
        except Exception as e:
            print(f"    Skipped: {e}")

    # 2. Time of day
    if "bars_since_rth_open" in df_rb.columns:
        print("\n  Time-of-day regimes:")
        bso = df_rb["bars_since_rth_open"]
        for lo, hi, label in [
            (0, 60, "First hour"),
            (60, 240, "Mid-day"),
            (240, 999, "Last hour"),
        ]:
            regime_mask = (bso >= lo) & (bso < hi)
            if regime_mask.sum() < 200:
                continue

            rev_mask = (df_rb[outcome_col] == 1) & regime_mask
            bo_mask = (df_rb[outcome_col] == 0) & regime_mask

            best_d, best_feat = 0.0, ""
            for feat in of_cols:
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
            print(
                f"    {label}: {regime_mask.sum():,} bars, "
                f"rev_rate={rev_rate:.3f}, "
                f"best d={best_d:.4f} ({best_feat}){marker}"
            )

    print(
        f"\n  Signal found (|d| >= 0.3 in any regime): "
        f"{'YES' if signal_found else 'NO'}"
    )
    return signal_found


# ── Phase 2E: Redundancy Check ──────────────────────────────────────────

def run_redundancy_check(df, of_cols, existing_cols, outcome_col="outcome"):
    print("\n" + "=" * 80)
    print("PHASE 2E: REDUNDANCY CHECK — ORDER FLOW vs EXISTING FEATURES")
    print("=" * 80)

    mask = df[outcome_col].isin([0, 1]) & df[of_cols[0]].notna()
    df_rb = df[mask].copy()

    print(f"\n  Checking correlation of {len(of_cols)} OF features vs {len(existing_cols)} existing...")

    redundant = []
    high_corr = []

    for of_feat in of_cols:
        if of_feat not in df_rb.columns:
            continue
        of_vals = df_rb[of_feat].values.astype(np.float64)
        of_finite = np.isfinite(of_vals)

        max_corr = 0.0
        max_corr_feat = ""

        for ex_feat in existing_cols:
            if ex_feat not in df_rb.columns:
                continue
            ex_vals = df_rb[ex_feat].values.astype(np.float64)
            valid = of_finite & np.isfinite(ex_vals)
            if valid.sum() < 100:
                continue

            try:
                r = np.corrcoef(of_vals[valid], ex_vals[valid])[0, 1]
            except (ValueError, FloatingPointError):
                r = 0.0

            if abs(r) > abs(max_corr):
                max_corr = r
                max_corr_feat = ex_feat

        if abs(max_corr) >= 0.9:
            redundant.append((of_feat, max_corr_feat, max_corr))
        elif abs(max_corr) >= 0.7:
            high_corr.append((of_feat, max_corr_feat, max_corr))

    print(f"\n  Redundant (|r| >= 0.9): {len(redundant)}")
    for of_f, ex_f, r in redundant:
        print(f"    {of_f:<35} ↔ {ex_f:<35} r={r:.3f}")

    print(f"\n  High correlation (|r| >= 0.7): {len(high_corr)}")
    for of_f, ex_f, r in high_corr:
        print(f"    {of_f:<35} ↔ {ex_f:<35} r={r:.3f}")

    novel = len(of_cols) - len(redundant)
    print(f"\n  Novel features (|r| < 0.9): {novel}/{len(of_cols)}")

    return redundant, high_corr


# ── Plotting ──────────────────────────────────────────────────────────

def save_plots(results_df, conditions, level_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 1. Cohen's d bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [
        "#e74c3c" if abs(d) >= 0.2 else "#f39c12" if abs(d) >= 0.1 else "#3498db"
        for d in results_df["cohens_d"]
    ]
    ax.barh(range(len(results_df)), results_df["abs_d"].values, color=colors)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df["feature"].values, fontsize=8)
    ax.axvline(x=0.2, color="red", linestyle="--", alpha=0.7, label="Signal threshold (0.2)")
    ax.axvline(x=0.1, color="orange", linestyle="--", alpha=0.5, label="Weak signal (0.1)")
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Order Flow Feature Effect Sizes: Reversal vs Breakout")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "of_cohens_d.png"), dpi=150)
    plt.close()

    # 2. Conditional reversal rates
    if conditions:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels_list = [c[0] for c in conditions]
        rates = [c[1] for c in conditions]
        base_rate = np.mean(rates)
        colors = [
            "#e74c3c" if abs(r - base_rate) >= 0.05 else "#3498db" for r in rates
        ]
        ax.bar(range(len(labels_list)), rates, color=colors)
        ax.axhline(y=base_rate, color="black", linestyle="--", alpha=0.7)
        ax.set_xticks(range(len(labels_list)))
        ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Reversal Rate")
        ax.set_title("Order Flow Conditional Reversal Rates")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "of_conditional_rates.png"), dpi=150)
        plt.close()

    # 3. Per-level best d
    if level_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = list(level_results.keys())
        d_vals = [abs(level_results[l]["best_d"]) for l in levels]
        colors = [
            "#e74c3c" if d >= 0.3 else "#f39c12" if d >= 0.2 else "#3498db"
            for d in d_vals
        ]
        ax.bar(range(len(levels)), d_vals, color=colors)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=45, ha="right")
        ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Signal (0.3)")
        ax.axhline(y=0.2, color="orange", linestyle="--", alpha=0.5, label="Weak (0.2)")
        ax.set_ylabel("|Cohen's d|")
        ax.set_title("Best Order Flow Feature per Level Type")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "of_per_level_d.png"), dpi=150)
        plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signal detection for order flow features"
    )
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-redundancy", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 70)
    print("Phase 2: Signal Detection — Order Flow Features")
    print("=" * 70)

    # Load and prepare 1-min data
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)

    # Compute existing features (for redundancy check)
    if not args.skip_redundancy:
        ohlcv, existing_cols = compute_all_features(ohlcv)
    else:
        existing_cols = []

    # Label reversals/breakouts
    ohlcv = label_reversals_breakouts(ohlcv)

    # Merge order flow features
    ohlcv, of_cols = merge_orderflow_features(ohlcv)

    # Run all analyses
    results_df, signal_2a = run_univariate_analysis(ohlcv, of_cols)
    conditions, signal_2b = run_conditional_analysis(ohlcv, of_cols)
    level_results, signal_2c = run_per_level_analysis(ohlcv, of_cols)
    signal_2d = run_regime_analysis(ohlcv, of_cols)

    # Redundancy check
    if not args.skip_redundancy and existing_cols:
        run_redundancy_check(ohlcv, of_cols, existing_cols)

    # Save plots
    if not args.skip_plots:
        save_plots(results_df, conditions, level_results)

    # ── GO/NO-GO Verdict ──
    print("\n" + "=" * 80)
    print("GO/NO-GO VERDICT — ORDER FLOW FEATURES")
    print("=" * 80)

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

    print(
        f"\n  Overall verdict: "
        f"{'GO — proceed to Phase 3' if any_signal else 'NO-GO — no actionable signal found'}"
    )
    print(f"\n  Total time: {time.time() - t0:.0f}s")

    return any_signal


if __name__ == "__main__":
    main()
