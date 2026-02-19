#!/usr/bin/env python
"""
Phase 3: XGB Enhancement with Order Flow Features.

Extends train_level_models.py by adding per-second order flow features
(aggregated to 1-min) to the existing 107 + 14 level-encoding features.

Trains two models per fold:
  - Baseline: 107 base + 14 level-encoding = 121 features (existing Phase 3)
  - Enhanced: 121 + ~32 order flow features

Walk-forward CV with honest early stopping, same splits/hyperparameters.

Success Criteria:
  | Metric         | Phase 3 Baseline | Target |
  |----------------|-----------------|--------|
  | OOS AUC        | 0.631           | > 0.65 |
  | WR at t=0.50   | 55.4%           | > 57%  |
  | E[PnL/trade]   | +1.54pt         | > +1.8 |
  | OVN level WR   | ~68%            | > 70%  |

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_level_models_orderflow.py
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "raw_data/es_min_3y_clean_td_gamma.csv"
FIGURE_DIR = "sandbox/figures/level_models_orderflow"

from strategies.labeling.reversal_zones import TRACKED_LEVELS
from sandbox.explore_sec_data import load_cached_features, ORDERFLOW_FEATURE_COLS

STOP_PTS = 4.0
TARGET_PTS = 6.0
MAX_BARS = 45


# ── Data loading (reused from train_level_models.py) ──────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
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


def compute_all_features(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
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

    has_bidask = "bidvolume" in ohlcv.columns
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
    print(f"  Total base features: {len(feature_cols)}")
    return ohlcv, feature_cols


# ── Labeling ──────────────────────────────────────────────────────────

def label_reversals_breakouts(
    ohlcv: pd.DataFrame,
    proximity_pts: float = 5.0,
    forward_bars: int = 45,
    reversal_pts: float = 6.0,
    breakout_pts: float = 4.0,
) -> pd.DataFrame:
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
    trade_dir_arr = np.zeros(n, dtype=np.int8)

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
        trade_dir_arr[i] = 1 if is_above else -1

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
    ohlcv["trade_direction"] = trade_dir_arr

    n_rev = (labels == 1).sum()
    n_bo = (labels == 0).sum()
    n_inc = (labels == 2).sum()
    print(f"  Reversal: {n_rev:,} ({100 * n_rev / max(n_near, 1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100 * n_bo / max(n_near, 1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100 * n_inc / max(n_near, 1):.1f}%)")

    return ohlcv


# ── Level-encoding features (from train_level_models.py) ────────────

def compute_level_encoding_features(
    ohlcv: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    print("\nComputing level-encoding features...")
    new_cols = []
    n = len(ohlcv)
    nearest = ohlcv["nearest_level_name"].values

    for lvl in TRACKED_LEVELS:
        col = f"is_{lvl}"
        ohlcv[col] = (nearest == lvl).astype(np.float32)
        new_cols.append(col)

    ohlcv["level_is_support"] = (ohlcv["side"] == 1).astype(np.float32)
    new_cols.append("level_is_support")

    print("  Computing trailing reversal rates (20d, 50d)...")
    ohlcv["level_trailing_rev_rate_20d"] = np.nan
    ohlcv["level_trailing_rev_rate_50d"] = np.nan

    days = sorted(ohlcv["trading_day"].unique())
    near_mask = ohlcv["outcome"].isin([0, 1])
    near_df = ohlcv.loc[
        near_mask, ["trading_day", "nearest_level_name", "outcome"]
    ].copy()

    daily_stats = (
        near_df.groupby(["trading_day", "nearest_level_name"])
        .agg(n_rev=("outcome", lambda x: (x == 1).sum()), n_total=("outcome", "count"))
        .reset_index()
    )

    for lvl in TRACKED_LEVELS:
        lvl_stats = daily_stats[daily_stats["nearest_level_name"] == lvl].copy()
        lvl_stats = lvl_stats.set_index("trading_day").reindex(days).fillna(0)

        lvl_stats["rev_20d"] = lvl_stats["n_rev"].rolling(20, min_periods=1).sum()
        lvl_stats["total_20d"] = lvl_stats["n_total"].rolling(20, min_periods=1).sum()
        lvl_stats["rev_rate_20d"] = lvl_stats["rev_20d"] / lvl_stats[
            "total_20d"
        ].clip(lower=1)

        lvl_stats["rev_50d"] = lvl_stats["n_rev"].rolling(50, min_periods=1).sum()
        lvl_stats["total_50d"] = lvl_stats["n_total"].rolling(50, min_periods=1).sum()
        lvl_stats["rev_rate_50d"] = lvl_stats["rev_50d"] / lvl_stats[
            "total_50d"
        ].clip(lower=1)

        lvl_stats["rev_rate_20d"] = lvl_stats["rev_rate_20d"].shift(1)
        lvl_stats["rev_rate_50d"] = lvl_stats["rev_rate_50d"].shift(1)

        rate_20d_map = lvl_stats["rev_rate_20d"].to_dict()
        rate_50d_map = lvl_stats["rev_rate_50d"].to_dict()

        lvl_mask = ohlcv["nearest_level_name"] == lvl
        if lvl_mask.any():
            ohlcv.loc[lvl_mask, "level_trailing_rev_rate_20d"] = ohlcv.loc[
                lvl_mask, "trading_day"
            ].map(rate_20d_map)
            ohlcv.loc[lvl_mask, "level_trailing_rev_rate_50d"] = ohlcv.loc[
                lvl_mask, "trading_day"
            ].map(rate_50d_map)

    new_cols.extend(["level_trailing_rev_rate_20d", "level_trailing_rev_rate_50d"])

    if "daily_bb_pct_b" in ohlcv.columns:
        ohlcv["level_side_bb_interaction"] = (
            ohlcv["level_is_support"] * ohlcv["daily_bb_pct_b"]
        )
        new_cols.append("level_side_bb_interaction")

    close_vals = ohlcv["close"].values.astype(np.float64)
    lvl_price_vals = ohlcv["nearest_level_price"].values.astype(np.float64)
    approach_dir = np.zeros(n, dtype=np.float32)
    for i in range(5, n):
        if np.isnan(lvl_price_vals[i]):
            continue
        price_change = close_vals[i] - close_vals[i - 5]
        if close_vals[i] >= lvl_price_vals[i]:
            approach_dir[i] = -1.0 if price_change < 0 else 1.0
        else:
            approach_dir[i] = 1.0 if price_change > 0 else -1.0
    ohlcv["approach_dir_vs_level"] = approach_dir
    new_cols.append("approach_dir_vs_level")

    print(f"  Added {len(new_cols)} level-encoding features")
    return ohlcv, new_cols


# ── Order Flow Feature Merge ─────────────────────────────────────────

def merge_orderflow_features(
    ohlcv: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge cached per-second order flow features into 1-min data."""
    print("\nLoading cached order flow features...")
    of_df = load_cached_features()

    of_df = of_df.set_index("minute")
    ohlcv_dt = ohlcv["dt"]

    of_cols = [c for c in ORDERFLOW_FEATURE_COLS if c in of_df.columns]
    print(f"  Order flow feature columns: {len(of_cols)}")

    for col in of_cols:
        ohlcv[col] = of_df[col].reindex(ohlcv_dt.values).values

    n_matched = ohlcv[of_cols[0]].notna().sum()
    print(
        f"  Matched: {n_matched:,} / {len(ohlcv):,} bars "
        f"({100 * n_matched / len(ohlcv):.1f}%)"
    )

    return ohlcv, of_cols


# ── Walk-forward splits ──────────────────────────────────────────────

def get_fold_splits(
    days: List, n_folds: int, min_train_days: int
) -> List[Tuple[List, List]]:
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds
    splits = []
    for fold in range(n_folds):
        train_end_idx = min_train_days + fold * test_days_per_fold
        test_end_idx = train_end_idx + test_days_per_fold
        if fold == n_folds - 1:
            test_end_idx = n_days
        train_d = days[:train_end_idx]
        test_d = days[train_end_idx:test_end_idx]
        if len(test_d) > 0:
            splits.append((train_d, test_d))
    return splits


# ── Trading simulation ──────────────────────────────────────────────

def simulate_trades(
    ohlcv: pd.DataFrame,
    predicted_indices: np.ndarray,
    stop_pts: float = STOP_PTS,
    target_pts: float = TARGET_PTS,
    max_bars: int = MAX_BARS,
) -> Tuple[int, float, float, float, List[float]]:
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0, []

    close = ohlcv["close"].values.astype(np.float64)
    high = ohlcv["high"].values.astype(np.float64)
    low = ohlcv["low"].values.astype(np.float64)
    trade_dir = ohlcv["trade_direction"].values
    n = len(close)

    wins = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n:
            continue
        direction = trade_dir[idx]
        if direction == 0:
            continue

        entry_price = close[idx]
        trade_pnl = 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    break
            else:
                if high[j] >= entry_price + stop_pts:
                    trade_pnl = -stop_pts
                    break
                if low[j] <= entry_price - target_pts:
                    trade_pnl = target_pts
                    break

        pnl_list.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1

    n_trades = len(pnl_list)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0, []

    wr = wins / n_trades
    mean_pnl = np.mean(pnl_list)
    total_pnl = np.sum(pnl_list)
    return n_trades, wr, mean_pnl, total_pnl, pnl_list


# ── Walk-Forward XGBoost Training ────────────────────────────────────

def train_walk_forward(
    samples_df: pd.DataFrame,
    feature_cols: List[str],
    ohlcv: pd.DataFrame,
    n_folds: int = 5,
    min_train_days: int = 100,
    label: str = "Model",
) -> Dict:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    print(f"\n{'─' * 70}")
    print(f"WALK-FORWARD TRAINING: {label}")
    print(f"{'─' * 70}")

    days = sorted(samples_df["trading_day"].unique())
    splits = get_fold_splits(days, n_folds, min_train_days)

    y = (samples_df["outcome"] == 1).astype(int).values
    pos_rate = y.mean()
    print(f"Samples: {len(samples_df):,}, positive rate: {pos_rate:.2%}")
    print(f"Features: {len(feature_cols)}")
    print(f"Folds: {len(splits)}")

    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_test_indices = []
    all_feature_imp = {}

    for fold, (train_days, test_days) in enumerate(splits):
        print(
            f"\n  Fold {fold + 1}/{len(splits)}: "
            f"{len(train_days)} train days, {len(test_days)} test days"
        )

        train_mask = samples_df["trading_day"].isin(train_days)
        test_mask = samples_df["trading_day"].isin(test_days)

        X_train_full = (
            samples_df.loc[train_mask, feature_cols]
            .fillna(0)
            .values.astype(np.float32)
        )
        X_test = (
            samples_df.loc[test_mask, feature_cols]
            .fillna(0)
            .values.astype(np.float32)
        )
        y_train_full = y[train_mask.values]
        y_test = y[test_mask.values]
        test_indices = samples_df.index[test_mask].values

        if len(y_test) == 0 or y_train_full.sum() < 5:
            print("    Skipping fold (insufficient data)")
            continue

        # Honest early stopping: temporal 80/20 split
        n_train_days = len(train_days)
        val_split_idx = int(n_train_days * 0.8)
        val_days = set(train_days[val_split_idx:])
        train_day_values = samples_df.loc[train_mask, "trading_day"].values
        val_within_train = np.array([d in val_days for d in train_day_values])
        train_within_train = ~val_within_train

        X_tr = X_train_full[train_within_train]
        y_tr = y_train_full[train_within_train]
        X_val = X_train_full[val_within_train]
        y_val = y_train_full[val_within_train]

        if len(y_val) == 0 or y_tr.sum() < 5:
            X_tr = X_train_full
            y_tr = y_train_full
            X_val = X_test
            y_val = y_test

        spw = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            min_child_weight=5,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            early_stopping_rounds=30,
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        print(f"    AUC={auc:.3f}, best_iteration={model.best_iteration}")

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_test_indices.extend(test_indices.tolist())

        fold_results.append({
            "fold": fold,
            "train_days": len(train_days),
            "test_days": len(test_days),
            "train_samples": len(X_tr),
            "test_samples": len(X_test),
            "auc": auc,
            "n_positive": int(y_test.sum()),
            "n_total": len(y_test),
            "best_iteration": model.best_iteration,
        })

        imp = model.get_booster().get_score(importance_type="gain")
        for fname, score in imp.items():
            if fname.startswith("f"):
                fidx = int(fname[1:])
                if fidx < len(feature_cols):
                    real_name = feature_cols[fidx]
                    all_feature_imp[real_name] = (
                        all_feature_imp.get(real_name, 0) + score
                    )

    if all_feature_imp:
        max_imp = max(all_feature_imp.values())
        if max_imp > 0:
            all_feature_imp = {k: v / max_imp for k, v in all_feature_imp.items()}

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_test_indices = np.array(all_test_indices)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    print(f"\n  {label} Overall OOS AUC: {overall_auc:.4f}")

    return {
        "fold_results": fold_results,
        "feature_importance": all_feature_imp,
        "all_y_true": all_y_true,
        "all_y_prob": all_y_prob,
        "all_test_indices": all_test_indices,
        "overall_auc": overall_auc,
        "feature_cols": feature_cols,
        "label": label,
    }


# ── Threshold evaluation ─────────────────────────────────────────────

def evaluate_thresholds(
    train_result: Dict,
    ohlcv: pd.DataFrame,
    thresholds: List[float] = None,
) -> Dict:
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    label = train_result.get("label", "Model")
    print(f"\n  Threshold Analysis — {label}:")
    print(
        f"  {'Thresh':>7} {'N pred':>8} {'N trades':>9} {'WR':>7} "
        f"{'E[PnL]':>8} {'Total PnL':>10}"
    )
    print("  " + "-" * 55)

    all_y_prob = train_result["all_y_prob"]
    all_test_indices = train_result["all_test_indices"]
    all_y_true = train_result["all_y_true"]

    threshold_results = {}

    for thresh in thresholds:
        pred_mask = all_y_prob >= thresh
        pred_indices = all_test_indices[pred_mask]

        y_pred = pred_mask.astype(int)
        tp = ((y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((y_pred == 1) & (all_y_true == 0)).sum()
        precision = tp / max(tp + fp, 1)

        n_trades, wr, mean_pnl, total_pnl, pnl_list = simulate_trades(
            ohlcv, pred_indices
        )

        threshold_results[thresh] = {
            "n_predicted": int(pred_mask.sum()),
            "precision": precision,
            "n_trades": n_trades,
            "win_rate": wr,
            "mean_pnl": mean_pnl,
            "total_pnl": total_pnl,
            "pnl_list": pnl_list,
            "pred_indices": pred_indices,
        }

        print(
            f"  {thresh:>7.2f} {pred_mask.sum():>8,} {n_trades:>9,} "
            f"{wr:>7.1%} {mean_pnl:>8.2f} {total_pnl:>10.1f}"
        )

    return threshold_results


# ── Post-analysis ─────────────────────────────────────────────────────

def run_comparison(
    baseline_result: Dict,
    enhanced_result: Dict,
    baseline_thresholds: Dict,
    enhanced_thresholds: Dict,
    ohlcv: pd.DataFrame,
    samples_df: pd.DataFrame,
    of_cols: List[str],
    level_encoding_cols: List[str],
):
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON: BASELINE vs ENHANCED (+ ORDER FLOW)")
    print("=" * 70)

    # AUC comparison
    print(f"\n  Overall OOS AUC:")
    print(f"    Baseline:  {baseline_result['overall_auc']:.4f}")
    print(f"    Enhanced:  {enhanced_result['overall_auc']:.4f}")
    auc_delta = enhanced_result["overall_auc"] - baseline_result["overall_auc"]
    print(f"    Delta:     {auc_delta:+.4f}")

    # Per-fold AUC comparison
    print(f"\n  Per-Fold AUC:")
    print(f"  {'Fold':>4} {'Baseline':>10} {'Enhanced':>10} {'Delta':>8}")
    print("  " + "-" * 35)
    for b, e in zip(
        baseline_result["fold_results"], enhanced_result["fold_results"]
    ):
        d = e["auc"] - b["auc"]
        print(f"  {b['fold']+1:>4} {b['auc']:>10.4f} {e['auc']:>10.4f} {d:>+8.4f}")

    # Threshold comparison at t=0.50
    thresh = 0.50
    b_t = baseline_thresholds.get(thresh, {})
    e_t = enhanced_thresholds.get(thresh, {})
    if b_t and e_t:
        print(f"\n  At threshold {thresh}:")
        for metric, label in [
            ("win_rate", "WR"),
            ("mean_pnl", "E[PnL]"),
            ("total_pnl", "Total PnL"),
            ("n_trades", "N trades"),
        ]:
            bv = b_t.get(metric, 0)
            ev = e_t.get(metric, 0)
            fmt = ".1%" if metric == "win_rate" else "+.2f" if "pnl" in metric.lower() else ","
            if metric == "win_rate":
                print(f"    {label:<12} Baseline={bv:.1%}  Enhanced={ev:.1%}  Δ={ev-bv:+.1%}")
            elif metric == "n_trades":
                print(f"    {label:<12} Baseline={bv:,}  Enhanced={ev:,}")
            else:
                print(f"    {label:<12} Baseline={bv:+.2f}  Enhanced={ev:+.2f}  Δ={ev-bv:+.2f}")

    # Per-level WR comparison at best threshold
    best_thresh = 0.50
    print(f"\n  Per-Level WR at t={best_thresh}:")
    print(f"  {'Level':<15} {'Base WR':>8} {'Enh WR':>8} {'Delta':>8}")
    print("  " + "-" * 42)

    b_indices = baseline_thresholds[best_thresh]["pred_indices"]
    e_indices = enhanced_thresholds[best_thresh]["pred_indices"]

    for lvl in TRACKED_LEVELS:
        b_lvl_mask = ohlcv.loc[b_indices, "nearest_level_name"].values == lvl
        b_lvl = b_indices[b_lvl_mask]
        e_lvl_mask = ohlcv.loc[e_indices, "nearest_level_name"].values == lvl
        e_lvl = e_indices[e_lvl_mask]

        _, b_wr, _, _, _ = simulate_trades(ohlcv, b_lvl) if len(b_lvl) > 0 else (0, 0, 0, 0, [])
        _, e_wr, _, _, _ = simulate_trades(ohlcv, e_lvl) if len(e_lvl) > 0 else (0, 0, 0, 0, [])

        d_wr = e_wr - b_wr
        print(f"  {lvl:<15} {b_wr:>8.1%} {e_wr:>8.1%} {d_wr:>+8.1%}")

    # Feature importance — do OF features appear in top 20?
    fi = enhanced_result["feature_importance"]
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 25 Features (Enhanced Model):")
    print(f"  {'Rank':>4} {'Feature':<40} {'Imp':>8} {'Type':>12}")
    print("  " + "-" * 68)
    for i, (feat, imp) in enumerate(fi_sorted[:25]):
        if feat in of_cols:
            ftype = "ORDERFLOW"
        elif feat in level_encoding_cols:
            ftype = "LEVEL-ENC"
        else:
            ftype = "base"
        print(f"  {i+1:>4} {feat:<40} {imp:>8.4f} {ftype:>12}")

    of_in_top20 = [f for f, _ in fi_sorted[:20] if f in of_cols]
    print(f"\n  Order flow features in top 20: {of_in_top20 if of_in_top20 else 'NONE'}")


# ── Plotting ─────────────────────────────────────────────────────────

def save_figures(
    baseline_result: Dict,
    enhanced_result: Dict,
    baseline_thresholds: Dict,
    enhanced_thresholds: Dict,
    ohlcv: pd.DataFrame,
    of_cols: List[str],
    level_encoding_cols: List[str],
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 1. Per-fold AUC comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [fr["fold"] + 1 for fr in baseline_result["fold_results"]]
    b_aucs = [fr["auc"] for fr in baseline_result["fold_results"]]
    e_aucs = [fr["auc"] for fr in enhanced_result["fold_results"]]
    x = np.arange(len(folds))
    width = 0.35
    ax.bar(x - width / 2, b_aucs, width, label="Baseline", color="#95a5a6")
    ax.bar(x + width / 2, e_aucs, width, label="+ Order Flow", color="#e74c3c")
    ax.axhline(y=0.5, color="black", linestyle=":", alpha=0.5)
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Per-Fold AUC: Baseline vs Enhanced")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "per_fold_auc_comparison.png"), dpi=150)
    plt.close()

    # 2. Enhanced feature importance (highlighting OF features)
    fi = enhanced_result["feature_importance"]
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:25]

    fig, ax = plt.subplots(figsize=(12, 9))
    names = [f[0] for f in fi_sorted]
    values = [f[1] for f in fi_sorted]
    colors = [
        "#2ecc71" if n in of_cols else "#e74c3c" if n in level_encoding_cols else "#3498db"
        for n in names
    ]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Normalized Gain")
    ax.set_title("Enhanced Model Feature Importance")
    ax.invert_yaxis()
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Order Flow"),
        Patch(facecolor="#e74c3c", label="Level-encoding"),
        Patch(facecolor="#3498db", label="Base features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "feature_importance_enhanced.png"), dpi=150)
    plt.close()

    # 3. Threshold comparison
    thresholds_sorted = sorted(baseline_thresholds.keys())
    b_wrs = [baseline_thresholds[t]["win_rate"] for t in thresholds_sorted]
    e_wrs = [enhanced_thresholds[t]["win_rate"] for t in thresholds_sorted]
    b_pnls = [baseline_thresholds[t]["mean_pnl"] for t in thresholds_sorted]
    e_pnls = [enhanced_thresholds[t]["mean_pnl"] for t in thresholds_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(thresholds_sorted, b_wrs, "o-", label="Baseline", color="#95a5a6")
    ax1.plot(thresholds_sorted, e_wrs, "s-", label="+ Order Flow", color="#e74c3c")
    ax1.axhline(y=0.5, color="black", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("WR vs Threshold")
    ax1.legend()

    ax2.plot(thresholds_sorted, b_pnls, "o-", label="Baseline", color="#95a5a6")
    ax2.plot(thresholds_sorted, e_pnls, "s-", label="+ Order Flow", color="#e74c3c")
    ax2.axhline(y=0, color="black", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("E[PnL] (pts)")
    ax2.set_title("E[PnL] vs Threshold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "threshold_comparison.png"), dpi=150)
    plt.close()

    # 4. Equity curves
    best_thresh = 0.50
    b_pnl = baseline_thresholds[best_thresh]["pnl_list"]
    e_pnl = enhanced_thresholds[best_thresh]["pnl_list"]

    if b_pnl and e_pnl:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(np.cumsum(b_pnl), color="#95a5a6", linewidth=1.0, label="Baseline")
        ax.plot(np.cumsum(e_pnl), color="#e74c3c", linewidth=1.0, label="+ Order Flow")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Cumulative PnL (pts)")
        ax.set_title(f"Equity Curves (threshold={best_thresh})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, "equity_curves.png"), dpi=150)
        plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: XGB enhancement with order flow features"
    )
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=100)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 70)
    print("Phase 3: XGB Enhancement with Order Flow Features")
    print("=" * 70)

    # ── Load data and compute features ──
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)
    ohlcv, feature_cols = compute_all_features(ohlcv)

    # ── Label ──
    ohlcv = label_reversals_breakouts(ohlcv)

    # ── Level-encoding features ──
    ohlcv, level_encoding_cols = compute_level_encoding_features(ohlcv)

    # ── Merge order flow features ──
    ohlcv, of_cols = merge_orderflow_features(ohlcv)

    # ── Build feature sets ──
    baseline_cols = feature_cols + level_encoding_cols
    baseline_cols = list(dict.fromkeys(baseline_cols))

    enhanced_cols = baseline_cols + of_cols
    enhanced_cols = list(dict.fromkeys(enhanced_cols))

    print(f"\n  Baseline features: {len(baseline_cols)}")
    print(f"  Enhanced features: {len(enhanced_cols)} (+{len(of_cols)} order flow)")

    # ── Filter to samples ──
    samples_mask = ohlcv["outcome"].isin([0, 1])
    samples_df = ohlcv.loc[samples_mask].copy()
    print(
        f"\nTraining samples: {len(samples_df):,} "
        f"(reversal={(samples_df['outcome'] == 1).sum():,}, "
        f"breakout={(samples_df['outcome'] == 0).sum():,})"
    )

    # Check OF coverage in samples
    of_available = samples_df[of_cols[0]].notna().sum()
    print(f"  Samples with OF data: {of_available:,} ({100*of_available/len(samples_df):.1f}%)")

    # ── Train both models ──
    baseline_result = train_walk_forward(
        samples_df, baseline_cols, ohlcv,
        n_folds=args.n_folds, min_train_days=args.min_train_days,
        label="Baseline (no OF)",
    )

    enhanced_result = train_walk_forward(
        samples_df, enhanced_cols, ohlcv,
        n_folds=args.n_folds, min_train_days=args.min_train_days,
        label="Enhanced (+ OF)",
    )

    # ── Evaluate thresholds ──
    baseline_thresholds = evaluate_thresholds(baseline_result, ohlcv)
    enhanced_thresholds = evaluate_thresholds(enhanced_result, ohlcv)

    # ── Comparison ──
    run_comparison(
        baseline_result, enhanced_result,
        baseline_thresholds, enhanced_thresholds,
        ohlcv, samples_df, of_cols, level_encoding_cols,
    )

    # ── Figures ──
    if not args.skip_plots:
        save_figures(
            baseline_result, enhanced_result,
            baseline_thresholds, enhanced_thresholds,
            ohlcv, of_cols, level_encoding_cols,
        )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    b_auc = baseline_result["overall_auc"]
    e_auc = enhanced_result["overall_auc"]
    print(f"  Baseline AUC:  {b_auc:.4f}")
    print(f"  Enhanced AUC:  {e_auc:.4f}  (Δ={e_auc - b_auc:+.4f})")

    thresh = 0.50
    b_t = baseline_thresholds[thresh]
    e_t = enhanced_thresholds[thresh]
    print(f"\n  At threshold {thresh}:")
    print(f"    Baseline WR={b_t['win_rate']:.1%}, E[PnL]={b_t['mean_pnl']:+.2f}")
    print(f"    Enhanced WR={e_t['win_rate']:.1%}, E[PnL]={e_t['mean_pnl']:+.2f}")

    # ── Success Criteria ──
    print(f"\n── Success Criteria ──")
    auc_pass = e_auc > 0.65
    wr_pass = e_t["win_rate"] > 0.57
    pnl_pass = e_t["mean_pnl"] > 1.8
    print(f"  AUC > 0.65:      {'PASS' if auc_pass else 'FAIL'} ({e_auc:.4f})")
    print(f"  WR > 57%:        {'PASS' if wr_pass else 'FAIL'} ({e_t['win_rate']:.1%})")
    print(f"  E[PnL] > 1.8pt:  {'PASS' if pnl_pass else 'FAIL'} ({e_t['mean_pnl']:+.2f})")

    fi = enhanced_result["feature_importance"]
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    of_in_top20 = [f for f, _ in fi_sorted[:20] if f in of_cols]
    fi_pass = len(of_in_top20) > 0
    print(f"  OF in top 20:    {'PASS' if fi_pass else 'FAIL'} ({of_in_top20})")

    overall = auc_pass or wr_pass or pnl_pass
    print(
        f"\n  Overall: {'IMPROVEMENT — proceed to Phase 4' if overall else 'NO IMPROVEMENT — stop'}"
    )
    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
