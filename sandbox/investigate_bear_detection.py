"""BEAR detection investigation: 4 experiments to improve BEAR detection.

Experiments:
  1. Binary BEAR detector (collapse 3-class → binary)
  2. Forward drawdown labels (causal-ish labels that stick from day 1)
  3. Leading indicator features (yield curve, SKEW, credit stress)
  4. DD exit + model re-entry (mechanical exit, model-guided re-entry)

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_bear_detection.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_bear_detection.py --quick --folds 2
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)

from strategies.swing.config import SwingConfig, INSTRUMENTS
from strategies.swing.data_loader import load_instruments
from strategies.swing.daily_aggregator import DailyAggregator, align_daily
from strategies.swing.features.daily_technical import (
    compute_daily_technical, FEATURE_NAMES as TECH_FEATURES,
)
from strategies.swing.features.volume_profile_daily import (
    compute_vp_daily_features, FEATURE_NAMES as VP_FEATURES,
)
from strategies.swing.features.cross_instrument import (
    compute_cross_features, get_feature_names as get_cross_names,
)
from strategies.swing.features.macro_context import (
    compute_macro_context, FEATURE_NAMES as MACRO_FEATURES,
)
from strategies.swing.features.external_daily import (
    compute_external_features, LEADING_FEATURE_NAMES,
)
from strategies.swing.labeling.structural_regime import compute_labels
from strategies.swing.labeling.hmm_regime import (
    compute_hmm_features_walkforward, FEATURE_NAMES as HMM_FEATURES,
)

FIGURES_DIR = Path("sandbox/figures/bear_detection")


def parse_args():
    parser = argparse.ArgumentParser(description="BEAR detection investigation")
    parser.add_argument("--quick", action="store_true", help="Quick debug run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--es-only", action="store_true")
    parser.add_argument("--skip-exp3", action="store_true",
                        help="Skip experiment 3 (needs downloaded data)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Part 0: Data Pipeline (shared across experiments)
# ─────────────────────────────────────────────────────────────────────

def load_data_pipeline(args) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load data, compute features and labels. Returns (df, es_daily, feature_cols)."""
    config = SwingConfig(
        n_folds=args.folds if not args.quick else 2,
        min_train_days=args.min_train if not args.quick else 200,
    )

    print("=" * 60)
    print("PART 0: Loading data & computing features")
    print("=" * 60)

    symbols = ["ES"]
    if not args.es_only:
        symbols += config.correlation_instruments
    minute_data = load_instruments(symbols)
    if "ES" not in minute_data:
        print("ERROR: ES data not found")
        sys.exit(1)

    aggregator = DailyAggregator()
    daily_data = {}
    for sym, minute_df in minute_data.items():
        compute_vp = (sym == "ES")
        print(f"  Aggregating {sym}...")
        daily_data[sym] = aggregator.aggregate(minute_df, compute_vp=compute_vp)
        print(f"    {len(daily_data[sym])} trading days")

    if len(daily_data) > 1:
        daily_data = align_daily(daily_data, primary="ES")

    es_daily = daily_data["ES"]
    print(f"  ES: {len(es_daily)} days, {es_daily.index.min().date()} -- {es_daily.index.max().date()}")

    # Features
    print("  Computing features...")
    tech_feats = compute_daily_technical(es_daily)
    feature_cols = list(TECH_FEATURES)

    vp_feats = pd.DataFrame(index=es_daily.index)
    if "vp_poc_rel" in es_daily.columns:
        vp_feats = compute_vp_daily_features(es_daily)
        feature_cols += VP_FEATURES

    cross_feats = pd.DataFrame(index=es_daily.index)
    if not args.es_only and len(daily_data) > 1:
        other_dailys = [(sym, df) for sym, df in daily_data.items() if sym != "ES"]
        cross_feats = compute_cross_features(es_daily, other_dailys)
        cross_names = get_cross_names([sym for sym, _ in other_dailys])
        cross_names = [c for c in cross_names if c in cross_feats.columns]
        feature_cols += cross_names

    other_for_macro = [(sym, df) for sym, df in daily_data.items() if sym != "ES"] if len(daily_data) > 1 else None
    macro_feats = compute_macro_context(es_daily, other_for_macro)
    feature_cols += MACRO_FEATURES

    ext_feats, ext_names = compute_external_features(es_daily)
    feature_cols += ext_names

    all_feats = pd.concat([tech_feats, vp_feats, cross_feats, macro_feats, ext_feats], axis=1)
    all_feats = all_feats.reindex(es_daily.index).fillna(0)

    # Labels (3-class structural)
    labels = compute_labels(es_daily)
    df = all_feats.join(labels)

    print(f"  Total features: {len(feature_cols)}")

    # Label distribution
    struct = df["y_structural"]
    valid = struct[struct.isin([0, 1, 2])]
    dist = valid.value_counts(normalize=True).sort_index()
    names = {0: "BEAR", 1: "BALANCE", 2: "BULL"}
    for k, v in dist.items():
        print(f"    {names[k]}: {v:.1%} ({valid.value_counts()[k]} days)")

    return df, es_daily, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Shared: Walk-forward binary CV
# ─────────────────────────────────────────────────────────────────────

def walk_forward_binary_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_folds: int = 5,
    min_train_days: int = 500,
    scale_pos_weight: float = 5.0,
    hmm_fn=None,
    verbose: bool = True,
) -> dict:
    """Walk-forward CV for binary classification (0 vs 1).

    Returns dict with OOS arrays and per-fold metrics.
    """
    valid_mask = df[target_col].isin([0, 1])
    df_valid = df[valid_mask].copy()
    days = sorted(df_valid.index.unique())
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds

    if verbose:
        print(f"  Binary CV: {n_days} days, {n_folds} folds, "
              f"~{test_days_per_fold} test days/fold, spw={scale_pos_weight:.1f}")

    all_preds = []
    all_actuals = []
    all_probas = []
    fold_metrics = []

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_days_per_fold
        test_end = train_end + test_days_per_fold if fold < n_folds - 1 else n_days

        train_days_list = days[:train_end]
        test_days_list = days[train_end:test_end]

        train_mask = df_valid.index.isin(train_days_list)
        test_mask = df_valid.index.isin(test_days_list)

        df_fold = df_valid.copy()

        # HMM per fold
        hmm_cols = []
        if hmm_fn is not None:
            hmm_feats = hmm_fn(df_fold, train_end)
            hmm_cols = [c for c in hmm_feats.columns if c not in df_fold.columns]
            for c in hmm_cols:
                df_fold[c] = hmm_feats[c].reindex(df_fold.index).fillna(0)

        fold_features = feature_cols + hmm_cols

        X_train = df_fold.loc[train_mask, fold_features].fillna(0).values
        y_train = df_fold.loc[train_mask, target_col].values.astype(int)
        X_test = df_fold.loc[test_mask, fold_features].fillna(0).values
        y_test = df_fold.loc[test_mask, target_col].values.astype(int)

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        model = XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.05,
            n_estimators=500,
            max_depth=5,
            min_child_weight=10,
            gamma=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train, verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_metrics.append({
            "fold": fold, "auc": auc, "precision": prec,
            "recall": rec, "f1": f1,
            "train_days": len(train_days_list),
            "test_days": len(test_days_list),
        })

        all_preds.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        all_probas.extend(y_proba.tolist())

        # Feature importance (last fold)
        importance = pd.DataFrame({
            "feature": fold_features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        if verbose:
            print(f"    Fold {fold+1}: AUC={auc:.3f}, P={prec:.3f}, "
                  f"R={rec:.3f}, F1={f1:.3f}")

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_probas = np.array(all_probas)

    # Aggregate
    agg_auc = roc_auc_score(all_actuals, all_probas) if len(np.unique(all_actuals)) > 1 else 0.5
    agg_prec = precision_score(all_actuals, all_preds, zero_division=0)
    agg_rec = recall_score(all_actuals, all_preds, zero_division=0)
    agg_f1 = f1_score(all_actuals, all_preds, zero_division=0)

    return {
        "preds": all_preds,
        "actuals": all_actuals,
        "probas": all_probas,
        "fold_metrics": fold_metrics,
        "agg_auc": agg_auc,
        "agg_precision": agg_prec,
        "agg_recall": agg_rec,
        "agg_f1": agg_f1,
        "importance": importance,  # from last fold
    }


# ─────────────────────────────────────────────────────────────────────
# Shared: Transition analysis for binary models
# ─────────────────────────────────────────────────────────────────────

def analyze_binary_transition_lag(actuals: np.ndarray, probas: np.ndarray,
                                   threshold: float = 0.5) -> dict:
    """Measure detection lag at BULL→BEAR (0→1) transitions."""
    preds = (probas >= threshold).astype(int)
    transitions = []
    for i in range(1, len(actuals)):
        if actuals[i] == 1 and actuals[i - 1] == 0:
            # NOT-BEAR → BEAR transition
            lag = None
            for j in range(i, len(preds)):
                if preds[j] == 1:
                    lag = j - i
                    break
            day1_prob = probas[i] if i < len(probas) else 0
            day1_pred = preds[i] if i < len(preds) else 0
            transitions.append({
                "idx": i,
                "lag": lag if lag is not None else len(preds) - i,
                "day1_prob": day1_prob,
                "day1_correct": day1_pred == 1,
            })
    return {
        "n_transitions": len(transitions),
        "mean_lag": np.mean([t["lag"] for t in transitions]) if transitions else float("inf"),
        "median_lag": np.median([t["lag"] for t in transitions]) if transitions else float("inf"),
        "day1_recall": np.mean([t["day1_correct"] for t in transitions]) if transitions else 0,
        "day1_mean_prob": np.mean([t["day1_prob"] for t in transitions]) if transitions else 0,
        "transitions": transitions,
    }


# ─────────────────────────────────────────────────────────────────────
# Shared: Strategy metrics
# ─────────────────────────────────────────────────────────────────────

def compute_strategy_metrics(daily_ret: np.ndarray, position: np.ndarray, label: str) -> dict:
    """Compute Sharpe, MaxDD, etc. for a position series."""
    # Causal: position at day t earns return at day t+1
    strat_ret = np.zeros(len(daily_ret))
    strat_ret[1:] = position[:-1] * daily_ret[1:]

    cum = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    n_years = len(daily_ret) / 252
    total_ret = cum[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = np.std(strat_ret) * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
    }


# ─────────────────────────────────────────────────────────────────────
# Experiment 1: Binary BEAR Detector
# ─────────────────────────────────────────────────────────────────────

def run_experiment_1(df: pd.DataFrame, es_daily: pd.DataFrame,
                     feature_cols: list[str], args) -> dict:
    """Binary BEAR detector: BULL+BALANCE→0, BEAR→1."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: Binary BEAR Detector")
    print("=" * 60)

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200

    # Create binary target
    df = df.copy()
    struct = df["y_structural"]
    df["y_bear_binary"] = -1  # invalid
    df.loc[struct.isin([1, 2]), "y_bear_binary"] = 0  # NOT-BEAR
    df.loc[struct == 0, "y_bear_binary"] = 1  # BEAR

    valid = df["y_bear_binary"].isin([0, 1])
    bear_rate = (df.loc[valid, "y_bear_binary"] == 1).mean()
    natural_spw = (1 - bear_rate) / bear_rate
    print(f"  BEAR rate: {bear_rate:.1%}, natural scale_pos_weight: {natural_spw:.1f}")

    # HMM callback
    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(es_daily, train_end_idx)

    # Sweep scale_pos_weight
    spw_values = [2, 3, 4, 5, 6, 8] if not args.quick else [3, 5]
    results = {}
    for spw in spw_values:
        print(f"\n  scale_pos_weight={spw}:")
        res = walk_forward_binary_cv(
            df, feature_cols, "y_bear_binary",
            n_folds=n_folds, min_train_days=min_train,
            scale_pos_weight=spw, hmm_fn=hmm_fn,
        )
        # Transition lag analysis
        trans = analyze_binary_transition_lag(res["actuals"], res["probas"])
        res["transition"] = trans
        results[spw] = res

        print(f"    Aggregate: AUC={res['agg_auc']:.3f}, P={res['agg_precision']:.3f}, "
              f"R={res['agg_recall']:.3f}, F1={res['agg_f1']:.3f}")
        print(f"    Day-1 BEAR recall: {trans['day1_recall']:.1%}, "
              f"mean lag: {trans['mean_lag']:.1f}d, "
              f"mean P(BEAR) at transition: {trans['day1_mean_prob']:.3f}")

    # Summary
    print(f"\n  {'SPW':>5s} {'AUC':>6s} {'Prec':>6s} {'Recall':>6s} "
          f"{'F1':>6s} {'Day1-R':>7s} {'Lag':>6s}")
    print("  " + "-" * 50)
    for spw, res in results.items():
        t = res["transition"]
        print(f"  {spw:>5.0f} {res['agg_auc']:>6.3f} {res['agg_precision']:>6.3f} "
              f"{res['agg_recall']:>6.3f} {res['agg_f1']:>6.3f} "
              f"{t['day1_recall']:>6.1%} {t['mean_lag']:>6.1f}")

    # Success check
    best_day1 = max(res["transition"]["day1_recall"] for res in results.values())
    print(f"\n  Success criterion: Day-1 BEAR recall > 25%")
    print(f"  Best day-1 recall: {best_day1:.1%} → {'PASS' if best_day1 > 0.25 else 'FAIL'}")

    # Top features from best AUC model
    best_spw = max(results, key=lambda k: results[k]["agg_auc"])
    print(f"\n  Top 15 features (spw={best_spw}):")
    for _, row in results[best_spw]["importance"].head(15).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Experiment 2: Forward Drawdown Labels
# ─────────────────────────────────────────────────────────────────────

def compute_forward_drawdown_labels(daily: pd.DataFrame, bear_threshold: float = 0.03) -> pd.Series:
    """Label each day by forward max drawdown before price recovers to a new high.

    For day t: look forward until close[t+k] > close[t] (recovery to new high).
    The max drawdown in close[t:t+k] is the forward MDD.
    If forward MDD > bear_threshold → BEAR (1), else NOT-BEAR (0).

    This is still non-causal but the label sticks from day 1 of any decline.
    """
    close = daily["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for t in range(n):
        ref_price = close[t]
        min_price = ref_price
        recovered = False

        for k in range(t + 1, n):
            min_price = min(min_price, close[k])
            if close[k] > ref_price:
                recovered = True
                break

        # Max drawdown from this point
        mdd = (min_price - ref_price) / ref_price
        if mdd < -bear_threshold:
            labels[t] = 1  # BEAR

    return pd.Series(labels, index=daily.index, name="y_fwd_dd")


def run_experiment_2(df: pd.DataFrame, es_daily: pd.DataFrame,
                     feature_cols: list[str], args) -> dict:
    """Forward drawdown labels experiment."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Forward Drawdown Labels")
    print("=" * 60)

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200

    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(es_daily, train_end_idx)

    thresholds = [0.02, 0.03, 0.05, 0.07] if not args.quick else [0.03, 0.05]
    results = {}

    for thresh in thresholds:
        print(f"\n  bear_threshold={thresh:.0%}:")
        fwd_labels = compute_forward_drawdown_labels(es_daily, bear_threshold=thresh)
        df_exp = df.copy()
        df_exp["y_fwd_dd"] = fwd_labels.reindex(df_exp.index).fillna(0).astype(int)

        bear_rate = df_exp["y_fwd_dd"].mean()
        natural_spw = (1 - bear_rate) / bear_rate if bear_rate > 0 else 5.0
        print(f"    BEAR rate: {bear_rate:.1%}, natural spw: {natural_spw:.1f}")

        # Use natural scale_pos_weight (capped at 8)
        spw = min(natural_spw, 8.0)

        res = walk_forward_binary_cv(
            df_exp, feature_cols, "y_fwd_dd",
            n_folds=n_folds, min_train_days=min_train,
            scale_pos_weight=spw, hmm_fn=hmm_fn,
        )

        # Transition lag
        trans = analyze_binary_transition_lag(res["actuals"], res["probas"])
        res["transition"] = trans
        res["bear_rate"] = bear_rate
        results[thresh] = res

        print(f"    Aggregate: AUC={res['agg_auc']:.3f}, P={res['agg_precision']:.3f}, "
              f"R={res['agg_recall']:.3f}, F1={res['agg_f1']:.3f}")
        print(f"    Day-1 recall: {trans['day1_recall']:.1%}, "
              f"mean lag: {trans['mean_lag']:.1f}d")

        # Precision at recall=50% (sweep thresholds)
        probas = res["probas"]
        actuals = res["actuals"]
        best_prec_at_50 = 0
        for p_thresh in np.arange(0.1, 0.9, 0.05):
            preds_t = (probas >= p_thresh).astype(int)
            rec_t = recall_score(actuals, preds_t, zero_division=0)
            prec_t = precision_score(actuals, preds_t, zero_division=0)
            if rec_t >= 0.50:
                best_prec_at_50 = max(best_prec_at_50, prec_t)
        res["precision_at_recall_50"] = best_prec_at_50
        print(f"    Precision at recall>=50%: {best_prec_at_50:.3f}")

    # Summary
    print(f"\n  {'Thresh':>7s} {'BearRate':>9s} {'AUC':>6s} {'Prec':>6s} "
          f"{'Recall':>6s} {'P@R50':>6s} {'Day1-R':>7s} {'Lag':>6s}")
    print("  " + "-" * 60)
    for thresh, res in results.items():
        t = res["transition"]
        print(f"  {thresh:>6.0%} {res['bear_rate']:>8.1%} {res['agg_auc']:>6.3f} "
              f"{res['agg_precision']:>6.3f} {res['agg_recall']:>6.3f} "
              f"{res['precision_at_recall_50']:>6.3f} "
              f"{t['day1_recall']:>6.1%} {t['mean_lag']:>6.1f}")

    # Success check
    for thresh, res in results.items():
        if res["agg_auc"] > 0.65 and res["precision_at_recall_50"] > 0.30:
            print(f"\n  SUCCESS at thresh={thresh:.0%}: AUC={res['agg_auc']:.3f} > 0.65 "
                  f"AND P@R50={res['precision_at_recall_50']:.3f} > 0.30")
            break
    else:
        print(f"\n  No threshold met success criteria (AUC>0.65 AND P@R50>0.30)")

    return results


# ─────────────────────────────────────────────────────────────────────
# Experiment 3: External Leading Indicators
# ─────────────────────────────────────────────────────────────────────

def run_experiment_3(df: pd.DataFrame, es_daily: pd.DataFrame,
                     feature_cols: list[str], args) -> dict:
    """Test whether leading indicator features improve BEAR detection."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: External Leading Indicators")
    print("=" * 60)

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200

    # Check which leading features are present
    leading_present = [f for f in LEADING_FEATURE_NAMES if f in df.columns and df[f].abs().sum() > 0]
    leading_missing = [f for f in LEADING_FEATURE_NAMES if f not in leading_present]

    if not leading_present:
        print("  ERROR: No leading indicator features found in data.")
        print("  Run: python scripts/download_leading_indicators.py")
        return {}

    print(f"  Leading indicator features present: {len(leading_present)}/{len(LEADING_FEATURE_NAMES)}")
    if leading_missing:
        print(f"  Missing: {leading_missing}")

    # Create binary target
    df = df.copy()
    struct = df["y_structural"]
    df["y_bear_binary"] = -1
    df.loc[struct.isin([1, 2]), "y_bear_binary"] = 0
    df.loc[struct == 0, "y_bear_binary"] = 1

    bear_rate = df.loc[df["y_bear_binary"].isin([0, 1]), "y_bear_binary"].mean()
    spw = min((1 - bear_rate) / bear_rate, 8.0)

    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(es_daily, train_end_idx)

    # Baseline: without leading features
    base_features = [f for f in feature_cols if f not in LEADING_FEATURE_NAMES]
    print(f"\n  Baseline: {len(base_features)} features (no leading indicators)")
    res_base = walk_forward_binary_cv(
        df, base_features, "y_bear_binary",
        n_folds=n_folds, min_train_days=min_train,
        scale_pos_weight=spw, hmm_fn=hmm_fn,
    )
    trans_base = analyze_binary_transition_lag(res_base["actuals"], res_base["probas"])
    res_base["transition"] = trans_base
    print(f"    AUC={res_base['agg_auc']:.3f}, Day-1 recall: {trans_base['day1_recall']:.1%}")

    # With leading features
    full_features = feature_cols  # already includes leading features
    print(f"\n  With leading: {len(full_features)} features")
    res_full = walk_forward_binary_cv(
        df, full_features, "y_bear_binary",
        n_folds=n_folds, min_train_days=min_train,
        scale_pos_weight=spw, hmm_fn=hmm_fn,
    )
    trans_full = analyze_binary_transition_lag(res_full["actuals"], res_full["probas"])
    res_full["transition"] = trans_full
    print(f"    AUC={res_full['agg_auc']:.3f}, Day-1 recall: {trans_full['day1_recall']:.1%}")

    # Compare
    auc_diff = res_full["agg_auc"] - res_base["agg_auc"]
    day1_diff = trans_full["day1_recall"] - trans_base["day1_recall"]
    print(f"\n  AUC change: {auc_diff:+.3f}")
    print(f"  Day-1 recall change: {day1_diff:+.1%}")

    # Check which leading features appear in top 20
    imp = res_full["importance"]
    top20 = set(imp.head(20)["feature"].values)
    leading_in_top20 = [f for f in leading_present if f in top20]
    print(f"\n  Leading features in top 20: {len(leading_in_top20)}")
    for f in leading_in_top20:
        rank = imp[imp["feature"] == f].index[0] + 1
        imp_val = imp.loc[imp["feature"] == f, "importance"].values[0]
        print(f"    {f}: rank={rank}, importance={imp_val:.4f}")

    # All leading feature importances
    print(f"\n  All leading feature importances:")
    for f in leading_present:
        if f in imp["feature"].values:
            imp_val = imp.loc[imp["feature"] == f, "importance"].values[0]
            rank = list(imp["feature"].values).index(f) + 1
            print(f"    {f:35s} rank={rank:3d}, imp={imp_val:.4f}")

    # Success check
    success = len(leading_in_top20) >= 2 and day1_diff > 0.05
    print(f"\n  Success: >=2 leading in top-20 AND day-1 recall +5pp")
    print(f"  Leading in top-20: {len(leading_in_top20)} >= 2 → {'PASS' if len(leading_in_top20) >= 2 else 'FAIL'}")
    print(f"  Day-1 recall change: {day1_diff:+.1%} > 5% → {'PASS' if day1_diff > 0.05 else 'FAIL'}")

    return {"baseline": res_base, "with_leading": res_full,
            "leading_in_top20": leading_in_top20}


# ─────────────────────────────────────────────────────────────────────
# Experiment 4: DD Exit + Model Re-entry
# ─────────────────────────────────────────────────────────────────────

def compute_dd_exit_model_reentry(
    daily_ret: np.ndarray,
    preds: np.ndarray,
    probas: np.ndarray,
    dd_threshold: float = 0.02,
    bull_consec_required: int = 3,
    p_bull_threshold: float = 0.50,
) -> np.ndarray:
    """DD exit + model re-entry strategy.

    Exit: mechanical trailing drawdown.
    Re-entry: require bull_consec_required consecutive NOT-BEAR predictions
              with P(NOT-BEAR) > p_bull_threshold.

    Args:
        daily_ret: daily returns array
        preds: binary predictions (0=NOT-BEAR, 1=BEAR)
        probas: P(BEAR) probabilities
        dd_threshold: drawdown exit threshold (positive, e.g. 0.02 = 2%)
        bull_consec_required: consecutive NOT-BEAR days for re-entry
        p_bull_threshold: min P(NOT-BEAR) = 1 - P(BEAR) for re-entry

    Returns:
        position array (1=long, 0=flat)
    """
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    in_market = True
    not_bear_consec = 0

    for i in range(1, n):
        # Update equity based on previous position
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if in_market:
            # Check DD exit
            if (equity - peak_equity) / peak_equity < -dd_threshold:
                in_market = False
                position[i] = 0.0
                not_bear_consec = 0
            else:
                position[i] = 1.0
        else:
            # Check model re-entry
            p_not_bear = 1 - probas[i] if i < len(probas) else 0.5
            pred_not_bear = preds[i] == 0 if i < len(preds) else False

            if pred_not_bear and p_not_bear > p_bull_threshold:
                not_bear_consec += 1
            else:
                not_bear_consec = 0

            if not_bear_consec >= bull_consec_required:
                in_market = True
                position[i] = 1.0
                peak_equity = equity  # reset peak
            else:
                position[i] = 0.0

    return position


def run_experiment_4(df: pd.DataFrame, es_daily: pd.DataFrame,
                     feature_cols: list[str], args,
                     binary_results: dict | None = None) -> dict:
    """DD exit + model re-entry sweep."""
    print(f"\n{'='*60}")
    print("EXPERIMENT 4: DD Exit + Model Re-entry")
    print("=" * 60)

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200

    # Get OOS predictions from binary model (use best from exp 1, or retrain)
    if binary_results is not None:
        # Use best AUC from experiment 1
        best_spw = max(binary_results, key=lambda k: binary_results[k]["agg_auc"])
        res = binary_results[best_spw]
        print(f"  Using binary model from Exp 1 (spw={best_spw})")
    else:
        # Retrain binary model
        df_exp = df.copy()
        struct = df_exp["y_structural"]
        df_exp["y_bear_binary"] = -1
        df_exp.loc[struct.isin([1, 2]), "y_bear_binary"] = 0
        df_exp.loc[struct == 0, "y_bear_binary"] = 1

        bear_rate = df_exp.loc[df_exp["y_bear_binary"].isin([0, 1]), "y_bear_binary"].mean()
        spw = min((1 - bear_rate) / bear_rate, 8.0)

        def hmm_fn(df_fold, train_end_idx):
            return compute_hmm_features_walkforward(es_daily, train_end_idx)

        res = walk_forward_binary_cv(
            df_exp, feature_cols, "y_bear_binary",
            n_folds=n_folds, min_train_days=min_train,
            scale_pos_weight=spw, hmm_fn=hmm_fn,
        )

    preds = res["preds"]
    probas = res["probas"]

    # Get daily returns for OOS period
    days = sorted(es_daily.index.unique())
    n_days = len(days)
    oos_days = days[n_days - len(preds):]
    daily_ret = es_daily["close"].pct_change().reindex(oos_days).fillna(0).values

    # Buy-and-hold benchmark
    bh_metrics = compute_strategy_metrics(daily_ret, np.ones(len(daily_ret)), "Buy&Hold")
    print(f"\n  Buy&Hold: Sharpe={bh_metrics['sharpe']:.2f}, MaxDD={bh_metrics['max_dd']:.1%}")

    # Pure DD exit benchmarks
    print(f"\n  Pure DD exit benchmarks:")
    dd_only_results = {}
    for dd_pct in [2, 3, 4]:
        pos = np.ones(len(daily_ret))
        equity = 1.0
        peak = 1.0
        flat_counter = 0
        for i in range(1, len(daily_ret)):
            equity *= (1 + pos[i-1] * daily_ret[i])
            peak = max(peak, equity)
            if flat_counter > 0:
                pos[i] = 0.0
                flat_counter -= 1
                if flat_counter == 0:
                    peak = equity
            elif (equity - peak) / peak < -dd_pct / 100:
                pos[i] = 0.0
                flat_counter = 5
            else:
                pos[i] = 1.0
        m = compute_strategy_metrics(daily_ret, pos, f"DD_{dd_pct}pct_only")
        dd_only_results[dd_pct] = m
        print(f"    DD {dd_pct}%: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # DD exit + model re-entry sweep
    dd_values = [2, 3, 4] if not args.quick else [2, 3]
    consec_values = [2, 3, 5] if not args.quick else [2, 3]
    p_thresh_values = [0.45, 0.50, 0.55] if not args.quick else [0.50]

    sweep_results = []
    print(f"\n  Sweep: DD x Consec x P_thresh ({len(dd_values)*len(consec_values)*len(p_thresh_values)} combos)")

    for dd_pct in dd_values:
        for consec in consec_values:
            for p_thresh in p_thresh_values:
                pos = compute_dd_exit_model_reentry(
                    daily_ret, preds, probas,
                    dd_threshold=dd_pct / 100,
                    bull_consec_required=consec,
                    p_bull_threshold=p_thresh,
                )
                m = compute_strategy_metrics(
                    daily_ret, pos,
                    f"DD{dd_pct}%_c{consec}_p{p_thresh:.2f}",
                )
                m["dd_pct"] = dd_pct
                m["consec"] = consec
                m["p_thresh"] = p_thresh
                m["time_in_market"] = (pos > 0).mean()
                sweep_results.append(m)

    # Sort by Sharpe
    sweep_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  {'Label':<25s} {'Sharpe':>7s} {'MaxDD':>8s} {'Calmar':>7s} {'InMkt':>6s}")
    print("  " + "-" * 60)
    for m in sweep_results[:15]:
        print(f"  {m['label']:<25s} {m['sharpe']:>7.2f} {m['max_dd']:>7.1%} "
              f"{m['calmar']:>7.2f} {m['time_in_market']:>5.0%}")

    # Best vs pure DD
    best = sweep_results[0]
    best_dd_only = max(dd_only_results.values(), key=lambda x: x["sharpe"])
    print(f"\n  Best combo: {best['label']}")
    print(f"    Sharpe={best['sharpe']:.2f} vs DD-only Sharpe={best_dd_only['sharpe']:.2f}")
    print(f"    MaxDD={best['max_dd']:.1%} vs DD-only MaxDD={best_dd_only['max_dd']:.1%}")

    if best["sharpe"] > best_dd_only["sharpe"]:
        print(f"    SUCCESS: Combo beats DD-only by {best['sharpe']-best_dd_only['sharpe']:.2f} Sharpe")
    elif best["max_dd"] > best_dd_only["max_dd"]:
        print(f"    PARTIAL: Lower MaxDD ({best['max_dd']:.1%} vs {best_dd_only['max_dd']:.1%})")
    else:
        print(f"    FAIL: Combo does not beat DD-only")

    return {
        "sweep": sweep_results,
        "dd_only": dd_only_results,
        "buy_hold": bh_metrics,
    }


# ─────────────────────────────────────────────────────────────────────
# Part 5: Summary & Figures
# ─────────────────────────────────────────────────────────────────────

def print_summary(exp1, exp2, exp3, exp4):
    """Print final comparison table across all experiments."""
    print(f"\n{'='*60}")
    print("SUMMARY: Cross-Experiment Comparison")
    print("=" * 60)

    rows = []

    # Exp 1 best
    if exp1:
        best_spw = max(exp1, key=lambda k: exp1[k]["agg_auc"])
        r = exp1[best_spw]
        rows.append({
            "experiment": f"Exp1: Binary (spw={best_spw})",
            "auc": r["agg_auc"],
            "precision": r["agg_precision"],
            "recall": r["agg_recall"],
            "day1_recall": r["transition"]["day1_recall"],
            "day1_lag": r["transition"]["mean_lag"],
        })

    # Exp 2 best
    if exp2:
        best_thresh = max(exp2, key=lambda k: exp2[k]["agg_auc"])
        r = exp2[best_thresh]
        rows.append({
            "experiment": f"Exp2: FwdDD (t={best_thresh:.0%})",
            "auc": r["agg_auc"],
            "precision": r["agg_precision"],
            "recall": r["agg_recall"],
            "day1_recall": r["transition"]["day1_recall"],
            "day1_lag": r["transition"]["mean_lag"],
        })

    # Exp 3
    if exp3 and "with_leading" in exp3:
        r = exp3["with_leading"]
        rows.append({
            "experiment": "Exp3: +Leading indicators",
            "auc": r["agg_auc"],
            "precision": r["agg_precision"],
            "recall": r["agg_recall"],
            "day1_recall": r["transition"]["day1_recall"],
            "day1_lag": r["transition"]["mean_lag"],
        })
        r2 = exp3["baseline"]
        rows.append({
            "experiment": "Exp3: Baseline (no leading)",
            "auc": r2["agg_auc"],
            "precision": r2["agg_precision"],
            "recall": r2["agg_recall"],
            "day1_recall": r2["transition"]["day1_recall"],
            "day1_lag": r2["transition"]["mean_lag"],
        })

    if rows:
        print(f"\n  {'Experiment':<35s} {'AUC':>6s} {'Prec':>6s} {'Rec':>6s} "
              f"{'Day1R':>6s} {'Lag':>6s}")
        print("  " + "-" * 70)
        for r in rows:
            print(f"  {r['experiment']:<35s} {r['auc']:>6.3f} {r['precision']:>6.3f} "
                  f"{r['recall']:>6.3f} {r['day1_recall']:>5.1%} {r['day1_lag']:>6.1f}")

    # Exp 4 top strategies
    if exp4 and "sweep" in exp4:
        print(f"\n  Top 5 DD+Model strategies:")
        print(f"  {'Label':<25s} {'Sharpe':>7s} {'MaxDD':>8s}")
        print("  " + "-" * 45)
        for m in exp4["sweep"][:5]:
            print(f"  {m['label']:<25s} {m['sharpe']:>7.2f} {m['max_dd']:>7.1%}")
        print(f"\n  Buy&Hold: Sharpe={exp4['buy_hold']['sharpe']:.2f}, "
              f"MaxDD={exp4['buy_hold']['max_dd']:.1%}")


def save_figures(exp1, exp2, exp4, es_daily):
    """Save analysis figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Experiment 1 — SPW sweep AUC/Recall tradeoff
    if exp1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        spws = sorted(exp1.keys())
        aucs = [exp1[s]["agg_auc"] for s in spws]
        recalls = [exp1[s]["agg_recall"] for s in spws]
        precisions = [exp1[s]["agg_precision"] for s in spws]
        day1_recalls = [exp1[s]["transition"]["day1_recall"] for s in spws]

        ax1.plot(spws, aucs, "o-", label="AUC")
        ax1.plot(spws, recalls, "s-", label="Recall")
        ax1.plot(spws, precisions, "^-", label="Precision")
        ax1.set_xlabel("scale_pos_weight")
        ax1.set_ylabel("Score")
        ax1.set_title("Exp 1: Binary BEAR Detector — SPW Sweep")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(spws)), day1_recalls, tick_label=[str(s) for s in spws])
        ax2.set_xlabel("scale_pos_weight")
        ax2.set_ylabel("Day-1 BEAR Recall")
        ax2.set_title("Day-1 Transition Recall")
        ax2.axhline(0.25, color="r", linestyle="--", label="Target (25%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "exp1_binary_bear.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {FIGURES_DIR / 'exp1_binary_bear.png'}")

    # Figure 2: Experiment 2 — Threshold sweep
    if exp2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        thresholds = sorted(exp2.keys())
        aucs = [exp2[t]["agg_auc"] for t in thresholds]
        p_at_r50 = [exp2[t]["precision_at_recall_50"] for t in thresholds]
        bear_rates = [exp2[t]["bear_rate"] for t in thresholds]

        ax1.plot([f"{t:.0%}" for t in thresholds], aucs, "o-", label="AUC")
        ax1.plot([f"{t:.0%}" for t in thresholds], p_at_r50, "s-", label="P@R50")
        ax1.set_xlabel("BEAR threshold")
        ax1.set_ylabel("Score")
        ax1.set_title("Exp 2: Forward Drawdown Labels")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(thresholds)), bear_rates,
                tick_label=[f"{t:.0%}" for t in thresholds])
        ax2.set_xlabel("BEAR threshold")
        ax2.set_ylabel("BEAR label rate")
        ax2.set_title("Label Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "exp2_fwd_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {FIGURES_DIR / 'exp2_fwd_drawdown.png'}")

    # Figure 3: Experiment 4 — Strategy comparison
    if exp4 and "sweep" in exp4:
        fig, ax = plt.subplots(figsize=(10, 6))
        sweep = exp4["sweep"][:10]
        labels = [m["label"] for m in sweep]
        sharpes = [m["sharpe"] for m in sweep]
        max_dds = [abs(m["max_dd"]) * 100 for m in sweep]

        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, sharpes, width, label="Sharpe", color="steelblue")
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, max_dds, width, label="MaxDD %", color="salmon", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Sharpe Ratio")
        ax2.set_ylabel("Max Drawdown (%)")
        ax.set_title("Exp 4: DD Exit + Model Re-entry — Top 10")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Add DD-only benchmark line
        best_dd = max(exp4["dd_only"].values(), key=lambda x: x["sharpe"])
        ax.axhline(best_dd["sharpe"], color="green", linestyle="--",
                   label=f"DD-only best ({best_dd['sharpe']:.2f})")
        ax.legend(loc="upper left")

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "exp4_dd_model_reentry.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {FIGURES_DIR / 'exp4_dd_model_reentry.png'}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    # Part 0: Load data
    df, es_daily, feature_cols = load_data_pipeline(args)

    # Experiment 1: Binary BEAR detector
    exp1_results = run_experiment_1(df, es_daily, feature_cols, args)

    # Experiment 2: Forward drawdown labels
    exp2_results = run_experiment_2(df, es_daily, feature_cols, args)

    # Experiment 3: Leading indicators
    exp3_results = {}
    if not args.skip_exp3:
        exp3_results = run_experiment_3(df, es_daily, feature_cols, args)
    else:
        print(f"\n{'='*60}")
        print("EXPERIMENT 3: SKIPPED (--skip-exp3)")
        print("=" * 60)

    # Experiment 4: DD exit + model re-entry
    exp4_results = run_experiment_4(df, es_daily, feature_cols, args,
                                     binary_results=exp1_results)

    # Summary
    print_summary(exp1_results, exp2_results, exp3_results, exp4_results)

    # Figures
    print(f"\n{'='*60}")
    print("Saving figures...")
    print("=" * 60)
    save_figures(exp1_results, exp2_results, exp4_results, es_daily)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
