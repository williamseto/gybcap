"""P(trend) v3 — non-tree models + fair heuristic comparison.

Expands on v2 by:
  1. Adding non-tree model families (logistic regression, RBF SVM, MLP)
  2. Walk-forward threshold selection for the heuristic (fair comparison)
  3. Same evaluation framework as v2 for direct comparison

Key question: is the heuristic threshold < 3 genuinely robust, or is it
an artifact of in-sample threshold selection?
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from vp.dalton_day_classifier import DayLabeler
from strategies.range_predictor.conditional import ConditionalRangeTrainer
from strategies.range_predictor.analysis import _generate_oos_predictions
from strategies.range_predictor.features import aggregate_to_daily, _compute_atr
from sandbox.range_signal_analysis import load_data, build_bar_level_predictions
from sandbox.range_signal_analysis_v2 import (
    find_fade_proximity_signals,
    evaluate_fixed_hold,
    evaluate_stop_target,
)
from sandbox.range_dalton_dynamic import (
    build_rolling_trend_scores,
    filter_signals_by_rolling_score,
    compute_trend_score_at_minute,
)
from sandbox.train_ptrend import (
    extract_checkpoint_features,
    FEATURE_COLS,
    FEATURE_COLS_NO_HEURISTIC,
    bootstrap_ci,
)
from sandbox.train_ptrend_v2 import (
    build_multi_target_dataset,
    make_xgb_stumps,
    make_xgb_shallow,
    apply_oos_filter,
    eval_filter,
    eval_stop_filter,
)


# ──────────────────────────────────────────────────────────────────────
# NON-TREE MODEL FACTORIES
# ──────────────────────────────────────────────────────────────────────

def make_logistic(scale_pos_weight=1.0):
    """Logistic regression with scaling — linear baseline."""
    # class_weight approximation from scale_pos_weight
    w = {0: 1.0, 1: scale_pos_weight}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0,
            class_weight=w,
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )),
    ])


def make_svm_rbf(scale_pos_weight=1.0):
    """RBF SVM with probability calibration."""
    w = {0: 1.0, 1: scale_pos_weight}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight=w,
            probability=True,
            random_state=42,
        )),
    ])


def make_mlp_small(scale_pos_weight=1.0):
    """Small MLP (2 hidden layers) — smooth nonlinear boundaries."""
    # MLPClassifier doesn't support class_weight, so we'll handle
    # imbalance via sample_weight in training (handled in walk-forward)
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


def make_mlp_tiny(scale_pos_weight=1.0):
    """Tiny MLP (1 hidden layer of 16) — almost-linear."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(16,),
            activation='relu',
            solver='adam',
            alpha=0.05,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD WITH SAMPLE WEIGHTS (for MLP imbalance handling)
# ──────────────────────────────────────────────────────────────────────

def walk_forward_eval(dataset, model_factory, target_col, feature_cols,
                      n_folds=5, min_train_days=100, verbose=True):
    """Walk-forward train/test. Returns OOS predictions.

    Handles MLP class imbalance via sample_weight when the model
    doesn't support class_weight directly.
    """
    days = sorted(dataset['trading_day'].unique())
    n_days = len(days)
    fold_size = n_days // n_folds

    oos_parts = []
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_days
        test_days = set(days[test_start:test_end])
        train_days = set(days[:test_start])

        if len(train_days) < min_train_days:
            continue

        train = dataset[dataset['trading_day'].isin(train_days)]
        test = dataset[dataset['trading_day'].isin(test_days)]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(1, n_pos)

        model = model_factory(spw)

        # Check if this is a pipeline with MLP (needs sample_weight)
        is_mlp = hasattr(model, 'named_steps') and 'mlp' in model.named_steps
        if is_mlp:
            # Create sample weights for imbalance
            sw = np.ones(len(y_train))
            sw[y_train == 1] = spw
            model.fit(X_train, y_train, mlp__sample_weight=sw)
        else:
            model.fit(X_train, y_train)

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba.shape[1] == 2:
                p_pos = proba[:, 1]
            else:
                p_pos = proba[:, 0]
        else:
            p_pos = model.predict(X_test).astype(float)

        out = test[['trading_day', 'checkpoint', target_col,
                     'heuristic_score', 'eod_score']].copy()
        out['p_pos'] = p_pos
        out['fold'] = fold
        oos_parts.append(out)

        if verbose:
            try:
                auc = roc_auc_score(y_test, p_pos)
            except ValueError:
                auc = 0.5
            print(f"    Fold {fold}: train={len(train_days)} days, "
                  f"test={len(test_days)} days, AUC={auc:.3f}")

    return pd.concat(oos_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD HEURISTIC THRESHOLD SELECTION
# ──────────────────────────────────────────────────────────────────────

def walk_forward_heuristic_threshold(sigs_all, rolling_scores, rth, dataset,
                                     n_folds=5, min_train_days=100,
                                     candidate_thresholds=None):
    """Walk-forward threshold selection for the heuristic score.

    At each fold, use only training-period signals to find the best
    threshold, then apply it to test-period signals.

    Returns:
        - dict of results keyed by method
        - details DataFrame with per-fold chosen thresholds
    """
    if candidate_thresholds is None:
        candidate_thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    days = sorted(dataset['trading_day'].unique())
    n_days = len(days)
    fold_size = n_days // n_folds

    fold_details = []
    oos_signals_best_wr = []
    oos_signals_best_pnl = []
    oos_signals_best_sharpe = []
    oos_signals_fixed3 = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_days
        test_days = set(days[test_start:test_end])
        train_days = set(days[:test_start])

        if len(train_days) < min_train_days:
            continue

        # Split signals into train/test by trading_day
        train_sigs = sigs_all[sigs_all['trading_day'].isin(train_days)]
        test_sigs = sigs_all[sigs_all['trading_day'].isin(test_days)]

        if len(train_sigs) < 20 or len(test_sigs) < 10:
            continue

        # Evaluate each threshold on TRAINING signals
        best_wr_thresh = 3.0
        best_pnl_thresh = 3.0
        best_sharpe_thresh = 3.0
        best_wr = 0
        best_pnl = -999
        best_sharpe = -999

        for thresh in candidate_thresholds:
            filtered = filter_signals_by_rolling_score(train_sigs, rolling_scores, thresh)
            if len(filtered) < 15:
                continue
            n, wr, pnl, _, _ = eval_filter(filtered, rth)
            if n < 15:
                continue

            # Compute Sharpe-like metric (PnL / std * sqrt(n))
            res = evaluate_fixed_hold(filtered, rth, [30])
            fade = res[res['signal_type'].isin(['fade_high', 'fade_low'])]
            if len(fade) < 15:
                continue
            pnl_std = fade['pnl'].std()
            sharpe = pnl / max(0.1, pnl_std) * np.sqrt(len(fade)) if pnl_std > 0 else 0

            if wr > best_wr:
                best_wr = wr
                best_wr_thresh = thresh
            if pnl > best_pnl:
                best_pnl = pnl
                best_pnl_thresh = thresh
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_sharpe_thresh = thresh

        fold_details.append({
            'fold': fold,
            'n_train_days': len(train_days),
            'n_test_days': len(test_days),
            'n_train_sigs': len(train_sigs),
            'n_test_sigs': len(test_sigs),
            'best_wr_thresh': best_wr_thresh,
            'best_pnl_thresh': best_pnl_thresh,
            'best_sharpe_thresh': best_sharpe_thresh,
        })

        # Apply chosen thresholds to TEST signals
        for thresh, out_list in [
            (best_wr_thresh, oos_signals_best_wr),
            (best_pnl_thresh, oos_signals_best_pnl),
            (best_sharpe_thresh, oos_signals_best_sharpe),
            (3.0, oos_signals_fixed3),
        ]:
            filtered = filter_signals_by_rolling_score(test_sigs, rolling_scores, thresh)
            if len(filtered) > 0:
                out_list.append(filtered)

    results = {}
    for label, parts in [
        ('WF heuristic (best-WR thresh)', oos_signals_best_wr),
        ('WF heuristic (best-PnL thresh)', oos_signals_best_pnl),
        ('WF heuristic (best-Sharpe thresh)', oos_signals_best_sharpe),
        ('WF heuristic (fixed thresh=3)', oos_signals_fixed3),
    ]:
        if parts:
            combined = pd.concat(parts, ignore_index=True)
            n, wr, pnl, ci_lo, ci_hi = eval_filter(combined, rth)
            results[label] = (n, wr, pnl, ci_lo, ci_hi)
        else:
            results[label] = (0, np.nan, np.nan, np.nan, np.nan)

    return results, pd.DataFrame(fold_details)


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD ML THRESHOLD SELECTION (fair comparison for ML too)
# ──────────────────────────────────────────────────────────────────────

def walk_forward_ml_threshold(sigs_all, oos_preds, rth, dataset, target_col,
                              invert=False, n_folds=5, min_train_days=100,
                              candidate_thresholds=None):
    """Walk-forward threshold selection for ML model predictions.

    Same approach as heuristic: at each fold, find best threshold
    on training signals, apply to test signals.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

    days = sorted(dataset['trading_day'].unique())
    n_days = len(days)
    fold_size = n_days // n_folds

    oos_signals = []

    fold_details = []
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_days
        test_days = set(days[test_start:test_end])
        train_days = set(days[:test_start])

        if len(train_days) < min_train_days:
            continue

        train_sigs = sigs_all[sigs_all['trading_day'].isin(train_days)]
        test_sigs = sigs_all[sigs_all['trading_day'].isin(test_days)]

        if len(train_sigs) < 20 or len(test_sigs) < 10:
            continue

        best_pnl = -999
        best_thresh = 0.5

        for thresh in candidate_thresholds:
            filtered = apply_oos_filter(train_sigs, oos_preds, rth, thresh, invert=invert)
            if len(filtered) < 15:
                continue
            n, wr, pnl, _, _ = eval_filter(filtered, rth)
            if n >= 15 and pnl > best_pnl:
                best_pnl = pnl
                best_thresh = thresh

        fold_details.append({
            'fold': fold,
            'best_thresh': best_thresh,
            'train_best_pnl': best_pnl,
        })

        # Apply to test
        filtered = apply_oos_filter(test_sigs, oos_preds, rth, best_thresh, invert=invert)
        if len(filtered) > 0:
            oos_signals.append(filtered)

    if oos_signals:
        combined = pd.concat(oos_signals, ignore_index=True)
        n, wr, pnl, ci_lo, ci_hi = eval_filter(combined, rth)
        return (n, wr, pnl, ci_lo, ci_hi), pd.DataFrame(fold_details)
    return (0, np.nan, np.nan, np.nan, np.nan), pd.DataFrame(fold_details)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("P(TREND) v3 — non-tree models + fair heuristic comparison")
    print("=" * W)

    df_1min, daily = load_data()
    print(f"\nLoaded {len(df_1min):,} 1-min bars → {len(daily)} daily bars")

    checkpoints = [15, 30, 60, 90, 120, 150, 210, 270, 330]

    # ── Build dataset ──
    print("\nBuilding multi-target checkpoint dataset...")
    dataset = build_multi_target_dataset(df_1min, checkpoints)
    print(f"  {len(dataset)} rows ({dataset['trading_day'].nunique()} days)")

    fcols = FEATURE_COLS_NO_HEURISTIC

    # ══════════════════════════════════════════════════════════════════
    # PART 1: NON-TREE MODELS — AUC comparison
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: NON-TREE MODELS — walk-forward AUC")
    print("=" * W)

    model_configs = [
        ('Logistic Regression', lambda spw: make_logistic(spw)),
        ('SVM RBF', lambda spw: make_svm_rbf(spw)),
        ('MLP small (64,32)', lambda spw: make_mlp_small(spw)),
        ('MLP tiny (16)', lambda spw: make_mlp_tiny(spw)),
        # Tree baselines for reference
        ('XGB stumps (d=1)', lambda spw: make_xgb_stumps(spw)),
        ('XGB shallow (d=2)', lambda spw: make_xgb_shallow(spw)),
    ]

    targets = [
        ('label_trend', 'Trend/Balance'),
        ('label_calm', 'Calm (score<3)'),
        ('label_narrow', 'Narrow range'),
    ]

    all_oos = {}

    for target_col, target_label in targets:
        print(f"\n  Target: {target_label} ({target_col})")
        for model_name, model_factory in model_configs:
            print(f"\n    {model_name}:")
            oos = walk_forward_eval(dataset, model_factory, target_col,
                                    fcols, verbose=True)
            all_oos[(model_name, target_col)] = oos

            try:
                overall_auc = roc_auc_score(oos[target_col].values, oos['p_pos'].values)
            except ValueError:
                overall_auc = 0.5
            print(f"    Overall AUC: {overall_auc:.3f}")

    # ── AUC summary table ──
    print(f"\n{'=' * W}")
    print("PART 1b: AUC SUMMARY TABLE")
    print("=" * W)

    print(f"\n  {'Model':<25}", end="")
    for _, tl in targets:
        print(f"  {tl:>16}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in targets:
        print(f"  {'-'*16}", end="")
    print()

    for model_name, _ in model_configs:
        print(f"  {model_name:<25}", end="")
        for target_col, _ in targets:
            oos = all_oos.get((model_name, target_col))
            if oos is not None:
                try:
                    auc = roc_auc_score(oos[target_col].values, oos['p_pos'].values)
                    print(f"  {auc:>16.3f}", end="")
                except ValueError:
                    print(f"  {'n/a':>16}", end="")
            else:
                print(f"  {'—':>16}", end="")
        print()

    # ── Per-checkpoint AUC for calm target (the most actionable) ──
    print(f"\n{'=' * W}")
    print("PART 1c: PER-CHECKPOINT AUC — Calm target (best for fade filtering)")
    print("=" * W)

    print(f"\n  {'Model':<25}", end="")
    for cp in checkpoints:
        print(f"  {f'@{cp}':>6}", end="")
    print()

    for model_name, _ in model_configs:
        key = (model_name, 'label_calm')
        if key not in all_oos:
            continue
        oos = all_oos[key]
        print(f"  {model_name:<25}", end="")
        for cp in checkpoints:
            sub = oos[oos['checkpoint'] == cp]
            if len(sub) < 20:
                print(f"  {'n/a':>6}", end="")
                continue
            try:
                auc = roc_auc_score(sub['label_calm'].values, sub['p_pos'].values)
                print(f"  {auc:>6.3f}", end="")
            except ValueError:
                print(f"  {'n/a':>6}", end="")
        print()

    # Heuristic reference (note: for calm, heuristic score is anti-correlated)
    ref_oos = all_oos.get(('XGB stumps (d=1)', 'label_calm'))
    if ref_oos is not None:
        print(f"  {'Heuristic (1 - AUC)':<25}", end="")
        for cp in checkpoints:
            sub = ref_oos[ref_oos['checkpoint'] == cp]
            if len(sub) < 20:
                print(f"  {'n/a':>6}", end="")
                continue
            try:
                raw_auc = roc_auc_score(sub['label_calm'].values,
                                        sub['heuristic_score'].values)
                # For calm, heuristic is anti-correlated, so effective AUC = 1 - raw
                print(f"  {1 - raw_auc:>6.3f}", end="")
            except ValueError:
                print(f"  {'n/a':>6}", end="")
        print()

    # ══════════════════════════════════════════════════════════════════
    # PART 2: FAIR HEURISTIC COMPARISON — walk-forward threshold
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: FAIR HEURISTIC COMPARISON — walk-forward threshold selection")
    print(f"  At each fold: pick best threshold on train signals, apply to test")
    print("=" * W)

    # Build range model predictions and signals
    print("\nBuilding conditional range model predictions...")
    static_oos = _generate_oos_predictions(daily, 'daily', n_folds=5,
                                           min_train_days=100)
    static_oos = static_oos.rename(columns={
        'pred_range_high': 'static_high', 'pred_range_low': 'static_low',
    })
    trainer = ConditionalRangeTrainer()
    cond_oos = trainer.walk_forward_evaluate(df_1min, daily, verbose=False)
    rth = build_bar_level_predictions(df_1min, daily, cond_oos, static_oos)

    rolling_scores = build_rolling_trend_scores(df_1min, checkpoints)

    prox_pts = 10
    sigs_all = find_fade_proximity_signals(rth, 'cond', prox_pts)
    print(f"  {len(sigs_all)} proximity fade signals")

    # Walk-forward heuristic threshold
    print("\n  Walk-forward heuristic threshold selection:")
    h_results, h_details = walk_forward_heuristic_threshold(
        sigs_all, rolling_scores, rth, dataset)

    print(f"\n  Per-fold selected thresholds:")
    print(f"  {'Fold':<6} {'Train days':<12} {'Best-WR':<10} {'Best-PnL':<10} {'Best-Sharpe':<12}")
    for _, row in h_details.iterrows():
        print(f"  {int(row['fold']):<6} {int(row['n_train_days']):<12} "
              f"{row['best_wr_thresh']:<10.1f} {row['best_pnl_thresh']:<10.1f} "
              f"{row['best_sharpe_thresh']:<12.1f}")

    print(f"\n  {'Method':<45} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18}")

    # In-sample heuristic (the unfair baseline we've been using)
    sigs_h3 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 3.0)
    n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_h3, rth)
    print(f"  {'In-sample heuristic (thresh=3, unfair)':<45} "
          f"{n:>5} {wr:>5.0%} {pnl:>+7.1f}  [{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    for label, (n, wr, pnl, ci_lo, ci_hi) in h_results.items():
        print(f"  {label:<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
              f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: ALL MODELS AS FADE FILTERS (with WF threshold for ML too)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: FILTER COMPARISON — fair walk-forward threshold for all")
    print("=" * W)

    print(f"\n  {'Filter':<50} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18}")

    # Baseline: no filter
    n0, wr0, pnl0, ci_lo0, ci_hi0 = eval_filter(sigs_all, rth)
    print(f"  {'No filter':<50} {n0:>5} {wr0:>5.0%} {pnl0:>+7.1f}  "
          f"[{ci_lo0:>+7.1f}, {ci_hi0:>+7.1f}]")

    # WF heuristic (best-PnL version = fairest comparison)
    for label, (n, wr, pnl, ci_lo, ci_hi) in h_results.items():
        print(f"  {label:<50} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
              f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # ML filters with WF threshold and fixed thresholds
    ml_filter_configs = [
        ('Logistic / calm', 'Logistic Regression', 'label_calm', True),
        ('SVM RBF / calm', 'SVM RBF', 'label_calm', True),
        ('MLP small / calm', 'MLP small (64,32)', 'label_calm', True),
        ('MLP tiny / calm', 'MLP tiny (16)', 'label_calm', True),
        ('XGB stumps / calm', 'XGB stumps (d=1)', 'label_calm', True),
        ('XGB shallow / calm', 'XGB shallow (d=2)', 'label_calm', True),
        ('Logistic / trend', 'Logistic Regression', 'label_trend', False),
        ('SVM RBF / trend', 'SVM RBF', 'label_trend', False),
        ('MLP small / trend', 'MLP small (64,32)', 'label_trend', False),
        ('MLP tiny / trend', 'MLP tiny (16)', 'label_trend', False),
        ('XGB stumps / trend', 'XGB stumps (d=1)', 'label_trend', False),
    ]

    wf_ml_results = {}
    for short_name, model_name, target_col, invert in ml_filter_configs:
        key = (model_name, target_col)
        if key not in all_oos:
            continue
        oos = all_oos[key]

        # Fixed thresholds
        for thresh in [0.4, 0.5, 0.6]:
            sigs_f = apply_oos_filter(sigs_all, oos, rth, thresh, invert=invert)
            n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_f, rth)
            op = '≥' if invert else '<'
            label = f"{short_name} / P{op}{thresh}"
            print(f"  {label:<50} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
                  f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

        # WF threshold (fairest comparison)
        (n, wr, pnl, ci_lo, ci_hi), ml_details = walk_forward_ml_threshold(
            sigs_all, oos, rth, dataset, target_col, invert=invert)
        label = f"{short_name} / WF-best-thresh"
        print(f"  {label:<50} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
              f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")
        wf_ml_results[short_name] = (n, wr, pnl, ci_lo, ci_hi)

    # ══════════════════════════════════════════════════════════════════
    # PART 4: EARLY CONVERGENCE — non-tree models in first hour
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: EARLY CONVERGENCE — first-hour signals only")
    print("=" * W)

    early_sigs = sigs_all[sigs_all['bar_offset'] < 60]
    mid_sigs = sigs_all[(sigs_all['bar_offset'] >= 60) & (sigs_all['bar_offset'] < 210)]
    late_sigs = sigs_all[sigs_all['bar_offset'] >= 210]

    print(f"\n  Signal counts: early(<1h)={len(early_sigs)}, "
          f"mid(1-3.5h)={len(mid_sigs)}, late(3.5h+)={len(late_sigs)}")

    filter_configs_convergence = [
        ('Heuristic < 3 (insample)', None, None, None, 3.0),
        ('Logistic / calm / P≥0.5', 'Logistic Regression', 'label_calm', True, 0.5),
        ('SVM RBF / calm / P≥0.5', 'SVM RBF', 'label_calm', True, 0.5),
        ('MLP small / calm / P≥0.5', 'MLP small (64,32)', 'label_calm', True, 0.5),
        ('MLP tiny / calm / P≥0.5', 'MLP tiny (16)', 'label_calm', True, 0.5),
        ('XGB stumps / calm / P≥0.5', 'XGB stumps (d=1)', 'label_calm', True, 0.5),
    ]

    for period_name, period_sigs in [('Early <1h', early_sigs),
                                      ('Mid 1-3.5h', mid_sigs),
                                      ('Late 3.5h+', late_sigs)]:
        print(f"\n  {period_name} (n={len(period_sigs)}):")
        for label, model_name, target_col, invert, thresh in filter_configs_convergence:
            if model_name is None:
                # Heuristic
                sigs_f = filter_signals_by_rolling_score(period_sigs, rolling_scores, thresh)
            else:
                key = (model_name, target_col)
                if key not in all_oos:
                    continue
                oos = all_oos[key]
                sigs_f = apply_oos_filter(period_sigs, oos, rth, thresh, invert=invert)
            n, wr, pnl, _, _ = eval_filter(sigs_f, rth)
            print(f"    {label:<42} n={n:>4} WR={wr:>5.0%} PnL={pnl:>+5.1f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: STOP+TARGET for top performers
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: STOP+TARGET (15pt stop, midpoint target)")
    print("=" * W)

    print(f"\n  {'Filter':<50} {'n':>5} {'WR':>6} {'PnL':>7} {'stop%':>6} {'tgt%':>6}")

    # Heuristic baselines
    for label, sigs_f in [
        ('In-sample heuristic (thresh=3)', sigs_h3),
    ]:
        n, wr, pnl, st, tg = eval_stop_filter(sigs_f, rth, 15)
        print(f"  {label:<50} {n:>5} {wr:>5.0%} {pnl:>+7.1f} {st:>5.0%} {tg:>5.0%}")

    # Top ML filters
    for short_name, model_name, target_col, invert in [
        ('Logistic / calm', 'Logistic Regression', 'label_calm', True),
        ('SVM RBF / calm', 'SVM RBF', 'label_calm', True),
        ('MLP small / calm', 'MLP small (64,32)', 'label_calm', True),
        ('MLP tiny / calm', 'MLP tiny (16)', 'label_calm', True),
        ('XGB stumps / calm', 'XGB stumps (d=1)', 'label_calm', True),
    ]:
        key = (model_name, target_col)
        if key not in all_oos:
            continue
        oos = all_oos[key]
        sigs_f = apply_oos_filter(sigs_all, oos, rth, 0.5, invert=invert)
        n, wr, pnl, st, tg = eval_stop_filter(sigs_f, rth, 15)
        label = f"{short_name} / P≥0.5"
        print(f"  {label:<50} {n:>5} {wr:>5.0%} {pnl:>+7.1f} {st:>5.0%} {tg:>5.0%}")

    # ══════════════════════════════════════════════════════════════════
    # PART 6: HEURISTIC STABILITY — does threshold 3 hold in sub-periods?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 6: HEURISTIC STABILITY — best threshold by sub-period")
    print(f"  Tests whether threshold=3 is universally best or period-specific")
    print("=" * W)

    # Split signals into thirds by date
    sig_days = sorted(sigs_all['trading_day'].unique())
    n_periods = 3
    period_size = len(sig_days) // n_periods

    for i in range(n_periods):
        start = i * period_size
        end = start + period_size if i < n_periods - 1 else len(sig_days)
        period_days = set(sig_days[start:end])
        period_sigs = sigs_all[sigs_all['trading_day'].isin(period_days)]

        date_range = f"{sig_days[start]} → {sig_days[min(end-1, len(sig_days)-1)]}"
        print(f"\n  Period {i+1}: {date_range} ({len(period_sigs)} signals)")
        print(f"  {'Threshold':<15} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18}")

        for thresh in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            filtered = filter_signals_by_rolling_score(period_sigs, rolling_scores, thresh)
            n, wr, pnl, ci_lo, ci_hi = eval_filter(filtered, rth)
            marker = " ←" if thresh == 3.0 else ""
            print(f"  {'< ' + str(thresh):<15} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
                  f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]{marker}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
