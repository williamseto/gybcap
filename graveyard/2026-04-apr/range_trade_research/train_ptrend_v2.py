"""P(trend) model v2 — alternative architectures for sharper cutoffs.

The v1 XGBoost logistic model smoothed over the hard boundaries that make
the heuristic score effective. This version tests:

  1. STUMPS (depth=1) — each tree is a single threshold, ensemble of thresholds
  2. DIFFERENT TARGET — predict "EOD score < 3" or "narrow range day" instead
     of trend/balance
  3. PER-CHECKPOINT MODELS — separate model for each time horizon
  4. ENSEMBLE hard-vote — random forest with class voting
  5. EARLY CONVERGENCE — how quickly does each approach reach useful accuracy?
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
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
from sandbox.range_dalton_dynamic import build_rolling_trend_scores
from sandbox.train_ptrend import (
    extract_checkpoint_features,
    FEATURE_COLS,
    FEATURE_COLS_NO_HEURISTIC,
    bootstrap_ci,
)


# ──────────────────────────────────────────────────────────────────────
# DATASET WITH MULTIPLE TARGETS
# ──────────────────────────────────────────────────────────────────────

def build_multi_target_dataset(df_1min, checkpoints):
    """Build dataset with multiple target labels per (day, checkpoint).

    Targets:
      - label_trend: EOD binary trend/balance (Dalton heuristic)
      - label_calm: EOD heuristic score < 3 (what actually drives fade PnL)
      - label_narrow: realized range in bottom 40% of days
      - eod_score: EOD heuristic score (for reference)
    """
    rth = df_1min[df_1min['ovn'] == 0].copy()
    eod_labeler = DayLabeler(binary_labels=True)

    # Pre-compute EOD scores and range stats
    day_stats = {}
    for td, grp in rth.groupby('trading_day'):
        prices = grp['close'].values
        volumes = grp['volume'].values if 'volume' in grp.columns else np.ones(len(grp))
        highs = grp['high'].values
        lows = grp['low'].values

        eod_label = eod_labeler.heuristic_label(
            prices, volumes, grp['open'].values[0], prices[-1], highs, lows)

        # EOD heuristic score (compute from full day)
        from sandbox.range_dalton_dynamic import compute_trend_score_at_minute
        eod_score = compute_trend_score_at_minute(grp, len(grp))

        realized_range = highs.max() - lows.min()

        day_stats[td] = {
            'is_trend': 1 if eod_label == 'Trend' else 0,
            'eod_score': eod_score,
            'realized_range': realized_range,
        }

    # Compute range percentiles
    ranges = [v['realized_range'] for v in day_stats.values()]
    p40 = np.percentile(ranges, 40)
    for v in day_stats.values():
        v['is_narrow'] = 1 if v['realized_range'] <= p40 else 0
        v['is_calm'] = 1 if v['eod_score'] < 3 else 0

    print(f"  Range P40 threshold: {p40:.1f} pts")
    n_calm = sum(v['is_calm'] for v in day_stats.values())
    n_narrow = sum(v['is_narrow'] for v in day_stats.values())
    n_trend = sum(v['is_trend'] for v in day_stats.values())
    print(f"  Trend: {n_trend}/{len(day_stats)} ({n_trend/len(day_stats):.0%})")
    print(f"  Calm (score<3): {n_calm}/{len(day_stats)} ({n_calm/len(day_stats):.0%})")
    print(f"  Narrow (range<P40): {n_narrow}/{len(day_stats)} ({n_narrow/len(day_stats):.0%})")

    # Build features at each checkpoint
    rows = []
    for td, grp in rth.groupby('trading_day'):
        if td not in day_stats:
            continue
        stats = day_stats[td]
        for cp in checkpoints:
            feats = extract_checkpoint_features(grp, cp)
            if feats is None:
                continue
            feats['trading_day'] = td
            feats['checkpoint'] = cp
            feats['label_trend'] = stats['is_trend']
            feats['label_calm'] = stats['is_calm']
            feats['label_narrow'] = stats['is_narrow']
            feats['eod_score'] = stats['eod_score']
            rows.append(feats)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# MODEL VARIANTS
# ──────────────────────────────────────────────────────────────────────

def make_xgb_stumps(scale_pos_weight=1.0):
    """XGBoost with depth=1 (stumps) — pure threshold ensemble."""
    return XGBClassifier(
        objective='binary:logistic',
        max_depth=1,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=30,
        reg_alpha=2.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

def make_xgb_shallow(scale_pos_weight=1.0):
    """XGBoost with depth=2 — one level of feature interactions."""
    return XGBClassifier(
        objective='binary:logistic',
        max_depth=2,
        learning_rate=0.03,
        n_estimators=400,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

def make_xgb_v1(scale_pos_weight=1.0):
    """XGBoost depth=3 (same as v1 for reference)."""
    return XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        learning_rate=0.02,
        n_estimators=500,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )

def make_rf(n_estimators=300, max_depth=3):
    """Random forest — hard voting by majority of trees."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=30,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
    )


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD EVALUATION
# ──────────────────────────────────────────────────────────────────────

def walk_forward_eval(dataset, model_factory, target_col, feature_cols,
                      n_folds=5, min_train_days=100, verbose=True):
    """Walk-forward train/test. Returns OOS predictions."""
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


def walk_forward_per_checkpoint(dataset, model_factory, target_col,
                                feature_cols, checkpoints_to_train,
                                n_folds=5, min_train_days=100):
    """Train a separate model per checkpoint for maximum early specialization."""
    days = sorted(dataset['trading_day'].unique())
    n_days = len(days)
    fold_size = n_days // n_folds

    oos_parts = []
    for cp in checkpoints_to_train:
        cp_data = dataset[dataset['checkpoint'] == cp]
        if len(cp_data) < 50:
            continue

        # Remove features that are always 0 at this checkpoint
        fcols = [f for f in feature_cols
                 if cp_data[f].std() > 1e-10]

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else n_days
            test_days = set(days[test_start:test_end])
            train_days = set(days[:test_start])

            if len(train_days) < min_train_days:
                continue

            train = cp_data[cp_data['trading_day'].isin(train_days)]
            test = cp_data[cp_data['trading_day'].isin(test_days)]

            if len(train) < 30 or len(test) < 10:
                continue

            X_train = train[fcols].values
            y_train = train[target_col].values
            X_test = test[fcols].values

            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            spw = n_neg / max(1, n_pos)

            model = model_factory(spw)
            model.fit(X_train, y_train)

            p_pos = model.predict_proba(X_test)[:, 1]

            out = test[['trading_day', 'checkpoint', target_col,
                         'heuristic_score', 'eod_score']].copy()
            out['p_pos'] = p_pos
            out['fold'] = fold
            oos_parts.append(out)

    if not oos_parts:
        return pd.DataFrame()
    return pd.concat(oos_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# FILTER EVALUATION ON FADE SIGNALS
# ──────────────────────────────────────────────────────────────────────

def apply_oos_filter(sigs, oos_preds, rth, threshold, invert=False):
    """Filter signals using OOS predictions. Keep signals where p_pos < threshold.

    If invert=True, keep where p_pos >= threshold (for "calm" targets where
    positive class = calm/narrow).
    """
    kept = []
    for _, sig in sigs.iterrows():
        td = sig['trading_day']
        bar = sig['bar_offset']
        day_preds = oos_preds[oos_preds['trading_day'] == td]
        if len(day_preds) == 0:
            continue
        valid = day_preds[day_preds['checkpoint'] <= bar]
        if len(valid) == 0:
            p = day_preds.iloc[0]['p_pos']
        else:
            p = valid.iloc[-1]['p_pos']

        if invert:
            if p >= threshold:
                kept.append(True)
            else:
                kept.append(False)
        else:
            if p < threshold:
                kept.append(True)
            else:
                kept.append(False)

    return sigs.iloc[:len(kept)][kept].copy()


def eval_filter(sigs, rth, label=""):
    """Evaluate fade signals, return (n, wr, pnl, ci_lo, ci_hi)."""
    if len(sigs) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    res = evaluate_fixed_hold(sigs, rth, [30])
    fade = res[res['signal_type'].isin(['fade_high', 'fade_low'])]
    if len(fade) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    mean, ci_lo, ci_hi = bootstrap_ci(fade['pnl'].values)
    wr = fade['win'].mean()
    return len(fade), wr, mean, ci_lo, ci_hi


def eval_stop_filter(sigs, rth, stop_pts=15):
    if len(sigs) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    st = evaluate_stop_target(sigs, rth, stop_pts, target_mode='midpoint')
    fade = st[st['signal_type'].isin(['fade_high', 'fade_low'])]
    if len(fade) == 0:
        return 0, np.nan, np.nan, np.nan, np.nan
    return (len(fade), fade['win'].mean(), fade['pnl'].mean(),
            (fade['exit_type'] == 'stop').mean(),
            (fade['exit_type'] == 'target').mean())


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("P(TREND) v2 — alternative models for sharper day-type filtering")
    print("=" * W)

    df_1min, daily = load_data()
    print(f"\nLoaded {len(df_1min):,} 1-min bars → {len(daily)} daily bars")

    checkpoints = [15, 30, 60, 90, 120, 150, 210, 270, 330]

    # ── Build dataset ──
    print("\nBuilding multi-target checkpoint dataset...")
    dataset = build_multi_target_dataset(df_1min, checkpoints)
    print(f"  {len(dataset)} rows ({dataset['trading_day'].nunique()} days)")

    fcols = FEATURE_COLS_NO_HEURISTIC  # Don't give ML the heuristic as input

    # ══════════════════════════════════════════════════════════════════
    # PART 1: MODEL × TARGET GRID — OOS AUC
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: MODEL × TARGET combinations — walk-forward AUC")
    print("=" * W)

    model_configs = [
        ('XGB stumps (d=1)', lambda spw: make_xgb_stumps(spw)),
        ('XGB shallow (d=2)', lambda spw: make_xgb_shallow(spw)),
        ('XGB d=3 (v1)', lambda spw: make_xgb_v1(spw)),
        ('RF d=3', lambda spw: make_rf(300, 3)),
        ('RF d=5', lambda spw: make_rf(300, 5)),
    ]

    targets = [
        ('label_trend', 'Trend/Balance'),
        ('label_calm', 'Calm (score<3)'),
        ('label_narrow', 'Narrow range'),
    ]

    all_oos = {}  # (model_name, target) -> oos_preds

    for target_col, target_label in targets:
        print(f"\n  Target: {target_label} ({target_col})")
        for model_name, model_factory in model_configs:
            print(f"\n    {model_name}:")
            oos = walk_forward_eval(dataset, model_factory, target_col,
                                    fcols, verbose=True)
            all_oos[(model_name, target_col)] = oos

            # Overall AUC
            try:
                overall_auc = roc_auc_score(oos[target_col].values, oos['p_pos'].values)
            except ValueError:
                overall_auc = 0.5
            print(f"    Overall AUC: {overall_auc:.3f}")

    # ── Per-checkpoint AUC ──
    print(f"\n{'=' * W}")
    print("PART 1b: AUC BY CHECKPOINT — where does each model converge?")
    print("=" * W)

    # Pick the best model per target from above, plus stumps
    for target_col, target_label in targets:
        print(f"\n  Target: {target_label}")
        print(f"  {'Model':<22}", end="")
        for cp in checkpoints:
            print(f"  {f'@{cp}':>6}", end="")
        print()

        for model_name, _ in model_configs:
            key = (model_name, target_col)
            if key not in all_oos:
                continue
            oos = all_oos[key]
            print(f"  {model_name:<22}", end="")
            for cp in checkpoints:
                sub = oos[oos['checkpoint'] == cp]
                if len(sub) < 20:
                    print(f"  {'n/a':>6}", end="")
                    continue
                try:
                    auc = roc_auc_score(sub[target_col].values, sub['p_pos'].values)
                    print(f"  {auc:>6.3f}", end="")
                except ValueError:
                    print(f"  {'n/a':>6}", end="")
            print()

        # Heuristic score as reference
        print(f"  {'Heuristic score':<22}", end="")
        for cp in checkpoints:
            sub = all_oos[(model_configs[0][0], target_col)]
            sub_cp = sub[sub['checkpoint'] == cp]
            if len(sub_cp) < 20:
                print(f"  {'n/a':>6}", end="")
                continue
            try:
                # Use heuristic_score as the "prediction" for the target
                auc = roc_auc_score(sub_cp[target_col].values,
                                    sub_cp['heuristic_score'].values)
                print(f"  {auc:>6.3f}", end="")
            except ValueError:
                print(f"  {'n/a':>6}", end="")
        print()

    # ══════════════════════════════════════════════════════════════════
    # PART 2: PER-CHECKPOINT MODELS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: PER-CHECKPOINT MODELS — specialized early detection")
    print(f"  Separate model trained for each checkpoint")
    print("=" * W)

    for target_col, target_label in [('label_calm', 'Calm'), ('label_trend', 'Trend')]:
        print(f"\n  Target: {target_label}, XGB stumps per checkpoint:")
        oos_per_cp = walk_forward_per_checkpoint(
            dataset, lambda spw: make_xgb_stumps(spw),
            target_col, fcols, checkpoints)

        if len(oos_per_cp) > 0:
            all_oos[('per-cp stumps', target_col)] = oos_per_cp
            print(f"  {'Checkpoint':<12} {'n':>5} {'AUC':>6}")
            for cp in checkpoints:
                sub = oos_per_cp[oos_per_cp['checkpoint'] == cp]
                if len(sub) < 20:
                    continue
                try:
                    auc = roc_auc_score(sub[target_col].values, sub['p_pos'].values)
                    print(f"  @{cp:<10} {len(sub):>5} {auc:>6.3f}")
                except ValueError:
                    pass

    # ══════════════════════════════════════════════════════════════════
    # PART 3: APPLY FILTERS TO FADE SIGNALS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: FILTER COMPARISON — all models as fade signal filters")
    print("=" * W)

    print("\nBuilding conditional range model...")
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

    # Baseline
    n0, wr0, pnl0, ci_lo0, ci_hi0 = eval_filter(sigs_all, rth)

    print(f"\n  30-bar hold:")
    print(f"  {'Filter':<45} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18}")
    print(f"  {'No filter':<45} {n0:>5} {wr0:>5.0%} {pnl0:>+7.1f}  "
          f"[{ci_lo0:>+7.1f}, {ci_hi0:>+7.1f}]")

    # Heuristic baseline
    from sandbox.range_dalton_dynamic import filter_signals_by_rolling_score
    sigs_h3 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 3.0)
    n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_h3, rth)
    print(f"  {'Heuristic score < 3':<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
          f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    sigs_h4 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 4.0)
    n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_h4, rth)
    print(f"  {'Heuristic score < 4':<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
          f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # ML filters
    # For "trend" target: keep where P(trend) < threshold
    # For "calm" target: keep where P(calm) > threshold (invert)
    best_configs = []
    for model_name, target_col in all_oos:
        oos = all_oos[(model_name, target_col)]
        if target_col == 'label_trend':
            for thresh in [0.4, 0.5, 0.6]:
                sigs_f = apply_oos_filter(sigs_all, oos, rth, thresh, invert=False)
                n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_f, rth)
                label = f"{model_name} / trend / P<{thresh}"
                print(f"  {label:<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
                      f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")
                best_configs.append((label, n, wr, pnl, ci_lo, ci_hi))
        elif target_col == 'label_calm':
            for thresh in [0.4, 0.5, 0.6]:
                sigs_f = apply_oos_filter(sigs_all, oos, rth, thresh, invert=True)
                n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_f, rth)
                label = f"{model_name} / calm / P≥{thresh}"
                print(f"  {label:<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
                      f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")
                best_configs.append((label, n, wr, pnl, ci_lo, ci_hi))
        elif target_col == 'label_narrow':
            for thresh in [0.4, 0.5, 0.6]:
                sigs_f = apply_oos_filter(sigs_all, oos, rth, thresh, invert=True)
                n, wr, pnl, ci_lo, ci_hi = eval_filter(sigs_f, rth)
                label = f"{model_name} / narrow / P≥{thresh}"
                print(f"  {label:<45} {n:>5} {wr:>5.0%} {pnl:>+7.1f}  "
                      f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")
                best_configs.append((label, n, wr, pnl, ci_lo, ci_hi))

    # ══════════════════════════════════════════════════════════════════
    # PART 4: EARLY CONVERGENCE — per-checkpoint filter quality
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: EARLY CONVERGENCE — filter quality by signal time")
    print(f"  How well does each filter work for signals in the first hour?")
    print("=" * W)

    # Split signals by time bucket
    early_sigs = sigs_all[sigs_all['bar_offset'] < 60]
    mid_sigs = sigs_all[(sigs_all['bar_offset'] >= 60) & (sigs_all['bar_offset'] < 210)]
    late_sigs = sigs_all[sigs_all['bar_offset'] >= 210]

    print(f"\n  Signal counts: early(<1h)={len(early_sigs)}, "
          f"mid(1-3.5h)={len(mid_sigs)}, late(3.5h+)={len(late_sigs)}")

    # Pick best ML configs + heuristic
    best_ml = sorted(best_configs, key=lambda x: -x[3] if not np.isnan(x[3]) else -999)[:5]

    for period_name, period_sigs in [('Early <1h', early_sigs),
                                      ('Mid 1-3.5h', mid_sigs),
                                      ('Late 3.5h+', late_sigs)]:
        print(f"\n  {period_name} (n={len(period_sigs)}):")
        # Heuristic
        sigs_f = filter_signals_by_rolling_score(period_sigs, rolling_scores, 3.0)
        n, wr, pnl, _, _ = eval_filter(sigs_f, rth)
        print(f"    {'Heuristic < 3':<42} n={n:>4} WR={wr:>5.0%} PnL={pnl:>+5.1f}")

        sigs_f = filter_signals_by_rolling_score(period_sigs, rolling_scores, 4.0)
        n, wr, pnl, _, _ = eval_filter(sigs_f, rth)
        print(f"    {'Heuristic < 4':<42} n={n:>4} WR={wr:>5.0%} PnL={pnl:>+5.1f}")

        # Best ML filters
        for model_name, target_col in [
            ('XGB stumps (d=1)', 'label_calm'),
            ('XGB shallow (d=2)', 'label_calm'),
            ('XGB stumps (d=1)', 'label_trend'),
            ('per-cp stumps', 'label_calm'),
        ]:
            key = (model_name, target_col)
            if key not in all_oos:
                continue
            oos = all_oos[key]
            for thresh in [0.5]:
                if target_col in ('label_calm', 'label_narrow'):
                    sigs_f = apply_oos_filter(period_sigs, oos, rth, thresh, invert=True)
                    label = f"{model_name}/{target_col}/P≥{thresh}"
                else:
                    sigs_f = apply_oos_filter(period_sigs, oos, rth, thresh, invert=False)
                    label = f"{model_name}/{target_col}/P<{thresh}"
                n, wr, pnl, _, _ = eval_filter(sigs_f, rth)
                print(f"    {label:<42} n={n:>4} WR={wr:>5.0%} PnL={pnl:>+5.1f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: STOP+TARGET for top performers
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: STOP+TARGET for top-performing filters (15pt stop, midpoint)")
    print("=" * W)

    print(f"\n  {'Filter':<40} {'n':>5} {'WR':>6} {'PnL':>7} {'stop%':>6} {'tgt%':>6}")

    # Heuristic
    for label, sigs_f in [
        ('Heuristic < 3', sigs_h3),
        ('Heuristic < 4', sigs_h4),
    ]:
        n, wr, pnl, st, tg = eval_stop_filter(sigs_f, rth, 15)
        print(f"  {label:<40} {n:>5} {wr:>5.0%} {pnl:>+7.1f} {st:>5.0%} {tg:>5.0%}")

    # ML filters
    for model_name, target_col, thresh in [
        ('XGB stumps (d=1)', 'label_calm', 0.5),
        ('XGB shallow (d=2)', 'label_calm', 0.5),
        ('XGB stumps (d=1)', 'label_trend', 0.5),
        ('RF d=3', 'label_calm', 0.5),
    ]:
        key = (model_name, target_col)
        if key not in all_oos:
            continue
        oos = all_oos[key]
        invert = target_col in ('label_calm', 'label_narrow')
        sigs_f = apply_oos_filter(sigs_all, oos, rth, thresh, invert=invert)
        label = f"{model_name}/{target_col}"
        n, wr, pnl, st, tg = eval_stop_filter(sigs_f, rth, 15)
        print(f"  {label:<40} {n:>5} {wr:>5.0%} {pnl:>+7.1f} {st:>5.0%} {tg:>5.0%}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
