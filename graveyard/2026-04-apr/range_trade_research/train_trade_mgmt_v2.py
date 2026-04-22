"""Trade management v2 — structural exit models.

v1 tried bar-level P(should_hold) → AUC 0.58, no improvement over fixed hold.
The problem: predicting short-term price direction is too noisy.

v2 tests three structural framings where a model has real signal:

  1. EARLY TRIAGE (bar 5-10): classify trade as runner/chopper/loser from
     initial post-entry behavior. Set exit strategy per category.

  2. REGIME CHANGE: predict P(day transitions to trend) at each checkpoint.
     If fade thesis breaks down mid-trade, exit.

  3. BOUNDARY RETEST: predict P(price retests the faded boundary within N bars).
     If retest is likely, exit (level is being challenged).

All three predict structural market properties, not random bar-by-bar moves.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from strategies.range_predictor.conditional import ConditionalRangeTrainer
from strategies.range_predictor.analysis import _generate_oos_predictions
from strategies.range_predictor.features import aggregate_to_daily
from sandbox.range_signal_analysis import load_data, build_bar_level_predictions
from sandbox.range_signal_analysis_v2 import (
    find_fade_proximity_signals,
    evaluate_fixed_hold,
    evaluate_stop_target,
)
from sandbox.range_dalton_dynamic import (
    build_rolling_trend_scores,
    filter_signals_by_rolling_score,
)
from sandbox.train_ptrend import (
    extract_checkpoint_features,
    bootstrap_ci,
    FEATURE_COLS_NO_HEURISTIC,
)
from sandbox.train_trade_mgmt import extract_trade_paths, compute_signal_stats


# ──────────────────────────────────────────────────────────────────────
# 1. EARLY TRIAGE — classify trade quality from first N bars
# ──────────────────────────────────────────────────────────────────────

def build_triage_dataset(paths, triage_bar=10):
    """At bar `triage_bar`, extract features and label trade outcome.

    Labels (based on PnL at bar 30 or EOD):
      - runner:  PnL > +5 pts
      - chopper: -3 <= PnL <= +5
      - loser:   PnL < -3 pts

    For binary, we also label: good_trade = PnL > 0 at bar 30.
    """
    rows = []
    for p in paths:
        if p['n_bars'] < triage_bar + 1:
            continue

        # Target: PnL at bar 30 (or end of path)
        target_bar = min(29, p['n_bars'] - 1)
        pnl_30 = p['unrealized'][target_bar]

        # Label categories
        if pnl_30 > 5:
            category = 'runner'
        elif pnl_30 < -3:
            category = 'loser'
        else:
            category = 'chopper'

        good_trade = 1 if pnl_30 > 0 else 0

        # Features at triage_bar
        t = triage_bar
        unr = p['unrealized'][:t]
        mfe = p['mfe'][t - 1]
        mae = p['mae'][t - 1]
        current_pnl = p['unrealized'][t - 1]

        # Price momentum
        if t >= 5:
            pnl_slope = (unr[-1] - unr[-5]) / 5.0
        else:
            pnl_slope = (unr[-1] - unr[0]) / max(1, t - 1)

        # MFE speed (how fast did we reach peak)
        mfe_speed = mfe / t

        # Giveback from peak
        giveback = mfe - current_pnl

        # Directional consistency: what fraction of bars moved in our direction?
        direction = 1.0 if p['is_bull'] else -1.0
        bar_returns = np.diff(p['closes'][:t])
        dir_consistency = np.sum(bar_returns * direction > 0) / max(1, len(bar_returns))

        # Volatility of unrealized PnL
        pnl_vol = np.std(unr) if len(unr) > 1 else 0.0

        # Range expansion in first t bars
        range_used = (p['highs'][:t].max() - p['lows'][:t].min())
        pred_width = max(1.0, p['pred_high'] - p['pred_low'])
        range_consumed = range_used / pred_width

        # Volume pattern
        vols = p['volumes'][:t]
        vol_trend = (vols[-3:].mean() / max(1, vols[:3].mean())) if t >= 6 else 1.0

        # Entry context
        entry_time_norm = p['bar_offset_entry'] / 390.0
        is_bull = 1.0 if p['is_bull'] else -1.0

        rows.append({
            'trading_day': p['trading_day'],
            'signal_idx': p['signal_idx'],
            # Targets
            'pnl_30': pnl_30,
            'category': category,
            'good_trade': good_trade,
            'mfe_total': p['mfe'][-1],
            # Features
            'current_pnl': current_pnl,
            'mfe_at_triage': mfe,
            'mae_at_triage': mae,
            'mfe_speed': mfe_speed,
            'giveback': giveback,
            'pnl_slope': pnl_slope,
            'dir_consistency': dir_consistency,
            'pnl_vol': pnl_vol,
            'range_consumed': range_consumed,
            'vol_trend': vol_trend,
            'entry_time_norm': entry_time_norm,
            'direction': is_bull,
        })

    return pd.DataFrame(rows)


TRIAGE_FEATURES = [
    'current_pnl', 'mfe_at_triage', 'mae_at_triage', 'mfe_speed',
    'giveback', 'pnl_slope', 'dir_consistency', 'pnl_vol',
    'range_consumed', 'vol_trend', 'entry_time_norm', 'direction',
]


# ──────────────────────────────────────────────────────────────────────
# 2. REGIME CHANGE — predict P(trend transition) at checkpoints
# ──────────────────────────────────────────────────────────────────────

def build_regime_change_dataset(df_1min, rolling_scores, checkpoints):
    """For each (day, checkpoint), predict if the trend score will rise above
    a threshold by EOD, given current state.

    Target: score_rises = 1 if EOD score > 4 AND current score < 3
    (i.e., the day STARTS calm but BECOMES trendy)
    """
    rth = df_1min[df_1min['ovn'] == 0]

    rows = []
    for td, grp in rth.groupby('trading_day'):
        if td not in rolling_scores:
            continue
        day_scores = rolling_scores[td]
        cps = sorted(day_scores.keys())
        if not cps:
            continue

        eod_score = day_scores.get(cps[-1], 0)

        for cp in checkpoints:
            if cp not in day_scores:
                continue
            current_score = day_scores[cp]

            # Only relevant for days that start calm
            if current_score >= 3.0:
                continue

            # Target: does the day become trendy?
            score_rises = 1 if eod_score >= 4.0 else 0

            # Features at this checkpoint
            feats = extract_checkpoint_features(grp, cp)
            if feats is None:
                continue

            feats['trading_day'] = td
            feats['checkpoint'] = cp
            feats['current_score'] = current_score
            feats['score_rises'] = score_rises
            rows.append(feats)

    return pd.DataFrame(rows)


REGIME_FEATURES = FEATURE_COLS_NO_HEURISTIC + ['current_score']


# ──────────────────────────────────────────────────────────────────────
# 3. BOUNDARY RETEST — predict P(price retests faded boundary)
# ──────────────────────────────────────────────────────────────────────

def build_retest_dataset(paths, eval_bars=30, retest_proximity=5):
    """After initial fade, predict whether price returns close to the
    faded boundary within eval_bars.

    Target: retests = 1 if at any point in [triage_bar, triage_bar+eval_bars],
    the price gets within retest_proximity of the faded boundary.
    """
    rows = []
    for p in paths:
        n = p['n_bars']
        if n < 15:
            continue

        # Check at multiple observation points (bar 5, 10, 15)
        for obs_bar in [5, 10, 15]:
            if n < obs_bar + 5:
                continue

            entry_price = p['entry_price']
            is_bull = p['is_bull']

            # The faded boundary
            if is_bull:
                # Faded the low boundary (bought) — retest = price goes back down near pred_low
                boundary = p['pred_low']
                # Check if future lows get within retest_proximity of boundary
                future_lows = p['lows'][obs_bar:min(obs_bar + eval_bars, n)]
                retest = 1 if len(future_lows) > 0 and future_lows.min() <= boundary + retest_proximity else 0
            else:
                # Faded the high boundary (sold) — retest = price goes back up near pred_high
                boundary = p['pred_high']
                future_highs = p['highs'][obs_bar:min(obs_bar + eval_bars, n)]
                retest = 1 if len(future_highs) > 0 and future_highs.max() >= boundary - retest_proximity else 0

            # Features at observation point
            t = obs_bar
            current_pnl = p['unrealized'][t - 1]
            mfe = p['mfe'][t - 1]
            mae = p['mae'][t - 1]

            # Distance from boundary right now
            if is_bull:
                dist_to_boundary = p['closes'][t - 1] - boundary
            else:
                dist_to_boundary = boundary - p['closes'][t - 1]

            pred_width = max(1.0, p['pred_high'] - p['pred_low'])
            dist_norm = dist_to_boundary / pred_width

            # Momentum toward/away from boundary
            if t >= 5:
                pnl_slope = (p['unrealized'][t - 1] - p['unrealized'][t - 5]) / 5.0
            else:
                pnl_slope = (p['unrealized'][t - 1] - p['unrealized'][0]) / max(1, t - 1)

            # Is price moving TOWARD or AWAY from the boundary?
            # Positive slope = moving away (good for fade)
            # Negative slope = moving back toward boundary (bad for fade)

            # Range consumed
            range_used = p['highs'][:t].max() - p['lows'][:t].min()
            range_consumed = range_used / pred_width

            # How far has price moved from boundary since entry?
            pnl_norm = current_pnl / pred_width

            rows.append({
                'trading_day': p['trading_day'],
                'signal_idx': p['signal_idx'],
                'obs_bar': obs_bar,
                # Target
                'retests': retest,
                # Features
                'current_pnl': current_pnl,
                'pnl_norm': pnl_norm,
                'mfe': mfe,
                'mae': mae,
                'dist_to_boundary': dist_to_boundary,
                'dist_norm': dist_norm,
                'pnl_slope': pnl_slope,
                'range_consumed': range_consumed,
                'mfe_speed': mfe / max(1, t),
                'giveback': mfe - current_pnl,
                'entry_time_norm': p['bar_offset_entry'] / 390.0,
                'obs_bar_norm': obs_bar / 60.0,
                'direction': 1.0 if is_bull else -1.0,
            })

    return pd.DataFrame(rows)


RETEST_FEATURES = [
    'current_pnl', 'pnl_norm', 'mfe', 'mae', 'dist_to_boundary',
    'dist_norm', 'pnl_slope', 'range_consumed', 'mfe_speed',
    'giveback', 'entry_time_norm', 'obs_bar_norm', 'direction',
]


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD MACHINERY
# ──────────────────────────────────────────────────────────────────────

def walk_forward_signal_level(dataset, feature_cols, target_col,
                              model_factory, n_folds=5, min_train_days=100):
    """Walk-forward at the signal level (one row per trade)."""
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

        if len(train) < 30 or len(test) < 10:
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(1, n_pos)

        model = model_factory(spw)
        model.fit(X_train, y_train)

        p_pos = model.predict_proba(X_test)[:, 1]

        out = test[['trading_day', 'signal_idx']].copy() if 'signal_idx' in test.columns else test[['trading_day']].copy()
        out[target_col] = y_test
        out['p_pos'] = p_pos
        out['fold'] = fold
        if 'obs_bar' in test.columns:
            out['obs_bar'] = test['obs_bar'].values
        if 'checkpoint' in test.columns:
            out['checkpoint'] = test['checkpoint'].values
        oos_parts.append(out)

        try:
            auc = roc_auc_score(y_test, p_pos)
        except ValueError:
            auc = 0.5
        print(f"    Fold {fold}: train={len(train)} ({len(train_days)}d), "
              f"test={len(test)} ({len(test_days)}d), AUC={auc:.3f}")

        # Feature importance for last fold
        if fold == n_folds - 1 and hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=feature_cols)
            imp = imp.sort_values(ascending=False)
            print(f"    Top features:")
            for feat, val in imp.head(6).items():
                print(f"      {feat:<25} {val:.3f}")

    if not oos_parts:
        return pd.DataFrame()
    return pd.concat(oos_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# EXIT STRATEGIES
# ──────────────────────────────────────────────────────────────────────

def simulate_triage_exit(paths, triage_preds, triage_bar=10, max_hold=60):
    """Use triage prediction to set exit strategy.

    Runners (P(good)>0.65): hold to bar 45
    Choppers (0.35 < P(good) < 0.65): hold to bar 20
    Losers (P(good)<0.35): exit at triage bar
    """
    pred_map = {}
    for _, row in triage_preds.iterrows():
        pred_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

    results = []
    for p in paths:
        key = (p['trading_day'], p['signal_idx'])
        p_good = pred_map.get(key, 0.5)

        if p_good >= 0.65:
            exit_bar = min(44, p['n_bars'] - 1)  # Runner: hold long
            exit_type = 'runner_hold'
        elif p_good < 0.35:
            exit_bar = min(triage_bar - 1, p['n_bars'] - 1)  # Loser: exit early
            exit_type = 'loser_exit'
        else:
            exit_bar = min(19, p['n_bars'] - 1)  # Chopper: moderate hold
            exit_type = 'chopper_hold'

        pnl = p['unrealized'][exit_bar]
        results.append({
            'signal_type': p['signal_type'],
            'trading_day': p['trading_day'],
            'bar_offset': p['bar_offset_entry'],
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': exit_bar + 1,
            'exit_type': exit_type,
            'p_good': p_good,
        })
    return pd.DataFrame(results)


def simulate_triage_exit_parametric(paths, triage_preds, triage_bar=10,
                                     runner_hold=45, chopper_hold=20,
                                     runner_thresh=0.6, loser_thresh=0.4):
    """Parametric version for grid search."""
    pred_map = {}
    for _, row in triage_preds.iterrows():
        pred_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

    results = []
    for p in paths:
        key = (p['trading_day'], p['signal_idx'])
        p_good = pred_map.get(key, 0.5)

        if p_good >= runner_thresh:
            exit_bar = min(runner_hold - 1, p['n_bars'] - 1)
        elif p_good < loser_thresh:
            exit_bar = min(triage_bar - 1, p['n_bars'] - 1)
        else:
            exit_bar = min(chopper_hold - 1, p['n_bars'] - 1)

        pnl = p['unrealized'][exit_bar]
        results.append({
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': exit_bar + 1,
            'p_good': p_good,
        })
    return pd.DataFrame(results)


def simulate_retest_exit(paths, retest_preds, obs_bar=10, max_hold=30,
                         exit_thresh=0.6):
    """If P(retest) is high at observation point, exit early.
    Otherwise hold to max_hold.
    """
    pred_map = {}
    for _, row in retest_preds.iterrows():
        if row['obs_bar'] == obs_bar:
            pred_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

    results = []
    for p in paths:
        key = (p['trading_day'], p['signal_idx'])
        p_retest = pred_map.get(key, 0.5)

        if p_retest >= exit_thresh:
            # Retest likely — exit now
            exit_bar = min(obs_bar - 1, p['n_bars'] - 1)
            exit_type = 'retest_exit'
        else:
            # Level holding — keep position
            exit_bar = min(max_hold - 1, p['n_bars'] - 1)
            exit_type = 'hold'

        pnl = p['unrealized'][exit_bar]
        results.append({
            'signal_type': p['signal_type'],
            'trading_day': p['trading_day'],
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': exit_bar + 1,
            'exit_type': exit_type,
            'p_retest': p_retest,
        })
    return pd.DataFrame(results)


def simulate_regime_exit(paths, regime_preds, rolling_scores, checkpoints,
                         exit_thresh=0.5, max_hold=30):
    """Exit if P(regime transition) rises above threshold mid-trade.

    At each checkpoint post-entry, check if the model predicts the day
    is transitioning to trend. If so, exit.
    """
    # Build lookup: (trading_day, checkpoint) -> p(regime_change)
    pred_map = {}
    if len(regime_preds) > 0:
        for _, row in regime_preds.iterrows():
            pred_map[(row['trading_day'], row['checkpoint'])] = row['p_pos']

    results = []
    for p in paths:
        td = p['trading_day']
        entry_offset = p['bar_offset_entry']

        exit_bar = min(max_hold - 1, p['n_bars'] - 1)
        exit_type = 'hold'

        # Check at each checkpoint that falls within our hold period
        for cp in checkpoints:
            if cp <= entry_offset:
                continue
            bars_into_trade = cp - entry_offset
            if bars_into_trade > exit_bar + 1:
                break

            p_regime = pred_map.get((td, cp), 0.0)
            if p_regime >= exit_thresh:
                exit_bar = min(bars_into_trade - 1, p['n_bars'] - 1)
                exit_type = 'regime_exit'
                break

        if exit_bar < 0:
            exit_bar = 0

        pnl = p['unrealized'][exit_bar]
        results.append({
            'signal_type': p['signal_type'],
            'trading_day': p['trading_day'],
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': exit_bar + 1,
            'exit_type': exit_type,
        })
    return pd.DataFrame(results)


def simulate_combined_exit(paths, triage_preds, retest_preds,
                           triage_bar=10, retest_obs_bar=10,
                           runner_hold=45, chopper_hold=20,
                           runner_thresh=0.6, loser_thresh=0.4,
                           retest_exit_thresh=0.6, stop_pts=15):
    """Combined strategy: triage sets hold target, retest triggers early exit,
    hard stop for protection."""
    triage_map = {}
    for _, row in triage_preds.iterrows():
        triage_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

    retest_map = {}
    for _, row in retest_preds.iterrows():
        if row['obs_bar'] == retest_obs_bar:
            retest_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

    results = []
    for p in paths:
        key = (p['trading_day'], p['signal_idx'])
        p_good = triage_map.get(key, 0.5)
        p_retest = retest_map.get(key, 0.5)

        # Triage sets the intended hold period
        if p_good >= runner_thresh:
            target_hold = runner_hold
        elif p_good < loser_thresh:
            target_hold = triage_bar
        else:
            target_hold = chopper_hold

        # Simulate bar by bar
        exit_bar = min(target_hold - 1, p['n_bars'] - 1)
        exit_type = 'triage_hold'

        for j in range(p['n_bars']):
            # Hard stop
            if stop_pts is not None:
                if p['mae'][j] >= stop_pts:
                    exit_bar = j
                    exit_type = 'stop'
                    break

            # Retest exit: if at obs_bar, retest is predicted, exit early
            if j == retest_obs_bar - 1 and p_retest >= retest_exit_thresh:
                # But only for non-runners
                if p_good < runner_thresh:
                    exit_bar = j
                    exit_type = 'retest_exit'
                    break

            # Reached target hold
            if j >= target_hold - 1:
                exit_bar = j
                exit_type = 'triage_hold'
                break

        pnl = p['unrealized'][exit_bar]
        results.append({
            'signal_type': p['signal_type'],
            'trading_day': p['trading_day'],
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': exit_bar + 1,
            'exit_type': exit_type,
            'p_good': p_good,
            'p_retest': p_retest,
        })
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# MODEL FACTORIES
# ──────────────────────────────────────────────────────────────────────

def make_xgb(spw=1.0):
    return XGBClassifier(
        max_depth=2, learning_rate=0.05, n_estimators=200,
        min_child_weight=20, reg_alpha=1.0, reg_lambda=3.0,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, verbosity=0,
    )

def make_lr(spw=1.0):
    w = {0: 1.0, 1: spw}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=1.0, class_weight=w, max_iter=1000,
                                  random_state=42)),
    ])


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("TRADE MANAGEMENT v2 — structural exit models")
    print("=" * W)

    df_1min, daily = load_data()
    print(f"\nLoaded {len(df_1min):,} 1-min bars → {len(daily)} daily bars")

    checkpoints = [15, 30, 60, 90, 120, 150, 210, 270, 330]

    print("\nBuilding range predictions...")
    static_oos = _generate_oos_predictions(daily, 'daily', n_folds=5,
                                           min_train_days=100)
    static_oos = static_oos.rename(columns={
        'pred_range_high': 'static_high', 'pred_range_low': 'static_low',
    })
    trainer = ConditionalRangeTrainer()
    cond_oos = trainer.walk_forward_evaluate(df_1min, daily, verbose=False)
    rth = build_bar_level_predictions(df_1min, daily, cond_oos, static_oos)

    rolling_scores = build_rolling_trend_scores(df_1min, checkpoints)

    sigs_all = find_fade_proximity_signals(rth, 'cond', 10)
    sigs_h3 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 3.0)
    print(f"  {len(sigs_h3)} signals (heuristic < 3)")

    MAX_HOLD = 60
    paths = extract_trade_paths(sigs_h3, rth, max_hold=MAX_HOLD)
    print(f"  {len(paths)} trade paths extracted")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: EARLY TRIAGE MODEL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: EARLY TRIAGE — classify trade quality from first N bars")
    print("=" * W)

    for triage_bar in [5, 10, 15]:
        triage_data = build_triage_dataset(paths, triage_bar=triage_bar)
        good_rate = triage_data['good_trade'].mean()
        cats = triage_data['category'].value_counts()
        print(f"\n  Triage at bar {triage_bar}: {len(triage_data)} trades, "
              f"good_trade={good_rate:.0%}")
        print(f"    Categories: {dict(cats)}")

    # Use bar 10 as primary
    triage_data = build_triage_dataset(paths, triage_bar=10)

    print(f"\n  Walk-forward triage models (target=good_trade, triage_bar=10):")
    for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
        print(f"\n    {name}:")
        oos = walk_forward_signal_level(triage_data, TRIAGE_FEATURES,
                                        'good_trade', factory)
        if len(oos) > 0:
            auc = roc_auc_score(oos['good_trade'], oos['p_pos'])
            print(f"    Overall AUC: {auc:.3f}")

    # Also try bar 5 (earlier decision)
    triage_data_5 = build_triage_dataset(paths, triage_bar=5)
    print(f"\n  Walk-forward triage at bar 5:")
    for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
        print(f"\n    {name}:")
        oos = walk_forward_signal_level(triage_data_5, TRIAGE_FEATURES,
                                        'good_trade', factory)
        if len(oos) > 0:
            auc = roc_auc_score(oos['good_trade'], oos['p_pos'])
            print(f"    Overall AUC: {auc:.3f}")

    # Save best triage predictions for exit simulation
    triage_oos = walk_forward_signal_level(triage_data, TRIAGE_FEATURES,
                                           'good_trade', make_xgb)

    # ══════════════════════════════════════════════════════════════════
    # PART 2: REGIME CHANGE MODEL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: REGIME CHANGE — predict calm→trend transition")
    print("=" * W)

    regime_data = build_regime_change_dataset(df_1min, rolling_scores, checkpoints)
    rise_rate = regime_data['score_rises'].mean()
    print(f"\n  {len(regime_data)} checkpoint-level observations (current score < 3)")
    print(f"  Score rises to ≥4 by EOD: {rise_rate:.0%}")

    print(f"\n  Walk-forward regime change model:")
    for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
        print(f"\n    {name}:")
        oos = walk_forward_signal_level(regime_data, REGIME_FEATURES,
                                        'score_rises', factory)
        if len(oos) > 0:
            auc = roc_auc_score(oos['score_rises'], oos['p_pos'])
            print(f"    Overall AUC: {auc:.3f}")

    regime_oos = walk_forward_signal_level(regime_data, REGIME_FEATURES,
                                           'score_rises', make_xgb)

    # ══════════════════════════════════════════════════════════════════
    # PART 3: BOUNDARY RETEST MODEL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: BOUNDARY RETEST — predict P(price retests faded boundary)")
    print("=" * W)

    retest_data = build_retest_dataset(paths, eval_bars=30, retest_proximity=5)
    for obs_bar in [5, 10, 15]:
        sub = retest_data[retest_data['obs_bar'] == obs_bar]
        if len(sub) > 0:
            print(f"  obs_bar={obs_bar}: {len(sub)} observations, "
                  f"retest_rate={sub['retests'].mean():.0%}")

    print(f"\n  Walk-forward retest models:")
    for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
        print(f"\n    {name}:")
        oos = walk_forward_signal_level(retest_data, RETEST_FEATURES,
                                        'retests', factory)
        if len(oos) > 0:
            auc = roc_auc_score(oos['retests'], oos['p_pos'])
            print(f"    Overall AUC: {auc:.3f}")

    retest_oos = walk_forward_signal_level(retest_data, RETEST_FEATURES,
                                           'retests', make_xgb)

    # ══════════════════════════════════════════════════════════════════
    # PART 4: EXIT SIMULATION — compare all strategies
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: EXIT SIMULATION — all strategies on OOS data")
    print("=" * W)

    # Restrict to OOS days
    if len(triage_oos) > 0:
        oos_days = set(triage_oos['trading_day'].unique())
    else:
        oos_days = set(pd.DataFrame([{'trading_day': p['trading_day']} for p in paths])['trading_day'].unique())
    oos_paths = [p for p in paths if p['trading_day'] in oos_days]
    oos_sigs = sigs_h3[sigs_h3['trading_day'].isin(oos_days)]

    print(f"\n  OOS: {len(oos_paths)} paths, {len(oos_days)} days")
    print(f"\n  {'Strategy':<50} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18} {'bars':>6}")

    # Baseline: fixed 30
    pnl_30 = np.array([p['unrealized'][min(29, p['n_bars'] - 1)] for p in oos_paths])
    m30, lo30, hi30 = bootstrap_ci(pnl_30)
    print(f"  {'Fixed 30-bar hold':<50} {len(pnl_30):>5} "
          f"{(pnl_30 > 0).mean():>5.0%} {m30:>+7.1f}  [{lo30:>+7.1f}, {hi30:>+7.1f}] {'30':>6}")

    # Baseline: stop+target
    st = evaluate_stop_target(oos_sigs, rth, 15, target_mode='midpoint')
    fade_st = st[st['signal_type'].isin(['fade_high', 'fade_low'])]
    if len(fade_st) > 0:
        mst, lost, hist = bootstrap_ci(fade_st['pnl'].values)
        print(f"  {'15pt stop + midpoint target':<50} {len(fade_st):>5} "
              f"{fade_st['win'].mean():>5.0%} {mst:>+7.1f}  [{lost:>+7.1f}, {hist:>+7.1f}] {'—':>6}")

    # Oracle
    pnl_oracle = np.array([p['unrealized'][np.argmax(p['unrealized'])] for p in oos_paths])
    mor, loor, hior = bootstrap_ci(pnl_oracle)
    print(f"  {'Oracle (exit at peak PnL)':<50} {len(pnl_oracle):>5} "
          f"{(pnl_oracle > 0).mean():>5.0%} {mor:>+7.1f}  [{loor:>+7.1f}, {hior:>+7.1f}] {'—':>6}")

    # ── Triage strategies ──
    if len(triage_oos) > 0:
        print()
        # Default triage
        res = simulate_triage_exit(oos_paths, triage_oos, triage_bar=10)
        if len(res) > 0:
            m, lo, hi = bootstrap_ci(res['pnl'].values)
            avg_bars = res['bars_held'].mean()
            exits = res['exit_type'].value_counts(normalize=True)
            print(f"  {'Triage (runner:45/chop:20/loser:exit@10)':<50} {len(res):>5} "
                  f"{res['win'].mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
                  f"{avg_bars:>5.0f}b")
            print(f"    Exit breakdown: {dict(exits)}")

        # Grid search over parameters
        print(f"\n  Triage parameter grid:")
        print(f"  {'runner_hold':>11} {'chop_hold':>10} {'runner_t':>8} {'loser_t':>8} "
              f"{'n':>5} {'WR':>6} {'PnL':>7} {'bars':>6}")
        best_pnl = -999
        best_params = {}
        for runner_hold in [30, 40, 50]:
            for chopper_hold in [15, 20, 25]:
                for runner_thresh in [0.55, 0.6, 0.65]:
                    for loser_thresh in [0.35, 0.4, 0.45]:
                        res = simulate_triage_exit_parametric(
                            oos_paths, triage_oos, triage_bar=10,
                            runner_hold=runner_hold, chopper_hold=chopper_hold,
                            runner_thresh=runner_thresh, loser_thresh=loser_thresh)
                        if len(res) > 0:
                            avg_pnl = res['pnl'].mean()
                            if avg_pnl > best_pnl:
                                best_pnl = avg_pnl
                                best_params = {
                                    'runner_hold': runner_hold,
                                    'chopper_hold': chopper_hold,
                                    'runner_thresh': runner_thresh,
                                    'loser_thresh': loser_thresh,
                                }

        if best_params:
            res = simulate_triage_exit_parametric(oos_paths, triage_oos, **best_params)
            m, lo, hi = bootstrap_ci(res['pnl'].values)
            print(f"  Best: rh={best_params['runner_hold']}, ch={best_params['chopper_hold']}, "
                  f"rt={best_params['runner_thresh']}, lt={best_params['loser_thresh']}")
            print(f"  {'Triage (best params)':<50} {len(res):>5} "
                  f"{res['win'].mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
                  f"{res['bars_held'].mean():>5.0f}b")

    # ── Retest strategies ──
    if len(retest_oos) > 0:
        print()
        for obs_bar in [5, 10]:
            for exit_thresh in [0.5, 0.6, 0.7]:
                res = simulate_retest_exit(oos_paths, retest_oos,
                                           obs_bar=obs_bar, max_hold=30,
                                           exit_thresh=exit_thresh)
                if len(res) > 0:
                    m, lo, hi = bootstrap_ci(res['pnl'].values)
                    avg_bars = res['bars_held'].mean()
                    label = f"Retest exit (obs@{obs_bar}, exit@P≥{exit_thresh})"
                    print(f"  {label:<50} {len(res):>5} "
                          f"{res['win'].mean():>5.0%} {m:>+7.1f}  "
                          f"[{lo:>+7.1f}, {hi:>+7.1f}] {avg_bars:>5.0f}b")

    # ── Regime strategies ──
    if len(regime_oos) > 0:
        print()
        for exit_thresh in [0.3, 0.4, 0.5]:
            res = simulate_regime_exit(oos_paths, regime_oos, rolling_scores,
                                       checkpoints, exit_thresh=exit_thresh,
                                       max_hold=30)
            if len(res) > 0:
                m, lo, hi = bootstrap_ci(res['pnl'].values)
                avg_bars = res['bars_held'].mean()
                label = f"Regime exit (exit@P≥{exit_thresh})"
                n_exit = (res['exit_type'] == 'regime_exit').sum()
                print(f"  {label:<50} {len(res):>5} "
                      f"{res['win'].mean():>5.0%} {m:>+7.1f}  "
                      f"[{lo:>+7.1f}, {hi:>+7.1f}] {avg_bars:>5.0f}b"
                      f"  ({n_exit} regime exits)")

    # ── Combined strategy ──
    if len(triage_oos) > 0 and len(retest_oos) > 0:
        print()
        for runner_hold in [40, 50]:
            for retest_thresh in [0.5, 0.6]:
                res = simulate_combined_exit(
                    oos_paths, triage_oos, retest_oos,
                    triage_bar=10, retest_obs_bar=10,
                    runner_hold=runner_hold, chopper_hold=20,
                    runner_thresh=0.6, loser_thresh=0.4,
                    retest_exit_thresh=retest_thresh, stop_pts=15)
                if len(res) > 0:
                    m, lo, hi = bootstrap_ci(res['pnl'].values)
                    avg_bars = res['bars_held'].mean()
                    exits = res['exit_type'].value_counts()
                    label = f"Combined (rh={runner_hold},retest@{retest_thresh},15pt stop)"
                    print(f"  {label:<50} {len(res):>5} "
                          f"{res['win'].mean():>5.0%} {m:>+7.1f}  "
                          f"[{lo:>+7.1f}, {hi:>+7.1f}] {avg_bars:>5.0f}b")
                    print(f"    Exits: {dict(exits)}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: DOES TRIAGE ACTUALLY DISCRIMINATE?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: TRIAGE CALIBRATION — does the model discriminate?")
    print("=" * W)

    if len(triage_oos) > 0:
        triage_with_pnl = triage_oos.merge(
            triage_data[['trading_day', 'signal_idx', 'pnl_30', 'category', 'mfe_total']],
            on=['trading_day', 'signal_idx'], how='left')

        # Bin by P(good) and check actual outcomes
        triage_with_pnl['p_bin'] = pd.cut(triage_with_pnl['p_pos'],
                                           bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
        print(f"\n  {'P(good) bin':<15} {'n':>5} {'Act WR':>8} {'Avg PnL':>9} "
              f"{'Avg MFE':>9} {'% Runner':>9} {'% Loser':>9}")
        for bin_label, grp in triage_with_pnl.groupby('p_bin', observed=True):
            if len(grp) < 5:
                continue
            act_wr = (grp['pnl_30'] > 0).mean()
            avg_pnl = grp['pnl_30'].mean()
            avg_mfe = grp['mfe_total'].mean()
            pct_runner = (grp['category'] == 'runner').mean()
            pct_loser = (grp['category'] == 'loser').mean()
            print(f"  {str(bin_label):<15} {len(grp):>5} {act_wr:>7.0%} {avg_pnl:>+9.1f} "
                  f"{avg_mfe:>+9.1f} {pct_runner:>8.0%} {pct_loser:>8.0%}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
