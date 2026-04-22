"""Small-winner analysis + conventional wisdom trade management.

Prior finding: trades with 0 < PnL ≤ +5 at bar 30 show +5.3 PnL if held
to EOD vs +2.3 at bar 30. But this is the "undecided" group — noisy price
action, no clear momentum signal.

This script:
  Part 1: Profile small winners — what separates eventual runners from reversals?
  Part 2: New feature categories beyond momentum (volume structure, boundary
          integrity, path shape, range prediction evolution)
  Part 3: ML model on small winners with new features
  Part 4: Conventional wisdom approach — cut losers early, let winners ride
  Part 5: Complete trade management framework comparison
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
from scipy.signal import find_peaks

from vp.dalton_day_classifier import VolumeProfileBuilder
from strategies.range_predictor.conditional import ConditionalRangeTrainer
from strategies.range_predictor.analysis import _generate_oos_predictions
from sandbox.range_signal_analysis import load_data, build_bar_level_predictions
from sandbox.range_signal_analysis_v2 import (
    find_fade_proximity_signals,
    evaluate_stop_target,
)
from sandbox.range_dalton_dynamic import (
    build_rolling_trend_scores,
    filter_signals_by_rolling_score,
)
from sandbox.train_ptrend import bootstrap_ci
from sandbox.train_runner_mgmt import extract_full_paths
from sandbox.train_trade_mgmt_v2 import (
    build_triage_dataset,
    TRIAGE_FEATURES,
    walk_forward_signal_level,
)


# ──────────────────────────────────────────────────────────────────────
# SMALL-WINNER FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────

def build_small_winner_features(paths, rth, rolling_scores, eval_bar=30):
    """Build enriched features for trades at eval_bar, focusing on
    features that distinguish eventual runners from reversals among
    the "undecided" trades.

    Feature categories beyond momentum:
      1. Path shape: HOW did we get here? (path roughness, retest count)
      2. Volume structure: is volume confirming the fade?
      3. Boundary integrity: is the faded level still holding?
      4. VP shape: is the day's volume profile developing one-sided?
      5. Range prediction context: how much room is left?
    """
    checkpoints = sorted(next(iter(rolling_scores.values())).keys()) if rolling_scores else []
    rth_closes = rth['close'].values
    rth_highs = rth['high'].values
    rth_lows = rth['low'].values
    rth_volumes = rth['volume'].values if 'volume' in rth.columns else np.ones(len(rth))
    rth_tds = rth['trading_day'].values

    rows = []
    for p in paths:
        if p['n_bars'] < eval_bar + 5:
            continue

        n = p['n_bars']
        unr = p['unrealized']
        mfe_arr = p['mfe']
        mae_arr = p['mae']
        eb = eval_bar - 1  # 0-indexed

        current_pnl = unr[eb]
        pred_width = max(1.0, p['pred_high'] - p['pred_low'])
        entry_idx = p['entry_idx']
        td = p['trading_day']
        is_bull = p['is_bull']
        direction = 1.0 if is_bull else -1.0

        # ── Targets ──
        # Primary: does PnL improve by bar 60?
        pnl_60 = unr[min(59, n - 1)]
        improves_60 = 1 if pnl_60 > current_pnl else 0

        # EOD PnL
        pnl_eod = unr[-1]
        improves_eod = 1 if pnl_eod > current_pnl else 0

        # Runner: PnL at EOD ≥ +5 from entry
        runner_eod = 1 if pnl_eod >= 5.0 else 0

        # MFE after bar 30 extends by 3+
        future_mfe = mfe_arr[-1]
        current_mfe = mfe_arr[eb]
        mfe_extends = 1 if (future_mfe - current_mfe) >= 3.0 else 0

        # ── 1. MOMENTUM FEATURES (baseline) ──
        slope_5 = (unr[eb] - unr[max(0, eb - 4)]) / 5.0
        slope_10 = (unr[eb] - unr[max(0, eb - 9)]) / 10.0
        slope_15 = (unr[eb] - unr[max(0, eb - 14)]) / 15.0

        mfe_at_eval = mfe_arr[eb]
        mae_at_eval = mae_arr[eb]
        giveback = mfe_at_eval - current_pnl
        giveback_pct = giveback / max(0.5, mfe_at_eval) if mfe_at_eval > 0.5 else 0

        # ── 2. PATH SHAPE — how did we get to this PnL? ──
        # Path roughness: std of bar-to-bar PnL changes
        pnl_changes = np.diff(unr[:eb + 1])
        path_roughness = np.std(pnl_changes) if len(pnl_changes) > 1 else 0

        # How many times did PnL cross zero? (choppy = many crossings)
        if eb >= 2:
            signs = np.sign(unr[:eb + 1])
            zero_crossings = np.sum(np.abs(np.diff(signs)) > 0) / max(1, eb)
        else:
            zero_crossings = 0

        # Time spent positive vs negative (pnl > 0 fraction)
        pnl_positive_frac = np.sum(unr[:eb + 1] > 0) / (eb + 1)

        # Was there a deep drawdown that recovered?
        recovered_from_dd = 1 if mae_at_eval >= 3.0 and current_pnl > 0 else 0

        # PnL acceleration: is the recent slope steeper than the overall slope?
        overall_slope = current_pnl / max(1, eval_bar)
        if eb >= 5:
            recent_slope = slope_5
            acceleration = recent_slope - overall_slope
        else:
            acceleration = 0

        # ── 3. VOLUME STRUCTURE ──
        vols = p['volumes'][:eb + 1]
        closes = p['closes'][:eb + 1]

        # Volume trend: recent vs earlier
        if eb >= 10:
            vol_early = vols[:eb // 2].mean()
            vol_late = vols[eb // 2:].mean()
            vol_trend = vol_late / max(1, vol_early)
        else:
            vol_trend = 1.0

        # Volume on favorable bars vs adverse bars
        bar_returns = np.diff(closes) * direction
        if len(bar_returns) >= 5:
            fav_mask = bar_returns > 0
            adv_mask = bar_returns < 0
            vol_on_fav = vols[1:][fav_mask].mean() if fav_mask.any() else 0
            vol_on_adv = vols[1:][adv_mask].mean() if adv_mask.any() else 0
            vol_fav_adv_ratio = vol_on_fav / max(1, vol_on_adv)
        else:
            vol_fav_adv_ratio = 1.0

        # Volume spike: max volume bar relative to mean
        vol_spike = vols.max() / max(1, vols.mean())

        # ── 4. BOUNDARY INTEGRITY ──
        # How far is current price from the faded boundary?
        if is_bull:
            boundary = p['pred_low']
            dist_to_faded_boundary = p['closes'][eb] - boundary
        else:
            boundary = p['pred_high']
            dist_to_faded_boundary = boundary - p['closes'][eb]

        dist_to_boundary_norm = dist_to_faded_boundary / pred_width

        # How many times did price come back close to the boundary (retests)?
        retest_count = 0
        retest_prox = 3.0  # within 3 pts
        for j in range(1, eb + 1):
            if is_bull:
                if p['lows'][j] <= boundary + retest_prox:
                    retest_count += 1
            else:
                if p['highs'][j] >= boundary - retest_prox:
                    retest_count += 1

        # Did price ever break through the boundary after entry?
        boundary_broken = 0
        for j in range(1, eb + 1):
            if is_bull and p['lows'][j] < boundary:
                boundary_broken = 1
                break
            elif not is_bull and p['highs'][j] > boundary:
                boundary_broken = 1
                break

        # ── 5. VP SHAPE (from session start to eval_bar) ──
        # Get all session bars from RTH start to current point
        session_start = entry_idx - int(p['bar_offset_entry'])
        if session_start < 0:
            session_start = entry_idx
        session_end = entry_idx + eb + 1

        session_prices = rth_closes[max(0, session_start):min(session_end, len(rth_closes))]
        session_volumes = rth_volumes[max(0, session_start):min(session_end, len(rth_volumes))]

        if len(session_prices) >= 20:
            vp_builder = VolumeProfileBuilder(bin_size=1.0)
            bin_centers, per_min_vbp = vp_builder.build_minute_vbp_matrix(
                session_prices, session_volumes)
            cum_vbp = per_min_vbp.sum(axis=0)

            poc = vp_builder.compute_poc(cum_vbp, bin_centers)
            va_lo, va_hi = vp_builder.compute_va70(cum_vbp, bin_centers)
            entropy = vp_builder.compute_entropy(cum_vbp)

            session_range = session_prices.max() - session_prices.min()
            va_width_rel = (va_hi - va_lo) / max(1, session_range)

            # Is POC above or below the midpoint? (skew)
            midpoint = (session_prices.max() + session_prices.min()) / 2
            poc_skew = (poc - midpoint) / max(1, session_range)

            # Is price above or below the POC?
            price_vs_poc = (p['closes'][eb] - poc) / max(1, session_range)
        else:
            va_width_rel = 0.5
            poc_skew = 0
            price_vs_poc = 0
            entropy = 3.0

        # ── 6. RANGE CONTEXT ──
        session_hi = p['highs'][:eb + 1].max()
        session_lo = p['lows'][:eb + 1].min()
        range_consumed = (session_hi - session_lo) / pred_width

        # How far is price from the opposite boundary?
        if is_bull:
            room_to_target = (p['pred_high'] - p['closes'][eb]) / pred_width
        else:
            room_to_target = (p['closes'][eb] - p['pred_low']) / pred_width

        # ── 7. TREND SCORE ──
        bar_offset = p['offsets'][eb] if eb < len(p['offsets']) else p['bar_offset_entry'] + eval_bar
        day_scores = rolling_scores.get(td, {})
        trend_score = 0
        for cp in checkpoints:
            if cp <= bar_offset:
                trend_score = day_scores.get(cp, 0)

        # Entry time
        entry_time = p['bar_offset_entry'] / 390.0
        session_elapsed = bar_offset / 390.0

        rows.append({
            'trading_day': td,
            'signal_idx': p['signal_idx'],
            'is_bull': is_bull,
            # Targets
            'current_pnl': current_pnl,
            'pnl_60': pnl_60,
            'pnl_eod': pnl_eod,
            'improves_60': improves_60,
            'improves_eod': improves_eod,
            'runner_eod': runner_eod,
            'mfe_extends': mfe_extends,
            # 1. Momentum (baseline)
            'slope_5': slope_5,
            'slope_10': slope_10,
            'slope_15': slope_15,
            'mfe_at_eval': mfe_at_eval,
            'mae_at_eval': mae_at_eval,
            'giveback': giveback,
            'giveback_pct': giveback_pct,
            # 2. Path shape
            'path_roughness': path_roughness,
            'zero_crossings': zero_crossings,
            'pnl_positive_frac': pnl_positive_frac,
            'recovered_from_dd': recovered_from_dd,
            'acceleration': acceleration,
            # 3. Volume
            'vol_trend': vol_trend,
            'vol_fav_adv_ratio': vol_fav_adv_ratio,
            'vol_spike': vol_spike,
            # 4. Boundary
            'dist_to_boundary_norm': dist_to_boundary_norm,
            'retest_count': retest_count,
            'boundary_broken': boundary_broken,
            # 5. VP shape
            'va_width_rel': va_width_rel,
            'poc_skew': poc_skew,
            'price_vs_poc': price_vs_poc,
            'entropy': entropy,
            # 6. Range
            'range_consumed': range_consumed,
            'room_to_target': room_to_target,
            # 7. Context
            'trend_score': trend_score,
            'entry_time': entry_time,
            'session_elapsed': session_elapsed,
            'direction': direction,
        })

    return pd.DataFrame(rows)


MOMENTUM_FEATURES = [
    'slope_5', 'slope_10', 'slope_15',
    'mfe_at_eval', 'mae_at_eval', 'giveback', 'giveback_pct',
]

PATH_FEATURES = [
    'path_roughness', 'zero_crossings', 'pnl_positive_frac',
    'recovered_from_dd', 'acceleration',
]

VOLUME_FEATURES = [
    'vol_trend', 'vol_fav_adv_ratio', 'vol_spike',
]

BOUNDARY_FEATURES = [
    'dist_to_boundary_norm', 'retest_count', 'boundary_broken',
]

VP_FEATURES = [
    'va_width_rel', 'poc_skew', 'price_vs_poc', 'entropy',
]

RANGE_FEATURES = [
    'range_consumed', 'room_to_target',
]

CONTEXT_FEATURES = [
    'trend_score', 'entry_time', 'session_elapsed', 'direction',
]

ALL_FEATURES = (MOMENTUM_FEATURES + PATH_FEATURES + VOLUME_FEATURES +
                BOUNDARY_FEATURES + VP_FEATURES + RANGE_FEATURES + CONTEXT_FEATURES)


# ──────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────

def make_xgb(spw=1.0):
    return XGBClassifier(
        max_depth=2, learning_rate=0.05, n_estimators=200,
        min_child_weight=15, reg_alpha=1.0, reg_lambda=3.0,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='logloss',
        random_state=42, verbosity=0,
    )

def make_lr(spw=1.0):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.5, class_weight={0: 1.0, 1: spw},
                                  max_iter=1000, random_state=42)),
    ])


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("SMALL-WINNER ANALYSIS + CONVENTIONAL WISDOM TRADE MANAGEMENT")
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

    paths = extract_full_paths(sigs_h3, rth)
    print(f"  {len(paths)} full-day paths")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: SMALL-WINNER PROFILING
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: SMALL-WINNER PROFILING (0 < PnL ≤ +5 at bar 30)")
    print("=" * W)

    small_winners = [p for p in paths if p['n_bars'] >= 31
                     and 0 < p['unrealized'][29] <= 5]
    print(f"\n  {len(small_winners)} small winners")

    # Split: eventual runners vs reversals
    eventual_runners = [p for p in small_winners if p['unrealized'][-1] >= 5]
    eventual_reversals = [p for p in small_winners if p['unrealized'][-1] < 0]
    eventual_flat = [p for p in small_winners
                     if 0 <= p['unrealized'][-1] < 5]

    print(f"  Eventual runners (EOD PnL ≥ +5):  {len(eventual_runners)} ({len(eventual_runners)/len(small_winners):.0%})")
    print(f"  Eventual flat (0 ≤ EOD < +5):      {len(eventual_flat)} ({len(eventual_flat)/len(small_winners):.0%})")
    print(f"  Eventual reversals (EOD PnL < 0):  {len(eventual_reversals)} ({len(eventual_reversals)/len(small_winners):.0%})")

    # Profile differences at bar 30
    print(f"\n  Characteristics at bar 30:")
    print(f"  {'Metric':<30} {'Runners':>10} {'Flat':>10} {'Reversals':>10}")

    for metric_name, metric_fn in [
        ('Avg PnL at bar 30', lambda p: p['unrealized'][29]),
        ('MFE at bar 30', lambda p: p['mfe'][29]),
        ('MAE at bar 30', lambda p: p['mae'][29]),
        ('PnL slope (last 10)', lambda p: (p['unrealized'][29] - p['unrealized'][19]) / 10),
        ('Path roughness', lambda p: np.std(np.diff(p['unrealized'][:30]))),
        ('Zero crossings', lambda p: np.sum(np.abs(np.diff(np.sign(p['unrealized'][:30]))) > 0) / 29),
        ('Vol trend (late/early)', lambda p: p['volumes'][15:30].mean() / max(1, p['volumes'][:15].mean())),
        ('Boundary dist (norm)', lambda p: (p['closes'][29] - p['pred_low']) / max(1, p['pred_high'] - p['pred_low']) if p['is_bull'] else (p['pred_high'] - p['closes'][29]) / max(1, p['pred_high'] - p['pred_low'])),
        ('Range consumed', lambda p: (p['highs'][:30].max() - p['lows'][:30].min()) / max(1, p['pred_high'] - p['pred_low'])),
        ('Entry time (norm)', lambda p: p['bar_offset_entry'] / 390.0),
        ('PnL at EOD', lambda p: p['unrealized'][-1]),
    ]:
        vals_r = [metric_fn(p) for p in eventual_runners] if eventual_runners else [0]
        vals_f = [metric_fn(p) for p in eventual_flat] if eventual_flat else [0]
        vals_v = [metric_fn(p) for p in eventual_reversals] if eventual_reversals else [0]
        print(f"  {metric_name:<30} {np.mean(vals_r):>+10.2f} {np.mean(vals_f):>+10.2f} "
              f"{np.mean(vals_v):>+10.2f}")

    # PnL curve for each group
    print(f"\n  PnL evolution by eventual outcome:")
    print(f"  {'Bar':<8} {'Runners':>10} {'Flat':>10} {'Reversals':>10}")
    for bar in [10, 20, 30, 45, 60, 90, 120, 180]:
        for group_name, group in [('Runners', eventual_runners),
                                   ('Flat', eventual_flat),
                                   ('Reversals', eventual_reversals)]:
            pass  # compute below
        r_pnl = np.mean([p['unrealized'][min(bar - 1, p['n_bars'] - 1)]
                         for p in eventual_runners]) if eventual_runners else np.nan
        f_pnl = np.mean([p['unrealized'][min(bar - 1, p['n_bars'] - 1)]
                         for p in eventual_flat]) if eventual_flat else np.nan
        v_pnl = np.mean([p['unrealized'][min(bar - 1, p['n_bars'] - 1)]
                         for p in eventual_reversals]) if eventual_reversals else np.nan
        print(f"  {bar:<8} {r_pnl:>+10.1f} {f_pnl:>+10.1f} {v_pnl:>+10.1f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: FEATURE ABLATION — which feature groups help?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: ML ON SMALL WINNERS — feature ablation")
    print("=" * W)

    print("\nBuilding enriched features for ALL trades at bar 30...")
    all_features_df = build_small_winner_features(paths, rth, rolling_scores, eval_bar=30)
    print(f"  {len(all_features_df)} trades with features")

    # Filter to small winners only
    sw_data = all_features_df[(all_features_df['current_pnl'] > 0) &
                               (all_features_df['current_pnl'] <= 5)].copy()
    print(f"  {len(sw_data)} small winners")
    print(f"  improves_eod rate: {sw_data['improves_eod'].mean():.0%}")
    print(f"  runner_eod rate: {sw_data['runner_eod'].mean():.0%}")

    # Test different feature sets
    feature_sets = [
        ('Momentum only', MOMENTUM_FEATURES),
        ('Path shape only', PATH_FEATURES),
        ('Volume only', VOLUME_FEATURES),
        ('Boundary only', BOUNDARY_FEATURES),
        ('VP shape only', VP_FEATURES),
        ('Range + context', RANGE_FEATURES + CONTEXT_FEATURES),
        ('Momentum + path', MOMENTUM_FEATURES + PATH_FEATURES),
        ('All new (no momentum)', PATH_FEATURES + VOLUME_FEATURES +
         BOUNDARY_FEATURES + VP_FEATURES + RANGE_FEATURES + CONTEXT_FEATURES),
        ('ALL features', ALL_FEATURES),
    ]

    targets = [
        ('improves_eod', 'PnL improves to EOD'),
        ('runner_eod', 'Becomes runner (EOD PnL ≥ +5)'),
        ('mfe_extends', 'MFE extends +3 pts'),
    ]

    for target_col, target_name in targets:
        print(f"\n  Target: {target_name} (base rate={sw_data[target_col].mean():.0%})")
        print(f"  {'Feature set':<35} {'AUC':>6}")

        for feat_name, feat_cols in feature_sets:
            # Quick walk-forward (3 folds to save time given small n)
            days = sorted(sw_data['trading_day'].unique())
            n_days = len(days)
            fold_size = n_days // 4

            oos_y, oos_p = [], []
            for fold in range(4):
                test_start = fold * fold_size
                test_end = test_start + fold_size if fold < 3 else n_days
                test_days = set(days[test_start:test_end])
                train_days = set(days[:test_start])

                if len(train_days) < 80:
                    continue

                train = sw_data[sw_data['trading_day'].isin(train_days)]
                test = sw_data[sw_data['trading_day'].isin(test_days)]

                if len(train) < 20 or len(test) < 5:
                    continue

                n_pos = train[target_col].sum()
                n_neg = len(train) - n_pos
                spw = n_neg / max(1, n_pos)

                model = make_xgb(spw)
                model.fit(train[feat_cols].values, train[target_col].values)
                p = model.predict_proba(test[feat_cols].values)[:, 1]
                oos_y.extend(test[target_col].values)
                oos_p.extend(p)

            if len(oos_y) >= 20:
                try:
                    auc = roc_auc_score(oos_y, oos_p)
                    print(f"  {feat_name:<35} {auc:>6.3f}")
                except ValueError:
                    print(f"  {feat_name:<35} {'n/a':>6}")
            else:
                print(f"  {feat_name:<35} {'n/a':>6}")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: ML ON ALL TRADES (not just small winners)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: ML ON ALL TRADES — predict outcome at bar 30")
    print("=" * W)

    print(f"\n  {len(all_features_df)} total trades")
    for target_col, target_name in targets:
        rate = all_features_df[target_col].mean()
        print(f"\n  Target: {target_name} (base rate={rate:.0%})")

        for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
            print(f"    {name}:")
            oos = walk_forward_signal_level(all_features_df, ALL_FEATURES,
                                            target_col, factory,
                                            n_folds=5, min_train_days=100)
            if len(oos) > 0:
                auc = roc_auc_score(oos[target_col], oos['p_pos'])
                print(f"    Overall AUC: {auc:.3f}")

                # Calibration
                oos_with_pnl = oos.merge(
                    all_features_df[['trading_day', 'signal_idx', 'current_pnl',
                                     'pnl_eod', 'pnl_60']],
                    on=['trading_day', 'signal_idx'], how='left')
                oos_with_pnl['p_bin'] = pd.cut(oos_with_pnl['p_pos'],
                                                bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
                print(f"    {'P bin':<12} {'n':>5} {'Act rate':>9} "
                      f"{'PnL@30':>8} {'PnL@60':>8} {'PnL EOD':>8}")
                for b, g in oos_with_pnl.groupby('p_bin', observed=True):
                    if len(g) < 3:
                        continue
                    print(f"    {str(b):<12} {len(g):>5} {g[target_col].mean():>8.0%} "
                          f"{g['current_pnl'].mean():>+8.1f} "
                          f"{g['pnl_60'].mean():>+8.1f} "
                          f"{g['pnl_eod'].mean():>+8.1f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: CONVENTIONAL WISDOM — cut losers, let winners ride
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: CONVENTIONAL WISDOM — cut losers early, trail winners")
    print("=" * W)

    # Build triage predictions (from v2, bar 10)
    print("\n  Training triage model (bar 10)...")
    from sandbox.train_trade_mgmt import extract_trade_paths as extract_60
    paths_60 = extract_60(sigs_h3, rth, max_hold=60)

    triage_data = build_triage_dataset(paths_60, triage_bar=10)
    triage_oos = walk_forward_signal_level(triage_data, TRIAGE_FEATURES,
                                           'good_trade', make_xgb)

    if len(triage_oos) > 0:
        triage_auc = roc_auc_score(triage_oos['good_trade'], triage_oos['p_pos'])
        print(f"  Triage AUC: {triage_auc:.3f}")

        # Map triage predictions to full paths
        triage_map = {}
        for _, row in triage_oos.iterrows():
            triage_map[(row['trading_day'], row['signal_idx'])] = row['p_pos']

        oos_days = set(triage_oos['trading_day'].unique())
        oos_paths = [p for p in paths if p['trading_day'] in oos_days]
        print(f"  OOS: {len(oos_paths)} paths ({len(oos_days)} days)")

        print(f"\n  {'Strategy':<55} {'n':>5} {'WR':>6} {'PnL':>7}  "
              f"{'95% CI':>18} {'bars':>6}")

        # Baseline
        pnl_30 = np.array([p['unrealized'][min(29, p['n_bars'] - 1)] for p in oos_paths])
        m, lo, hi = bootstrap_ci(pnl_30)
        print(f"  {'Exit at bar 30 (baseline)':<55} {len(pnl_30):>5} "
              f"{(pnl_30 > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] {'30':>6}")

        # Conventional wisdom strategies
        strategies = {}

        # Strategy 1: Cut losers at bar 10, hold others to bar 30
        pnls, bars = [], []
        for p in oos_paths:
            key = (p['trading_day'], p['signal_idx'])
            p_good = triage_map.get(key, 0.5)
            if p_good < 0.35:
                eb = min(9, p['n_bars'] - 1)  # cut at bar 10
            else:
                eb = min(29, p['n_bars'] - 1)  # hold to 30
            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['Cut losers@10'] = pnls
        print(f"  {'Cut losers (P<0.35) at bar 10, others hold 30':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

        # Strategy 2: Cut losers at 10, trail winners (MFE trail 5pt)
        pnls, bars = [], []
        for p in oos_paths:
            key = (p['trading_day'], p['signal_idx'])
            p_good = triage_map.get(key, 0.5)
            if p_good < 0.35:
                eb = min(9, p['n_bars'] - 1)
            elif p_good >= 0.6:
                # Winner: trail 5pt from MFE, max 90 bars
                eb = min(89, p['n_bars'] - 1)
                for j in range(10, eb + 1):
                    if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                        eb = j
                        break
            else:
                eb = min(29, p['n_bars'] - 1)
            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['Cut+Trail'] = pnls
        print(f"  {'Cut losers@10, trail winners(P≥0.6), others@30':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

        # Strategy 3: Three-tier based on PnL at bar 30
        pnls, bars = [], []
        for p in oos_paths:
            if p['n_bars'] < 31:
                pnls.append(p['unrealized'][-1])
                bars.append(p['n_bars'])
                continue
            pnl_30 = p['unrealized'][29]
            if pnl_30 > 10:
                # Big winner: trail 5pt, max 120 bars
                eb = min(119, p['n_bars'] - 1)
                for j in range(30, eb + 1):
                    if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                        eb = j
                        break
            elif pnl_30 <= 0:
                eb = 29  # Loser: exit now
            else:
                eb = 29  # Small/moderate winner: exit at 30
            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['PnL-based tier'] = pnls
        print(f"  {'PnL tier: big winners trail, rest exit@30':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

        # Strategy 4: Triage + PnL conditional
        pnls, bars = [], []
        for p in oos_paths:
            key = (p['trading_day'], p['signal_idx'])
            p_good = triage_map.get(key, 0.5)
            if p_good < 0.35:
                eb = min(9, p['n_bars'] - 1)  # Cut early loser
            elif p['n_bars'] >= 31 and p['unrealized'][29] > 10 and p_good >= 0.6:
                # High-confidence big winner: trail
                eb = min(89, p['n_bars'] - 1)
                for j in range(30, eb + 1):
                    if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                        eb = j
                        break
            else:
                eb = min(29, p['n_bars'] - 1)  # Default: hold 30
            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['Triage+PnL'] = pnls
        print(f"  {'Triage cut@10 + big winner trail + default@30':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

        # Strategy 5: 15pt hard stop + exit at 30
        pnls, bars = [], []
        for p in oos_paths:
            eb = min(29, p['n_bars'] - 1)
            for j in range(p['n_bars']):
                if p['mae'][j] >= 15:
                    eb = j
                    break
                if j >= 29:
                    break
            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['Stop15+30'] = pnls
        print(f"  {'15pt stop + exit at bar 30':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

        # Strategy 6: Full framework - triage cut + 15pt stop + conditional hold
        pnls, bars = [], []
        for p in oos_paths:
            key = (p['trading_day'], p['signal_idx'])
            p_good = triage_map.get(key, 0.5)

            if p_good < 0.35:
                eb = min(9, p['n_bars'] - 1)
            else:
                # Default hold to 30 with 15pt stop
                eb = min(29, p['n_bars'] - 1)
                stopped = False
                for j in range(min(30, p['n_bars'])):
                    if p['mae'][j] >= 15:
                        eb = j
                        stopped = True
                        break

                # If at bar 30 with big profit and high triage → trail
                if not stopped and p['n_bars'] >= 31 and p_good >= 0.6:
                    pnl_at_30 = p['unrealized'][29]
                    if pnl_at_30 > 10:
                        # Trail 5pt from MFE, max 60 more bars
                        trail_end = min(89, p['n_bars'] - 1)
                        for j in range(30, trail_end + 1):
                            if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                                eb = j
                                break
                        else:
                            eb = trail_end

            pnls.append(p['unrealized'][eb])
            bars.append(eb + 1)
        pnls = np.array(pnls)
        m, lo, hi = bootstrap_ci(pnls)
        strategies['Full framework'] = pnls
        print(f"  {'FULL: triage cut + 15pt stop + winner trail':<55} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] "
              f"{np.mean(bars):>5.0f}b")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: WHERE DOES VALUE COME FROM?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: VALUE DECOMPOSITION — where does each component help?")
    print("=" * W)

    if len(triage_oos) > 0:
        # Break down by triage bucket
        print(f"\n  By triage P(good) bucket:")
        print(f"  {'Bucket':<15} {'n':>5} {'Baseline@30':>12} {'Cut@10':>12} {'Trail':>12}")

        for lo_t, hi_t, label in [
            (0, 0.35, 'P < 0.35'),
            (0.35, 0.5, '0.35-0.50'),
            (0.5, 0.65, '0.50-0.65'),
            (0.65, 1.0, 'P ≥ 0.65'),
        ]:
            bucket = [p for p in oos_paths
                      if lo_t <= triage_map.get((p['trading_day'], p['signal_idx']), 0.5) < hi_t]
            if len(bucket) < 5:
                continue

            base_pnl = np.mean([p['unrealized'][min(29, p['n_bars'] - 1)] for p in bucket])
            cut_pnl = np.mean([p['unrealized'][min(9, p['n_bars'] - 1)] for p in bucket])

            trail_pnls = []
            for p in bucket:
                eb = min(89, p['n_bars'] - 1)
                for j in range(10, eb + 1):
                    if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                        eb = j
                        break
                trail_pnls.append(p['unrealized'][eb])
            trail_pnl = np.mean(trail_pnls)

            print(f"  {label:<15} {len(bucket):>5} {base_pnl:>+12.1f} "
                  f"{cut_pnl:>+12.1f} {trail_pnl:>+12.1f}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
