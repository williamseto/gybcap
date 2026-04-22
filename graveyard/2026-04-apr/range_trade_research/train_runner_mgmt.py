"""Runner management — hold or take profits on winning trades?

The triage model (v2) identifies runners at bar 10 with 97% WR and +22 PnL.
The question: once you're 30+ bars in and profitable, should you:
  - Keep holding for a bigger move?
  - Take profits at some level?
  - Trail a stop?

This script:
  1. Profiles what happens to runners beyond bar 30
  2. Tests heuristic continuation rules (trail stops, targets, score-based)
  3. Builds an ML model for "should I keep holding this runner?"
  4. Compares all approaches on OOS data
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
from sandbox.train_ptrend import bootstrap_ci


# ──────────────────────────────────────────────────────────────────────
# FULL-DAY TRADE PATHS (no max_hold cap)
# ──────────────────────────────────────────────────────────────────────

def extract_full_paths(signals_df, rth):
    """Extract bar-by-bar paths through end of day (no max_hold cap)."""
    closes = rth['close'].values
    highs = rth['high'].values
    lows = rth['low'].values
    volumes = rth['volume'].values if 'volume' in rth.columns else np.ones(len(rth))
    tds = rth['trading_day'].values
    offsets = rth['bar_offset'].values

    paths = []
    for sig_i, (_, sig) in enumerate(signals_df.iterrows()):
        entry_idx = int(sig['idx'])
        entry_price = sig['entry_price']
        is_bull = sig['is_bull']
        entry_td = sig['trading_day']

        pc, ph, pl, pv, po = [], [], [], [], []
        for j in range(entry_idx + 1, len(rth)):
            if tds[j] != entry_td:
                break
            pc.append(closes[j])
            ph.append(highs[j])
            pl.append(lows[j])
            pv.append(volumes[j])
            po.append(offsets[j])

        if len(pc) < 10:
            continue

        pc = np.array(pc)
        ph = np.array(ph)
        pl = np.array(pl)
        pv = np.array(pv)
        direction = 1.0 if is_bull else -1.0

        unrealized = (pc - entry_price) * direction
        if is_bull:
            favorable = ph - entry_price
            adverse = entry_price - pl
        else:
            favorable = entry_price - pl
            adverse = ph - entry_price

        mfe = np.maximum.accumulate(favorable)
        mae = np.maximum.accumulate(adverse)

        paths.append({
            'signal_idx': sig_i,
            'entry_idx': entry_idx,
            'trading_day': entry_td,
            'entry_price': entry_price,
            'is_bull': is_bull,
            'bar_offset_entry': sig['bar_offset'],
            'pred_high': sig['pred_high'],
            'pred_low': sig['pred_low'],
            'signal_type': sig['signal_type'],
            'n_bars': len(pc),
            'unrealized': unrealized,
            'mfe': mfe,
            'mae': mae,
            'closes': pc,
            'highs': ph,
            'lows': pl,
            'volumes': pv,
            'offsets': np.array(po),
        })

    return paths


# ──────────────────────────────────────────────────────────────────────
# RUNNER PROFILING
# ──────────────────────────────────────────────────────────────────────

def profile_runners(paths, profitable_at=30, min_pnl=0):
    """Select trades that are profitable at bar `profitable_at`, then
    profile what happens to them beyond that point."""
    runners = []
    for p in paths:
        if p['n_bars'] < profitable_at + 1:
            continue
        pnl_at = p['unrealized'][profitable_at - 1]
        if pnl_at > min_pnl:
            runners.append(p)
    return runners


# ──────────────────────────────────────────────────────────────────────
# HEURISTIC EXIT RULES (applied at eval_bar onward)
# ──────────────────────────────────────────────────────────────────────

def apply_exit_rule(p, eval_bar, rule_fn, max_hold_after=None):
    """Apply an exit rule starting from eval_bar.

    rule_fn(p, j, eval_bar) -> True if should exit at bar j.
    Returns (exit_bar, exit_type).
    """
    max_bar = p['n_bars'] - 1
    if max_hold_after is not None:
        max_bar = min(max_bar, eval_bar + max_hold_after - 1)

    for j in range(eval_bar, max_bar + 1):
        if rule_fn(p, j, eval_bar):
            return j, 'rule'

    return max_bar, 'time'


def simulate_rule(paths, eval_bar, rule_fn, rule_name, max_hold_after=None):
    """Run an exit rule on all paths. Returns results DataFrame."""
    results = []
    for p in paths:
        exit_bar, exit_type = apply_exit_rule(p, eval_bar, rule_fn, max_hold_after)
        pnl = p['unrealized'][exit_bar]
        pnl_at_eval = p['unrealized'][eval_bar - 1]
        incremental = pnl - pnl_at_eval
        results.append({
            'trading_day': p['trading_day'],
            'signal_idx': p['signal_idx'],
            'pnl_total': pnl,
            'pnl_at_eval': pnl_at_eval,
            'pnl_incremental': incremental,
            'exit_bar': exit_bar + 1,
            'bars_after_eval': exit_bar - eval_bar + 1,
            'exit_type': exit_type,
            'win': pnl > 0,
        })
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# CONTINUATION FEATURES (at eval_bar)
# ──────────────────────────────────────────────────────────────────────

def build_continuation_features(paths, rolling_scores, eval_bar=30,
                                 future_horizon=30):
    """Build features at eval_bar for predicting continuation.

    Target: does the trade improve further?
      - continues = 1 if PnL at min(eval_bar + future_horizon, EOD) > PnL at eval_bar
      - runner_continues = 1 if MFE increases by ≥3 pts after eval_bar
    """
    checkpoints = sorted(next(iter(rolling_scores.values())).keys()) if rolling_scores else []
    rows = []

    for p in paths:
        if p['n_bars'] < eval_bar + 5:
            continue

        n = p['n_bars']
        unr = p['unrealized']
        mfe_arr = p['mfe']
        mae_arr = p['mae']
        pred_width = max(1.0, p['pred_high'] - p['pred_low'])

        # Current state at eval_bar
        eb = eval_bar - 1  # 0-indexed
        current_pnl = unr[eb]
        current_mfe = mfe_arr[eb]
        current_mae = mae_arr[eb]

        # ── Targets ──
        future_bar = min(eb + future_horizon, n - 1)
        future_pnl = unr[future_bar]
        continues = 1 if future_pnl > current_pnl else 0

        # MFE continues (runner gets even better)
        future_mfe = mfe_arr[future_bar]
        mfe_continues = 1 if (future_mfe - current_mfe) >= 3.0 else 0

        # Maximum drawdown from eval_bar forward
        future_pnls = unr[eb:future_bar + 1]
        max_dd = current_pnl - future_pnls.min()

        # ── Features ──

        # Trade state
        pnl_norm = current_pnl / pred_width
        mfe_norm = current_mfe / pred_width
        giveback = current_mfe - current_pnl
        giveback_pct = giveback / max(0.5, current_mfe)

        # Recent momentum (last 5, 10, 15 bars)
        slope_5 = (unr[eb] - unr[max(0, eb - 4)]) / 5.0
        slope_10 = (unr[eb] - unr[max(0, eb - 9)]) / 10.0
        slope_15 = (unr[eb] - unr[max(0, eb - 14)]) / 15.0

        # MFE timing — when was the peak reached?
        mfe_bar = np.argmax(mfe_arr[:eb + 1] == mfe_arr[eb])
        bars_since_mfe = eb - mfe_bar
        mfe_recency = bars_since_mfe / max(1, eval_bar)

        # Range consumed
        session_hi = p['highs'][:eb + 1].max()
        session_lo = p['lows'][:eb + 1].min()
        range_consumed = (session_hi - session_lo) / pred_width

        # Price position in predicted range
        price_pos = (p['closes'][eb] - p['pred_low']) / pred_width

        # Is price near a boundary? (potential for more move or reversal)
        dist_to_near_boundary = min(
            abs(p['closes'][eb] - p['pred_high']),
            abs(p['closes'][eb] - p['pred_low'])
        ) / pred_width

        # Trend score at eval_bar
        bar_offset = p['offsets'][eb] if eb < len(p['offsets']) else p['bar_offset_entry'] + eval_bar
        td = p['trading_day']
        day_scores = rolling_scores.get(td, {})
        trend_score = 0
        for cp in checkpoints:
            if cp <= bar_offset:
                trend_score = day_scores.get(cp, 0)

        # Session time
        session_elapsed = bar_offset / 390.0

        # Volume dynamics
        recent_vol = p['volumes'][max(0, eb - 5):eb + 1].mean()
        earlier_vol = p['volumes'][max(0, eb - 15):max(1, eb - 5)].mean()
        vol_ratio = recent_vol / max(1.0, earlier_vol)

        # Direction
        direction = 1.0 if p['is_bull'] else -1.0

        # Directional consistency over last 10 bars
        if eb >= 10:
            recent_returns = np.diff(p['closes'][eb - 10:eb + 1])
            dir_consistency = np.sum(recent_returns * direction > 0) / len(recent_returns)
        else:
            dir_consistency = 0.5

        # Entry time
        entry_time = p['bar_offset_entry'] / 390.0

        rows.append({
            'trading_day': td,
            'signal_idx': p['signal_idx'],
            # Targets
            'continues': continues,
            'mfe_continues': mfe_continues,
            'future_pnl': future_pnl,
            'current_pnl': current_pnl,
            'max_dd_forward': max_dd,
            # Features
            'pnl_norm': pnl_norm,
            'mfe_norm': mfe_norm,
            'giveback': giveback,
            'giveback_pct': giveback_pct,
            'slope_5': slope_5,
            'slope_10': slope_10,
            'slope_15': slope_15,
            'mfe_recency': mfe_recency,
            'bars_since_mfe': bars_since_mfe / 60.0,
            'range_consumed': range_consumed,
            'price_pos': price_pos,
            'dist_to_near_boundary': dist_to_near_boundary,
            'trend_score': trend_score,
            'session_elapsed': session_elapsed,
            'vol_ratio': vol_ratio,
            'direction': direction,
            'dir_consistency': dir_consistency,
            'entry_time': entry_time,
        })

    return pd.DataFrame(rows)


CONT_FEATURES = [
    'pnl_norm', 'mfe_norm', 'giveback', 'giveback_pct',
    'slope_5', 'slope_10', 'slope_15',
    'mfe_recency', 'bars_since_mfe',
    'range_consumed', 'price_pos', 'dist_to_near_boundary',
    'trend_score', 'session_elapsed', 'vol_ratio',
    'direction', 'dir_consistency', 'entry_time',
]


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD
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
        ('lr', LogisticRegression(C=1.0, class_weight={0: 1.0, 1: spw},
                                  max_iter=1000, random_state=42)),
    ])


def walk_forward(dataset, features, target, model_factory,
                 n_folds=5, min_train_days=100):
    """Walk-forward at signal level."""
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

        X_train = train[features].values
        y_train = train[target].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(1, n_pos)

        model = model_factory(spw)
        model.fit(X_train, y_train)

        p_pos = model.predict_proba(test[features].values)[:, 1]
        out = test[['trading_day', 'signal_idx', target, 'current_pnl']].copy()
        out['p_continue'] = p_pos
        out['fold'] = fold
        oos_parts.append(out)

        try:
            auc = roc_auc_score(test[target].values, p_pos)
        except ValueError:
            auc = 0.5
        print(f"    Fold {fold}: train={len(train)} ({len(train_days)}d), "
              f"test={len(test)} ({len(test_days)}d), AUC={auc:.3f}")

        if fold == n_folds - 1 and hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=features)
            for f, v in imp.sort_values(ascending=False).head(8).items():
                print(f"      {f:<25} {v:.3f}")

    if not oos_parts:
        return pd.DataFrame()
    return pd.concat(oos_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("RUNNER MANAGEMENT — hold or take profits on winning trades?")
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

    # Full-day paths (no max_hold cap)
    paths = extract_full_paths(sigs_h3, rth)
    print(f"  {len(paths)} full-day paths")
    avg_bars = np.mean([p['n_bars'] for p in paths])
    print(f"  Average path length: {avg_bars:.0f} bars")

    # ══════════════════════════════════════════════════════════════════
    # PART 1: PROFILING — what happens to runners after bar 30?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: RUNNER PROFILING — what happens after bar 30?")
    print("=" * W)

    # Select trades profitable at bar 30
    runners_30 = profile_runners(paths, profitable_at=30, min_pnl=0)
    big_runners = profile_runners(paths, profitable_at=30, min_pnl=5)
    losers_30 = [p for p in paths if p['n_bars'] >= 31 and p['unrealized'][29] <= 0]

    print(f"\n  At bar 30: {len(runners_30)} profitable, {len(losers_30)} losing, "
          f"{len(big_runners)} with PnL>5")

    # For runners at bar 30, what happens at later bars?
    print(f"\n  Runners (profitable at bar 30) — future PnL from bar 30:")
    print(f"  {'Exit bar':<10} {'n':>5} {'Avg PnL':>8} {'Med PnL':>8} "
          f"{'Incr PnL':>9} {'WR(incr)':>9} {'Max DD':>7}")

    for exit_bar in [30, 45, 60, 90, 120, 180, 250]:
        pnls = []
        incrs = []
        dds = []
        for p in runners_30:
            if p['n_bars'] < exit_bar:
                continue
            eb = exit_bar - 1
            pnl = p['unrealized'][eb]
            pnl_30 = p['unrealized'][29]
            incr = pnl - pnl_30
            # Max drawdown from bar 30 to exit
            dd = pnl_30 - p['unrealized'][29:eb + 1].min()
            pnls.append(pnl)
            incrs.append(incr)
            dds.append(dd)

        if pnls:
            pnls = np.array(pnls)
            incrs = np.array(incrs)
            dds = np.array(dds)
            label = f"Bar {exit_bar}" if exit_bar > 30 else "Bar 30 (ref)"
            print(f"  {label:<10} {len(pnls):>5} {pnls.mean():>+8.1f} "
                  f"{np.median(pnls):>+8.1f} {incrs.mean():>+9.1f} "
                  f"{(incrs > 0).mean():>8.0%} {dds.mean():>7.1f}")

    # Big runners (PnL>5 at bar 30)
    print(f"\n  Big runners (PnL>5 at bar 30) — future PnL:")
    print(f"  {'Exit bar':<10} {'n':>5} {'Avg PnL':>8} {'Med PnL':>8} "
          f"{'Incr PnL':>9} {'WR(incr)':>9}")
    for exit_bar in [30, 45, 60, 90, 120, 180]:
        pnls = []
        incrs = []
        for p in big_runners:
            if p['n_bars'] < exit_bar:
                continue
            pnl = p['unrealized'][exit_bar - 1]
            incr = pnl - p['unrealized'][29]
            pnls.append(pnl)
            incrs.append(incr)
        if pnls:
            pnls = np.array(pnls)
            incrs = np.array(incrs)
            print(f"  {'Bar ' + str(exit_bar):<10} {len(pnls):>5} {pnls.mean():>+8.1f} "
                  f"{np.median(pnls):>+8.1f} {incrs.mean():>+9.1f} "
                  f"{(incrs > 0).mean():>8.0%}")

    # What about losers — do they recover?
    print(f"\n  Losers (PnL≤0 at bar 30) — do they recover?")
    for exit_bar in [45, 60, 90, 120]:
        pnls = []
        for p in losers_30:
            if p['n_bars'] >= exit_bar:
                pnls.append(p['unrealized'][exit_bar - 1])
        if pnls:
            pnls = np.array(pnls)
            print(f"    Bar {exit_bar}: n={len(pnls)}, avg={pnls.mean():+.1f}, "
                  f"recover rate={(pnls > 0).mean():.0%}")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: HEURISTIC EXIT RULES — tested on runners from bar 30
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: HEURISTIC EXIT RULES — applied to ALL trades from bar 30")
    print("=" * W)

    # We compare: given you're at bar 30, what exit rule maximizes total PnL?
    # (including both runners and losers — the rule applies to everyone)
    all_at_30 = [p for p in paths if p['n_bars'] >= 31]
    print(f"\n  {len(all_at_30)} trades with ≥31 bars")
    print(f"\n  {'Rule':<50} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18} {'bars':>6}")

    # Baseline: exit at bar 30 (current strategy)
    pnl_30 = np.array([p['unrealized'][29] for p in all_at_30])
    m, lo, hi = bootstrap_ci(pnl_30)
    print(f"  {'Exit at bar 30 (baseline)':<50} {len(pnl_30):>5} "
          f"{(pnl_30 > 0).mean():>5.0%} {m:>+7.1f}  [{lo:>+7.1f}, {hi:>+7.1f}] {'30':>6}")

    # Define rules
    rules = []

    # Fixed holds
    for hold_to in [45, 60, 90]:
        def make_rule(h):
            return lambda p, j, eb: j >= h - 1
        rules.append((f'Hold to bar {hold_to}', make_rule(hold_to), None))

    # Hold to EOD
    rules.append(('Hold to EOD', lambda p, j, eb: j >= p['n_bars'] - 1, None))

    # Trail from MFE (activate when MFE exceeds threshold)
    for trail in [3, 5, 8]:
        def make_trail(t):
            return lambda p, j, eb: (p['mfe'][j] >= t and
                                      (p['mfe'][j] - p['unrealized'][j]) >= t)
        rules.append((f'Trail {trail}pt from MFE≥{trail}', make_trail(trail), 120))

    # Trail from bar-30 PnL (give back no more than X pts from bar-30 level)
    for give in [3, 5, 8]:
        def make_give(g):
            return lambda p, j, eb: p['unrealized'][j] < p['unrealized'][eb - 1] - g
        rules.append((f'Exit if PnL drops {give}pts below bar-30', make_give(give), 120))

    # Fixed profit targets from entry
    for target in [10, 15, 20]:
        def make_target(t):
            return lambda p, j, eb: p['unrealized'][j] >= t
        rules.append((f'Take profit at +{target} pts', make_target(target), 120))

    # Incremental profit target (from bar-30 PnL)
    for incr_tgt in [5, 8, 10]:
        def make_incr(t):
            return lambda p, j, eb: p['unrealized'][j] >= p['unrealized'][eb - 1] + t
        rules.append((f'Take +{incr_tgt} incremental from bar 30', make_incr(incr_tgt), 120))

    # Range consumed threshold
    for rc in [0.8, 0.9, 1.0]:
        def make_rc(threshold):
            def rule(p, j, eb):
                pw = max(1.0, p['pred_high'] - p['pred_low'])
                session_range = p['highs'][:j+1].max() - p['lows'][:j+1].min()
                return session_range / pw >= threshold
            return rule
        rules.append((f'Exit if range consumed ≥ {rc:.0%}', make_rc(rc), 120))

    # Trend score based exit
    for score_thresh in [3, 4, 5]:
        def make_score_exit(st, rs=rolling_scores, cps=checkpoints):
            def rule(p, j, eb):
                td = p['trading_day']
                ds = rs.get(td, {})
                bar_off = p['offsets'][j] if j < len(p['offsets']) else p['bar_offset_entry'] + j + 1
                score = 0
                for cp in cps:
                    if cp <= bar_off:
                        score = ds.get(cp, 0)
                return score >= st
            return rule
        rules.append((f'Exit if trend score ≥ {score_thresh}', make_score_exit(score_thresh), 120))

    # Combined: trail + time limit
    def make_combined(t, tlim):
        def rule(p, j, eb):
            bars_after = j - eb + 1
            if bars_after >= tlim:
                return True
            if p['mfe'][j] >= t and (p['mfe'][j] - p['unrealized'][j]) >= t:
                return True
            return False
        return rule
    for trail, time_limit in [(5, 60), (5, 90), (3, 45)]:
        rules.append((f'Trail {trail}pt OR time limit {time_limit} bars', make_combined(trail, time_limit), None))

    for rule_name, rule_fn, max_after in rules:
        res = simulate_rule(all_at_30, 30, rule_fn, rule_name, max_hold_after=max_after)
        pnls = res['pnl_total'].values
        m, lo, hi = bootstrap_ci(pnls)
        avg_bars = res['exit_bar'].mean()
        wr = (pnls > 0).mean()
        print(f"  {rule_name:<50} {len(pnls):>5} {wr:>5.0%} {m:>+7.1f}  "
              f"[{lo:>+7.1f}, {hi:>+7.1f}] {avg_bars:>5.0f}b")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: CONDITIONAL — rules that differ by trade state at bar 30
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: CONDITIONAL RULES — different strategy by PnL at bar 30")
    print("=" * W)

    # Split into PnL buckets at bar 30
    for bucket_name, filter_fn in [
        ('PnL > +10', lambda p: p['unrealized'][29] > 10),
        ('+5 < PnL ≤ +10', lambda p: 5 < p['unrealized'][29] <= 10),
        ('0 < PnL ≤ +5', lambda p: 0 < p['unrealized'][29] <= 5),
        ('-5 < PnL ≤ 0', lambda p: -5 < p['unrealized'][29] <= 0),
        ('PnL ≤ -5', lambda p: p['unrealized'][29] <= -5),
    ]:
        bucket = [p for p in all_at_30 if filter_fn(p)]
        if len(bucket) < 10:
            continue

        print(f"\n  {bucket_name} (n={len(bucket)}):")
        print(f"  {'Rule':<45} {'WR':>6} {'PnL':>7} {'bars':>6}")

        # Baseline
        pnl_b = np.array([p['unrealized'][29] for p in bucket])
        print(f"  {'Exit at bar 30':<45} {(pnl_b > 0).mean():>5.0%} "
              f"{pnl_b.mean():>+7.1f} {'30':>6}")

        for hold_to in [45, 60, 90]:
            pnls = []
            for p in bucket:
                eb = min(hold_to - 1, p['n_bars'] - 1)
                pnls.append(p['unrealized'][eb])
            pnls = np.array(pnls)
            print(f"  {f'Hold to bar {hold_to}':<45} {(pnls > 0).mean():>5.0%} "
                  f"{pnls.mean():>+7.1f} {hold_to:>6}")

        # Trail 5pt from MFE
        res = simulate_rule(bucket, 30,
                           lambda p, j, eb: (p['mfe'][j] >= 5 and
                                             (p['mfe'][j] - p['unrealized'][j]) >= 5),
                           'trail', max_hold_after=120)
        print(f"  {'Trail 5pt from MFE≥5':<45} {(res['pnl_total'] > 0).mean():>5.0%} "
              f"{res['pnl_total'].mean():>+7.1f} {res['exit_bar'].mean():>5.0f}b")

        # EOD
        pnls_eod = np.array([p['unrealized'][-1] for p in bucket])
        print(f"  {'Hold to EOD':<45} {(pnls_eod > 0).mean():>5.0%} "
              f"{pnls_eod.mean():>+7.1f} {'EOD':>6}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: ML CONTINUATION MODEL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: ML CONTINUATION MODEL — P(trade improves after bar 30)")
    print("=" * W)

    for eval_bar, future_horizon in [(30, 30), (30, 60), (15, 30)]:
        cont_data = build_continuation_features(all_at_30, rolling_scores,
                                                 eval_bar=eval_bar,
                                                 future_horizon=future_horizon)
        cont_rate = cont_data['continues'].mean()
        mfe_rate = cont_data['mfe_continues'].mean()
        print(f"\n  Eval@bar {eval_bar}, horizon={future_horizon}: "
              f"{len(cont_data)} trades, continues={cont_rate:.0%}, "
              f"MFE continues(+3)={mfe_rate:.0%}")

        for target_name, target_col in [('continues', 'continues'),
                                         ('MFE continues +3', 'mfe_continues')]:
            print(f"\n  Target: {target_name}")
            for name, factory in [('XGB', make_xgb), ('Logistic', make_lr)]:
                print(f"    {name}:")
                oos = walk_forward(cont_data, CONT_FEATURES, target_col, factory)
                if len(oos) > 0:
                    auc = roc_auc_score(oos[target_col], oos['p_continue'])
                    print(f"    Overall AUC: {auc:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: ML-GUIDED EXIT SIMULATION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: ML-GUIDED EXIT vs HEURISTICS vs BASELINE")
    print("=" * W)

    # Train continuation model with best config
    cont_data_30 = build_continuation_features(all_at_30, rolling_scores,
                                                eval_bar=30, future_horizon=30)
    cont_oos = walk_forward(cont_data_30, CONT_FEATURES, 'continues', make_xgb)

    if len(cont_oos) > 0:
        oos_days = set(cont_oos['trading_day'].unique())
        oos_paths = [p for p in all_at_30 if p['trading_day'] in oos_days]
        print(f"\n  OOS: {len(oos_paths)} paths ({len(oos_days)} days)")

        # Build prediction lookup
        pred_map = {}
        for _, row in cont_oos.iterrows():
            pred_map[(row['trading_day'], row['signal_idx'])] = row['p_continue']

        print(f"\n  {'Strategy':<50} {'n':>5} {'WR':>6} {'PnL':>7}  "
              f"{'95% CI':>18} {'bars':>6}")

        # Baseline: exit at bar 30
        pnl_30_oos = np.array([p['unrealized'][29] for p in oos_paths])
        m, lo, hi = bootstrap_ci(pnl_30_oos)
        print(f"  {'Exit at bar 30':<50} {len(pnl_30_oos):>5} "
              f"{(pnl_30_oos > 0).mean():>5.0%} {m:>+7.1f}  "
              f"[{lo:>+7.1f}, {hi:>+7.1f}] {'30':>6}")

        # Top heuristics on OOS paths
        for rule_name, rule_fn, max_after in [
            ('Hold to bar 45', lambda p, j, eb: j >= 44, None),
            ('Hold to bar 60', lambda p, j, eb: j >= 59, None),
            ('Trail 5pt from MFE≥5', lambda p, j, eb: p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5, 120),
            ('Trail 5pt OR time 60', lambda p, j, eb: (j - eb + 1 >= 60) or (p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5), None),
        ]:
            res = simulate_rule(oos_paths, 30, rule_fn, rule_name, max_hold_after=max_after)
            pnls = res['pnl_total'].values
            m, lo, hi = bootstrap_ci(pnls)
            avg_bars = res['exit_bar'].mean()
            print(f"  {rule_name:<50} {len(pnls):>5} "
                  f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  "
                  f"[{lo:>+7.1f}, {hi:>+7.1f}] {avg_bars:>5.0f}b")

        # ML-guided: if P(continue) high, hold to 60; if low, exit at 30
        for cont_thresh in [0.4, 0.5, 0.6]:
            pnls = []
            bars_list = []
            for p in oos_paths:
                key = (p['trading_day'], p['signal_idx'])
                p_cont = pred_map.get(key, 0.5)
                if p_cont >= cont_thresh:
                    eb = min(59, p['n_bars'] - 1)  # Hold to 60
                else:
                    eb = 29  # Exit at 30
                pnls.append(p['unrealized'][eb])
                bars_list.append(eb + 1)
            pnls = np.array(pnls)
            bars_arr = np.array(bars_list)
            m, lo, hi = bootstrap_ci(pnls)
            label = f"ML: P(cont)≥{cont_thresh} → hold60, else exit30"
            print(f"  {label:<50} {len(pnls):>5} "
                  f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  "
                  f"[{lo:>+7.1f}, {hi:>+7.1f}] {bars_arr.mean():>5.0f}b")

        # ML-guided with trail: P(cont) high → trail, else exit at 30
        for cont_thresh in [0.4, 0.5, 0.6]:
            pnls = []
            bars_list = []
            for p in oos_paths:
                key = (p['trading_day'], p['signal_idx'])
                p_cont = pred_map.get(key, 0.5)
                if p_cont >= cont_thresh:
                    # Hold with 5pt trail, max 120 bars after bar 30
                    exit_bar = min(p['n_bars'] - 1, 29 + 120)
                    for j in range(30, exit_bar + 1):
                        if p['mfe'][j] >= 5 and (p['mfe'][j] - p['unrealized'][j]) >= 5:
                            exit_bar = j
                            break
                    pnls.append(p['unrealized'][exit_bar])
                    bars_list.append(exit_bar + 1)
                else:
                    pnls.append(p['unrealized'][29])
                    bars_list.append(30)
            pnls = np.array(pnls)
            bars_arr = np.array(bars_list)
            m, lo, hi = bootstrap_ci(pnls)
            label = f"ML: P≥{cont_thresh} → trail5, else exit30"
            print(f"  {label:<50} {len(pnls):>5} "
                  f"{(pnls > 0).mean():>5.0%} {m:>+7.1f}  "
                  f"[{lo:>+7.1f}, {hi:>+7.1f}] {bars_arr.mean():>5.0f}b")

        # Calibration of continuation model
        print(f"\n  Continuation model calibration:")
        cont_with_outcome = cont_oos.copy()
        # Add actual future PnL
        for i, row in cont_with_outcome.iterrows():
            td, si = row['trading_day'], row['signal_idx']
            for p in oos_paths:
                if p['trading_day'] == td and p['signal_idx'] == si:
                    future_bar = min(59, p['n_bars'] - 1)
                    cont_with_outcome.at[i, 'pnl_at_60'] = p['unrealized'][future_bar]
                    cont_with_outcome.at[i, 'pnl_at_30'] = p['unrealized'][29]
                    break

        if 'pnl_at_60' in cont_with_outcome.columns:
            cont_with_outcome['p_bin'] = pd.cut(cont_with_outcome['p_continue'],
                                                 bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
            print(f"  {'P(cont) bin':<15} {'n':>5} {'Act cont%':>10} {'Avg PnL@30':>12} "
                  f"{'Avg PnL@60':>12} {'Incr PnL':>10}")
            for bin_label, grp in cont_with_outcome.groupby('p_bin', observed=True):
                if len(grp) < 5:
                    continue
                act_cont = grp['continues'].mean()
                pnl30 = grp['pnl_at_30'].mean() if 'pnl_at_30' in grp.columns else np.nan
                pnl60 = grp['pnl_at_60'].mean() if 'pnl_at_60' in grp.columns else np.nan
                incr = pnl60 - pnl30 if not np.isnan(pnl60) else np.nan
                print(f"  {str(bin_label):<15} {len(grp):>5} {act_cont:>9.0%} "
                      f"{pnl30:>+12.1f} {pnl60:>+12.1f} {incr:>+10.1f}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
