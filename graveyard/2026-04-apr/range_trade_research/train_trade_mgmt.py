"""ML-based trade management — dynamic exit timing for fade trades.

Current system: proximity fade with heuristic filter uses naive 30-bar hold.
This script investigates whether an ML model can improve exits by learning
when to take profits early vs hold for runners, conditioned on post-entry
price action and evolving market state.

Approach:
  - At each bar post-entry, classify P(should_hold) = P(trade improves over next N bars)
  - Features: trade state (unrealized PnL, MFE/MAE, momentum) + market context
  - Walk-forward evaluation, compared against fixed hold and stop+target baselines
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
from sandbox.train_ptrend import bootstrap_ci


# ──────────────────────────────────────────────────────────────────────
# TRADE PATH EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def extract_trade_paths(signals_df, rth, max_hold=60):
    """Extract bar-by-bar post-entry paths for each signal.

    Returns list of dicts, each containing:
      - signal metadata (idx, trading_day, entry_price, is_bull, etc.)
      - arrays: unrealized_pnl, mfe, mae, closes, highs, lows, volumes
    """
    closes = rth['close'].values
    highs = rth['high'].values
    lows = rth['low'].values
    volumes = rth['volume'].values if 'volume' in rth.columns else np.ones(len(rth))
    tds = rth['trading_day'].values
    offsets = rth['bar_offset'].values

    paths = []
    for _, sig in signals_df.iterrows():
        entry_idx = int(sig['idx'])
        entry_price = sig['entry_price']
        is_bull = sig['is_bull']
        entry_td = sig['trading_day']

        # Find end of day or max_hold, whichever comes first
        path_closes = []
        path_highs = []
        path_lows = []
        path_volumes = []
        path_offsets = []

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(rth))):
            if tds[j] != entry_td:
                break
            path_closes.append(closes[j])
            path_highs.append(highs[j])
            path_lows.append(lows[j])
            path_volumes.append(volumes[j])
            path_offsets.append(offsets[j])

        if len(path_closes) < 3:
            continue

        path_closes = np.array(path_closes)
        path_highs = np.array(path_highs)
        path_lows = np.array(path_lows)
        path_volumes = np.array(path_volumes)

        direction = 1.0 if is_bull else -1.0

        # Unrealized PnL at each bar (using close)
        unrealized = (path_closes - entry_price) * direction

        # MFE/MAE using high/low (more accurate than close-only)
        if is_bull:
            favorable = path_highs - entry_price
            adverse = entry_price - path_lows
        else:
            favorable = entry_price - path_lows
            adverse = path_highs - entry_price

        mfe = np.maximum.accumulate(favorable)
        mae = np.maximum.accumulate(adverse)

        paths.append({
            'signal_idx': sig.name if hasattr(sig, 'name') else len(paths),
            'entry_idx': entry_idx,
            'trading_day': entry_td,
            'entry_price': entry_price,
            'is_bull': is_bull,
            'bar_offset_entry': sig['bar_offset'],
            'pred_high': sig['pred_high'],
            'pred_low': sig['pred_low'],
            'signal_type': sig['signal_type'],
            'n_bars': len(path_closes),
            'unrealized': unrealized,
            'mfe': mfe,
            'mae': mae,
            'closes': path_closes,
            'highs': path_highs,
            'lows': path_lows,
            'volumes': path_volumes,
            'offsets': np.array(path_offsets),
        })

    return paths


# ──────────────────────────────────────────────────────────────────────
# SIGNAL-LEVEL DESCRIPTIVE STATS
# ──────────────────────────────────────────────────────────────────────

def compute_signal_stats(paths):
    """Compute per-signal summary stats: MFE, MAE, optimal exit bar, etc."""
    rows = []
    for p in paths:
        n = p['n_bars']
        unr = p['unrealized']
        mfe_arr = p['mfe']
        mae_arr = p['mae']

        optimal_bar = np.argmax(unr) + 1  # 1-indexed bars held
        mfe_total = mfe_arr[-1]
        mae_total = mae_arr[-1]
        pnl_at_30 = unr[min(29, n - 1)]
        pnl_at_eod = unr[-1]

        # Quick mover: MFE reaches +5 within 10 bars
        early_mfe = mfe_arr[:min(10, n)]
        quick_mover = 1 if early_mfe.max() >= 5.0 else 0

        # Chopper: spends >20 bars with |unrealized| < 3
        chop_bars = np.sum(np.abs(unr[:min(30, n)]) < 3.0)
        chopper = 1 if chop_bars > 20 else 0

        rows.append({
            'trading_day': p['trading_day'],
            'bar_offset_entry': p['bar_offset_entry'],
            'signal_type': p['signal_type'],
            'n_bars': n,
            'mfe_total': mfe_total,
            'mae_total': mae_total,
            'optimal_bar': optimal_bar,
            'pnl_at_30': pnl_at_30,
            'pnl_at_eod': pnl_at_eod,
            'quick_mover': quick_mover,
            'chopper': chopper,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# BAR-LEVEL DATASET (features + targets)
# ──────────────────────────────────────────────────────────────────────

def build_bar_level_dataset(paths, rolling_scores, lookahead=15):
    """Build bar-level features and targets for each post-entry bar.

    Target: should_hold = 1 if PnL at bar j+lookahead > PnL at bar j
            (i.e., the trade improves if you keep holding)

    Features:
      Trade state: bars_held, unrealized PnL, MFE, MAE, momentum, giveback
      Market context: trend score, range consumed, session elapsed
      Entry context: entry time, direction
    """
    checkpoints = sorted(next(iter(rolling_scores.values())).keys()) if rolling_scores else []

    rows = []
    for p in paths:
        n = p['n_bars']
        unr = p['unrealized']
        mfe_arr = p['mfe']
        mae_arr = p['mae']
        pred_width = max(1.0, p['pred_high'] - p['pred_low'])
        td = p['trading_day']
        day_scores = rolling_scores.get(td, {})

        # Session high/low up to each bar (for range_consumed)
        cum_session_hi = np.maximum.accumulate(p['highs'])
        cum_session_lo = np.minimum.accumulate(p['lows'])

        for j in range(n):
            bars_held = j + 1

            # ── Target ──
            future_idx = min(j + lookahead, n - 1)
            future_pnl = unr[future_idx]
            current_pnl = unr[j]
            should_hold = 1 if future_pnl > current_pnl else 0

            # Also compute alternative targets
            # More aggressive: does MFE improve?
            future_mfe = mfe_arr[future_idx]
            current_mfe = mfe_arr[j]

            # ── Trade state features ──
            unrealized_pnl = current_pnl
            unrealized_pnl_norm = current_pnl / pred_width
            mfe_so_far = mfe_arr[j]
            mae_so_far = mae_arr[j]

            # Giveback from peak
            pnl_vs_mfe = current_pnl / max(0.25, mfe_so_far) if mfe_so_far > 0.25 else 0.0

            # Speed of MFE
            mfe_speed = mfe_so_far / bars_held

            # PnL slope (momentum over last 5 bars)
            if j >= 4:
                pnl_slope_5 = (unr[j] - unr[j - 4]) / 5.0
            elif j >= 1:
                pnl_slope_5 = (unr[j] - unr[0]) / max(1, j)
            else:
                pnl_slope_5 = 0.0

            # PnL slope over last 15 bars
            if j >= 14:
                pnl_slope_15 = (unr[j] - unr[j - 14]) / 15.0
            elif j >= 1:
                pnl_slope_15 = (unr[j] - unr[0]) / max(1, j)
            else:
                pnl_slope_15 = 0.0

            # Bars since MFE was set
            mfe_bar = np.argmax(mfe_arr[:j + 1] == mfe_arr[j])
            bars_since_mfe = j - mfe_bar

            # ── Market context features ──
            # Trend score at most recent checkpoint
            bar_offset_j = p['offsets'][j] if j < len(p['offsets']) else p['bar_offset_entry'] + bars_held
            trend_score = 0.0
            for cp in checkpoints:
                if cp <= bar_offset_j:
                    trend_score = day_scores.get(cp, 0.0)

            # Range consumed
            session_range = cum_session_hi[j] - cum_session_lo[j]
            range_consumed = session_range / pred_width

            # Price position within predicted range
            price_position = (p['closes'][j] - p['pred_low']) / pred_width

            # Session elapsed
            session_elapsed = bar_offset_j / 390.0

            # Volume rate (vs trailing avg)
            if j >= 5:
                avg_vol = p['volumes'][max(0, j - 30):j].mean()
                vol_rate = p['volumes'][j] / max(1.0, avg_vol)
            else:
                vol_rate = 1.0

            # ── Entry context features ──
            entry_time_norm = p['bar_offset_entry'] / 390.0
            direction = 1.0 if p['is_bull'] else -1.0

            rows.append({
                'trading_day': td,
                'signal_idx': p['signal_idx'],
                'bar_j': j,
                'bars_held': bars_held,
                # Targets
                'should_hold': should_hold,
                'future_pnl': future_pnl,
                'current_pnl': current_pnl,
                # Trade state features
                'bars_held_norm': bars_held / 60.0,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_norm': unrealized_pnl_norm,
                'mfe_so_far': mfe_so_far,
                'mae_so_far': mae_so_far,
                'pnl_vs_mfe': pnl_vs_mfe,
                'mfe_speed': mfe_speed,
                'pnl_slope_5': pnl_slope_5,
                'pnl_slope_15': pnl_slope_15,
                'bars_since_mfe': bars_since_mfe / 60.0,
                # Market context
                'trend_score': trend_score,
                'range_consumed': range_consumed,
                'price_position': price_position,
                'session_elapsed': session_elapsed,
                'vol_rate': vol_rate,
                # Entry context
                'entry_time_norm': entry_time_norm,
                'direction': direction,
            })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    'bars_held_norm', 'unrealized_pnl', 'unrealized_pnl_norm',
    'mfe_so_far', 'mae_so_far', 'pnl_vs_mfe', 'mfe_speed',
    'pnl_slope_5', 'pnl_slope_15', 'bars_since_mfe',
    'trend_score', 'range_consumed', 'price_position',
    'session_elapsed', 'vol_rate',
    'entry_time_norm', 'direction',
]


# ──────────────────────────────────────────────────────────────────────
# EXIT SIMULATION
# ──────────────────────────────────────────────────────────────────────

def simulate_ml_exit(paths, bar_dataset, oos_preds, threshold=0.5,
                     max_hold=60, stop_pts=None):
    """Simulate ML-driven exits.

    At each bar post-entry:
      1. Check stop (if stop_pts set)
      2. Get P(should_hold) from oos_preds
      3. If P < threshold, exit at close
      4. Otherwise hold until max_hold or EOD

    Returns DataFrame with columns matching evaluate_fixed_hold output.
    """
    results = []
    for p in paths:
        td = p['trading_day']
        sig_idx = p['signal_idx']
        n = p['n_bars']
        entry_price = p['entry_price']
        is_bull = p['is_bull']
        direction = 1.0 if is_bull else -1.0

        # Get this signal's OOS predictions
        sig_preds = oos_preds[(oos_preds['trading_day'] == td) &
                              (oos_preds['signal_idx'] == sig_idx)]
        if len(sig_preds) == 0:
            continue

        pred_map = dict(zip(sig_preds['bar_j'].values, sig_preds['p_hold'].values))

        exit_bar = n - 1  # default: hold to end
        exit_type = 'max_hold'

        for j in range(n):
            # Check stop first
            if stop_pts is not None:
                if is_bull and (p['lows'][j] <= entry_price - stop_pts):
                    exit_bar = j
                    exit_type = 'stop'
                    break
                elif not is_bull and (p['highs'][j] >= entry_price + stop_pts):
                    exit_bar = j
                    exit_type = 'stop'
                    break

            # ML exit decision
            p_hold = pred_map.get(j, 0.5)
            if p_hold < threshold:
                exit_bar = j
                exit_type = 'ml_exit'
                break

        pnl = p['unrealized'][exit_bar]
        bars_held = exit_bar + 1

        results.append({
            'signal_type': p['signal_type'],
            'trading_day': td,
            'bar_offset': p['bar_offset_entry'],
            'pnl': pnl,
            'win': pnl > 0,
            'bars_held': bars_held,
            'exit_type': exit_type,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD TRAINING
# ──────────────────────────────────────────────────────────────────────

def make_xgb_exit(scale_pos_weight=1.0):
    return XGBClassifier(
        max_depth=2,
        learning_rate=0.05,
        n_estimators=200,
        min_child_weight=50,
        reg_alpha=2.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0,
    )


def make_logistic_exit(scale_pos_weight=1.0):
    w = {0: 1.0, 1: scale_pos_weight}
    return Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.5, class_weight=w, max_iter=1000,
                                  solver='lbfgs', random_state=42)),
    ])


def walk_forward_exit_model(bar_dataset, paths, model_factory,
                            n_folds=5, min_train_days=100):
    """Walk-forward train/test for exit model.

    Splits by trading_day. Returns OOS predictions at the bar level.
    """
    days = sorted(bar_dataset['trading_day'].unique())
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

        train = bar_dataset[bar_dataset['trading_day'].isin(train_days)]
        test = bar_dataset[bar_dataset['trading_day'].isin(test_days)]

        if len(train) < 100 or len(test) < 50:
            continue

        X_train = train[FEATURE_COLS].values
        y_train = train['should_hold'].values
        X_test = test[FEATURE_COLS].values
        y_test = test['should_hold'].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(1, n_pos)

        model = model_factory(spw)
        model.fit(X_train, y_train)

        p_hold = model.predict_proba(X_test)[:, 1]

        out = test[['trading_day', 'signal_idx', 'bar_j',
                     'should_hold', 'current_pnl']].copy()
        out['p_hold'] = p_hold
        out['fold'] = fold
        oos_parts.append(out)

        try:
            auc = roc_auc_score(y_test, p_hold)
        except ValueError:
            auc = 0.5
        print(f"    Fold {fold}: train={len(train_days)} days ({len(train)} bars), "
              f"test={len(test_days)} days ({len(test)} bars), AUC={auc:.3f}")

        # Feature importance for last fold
        if fold == n_folds - 1 and hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            imp = imp.sort_values(ascending=False)
            print(f"\n    Top features (last fold):")
            for feat, val in imp.head(8).items():
                print(f"      {feat:<25} {val:.3f}")

    if not oos_parts:
        return pd.DataFrame()
    return pd.concat(oos_parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("TRADE MANAGEMENT — ML-based dynamic exit timing")
    print("=" * W)

    # ── Load data and build signals ──
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
    print(f"  {len(sigs_all)} total proximity fade signals")

    # Apply heuristic filters at different strictness levels
    sigs_h3 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 3.0)
    sigs_h15 = filter_signals_by_rolling_score(sigs_all, rolling_scores, 1.5)
    print(f"  After heuristic < 3: {len(sigs_h3)} signals")
    print(f"  After heuristic < 1.5: {len(sigs_h15)} signals")

    # Use thresh < 3 for more training data; evaluate on both
    sigs = sigs_h3
    MAX_HOLD = 60

    # ══════════════════════════════════════════════════════════════════
    # PART 1: DESCRIPTIVE — trade path characteristics
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 1: TRADE PATH CHARACTERISTICS")
    print("=" * W)

    paths = extract_trade_paths(sigs, rth, max_hold=MAX_HOLD)
    print(f"\n  Extracted {len(paths)} trade paths (max {MAX_HOLD} bars)")

    stats = compute_signal_stats(paths)
    print(f"\n  Signal-level summary:")
    print(f"    MFE:  mean={stats['mfe_total'].mean():+.1f}, "
          f"median={stats['mfe_total'].median():+.1f}, "
          f"P25={stats['mfe_total'].quantile(0.25):+.1f}, "
          f"P75={stats['mfe_total'].quantile(0.75):+.1f}")
    print(f"    MAE:  mean={stats['mae_total'].mean():.1f}, "
          f"median={stats['mae_total'].median():.1f}, "
          f"P25={stats['mae_total'].quantile(0.25):.1f}, "
          f"P75={stats['mae_total'].quantile(0.75):.1f}")
    print(f"    Optimal exit bar: mean={stats['optimal_bar'].mean():.0f}, "
          f"median={stats['optimal_bar'].median():.0f}")
    print(f"    PnL at bar 30:   mean={stats['pnl_at_30'].mean():+.1f}")
    print(f"    PnL at EOD/60:   mean={stats['pnl_at_eod'].mean():+.1f}")
    print(f"    Quick movers (MFE≥5 in 10 bars): {stats['quick_mover'].sum()} "
          f"({stats['quick_mover'].mean():.0%})")
    print(f"    Choppers (|PnL|<3 for >20 bars):  {stats['chopper'].sum()} "
          f"({stats['chopper'].mean():.0%})")

    # Average PnL curve over hold time
    print(f"\n  Average unrealized PnL by bars held:")
    print(f"  {'Bars':<8} {'Avg PnL':>8} {'Med PnL':>8} {'WR':>6} {'n':>5}")
    for b in [1, 3, 5, 10, 15, 20, 30, 45, 60]:
        pnls = []
        for p in paths:
            if p['n_bars'] >= b:
                pnls.append(p['unrealized'][b - 1])
        if pnls:
            pnls = np.array(pnls)
            print(f"  {b:<8} {pnls.mean():>+8.1f} {np.median(pnls):>+8.1f} "
                  f"{(pnls > 0).mean():>5.0%} {len(pnls):>5}")

    # ══════════════════════════════════════════════════════════════════
    # PART 2: ORACLE ANALYSIS — what's the ceiling?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 2: ORACLE ANALYSIS — theoretical ceiling for exit optimization")
    print("=" * W)

    # Compare: fixed hold vs oracle optimal exit
    print(f"\n  {'Strategy':<40} {'n':>5} {'WR':>6} {'PnL':>7} {'Avg bars':>9}")

    # Fixed 30-bar
    pnl_30 = np.array([p['unrealized'][min(29, p['n_bars'] - 1)] for p in paths])
    wr_30 = (pnl_30 > 0).mean()
    print(f"  {'Fixed 30-bar hold':<40} {len(pnl_30):>5} {wr_30:>5.0%} "
          f"{pnl_30.mean():>+7.1f} {'30':>9}")

    # Fixed 60-bar / EOD
    pnl_eod = np.array([p['unrealized'][-1] for p in paths])
    wr_eod = (pnl_eod > 0).mean()
    avg_bars_eod = np.mean([p['n_bars'] for p in paths])
    print(f"  {'Fixed 60-bar / EOD':<40} {len(pnl_eod):>5} {wr_eod:>5.0%} "
          f"{pnl_eod.mean():>+7.1f} {avg_bars_eod:>9.0f}")

    # Oracle: exit at optimal bar
    pnl_oracle = np.array([p['unrealized'][int(stats.iloc[i]['optimal_bar']) - 1]
                           for i, p in enumerate(paths)])
    bars_oracle = stats['optimal_bar'].values
    wr_oracle = (pnl_oracle > 0).mean()
    print(f"  {'Oracle optimal exit':<40} {len(pnl_oracle):>5} {wr_oracle:>5.0%} "
          f"{pnl_oracle.mean():>+7.1f} {bars_oracle.mean():>9.1f}")

    # Simple rule: exit when unrealized PnL first reaches +5 pts (or hold 30)
    pnl_tp5 = []
    bars_tp5 = []
    for p in paths:
        exited = False
        for j in range(p['n_bars']):
            if p['unrealized'][j] >= 5.0:
                pnl_tp5.append(p['unrealized'][j])
                bars_tp5.append(j + 1)
                exited = True
                break
        if not exited:
            pnl_tp5.append(p['unrealized'][min(29, p['n_bars'] - 1)])
            bars_tp5.append(min(30, p['n_bars']))
    pnl_tp5 = np.array(pnl_tp5)
    bars_tp5 = np.array(bars_tp5)
    print(f"  {'Take profit at +5 or hold 30':<40} {len(pnl_tp5):>5} "
          f"{(pnl_tp5 > 0).mean():>5.0%} {pnl_tp5.mean():>+7.1f} "
          f"{bars_tp5.mean():>9.1f}")

    # Simple rule: exit when unrealized PnL first reaches +8 pts (or hold 30)
    pnl_tp8 = []
    bars_tp8 = []
    for p in paths:
        exited = False
        for j in range(p['n_bars']):
            if p['unrealized'][j] >= 8.0:
                pnl_tp8.append(p['unrealized'][j])
                bars_tp8.append(j + 1)
                exited = True
                break
        if not exited:
            pnl_tp8.append(p['unrealized'][min(29, p['n_bars'] - 1)])
            bars_tp8.append(min(30, p['n_bars']))
    pnl_tp8 = np.array(pnl_tp8)
    bars_tp8 = np.array(bars_tp8)
    print(f"  {'Take profit at +8 or hold 30':<40} {len(pnl_tp8):>5} "
          f"{(pnl_tp8 > 0).mean():>5.0%} {pnl_tp8.mean():>+7.1f} "
          f"{bars_tp8.mean():>9.1f}")

    # Trailing stop: exit when PnL drops 3 pts from MFE (after MFE >= 3)
    pnl_trail = []
    bars_trail = []
    for p in paths:
        exited = False
        for j in range(p['n_bars']):
            if p['mfe'][j] >= 3.0:
                giveback = p['mfe'][j] - p['unrealized'][j]
                if giveback >= 3.0:
                    pnl_trail.append(p['unrealized'][j])
                    bars_trail.append(j + 1)
                    exited = True
                    break
        if not exited:
            pnl_trail.append(p['unrealized'][min(29, p['n_bars'] - 1)])
            bars_trail.append(min(30, p['n_bars']))
    pnl_trail = np.array(pnl_trail)
    bars_trail = np.array(bars_trail)
    print(f"  {'Trail 3pt after MFE≥3, else hold 30':<40} {len(pnl_trail):>5} "
          f"{(pnl_trail > 0).mean():>5.0%} {pnl_trail.mean():>+7.1f} "
          f"{bars_trail.mean():>9.1f}")

    # Trailing stop: 5pt trail after MFE≥5
    pnl_trail5 = []
    bars_trail5 = []
    for p in paths:
        exited = False
        for j in range(p['n_bars']):
            if p['mfe'][j] >= 5.0:
                giveback = p['mfe'][j] - p['unrealized'][j]
                if giveback >= 5.0:
                    pnl_trail5.append(p['unrealized'][j])
                    bars_trail5.append(j + 1)
                    exited = True
                    break
        if not exited:
            pnl_trail5.append(p['unrealized'][min(29, p['n_bars'] - 1)])
            bars_trail5.append(min(30, p['n_bars']))
    pnl_trail5 = np.array(pnl_trail5)
    bars_trail5 = np.array(bars_trail5)
    print(f"  {'Trail 5pt after MFE≥5, else hold 30':<40} {len(pnl_trail5):>5} "
          f"{(pnl_trail5 > 0).mean():>5.0%} {pnl_trail5.mean():>+7.1f} "
          f"{bars_trail5.mean():>9.1f}")

    # Exit on score change: if trend score rises above 3 mid-trade, exit
    pnl_score_exit = []
    bars_score_exit = []
    for p in paths:
        td = p['trading_day']
        day_scores = rolling_scores.get(td, {})
        cps = sorted(day_scores.keys())
        exited = False
        for j in range(p['n_bars']):
            bar_off = p['offsets'][j] if j < len(p['offsets']) else p['bar_offset_entry'] + j + 1
            # Check if trend score at nearest checkpoint >= 3
            score = 0
            for cp in cps:
                if cp <= bar_off:
                    score = day_scores.get(cp, 0)
            if score >= 4.0 and j >= 5:  # Only exit after 5 bars
                pnl_score_exit.append(p['unrealized'][j])
                bars_score_exit.append(j + 1)
                exited = True
                break
        if not exited:
            pnl_score_exit.append(p['unrealized'][min(29, p['n_bars'] - 1)])
            bars_score_exit.append(min(30, p['n_bars']))
    pnl_score_exit = np.array(pnl_score_exit)
    bars_score_exit = np.array(bars_score_exit)
    print(f"  {'Exit if score≥4 (after 5 bars)':<40} {len(pnl_score_exit):>5} "
          f"{(pnl_score_exit > 0).mean():>5.0%} {pnl_score_exit.mean():>+7.1f} "
          f"{bars_score_exit.mean():>9.1f}")

    print(f"\n  Oracle ceiling: {pnl_oracle.mean():+.1f} PnL "
          f"(+{pnl_oracle.mean() - pnl_30.mean():.1f} vs fixed 30-bar)")

    # ══════════════════════════════════════════════════════════════════
    # PART 3: BAR-LEVEL DATASET
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 3: BAR-LEVEL DATASET CONSTRUCTION")
    print("=" * W)

    for lookahead in [10, 15, 20]:
        bar_data = build_bar_level_dataset(paths, rolling_scores, lookahead=lookahead)
        hold_rate = bar_data['should_hold'].mean()
        print(f"  Lookahead={lookahead}: {len(bar_data)} rows, "
              f"{bar_data['trading_day'].nunique()} days, "
              f"should_hold={hold_rate:.0%}")

    # Use lookahead=15 as primary
    bar_data = build_bar_level_dataset(paths, rolling_scores, lookahead=15)

    print(f"\n  Feature distributions:")
    for f in FEATURE_COLS:
        vals = bar_data[f].values
        print(f"    {f:<25} mean={vals.mean():>7.2f}  std={vals.std():>7.2f}  "
              f"min={vals.min():>7.2f}  max={vals.max():>7.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 4: WALK-FORWARD ML EXIT MODEL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 4: WALK-FORWARD EXIT MODEL TRAINING")
    print("=" * W)

    model_configs = [
        ('XGB shallow (d=2)', make_xgb_exit),
        ('Logistic Regression', make_logistic_exit),
    ]

    all_oos = {}
    for model_name, factory in model_configs:
        print(f"\n  {model_name}:")
        oos = walk_forward_exit_model(bar_data, paths, factory)
        if len(oos) > 0:
            all_oos[model_name] = oos
            overall_auc = roc_auc_score(oos['should_hold'].values, oos['p_hold'].values)
            print(f"    Overall AUC: {overall_auc:.3f}")

    # ══════════════════════════════════════════════════════════════════
    # PART 5: EXIT SIMULATION — ML vs baselines
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 5: EXIT SIMULATION — ML vs baselines")
    print("=" * W)

    # Restrict paths to OOS days only (for fair comparison)
    if all_oos:
        oos_ref = list(all_oos.values())[0]
        oos_days = set(oos_ref['trading_day'].unique())
        oos_paths = [p for p in paths if p['trading_day'] in oos_days]
        oos_sigs = sigs[sigs['trading_day'].isin(oos_days)]
    else:
        oos_paths = paths
        oos_sigs = sigs

    print(f"\n  OOS signals: {len(oos_paths)} ({len(oos_days)} days)")
    print(f"\n  {'Strategy':<50} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18} {'Avg bars':>9}")

    # Baselines on OOS paths only
    pnl_30_oos = np.array([p['unrealized'][min(29, p['n_bars'] - 1)] for p in oos_paths])
    mean_30, ci_lo_30, ci_hi_30 = bootstrap_ci(pnl_30_oos)
    print(f"  {'Fixed 30-bar hold':<50} {len(pnl_30_oos):>5} "
          f"{(pnl_30_oos > 0).mean():>5.0%} {mean_30:>+7.1f}  "
          f"[{ci_lo_30:>+7.1f}, {ci_hi_30:>+7.1f}] {'30':>9}")

    # Stop+target baseline
    st_res = evaluate_stop_target(oos_sigs, rth, 15, target_mode='midpoint')
    fade_st = st_res[st_res['signal_type'].isin(['fade_high', 'fade_low'])]
    if len(fade_st) > 0:
        mean_st, ci_lo_st, ci_hi_st = bootstrap_ci(fade_st['pnl'].values)
        print(f"  {'15pt stop + midpoint target':<50} {len(fade_st):>5} "
              f"{fade_st['win'].mean():>5.0%} {mean_st:>+7.1f}  "
              f"[{ci_lo_st:>+7.1f}, {ci_hi_st:>+7.1f}] {'—':>9}")

    # Simple rule baselines on OOS paths
    for rule_name, rule_fn in [
        ('Take profit at +5 or hold 30', lambda p, j: p['unrealized'][j] >= 5.0),
        ('Take profit at +8 or hold 30', lambda p, j: p['unrealized'][j] >= 8.0),
        ('Trail 3pt from MFE≥3', lambda p, j: p['mfe'][j] >= 3.0 and (p['mfe'][j] - p['unrealized'][j]) >= 3.0),
    ]:
        pnls = []
        bars = []
        for p in oos_paths:
            exited = False
            for j in range(p['n_bars']):
                if rule_fn(p, j):
                    pnls.append(p['unrealized'][j])
                    bars.append(j + 1)
                    exited = True
                    break
            if not exited:
                pnls.append(p['unrealized'][min(29, p['n_bars'] - 1)])
                bars.append(min(30, p['n_bars']))
        pnls = np.array(pnls)
        bars = np.array(bars)
        mean_r, ci_lo_r, ci_hi_r = bootstrap_ci(pnls)
        print(f"  {rule_name:<50} {len(pnls):>5} "
              f"{(pnls > 0).mean():>5.0%} {mean_r:>+7.1f}  "
              f"[{ci_lo_r:>+7.1f}, {ci_hi_r:>+7.1f}] {bars.mean():>9.1f}")

    # ML exit strategies
    for model_name, oos_preds in all_oos.items():
        for threshold in [0.3, 0.4, 0.5, 0.6]:
            res = simulate_ml_exit(oos_paths, bar_data, oos_preds,
                                   threshold=threshold, max_hold=60)
            if len(res) == 0:
                continue
            mean_ml, ci_lo_ml, ci_hi_ml = bootstrap_ci(res['pnl'].values)
            avg_bars = res['bars_held'].mean()
            exit_types = res['exit_type'].value_counts(normalize=True)
            ml_pct = exit_types.get('ml_exit', 0)
            print(f"  {f'{model_name} / exit@P<{threshold}':<50} {len(res):>5} "
                  f"{res['win'].mean():>5.0%} {mean_ml:>+7.1f}  "
                  f"[{ci_lo_ml:>+7.1f}, {ci_hi_ml:>+7.1f}] {avg_bars:>8.1f}b"
                  f"  (ml_exit={ml_pct:.0%})")

        # ML + stop
        for threshold in [0.4, 0.5]:
            res = simulate_ml_exit(oos_paths, bar_data, oos_preds,
                                   threshold=threshold, max_hold=60,
                                   stop_pts=15)
            if len(res) == 0:
                continue
            mean_ml, ci_lo_ml, ci_hi_ml = bootstrap_ci(res['pnl'].values)
            avg_bars = res['bars_held'].mean()
            exit_types = res['exit_type'].value_counts(normalize=True)
            print(f"  {f'{model_name} / exit@P<{threshold} + 15pt stop':<50} {len(res):>5} "
                  f"{res['win'].mean():>5.0%} {mean_ml:>+7.1f}  "
                  f"[{ci_lo_ml:>+7.1f}, {ci_hi_ml:>+7.1f}] {avg_bars:>8.1f}b"
                  f"  ({dict(exit_types)})")

    # ══════════════════════════════════════════════════════════════════
    # PART 6: CONDITIONAL ANALYSIS — when does ML help most?
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 6: CONDITIONAL ANALYSIS — when does ML add value?")
    print("=" * W)

    if all_oos:
        # Use best model
        best_model = max(all_oos.keys(), key=lambda k: roc_auc_score(
            all_oos[k]['should_hold'], all_oos[k]['p_hold']))
        oos_preds = all_oos[best_model]
        print(f"\n  Using {best_model}")

        # Best threshold from Part 5 (pick the one with highest PnL)
        best_thresh = 0.5
        best_pnl = -999
        for t in [0.3, 0.4, 0.5, 0.6]:
            res = simulate_ml_exit(oos_paths, bar_data, oos_preds,
                                   threshold=t, max_hold=60)
            if len(res) > 0 and res['pnl'].mean() > best_pnl:
                best_pnl = res['pnl'].mean()
                best_thresh = t
        print(f"  Best exit threshold: P<{best_thresh}")

        # Analysis by entry time
        print(f"\n  By entry time:")
        print(f"  {'Period':<20} {'Fixed30':>12} {'ML exit':>12} {'Diff':>8}")
        for label, lo, hi in [('Early <1h', 0, 60), ('Mid 1-3.5h', 60, 210), ('Late >3.5h', 210, 999)]:
            period_paths = [p for p in oos_paths if lo <= p['bar_offset_entry'] < hi]
            if len(period_paths) < 10:
                continue
            fix_pnl = np.mean([p['unrealized'][min(29, p['n_bars'] - 1)] for p in period_paths])
            ml_res = simulate_ml_exit(period_paths, bar_data, oos_preds,
                                      threshold=best_thresh, max_hold=60)
            if len(ml_res) > 0:
                ml_pnl = ml_res['pnl'].mean()
                diff = ml_pnl - fix_pnl
                print(f"  {label:<20} {fix_pnl:>+12.1f} {ml_pnl:>+12.1f} {diff:>+8.1f}")

        # Analysis by quick movers vs choppers
        print(f"\n  By trade character:")
        print(f"  {'Type':<20} {'Fixed30':>12} {'ML exit':>12} {'Diff':>8} {'n':>5}")
        for label, selector in [
            ('Quick movers', lambda p: p['mfe'][min(9, p['n_bars']-1)] >= 5.0),
            ('Choppers', lambda p: np.sum(np.abs(p['unrealized'][:min(30, p['n_bars'])]) < 3.0) > 20),
            ('Losers (MAE>8)', lambda p: p['mae'][min(29, p['n_bars']-1)] >= 8.0),
        ]:
            subset = [p for p in oos_paths if selector(p)]
            if len(subset) < 5:
                continue
            fix_pnl = np.mean([p['unrealized'][min(29, p['n_bars'] - 1)] for p in subset])
            ml_res = simulate_ml_exit(subset, bar_data, oos_preds,
                                      threshold=best_thresh, max_hold=60)
            if len(ml_res) > 0:
                ml_pnl = ml_res['pnl'].mean()
                diff = ml_pnl - fix_pnl
                print(f"  {label:<20} {fix_pnl:>+12.1f} {ml_pnl:>+12.1f} "
                      f"{diff:>+8.1f} {len(subset):>5}")

    # ══════════════════════════════════════════════════════════════════
    # PART 7: ROBUSTNESS — vary lookahead and threshold
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * W}")
    print("PART 7: ROBUSTNESS — different lookaheads and exit thresholds")
    print("=" * W)

    for lookahead in [10, 20]:
        print(f"\n  Lookahead = {lookahead} bars:")
        bar_data_alt = build_bar_level_dataset(paths, rolling_scores, lookahead=lookahead)
        oos_alt = walk_forward_exit_model(bar_data_alt, paths, make_xgb_exit)
        if len(oos_alt) == 0:
            print("    (no OOS predictions)")
            continue

        try:
            auc = roc_auc_score(oos_alt['should_hold'].values, oos_alt['p_hold'].values)
            print(f"    Overall AUC: {auc:.3f}")
        except ValueError:
            pass

        for threshold in [0.4, 0.5, 0.6]:
            res = simulate_ml_exit(oos_paths, bar_data_alt, oos_alt,
                                   threshold=threshold, max_hold=60)
            if len(res) == 0:
                continue
            mean_ml, ci_lo_ml, ci_hi_ml = bootstrap_ci(res['pnl'].values)
            avg_bars = res['bars_held'].mean()
            print(f"    exit@P<{threshold}: n={len(res)}, WR={res['win'].mean():.0%}, "
                  f"PnL={mean_ml:+.1f} [{ci_lo_ml:+.1f}, {ci_hi_ml:+.1f}], "
                  f"avg_bars={avg_bars:.1f}")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
