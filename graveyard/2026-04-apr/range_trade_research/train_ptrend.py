"""Train walk-forward P(trend) model and compare against heuristic score.

Builds a binary XGBoost classifier that outputs P(trend) at each checkpoint,
using the same rich feature set as the Dalton classifier (VBP metrics, IB
extensions, OTF, OR breakouts, etc).

Walk-forward: train on earlier days, predict later days (same fold structure
as the conditional range model).

Comparison:
  - ML P(trend) as fade filter vs heuristic trend score as fade filter
  - Calibration: does P(trend) predict actual trend days?
  - Combined: can ML identify the "score 0-2" regime more precisely?
"""

import math
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from scipy.signal import find_peaks

from vp.dalton_day_classifier import VolumeProfileBuilder, DayLabeler
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
    label_days_eod,
    build_rolling_trend_scores,
    compute_trend_score_at_minute,
)


# ──────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION AT CHECKPOINTS
# ──────────────────────────────────────────────────────────────────────

def extract_checkpoint_features(day_bars, checkpoint_minute):
    """Extract features from the first `checkpoint_minute` bars of a day.

    Returns a dict of features, or None if insufficient data.
    """
    if len(day_bars) < checkpoint_minute or checkpoint_minute < 10:
        return None

    prices = day_bars['close'].values[:checkpoint_minute]
    highs = day_bars['high'].values[:checkpoint_minute]
    lows = day_bars['low'].values[:checkpoint_minute]
    opens = day_bars['open'].values[:checkpoint_minute]
    volumes = day_bars['volume'].values[:checkpoint_minute] if 'volume' in day_bars.columns else np.ones(checkpoint_minute)

    T = checkpoint_minute
    running_high = highs.max()
    running_low = lows.min()
    running_range = max(0.5, running_high - running_low)
    open_price = prices[0]
    current_price = prices[-1]

    # ── Volume profile metrics ──
    vp_builder = VolumeProfileBuilder(bin_size=0.5)
    bin_centers, per_minute_vbp = vp_builder.build_minute_vbp_matrix(prices, volumes)
    cum_vbp = per_minute_vbp.sum(axis=0)

    poc_price = vp_builder.compute_poc(cum_vbp, bin_centers)
    va_low, va_high = vp_builder.compute_va70(cum_vbp, bin_centers)
    entropy = vp_builder.compute_entropy(cum_vbp)
    n_peaks, peak_sep = vp_builder.find_peaks_in_profile(cum_vbp)

    poc_rel = (poc_price - running_low) / running_range
    va_width_rel = (va_high - va_low) / running_range

    # ── OTF (one-time-framing) ──
    rh = np.maximum.accumulate(highs)
    rl = np.minimum.accumulate(lows)
    new_highs = np.sum(rh[1:] > rh[:-1])
    new_lows = np.sum(rl[1:] < rl[:-1])
    total_ext = new_highs + new_lows
    otf_ratio = abs(new_highs - new_lows) / max(1, total_ext)
    otf_direction = (new_highs - new_lows) / max(1, total_ext)

    # ── VWAP ──
    cum_pv = np.sum(prices * volumes)
    cum_vol = np.sum(volumes)
    vwap = cum_pv / max(1.0, cum_vol)
    price_vs_vwap = (current_price - vwap) / max(1.0, vwap) * 100

    # ── Opening range breakouts ──
    def or_breakout(n_bars):
        if T < n_bars:
            return 0.0, 0.0
        or_h = highs[:n_bars].max()
        or_l = lows[:n_bars].min()
        or_w = or_h - or_l + 0.01
        if current_price > or_h:
            return (current_price - or_h) / or_w, or_w / running_range
        elif current_price < or_l:
            return (current_price - or_l) / or_w, or_w / running_range
        return 0.0, or_w / running_range

    or5_break, or5_width_rel = or_breakout(5) if T >= 5 else (0.0, 0.0)
    or15_break, or15_width_rel = or_breakout(15) if T >= 15 else (0.0, 0.0)
    or30_break, or30_width_rel = or_breakout(30) if T >= 30 else (0.0, 0.0)

    # ── IB (initial balance, first 60 min) ──
    ib_high = highs[:min(60, T)].max()
    ib_low = lows[:min(60, T)].min()
    ib_range = max(0.5, ib_high - ib_low)

    if T >= 60:
        ib_high_dist = (current_price - ib_high) / ib_range
        ib_low_dist = (ib_low - current_price) / ib_range
        range_ext_up = max(0, running_high - ib_high) / ib_range
        range_ext_down = max(0, ib_low - running_low) / ib_range
    else:
        ib_high_dist = 0.0
        ib_low_dist = 0.0
        range_ext_up = 0.0
        range_ext_down = 0.0

    # ── Price dynamics ──
    move_from_open = (current_price - open_price) / running_range
    close_in_range = (current_price - running_low) / running_range

    ret_1m = (prices[-1] - prices[-2]) / prices[-2] if T >= 2 else 0.0
    ret_5m = (prices[-1] - prices[-6]) / prices[-6] if T >= 6 else 0.0

    # ── Range expansion rate ──
    range_exp = 0.0
    if T >= 10:
        prev_range = highs[:T-5].max() - lows[:T-5].min()
        if prev_range > 0:
            range_exp = (running_range - prev_range) / prev_range

    # ── Volume dynamics ──
    vol_share = volumes[-1] / max(1.0, volumes.mean()) if T > 0 else 1.0
    cum_delta_rate = 0.0  # No bid/ask split in our data

    # ── Bar structure features ──
    # Fraction of bars that are directional (close > open for up, close < open for down)
    up_bars = np.sum(prices > opens) / T
    # Average bar range relative to day range
    bar_ranges = highs - lows
    avg_bar_range = bar_ranges.mean() / running_range

    # ── Heuristic trend score (for comparison/as feature) ──
    heuristic_score = 0.0
    if otf_ratio > 0.65:
        heuristic_score += 2.5
    elif otf_ratio > 0.50:
        heuristic_score += 1.5
    if va_width_rel < 0.40:
        heuristic_score += 2.0
    elif va_width_rel < 0.55:
        heuristic_score += 1.0
    if poc_rel > 0.70 or poc_rel < 0.30:
        heuristic_score += 1.5
    elif poc_rel > 0.60 or poc_rel < 0.40:
        heuristic_score += 0.75
    if abs(or15_break) > 1.0:
        heuristic_score += 2.0
    elif abs(or15_break) > 0.5:
        heuristic_score += 1.0
    if abs(move_from_open) > 0.6:
        heuristic_score += 1.5
    elif abs(move_from_open) > 0.4:
        heuristic_score += 0.75

    return {
        'minute_norm': T / 450,
        # VBP shape
        'poc_rel': poc_rel,
        'va_width_rel': va_width_rel,
        'entropy': entropy,
        'n_peaks': n_peaks,
        'peak_sep': peak_sep,
        # OTF
        'otf_ratio': otf_ratio,
        'otf_direction': otf_direction,
        # VWAP
        'price_vs_vwap': price_vs_vwap,
        # OR breakouts
        'or5_break': or5_break,
        'or15_break': or15_break,
        'or30_break': or30_break,
        'or5_width_rel': or5_width_rel,
        'or15_width_rel': or15_width_rel,
        'or30_width_rel': or30_width_rel,
        # IB
        'ib_high_dist': ib_high_dist,
        'ib_low_dist': ib_low_dist,
        'range_ext_up': range_ext_up,
        'range_ext_down': range_ext_down,
        # Price dynamics
        'move_from_open': move_from_open,
        'close_in_range': close_in_range,
        'ret_1m': ret_1m,
        'ret_5m': ret_5m,
        'range_exp': range_exp,
        # Volume
        'vol_share': vol_share,
        # Bar structure
        'up_bar_frac': up_bars,
        'avg_bar_range_rel': avg_bar_range,
        # Heuristic (included for comparison, can ablate)
        'heuristic_score': heuristic_score,
    }


FEATURE_COLS = [
    'minute_norm', 'poc_rel', 'va_width_rel', 'entropy', 'n_peaks', 'peak_sep',
    'otf_ratio', 'otf_direction', 'price_vs_vwap',
    'or5_break', 'or15_break', 'or30_break',
    'or5_width_rel', 'or15_width_rel', 'or30_width_rel',
    'ib_high_dist', 'ib_low_dist', 'range_ext_up', 'range_ext_down',
    'move_from_open', 'close_in_range', 'ret_1m', 'ret_5m', 'range_exp',
    'vol_share', 'up_bar_frac', 'avg_bar_range_rel',
    'heuristic_score',
]

FEATURE_COLS_NO_HEURISTIC = [c for c in FEATURE_COLS if c != 'heuristic_score']


# ──────────────────────────────────────────────────────────────────────
# DATASET BUILDING
# ──────────────────────────────────────────────────────────────────────

def build_ptrend_dataset(df_1min, checkpoints):
    """Build dataset: one row per (day, checkpoint) with features and EOD label.

    Returns DataFrame with columns: trading_day, checkpoint, features..., label
    """
    rth = df_1min[df_1min['ovn'] == 0].copy()
    eod_labeler = DayLabeler(binary_labels=True)

    rows = []
    for td, grp in rth.groupby('trading_day'):
        # EOD label
        prices_all = grp['close'].values
        volumes_all = grp['volume'].values if 'volume' in grp.columns else np.ones(len(grp))
        highs_all = grp['high'].values
        lows_all = grp['low'].values

        eod_label = eod_labeler.heuristic_label(
            prices_all, volumes_all,
            grp['open'].values[0], prices_all[-1],
            highs_all, lows_all)
        is_trend = 1 if eod_label == 'Trend' else 0

        for cp in checkpoints:
            feats = extract_checkpoint_features(grp, cp)
            if feats is None:
                continue
            feats['trading_day'] = td
            feats['checkpoint'] = cp
            feats['label'] = is_trend
            rows.append(feats)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# WALK-FORWARD TRAINING
# ──────────────────────────────────────────────────────────────────────

def walk_forward_ptrend(dataset, n_folds=5, min_train_days=100,
                        feature_cols=None):
    """Walk-forward train/test for P(trend) model.

    Returns DataFrame with OOS predictions.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    days = sorted(dataset['trading_day'].unique())
    n_days = len(days)
    fold_size = n_days // n_folds

    oos_results = []

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
        y_train = train['label'].values
        X_test = test[feature_cols].values

        # Train XGBoost binary classifier
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / max(1, n_pos)

        model = XGBClassifier(
            objective='binary:logistic',
            max_depth=3,
            learning_rate=0.02,
            n_estimators=500,
            min_child_weight=20,
            reg_alpha=1.0,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Predict probabilities
        p_trend = model.predict_proba(X_test)[:, 1]

        test_out = test[['trading_day', 'checkpoint', 'label',
                         'heuristic_score']].copy()
        test_out['p_trend'] = p_trend
        test_out['fold'] = fold
        oos_results.append(test_out)

        print(f"  Fold {fold}: train={len(train)} rows ({len(train_days)} days), "
              f"test={len(test)} rows ({len(test_days)} days), "
              f"AUC={_roc_auc(test_out['label'].values, p_trend):.3f}")

        # Feature importance for last fold
        if fold == n_folds - 1:
            imp = pd.Series(model.feature_importances_, index=feature_cols)
            imp = imp.sort_values(ascending=False)
            print(f"\n  Feature importance (last fold):")
            for feat, val in imp.head(10).items():
                print(f"    {feat:<25} {val:.3f}")

    return pd.concat(oos_results, ignore_index=True)


def _roc_auc(y_true, y_score):
    """Simple ROC AUC."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


# ──────────────────────────────────────────────────────────────────────
# COMPARISON WITH HEURISTIC
# ──────────────────────────────────────────────────────────────────────

def compare_filters(sigs_all, rth, rolling_scores, oos_preds, checkpoints):
    """Compare ML P(trend) vs heuristic score as fade filters."""

    def get_ml_ptrend(sig, oos_preds):
        """Get the ML P(trend) prediction for a signal."""
        td = sig['trading_day']
        bar = sig['bar_offset']
        day_preds = oos_preds[oos_preds['trading_day'] == td]
        if len(day_preds) == 0:
            return np.nan
        # Find most recent checkpoint <= bar
        valid = day_preds[day_preds['checkpoint'] <= bar]
        if len(valid) == 0:
            return day_preds.iloc[0]['p_trend']
        return valid.iloc[-1]['p_trend']

    def get_heuristic_score(sig, rolling_scores, checkpoints):
        td = sig['trading_day']
        bar = sig['bar_offset']
        if td not in rolling_scores:
            return np.nan
        day_scores = rolling_scores[td]
        recent_cp = None
        for cp in checkpoints:
            if cp <= bar:
                recent_cp = cp
        return day_scores.get(recent_cp, 0) if recent_cp else 0

    # Get scores for all signals
    ml_scores = []
    heur_scores = []
    for _, sig in sigs_all.iterrows():
        ml_scores.append(get_ml_ptrend(sig, oos_preds))
        heur_scores.append(get_heuristic_score(sig, rolling_scores, checkpoints))

    return np.array(ml_scores), np.array(heur_scores)


def fmt(val, kind='pct'):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'n/a'.rjust(7)
    if kind == 'pct':
        return f"{val:.0%}".rjust(7)
    elif kind == 'pnl':
        return f"{val:+.1f}".rjust(7)
    return f"{val}".rjust(7)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return np.nan, np.nan, np.nan
    means = [np.random.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    means = np.array(means)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return values.mean(), lo, hi


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    W = 95
    print("=" * W)
    print("P(TREND) ML MODEL — walk-forward training & comparison vs heuristic")
    print("=" * W)

    df_1min, daily = load_data()
    print(f"\nLoaded {len(df_1min):,} 1-min bars → {len(daily)} daily bars")

    checkpoints = [15, 30, 60, 90, 120, 150, 210, 270, 330]

    # ── Build dataset ──
    print("\nBuilding P(trend) checkpoint dataset...")
    dataset = build_ptrend_dataset(df_1min, checkpoints)
    n_trend = dataset.groupby('trading_day')['label'].first().sum()
    n_total = dataset['trading_day'].nunique()
    print(f"  {len(dataset)} rows ({n_total} days, {n_trend} trend, "
          f"{n_total - n_trend} balance)")

    # ── Walk-forward training ──
    print(f"\n{'=' * W}")
    print("WALK-FORWARD TRAINING — P(trend) binary XGBoost")
    print("=" * W)

    print("\n  Full feature set (28 features):")
    oos_full = walk_forward_ptrend(dataset, n_folds=5, min_train_days=100,
                                    feature_cols=FEATURE_COLS)

    print(f"\n  Without heuristic_score feature (27 features):")
    oos_no_heur = walk_forward_ptrend(dataset, n_folds=5, min_train_days=100,
                                       feature_cols=FEATURE_COLS_NO_HEURISTIC)

    # ── Calibration ──
    print(f"\n{'=' * W}")
    print("CALIBRATION — does P(trend) predict actual trend days?")
    print("=" * W)

    for name, oos in [('ML (full)', oos_full), ('ML (no heuristic)', oos_no_heur)]:
        print(f"\n  {name}:")
        print(f"  {'P(trend) bin':<20} {'n':>5} {'actual trend%':>14} {'calibration':>12}")
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5), (0.5, 0.6),
                       (0.6, 0.8), (0.8, 1.0)]:
            sub = oos[(oos['p_trend'] >= lo) & (oos['p_trend'] < hi)]
            if len(sub) >= 10:
                actual = sub['label'].mean()
                expected = (lo + hi) / 2
                cal = actual / expected if expected > 0 else 0
                print(f"  [{lo:.1f}, {hi:.1f})          {len(sub):>5} {actual:>14.1%} "
                      f"{cal:>12.2f}×")

    # Compare heuristic score calibration
    print(f"\n  Heuristic score:")
    print(f"  {'Score range':<20} {'n':>5} {'actual trend%':>14}")
    for lo, hi, label in [(0, 2, '[0, 2)'), (2, 4, '[2, 4)'),
                           (4, 5, '[4, 5)'), (5, 7, '[5, 7)'), (7, 15, '[7, 15)')]:
        sub = oos_full[(oos_full['heuristic_score'] >= lo) &
                       (oos_full['heuristic_score'] < hi)]
        if len(sub) >= 10:
            actual = sub['label'].mean()
            print(f"  {label:<20} {len(sub):>5} {actual:>14.1%}")

    # ── Build range model + apply filters ──
    print(f"\n{'=' * W}")
    print("FILTER COMPARISON — ML P(trend) vs heuristic for conditional proximity fade")
    print("=" * W)

    print("\nBuilding static Model-OOS...")
    static_oos = _generate_oos_predictions(daily, 'daily', n_folds=5,
                                           min_train_days=100)
    static_oos = static_oos.rename(columns={
        'pred_range_high': 'static_high', 'pred_range_low': 'static_low',
    })

    print("Building conditional model (walk-forward)...")
    trainer = ConditionalRangeTrainer()
    cond_oos = trainer.walk_forward_evaluate(df_1min, daily, verbose=True)

    print("\nAssigning bar-level predictions...")
    rth = build_bar_level_predictions(df_1min, daily, cond_oos, static_oos)
    n_days = rth['trading_day'].nunique()
    print(f"  {len(rth):,} RTH bars across {n_days} OOS days")

    rolling_scores = build_rolling_trend_scores(df_1min, checkpoints)

    prox_pts = 10
    sigs_all = find_fade_proximity_signals(rth, 'cond', prox_pts)
    print(f"  {len(sigs_all)} proximity fade signals")

    # Get ML and heuristic scores for each signal
    print("\nAssigning ML P(trend) and heuristic scores to signals...")
    ml_scores, heur_scores = compare_filters(
        sigs_all, rth, rolling_scores, oos_full, checkpoints)

    # Evaluate fade PnL
    fade_res = evaluate_fixed_hold(sigs_all, rth, [30])
    fade_30 = fade_res[fade_res['signal_type'].isin(['fade_high', 'fade_low'])]
    pnls = fade_30['pnl'].values

    # Drop signals without ML predictions
    valid = ~np.isnan(ml_scores)
    ml_v = ml_scores[valid]
    heur_v = heur_scores[valid]
    pnls_v = pnls[valid]
    sigs_v = sigs_all[valid]

    print(f"  {len(ml_v)} signals with valid ML predictions")

    # ── Head-to-head comparison ──
    print(f"\n  {'Filter':<35} {'n':>5} {'WR':>6} {'PnL':>7}  {'95% CI':>18}")

    # No filter
    mean, ci_lo, ci_hi = bootstrap_ci(pnls_v)
    print(f"  {'No filter':<35} {len(pnls_v):>5} {(pnls_v>0).mean():>5.0%} "
          f"{mean:>+7.1f}  [{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # Heuristic thresholds
    for thresh in [3.0, 4.0, 5.0]:
        mask = heur_v < thresh
        bp = pnls_v[mask]
        if len(bp) >= 5:
            mean, ci_lo, ci_hi = bootstrap_ci(bp)
            print(f"  {f'Heuristic < {thresh:.0f}':<35} {len(bp):>5} "
                  f"{(bp>0).mean():>5.0%} {mean:>+7.1f}  "
                  f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # ML thresholds
    for thresh in [0.3, 0.4, 0.5, 0.6]:
        mask = ml_v < thresh
        bp = pnls_v[mask]
        if len(bp) >= 5:
            mean, ci_lo, ci_hi = bootstrap_ci(bp)
            print(f"  {f'ML P(trend) < {thresh:.1f}':<35} {len(bp):>5} "
                  f"{(bp>0).mean():>5.0%} {mean:>+7.1f}  "
                  f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    # ── PnL by ML score bucket ──
    print(f"\n  PnL by ML P(trend) bucket:")
    print(f"  {'P(trend) bin':<18} {'n':>5} {'WR':>6} {'PnL':>7}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (ml_v >= lo) & (ml_v < hi)
        bp = pnls_v[mask]
        if len(bp) >= 5:
            print(f"  [{lo:.1f}, {hi:.1f})          {len(bp):>5} "
                  f"{(bp>0).mean():>5.0%} {bp.mean():>+7.1f}")

    # ── Correlation between ML and heuristic ──
    corr = np.corrcoef(ml_v, heur_v)[0, 1]
    rank_corr = pd.Series(ml_v).rank().corr(pd.Series(heur_v).rank())
    print(f"\n  ML ↔ Heuristic correlation: r={corr:.3f}, rank={rank_corr:.3f}")

    # ── Stop+target comparison ──
    print(f"\n  Stop=15 / target=midpoint:")
    print(f"  {'Filter':<35} {'n':>5} {'WR':>6} {'PnL':>7} {'stop%':>6} {'tgt%':>6}")

    for label, mask in [
        ('No filter', np.ones(len(sigs_v), dtype=bool)),
        ('Heuristic < 3', heur_v < 3),
        ('ML P(trend) < 0.4', ml_v < 0.4),
        ('ML P(trend) < 0.5', ml_v < 0.5),
    ]:
        sb = sigs_v[mask]
        if len(sb) < 5:
            continue
        st = evaluate_stop_target(sb, rth, 15, target_mode='midpoint')
        fade_st = st[st['signal_type'].isin(['fade_high', 'fade_low'])]
        if len(fade_st) < 5:
            continue
        wr = fade_st['win'].mean()
        avg = fade_st['pnl'].mean()
        pst = (fade_st['exit_type'] == 'stop').mean()
        ptg = (fade_st['exit_type'] == 'target').mean()
        print(f"  {label:<35} {len(fade_st):>5} {wr:>5.0%} {avg:>+7.1f} "
              f"{pst:>5.0%} {ptg:>5.0%}")

    # ── Does ML add anything beyond heuristic? ──
    print(f"\n  Does ML add beyond heuristic? (Heuristic < 3 AND ML filter):")
    base_mask = heur_v < 3
    base_bp = pnls_v[base_mask]
    mean_base, _, _ = bootstrap_ci(base_bp)
    print(f"  {'Heuristic < 3 alone':<35} {len(base_bp):>5} "
          f"{(base_bp>0).mean():>5.0%} {mean_base:>+7.1f}")

    for ml_thresh in [0.3, 0.4, 0.5]:
        mask = base_mask & (ml_v < ml_thresh)
        bp = pnls_v[mask]
        if len(bp) >= 10:
            mean, ci_lo, ci_hi = bootstrap_ci(bp)
            print(f"  {f'+ ML < {ml_thresh}':<35} {len(bp):>5} "
                  f"{(bp>0).mean():>5.0%} {mean:>+7.1f}  "
                  f"[{ci_lo:>+7.1f}, {ci_hi:>+7.1f}]")

    print(f"\n{'=' * W}")


if __name__ == '__main__':
    main()
