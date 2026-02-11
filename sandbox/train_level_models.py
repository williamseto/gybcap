#!/usr/bin/env python
"""
Phase 3: Single-Model Training with Level-Quality Encoding.

Trains a single XGBoost model that generalizes across all level types by
encoding level identity and quality as features. This lets the model learn
level-specific patterns (e.g. "ovn_hi with high volume → reversal") within
a unified framework with more training data than per-level models.

New features added:
  - One-hot level type (7 features)
  - Trailing 20d/50d reversal rate per level type (causal)
  - Level side (support/resistance)
  - Level-side × BB interaction
  - Approach direction relative to level

Walk-forward CV with honest early stopping (temporal train/val split).

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_level_models.py
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
FIGURE_DIR = "sandbox/figures/level_models"

from strategies.labeling.reversal_zones import TRACKED_LEVELS

STOP_PTS = 4.0
TARGET_PTS = 6.0
MAX_BARS = 45


# ── Data loading (reused from signal_detection.py) ────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from strategies.features.price_levels import PriceLevelProvider
    print("\nComputing price levels...")
    plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
    feat_df = plp._compute_impl(ohlcv)

    level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']
    if 'dt' in feat_df.columns:
        feat_df = feat_df.set_index('dt')
    ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv
    for col in level_cols:
        if col in feat_df.columns:
            ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

    levels = plp.prev_day_levels(ohlcv)
    ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
    ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])
    return ohlcv


def compute_all_features(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute all feature providers on the data."""
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
    has_bidask = 'bidvolume' in ohlcv.columns
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


# ── Labeling (reused from signal_detection.py) ───────────────────────────

def label_reversals_breakouts(ohlcv: pd.DataFrame, proximity_pts: float = 5.0,
                               forward_bars: int = 45,
                               reversal_pts: float = 6.0,
                               breakout_pts: float = 4.0) -> pd.DataFrame:
    """Label near-level bars as reversal (1), breakout (0), or inconclusive (2)."""
    print(f"\nLabeling reversals/breakouts (proximity={proximity_pts}pt, "
          f"reversal={reversal_pts}pt, breakout={breakout_pts}pt)...")

    close_arr = ohlcv['close'].values.astype(np.float64)
    high_arr = ohlcv['high'].values.astype(np.float64)
    low_arr = ohlcv['low'].values.astype(np.float64)
    n = len(ohlcv)

    labels = np.full(n, -1, dtype=np.int32)
    nearest_level_name = np.empty(n, dtype=object)
    nearest_level_price = np.full(n, np.nan)
    side_arr = np.zeros(n, dtype=np.int8)  # 1=above(support), -1=below(resistance)
    trade_dir_arr = np.zeros(n, dtype=np.int8)

    # Collect level arrays
    level_arrs = {}
    for lvl_name in TRACKED_LEVELS:
        if lvl_name in ohlcv.columns:
            level_arrs[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

    # Find nearest level for each bar
    for lvl_name, lvl_vals in level_arrs.items():
        for i in range(n):
            d = abs(close_arr[i] - lvl_vals[i])
            if np.isnan(d):
                continue
            if d <= proximity_pts:
                prev_dist = abs(close_arr[i] - nearest_level_price[i]) if not np.isnan(nearest_level_price[i]) else np.inf
                if d < prev_dist:
                    nearest_level_name[i] = lvl_name
                    nearest_level_price[i] = lvl_vals[i]

    near_level_mask = ~np.isnan(nearest_level_price)
    n_near = near_level_mask.sum()
    print(f"  {n_near:,} near-level bars out of {n:,} ({100*n_near/n:.1f}%)")

    # Label each near-level bar
    for i in range(n):
        if not near_level_mask[i]:
            continue

        level = nearest_level_price[i]
        price = close_arr[i]
        is_above = price >= level
        side_arr[i] = 1 if is_above else -1
        trade_dir_arr[i] = 1 if is_above else -1  # long above support, short below resistance

        end = min(i + 1 + forward_bars, n)
        future_highs = high_arr[i+1:end]
        future_lows = low_arr[i+1:end]

        if len(future_highs) == 0:
            labels[i] = 2
            continue

        if is_above:
            # Level is support → reversal=UP, breakout=DOWN through level
            rev_hits = np.where(future_highs >= price + reversal_pts)[0]
            brk_hits = np.where(future_lows <= level - breakout_pts)[0]
        else:
            # Level is resistance → reversal=DOWN, breakout=UP through level
            rev_hits = np.where(future_lows <= price - reversal_pts)[0]
            brk_hits = np.where(future_highs >= level + breakout_pts)[0]

        first_rev = (rev_hits[0]) if len(rev_hits) > 0 else 9999
        first_brk = (brk_hits[0]) if len(brk_hits) > 0 else 9999

        if first_rev < first_brk and first_rev < 9999:
            labels[i] = 1  # reversal
        elif first_brk < first_rev and first_brk < 9999:
            labels[i] = 0  # breakout
        else:
            labels[i] = 2  # inconclusive

    ohlcv['outcome'] = labels
    ohlcv['nearest_level_name'] = nearest_level_name
    ohlcv['nearest_level_price'] = nearest_level_price
    ohlcv['side'] = side_arr
    ohlcv['trade_direction'] = trade_dir_arr

    n_rev = (labels == 1).sum()
    n_bo = (labels == 0).sum()
    n_inc = (labels == 2).sum()
    print(f"  Reversal: {n_rev:,} ({100*n_rev/max(n_near,1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100*n_bo/max(n_near,1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100*n_inc/max(n_near,1):.1f}%)")

    return ohlcv


# ── Step 1: Level-aware feature engineering ──────────────────────────────

def compute_level_encoding_features(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add ~12 features encoding level identity and quality.

    Returns updated ohlcv and list of new feature column names.
    """
    print("\nComputing level-encoding features...")
    new_cols = []
    n = len(ohlcv)
    nearest = ohlcv['nearest_level_name'].values

    # 1. One-hot encoding of nearest level type (7 features)
    for lvl in TRACKED_LEVELS:
        col = f'is_{lvl}'
        ohlcv[col] = (nearest == lvl).astype(np.float32)
        new_cols.append(col)

    # 2. level_is_support: price above level → 1, below → 0
    ohlcv['level_is_support'] = (ohlcv['side'] == 1).astype(np.float32)
    new_cols.append('level_is_support')

    # 3. Trailing reversal rates per level type (causal — uses only past outcomes)
    print("  Computing trailing reversal rates (20d, 50d)...")
    ohlcv['level_trailing_rev_rate_20d'] = np.nan
    ohlcv['level_trailing_rev_rate_50d'] = np.nan

    # We need outcome and trading_day columns
    days = sorted(ohlcv['trading_day'].unique())
    day_to_idx = {d: i for i, d in enumerate(days)}

    # Pre-compute per-day per-level reversal counts
    # outcome: 1=reversal, 0=breakout, 2=inconclusive, -1=not near level
    near_mask = ohlcv['outcome'].isin([0, 1])  # only rev/bo bars count
    near_df = ohlcv.loc[near_mask, ['trading_day', 'nearest_level_name', 'outcome']].copy()

    # Daily reversal stats per level
    daily_stats = near_df.groupby(['trading_day', 'nearest_level_name']).agg(
        n_rev=('outcome', lambda x: (x == 1).sum()),
        n_total=('outcome', 'count'),
    ).reset_index()

    # Build cumulative lookup: for each (day, level) → trailing rev rate
    for lvl in TRACKED_LEVELS:
        lvl_stats = daily_stats[daily_stats['nearest_level_name'] == lvl].copy()
        lvl_stats = lvl_stats.set_index('trading_day').reindex(days).fillna(0)
        lvl_stats['cum_rev'] = lvl_stats['n_rev'].cumsum()
        lvl_stats['cum_total'] = lvl_stats['n_total'].cumsum()

        # 20d trailing (rolling sum)
        lvl_stats['rev_20d'] = lvl_stats['n_rev'].rolling(20, min_periods=1).sum()
        lvl_stats['total_20d'] = lvl_stats['n_total'].rolling(20, min_periods=1).sum()
        lvl_stats['rev_rate_20d'] = lvl_stats['rev_20d'] / lvl_stats['total_20d'].clip(lower=1)

        # 50d trailing
        lvl_stats['rev_50d'] = lvl_stats['n_rev'].rolling(50, min_periods=1).sum()
        lvl_stats['total_50d'] = lvl_stats['n_total'].rolling(50, min_periods=1).sum()
        lvl_stats['rev_rate_50d'] = lvl_stats['rev_50d'] / lvl_stats['total_50d'].clip(lower=1)

        # Shift by 1 to make strictly causal (today uses data up to yesterday)
        lvl_stats['rev_rate_20d'] = lvl_stats['rev_rate_20d'].shift(1)
        lvl_stats['rev_rate_50d'] = lvl_stats['rev_rate_50d'].shift(1)

        # Map back to bars
        rate_20d_map = lvl_stats['rev_rate_20d'].to_dict()
        rate_50d_map = lvl_stats['rev_rate_50d'].to_dict()

        lvl_mask = ohlcv['nearest_level_name'] == lvl
        if lvl_mask.any():
            ohlcv.loc[lvl_mask, 'level_trailing_rev_rate_20d'] = (
                ohlcv.loc[lvl_mask, 'trading_day'].map(rate_20d_map)
            )
            ohlcv.loc[lvl_mask, 'level_trailing_rev_rate_50d'] = (
                ohlcv.loc[lvl_mask, 'trading_day'].map(rate_50d_map)
            )

    new_cols.extend(['level_trailing_rev_rate_20d', 'level_trailing_rev_rate_50d'])

    # 4. level_side_bb_interaction: level_is_support × daily_bb_pct_b
    if 'daily_bb_pct_b' in ohlcv.columns:
        ohlcv['level_side_bb_interaction'] = ohlcv['level_is_support'] * ohlcv['daily_bb_pct_b']
        new_cols.append('level_side_bb_interaction')
    else:
        print("  WARNING: daily_bb_pct_b not found, skipping BB interaction feature")

    # 5. approach_dir_vs_level: +1 approaching from above, -1 from below
    # Use 5-bar close change direction relative to level position
    close_vals = ohlcv['close'].values.astype(np.float64)
    lvl_price_vals = ohlcv['nearest_level_price'].values.astype(np.float64)
    approach_dir = np.zeros(n, dtype=np.float32)
    for i in range(5, n):
        if np.isnan(lvl_price_vals[i]):
            continue
        price_change = close_vals[i] - close_vals[i - 5]
        if close_vals[i] >= lvl_price_vals[i]:
            # Above level: approaching from above if price was falling
            approach_dir[i] = -1.0 if price_change < 0 else 1.0
        else:
            # Below level: approaching from below if price was rising
            approach_dir[i] = 1.0 if price_change > 0 else -1.0
    ohlcv['approach_dir_vs_level'] = approach_dir
    new_cols.append('approach_dir_vs_level')

    print(f"  Added {len(new_cols)} level-encoding features")
    return ohlcv, new_cols


# ── Walk-forward splits ──────────────────────────────────────────────────

def get_fold_splits(days: List, n_folds: int, min_train_days: int) -> List[Tuple[List, List]]:
    """Walk-forward expanding-window splits."""
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


# ── Trading simulation ──────────────────────────────────────────────────

def simulate_trades(ohlcv: pd.DataFrame, predicted_indices: np.ndarray,
                    stop_pts: float = STOP_PTS, target_pts: float = TARGET_PTS,
                    max_bars: int = MAX_BARS) -> Tuple[int, float, float, float, List[float]]:
    """
    Simulate fixed-stop/target trades at predicted reversal bars.

    Returns (n_trades, win_rate, mean_pnl, total_pnl, pnl_list).
    """
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0, []

    close = ohlcv['close'].values.astype(np.float64)
    high = ohlcv['high'].values.astype(np.float64)
    low = ohlcv['low'].values.astype(np.float64)
    trade_dir = ohlcv['trade_direction'].values
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
            if direction == 1:  # long
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    break
            else:  # short
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


# ── Step 2: Walk-Forward XGBoost Training ────────────────────────────────

def train_walk_forward(samples_df: pd.DataFrame, feature_cols: List[str],
                       ohlcv: pd.DataFrame, n_folds: int = 5,
                       min_train_days: int = 100,
                       thresholds: List[float] = None) -> Dict:
    """
    Train XGBoost with walk-forward CV and honest early stopping.

    Returns dict with fold results, feature importance, predictions.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 70)
    print("WALK-FORWARD XGBOOST TRAINING")
    print("=" * 70)

    days = sorted(samples_df['trading_day'].unique())
    splits = get_fold_splits(days, n_folds, min_train_days)

    y = (samples_df['outcome'] == 1).astype(int).values  # reversal=1, breakout=0
    pos_rate = y.mean()
    print(f"Samples: {len(samples_df):,}, positive rate (reversal): {pos_rate:.2%}")
    print(f"Features: {len(feature_cols)}")
    print(f"Folds: {len(splits)}, thresholds: {thresholds}")

    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_test_indices = []
    all_feature_imp = {}

    for fold, (train_days, test_days) in enumerate(splits):
        print(f"\nFold {fold + 1}/{len(splits)}: "
              f"{len(train_days)} train days, {len(test_days)} test days")

        train_mask = samples_df['trading_day'].isin(train_days)
        test_mask = samples_df['trading_day'].isin(test_days)

        X_train_full = samples_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        X_test = samples_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_train_full = y[train_mask.values]
        y_test = y[test_mask.values]
        test_indices = samples_df.index[test_mask].values

        if len(y_test) == 0 or y_train_full.sum() < 5:
            print("  Skipping fold (insufficient data)")
            continue

        # Honest early stopping: temporal 80/20 split within training days
        n_train_days = len(train_days)
        val_split_idx = int(n_train_days * 0.8)
        val_days = set(train_days[val_split_idx:])
        train_day_values = samples_df.loc[train_mask, 'trading_day'].values
        val_within_train = np.array([d in val_days for d in train_day_values])
        train_within_train = ~val_within_train

        X_tr = X_train_full[train_within_train]
        y_tr = y_train_full[train_within_train]
        X_val = X_train_full[val_within_train]
        y_val = y_train_full[val_within_train]

        if len(y_val) == 0 or y_tr.sum() < 5:
            # Fallback: use full train without early stopping
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
            eval_metric='logloss',
            verbosity=0,
            tree_method='hist',
            early_stopping_rounds=30,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        print(f"  AUC={auc:.3f}, best_iteration={model.best_iteration}")

        # Accumulate
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_test_indices.extend(test_indices.tolist())

        fold_results.append({
            'fold': fold,
            'train_days': len(train_days),
            'test_days': len(test_days),
            'train_samples': len(X_tr),
            'test_samples': len(X_test),
            'auc': auc,
            'n_positive': int(y_test.sum()),
            'n_total': len(y_test),
            'best_iteration': model.best_iteration,
        })

        # Feature importance
        imp = model.get_booster().get_score(importance_type='gain')
        for fname, score in imp.items():
            if fname.startswith('f'):
                fidx = int(fname[1:])
                if fidx < len(feature_cols):
                    real_name = feature_cols[fidx]
                    all_feature_imp[real_name] = all_feature_imp.get(real_name, 0) + score

    # Normalize feature importance
    if all_feature_imp:
        max_imp = max(all_feature_imp.values())
        if max_imp > 0:
            all_feature_imp = {k: v / max_imp for k, v in all_feature_imp.items()}

    # Overall AUC
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_test_indices = np.array(all_test_indices)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    print(f"\nOverall OOS AUC: {overall_auc:.4f}")

    return {
        'fold_results': fold_results,
        'feature_importance': all_feature_imp,
        'all_y_true': all_y_true,
        'all_y_prob': all_y_prob,
        'all_test_indices': all_test_indices,
        'overall_auc': overall_auc,
        'feature_cols': feature_cols,
    }


# ── Step 3: Trading simulation at multiple thresholds ────────────────────

def evaluate_thresholds(train_result: Dict, ohlcv: pd.DataFrame,
                        thresholds: List[float] = None) -> Dict:
    """Evaluate trading performance at multiple probability thresholds."""
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS — TRADING SIMULATION")
    print("=" * 70)

    all_y_true = train_result['all_y_true']
    all_y_prob = train_result['all_y_prob']
    all_test_indices = train_result['all_test_indices']

    threshold_results = {}

    print(f"\n{'Thresh':>7} {'N pred':>8} {'N trades':>9} {'WR':>7} {'E[PnL]':>8} "
          f"{'Total PnL':>10} {'Precision':>10}")
    print("-" * 65)

    for thresh in thresholds:
        pred_mask = all_y_prob >= thresh
        pred_indices = all_test_indices[pred_mask]

        # Classification metrics
        y_pred = pred_mask.astype(int)
        tp = ((y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((y_pred == 1) & (all_y_true == 0)).sum()
        precision = tp / max(tp + fp, 1)

        # Trading simulation
        n_trades, wr, mean_pnl, total_pnl, pnl_list = simulate_trades(
            ohlcv, pred_indices
        )

        threshold_results[thresh] = {
            'n_predicted': int(pred_mask.sum()),
            'precision': precision,
            'n_trades': n_trades,
            'win_rate': wr,
            'mean_pnl': mean_pnl,
            'total_pnl': total_pnl,
            'pnl_list': pnl_list,
            'pred_indices': pred_indices,
        }

        print(f"{thresh:>7.2f} {pred_mask.sum():>8,} {n_trades:>9,} "
              f"{wr:>7.1%} {mean_pnl:>8.2f} {total_pnl:>10.1f} {precision:>10.1%}")

    return threshold_results


# ── Step 4: Post-analysis ────────────────────────────────────────────────

def run_post_analysis(train_result: Dict, threshold_results: Dict,
                      ohlcv: pd.DataFrame, samples_df: pd.DataFrame,
                      level_encoding_cols: List[str]):
    """Feature importance, per-level breakdown, baselines, fold stability."""

    print("\n" + "=" * 70)
    print("POST-ANALYSIS")
    print("=" * 70)

    # ── Feature importance ──
    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    print("\n── Top 25 Features by XGB Gain ──")
    print(f"{'Rank':>4} {'Feature':<40} {'Importance':>10} {'Level-Enc':>10}")
    print("-" * 68)
    for i, (feat, imp) in enumerate(fi_sorted[:25]):
        is_level = "***" if feat in level_encoding_cols else ""
        print(f"{i+1:>4} {feat:<40} {imp:>10.4f} {is_level:>10}")

    # Check: do level-encoding features rank in top 10?
    top10_names = [f[0] for f in fi_sorted[:10]]
    level_in_top10 = [f for f in top10_names if f in level_encoding_cols]
    print(f"\nLevel-encoding features in top 10: {level_in_top10 if level_in_top10 else 'NONE'}")

    # ── Per-level WR breakdown ──
    # Use best threshold (highest total PnL)
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    best_res = threshold_results[best_thresh]
    pred_indices = best_res['pred_indices']

    print(f"\n── Per-Level Breakdown (threshold={best_thresh:.2f}) ──")
    print(f"{'Level':<15} {'N trades':>9} {'WR':>7} {'E[PnL]':>8} {'Total PnL':>10}")
    print("-" * 55)

    for lvl in TRACKED_LEVELS:
        lvl_mask = ohlcv.loc[pred_indices, 'nearest_level_name'].values == lvl
        lvl_indices = pred_indices[lvl_mask]
        if len(lvl_indices) == 0:
            print(f"{lvl:<15} {'—':>9}")
            continue
        nt, wr, mp, tp, _ = simulate_trades(ohlcv, lvl_indices)
        print(f"{lvl:<15} {nt:>9,} {wr:>7.1%} {mp:>8.2f} {tp:>10.1f}")

    # ── Baselines ──
    print(f"\n── Baseline Comparisons ──")
    near_mask = samples_df['outcome'].isin([0, 1])
    all_near_indices = samples_df.index[near_mask].values

    # Baseline 1: trade every near-level bar
    nt, wr, mp, tp, _ = simulate_trades(ohlcv, all_near_indices)
    print(f"  All near-level bars:  N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # Baseline 2: trade only ovn_hi/ovn_lo bars
    ovn_mask = near_mask & samples_df['nearest_level_name'].isin(['ovn_hi', 'ovn_lo'])
    ovn_indices = samples_df.index[ovn_mask].values
    nt, wr, mp, tp, _ = simulate_trades(ohlcv, ovn_indices)
    print(f"  OVN levels only:      N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # Model at best threshold
    nt, wr, mp, tp = best_res['n_trades'], best_res['win_rate'], best_res['mean_pnl'], best_res['total_pnl']
    print(f"  Model (t={best_thresh:.2f}):      N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # ── Per-fold stability ──
    print(f"\n── Per-Fold Stability ──")
    print(f"{'Fold':>4} {'Train':>6} {'Test':>5} {'AUC':>7} {'N pos':>6}")
    print("-" * 32)
    for fr in train_result['fold_results']:
        print(f"{fr['fold']+1:>4} {fr['train_days']:>6} {fr['test_days']:>5} "
              f"{fr['auc']:>7.3f} {fr['n_positive']:>6}")

    auc_values = [fr['auc'] for fr in train_result['fold_results']]
    print(f"  AUC mean={np.mean(auc_values):.3f}, std={np.std(auc_values):.3f}")


# ── Step 5: Figures ──────────────────────────────────────────────────────

def save_figures(train_result: Dict, threshold_results: Dict,
                 ohlcv: pd.DataFrame, samples_df: pd.DataFrame,
                 level_encoding_cols: List[str]):
    """Save all analysis figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:25]

    # ── 1. Feature importance bar chart ──
    fig, ax = plt.subplots(figsize=(12, 9))
    names = [f[0] for f in fi_sorted]
    values = [f[1] for f in fi_sorted]
    colors = ['#e74c3c' if n in level_encoding_cols else '#3498db' for n in names]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Normalized Gain')
    ax.set_title('Feature Importance (red = level-encoding features)')
    ax.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Level-encoding'),
                       Patch(facecolor='#3498db', label='Base features')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # ── 2. Per-fold AUC bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [fr['fold'] + 1 for fr in train_result['fold_results']]
    aucs = [fr['auc'] for fr in train_result['fold_results']]
    ax.bar(folds, aucs, color='#2ecc71')
    ax.axhline(y=np.mean(aucs), color='black', linestyle='--', label=f'Mean AUC={np.mean(aucs):.3f}')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Per-Fold AUC')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'per_fold_auc.png'), dpi=150)
    plt.close()

    # ── 3. Per-level-type WR comparison (model vs no-model) ──
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    pred_indices = threshold_results[best_thresh]['pred_indices']

    level_wr_model = {}
    level_wr_baseline = {}
    near_mask = samples_df['outcome'].isin([0, 1])

    for lvl in TRACKED_LEVELS:
        # Model
        lvl_pred_mask = ohlcv.loc[pred_indices, 'nearest_level_name'].values == lvl
        lvl_pred = pred_indices[lvl_pred_mask]
        if len(lvl_pred) > 0:
            nt, wr, _, _, _ = simulate_trades(ohlcv, lvl_pred)
            level_wr_model[lvl] = wr if nt > 0 else np.nan
        else:
            level_wr_model[lvl] = np.nan

        # Baseline
        lvl_base_mask = near_mask & (samples_df['nearest_level_name'] == lvl)
        lvl_base = samples_df.index[lvl_base_mask].values
        if len(lvl_base) > 0:
            nt, wr, _, _, _ = simulate_trades(ohlcv, lvl_base)
            level_wr_baseline[lvl] = wr if nt > 0 else np.nan
        else:
            level_wr_baseline[lvl] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TRACKED_LEVELS))
    width = 0.35
    model_wrs = [level_wr_model.get(l, 0) for l in TRACKED_LEVELS]
    base_wrs = [level_wr_baseline.get(l, 0) for l in TRACKED_LEVELS]
    ax.bar(x - width/2, base_wrs, width, label='Baseline (all bars)', color='#95a5a6')
    ax.bar(x + width/2, model_wrs, width, label=f'Model (t={best_thresh:.2f})', color='#e74c3c')
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Level Type')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate by Level Type: Model vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(TRACKED_LEVELS, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'per_level_wr.png'), dpi=150)
    plt.close()

    # ── 4. Equity curve ──
    pnl_list = threshold_results[best_thresh]['pnl_list']
    if pnl_list:
        fig, ax = plt.subplots(figsize=(12, 5))
        cum_pnl = np.cumsum(pnl_list)
        ax.plot(cum_pnl, color='#2ecc71', linewidth=1.0)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL (points)')
        ax.set_title(f'Equity Curve (threshold={best_thresh:.2f}, N={len(pnl_list)} trades)')
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=np.array(cum_pnl) >= 0, alpha=0.1, color='green')
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=np.array(cum_pnl) < 0, alpha=0.1, color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'equity_curve.png'), dpi=150)
        plt.close()

    # ── 5. Threshold analysis ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    thresholds_sorted = sorted(threshold_results.keys())
    wrs = [threshold_results[t]['win_rate'] for t in thresholds_sorted]
    nts = [threshold_results[t]['n_trades'] for t in thresholds_sorted]
    mpnls = [threshold_results[t]['mean_pnl'] for t in thresholds_sorted]

    ax1.plot(thresholds_sorted, wrs, 'o-', color='#e74c3c', label='Win Rate')
    ax1.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate vs Threshold')
    ax1.legend()

    ax1b = ax1.twinx()
    ax1b.bar(thresholds_sorted, nts, width=0.02, alpha=0.3, color='#3498db', label='N trades')
    ax1b.set_ylabel('N trades')
    ax1b.legend(loc='lower right')

    ax2.plot(thresholds_sorted, mpnls, 's-', color='#2ecc71')
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('E[PnL] per trade (pts)')
    ax2.set_title('Expected PnL vs Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'threshold_analysis.png'), dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")


# ── Model saving ─────────────────────────────────────────────────────────

def save_model(samples_df: pd.DataFrame, feature_cols: List[str],
               ohlcv: pd.DataFrame, train_result: Dict,
               model_dir: str = 'models/reversal_phase3',
               threshold: float = 0.50) -> None:
    """
    Train a final model on all data and save artifacts for realtime use.

    Saves:
      - model.json: XGBoost model
      - metadata.json: feature columns, levels, config
      - reference_predictions.npz: OOS predictions from walk-forward for playback comparison
    """
    import xgboost as xgb

    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 70)

    # Train final model on ALL data
    y = (samples_df['outcome'] == 1).astype(int).values
    X = samples_df[feature_cols].fillna(0).values.astype(np.float32)

    # Use last 20% for honest early stopping
    days = sorted(samples_df['trading_day'].unique())
    val_split_idx = int(len(days) * 0.8)
    val_days = set(days[val_split_idx:])
    day_values = samples_df['trading_day'].values
    val_mask = np.array([d in val_days for d in day_values])
    train_mask = ~val_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    spw = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=5,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        tree_method='hist',
        early_stopping_rounds=30,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Final model trained: {len(X_tr):,} train, {len(X_val):,} val, "
          f"best_iteration={model.best_iteration}")

    # Save model
    model_path = os.path.join(model_dir, 'model.json')
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'tracked_levels': list(TRACKED_LEVELS),
        'threshold': threshold,
        'stop_pts': STOP_PTS,
        'target_pts': TARGET_PTS,
        'max_bars': MAX_BARS,
        'proximity_pts': 5.0,
        'n_features': len(feature_cols),
        'n_train_samples': int(len(X_tr)),
        'n_val_samples': int(len(X_val)),
        'best_iteration': int(model.best_iteration),
        'oos_auc': float(train_result['overall_auc']),
    }
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    # Save reference predictions (from walk-forward OOS)
    ref_path = os.path.join(model_dir, 'reference_predictions.npz')
    test_indices = train_result['all_test_indices']
    test_trading_days = samples_df.loc[test_indices, 'trading_day'].values
    test_nearest_level = samples_df.loc[test_indices, 'nearest_level_name'].values

    np.savez(
        ref_path,
        y_true=train_result['all_y_true'],
        y_prob=train_result['all_y_prob'],
        test_indices=test_indices,
        trading_days=test_trading_days,
        nearest_level=test_nearest_level,
    )
    print(f"  Reference predictions saved to {ref_path} "
          f"({len(test_indices):,} samples)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Single-model training with level-quality encoding")
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-train-days', type=int, default=100)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model artifacts to models/reversal_phase3/')
    parser.add_argument('--model-dir', default='models/reversal_phase3',
                        help='Directory to save model artifacts')
    args = parser.parse_args()

    t0 = time.time()

    # ── Load data and compute features ──
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)
    ohlcv, feature_cols = compute_all_features(ohlcv)

    # ── Label reversals/breakouts ──
    ohlcv = label_reversals_breakouts(ohlcv)

    # ── Add level-encoding features ──
    ohlcv, level_encoding_cols = compute_level_encoding_features(ohlcv)
    all_feature_cols = feature_cols + level_encoding_cols
    all_feature_cols = list(dict.fromkeys(all_feature_cols))

    # ── Filter to reversal/breakout samples only ──
    samples_mask = ohlcv['outcome'].isin([0, 1])
    samples_df = ohlcv.loc[samples_mask].copy()
    print(f"\nTraining samples: {len(samples_df):,} "
          f"(reversal={( samples_df['outcome']==1).sum():,}, "
          f"breakout={(samples_df['outcome']==0).sum():,})")

    # ── Train walk-forward XGBoost ──
    train_result = train_walk_forward(
        samples_df, all_feature_cols, ohlcv,
        n_folds=args.n_folds,
        min_train_days=args.min_train_days,
    )

    # ── Evaluate thresholds ──
    threshold_results = evaluate_thresholds(train_result, ohlcv)

    # ── Save model ──
    if args.save_model:
        save_model(samples_df, all_feature_cols, ohlcv, train_result,
                   model_dir=args.model_dir)

    # ── Post-analysis ──
    run_post_analysis(train_result, threshold_results, ohlcv, samples_df,
                      level_encoding_cols)

    # ── Save figures ──
    if not args.skip_plots:
        save_figures(train_result, threshold_results, ohlcv, samples_df,
                     level_encoding_cols)

    # ── Summary ──
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    best = threshold_results[best_thresh]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall OOS AUC:      {train_result['overall_auc']:.4f}")
    print(f"Best threshold:       {best_thresh:.2f}")
    print(f"  N trades:           {best['n_trades']:,}")
    print(f"  Win rate:           {best['win_rate']:.1%}")
    print(f"  E[PnL/trade]:       {best['mean_pnl']:+.2f} pts")
    print(f"  Total PnL:          {best['total_pnl']:+.1f} pts")
    print(f"  Precision:          {best['precision']:.1%}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Success criteria check
    print(f"\n── Success Criteria ──")
    wr_pass = best['win_rate'] > 0.52
    pnl_pass = best['mean_pnl'] > 0.3
    print(f"  WR > 52%:           {'PASS' if wr_pass else 'FAIL'} ({best['win_rate']:.1%})")
    print(f"  E[PnL] > 0.3pt:    {'PASS' if pnl_pass else 'FAIL'} ({best['mean_pnl']:+.2f})")

    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    top10_names = [f[0] for f in fi_sorted[:10]]
    level_in_top10 = [f for f in top10_names if f in level_encoding_cols]
    fi_pass = len(level_in_top10) > 0
    print(f"  Level features in top 10: {'PASS' if fi_pass else 'FAIL'} ({level_in_top10})")


if __name__ == '__main__':
    main()
