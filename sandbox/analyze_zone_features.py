#!/usr/bin/env python
"""
Feature activation analysis + honest evaluation with structural labels.

Part 1: Label bars with ReversalBreakoutLabeler (structural outcomes)
Part 2: Feature distribution analysis (reversal vs breakout features)
Part 3: Train XGBoost with structural labels (P(reversal), not trade viability)
Part 4: Honest trading sim on ALL near-level bars at real base rate

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_zone_features.py
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Data loading (from train_causal.py) ──────────────────────────────────

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

    level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi']
    if 'dt' in feat_df.columns:
        feat_df = feat_df.set_index('dt')
    ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv

    for col in level_cols:
        if col in feat_df.columns:
            ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

    levels = plp.prev_day_levels(ohlcv)
    ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
    ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])

    for col in level_cols + ['prev_high', 'prev_low']:
        if col in ohlcv.columns:
            valid = ohlcv[col].notna() & (ohlcv[col] != 0)
            print(f"  {col:15s}: {valid.mean():.1%} valid")

    return ohlcv


# ── Feature extraction ───────────────────────────────────────────────────

def extract_features(ohlcv: pd.DataFrame) -> tuple:
    """Compute all scalar features on the full dataset. Returns (ohlcv, feature_cols)."""
    feature_cols = []

    print("\nExtracting features (full dataset)...")

    from strategies.features.higher_timeframe import HigherTimeframeProvider
    htf = HigherTimeframeProvider()
    htf_df = htf._compute_impl(ohlcv)
    for col in htf.feature_names:
        if col in htf_df.columns:
            ohlcv[col] = htf_df[col].values
            feature_cols.append(col)
    print(f"  Higher timeframe: {len(htf.feature_names)} features")

    from strategies.features.volume_microstructure import VolumeMicrostructureProvider
    vmp = VolumeMicrostructureProvider(include_bidask='bidvolume' in ohlcv.columns)
    vol_df = vmp._compute_impl(ohlcv)
    for col in vmp.feature_names:
        if col in vol_df.columns:
            ohlcv[col] = vol_df[col].values
            feature_cols.append(col)
    print(f"  Volume microstructure: {len(vmp.feature_names)} features")

    from strategies.features.reversion_quality import ReversionQualityProvider
    rqp = ReversionQualityProvider()
    qual_df = rqp._compute_impl(ohlcv)
    for col in rqp.feature_names:
        if col in qual_df.columns:
            ohlcv[col] = qual_df[col].values
            feature_cols.append(col)
    print(f"  Reversion quality: {len(rqp.feature_names)} features")

    from strategies.features.temporal_interactions import TemporalInteractionProvider
    tip = TemporalInteractionProvider()
    temp_df = tip._compute_impl(ohlcv)
    for col in tip.feature_names:
        if col in temp_df.columns:
            ohlcv[col] = temp_df[col].values
            feature_cols.append(col)
    print(f"  Temporal interactions: {len(tip.feature_names)} features")

    from strategies.features.price_levels import PriceLevelProvider
    plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
    pl_df = plp._compute_impl(ohlcv)
    for col in plp.feature_names:
        if col in pl_df.columns:
            ohlcv[col] = pl_df[col].values
            feature_cols.append(col)
    print(f"  Price levels: {len(plp.feature_names)} features")

    # level_distance_norm computed for ALL bars (not from zone labels)
    from strategies.labeling.reversal_zones import ReversalBreakoutLabeler
    close_vals = ohlcv['close'].values
    min_dist = np.full(len(ohlcv), np.inf)
    for lvl_name in ReversalBreakoutLabeler.TRACKED_LEVELS:
        if lvl_name in ohlcv.columns:
            lvl_vals = ohlcv[lvl_name].values
            dist = np.abs(close_vals - lvl_vals)
            valid = ~np.isnan(dist)
            min_dist[valid] = np.minimum(min_dist[valid], dist[valid])
    min_dist[np.isinf(min_dist)] = np.nan
    ohlcv['level_distance_norm'] = min_dist / np.maximum(close_vals, 1) * 1000
    feature_cols.append('level_distance_norm')

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"  Total: {len(feature_cols)} features")
    return ohlcv, feature_cols


# ── Part 1: Feature distribution analysis ────────────────────────────────

def analyze_feature_distributions(
    near_level_df: pd.DataFrame,
    feature_cols: list,
    fig_dir: str,
):
    """Compare feature distributions: reversal vs breakout bars."""

    rev_mask = near_level_df['outcome'] == 'reversal'
    brk_mask = near_level_df['outcome'] == 'breakout'
    rev_df = near_level_df[rev_mask]
    brk_df = near_level_df[brk_mask]

    print(f"\n{'='*70}")
    print("FEATURE ACTIVATION ANALYSIS: REVERSAL vs BREAKOUT")
    print(f"{'='*70}")
    print(f"Reversal bars:  {len(rev_df):,}")
    print(f"Breakout bars:  {len(brk_df):,}")

    rows = []
    for col in feature_cols:
        rv = rev_df[col].dropna().values
        bv = brk_df[col].dropna().values
        if len(rv) < 20 or len(bv) < 20:
            continue

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(rv) - 1) * rv.std()**2 + (len(bv) - 1) * bv.std()**2)
            / (len(rv) + len(bv) - 2)
        )
        cohens_d = (rv.mean() - bv.mean()) / max(pooled_std, 1e-10)

        # KS test
        ks_stat, ks_p = ks_2samp(rv, bv)

        # t-test
        t_stat, t_p = ttest_ind(rv, bv, equal_var=False)

        rows.append({
            'feature': col,
            'rev_mean': rv.mean(),
            'rev_std': rv.std(),
            'brk_mean': bv.mean(),
            'brk_std': bv.std(),
            'cohens_d': cohens_d,
            'abs_d': abs(cohens_d),
            'ks_stat': ks_stat,
            'ks_pval': ks_p,
            't_pval': t_p,
        })

    df = pd.DataFrame(rows).sort_values('abs_d', ascending=False)

    # Print top features by effect size
    print(f"\nTop 25 features by |Cohen's d| (reversal vs breakout):")
    print(f"{'Feature':35s} {'Rev mean':>10s} {'Brk mean':>10s} {'Cohen d':>9s} {'KS stat':>9s} {'KS p':>10s}")
    print("-" * 85)
    for _, r in df.head(25).iterrows():
        sig = "***" if r['ks_pval'] < 0.001 else "**" if r['ks_pval'] < 0.01 else "*" if r['ks_pval'] < 0.05 else ""
        print(f"{r['feature']:35s} {r['rev_mean']:10.4f} {r['brk_mean']:10.4f} "
              f"{r['cohens_d']:+9.3f} {r['ks_stat']:9.3f} {r['ks_pval']:10.2e} {sig}")

    # Flag potentially circular features
    print(f"\n--- Circularity check ---")
    suspect = ['mae_estimate', 'level_distance_norm', 'bar_range_z']
    for s in suspect:
        match = df[df['feature'] == s]
        if len(match) > 0:
            r = match.iloc[0]
            print(f"  {s:35s}: d={r['cohens_d']:+.3f}, KS={r['ks_stat']:.3f} "
                  f"{'<-- possibly circular' if r['abs_d'] > 0.3 else '(OK)'}")

    # Summary: how many features have meaningful effect sizes?
    n_small = (df['abs_d'] >= 0.2).sum()
    n_medium = (df['abs_d'] >= 0.5).sum()
    n_large = (df['abs_d'] >= 0.8).sum()
    print(f"\nEffect size summary:")
    print(f"  |d| >= 0.2 (small):  {n_small} features")
    print(f"  |d| >= 0.5 (medium): {n_medium} features")
    print(f"  |d| >= 0.8 (large):  {n_large} features")

    # Save figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(fig_dir, exist_ok=True)

        # Top 12 features violin plots
        top_features = df.head(12)['feature'].tolist()
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()

        for i, feat in enumerate(top_features):
            ax = axes[i]
            rv = rev_df[feat].dropna().values
            bv = brk_df[feat].dropna().values

            # Clip outliers for visualization
            for arr in [rv, bv]:
                p1, p99 = np.percentile(arr, [1, 99])
                arr_range = p99 - p1
                if arr_range > 0:
                    np.clip(arr, p1 - 0.5 * arr_range, p99 + 0.5 * arr_range, out=arr)

            parts = ax.violinplot([rv, bv], positions=[0, 1], showmedians=True)
            for pc, color in zip(parts['bodies'], ['tab:blue', 'tab:red']):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Reversal', 'Breakout'])
            d_val = df[df['feature'] == feat]['cohens_d'].values[0]
            ax.set_title(f"{feat}\nd={d_val:+.3f}", fontsize=9)

        plt.suptitle('Feature Distributions: Reversal vs Breakout', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'feature_activations.png'), dpi=120)
        plt.close()
        print(f"\nFigure saved: {fig_dir}/feature_activations.png")
    except ImportError:
        print("matplotlib not available - skipping figures")

    return df


# ── Part 2: XGBoost with structural labels ───────────────────────────────

def train_xgboost_structural(
    near_level_df: pd.DataFrame,
    feature_cols: list,
    ohlcv: pd.DataFrame,
    n_folds: int = 5,
    min_train_days: int = 100,
):
    """
    Train XGBoost on structural labels (reversal=1, breakout/inconclusive=0).
    Walk-forward CV, evaluated on ALL near-level bars at real base rate.
    """
    import xgboost as xgb
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    print(f"\n{'='*70}")
    print("XGBOOST WITH STRUCTURAL LABELS (REVERSAL vs BREAKOUT)")
    print(f"{'='*70}")

    # Binary target: reversal=1, everything else=0
    y = (near_level_df['outcome'] == 'reversal').astype(int).values
    pos_rate = y.mean()
    print(f"Near-level bars: {len(near_level_df):,}")
    print(f"Real positive rate: {pos_rate:.2%} ({y.sum():,} reversals)")
    print(f"Features: {len(feature_cols)}")

    days = sorted(near_level_df['trading_day'].unique())
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds

    splits = []
    for fold in range(n_folds):
        train_end = min_train_days + fold * test_days_per_fold
        test_end = train_end + test_days_per_fold
        if fold == n_folds - 1:
            test_end = n_days
        train_d = days[:train_end]
        test_d = days[train_end:test_end]
        if len(test_d) > 0:
            splits.append((train_d, test_d))

    all_y_true = []
    all_y_prob = []
    all_test_indices = []
    fold_results = []

    for fold, (train_days, test_days) in enumerate(splits):
        print(f"\nFold {fold+1}/{len(splits)}: "
              f"{len(train_days)} train, {len(test_days)} test days")

        train_mask = near_level_df['trading_day'].isin(train_days)
        test_mask = near_level_df['trading_day'].isin(test_days)

        X_train = near_level_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        X_test = near_level_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_train = y[train_mask.values]
        y_test = y[test_mask.values]
        test_idx = near_level_df.index[test_mask].values

        if y_train.sum() < 10:
            continue

        spw = max(1.0, (1 - y_train.mean()) / max(y_train.mean(), 1e-6))

        # Use train/val split for early stopping (NOT test set)
        n_train = len(X_train)
        val_size = max(int(n_train * 0.15), 100)
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]

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

        y_prob = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_test_indices.extend(test_idx.tolist())

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        # Evaluate at several thresholds
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = (y_prob >= thresh).astype(int)
            n_pred = preds.sum()
            if n_pred == 0:
                continue
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            # Honest trading sim
            pred_idx = test_idx[preds == 1]
            n_trades, wr, mean_pnl, total_pnl, detail = simulate_trades_honest(
                ohlcv, pred_idx, near_level_df,
            )

            if thresh == 0.5:  # Print primary threshold
                print(f"  T={thresh:.1f} | P={prec:.2%} R={rec:.2%} F1={f1:.2%} "
                      f"AUC={auc:.3f} | Trades={n_trades} WR={wr:.1%} "
                      f"E[PnL]={mean_pnl:+.2f}pt")

            fold_results.append({
                'fold': fold, 'threshold': thresh,
                'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
                'n_pred': int(n_pred), 'n_trades': n_trades,
                'win_rate': wr, 'mean_pnl': mean_pnl, 'total_pnl': total_pnl,
            })

    # Overall results
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    print(f"\n{'='*70}")
    print("STRUCTURAL XGBOOST SUMMARY")
    print(f"{'='*70}")
    print(f"Overall AUC: {overall_auc:.3f}")
    print(f"Real positive rate: {all_y_true.mean():.2%}")

    # Show per-threshold aggregate results
    fdf = pd.DataFrame(fold_results)
    print(f"\nPer-threshold aggregate (all folds):")
    print(f"{'Thresh':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} "
          f"{'Trades':>7s} {'WR':>7s} {'E[PnL]':>8s} {'Total':>9s}")
    print("-" * 65)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        t_df = fdf[fdf['threshold'] == thresh]
        if len(t_df) == 0:
            continue
        total_trades = t_df['n_trades'].sum()
        total_pnl = t_df['total_pnl'].sum()
        avg_wr = (t_df['n_trades'] * t_df['win_rate']).sum() / max(total_trades, 1)
        avg_pnl = total_pnl / max(total_trades, 1)
        avg_prec = t_df['precision'].mean()
        avg_rec = t_df['recall'].mean()
        avg_f1 = t_df['f1'].mean()
        print(f"{thresh:7.1f} {avg_prec:7.2%} {avg_rec:7.2%} {avg_f1:7.2%} "
              f"{total_trades:7d} {avg_wr:7.1%} {avg_pnl:+8.2f} {total_pnl:+9.1f}")

    # Feature importance from last fold's model
    if 'model' in dir():
        imp = model.get_booster().get_score(importance_type='gain')
        sorted_imp = sorted(
            ((feature_cols[int(k[1:])] if k.startswith('f') and int(k[1:]) < len(feature_cols) else k, v)
             for k, v in imp.items()),
            key=lambda x: x[1], reverse=True,
        )
        max_v = sorted_imp[0][1] if sorted_imp else 1
        print(f"\nTop 15 features (gain, last fold):")
        for name, score in sorted_imp[:15]:
            print(f"  {name:35s}: {score/max_v:.3f}")

    return fdf, all_y_true, all_y_prob, all_test_indices


# ── Honest trading simulation ────────────────────────────────────────────

def simulate_trades_honest(
    ohlcv: pd.DataFrame,
    predicted_indices: np.ndarray,
    labeled_df: pd.DataFrame,
    stop_pts: float = 4.0,
    target_pts: float = 6.0,
    max_bars: int = 45,
) -> tuple:
    """
    Simulate trades at predicted bars using structural trade direction.
    Honest: includes all FP stops. Direction from side, not future label.
    """
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0, {}

    close = ohlcv['close'].values
    high = ohlcv['high'].values
    low = ohlcv['low'].values
    n = len(close)

    wins = 0
    stops = 0
    timeouts = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n or idx not in labeled_df.index:
            continue

        direction = int(labeled_df.loc[idx, 'trade_direction'])
        if direction == 0:
            continue

        entry = close[idx]
        trade_pnl = 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:  # long
                if low[j] <= entry - stop_pts:
                    trade_pnl = -stop_pts
                    stops += 1
                    break
                if high[j] >= entry + target_pts:
                    trade_pnl = target_pts
                    wins += 1
                    break
            else:  # short
                if high[j] >= entry + stop_pts:
                    trade_pnl = -stop_pts
                    stops += 1
                    break
                if low[j] <= entry - target_pts:
                    trade_pnl = target_pts
                    wins += 1
                    break
        else:
            timeouts += 1

        pnl_list.append(trade_pnl)

    n_trades = len(pnl_list)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0, {}

    wr = wins / n_trades
    mean_pnl = np.mean(pnl_list)
    total_pnl = np.sum(pnl_list)

    return n_trades, wr, mean_pnl, total_pnl, {
        'wins': wins, 'stops': stops, 'timeouts': timeouts,
    }


# ── Baseline: enter every near-level bar ─────────────────────────────────

def compute_baseline(ohlcv, near_level_df):
    """What happens if you enter EVERY near-level bar?"""
    print(f"\n{'='*70}")
    print("BASELINE: ENTER EVERY NEAR-LEVEL BAR")
    print(f"{'='*70}")

    all_idx = near_level_df.index.values
    n_trades, wr, mean_pnl, total_pnl, detail = simulate_trades_honest(
        ohlcv, all_idx, near_level_df,
    )
    print(f"Trades: {n_trades:,}, WR={wr:.1%}, E[PnL]={mean_pnl:+.2f}pt, "
          f"Total={total_pnl:+,.0f}pt")
    if detail:
        print(f"  Wins: {detail['wins']:,}, Stops: {detail['stops']:,}, "
              f"Timeouts: {detail['timeouts']:,}")

    # Breakdown by outcome
    for outcome in ['reversal', 'breakout', 'inconclusive']:
        mask = near_level_df['outcome'] == outcome
        idx = near_level_df.index[mask].values
        nt, w, mp, tp, d = simulate_trades_honest(ohlcv, idx, near_level_df)
        print(f"  {outcome:15s}: {nt:6d} trades, WR={w:.1%}, "
              f"E[PnL]={mp:+.2f}pt")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='raw_data/es_min_3y_clean_td_gamma.csv')
    parser.add_argument('--proximity', type=float, default=5.0)
    parser.add_argument('--rev-threshold', type=float, default=6.0)
    parser.add_argument('--brk-threshold', type=float, default=4.0)
    parser.add_argument('--fig-dir', default='sandbox/figures/feature_analysis')
    parser.add_argument('--skip-training', action='store_true')
    args = parser.parse_args()

    t0 = time.time()

    # 1. Load data + levels
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)

    # 2. Structural labeling
    print("\nLabeling with ReversalBreakoutLabeler...")
    from strategies.labeling.reversal_zones import ReversalBreakoutLabeler
    labeler = ReversalBreakoutLabeler(
        proximity_pts=args.proximity,
        forward_window=45,
        reversal_threshold_pts=args.rev_threshold,
        breakout_threshold_pts=args.brk_threshold,
        decay_alpha=0.3,
    )
    labeled = labeler.fit(ohlcv)
    labeler.print_summary()

    # 3. Extract features
    ohlcv, feature_cols = extract_features(ohlcv)

    # Get near-level bars with features
    near_mask = labeled['near_level'] == True
    near_level_df = ohlcv.loc[near_mask].copy()
    for col in ['outcome', 'p_reversal', 'nearest_level', 'level_distance',
                'side', 'bars_to_event', 'trade_direction', 'near_level']:
        near_level_df[col] = labeled.loc[near_mask, col].values

    print(f"\nNear-level bars with features: {len(near_level_df):,}")

    # 4. Feature distribution analysis
    feat_df = analyze_feature_distributions(
        near_level_df, feature_cols, args.fig_dir,
    )

    # 5. Baseline
    compute_baseline(ohlcv, near_level_df)

    # 6. XGBoost with structural labels
    if not args.skip_training:
        train_xgboost_structural(
            near_level_df, feature_cols, ohlcv,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
