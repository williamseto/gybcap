#!/usr/bin/env python
"""
V3 neural model analysis with structural labels.

Part A: Sequence pattern comparison (do reversals and breakouts LOOK different
        in raw price/volume data?)
Part B: Train V3 CausalZoneModel with structural labels (honest evaluation)

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_v3_structural.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_v3_structural.py --sequence-only
"""

import argparse
import os
import sys
import time
import gc
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Data loading (reuse from train_causal.py) ────────────────────────────

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv):
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
    return ohlcv


# ── Part A: Sequence pattern comparison ──────────────────────────────────

def build_sequence(opens, highs, lows, closes, volumes, level_price):
    """Build 6-channel sequence (same as ZoneFeatureExtractor)."""
    T = len(closes)
    seq = np.zeros((T, 6), dtype=np.float32)

    if level_price > 0:
        seq[:, 0] = (closes - level_price) / level_price * 1000
    vol_mean, vol_std = volumes.mean(), volumes.std()
    if vol_std > 0:
        seq[:, 1] = (volumes - vol_mean) / vol_std
    signs = np.sign(closes - opens)
    tick_delta = signs * volumes
    td_std = tick_delta.std()
    if td_std > 0:
        seq[:, 2] = tick_delta / td_std
    if level_price > 0:
        seq[:, 3] = (highs - lows) / level_price * 1000
    bar_range = highs - lows
    valid = bar_range > 0
    seq[valid, 4] = (closes[valid] - lows[valid]) / bar_range[valid]
    seq[~valid, 4] = 0.5
    cum_delta = np.cumsum(tick_delta)
    cd_std = cum_delta.std()
    if cd_std > 0:
        seq[:, 5] = cum_delta / cd_std

    return seq


def analyze_sequences(ohlcv, labeled, fig_dir, n_sample=5000, seq_window=60):
    """
    Compare average 60-bar sequences for reversals vs breakouts.
    This directly shows whether raw price/volume dynamics differ.
    """
    print(f"\n{'='*70}")
    print("SEQUENCE PATTERN ANALYSIS: REVERSAL vs BREAKOUT")
    print(f"{'='*70}")

    close_arr = ohlcv['close'].values.astype(np.float64)
    high_arr = ohlcv['high'].values.astype(np.float64)
    low_arr = ohlcv['low'].values.astype(np.float64)
    open_arr = ohlcv['open'].values.astype(np.float64)
    vol_arr = ohlcv['volume'].values.astype(np.float64)

    near_mask = labeled['near_level'].values.astype(bool)
    outcome_arr = labeled['outcome'].values
    nearest_level_arr = labeled['nearest_level'].values

    channel_names = [
        'normalized_close', 'volume_z', 'tick_delta',
        'bar_range', 'close_position', 'cumulative_delta',
    ]

    rev_sequences = []
    brk_sequences = []

    # Collect sequences
    rng = np.random.RandomState(42)
    rev_indices = np.where(near_mask & (outcome_arr == 'reversal'))[0]
    brk_indices = np.where(near_mask & (outcome_arr == 'breakout'))[0]

    # Filter to indices with enough history
    rev_indices = rev_indices[rev_indices >= seq_window]
    brk_indices = brk_indices[brk_indices >= seq_window]

    if len(rev_indices) > n_sample:
        rev_indices = rng.choice(rev_indices, n_sample, replace=False)
    if len(brk_indices) > n_sample:
        brk_indices = rng.choice(brk_indices, n_sample, replace=False)

    print(f"Sampling {len(rev_indices)} reversal, {len(brk_indices)} breakout sequences")

    for idx in rev_indices:
        start = idx - seq_window
        lvl_name = nearest_level_arr[idx]
        lvl_price = float(ohlcv.loc[idx, lvl_name]) if lvl_name and lvl_name in ohlcv.columns else close_arr[idx]
        if np.isnan(lvl_price) or lvl_price == 0:
            lvl_price = close_arr[idx]
        seq = build_sequence(
            open_arr[start:idx], high_arr[start:idx],
            low_arr[start:idx], close_arr[start:idx],
            vol_arr[start:idx], lvl_price,
        )
        if len(seq) == seq_window:
            rev_sequences.append(seq)

    for idx in brk_indices:
        start = idx - seq_window
        lvl_name = nearest_level_arr[idx]
        lvl_price = float(ohlcv.loc[idx, lvl_name]) if lvl_name and lvl_name in ohlcv.columns else close_arr[idx]
        if np.isnan(lvl_price) or lvl_price == 0:
            lvl_price = close_arr[idx]
        seq = build_sequence(
            open_arr[start:idx], high_arr[start:idx],
            low_arr[start:idx], close_arr[start:idx],
            vol_arr[start:idx], lvl_price,
        )
        if len(seq) == seq_window:
            brk_sequences.append(seq)

    rev_seqs = np.array(rev_sequences)  # (N_rev, 60, 6)
    brk_seqs = np.array(brk_sequences)  # (N_brk, 60, 6)

    print(f"Collected: {len(rev_seqs)} reversal, {len(brk_seqs)} breakout sequences")

    # Compute per-timestep per-channel statistics
    rev_mean = rev_seqs.mean(axis=0)  # (60, 6)
    brk_mean = brk_seqs.mean(axis=0)
    rev_std = rev_seqs.std(axis=0)
    brk_std = brk_seqs.std(axis=0)

    # Per-channel Cohen's d at each timestep
    print(f"\nPer-channel Cohen's d (avg across last 10 bars, first 10 bars, all 60 bars):")
    print(f"{'Channel':25s} {'Last 10':>10s} {'First 10':>10s} {'All 60':>10s} {'Max |d|':>10s}")
    print("-" * 70)

    for ch in range(6):
        rv = rev_seqs[:, :, ch]  # (N, 60)
        bv = brk_seqs[:, :, ch]

        # Per-timestep Cohen's d
        d_per_t = np.zeros(seq_window)
        for t in range(seq_window):
            r_vals = rv[:, t]
            b_vals = bv[:, t]
            pooled = np.sqrt(
                ((len(r_vals)-1)*r_vals.std()**2 + (len(b_vals)-1)*b_vals.std()**2)
                / (len(r_vals)+len(b_vals)-2)
            )
            d_per_t[t] = (r_vals.mean() - b_vals.mean()) / max(pooled, 1e-10)

        avg_last10 = np.abs(d_per_t[-10:]).mean()
        avg_first10 = np.abs(d_per_t[:10]).mean()
        avg_all = np.abs(d_per_t).mean()
        max_d = np.max(np.abs(d_per_t))

        print(f"{channel_names[ch]:25s} {avg_last10:10.4f} {avg_first10:10.4f} "
              f"{avg_all:10.4f} {max_d:10.4f}")

    # Overall: flatten sequences and compare
    rev_flat = rev_seqs.reshape(len(rev_seqs), -1)  # (N, 360)
    brk_flat = brk_seqs.reshape(len(brk_seqs), -1)

    # Multivariate: average element-wise d
    d_elements = []
    for j in range(rev_flat.shape[1]):
        rv = rev_flat[:, j]
        bv = brk_flat[:, j]
        ps = np.sqrt(((len(rv)-1)*rv.std()**2 + (len(bv)-1)*bv.std()**2)/(len(rv)+len(bv)-2))
        d_elements.append(abs((rv.mean() - bv.mean()) / max(ps, 1e-10)))
    d_elements = np.array(d_elements)
    print(f"\nSequence elements with |d| >= 0.1: {(d_elements >= 0.1).sum()} / {len(d_elements)}")
    print(f"Sequence elements with |d| >= 0.2: {(d_elements >= 0.2).sum()} / {len(d_elements)}")
    print(f"Max element |d|: {d_elements.max():.4f}")
    print(f"Mean element |d|: {d_elements.mean():.4f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(fig_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        x = np.arange(seq_window)

        for ch in range(6):
            ax = axes[ch]
            rm = rev_mean[:, ch]
            bm = brk_mean[:, ch]
            rs = rev_std[:, ch] / np.sqrt(len(rev_seqs))  # SEM
            bs = brk_std[:, ch] / np.sqrt(len(brk_seqs))

            ax.plot(x, rm, 'b-', label='Reversal', alpha=0.9, linewidth=1.5)
            ax.fill_between(x, rm - 2*rs, rm + 2*rs, alpha=0.2, color='blue')
            ax.plot(x, bm, 'r-', label='Breakout', alpha=0.9, linewidth=1.5)
            ax.fill_between(x, bm - 2*bs, bm + 2*bs, alpha=0.2, color='red')

            ax.set_title(channel_names[ch], fontsize=11)
            ax.set_xlabel('Bar (0=oldest, 59=most recent)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Average 60-bar Sequences: Reversal (n={len(rev_seqs)}) vs '
            f'Breakout (n={len(brk_seqs)})',
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'sequence_comparison.png'), dpi=120)
        plt.close()
        print(f"\nFigure saved: {fig_dir}/sequence_comparison.png")

        # Also plot per-timestep Cohen's d for each channel
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for ch in range(6):
            ax = axes[ch]
            rv = rev_seqs[:, :, ch]
            bv = brk_seqs[:, :, ch]
            d_per_t = np.zeros(seq_window)
            for t in range(seq_window):
                r_vals = rv[:, t]
                b_vals = bv[:, t]
                ps = np.sqrt(
                    ((len(r_vals)-1)*r_vals.std()**2 +
                     (len(b_vals)-1)*b_vals.std()**2)
                    / (len(r_vals)+len(b_vals)-2)
                )
                d_per_t[t] = (r_vals.mean() - b_vals.mean()) / max(ps, 1e-10)

            ax.bar(x, d_per_t, color=np.where(d_per_t > 0, 'steelblue', 'salmon'),
                   alpha=0.7, width=1.0)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axhline(0.2, color='green', linewidth=0.5, linestyle='--', label='small d')
            ax.axhline(-0.2, color='green', linewidth=0.5, linestyle='--')
            ax.set_title(f"{channel_names[ch]} — Cohen's d per bar", fontsize=11)
            ax.set_xlabel('Bar')
            ax.set_ylabel("Cohen's d")
            ax.set_ylim(-0.4, 0.4)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Per-Timestep Effect Size: Reversal vs Breakout", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'sequence_cohens_d.png'), dpi=120)
        plt.close()
        print(f"Figure saved: {fig_dir}/sequence_cohens_d.png")

    except ImportError:
        print("matplotlib not available")

    return rev_seqs, brk_seqs


# ── Part B: V3 model with structural labels ──────────────────────────────

def train_v3_structural(
    ohlcv, labeled, fig_dir,
    n_sample_per_class=15000,
    n_folds=5, min_train_days=100,
    epochs=50, batch_size=64, lr=1e-3,
):
    """Train V3 CausalZoneModel with structural labels."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from strategies.reversal.causal_model import CausalZoneModel
    from strategies.features.zone_features import ZoneFeatureExtractor
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    print(f"\n{'='*70}")
    print("V3 NEURAL MODEL WITH STRUCTURAL LABELS")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Subsample: balanced reversal + breakout, skip inconclusive for training
    near_mask = labeled['near_level'].values.astype(bool)
    outcome = labeled['outcome'].values

    rev_idx = np.where(near_mask & (outcome == 'reversal'))[0]
    brk_idx = np.where(near_mask & (outcome == 'breakout'))[0]

    rng = np.random.RandomState(42)
    if len(rev_idx) > n_sample_per_class:
        rev_idx = rng.choice(rev_idx, n_sample_per_class, replace=False)
    if len(brk_idx) > n_sample_per_class:
        brk_idx = rng.choice(brk_idx, n_sample_per_class, replace=False)

    sample_indices = np.sort(np.concatenate([rev_idx, brk_idx]))
    print(f"Training samples: {len(rev_idx)} reversal + {len(brk_idx)} breakout "
          f"= {len(sample_indices)}")

    # Extract scalar features
    print("\nExtracting scalar features...")
    from strategies.features.higher_timeframe import HigherTimeframeProvider
    from strategies.features.volume_microstructure import VolumeMicrostructureProvider
    from strategies.features.reversion_quality import ReversionQualityProvider
    from strategies.features.temporal_interactions import TemporalInteractionProvider
    from strategies.features.price_levels import PriceLevelProvider

    feature_cols = []
    providers = [
        HigherTimeframeProvider(),
        VolumeMicrostructureProvider(include_bidask='bidvolume' in ohlcv.columns),
        ReversionQualityProvider(),
        TemporalInteractionProvider(),
        PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns),
    ]
    for prov in providers:
        df = prov._compute_impl(ohlcv)
        for col in prov.feature_names:
            if col in df.columns:
                ohlcv[col] = df[col].values
                feature_cols.append(col)

    # level_distance_norm for all bars
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
    print(f"  Scalar features: {len(feature_cols)}")

    # Extract heatmaps for subsample
    print("\nExtracting VP heatmaps...")
    extractor = ZoneFeatureExtractor()

    # Build a zone_labels-like DataFrame for heatmap extraction
    # (just needs nearest_level column for centering)
    zone_proxy = pd.DataFrame(index=ohlcv.index)
    zone_proxy['nearest_level'] = labeled['nearest_level'].values
    zone_proxy['zone_label'] = 0  # unused but needed by interface

    heatmaps = extractor.extract_heatmaps(
        ohlcv, zone_proxy, sample_indices,
        cache_dir=None,  # no cache, structural labels are different
    )

    print(f"  micro_vp: {heatmaps['micro_vp'].shape}")
    print(f"  meso_vp:  {heatmaps['meso_vp'].shape}")
    print(f"  macro_vp: {heatmaps['macro_vp'].shape}")
    print(f"  sequence: {heatmaps['sequence'].shape}")

    # Prepare labels
    # y_soft: P(reversal) from structural labeler
    p_rev = labeled['p_reversal'].values
    y_soft = p_rev[sample_indices].astype(np.float32)
    y_hard = (outcome[sample_indices] == 'reversal').astype(int)

    # Build samples DataFrame for walk-forward splits
    samples_df = ohlcv.loc[sample_indices].copy()
    samples_df['outcome'] = outcome[sample_indices]
    samples_df['trade_direction'] = labeled['trade_direction'].values[sample_indices]
    samples_df['p_reversal'] = p_rev[sample_indices]

    pos_rate = y_hard.mean()
    print(f"\nSample balance: {y_hard.sum()} reversal, {(1-y_hard).sum()} breakout "
          f"({pos_rate:.1%} positive)")

    # Walk-forward splits
    days = sorted(samples_df['trading_day'].unique())
    n_days = len(days)
    test_per_fold = (n_days - min_train_days) // n_folds
    splits = []
    for fold in range(n_folds):
        t_end = min_train_days + fold * test_per_fold
        te_end = t_end + test_per_fold
        if fold == n_folds - 1:
            te_end = n_days
        train_d = days[:t_end]
        test_d = days[t_end:te_end]
        if len(test_d) > 0:
            splits.append((train_d, test_d))

    all_y_true = []
    all_y_prob = []
    fold_results = []

    for fold, (train_days, test_days) in enumerate(splits):
        print(f"\nFold {fold+1}/{len(splits)}: "
              f"{len(train_days)} train, {len(test_days)} test days")

        train_mask = samples_df['trading_day'].isin(train_days).values
        test_mask = samples_df['trading_day'].isin(test_days).values

        if test_mask.sum() == 0 or y_hard[train_mask].sum() < 10:
            continue

        # Tensors
        X_sc_train = torch.tensor(
            samples_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        )
        X_sc_test = torch.tensor(
            samples_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
        )

        micro_train = torch.tensor(heatmaps['micro_vp'][train_mask])
        micro_test = torch.tensor(heatmaps['micro_vp'][test_mask])
        meso_train = torch.tensor(heatmaps['meso_vp'][train_mask])
        meso_test = torch.tensor(heatmaps['meso_vp'][test_mask])
        macro_train = torch.tensor(heatmaps['macro_vp'][train_mask])
        macro_test = torch.tensor(heatmaps['macro_vp'][test_mask])
        seq_train = torch.tensor(heatmaps['sequence'][train_mask])
        seq_test = torch.tensor(heatmaps['sequence'][test_mask])

        y_train_soft = torch.tensor(y_soft[train_mask])
        y_test_hard = y_hard[test_mask]
        test_idx = sample_indices[test_mask]

        # Build model
        model = CausalZoneModel(
            scalar_dim=len(feature_cols),
            seq_channels=heatmaps['sequence'].shape[-1],
        ).to(device)

        if fold == 0:
            print(f"  Model params: {model.count_parameters():,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        train_ds = TensorDataset(
            micro_train, meso_train, macro_train, seq_train, X_sc_train,
            y_train_soft,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        steps_per_epoch = len(train_loader)
        total_steps = max(epochs * steps_per_epoch, 1)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps,
        )

        # Train
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                b_micro, b_meso, b_macro, b_seq, b_sc, b_y = [
                    x.to(device) for x in batch
                ]
                pred = model(b_micro, b_meso, b_macro, b_seq, b_sc)
                loss = loss_fn(pred, b_y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_prob_list = []
            eval_bs = 256
            for i in range(0, len(X_sc_test), eval_bs):
                pred = model(
                    micro_test[i:i+eval_bs].to(device),
                    meso_test[i:i+eval_bs].to(device),
                    macro_test[i:i+eval_bs].to(device),
                    seq_test[i:i+eval_bs].to(device),
                    X_sc_test[i:i+eval_bs].to(device),
                )
                y_prob_list.append(pred.cpu().numpy())
            y_prob = np.concatenate(y_prob_list)

        all_y_true.extend(y_test_hard.tolist())
        all_y_prob.extend(y_prob.tolist())

        try:
            auc = roc_auc_score(y_test_hard, y_prob)
        except ValueError:
            auc = 0.5

        # Evaluate at thresholds
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds = (y_prob >= thresh).astype(int)
            n_pred = preds.sum()
            if n_pred == 0:
                continue
            prec = precision_score(y_test_hard, preds, zero_division=0)
            rec = recall_score(y_test_hard, preds, zero_division=0)
            f1 = f1_score(y_test_hard, preds, zero_division=0)

            # Honest trading sim
            pred_global = test_idx[preds == 1]
            n_trades, wr, mean_pnl, total_pnl = _sim_trades(
                ohlcv, pred_global, labeled,
            )

            fold_results.append({
                'fold': fold, 'threshold': thresh,
                'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
                'n_pred': int(n_pred), 'n_trades': n_trades,
                'win_rate': wr, 'mean_pnl': mean_pnl, 'total_pnl': total_pnl,
            })

            if thresh == 0.5:
                print(f"  T={thresh:.1f} | P={prec:.2%} R={rec:.2%} F1={f1:.2%} "
                      f"AUC={auc:.3f} | Trades={n_trades} WR={wr:.1%} "
                      f"E[PnL]={mean_pnl:+.2f}pt")

        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    print(f"\n{'='*70}")
    print("V3 STRUCTURAL SUMMARY")
    print(f"{'='*70}")
    print(f"Overall AUC: {overall_auc:.3f}")

    fdf = pd.DataFrame(fold_results)
    print(f"\nPer-threshold aggregate:")
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

    return fdf


def _sim_trades(ohlcv, predicted_indices, labeled,
                stop_pts=4.0, target_pts=6.0, max_bars=45):
    """Honest trade simulation."""
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0

    close = ohlcv['close'].values
    high = ohlcv['high'].values
    low = ohlcv['low'].values
    n = len(close)
    td = labeled['trade_direction'].values

    wins = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n:
            continue
        direction = int(td[idx])
        if direction == 0:
            continue
        entry = close[idx]
        trade_pnl = 0.0
        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:
                if low[j] <= entry - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry + target_pts:
                    trade_pnl = target_pts
                    break
            else:
                if high[j] >= entry + stop_pts:
                    trade_pnl = -stop_pts
                    break
                if low[j] <= entry - target_pts:
                    trade_pnl = target_pts
                    break
        pnl_list.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1

    n_trades = len(pnl_list)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0
    return n_trades, wins / n_trades, np.mean(pnl_list), np.sum(pnl_list)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='raw_data/es_min_3y_clean_td_gamma.csv')
    parser.add_argument('--fig-dir', default='sandbox/figures/feature_analysis')
    parser.add_argument('--sequence-only', action='store_true',
                        help='Only run sequence analysis, skip V3 training')
    parser.add_argument('--n-sample', type=int, default=15000,
                        help='Samples per class for V3 training')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    t0 = time.time()

    # Load + levels
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv)

    # Structural labeling
    print("\nLabeling with ReversalBreakoutLabeler...")
    from strategies.labeling.reversal_zones import ReversalBreakoutLabeler
    labeler = ReversalBreakoutLabeler(
        proximity_pts=5.0,
        forward_window=45,
        reversal_threshold_pts=6.0,
        breakout_threshold_pts=4.0,
        decay_alpha=0.3,
    )
    labeled = labeler.fit(ohlcv)
    labeler.print_summary()

    # Part A: Sequence analysis
    rev_seqs, brk_seqs = analyze_sequences(
        ohlcv, labeled, args.fig_dir, n_sample=5000,
    )

    # Part B: V3 training
    if not args.sequence_only:
        train_v3_structural(
            ohlcv, labeled, args.fig_dir,
            n_sample_per_class=args.n_sample,
            epochs=args.epochs,
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
