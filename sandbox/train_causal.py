#!/usr/bin/env python
"""
V3 Causal Zone Prediction — experiment script.

Train XGBoost (scalar features) and V3 neural model (VP heatmaps + TCN + scalars)
to predict P(pre-reversal zone) at bars near tracked price levels.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py --xgboost-only
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py --v3-only
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py --levels vwap ovn_lo rth_lo
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='V3 Causal Zone Prediction')
    parser.add_argument('--data', type=str,
                        default='raw_data/es_min_3y_clean_td_gamma.csv',
                        help='Path to 1-min CSV')
    parser.add_argument('--xgboost-only', action='store_true',
                        help='Only train XGBoost (skip V3)')
    parser.add_argument('--v3-only', action='store_true',
                        help='Only train V3 (skip XGBoost)')
    parser.add_argument('--levels', nargs='*', default=None,
                        help='Levels to track (default: all)')
    parser.add_argument('--proximity', type=float, default=3.0,
                        help='Level proximity in points (default: 3.0)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of walk-forward folds')
    parser.add_argument('--epochs', type=int, default=50,
                        help='V3 training epochs per fold')
    parser.add_argument('--cache-dir', type=str,
                        default='strategies/data/zone_cache',
                        help='Cache directory for heatmaps')
    parser.add_argument('--fig-dir', type=str,
                        default='sandbox/figures/causal_zones',
                        help='Directory for output figures')
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    """Load 1-min OHLCV data."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    # Combine Date + Time into dt column, lowercase all columns
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()

    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute price level columns (vwap, ovn_lo, etc.) + prev_high/prev_low."""
    from strategies.features.price_levels import PriceLevelProvider

    print("\nComputing price levels...")
    plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
    feat_df = plp._compute_impl(ohlcv)

    # Attach level columns to ohlcv
    level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi']
    if 'dt' in feat_df.columns:
        feat_df = feat_df.set_index('dt')
    if 'dt' in ohlcv.columns:
        ohlcv_dt = ohlcv.set_index('dt')
    else:
        ohlcv_dt = ohlcv

    for col in level_cols:
        if col in feat_df.columns:
            ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

    # Compute prev_high/prev_low
    levels = plp.prev_day_levels(ohlcv)
    ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
    ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])

    # Report level coverage
    for col in level_cols + ['prev_high', 'prev_low']:
        if col in ohlcv.columns:
            valid = ohlcv[col].notna() & (ohlcv[col] != 0)
            print(f"  {col:15s}: {valid.mean():.1%} valid")

    return ohlcv


def run_zone_labeling(ohlcv: pd.DataFrame, proximity: float, levels: list = None):
    """Run level-anchored zone labeling."""
    from strategies.labeling.reversal_zones import LevelAnchoredZoneLabeler, ZoneConfig

    print("\nLabeling zones...")
    labeler = LevelAnchoredZoneLabeler(
        level_proximity_pts=proximity,
        zone_config=ZoneConfig(
            stop_loss_pct=0.0008,    # ~4pt at ES 5000
            min_reward_risk=1.5,
            max_lookback_bars=10,
            min_move_pct=0.0012,     # ~6pt at ES 5000
            slope_window=20,
            validation_bars=15,
        ),
        decay_alpha=0.5,
    )

    # Optionally filter to specific levels
    if levels:
        labeler.TRACKED_LEVELS = levels

    zone_labels = labeler.fit(ohlcv)
    labeler.print_summary()

    return zone_labels


def extract_features(ohlcv, zone_labels):
    """Extract scalar features and select sample bars."""
    from strategies.features.zone_features import ZoneFeatureExtractor

    print("\nExtracting scalar features...")
    extractor = ZoneFeatureExtractor()
    samples_df, feature_cols = extractor.extract_scalars(
        ohlcv, zone_labels,
        include_htf=True,
        include_volume=True,
        include_quality=True,
        include_temporal=True,
        include_price_level=True,
    )

    return samples_df, feature_cols, extractor


def extract_heatmaps(extractor, ohlcv, zone_labels, sample_indices, cache_dir):
    """Extract VP heatmaps for V3 model."""
    print("\nExtracting VP heatmaps...")
    heatmaps = extractor.extract_heatmaps(
        ohlcv, zone_labels, sample_indices,
        cache_dir=cache_dir,
    )

    print(f"  micro_vp: {heatmaps['micro_vp'].shape}")
    print(f"  meso_vp:  {heatmaps['meso_vp'].shape}")
    print(f"  macro_vp: {heatmaps['macro_vp'].shape}")
    print(f"  sequence: {heatmaps['sequence'].shape}")

    return heatmaps


def save_figures(trainer, fig_dir):
    """Save comparison figures."""
    os.makedirs(fig_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Per-fold results
        for name, result in trainer.path_results.items():
            folds = result.fold_results
            if not folds:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'{name.upper()} — Per-Fold Results')

            x = [f.fold + 1 for f in folds]

            axes[0].bar(x, [f.precision for f in folds], alpha=0.7, label='Precision')
            axes[0].bar(x, [f.recall for f in folds], alpha=0.5, label='Recall')
            axes[0].set_xlabel('Fold')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Zone Detection')
            axes[0].legend()

            axes[1].bar(x, [f.roc_auc for f in folds], color='green', alpha=0.7)
            axes[1].axhline(0.5, color='red', linestyle='--', label='Random')
            axes[1].set_xlabel('Fold')
            axes[1].set_ylabel('ROC-AUC')
            axes[1].set_title('Discrimination')
            axes[1].legend()

            axes[2].bar(x, [f.mean_pnl for f in folds], color='blue', alpha=0.7)
            axes[2].axhline(0, color='red', linestyle='--')
            axes[2].set_xlabel('Fold')
            axes[2].set_ylabel('E[PnL] per trade (pts)')
            axes[2].set_title('Trading Simulation')

            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f'{name}_folds.png'), dpi=100)
            plt.close()

        # Feature importance (XGBoost)
        xgb_result = trainer.path_results.get('xgboost')
        if xgb_result and xgb_result.feature_importance:
            sorted_imp = sorted(
                xgb_result.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:20]

            fig, ax = plt.subplots(figsize=(10, 8))
            names = [x[0] for x in sorted_imp]
            values = [x[1] for x in sorted_imp]
            ax.barh(range(len(names)), values, color='steelblue')
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Normalized Gain')
            ax.set_title('XGBoost Feature Importance (Top 20)')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'xgb_feature_importance.png'), dpi=100)
            plt.close()

        print(f"\nFigures saved to {fig_dir}/")

    except ImportError:
        print("matplotlib not available — skipping figures")


def main():
    args = parse_args()
    t0 = time.time()

    # 1. Load data
    ohlcv = load_data(args.data)

    # 2. Compute levels
    ohlcv = compute_levels(ohlcv)

    # 3. Zone labeling
    zone_labels = run_zone_labeling(ohlcv, args.proximity, args.levels)

    # 4. Extract features
    samples_df, feature_cols, extractor = extract_features(ohlcv, zone_labels)

    # Sanity check
    n_pos = (samples_df['zone_label'] != 0).sum()
    n_neg = (samples_df['zone_label'] == 0).sum()
    print(f"\nSample balance: {n_pos} positive, {n_neg} negative "
          f"({n_pos / (n_pos + n_neg):.1%} positive rate)")

    if n_pos < 50:
        print("ERROR: Too few positive samples. Check zone labeling parameters.")
        sys.exit(1)

    # 5. Train
    from strategies.reversal.causal_trainer import CausalZoneTrainer
    trainer = CausalZoneTrainer(n_folds=args.n_folds)

    if not args.v3_only:
        print("\n" + "#" * 70)
        print("# TRAINING XGBOOST")
        print("#" * 70)
        trainer.train_xgboost(
            samples_df, feature_cols, ohlcv, zone_labels, verbose=True,
        )

    if not args.xgboost_only:
        print("\n" + "#" * 70)
        print("# TRAINING V3 CAUSAL MODEL")
        print("#" * 70)

        sample_indices = samples_df.index.values
        heatmaps = extract_heatmaps(
            extractor, ohlcv, zone_labels, sample_indices, args.cache_dir,
        )

        trainer.train_v3(
            samples_df, feature_cols, heatmaps,
            ohlcv, zone_labels,
            epochs=args.epochs, verbose=True,
        )

    # 6. Compare
    if len(trainer.path_results) > 1:
        trainer.compare_paths()

    # Baseline: what if we entered every near-level bar?
    n_days = ohlcv['trading_day'].nunique()
    n_total_zone_bars = (zone_labels['zone_label'] != 0).sum()
    base_rate = n_total_zone_bars / len(ohlcv)
    print(f"\nBaseline positive rate (all bars): {base_rate:.4%}")
    print(f"Zone bars across {n_days} days: {n_total_zone_bars}")

    # 7. Figures
    save_figures(trainer, args.fig_dir)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")


if __name__ == '__main__':
    main()
