#!/usr/bin/env python3
"""
End-to-end reversal predictor experiment.

This script implements the head-to-head comparison between:
1. XGBoost with hand-crafted features
2. TCN with raw sequences
3. Hybrid combining both approaches
4. Anomaly detection (feature-only and hybrid)

Phase 2 addition: Anomaly detection approach to address 499:1 class imbalance.
Train on reversal samples ONLY, detect similar patterns.

Usage:
    source ~/ml-venv/bin/activate
    python sandbox/test_reversal_predictor.py --data raw_data/es_min_3y_clean_td_gamma.csv

    # With anomaly detection
    python sandbox/test_reversal_predictor.py --method anomaly --target-recall 0.7

    # Compare all methods
    python sandbox/test_reversal_predictor.py --method all

Outputs:
1. Label distribution report
2. Feature correlation matrix
3. Model comparison table
4. Precision-recall curves (for anomaly detection)
5. Trading simulation results
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and parse OHLCV data."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=dt_format_str)

    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df):,} bars across {df['trading_day'].nunique()} days")

    return df


def analyze_labels(labels_df: pd.DataFrame) -> dict:
    """Analyze label distribution."""
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 60)

    total = len(labels_df)
    n_bull = (labels_df['reversal_label'] == 1).sum()
    n_bear = (labels_df['reversal_label'] == -1).sum()
    n_none = (labels_df['reversal_label'] == 0).sum()

    print(f"\nTotal bars: {total:,}")
    print(f"Bull reversals: {n_bull:,} ({n_bull/total:.2%})")
    print(f"Bear reversals: {n_bear:,} ({n_bear/total:.2%})")
    print(f"No reversal: {n_none:,} ({n_none/total:.2%})")

    # Per-day stats
    daily = labels_df.groupby('trading_day').agg({
        'reversal_label': lambda x: (x != 0).sum()
    }).rename(columns={'reversal_label': 'reversals'})

    print(f"\nReversals per day:")
    print(f"  Mean: {daily['reversals'].mean():.1f}")
    print(f"  Std: {daily['reversals'].std():.1f}")
    print(f"  Min: {daily['reversals'].min()}")
    print(f"  Max: {daily['reversals'].max()}")

    # Magnitude distribution
    rev_df = labels_df[labels_df['reversal_label'] != 0]
    if len(rev_df) > 0:
        print(f"\nReversal magnitude (% move):")
        print(f"  Mean: {rev_df['reversal_magnitude'].mean():.3%}")
        print(f"  Median: {rev_df['reversal_magnitude'].median():.3%}")
        print(f"  Std: {rev_df['reversal_magnitude'].std():.3%}")

    return {
        'total_bars': total,
        'n_bull': n_bull,
        'n_bear': n_bear,
        'n_none': n_none,
        'avg_per_day': daily['reversals'].mean(),
        'avg_magnitude': rev_df['reversal_magnitude'].mean() if len(rev_df) > 0 else 0
    }


def get_feature_cols(features_df: pd.DataFrame) -> list:
    """Get list of feature columns for training."""
    # Higher timeframe features
    htf_features = [
        'daily_rsi_14', 'daily_close_vs_sma20', 'daily_close_vs_sma50',
        'daily_atr_14', 'daily_trend_5d', 'daily_trend_20d',
        'daily_bb_pct_b', 'daily_bb_upper_dist', 'daily_bb_lower_dist', 'daily_bb_width',
        'prior_day_range', 'prior_day_range_pct', 'prior_poc_rel',
        'gap_pct', 'gap_vs_prior_range', 'open_vs_prior_va', 'gap_filled',
        'weekly_rsi_14', 'weekly_close_vs_sma10', 'weekly_bb_pct_b', 'weekly_bb_width',
        'intraday_bb_pct_b', 'intraday_bb_upper_dist', 'intraday_bb_lower_dist', 'intraday_bb_width_z',
    ]

    # Volume features
    volume_features = [
        'vol_at_level', 'vol_at_level_pct', 'vol_concentration_z', 'vol_profile_entropy',
        'vol_into_level', 'vol_at_rejection_bar', 'vol_ratio_rejection',
        'poc_distance', 'va_contains_level', 'profile_skew_at_level', 'n_peaks_near_level',
    ]

    # Quality features
    quality_features = [
        'wick_to_body_ratio', 'rejection_penetration', 'close_position',
        'level_touch_count', 'bars_since_level_test',
    ]

    # Filter to columns that exist
    all_features = htf_features + volume_features + quality_features
    available = [c for c in all_features if c in features_df.columns]

    print(f"\nAvailable features: {len(available)}/{len(all_features)}")
    return available


def analyze_anomaly_results(trainer, features_df, labels_df):
    """
    Analyze anomaly detection results in detail.

    Shows precision-recall tradeoffs and optimal thresholds.
    """
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION ANALYSIS")
    print("=" * 60)

    # Check if anomaly path was trained
    anomaly_results = {k: v for k, v in trainer.path_results.items() if 'anomaly' in k}
    if not anomaly_results:
        print("No anomaly detection results to analyze.")
        return

    for path_name, result in anomaly_results.items():
        print(f"\n{path_name.upper()}:")
        print(f"  Recall: {result.reversal_recall:.1%}")
        print(f"  Precision: {result.reversal_precision:.1%}")
        print(f"  F1 Score: {result.overall_f1:.1%}")
        print(f"  Predictions: {result.n_filtered_samples:,} / {result.n_total_samples:,}")

        # Daily prediction rate
        n_days = features_df['trading_day'].nunique()
        predictions_per_day = result.n_filtered_samples / n_days
        print(f"  Predictions/day: {predictions_per_day:.1f}")

    # Compare to XGBoost baseline
    if 'xgboost' in trainer.path_results:
        xgb = trainer.path_results['xgboost']
        best_anomaly = max(anomaly_results.values(), key=lambda x: x.overall_f1)

        print("\n" + "-" * 40)
        print("COMPARISON: Anomaly vs XGBoost")
        print("-" * 40)
        print(f"{'Metric':<20} {'XGBoost':<15} {'Anomaly':<15} {'Diff':<15}")
        print("-" * 60)
        print(f"{'Recall':<20} {xgb.reversal_recall:>14.1%} {best_anomaly.reversal_recall:>14.1%} {best_anomaly.reversal_recall - xgb.reversal_recall:>+14.1%}")
        print(f"{'Precision':<20} {xgb.reversal_precision:>14.1%} {best_anomaly.reversal_precision:>14.1%} {best_anomaly.reversal_precision - xgb.reversal_precision:>+14.1%}")
        print(f"{'F1':<20} {xgb.overall_f1:>14.1%} {best_anomaly.overall_f1:>14.1%} {best_anomaly.overall_f1 - xgb.overall_f1:>+14.1%}")


def simulate_trading(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    ohlcv: pd.DataFrame,
    threshold: float = 0.6,
    stop_pct: float = 0.002,
    rr: float = 2.0
) -> dict:
    """
    Simulate trading based on predictions.

    Args:
        predictions_df: DataFrame with reversal_prob, bull_prob, bear_prob
        labels_df: DataFrame with actual reversal labels
        ohlcv: OHLCV data
        threshold: Probability threshold for taking trades
        stop_pct: Stop loss as % of entry price
        rr: Reward-to-risk ratio

    Returns:
        Trading metrics dictionary
    """
    print("\n" + "=" * 60)
    print(f"TRADING SIMULATION (threshold={threshold}, stop={stop_pct:.2%}, rr={rr})")
    print("=" * 60)

    # Identify trade entries
    bull_signals = predictions_df['bull_prob'] > threshold
    bear_signals = predictions_df['bear_prob'] > threshold

    n_bull_signals = bull_signals.sum()
    n_bear_signals = bear_signals.sum()

    print(f"\nSignals generated:")
    print(f"  Bull: {n_bull_signals:,}")
    print(f"  Bear: {n_bear_signals:,}")
    print(f"  Total: {n_bull_signals + n_bear_signals:,}")

    # Check actual outcomes
    actual_reversals = labels_df['reversal_label'].values
    magnitudes = labels_df['reversal_magnitude'].values

    # True positives for bull
    bull_tp = (bull_signals & (actual_reversals == 1)).sum()
    bull_fp = (bull_signals & (actual_reversals != 1)).sum()

    # True positives for bear
    bear_tp = (bear_signals & (actual_reversals == -1)).sum()
    bear_fp = (bear_signals & (actual_reversals != -1)).sum()

    total_signals = n_bull_signals + n_bear_signals
    total_tp = bull_tp + bear_tp
    total_fp = bull_fp + bear_fp

    win_rate = total_tp / max(total_signals, 1)
    precision = total_tp / max(total_signals, 1)

    # Estimate PnL
    # True positive: win RR * stop
    # False positive: lose 1 * stop
    avg_win = rr * stop_pct
    avg_loss = stop_pct

    expected_pnl_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
    total_pnl_pts = expected_pnl_per_trade * total_signals * 5000  # ES point value

    print(f"\nTrade Outcomes:")
    print(f"  True Positives: {total_tp:,} ({total_tp/max(total_signals,1):.1%})")
    print(f"  False Positives: {total_fp:,} ({total_fp/max(total_signals,1):.1%})")
    print(f"  Win Rate: {win_rate:.1%}")

    print(f"\nEstimated Performance:")
    print(f"  Expected PnL/trade: {expected_pnl_per_trade:.4f} ({expected_pnl_per_trade*100:.2f}%)")
    print(f"  Signals/day: {total_signals / labels_df['trading_day'].nunique():.1f}")

    return {
        'n_signals': total_signals,
        'win_rate': win_rate,
        'precision': precision,
        'expected_pnl_per_trade': expected_pnl_per_trade,
        'signals_per_day': total_signals / labels_df['trading_day'].nunique()
    }


def main():
    parser = argparse.ArgumentParser(description='Reversal Predictor Experiment')
    parser.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input data CSV path'
    )
    parser.add_argument(
        '--min-move',
        type=float,
        default=0.002,
        help='Minimum move % for reversal labeling (default: 0.2%%)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of walk-forward folds (default: 5)'
    )
    parser.add_argument(
        '--skip-tcn',
        action='store_true',
        help='Skip TCN path (faster but less complete comparison)'
    )
    parser.add_argument(
        '--skip-hybrid',
        action='store_true',
        help='Skip hybrid path'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Probability threshold for trading simulation (default: 0.6)'
    )
    parser.add_argument(
        '--method',
        choices=['xgboost', 'tcn', 'hybrid', 'anomaly', 'all'],
        default='xgboost',
        help='Training method (default: xgboost, use "all" to compare all methods)'
    )
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.7,
        help='Target recall for anomaly detection threshold tuning (default: 0.7)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Training epochs for anomaly/NN models (default: 50)'
    )
    parser.add_argument(
        '--use-sequences',
        action='store_true',
        help='Use hybrid autoencoder with sequences (default: feature-only)'
    )

    args = parser.parse_args()

    # Import modules
    from strategies.labeling.reversal_labels import ReversalLabeler
    from strategies.reversal.trainer import ReversalTrainer

    # Load data
    ohlcv = load_data(args.data)

    # Initialize trainer
    trainer = ReversalTrainer(
        min_move_pct=args.min_move,
        n_folds=args.n_folds,
        min_train_days=100
    )

    # Step 1: Compute labels
    print("\n" + "=" * 60)
    print("STEP 1: COMPUTING REVERSAL LABELS")
    print("=" * 60)
    labels_df = trainer.prepare_labels(ohlcv)
    label_stats = analyze_labels(labels_df)

    # Step 2: Compute features
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING FEATURES")
    print("=" * 60)
    features_df = trainer.prepare_features(
        ohlcv,
        include_htf=True,
        include_volume=True,
        include_quality=True
    )

    feature_cols = get_feature_cols(features_df)

    # Step 3: Analyze feature correlations
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE CORRELATION ANALYSIS")
    print("=" * 60)
    corr_df = trainer.analyze_feature_correlations(
        features_df, labels_df, feature_cols, top_n=20
    )

    # Step 4: Train models based on method
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MODELS")
    print("=" * 60)

    method = args.method

    # Train XGBoost (baseline)
    if method in ['xgboost', 'all']:
        xgb_result = trainer.train_xgboost_path(
            ohlcv, labels_df, features_df, feature_cols, verbose=True
        )

    # Train TCN path
    if method in ['tcn', 'all'] and not args.skip_tcn:
        try:
            tcn_result = trainer.train_tcn_path(
                ohlcv, labels_df, verbose=True
            )
        except Exception as e:
            print(f"\nTCN training failed: {e}")
            print("Skipping TCN path.")

    # Train Hybrid path
    if method in ['hybrid', 'all'] and not args.skip_hybrid:
        try:
            hybrid_result = trainer.train_hybrid_path(
                ohlcv, labels_df, features_df, feature_cols, verbose=True
            )
        except Exception as e:
            print(f"\nHybrid training failed: {e}")
            print("Skipping hybrid path.")

    # Train Anomaly Detection path (Phase 2)
    if method in ['anomaly', 'all']:
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION PATH")
        print("=" * 60)
        try:
            # Feature-only anomaly detection
            anomaly_result = trainer.train_anomaly_path(
                ohlcv, labels_df, features_df, feature_cols,
                use_sequences=False,
                target_recall=args.target_recall,
                epochs=args.epochs,
                verbose=True
            )

            # Hybrid anomaly detection (with sequences)
            if args.use_sequences or method == 'all':
                try:
                    anomaly_hybrid_result = trainer.train_anomaly_path(
                        ohlcv, labels_df, features_df, feature_cols,
                        use_sequences=True,
                        target_recall=args.target_recall,
                        epochs=args.epochs,
                        verbose=True
                    )
                except Exception as e:
                    print(f"\nHybrid anomaly training failed: {e}")
                    print("Sequences may require more data or different parameters.")

        except Exception as e:
            print(f"\nAnomaly detection training failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 7: Compare paths
    print("\n" + "=" * 60)
    print("STEP 5: MODEL COMPARISON")
    print("=" * 60)
    if len(trainer.path_results) > 0:
        comparison_df = trainer.compare_paths()

        # Analyze anomaly detection results if present
        if method in ['anomaly', 'all']:
            analyze_anomaly_results(trainer, features_df, labels_df)
    else:
        print("No models were trained successfully.")
        comparison_df = pd.DataFrame()

    # Step 8: Trading simulation with best model
    # For now, use XGBoost predictions
    print("\n" + "=" * 60)
    print("STEP 6: TRADING SIMULATION")
    print("=" * 60)

    # Train final XGBoost model on all data for simulation
    from strategies.reversal.predictor import XGBoostReversalPredictor

    final_predictor = XGBoostReversalPredictor(feature_cols)
    final_predictor.train(ohlcv, labels_df, features_df)
    predictions = final_predictor.predict(features_df)
    predictions_df = predictions.to_dataframe()

    trade_stats = simulate_trading(
        predictions_df, labels_df, ohlcv,
        threshold=args.threshold,
        stop_pct=0.002,
        rr=2.0
    )

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {len(ohlcv):,} bars, {ohlcv['trading_day'].nunique()} days")
    print(f"Reversals found: {label_stats['n_bull'] + label_stats['n_bear']:,} "
          f"({label_stats['avg_per_day']:.1f}/day avg)")

    if len(comparison_df) > 0:
        print(f"\nBest model: {comparison_df.iloc[0]['path']} "
              f"(F1: {comparison_df.iloc[0]['f1']:.2%})")

        # Highlight anomaly detection results
        if method in ['anomaly', 'all']:
            anomaly_rows = comparison_df[comparison_df['path'].str.contains('anomaly')]
            if len(anomaly_rows) > 0:
                best_anomaly = anomaly_rows.iloc[0]
                print(f"\nAnomaly Detection Results:")
                print(f"  Best: {best_anomaly['path']}")
                print(f"  Recall: {best_anomaly['recall']:.1%}")
                print(f"  Precision: {best_anomaly['precision']:.1%}")
                print(f"  F1: {best_anomaly['f1']:.1%}")

    print(f"\nTrading simulation:")
    print(f"  Signals/day: {trade_stats['signals_per_day']:.1f}")
    print(f"  Expected win rate: {trade_stats['win_rate']:.1%}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'sandbox/reversal_experiment_results'
    os.makedirs(output_dir, exist_ok=True)

    comparison_df.to_csv(f'{output_dir}/comparison_{timestamp}.csv', index=False)
    corr_df.to_csv(f'{output_dir}/correlations_{timestamp}.csv', index=False)

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
