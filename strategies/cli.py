"""Command-line interface for trading strategy pipeline."""

import argparse
import os
import sys
import pandas as pd

from strategies.core.base import Trade
from strategies.features.price_levels import PriceLevelProvider
from strategies.training.trainer import Trainer
from strategies.training.evaluation import calculate_trading_metrics


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and parse data from CSV."""
    df = pd.read_csv(csv_path)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=dt_format_str)

    df.columns = df.columns.str.lower()
    return df


def resample_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-min data to higher timeframe."""
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Parse timeframe (e.g., "5min", "15min")
    freq = timeframe.replace('min', 'T').replace('m', 'T')

    bars = df.set_index('dt').groupby('trading_day').resample(freq).agg(agg_dict)
    bars = bars.dropna().reset_index()

    return bars


def train_strategy(args):
    """Train a strategy model."""
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)

    # Determine timeframe (both strategies default to 5min)
    if args.timeframe:
        timeframe = args.timeframe
    else:
        timeframe = '5min'

    print(f"Resampling to {timeframe}...")
    bars = resample_bars(df, timeframe)

    # Compute price level features
    print("Computing features...")
    plp = PriceLevelProvider(include_gamma='nearby_gamma_score' in df.columns)
    plp._compute_impl(df)
    bars = plp.attach_levels_to_bars(bars, df)

    # Set up level columns (different defaults per strategy based on edge analysis)
    if args.strategy == 'reversion':
        # Only levels with positive out-of-sample edge
        level_cols = ['vwap', 'rth_lo', 'ovn_lo']
    else:
        level_cols = [
            'prev_high', 'prev_low', 'vwap',
            'ovn_lo', 'ovn_hi',
            'rth_lo', 'rth_hi'
        ]

    # Create strategy
    if args.strategy == 'breakout':
        from strategies.breakout.strategy import BreakoutRetestStrategy
        strat = BreakoutRetestStrategy(
            bars,
            level_cols=level_cols,
            threshold_pct=0.0012,
            lookahead_bars=args.lookahead
        )
    elif args.strategy == 'reversion':
        from strategies.reversion.strategy import ReversionStrategy
        strat = ReversionStrategy(
            bars,
            level_cols=level_cols,
            threshold_pct=0.0012,
            lookahead_bars=args.lookahead
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    print(f"Running {args.strategy} strategy...")
    trades, trade_features_df = strat.find_retest_and_build_trades(
        stop_buffer_pct=args.stop_buffer,
        rr=args.rr,
        fixed_size=1.0
    )

    print(f"Generated {len(trades)} trades")

    # Calculate metrics
    total_pnl = sum(t.pnl for t in trades)
    win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) if trades else 0

    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Number of trading days: {df['trading_day'].nunique()}")

    # Train model
    feature_cols = plp.feature_names + ['bear']

    # Add Dalton features if requested
    if args.dalton:
        print("Adding Dalton features...")
        from strategies.features.dalton import DaltonFeatureProvider
        dalton = DaltonFeatureProvider(args.dalton_model)
        dalton_features = dalton._compute_impl(df)
        for col in dalton.feature_names:
            if col in dalton_features.columns:
                trade_features_df[col] = dalton_features.loc[
                    trade_features_df.index, col
                ].values
        feature_cols.extend(dalton.feature_names)

    # Disable augmentation for reversion by default (direction flip doesn't make sense)
    augment_data = not args.no_augment
    if args.strategy == 'reversion' and not args.no_augment:
        print("Note: Data augmentation disabled for reversion (use --no-augment=False to enable)")
        augment_data = False

    print(f"Training model with features: {feature_cols}")
    trainer = Trainer(
        feature_cols,
        augment_data=augment_data,
        n_jobs=args.n_jobs
    )

    if args.cv:
        print(f"Running cross-validation with {args.cv_folds} folds...")
        model = trainer.train_with_cv(
            trade_features_df,
            quick=args.cv_quick,
            n_folds=args.cv_folds,
            verbose=True
        )
    else:
        model = trainer.train(trade_features_df, verbose=True)

    # Evaluate on training data
    train_results = trainer.evaluate_trades(trades, trade_features_df, verbose=True)

    # Save model
    output_path = args.output or f"{args.strategy}_model.json"
    trainer.save_model(output_path)
    print(f"Model saved to {output_path}")

    return model, trades, trade_features_df


def backtest_strategy(args):
    """Backtest a strategy with saved model."""
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)

    # Determine timeframe (both strategies default to 5min)
    if args.timeframe:
        timeframe = args.timeframe
    else:
        timeframe = '5min'

    print(f"Resampling to {timeframe}...")
    bars = resample_bars(df, timeframe)

    # Compute features
    print("Computing features...")
    plp = PriceLevelProvider(include_gamma='nearby_gamma_score' in df.columns)
    plp._compute_impl(df)
    bars = plp.attach_levels_to_bars(bars, df)

    # Set up level columns
    level_cols = [
        'prev_high', 'prev_low', 'vwap',
        'ovn_lo', 'ovn_hi',
        'rth_lo', 'rth_hi'
    ]

    # Create strategy
    if args.strategy == 'breakout':
        from strategies.breakout.strategy import BreakoutRetestStrategy
        strat = BreakoutRetestStrategy(
            bars,
            level_cols=level_cols,
            threshold_pct=0.0012,
            lookahead_bars=args.lookahead
        )
    else:
        from strategies.reversion.strategy import ReversionStrategy
        strat = ReversionStrategy(
            bars,
            level_cols=level_cols,
            threshold_pct=0.0012,
            lookahead_bars=args.lookahead
        )

    print(f"Running {args.strategy} strategy...")
    trades, trade_features_df = strat.find_retest_and_build_trades(
        stop_buffer_pct=args.stop_buffer,
        rr=args.rr,
        fixed_size=1.0
    )

    print(f"Generated {len(trades)} trades")

    # Load model
    print(f"Loading model from {args.model}...")
    feature_cols = plp.feature_names + ['bear']
    trainer = Trainer(feature_cols)
    trainer.load_model(args.model)

    # Evaluate
    results = trainer.evaluate_trades(trades, trade_features_df, verbose=True)

    # Calculate detailed metrics
    preds = trainer.predict(trade_features_df)
    all_metrics = calculate_trading_metrics(trades)
    filtered_metrics = calculate_trading_metrics(trades, preds)

    print("\n--- All Trades ---")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- Filtered Trades ---")
    for k, v in filtered_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.plot:
        print("\nPlotting not yet implemented in CLI. Use mplfinance directly.")

    return trades, trade_features_df, preds


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Trading Strategy Pipeline',
        prog='strategies'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a strategy model')
    train_parser.add_argument(
        '--strategy', '-s',
        choices=['breakout', 'reversion', 'all'],
        default='breakout',
        help='Strategy to train (default: breakout)'
    )
    train_parser.add_argument(
        '--timeframe', '-t',
        help='Bar timeframe (e.g., 5min, 15min). Default: 5min'
    )
    train_parser.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input data CSV path'
    )
    train_parser.add_argument(
        '--output', '-o',
        help='Output model path (default: <strategy>_model.json)'
    )
    train_parser.add_argument(
        '--dalton',
        action='store_true',
        help='Include Dalton day type features'
    )
    train_parser.add_argument(
        '--dalton-model',
        default='vp/dalton_artifacts/dalton_classifier.joblib',
        help='Path to Dalton classifier model'
    )
    train_parser.add_argument(
        '--stop-buffer',
        type=float,
        default=0.0025,
        help='Stop buffer percentage (default: 0.0025)'
    )
    train_parser.add_argument(
        '--rr',
        type=float,
        default=2.0,
        help='Reward-to-risk ratio (default: 2.0)'
    )
    train_parser.add_argument(
        '--lookahead',
        type=int,
        default=10,
        help='Lookahead bars for retest (default: 10)'
    )
    train_parser.add_argument(
        '--cv',
        action='store_true',
        help='Use cross-validation for hyperparameter tuning'
    )
    train_parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    train_parser.add_argument(
        '--cv-quick',
        action='store_true',
        default=True,
        help='Use quick grid search instead of randomized search (default: True)'
    )
    train_parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation (recommended for reversion strategy)'
    )
    train_parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs for CV (default: 1, -1 uses all cores but can cause OOM)'
    )

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest a strategy')
    backtest_parser.add_argument(
        '--strategy', '-s',
        choices=['breakout', 'reversion'],
        required=True,
        help='Strategy to backtest'
    )
    backtest_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to trained model'
    )
    backtest_parser.add_argument(
        '--timeframe', '-t',
        help='Bar timeframe (e.g., 5min, 15min)'
    )
    backtest_parser.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input data CSV path'
    )
    backtest_parser.add_argument(
        '--stop-buffer',
        type=float,
        default=0.0025,
        help='Stop buffer percentage'
    )
    backtest_parser.add_argument(
        '--rr',
        type=float,
        default=2.0,
        help='Reward-to-risk ratio'
    )
    backtest_parser.add_argument(
        '--lookahead',
        type=int,
        default=10,
        help='Lookahead bars for retest'
    )
    backtest_parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot trades (not yet implemented)'
    )

    args = parser.parse_args()

    if args.command == 'train':
        if args.strategy == 'all':
            print("=== Training Breakout Strategy ===")
            args.strategy = 'breakout'
            train_strategy(args)

            print("\n=== Training Reversion Strategy ===")
            args.strategy = 'reversion'
            train_strategy(args)
        else:
            train_strategy(args)
    elif args.command == 'backtest':
        backtest_strategy(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
