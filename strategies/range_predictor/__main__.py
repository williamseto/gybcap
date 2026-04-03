"""CLI entry point for range predictor.

Usage:
    python -m strategies.range_predictor train [--data PATH] [--timeframes daily weekly]
    python -m strategies.range_predictor predict [--model-dir PATH]
    python -m strategies.range_predictor signal-study [--newsletter PATH]
    python -m strategies.range_predictor merge-data --historical PATH --existing PATH --output PATH

Newsletter commands have moved to:
    python -m strategies.range_predictor.newsletter --help
"""

import argparse
import sys

import pandas as pd

from strategies.range_predictor.config import RangePredictorConfig
from strategies.range_predictor.features import aggregate_to_daily, aggregate_to_rth_daily


def _load_daily(data_path: str) -> pd.DataFrame:
    """Load 1-min data and aggregate to daily."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=dt_format_str)

    df.columns = df.columns.str.lower()

    print(f"Loaded {len(df):,} 1-min bars across {df['trading_day'].nunique()} days")
    daily = aggregate_to_daily(df)
    print(f"Aggregated to {len(daily)} daily bars")
    return daily


def _load_1min(data_path: str) -> pd.DataFrame:
    """Load 1-min data with dt column parsed, columns lowercased."""
    print(f"Loading 1-min data from {data_path}...")
    df = pd.read_csv(data_path)

    dt_format_str = "%m/%d/%Y %H:%M:%S"
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=dt_format_str)

    df.columns = df.columns.str.lower()

    print(f"Loaded {len(df):,} 1-min bars across {df['trading_day'].nunique()} days")
    return df


def cmd_train(args):
    """Train range prediction models."""
    from strategies.range_predictor.trainer import RangeTrainer

    config = RangePredictorConfig(
        timeframes=args.timeframes,
        model_dir=args.model_dir,
        walk_forward_folds=args.folds,
        min_train_days=args.min_train_days,
    )

    if args.rth:
        # RTH model needs 1-min data for RTH aggregation
        df_1min = _load_1min(args.data)
        daily = aggregate_to_daily(df_1min)
        rth_daily = aggregate_to_rth_daily(df_1min)
        print(f"RTH daily: {len(rth_daily)} days")
    else:
        daily = _load_daily(args.data)

    trainer = RangeTrainer(config)

    if args.correlations:
        for tf in config.timeframes:
            trainer.analyze_feature_correlations(daily, timeframe=tf)

    trainer.train_all(daily, verbose=True)

    if args.rth:
        trainer.train_rth(daily, rth_daily, verbose=True)

    trainer.save_models()


def cmd_predict(args):
    """Predict today's ranges."""
    from strategies.range_predictor.predictor import RangePredictor

    config = RangePredictorConfig(model_dir=args.model_dir)
    predictor = RangePredictor(config)
    predictor.load_models()

    daily = _load_daily(args.data)
    predictions = predictor.predict(daily)

    print(f"\n{'='*60}")
    print("RANGE PREDICTIONS")
    print(f"{'='*60}")
    last_close = daily['close'].iloc[-1]
    print(f"Last close: {last_close:.2f}")
    print()

    for tf, vals in predictions.items():
        if tf == 'rth':
            continue  # Print RTH separately below
        low = vals.get('range_low', '?')
        high = vals.get('range_high', '?')
        width = vals.get('range_width', '?')
        print(f"  {tf:>12s}: {low} — {high}  (width: {width})")

    # RTH predictions
    if args.rth and args.rth_open is not None:
        rth_open = args.rth_open
        rth_daily = None
        if hasattr(args, 'data'):
            # Try to load RTH daily for gap features
            try:
                df_1min = _load_1min(args.data)
                rth_daily = aggregate_to_rth_daily(df_1min)
            except Exception:
                pass

        rth_preds = predictor.predict_rth(daily, rth_open, rth_daily)
        print(f"\n  RTH (open={rth_open:.2f}):")
        print(f"    High: {rth_preds.get('pred_rth_high', '?')}")
        print(f"    Low:  {rth_preds.get('pred_rth_low', '?')}")
        print(f"    Width: {rth_preds.get('pred_rth_width', '?')}")

    if args.output:
        predictor.save_predictions_csv(predictions, args.output)


def cmd_merge_data(args):
    """Merge historical ES data with existing dataset."""
    from strategies.range_predictor.merge_data import merge_historical_data

    merge_historical_data(args.historical, args.existing, args.output)


def cmd_signal_study(args):
    """Run intraday range signal backtest."""
    from strategies.range_predictor.signal_study import (
        SignalStudyConfig,
        backtest_range_signals,
    )

    df_1min = _load_1min(args.data)
    daily = aggregate_to_daily(df_1min)

    newsletter = None
    if args.newsletter:
        newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    config = SignalStudyConfig(
        stop_buffer_pts=args.stop_buffer,
        target_frac=args.target_frac,
        min_range_width_pts=args.min_width,
        cooldown_bars=args.cooldown,
        rth_only=not args.include_ovn,
        n_folds=args.folds,
        min_train_days=args.min_train_days,
        atr_mult=args.atr_mult,
    )

    backtest_range_signals(df_1min, daily, newsletter, config)


def main():
    parser = argparse.ArgumentParser(
        description='ES Range Predictor',
        prog='strategies.range_predictor',
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # ── train ──
    train_p = subparsers.add_parser('train', help='Train range prediction models')
    train_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )
    train_p.add_argument(
        '--model-dir',
        default='models/range_predictor',
        help='Output model directory',
    )
    train_p.add_argument(
        '--timeframes', '-t',
        nargs='+',
        default=['daily', 'weekly', 'monthly', 'quarterly'],
        choices=['daily', 'weekly', 'monthly', 'quarterly'],
        help='Timeframes to train',
    )
    train_p.add_argument('--folds', type=int, default=5, help='Walk-forward folds')
    train_p.add_argument('--min-train-days', type=int, default=100)
    train_p.add_argument(
        '--correlations', action='store_true',
        help='Run feature correlation analysis before training',
    )
    train_p.add_argument(
        '--rth', action='store_true',
        help='Also train RTH range model (requires 1-min data with ovn column)',
    )

    # ── predict ──
    predict_p = subparsers.add_parser('predict', help="Predict today's ranges")
    predict_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
    )
    predict_p.add_argument('--model-dir', default='models/range_predictor')
    predict_p.add_argument('--output', '-o', help='Output CSV path')
    predict_p.add_argument(
        '--rth', action='store_true',
        help='Also predict RTH range (requires --rth-open)',
    )
    predict_p.add_argument(
        '--rth-open', type=float, default=None,
        help='RTH open price for RTH range prediction',
    )

    # ── merge-data ──
    merge_p = subparsers.add_parser(
        'merge-data',
        help='Merge historical ES data with existing dataset for extended training',
    )
    merge_p.add_argument(
        '--historical', required=True,
        help='Historical data file (e.g. es_historical_public222.txt)',
    )
    merge_p.add_argument(
        '--existing',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Existing 1-min data CSV',
    )
    merge_p.add_argument(
        '--output', '-o',
        default='raw_data/es_min_extended.csv',
        help='Output merged CSV path',
    )

    # ── signal-study ──
    ss_p = subparsers.add_parser(
        'signal-study',
        help='Intraday range signal backtest (fade at predicted boundaries)',
    )
    ss_p.add_argument(
        '--newsletter',
        help='Newsletter predictions CSV path (optional)',
    )
    ss_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )
    ss_p.add_argument('--stop-buffer', type=float, default=5.0,
                       help='Stop buffer beyond boundary (pts)')
    ss_p.add_argument('--target-frac', type=float, default=0.5,
                       help='Target as fraction of range width (0.5 = midpoint)')
    ss_p.add_argument('--min-width', type=float, default=8.0,
                       help='Skip days with predicted range < this width (pts)')
    ss_p.add_argument('--cooldown', type=int, default=15,
                       help='Bars to wait after trade exit before re-entry')
    ss_p.add_argument('--include-ovn', action='store_true',
                       help='Allow entries during overnight session')
    ss_p.add_argument('--folds', type=int, default=5,
                       help='Walk-forward folds for OOS predictions')
    ss_p.add_argument('--min-train-days', type=int, default=100)
    ss_p.add_argument('--atr-mult', type=float, default=0.7,
                       help='ATR multiplier for baseline')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'merge-data':
        cmd_merge_data(args)
    elif args.command == 'signal-study':
        cmd_signal_study(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
