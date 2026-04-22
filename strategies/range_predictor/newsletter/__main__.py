"""CLI entry point for newsletter comparison tools.

Usage:
    python -m strategies.range_predictor.newsletter analyze --newsletter PATH
    python -m strategies.range_predictor.newsletter reverse-engineer --newsletter PATH
    python -m strategies.range_predictor.newsletter formula --newsletter PATH
    python -m strategies.range_predictor.newsletter compare --newsletter PATH
    python -m strategies.range_predictor.newsletter compare-tf --newsletter PATH
    python -m strategies.range_predictor.newsletter fetch-emails [--since DATE]
"""

import argparse
import sys

import pandas as pd

from strategies.range_predictor.config import RangePredictorConfig
from strategies.range_predictor.features import aggregate_to_daily


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


def cmd_analyze(args):
    """Compare model predictions against newsletter."""
    from strategies.range_predictor.newsletter.comparison import run_analysis
    from strategies.range_predictor.predictor import RangePredictor

    config = RangePredictorConfig(model_dir=args.model_dir)
    predictor = RangePredictor(config)
    predictor.load_models()

    daily = _load_daily(args.data)
    newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    run_analysis(predictor, daily, newsletter, verbose=True)


def cmd_reverse_engineer(args):
    """Reverse-engineer what drives newsletter range predictions."""
    from strategies.range_predictor.newsletter.comparison import reverse_engineer_newsletter

    daily = _load_daily(args.data)
    newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    reverse_engineer_newsletter(daily, newsletter, verbose=True)


def cmd_compare_tf(args):
    """Compare model vs newsletter across all timeframes."""
    from strategies.range_predictor.newsletter.comparison import compare_all_timeframes
    from strategies.range_predictor.predictor import RangePredictor

    config = RangePredictorConfig(model_dir=args.model_dir)
    predictor = RangePredictor(config)
    predictor.load_models()

    daily = _load_daily(args.data)
    newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    compare_all_timeframes(
        predictor, daily, newsletter,
        oos_mode=not args.no_oos,
        verbose=True,
    )


def cmd_compare(args):
    """Comprehensive model vs newsletter vs realized comparison."""
    from strategies.range_predictor.newsletter.comparison import comprehensive_comparison
    from strategies.range_predictor.predictor import RangePredictor

    config = RangePredictorConfig(model_dir=args.model_dir)
    predictor = RangePredictor(config)
    predictor.load_models()

    daily = _load_daily(args.data)
    newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    comprehensive_comparison(
        predictor, daily, newsletter,
        output_path=args.output,
        verbose=True,
    )


def cmd_formula(args):
    """Fit and evaluate the reverse-engineered newsletter formula."""
    from strategies.range_predictor.newsletter.formula import run_formula_analysis

    daily = _load_daily(args.data)
    newsletter = pd.read_csv(args.newsletter, parse_dates=['date'])

    run_formula_analysis(
        daily, newsletter,
        n_folds=args.folds,
        min_train_days=args.min_train_days,
        verbose=True,
    )


def cmd_fetch_emails(args):
    """Fetch and parse newsletter emails."""
    from strategies.range_predictor.newsletter.email_pipeline.fetcher import fetch_emails
    from strategies.range_predictor.newsletter.email_pipeline.parser import parse_all_emails
    from strategies.range_predictor.newsletter.email_pipeline.cache import EmailCache

    cache = EmailCache(cache_dir=args.cache_dir)
    raw_emails = fetch_emails(since=args.since, cache=cache)
    print(f"Fetched {len(raw_emails)} emails")

    parsed = parse_all_emails(raw_emails, cache=cache)
    print(f"Parsed {len(parsed)} emails")

    # Export to CSV
    if args.output:
        rows = []
        for entry in parsed:
            for tf, predictions in entry.get('predictions', {}).items():
                for pred in predictions:
                    rows.append({
                        'date': entry['date'],
                        'timeframe': tf,
                        **pred,
                    })
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='Newsletter comparison tools for ES Range Predictor',
        prog='strategies.range_predictor.newsletter',
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # ── analyze ──
    analyze_p = subparsers.add_parser(
        'analyze', help='Compare model vs newsletter predictions',
    )
    analyze_p.add_argument(
        '--newsletter', required=True,
        help='Newsletter predictions CSV path',
    )
    analyze_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
    )
    analyze_p.add_argument('--model-dir', default='models/range_predictor')

    # ── reverse-engineer ──
    re_p = subparsers.add_parser(
        'reverse-engineer',
        help='Reverse-engineer what drives newsletter range predictions',
    )
    re_p.add_argument(
        '--newsletter', required=True,
        help='Newsletter predictions CSV path',
    )
    re_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )

    # ── compare-tf ──
    ctf_p = subparsers.add_parser(
        'compare-tf',
        help='Compare model vs newsletter for all timeframes (daily/weekly/monthly/quarterly)',
    )
    ctf_p.add_argument(
        '--newsletter', required=True,
        help='Newsletter predictions CSV path',
    )
    ctf_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )
    ctf_p.add_argument('--model-dir', default='models/range_predictor')
    ctf_p.add_argument(
        '--no-oos', action='store_true',
        help='Use final model (in-sample) instead of walk-forward OOS predictions',
    )

    # ── compare ──
    cmp_p = subparsers.add_parser(
        'compare',
        help='Comprehensive model vs newsletter vs realized comparison',
    )
    cmp_p.add_argument(
        '--newsletter', required=True,
        help='Newsletter predictions CSV path',
    )
    cmp_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )
    cmp_p.add_argument('--model-dir', default='models/range_predictor')
    cmp_p.add_argument(
        '--output', '-o',
        default='data/range_comparison_results.csv',
        help='Output CSV path for comparison results',
    )

    # ── formula ──
    fm_p = subparsers.add_parser(
        'formula',
        help='Fit and evaluate the linear newsletter replica formula '
             '(nl_width_t ~ AR(1) + prev_ret + |prev_ret|)',
    )
    fm_p.add_argument(
        '--newsletter', required=True,
        help='Newsletter predictions CSV path',
    )
    fm_p.add_argument(
        '--data', '-d',
        default='raw_data/es_min_3y_clean_td_gamma.csv',
        help='Input 1-min data CSV path',
    )
    fm_p.add_argument('--folds', type=int, default=5,
                      help='Walk-forward folds for OOS evaluation')
    fm_p.add_argument('--min-train-days', type=int, default=100)

    # ── fetch-emails ──
    fetch_p = subparsers.add_parser(
        'fetch-emails', help='Fetch and parse newsletter emails',
    )
    fetch_p.add_argument('--since', help='Fetch emails since date (YYYY-MM-DD)')
    fetch_p.add_argument('--cache-dir', default='data/email_cache')
    fetch_p.add_argument('--output', '-o', help='Output CSV path')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'reverse-engineer':
        cmd_reverse_engineer(args)
    elif args.command == 'formula':
        cmd_formula(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'compare-tf':
        cmd_compare_tf(args)
    elif args.command == 'fetch-emails':
        cmd_fetch_emails(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
