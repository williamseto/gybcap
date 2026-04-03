#!/usr/bin/env python3
"""NQ Generalization Test for Range Predictor.

Fetches NQ futures daily bars from yfinance and trains the same range model
used for ES. Tests whether our feature engineering generalizes beyond ES.

Usage:
    source ~/ml-venv/bin/activate
    python strategies/range_predictor/train_nq.py
    python strategies/range_predictor/train_nq.py --start 2020-01-01
    python strategies/range_predictor/train_nq.py --model-dir models/range_predictor_nq
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def fetch_nq_daily(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """Fetch NQ futures daily bars from yfinance.

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Daily OHLCV DataFrame with lowercase column names, DatetimeIndex.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance not installed. Run: pip install yfinance"
        )

    print(f"Fetching NQ=F from yfinance ({start} to {end or 'today'})...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        raise RuntimeError("yfinance returned empty DataFrame for NQ=F")

    # Normalize columns to lowercase
    df.columns = df.columns.str.lower()

    # Keep only OHLCV
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    # Ensure DatetimeIndex is tz-naive for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    print(f"Fetched {len(df)} NQ daily bars  ({df.index[0].date()} — {df.index[-1].date()})")
    return df


def train_nq_model(
    nq_daily: pd.DataFrame,
    model_dir: str = 'models/range_predictor_nq',
    timeframe: str = 'daily',
    verbose: bool = True,
) -> dict:
    """Train range predictor on NQ daily data.

    Args:
        nq_daily: NQ daily OHLCV DataFrame (DatetimeIndex).
        model_dir: Directory to save NQ models.
        timeframe: Timeframe to train (default 'daily').
        verbose: Print training progress.

    Returns:
        Dict with CV metrics.
    """
    from strategies.range_predictor.config import RangePredictorConfig
    from strategies.range_predictor.trainer import RangeTrainer

    config = RangePredictorConfig(
        model_dir=model_dir,
        timeframes=[timeframe],
    )
    trainer = RangeTrainer(config)
    trainer.train_timeframe(nq_daily, timeframe=timeframe, verbose=verbose)
    trainer.save_models()

    # Extract CV results
    cv = trainer.results[timeframe]['cv_results']
    return {
        'timeframe': timeframe,
        'n_samples': trainer.results[timeframe]['n_samples'],
        'range_high_pct': cv['range_high_pct'],
        'range_low_pct': cv['range_low_pct'],
    }


def load_es_metrics(
    es_model_dir: str = 'models/range_predictor',
) -> dict:
    """Load existing ES model CV metrics from saved metadata.

    Returns:
        Dict with ES daily CV metrics, or empty dict if not found.
    """
    import json

    meta_path = os.path.join(es_model_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return {}

    with open(meta_path) as f:
        metadata = json.load(f)

    es_metrics = {}
    for key, val in metadata.items():
        if isinstance(val, dict) and 'cv_results' in val:
            es_metrics[key] = val['cv_results']

    return es_metrics


def compare_es_nq(
    nq_results: dict,
    es_model_dir: str = 'models/range_predictor',
    verbose: bool = True,
) -> None:
    """Print side-by-side ES vs NQ walk-forward metrics."""
    es_metrics = load_es_metrics(es_model_dir)

    if verbose:
        print(f"\n{'='*65}")
        print("ES vs NQ WALK-FORWARD COMPARISON  (daily timeframe)")
        print(f"{'='*65}")
        print(f"{'Metric':<30} {'ES Daily':>15} {'NQ Daily':>15}")
        print(f"{'─'*65}")

    for target in ['range_high_pct', 'range_low_pct']:
        nq_cv = nq_results.get(target, {})
        es_key = f"daily/{target}"
        es_cv = es_metrics.get(es_key, {})

        if verbose:
            label = target.replace('range_', '').replace('_pct', '')
            print(f"\n  Target: {target}")

            metrics_rows = [
                ('R²', 'overall_r2', '{:.4f}'),
                ('Correlation', 'correlation', '{:.4f}'),
                ('MAE', 'overall_mae', '{:.6f}'),
                ('RMSE', 'overall_rmse', '{:.6f}'),
                ('OOS samples', 'n_oos_samples', '{:d}'),
            ]
            for metric_label, key, fmt in metrics_rows:
                nq_val = nq_cv.get(key, float('nan'))
                es_val = es_cv.get(key, float('nan'))
                try:
                    nq_str = fmt.format(nq_val)
                    es_str = fmt.format(es_val)
                except (ValueError, TypeError):
                    nq_str = str(nq_val)
                    es_str = str(es_val)
                print(f"    {metric_label:<26} {es_str:>15} {nq_str:>15}")

    if verbose:
        print(f"\n  NQ samples: {nq_results.get('n_samples', 'N/A')}")
        print(f"{'='*65}")


def compare_nq_vs_realized(
    nq_daily: pd.DataFrame,
    nq_model_dir: str = 'models/range_predictor_nq',
    verbose: bool = True,
) -> dict:
    """Compare NQ model predictions vs realized ranges.

    Since the newsletter has no NQ daily predictions, we compare
    model vs realized only (no newsletter benchmark).

    Returns:
        Dict with containment metrics.
    """
    from strategies.range_predictor.predictor import RangePredictor
    from strategies.range_predictor.config import RangePredictorConfig
    from strategies.range_predictor.analysis import compute_containment_rate
    from strategies.range_predictor.features import _compute_atr

    config = RangePredictorConfig(model_dir=nq_model_dir)
    predictor = RangePredictor(config)
    predictor.load_models()

    preds = predictor.predict_series(nq_daily, timeframe='daily')

    # Drop NaN predictions (first few rows before features are ready)
    valid = preds.dropna(subset=['pred_range_high', 'pred_range_low'])
    realized = nq_daily.loc[valid.index]

    model_metrics = compute_containment_rate(
        valid['pred_range_low'],
        valid['pred_range_high'],
        realized['low'],
        realized['high'],
    )

    # ATR 0.7x baseline
    atr_14 = _compute_atr(nq_daily['high'], nq_daily['low'], nq_daily['close'], 14)
    prev_close = nq_daily['close'].shift(1)
    atr_low = (prev_close - 0.7 * atr_14).loc[valid.index]
    atr_high = (prev_close + 0.7 * atr_14).loc[valid.index]

    atr_metrics = compute_containment_rate(
        atr_low, atr_high,
        realized['low'], realized['high'],
    )

    if verbose:
        print(f"\n{'─'*50}")
        print(f"NQ MODEL vs REALIZED  (daily, {len(valid)} days)")
        print(f"{'─'*50}")
        print(f"{'Metric':<28} {'Model':>10} {'ATR-0.7x':>10}")
        rows = [
            ('Full containment', 'full_containment', '{:.1%}'),
            ('High contained', 'high_contained', '{:.1%}'),
            ('Low contained', 'low_contained', '{:.1%}'),
            ('Avg pred width', 'avg_pred_width', '{:.1f}'),
            ('Avg realized width', 'avg_realized_width', '{:.1f}'),
            ('Width ratio', 'width_ratio', '{:.3f}'),
        ]
        for label, key, fmt in rows:
            m_val = fmt.format(model_metrics[key])
            a_val = fmt.format(atr_metrics[key])
            print(f"  {label:<26} {m_val:>10} {a_val:>10}")

    return {'model': model_metrics, 'atr_0.7x': atr_metrics}


def main():
    parser = argparse.ArgumentParser(
        description='NQ Range Predictor — generalization test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--start', default='2020-01-01',
        help='Start date for NQ data (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--end', default=None,
        help='End date for NQ data (YYYY-MM-DD, default: today)',
    )
    parser.add_argument(
        '--model-dir', default='models/range_predictor_nq',
        help='Directory to save NQ models',
    )
    parser.add_argument(
        '--es-model-dir', default='models/range_predictor',
        help='Directory of existing ES models for comparison',
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Skip saving models to disk',
    )
    args = parser.parse_args()

    # ── Step 1: Fetch NQ data ──────────────────────────────────────────
    nq_daily = fetch_nq_daily(start=args.start, end=args.end)

    # ── Step 2: Train NQ model ─────────────────────────────────────────
    nq_results = train_nq_model(
        nq_daily,
        model_dir=args.model_dir,
        timeframe='daily',
        verbose=True,
    )

    # ── Step 3: ES vs NQ comparison ───────────────────────────────────
    compare_es_nq(nq_results, es_model_dir=args.es_model_dir, verbose=True)

    # ── Step 4: NQ model vs realized containment ──────────────────────
    compare_nq_vs_realized(nq_daily, nq_model_dir=args.model_dir, verbose=True)


if __name__ == '__main__':
    # Allow running from repo root without installing
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    main()
