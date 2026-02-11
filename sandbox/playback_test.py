#!/usr/bin/env python
"""
End-to-end playback test for Phase 3 reversal predictor.

Replays historical CSV data through the ReversalPredictorStrategy,
then compares predictions to the reference OOS predictions from training.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/playback_test.py \
        --csv raw_data/es_min_3y_clean_td_gamma.csv \
        --model-dir models/reversal_phase3 \
        --test-days 10 \
        --warmup-days 60
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.realtime.protocol import RealtimeSignal


# ── Data helpers ──────────────────────────────────────────────────────────

def load_csv_data(path: str) -> pd.DataFrame:
    """Load CSV data matching train_level_models.load_data format."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


# ── PlaybackResult ────────────────────────────────────────────────────────

@dataclass
class PlaybackResult:
    """Collected signals and predictions from a playback run."""
    signals: List[RealtimeSignal] = field(default_factory=list)
    predictions: Dict[str, List] = field(default_factory=lambda: {
        'trading_day': [], 'bar_index': [], 'nearest_level': [],
        'y_prob': [], 'direction': [], 'entry_price': [],
    })
    test_days: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0


@dataclass
class ConsistencyReport:
    """Result of comparing playback vs reference predictions."""
    n_playback: int = 0
    n_reference: int = 0
    n_matched: int = 0
    correlation: float = 0.0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    passed: bool = False
    details: str = ""


# ── PlaybackRunner ────────────────────────────────────────────────────────

class PlaybackRunner:
    """Replay historical CSV data through ReversalPredictorStrategy end-to-end."""

    def __init__(
        self,
        csv_path: str,
        model_dir: str,
        test_start_idx: int = 0,
        test_days: int = 10,
        warmup_days: int = 60,
        pred_threshold: float = 0.0,  # capture ALL predictions for comparison
        proximity_pts: float = 5.0,
    ):
        self.csv_path = csv_path
        self.model_dir = model_dir
        self.test_start_idx = test_start_idx
        self.test_days = test_days
        self.warmup_days = warmup_days
        self.pred_threshold = pred_threshold
        self.proximity_pts = proximity_pts

    def run(self) -> PlaybackResult:
        """Step through test days, collect all signals.

        Processes each day's full set of bars at once (batch mode) for
        performance.  This matches the real-time engine behavior at end-of-day
        and validates that features and predictions are correct.
        """
        from strategies.reversal.realtime_strategy import ReversalPredictorStrategy

        t0 = time.time()
        result = PlaybackResult()

        # Load data
        ohlcv = load_csv_data(self.csv_path)
        all_days = sorted(ohlcv['trading_day'].unique())
        print(f"Total trading days: {len(all_days)}")

        # Split into warmup + test
        start_idx = self.warmup_days + self.test_start_idx
        if start_idx >= len(all_days):
            print(f"ERROR: warmup_days ({self.warmup_days}) + test_start_idx "
                  f"({self.test_start_idx}) >= total days ({len(all_days)})")
            return result

        end_idx = min(start_idx + self.test_days, len(all_days))
        warmup_days_list = all_days[:start_idx]
        test_days_list = all_days[start_idx:end_idx]

        print(f"Warmup: {len(warmup_days_list)} days "
              f"({warmup_days_list[0]} to {warmup_days_list[-1]})")
        print(f"Test:   {len(test_days_list)} days "
              f"({test_days_list[0]} to {test_days_list[-1]})")

        # Prepare historical context (warmup period)
        warmup_df = ohlcv[ohlcv['trading_day'].isin(warmup_days_list)].copy()

        # Create strategy
        strategy = ReversalPredictorStrategy(
            model_dir=self.model_dir,
            pred_threshold=self.pred_threshold,
            proximity_pts=self.proximity_pts,
        )
        strategy.set_historical_context(warmup_df)

        # Step through each test day — process full day at once
        for day_idx, test_day in enumerate(test_days_list):
            day_df = ohlcv[ohlcv['trading_day'] == test_day].copy()
            if day_df.empty:
                continue

            print(f"  Day {day_idx + 1}/{len(test_days_list)}: {test_day} "
                  f"({len(day_df)} bars)", end="", flush=True)

            strategy.reset_day()

            # Build datetime-indexed 1-min bars like the engine provides
            day_bars = self._prepare_day_bars(day_df)
            if day_bars.empty:
                print(" (skip: no bars)")
                continue

            # Process full day at once (batch mode)
            day_signals = strategy.process(day_bars)

            # Record results
            result.signals.extend(day_signals)
            result.test_days.append(test_day)

            for sig in day_signals:
                result.predictions['trading_day'].append(test_day)
                result.predictions['bar_index'].append(str(sig.trigger_ts))
                result.predictions['nearest_level'].append(sig.level_name)
                result.predictions['y_prob'].append(sig.pred_proba)
                result.predictions['direction'].append(sig.direction)
                result.predictions['entry_price'].append(sig.entry_price)

            print(f" -> {len(day_signals)} signals")

            # Update historical context for next day (expanding window)
            strategy._historical_ohlcv = pd.concat(
                [strategy._historical_ohlcv, day_df], ignore_index=True
            )

        result.elapsed_sec = time.time() - t0
        print(f"\nPlayback complete: {len(result.signals)} total signals "
              f"in {result.elapsed_sec:.1f}s")
        return result

    @staticmethod
    def _prepare_day_bars(day_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw CSV rows to datetime-indexed 1-min bars."""
        if 'dt' not in day_df.columns:
            return pd.DataFrame()

        bars = day_df.copy()
        bars = bars.set_index('dt')

        # Ensure tz-aware index (LA timezone)
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize('America/Los_Angeles')

        return bars


# ── ConsistencyChecker ────────────────────────────────────────────────────

class ConsistencyChecker:
    """Compare playback predictions to offline training reference."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._ref = None

    def load_reference(self) -> bool:
        """Load reference predictions from training."""
        ref_path = os.path.join(self.model_dir, 'reference_predictions.npz')
        if not os.path.exists(ref_path):
            print(f"Reference file not found: {ref_path}")
            return False

        self._ref = np.load(ref_path, allow_pickle=True)
        print(f"Reference loaded: {len(self._ref['y_prob'])} predictions")
        return True

    def compare(self, playback_result: PlaybackResult,
                test_days: Optional[List[str]] = None) -> ConsistencyReport:
        """Compare playback predictions to reference.

        Matches by (trading_day, nearest_level) since bar timestamps
        may differ between batch and incremental processing.
        """
        report = ConsistencyReport()

        if self._ref is None:
            report.details = "No reference loaded"
            return report

        ref_days = self._ref['trading_days']
        ref_levels = self._ref['nearest_level']
        ref_probs = self._ref['y_prob']

        report.n_reference = len(ref_probs)

        # Filter reference to test days if specified
        if test_days:
            day_mask = np.isin(ref_days, test_days)
            ref_days = ref_days[day_mask]
            ref_levels = ref_levels[day_mask]
            ref_probs = ref_probs[day_mask]
            print(f"Reference filtered to {len(test_days)} test days: "
                  f"{len(ref_probs)} predictions")

        # Build lookup from playback predictions
        pb_preds = playback_result.predictions
        report.n_playback = len(pb_preds['y_prob'])

        if report.n_playback == 0:
            report.details = "No playback predictions to compare"
            return report

        # Match by (trading_day, nearest_level) — aggregate by day/level
        # since there may be many bars per day per level
        ref_day_level_probs = {}
        for i in range(len(ref_days)):
            key = (str(ref_days[i]), str(ref_levels[i]))
            if key not in ref_day_level_probs:
                ref_day_level_probs[key] = []
            ref_day_level_probs[key].append(float(ref_probs[i]))

        pb_day_level_probs = {}
        for i in range(len(pb_preds['trading_day'])):
            key = (str(pb_preds['trading_day'][i]), str(pb_preds['nearest_level'][i]))
            if key not in pb_day_level_probs:
                pb_day_level_probs[key] = []
            pb_day_level_probs[key].append(float(pb_preds['y_prob'][i]))

        # Find matching keys
        common_keys = set(ref_day_level_probs.keys()) & set(pb_day_level_probs.keys())
        report.n_matched = len(common_keys)

        if report.n_matched == 0:
            report.details = (
                f"No matching (day, level) pairs found. "
                f"Ref keys sample: {list(ref_day_level_probs.keys())[:3]}, "
                f"Playback keys sample: {list(pb_day_level_probs.keys())[:3]}"
            )
            return report

        # Compare mean probabilities per (day, level)
        ref_means = []
        pb_means = []
        for key in sorted(common_keys):
            ref_means.append(np.mean(ref_day_level_probs[key]))
            pb_means.append(np.mean(pb_day_level_probs[key]))

        ref_arr = np.array(ref_means)
        pb_arr = np.array(pb_means)

        diffs = np.abs(ref_arr - pb_arr)
        report.max_abs_diff = float(diffs.max())
        report.mean_abs_diff = float(diffs.mean())

        if len(ref_arr) > 1:
            report.correlation = float(np.corrcoef(ref_arr, pb_arr)[0, 1])
        else:
            report.correlation = 1.0 if report.max_abs_diff < 0.01 else 0.0

        # Note: we use relaxed criteria because:
        # - Playback uses a DIFFERENT model (final retrained on all data)
        #   while reference uses walk-forward OOS fold models
        # - Feature computation may have minor numerical differences
        report.passed = (
            report.n_matched >= 3
            and report.correlation > 0.50
        )

        report.details = (
            f"Matched {report.n_matched}/{len(ref_day_level_probs)} ref (day,level) pairs. "
            f"Correlation={report.correlation:.4f}, "
            f"max|diff|={report.max_abs_diff:.4f}, "
            f"mean|diff|={report.mean_abs_diff:.4f}"
        )

        return report


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Playback test for Phase 3 reversal predictor"
    )
    parser.add_argument('--csv', default='raw_data/es_min_3y_clean_td_gamma.csv',
                        help='Path to 1-min CSV data')
    parser.add_argument('--model-dir', default='models/reversal_phase3',
                        help='Directory with model artifacts')
    parser.add_argument('--test-days', type=int, default=10,
                        help='Number of test days to replay')
    parser.add_argument('--warmup-days', type=int, default=60,
                        help='Number of warmup days for feature context')
    parser.add_argument('--test-start-idx', type=int, default=0,
                        help='Index into test days to start from (after warmup)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Prediction threshold (0 = capture all)')
    parser.add_argument('--skip-consistency', action='store_true',
                        help='Skip consistency check with reference predictions')
    args = parser.parse_args()

    print("=" * 70)
    print("PLAYBACK TEST — Phase 3 Reversal Predictor")
    print("=" * 70)

    # Verify model exists
    model_path = os.path.join(args.model_dir, 'model.json')
    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Run first: PYTHONPATH=/home/william/gybcap python -u "
              "sandbox/train_level_models.py --save-model")
        sys.exit(1)

    # Run playback
    runner = PlaybackRunner(
        csv_path=args.csv,
        model_dir=args.model_dir,
        test_start_idx=args.test_start_idx,
        test_days=args.test_days,
        warmup_days=args.warmup_days,
        pred_threshold=args.threshold,
    )

    result = runner.run()

    # Summary
    print("\n" + "=" * 70)
    print("PLAYBACK SUMMARY")
    print("=" * 70)
    print(f"Test days:      {len(result.test_days)}")
    print(f"Total signals:  {len(result.signals)}")
    print(f"Elapsed time:   {result.elapsed_sec:.1f}s")

    if result.signals:
        probs = [s.pred_proba for s in result.signals]
        print(f"Prob range:     [{min(probs):.3f}, {max(probs):.3f}]")
        print(f"Mean prob:      {np.mean(probs):.3f}")

        # Per-level breakdown
        level_counts = {}
        for s in result.signals:
            level_counts[s.level_name] = level_counts.get(s.level_name, 0) + 1
        print("\nSignals per level:")
        for lvl, cnt in sorted(level_counts.items(), key=lambda x: -x[1]):
            print(f"  {lvl:<15} {cnt:>6}")

    # Consistency check
    if not args.skip_consistency:
        print("\n" + "=" * 70)
        print("CONSISTENCY CHECK")
        print("=" * 70)

        checker = ConsistencyChecker(args.model_dir)
        if checker.load_reference():
            report = checker.compare(result, test_days=result.test_days)
            print(f"\nPlayback predictions: {report.n_playback}")
            print(f"Reference predictions: {report.n_reference}")
            print(f"Matched (day, level) pairs: {report.n_matched}")
            print(f"Correlation:    {report.correlation:.4f}")
            print(f"Max |diff|:     {report.max_abs_diff:.4f}")
            print(f"Mean |diff|:    {report.mean_abs_diff:.4f}")
            print(f"\n{report.details}")
            print(f"\nVerdict: {'PASS' if report.passed else 'FAIL'}")
        else:
            print("Skipping consistency check (no reference predictions)")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
