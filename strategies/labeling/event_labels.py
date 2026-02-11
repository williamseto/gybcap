"""
Event-triggered labeling for reversal prediction.

Instead of labeling every bar (0.1-0.5% positive rate), this module labels
only bars where a level interaction was detected (30-50% positive rate).
The ML model's job changes from "find needles in haystacks" to
"score pre-selected candidates."
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from strategies.core.types import Direction, TriggerEvent
from strategies.reversion.detector import ReversionDetector


@dataclass
class EventLabelConfig:
    """Configuration for event-triggered labeling."""
    stop_loss_points: float = 4.0
    min_reward_multiplier: float = 1.5
    time_window_minutes: int = 45

    # Levels to detect events at
    level_cols: List[str] = field(default_factory=lambda: [
        'vwap', 'rth_lo', 'ovn_lo', 'rth_hi', 'ovn_hi',
        'prev_high', 'prev_low'
    ])

    # Whether to restrict to RTH only
    rth_only: bool = True

    @property
    def take_profit_points(self) -> float:
        return self.stop_loss_points * self.min_reward_multiplier


@dataclass
class LabeledEvent:
    """A trigger event with success/failure label and metadata."""
    level_name: str
    level_price: float
    trigger_ts: pd.Timestamp
    direction: str  # 'bull' or 'bear'
    bar_index: int
    trading_day: object  # date

    # Label
    success: int  # 1 = target hit first, 0 = stop hit or timeout
    exit_type: str  # 'target', 'stop', 'timeout'
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_points: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0


class EventTriggeredLabeler:
    """
    Labels level-interaction events as success/failure.

    Uses ReversionDetector to find level touches, then simulates
    each trade with fixed stop/target to determine outcome.

    Expected positive rate: 30-50% (vs 0.1-0.5% for all-bar prediction).
    """

    def __init__(self, config: Optional[EventLabelConfig] = None):
        self.config = config or EventLabelConfig()

    def detect_events(self, bars: pd.DataFrame) -> List[TriggerEvent]:
        """
        Detect level-interaction events using ReversionDetector.

        Args:
            bars: OHLCV DataFrame with level columns

        Returns:
            List of TriggerEvent objects
        """
        available_levels = [c for c in self.config.level_cols if c in bars.columns]
        if not available_levels:
            raise ValueError(
                f"No level columns found in data. Expected any of: {self.config.level_cols}"
            )

        detector = ReversionDetector(
            bars=bars,
            level_cols=available_levels,
            rth_only=self.config.rth_only,
        )
        return detector.detect()

    def label_events(
        self,
        bars: pd.DataFrame,
        events: Optional[List[TriggerEvent]] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Label each event as success (1) or failure (0).

        For bull events: success = price rises by take_profit before
        dropping by stop_loss (within time window).

        For bear events: success = price drops by take_profit before
        rising by stop_loss (within time window).

        Args:
            bars: OHLCV DataFrame (1-min bars)
            events: Pre-detected events. If None, detects them.
            verbose: Print summary statistics

        Returns:
            DataFrame with one row per event and columns:
                level_name, level_price, trigger_ts, direction,
                bar_index, trading_day, success, exit_type, pnl_points, etc.
        """
        if events is None:
            events = self.detect_events(bars)

        if not events:
            if verbose:
                print("No events detected.")
            return pd.DataFrame()

        cfg = self.config
        stop_pts = cfg.stop_loss_points
        target_pts = cfg.take_profit_points
        window_bars = cfg.time_window_minutes  # 1 bar = 1 min

        # Pre-extract arrays for fast simulation
        highs = bars['high'].values.astype(np.float64)
        lows = bars['low'].values.astype(np.float64)
        closes = bars['close'].values.astype(np.float64)
        bar_idx_array = bars.index

        # Trading day column
        if 'trading_day' in bars.columns:
            trading_days = bars['trading_day'].values
        else:
            trading_days = np.array([None] * len(bars))

        labeled = []

        for event in events:
            entry_idx = event.bar_index
            entry_price = closes[entry_idx]
            is_bull = event.direction == Direction.BULL

            if is_bull:
                target_price = entry_price + target_pts
                stop_price = entry_price - stop_pts
            else:
                target_price = entry_price - target_pts
                stop_price = entry_price + stop_pts

            # Simulate forward
            end_idx = min(entry_idx + window_bars + 1, len(bars))
            exit_type = 'timeout'
            exit_ts = None
            exit_price = entry_price
            max_favorable = 0.0
            max_adverse = 0.0
            success = 0

            for j in range(entry_idx + 1, end_idx):
                if is_bull:
                    favorable = highs[j] - entry_price
                    adverse = entry_price - lows[j]

                    max_favorable = max(max_favorable, favorable)
                    max_adverse = max(max_adverse, adverse)

                    # Check target first (optimistic)
                    if highs[j] >= target_price:
                        exit_type = 'target'
                        exit_ts = bar_idx_array[j]
                        exit_price = target_price
                        success = 1
                        break
                    # Check stop
                    if lows[j] <= stop_price:
                        exit_type = 'stop'
                        exit_ts = bar_idx_array[j]
                        exit_price = stop_price
                        success = 0
                        break
                else:
                    favorable = entry_price - lows[j]
                    adverse = highs[j] - entry_price

                    max_favorable = max(max_favorable, favorable)
                    max_adverse = max(max_adverse, adverse)

                    if lows[j] <= target_price:
                        exit_type = 'target'
                        exit_ts = bar_idx_array[j]
                        exit_price = target_price
                        success = 1
                        break
                    if highs[j] >= stop_price:
                        exit_type = 'stop'
                        exit_ts = bar_idx_array[j]
                        exit_price = stop_price
                        success = 0
                        break

            # Compute PnL
            if is_bull:
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price

            # Get trading day
            td = trading_days[entry_idx] if trading_days[entry_idx] is not None else None

            labeled.append(LabeledEvent(
                level_name=event.level_name,
                level_price=event.level_price,
                trigger_ts=event.trigger_ts,
                direction=str(event.direction),
                bar_index=event.bar_index,
                trading_day=td,
                success=success,
                exit_type=exit_type,
                exit_ts=exit_ts,
                exit_price=exit_price,
                pnl_points=pnl,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
            ))

        df = pd.DataFrame([vars(le) for le in labeled])

        if verbose:
            self._print_summary(df)

        return df

    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics for labeled events."""
        n = len(df)
        n_success = df['success'].sum()
        pos_rate = n_success / n if n > 0 else 0

        print(f"\n{'='*60}")
        print("EVENT-TRIGGERED LABELING SUMMARY")
        print(f"{'='*60}")
        print(f"Total events: {n:,}")
        print(f"Successes:    {n_success:,} ({pos_rate:.1%})")
        print(f"Failures:     {n - n_success:,} ({1-pos_rate:.1%})")
        print(f"\nConfig: stop={self.config.stop_loss_points}pts, "
              f"target={self.config.take_profit_points}pts, "
              f"window={self.config.time_window_minutes}min")

        # Breakdown by exit type
        print(f"\nExit types:")
        for etype, count in df['exit_type'].value_counts().items():
            print(f"  {etype}: {count:,} ({count/n:.1%})")

        # Breakdown by level
        print(f"\nBy level:")
        for level, grp in df.groupby('level_name'):
            lr = grp['success'].mean()
            print(f"  {level}: {len(grp):,} events, {lr:.1%} success rate")

        # Breakdown by direction
        print(f"\nBy direction:")
        for direction, grp in df.groupby('direction'):
            dr = grp['success'].mean()
            print(f"  {direction}: {len(grp):,} events, {dr:.1%} success rate")

        # Events per year
        if 'trading_day' in df.columns and df['trading_day'].notna().any():
            try:
                td_vals = df['trading_day'].dropna()
                # Handle integer day indices
                if td_vals.dtype in ('int64', 'float64', 'int32'):
                    n_unique_days = td_vals.nunique()
                    years = n_unique_days / 252  # ~252 trading days/year
                else:
                    days = pd.to_datetime(td_vals.astype(str))
                    years = (days.max() - days.min()).days / 365.25
                if years > 0:
                    print(f"\nEvents per year: {n/years:.0f}")
            except Exception:
                pass

        # PnL summary
        avg_pnl = df['pnl_points'].mean()
        total_pnl = df['pnl_points'].sum()
        print(f"\nAvg PnL: {avg_pnl:+.2f} pts")
        print(f"Total PnL: {total_pnl:+.1f} pts")
