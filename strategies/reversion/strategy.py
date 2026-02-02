"""Reversion trading strategy."""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

from strategies.core.base import BaseStrategy, Trade
from strategies.core.types import Direction, TriggerEvent, TradeSignal
from strategies.reversion.detector import ReversionDetector
from strategies.reversion.config import ReversionConfig


class ReversionStrategy(BaseStrategy):
    """
    Reversion (rejection) trading strategy.

    Detects rejection/reversion at key price levels, then enters
    in the direction of the rejection.

    For bull reversion (failed breakdown):
    - Price opens above level
    - Low touches or pierces level
    - Close back above level
    - Enter long on retest of level

    For bear reversion (failed breakout):
    - Price opens below level
    - High touches or pierces level
    - Close back below level
    - Enter short on retest of level
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        level_cols: Optional[List[str]] = None,
        threshold_pct: float = 0.0012,
        lookahead_bars: int = 10,
        config: Optional[ReversionConfig] = None
    ):
        """
        Initialize strategy.

        Args:
            bars: OHLCV DataFrame with price level columns
            level_cols: Columns containing price levels to trade
            threshold_pct: Distance threshold for retest entry (% of level)
            lookahead_bars: Number of bars to look for retest
            config: Optional ReversionConfig object
        """
        if config is not None:
            level_cols = config.level_cols
            threshold_pct = config.threshold_pct
            lookahead_bars = config.lookahead_bars

        if level_cols is None:
            level_cols = ['prev_high', 'prev_low', 'prev_mid']

        super().__init__(bars, level_cols, threshold_pct, lookahead_bars)
        self.config = config or ReversionConfig(
            level_cols=level_cols,
            threshold_pct=threshold_pct,
            lookahead_bars=lookahead_bars
        )

        # Create detector
        self._detector = ReversionDetector(self.bars, self.level_cols)

    def detect_triggers(self) -> List[TriggerEvent]:
        """Detect reversion trigger events."""
        return self._detector.detect()

    def get_trigger_events(self) -> List[tuple]:
        """
        Legacy interface for reversion detection.

        Returns:
            List of tuples: (level_name, level_price, timestamp, direction_str)
        """
        events = self.detect_triggers()
        return [e.to_tuple() for e in events]

    def find_entries(
        self,
        triggers: List[TriggerEvent],
        stop_buffer_pct: float,
        rr: float
    ) -> List[TradeSignal]:
        """
        Find confirmed entry signals from reversion triggers.

        Looks for price to retest the rejection level within lookahead window.
        """
        signals = []

        for trigger in triggers:
            idx0 = trigger.bar_index

            # Get lookahead window
            look_slice = self.bars.iloc[idx0 + 1: idx0 + 1 + self.lookahead_bars]
            if look_slice.empty:
                continue

            L = trigger.level_price

            # Calculate risk
            risk = min(stop_buffer_pct * L, self.config.max_risk_points)
            threshold = min(self.threshold_pct * L, risk * 0.5)

            # Find retest: bar where close is within threshold of level
            distances = (look_slice['close'] - L).abs()
            hits = distances <= threshold

            if not hits.any():
                continue

            # First bar that hits threshold
            hit_idx = hits.idxmax()
            entry_price = float(look_slice.loc[hit_idx, 'close'])

            # Get integer index for the entry bar
            entry_bar_idx = self.bars.index.get_loc(hit_idx)

            # Calculate stop and take
            if trigger.direction == Direction.BULL:
                stop_price = entry_price - risk
                take_price = entry_price + rr * risk
            else:
                stop_price = entry_price + risk
                take_price = entry_price - rr * risk

            signals.append(TradeSignal(
                trigger_event=trigger,
                entry_ts=hit_idx,
                entry_price=entry_price,
                entry_bar_index=entry_bar_idx,
                stop_price=stop_price,
                take_price=take_price,
                risk=risk
            ))

        return signals

    def find_retest_and_build_trades(
        self,
        stop_buffer_pct: float = 0.0025,
        rr: float = 1.5,
        fixed_size: float = 1.0
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Execute strategy and return trades with features.

        This provides the same interface as the legacy code.
        """
        return super().find_retest_and_build_trades(stop_buffer_pct, rr, fixed_size)
