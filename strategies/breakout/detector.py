"""Breakout event detection with vectorized implementation."""

from typing import List, Optional
import pandas as pd
import numpy as np

from strategies.core.types import Direction, TriggerEvent


class BreakoutDetector:
    """
    Detects breakout events from price level crossings.

    A breakout occurs when price crosses a key level:
    - Bull breakout: close crosses above level (prev_close <= level < close)
    - Bear breakout: close crosses below level (prev_close >= level > close)
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        level_cols: List[str],
        rth_only: bool = True
    ):
        """
        Initialize detector.

        Args:
            bars: DataFrame with OHLCV data and level columns
            level_cols: List of column names containing price levels
            rth_only: Only detect breakouts during RTH (when ovn == 0)
        """
        self.bars = bars
        self.level_cols = level_cols
        self.rth_only = rth_only

    def detect(self) -> List[TriggerEvent]:
        """
        Detect all breakout events.

        Returns:
            List of TriggerEvent sorted by timestamp
        """
        events = []

        c = self.bars['close'].values.astype(np.float64)
        prev_close = np.roll(c, 1)
        prev_close[0] = np.nan

        # RTH mask
        if self.rth_only and 'ovn' in self.bars.columns:
            rth_mask = (self.bars['ovn'] == 0).values
        else:
            rth_mask = np.ones(len(self.bars), dtype=bool)

        bar_index = self.bars.index

        for lvl_col in self.level_cols:
            if lvl_col not in self.bars.columns:
                continue

            L = self.bars[lvl_col].values

            # Bull breakout: price crosses up through level
            bull_mask = (prev_close <= L) & (c >= L) & rth_mask & ~np.isnan(L)

            # Bear breakout: price crosses down through level
            bear_mask = (prev_close >= L) & (c <= L) & rth_mask & ~np.isnan(L)

            # Collect bull breakouts
            bull_indices = np.where(bull_mask)[0]
            for idx in bull_indices:
                events.append(TriggerEvent(
                    level_name=lvl_col,
                    level_price=float(L[idx]),
                    trigger_ts=bar_index[idx],
                    direction=Direction.BULL,
                    bar_index=idx
                ))

            # Collect bear breakouts
            bear_indices = np.where(bear_mask)[0]
            for idx in bear_indices:
                events.append(TriggerEvent(
                    level_name=lvl_col,
                    level_price=float(L[idx]),
                    trigger_ts=bar_index[idx],
                    direction=Direction.BEAR,
                    bar_index=idx
                ))

        # Sort by timestamp
        events.sort(key=lambda e: e.trigger_ts)

        return events

    def detect_vectorized(self) -> pd.DataFrame:
        """
        Detect breakouts and return as DataFrame.

        Returns:
            DataFrame with columns: level_name, level_price, trigger_ts, direction, bar_index
        """
        events = self.detect()

        if not events:
            return pd.DataFrame(columns=[
                'level_name', 'level_price', 'trigger_ts', 'direction', 'bar_index'
            ])

        return pd.DataFrame([
            {
                'level_name': e.level_name,
                'level_price': e.level_price,
                'trigger_ts': e.trigger_ts,
                'direction': str(e.direction),
                'bar_index': e.bar_index
            }
            for e in events
        ])
