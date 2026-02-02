"""Reversion event detection with vectorized implementation."""

from typing import List, Optional
import pandas as pd
import numpy as np

from strategies.core.types import Direction, TriggerEvent


class ReversionDetector:
    """
    Detects reversion (rejection) events at price levels.

    A reversion occurs when price touches a level but closes on the
    opposite side, indicating rejection:

    - Bull reversion: Open above level, low touches level, close above level
      (Price tried to break down but was rejected)

    - Bear reversion: Open below level, high touches level, close below level
      (Price tried to break up but was rejected)
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
            rth_only: Only detect reversions during RTH (when ovn == 0)
        """
        self.bars = bars
        self.level_cols = level_cols
        self.rth_only = rth_only

    def detect(self) -> List[TriggerEvent]:
        """
        Detect all reversion events.

        Returns:
            List of TriggerEvent sorted by timestamp
        """
        events = []

        o = self.bars['open'].values.astype(np.float64)
        h = self.bars['high'].values.astype(np.float64)
        l = self.bars['low'].values.astype(np.float64)
        c = self.bars['close'].values.astype(np.float64)

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

            # Bull reversion: open above level, low touches level, close above level
            # This indicates a failed breakdown - bullish signal
            bull_mask = (o > L) & (l <= L) & (c > L) & rth_mask & ~np.isnan(L)

            # Bear reversion: open below level, high touches level, close below level
            # This indicates a failed breakout - bearish signal
            bear_mask = (o < L) & (h >= L) & (c < L) & rth_mask & ~np.isnan(L)

            # Collect bull reversions
            bull_indices = np.where(bull_mask)[0]
            for idx in bull_indices:
                events.append(TriggerEvent(
                    level_name=lvl_col,
                    level_price=float(L[idx]),
                    trigger_ts=bar_index[idx],
                    direction=Direction.BULL,
                    bar_index=idx
                ))

            # Collect bear reversions
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
        Detect reversions and return as DataFrame.

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
