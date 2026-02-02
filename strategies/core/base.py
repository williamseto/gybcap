"""Base classes and protocols for strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Protocol
import pandas as pd
import numpy as np

from strategies.core.types import Direction, TriggerEvent, TradeSignal, TradeResult


@dataclass
class Trade:
    """
    Complete trade record with entry, exit, and P&L.

    This is the primary output type for strategy simulations.
    Compatible with legacy Trade dataclass from test_bo_retest.py.
    """
    entry_ts: pd.Timestamp
    entry_price: float
    direction: str  # 'bull' or 'bear' for legacy compatibility
    size: float
    stop: float
    take: Optional[float]
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    # Extended fields for new system
    level_name: Optional[str] = None
    entry_bar_index: Optional[int] = None
    exit_bar_index: Optional[int] = None
    exit_type: Optional[str] = None  # 'stop', 'take', 'close'
    features: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_signal_and_result(
        cls,
        signal: TradeSignal,
        result: TradeResult,
        size: float = 1.0
    ) -> "Trade":
        """Create Trade from TradeSignal and TradeResult."""
        return cls(
            entry_ts=signal.entry_ts,
            entry_price=signal.entry_price,
            direction=str(signal.direction),
            size=size,
            stop=signal.stop_price,
            take=signal.take_price,
            exit_ts=result.exit_ts,
            exit_price=result.exit_price,
            pnl=result.pnl * size,
            level_name=signal.level_name,
            entry_bar_index=signal.entry_bar_index,
            exit_bar_index=result.exit_bar_index,
            exit_type=result.exit_type,
        )

    def compute_pnl(self) -> float:
        """Calculate P&L from entry/exit prices."""
        if self.exit_price is None:
            return 0.0
        if self.direction == "bull":
            return (self.exit_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.exit_price) * self.size


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Subclasses must implement:
    - detect_triggers(): Find potential trade opportunities
    - find_entries(): Confirm entries from triggers
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        level_cols: List[str],
        threshold_pct: float = 0.0012,
        lookahead_bars: int = 40,
    ):
        """
        Initialize strategy.

        Args:
            bars: OHLCV DataFrame with price level columns attached
            level_cols: Names of columns containing price levels to trade
            threshold_pct: Threshold for entry confirmation (% of price)
            lookahead_bars: Number of bars to look ahead for entry
        """
        self.bars = bars.copy()
        self.level_cols = level_cols
        self.threshold_pct = threshold_pct
        self.lookahead_bars = lookahead_bars

        # Ensure bars are properly indexed
        if 'dt' in self.bars.columns and not isinstance(self.bars.index, pd.DatetimeIndex):
            self.bars = self.bars.set_index('dt')

    @abstractmethod
    def detect_triggers(self) -> List[TriggerEvent]:
        """
        Detect potential trade trigger events.

        Returns:
            List of TriggerEvent objects sorted by timestamp
        """
        pass

    @abstractmethod
    def find_entries(
        self,
        triggers: List[TriggerEvent],
        stop_buffer_pct: float,
        rr: float
    ) -> List[TradeSignal]:
        """
        Find confirmed entry signals from trigger events.

        Args:
            triggers: List of trigger events to evaluate
            stop_buffer_pct: Stop loss buffer as percentage
            rr: Reward-to-risk ratio for take profit

        Returns:
            List of TradeSignal objects with entry/exit prices
        """
        pass

    def find_retest_and_build_trades(
        self,
        stop_buffer_pct: float = 0.0025,
        rr: float = 1.5,
        fixed_size: float = 1.0
    ) -> Tuple[List[Trade], pd.DataFrame]:
        """
        Execute full strategy and return trades with features.

        This is the main entry point matching the legacy interface.

        Returns:
            trades: List of Trade objects
            trade_features_df: DataFrame with features for each trade
        """
        from strategies.core.trade_simulator import VectorizedTradeSimulator

        triggers = self.detect_triggers()
        signals = self.find_entries(triggers, stop_buffer_pct, rr)

        if not signals:
            return [], pd.DataFrame()

        # Build trade simulation inputs
        simulator = VectorizedTradeSimulator(self.bars)

        trades = []
        trade_features = []
        trade_succ = []
        trade_bear = []

        for signal in signals:
            result = simulator.simulate_single_trade(
                entry_idx=signal.entry_bar_index,
                entry_price=signal.entry_price,
                stop=signal.stop_price,
                take=signal.take_price,
                is_bull=(signal.direction == Direction.BULL)
            )

            trade = Trade.from_signal_and_result(signal, result, fixed_size)
            trades.append(trade)

            # Collect features for ML training
            trade_row = self.bars.iloc[signal.entry_bar_index].copy()
            trade_features.append(trade_row)
            trade_succ.append(1 if trade.pnl > 0 else 0)
            trade_bear.append(1 if signal.direction == Direction.BEAR else 0)

        trade_features_df = pd.DataFrame(trade_features).reset_index(drop=True)
        trade_features_df['y_succ'] = trade_succ
        trade_features_df['bear'] = trade_bear

        return trades, trade_features_df
