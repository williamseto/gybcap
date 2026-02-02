"""Core type definitions for trading strategies."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class Direction(Enum):
    """Trade direction."""
    BULL = "bull"
    BEAR = "bear"

    @classmethod
    def from_string(cls, s: str) -> "Direction":
        """Convert string to Direction enum."""
        if s.lower() == "bull":
            return cls.BULL
        elif s.lower() == "bear":
            return cls.BEAR
        else:
            raise ValueError(f"Unknown direction: {s}")

    def __str__(self) -> str:
        return self.value


@dataclass
class TriggerEvent:
    """Represents a potential trade trigger (breakout or reversion event)."""
    level_name: str       # Name of the price level (e.g., 'prev_high', 'vwap')
    level_price: float    # Price of the level
    trigger_ts: pd.Timestamp  # Timestamp when trigger occurred
    direction: Direction  # Bull or bear
    bar_index: int       # Integer index into bars DataFrame

    def to_tuple(self) -> tuple:
        """Convert to tuple for compatibility with legacy code."""
        return (self.level_name, self.level_price, self.trigger_ts, str(self.direction))


@dataclass
class TradeSignal:
    """
    A trade signal with entry/exit parameters.

    Created from a TriggerEvent after retest confirmation.
    """
    trigger_event: TriggerEvent
    entry_ts: pd.Timestamp
    entry_price: float
    entry_bar_index: int
    stop_price: float
    take_price: float
    risk: float

    @property
    def direction(self) -> Direction:
        return self.trigger_event.direction

    @property
    def level_name(self) -> str:
        return self.trigger_event.level_name

    @property
    def level_price(self) -> float:
        return self.trigger_event.level_price


@dataclass
class TradeResult:
    """Result of simulating a trade."""
    exit_ts: pd.Timestamp
    exit_price: float
    exit_bar_index: int
    exit_type: str  # 'stop', 'take', or 'close' (end of data)
    pnl: float

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
