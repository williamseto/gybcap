"""
strategies - Modular trading strategy framework

This package provides a plugin-based system for building and testing
trading strategies with support for:
- Vectorized trade simulation (numba)
- Pluggable feature providers (price levels, gamma, Dalton)
- Separate breakout and reversion strategies
- CLI for training and backtesting

Usage:
    python -m strategies train --strategy breakout --timeframe 5min
    python -m strategies train --strategy reversion --timeframe 15min
    python -m strategies backtest --strategy breakout --model models/bo_model.json
"""

from strategies.core.types import Direction, TriggerEvent, TradeSignal
from strategies.core.base import Trade, BaseStrategy
from strategies.core.trade_simulator import VectorizedTradeSimulator
from strategies.features.registry import FeatureRegistry, FeaturePipeline
from strategies.breakout.strategy import BreakoutRetestStrategy
from strategies.reversion.strategy import ReversionStrategy

__version__ = "1.0.0"

__all__ = [
    "Direction",
    "TriggerEvent",
    "TradeSignal",
    "Trade",
    "BaseStrategy",
    "VectorizedTradeSimulator",
    "FeatureRegistry",
    "FeaturePipeline",
    "BreakoutRetestStrategy",
    "ReversionStrategy",
]
