"""
strategies.realtime - Modular real-time trading engine.

Plug-and-play strategy system: any object implementing the
RealtimeStrategy protocol can be registered with the engine.

Usage:
    python -m strategies.realtime                     # realtime engine (batch strategies off by default)
    python -m strategies.realtime --no-gex            # disable GEX
    python -m strategies.realtime --no-discord         # log-only mode
    python -m strategies.realtime --strategies breakout  # enable breakout batch strategy
"""

from strategies.realtime.protocol import RealtimeSignal, RealtimeStrategy, BatchStrategyAdapter
from strategies.realtime.config import (
    EngineConfig,
    RealtimeStrategyConfig,
    DatabaseConfig,
    DEFAULT_LEVEL_COLS,
    PlaybackConfig,
)
from strategies.realtime.engine import RealtimeEngine
from strategies.realtime.playback import PlaybackRunner
from strategies.realtime.strategy_factory import RealtimeStrategyFactory, create_default_strategy_factory

__all__ = [
    "RealtimeSignal",
    "RealtimeStrategy",
    "BatchStrategyAdapter",
    "EngineConfig",
    "RealtimeStrategyConfig",
    "DatabaseConfig",
    "DEFAULT_LEVEL_COLS",
    "PlaybackConfig",
    "RealtimeEngine",
    "PlaybackRunner",
    "RealtimeStrategyFactory",
    "create_default_strategy_factory",
]
