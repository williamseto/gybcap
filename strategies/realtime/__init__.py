"""
strategies.realtime - Modular real-time trading engine.

Plug-and-play strategy system: any object implementing the
RealtimeStrategy protocol can be registered with the engine.

Usage:
    python -m strategies.realtime                     # default config
    python -m strategies.realtime --no-gex            # disable GEX
    python -m strategies.realtime --no-discord         # log-only mode
    python -m strategies.realtime --strategies breakout  # only breakout
"""

from strategies.realtime.protocol import RealtimeSignal, RealtimeStrategy, BatchStrategyAdapter
from strategies.realtime.config import EngineConfig, StrategySlotConfig, DatabaseConfig
from strategies.realtime.engine import RealtimeEngine

__all__ = [
    "RealtimeSignal",
    "RealtimeStrategy",
    "BatchStrategyAdapter",
    "EngineConfig",
    "StrategySlotConfig",
    "DatabaseConfig",
    "RealtimeEngine",
]
