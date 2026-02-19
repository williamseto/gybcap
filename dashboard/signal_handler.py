"""DashboardSignalHandler: bridges realtime engine signals to WebSocket clients."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

MAX_SIGNALS = 50


class DashboardSignalHandler:
    """Implements the SignalHandler protocol from strategies/realtime/signal_handler.py.

    Stores intraday signals on the DashboardState and queues them for
    WebSocket broadcast. Attach to the realtime engine via:

        engine.signal_handler.add(DashboardSignalHandler(state, ws_queue))
    """

    def __init__(
        self,
        state_ref: "DashboardState",  # noqa: F821
        ws_queue: asyncio.Queue,
        max_signals: int = MAX_SIGNALS,
    ):
        from dashboard.state import DashboardState  # local import to avoid circulars
        self._state = state_ref
        self._ws_queue = ws_queue
        self._max_signals = max_signals

    def handle(self, signals: list) -> None:
        """Receive signals from the realtime engine (sync context)."""
        for sig in signals:
            data = self._serialize(sig)
            # Maintain rolling window
            self._state.intraday_signals.append(data)
            if len(self._state.intraday_signals) > self._max_signals:
                self._state.intraday_signals = self._state.intraday_signals[-self._max_signals:]
            # Queue for WebSocket broadcast (non-blocking)
            try:
                self._ws_queue.put_nowait({"type": "signal", "data": data})
            except asyncio.QueueFull:
                logger.warning("WebSocket queue full — dropping signal")

    @staticmethod
    def _serialize(sig) -> dict:
        """Convert a RealtimeSignal (or any object) to a plain dict."""
        if hasattr(sig, "__dict__"):
            d = {k: v for k, v in sig.__dict__.items()}
        elif hasattr(sig, "_asdict"):
            d = sig._asdict()
        else:
            d = {"raw": str(sig)}
        # Ensure timestamp is JSON-serializable
        if "timestamp" in d and isinstance(d["timestamp"], datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d
