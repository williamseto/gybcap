"""FastAPI application: REST endpoints + WebSocket for live signals."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path

from dashboard.state import DashboardState, make_empty_state

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(
    runner: "SwingDashboardRunner",  # noqa: F821
    ws_queue: asyncio.Queue,
    refresh_secret: str = "changeme",
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        runner: SwingDashboardRunner instance (holds last_state).
        ws_queue: Shared asyncio.Queue for pushing messages to WebSocket clients.
        refresh_secret: Secret header value required for POST /api/refresh.
    """
    app = FastAPI(title="Swing Strategy Dashboard", version="1.0.0")

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _current_state() -> DashboardState:
        state = runner.last_state
        if state is None:
            return make_empty_state("Initializing — first refresh not yet complete")
        return state

    # ────────────────────────────────────────────────────────────────────
    # REST endpoints
    # ────────────────────────────────────────────────────────────────────

    @app.get("/api/state")
    async def get_state():
        """Full DashboardState JSON."""
        return JSONResponse(content=_current_state().to_dict())

    @app.get("/api/today")
    async def get_today():
        """Today's DayState only (lightweight)."""
        state = _current_state()
        if state.today is None:
            return JSONResponse(content={"error": "Not yet computed"}, status_code=503)
        return JSONResponse(content=state.today.to_dict())

    @app.get("/api/history")
    async def get_history(days: int = 252):
        """Array of DayState for the last N days."""
        state = _current_state()
        history = state.history[-days:]
        return JSONResponse(content=[h.to_dict() for h in history])

    @app.get("/api/signals")
    async def get_signals():
        """Intraday signals list."""
        return JSONResponse(content=_current_state().intraday_signals)

    @app.post("/api/signals")
    async def post_signal(payload: dict):
        """Receive a signal from the realtime engine (or any external caller)."""
        state = _current_state()
        state.intraday_signals.append(payload)
        if len(state.intraday_signals) > 50:
            state.intraday_signals = state.intraday_signals[-50:]
        await ws_queue.put({"type": "signal", "data": payload})
        return {"status": "ok"}

    @app.post("/api/refresh")
    async def manual_refresh(x_refresh_secret: Optional[str] = Header(None)):
        """Trigger a manual pipeline refresh (protected by X-Refresh-Secret header)."""
        if x_refresh_secret != refresh_secret:
            raise HTTPException(status_code=403, detail="Invalid refresh secret")
        # Run refresh in background so we don't block the request
        asyncio.create_task(_background_refresh(runner, ws_queue))
        return {"status": "refresh_started"}

    @app.get("/health")
    async def health():
        state = _current_state()
        return {
            "status": "ok" if state.error is None else "error",
            "computed_at": state.computed_at,
            "as_of_date": state.as_of_date,
            "error": state.error,
        }

    # ────────────────────────────────────────────────────────────────────
    # WebSocket
    # ────────────────────────────────────────────────────────────────────

    _ws_clients: list[WebSocket] = []

    @app.websocket("/ws/signals")
    async def ws_signals(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        logger.info("WebSocket client connected (%d total)", len(_ws_clients))
        try:
            while True:
                # Wait for a message in the queue with a 30s timeout (heartbeat)
                try:
                    msg = await asyncio.wait_for(ws_queue.get(), timeout=30.0)
                    await websocket.send_text(json.dumps(msg))
                except asyncio.TimeoutError:
                    # Heartbeat: send current risk score
                    state = _current_state()
                    risk = state.today.risk_score if state.today else 0.0
                    regime = state.today.risk_regime if state.today else 0
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat",
                        "risk_score": risk,
                        "risk_regime": regime,
                        "computed_at": state.computed_at,
                    }))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WebSocket error: %s", e)
        finally:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
            logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))

    # ────────────────────────────────────────────────────────────────────
    # Broadcast helper (used by scheduler after each refresh)
    # ────────────────────────────────────────────────────────────────────

    async def broadcast_state_update():
        """Broadcast a state_update message to all connected WebSocket clients."""
        state = _current_state()
        msg = json.dumps({
            "type": "state_update",
            "computed_at": state.computed_at,
            "as_of_date": state.as_of_date,
        })
        dead = []
        for ws in list(_ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in _ws_clients:
                _ws_clients.remove(ws)

    # Attach broadcast helper to app so scheduler can call it
    app.state.broadcast_state_update = broadcast_state_update
    app.state.ws_clients = _ws_clients

    # ────────────────────────────────────────────────────────────────────
    # Static files (served last so API routes take priority)
    # ────────────────────────────────────────────────────────────────────

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


async def _background_refresh(runner, ws_queue: asyncio.Queue):
    """Run the pipeline refresh in a thread pool to avoid blocking the event loop."""
    import concurrent.futures
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        await loop.run_in_executor(pool, runner.refresh)
    # Notify WebSocket clients that state has been updated
    await ws_queue.put({"type": "state_update"})
    logger.info("Manual refresh complete")
