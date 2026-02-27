"""Entry point: uvicorn + APScheduler.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -m dashboard.server

Environment variables:
    DASHBOARD_DATA_MODE   csv_plus_yfinance (default) | yfinance_only
    DASHBOARD_REFRESH_SECRET  secret for POST /api/refresh (default: changeme)
    DASHBOARD_STRICT_BACKFILL  optional bool override (true/false)
    DASHBOARD_PORT        port (default: 8000)
    DASHBOARD_HOST        host (default: 0.0.0.0)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import concurrent.futures
from pathlib import Path

import uvicorn

# ── Logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dashboard.server")

# ── Config from env ────────────────────────────────────────────────────
DATA_MODE = os.getenv("DASHBOARD_DATA_MODE", "csv_plus_yfinance")
REFRESH_SECRET = os.getenv("DASHBOARD_REFRESH_SECRET", "changeme")
PORT = int(os.getenv("DASHBOARD_PORT", "8000"))
HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r (expected true/false). Using default=%s.", name, raw, default)
    return default


STRICT_BACKFILL = _parse_bool_env("DASHBOARD_STRICT_BACKFILL", default=False)


async def main():
    from dashboard.runner import SwingDashboardRunner, DashboardRunnerConfig
    from dashboard.app import create_app, _background_refresh
    from dashboard.scheduler import create_scheduler

    # ── 1. Build runner ────────────────────────────────────────────────
    config = DashboardRunnerConfig(
        data_mode=DATA_MODE,
        refresh_secret=REFRESH_SECRET,
        strict_backfill=STRICT_BACKFILL,
    )
    runner = SwingDashboardRunner(config)

    # ── 2. Try loading cached state (fast path) ────────────────────────
    cached = runner.load_cached_state()
    if cached is not None:
        runner._last_state = cached
        logger.info("Serving stale cached state while fresh refresh runs in background")

    # ── 3. Shared WebSocket queue ──────────────────────────────────────
    ws_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    # ── 4. Build FastAPI app ───────────────────────────────────────────
    app = create_app(runner, ws_queue, refresh_secret=REFRESH_SECRET)

    # ── 5. Build scheduler ────────────────────────────────────────────
    scheduler = create_scheduler(runner, app)

    # ── 6. Configure uvicorn ──────────────────────────────────────────
    uv_config = uvicorn.Config(
        app=app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    # ── 7. Startup: initial refresh in background ──────────────────────
    async def startup():
        scheduler.start()
        logger.info("Scheduler started")

        if cached is None:
            logger.info("No cached state — running initial refresh (may take 1-2 min)...")
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                await loop.run_in_executor(pool, runner.refresh)
            logger.info("Initial refresh complete — state computed in %.1fs",
                        runner.last_state.refresh_duration_sec if runner.last_state else 0)
        else:
            # Schedule background refresh to get fresh data
            logger.info("Triggering background refresh for fresh data...")
            asyncio.create_task(_background_refresh(runner, ws_queue))

    # ── 8. Run everything ─────────────────────────────────────────────
    # Register startup hook with uvicorn
    original_startup = server.startup

    async def patched_startup(sockets=None):
        await startup()
        await original_startup(sockets=sockets)

    server.startup = patched_startup

    logger.info("Starting dashboard server on http://%s:%d", HOST, PORT)
    await server.serve()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
