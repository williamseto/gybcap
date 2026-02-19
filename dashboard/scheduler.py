"""APScheduler: fires daily refresh at 4:30 PM and 5:00 PM ET on weekdays."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def create_scheduler(runner, app) -> AsyncIOScheduler:
    """Build and return the APScheduler instance.

    Jobs:
      - 4:30 PM ET Mon-Fri: primary daily refresh
      - 5:00 PM ET Mon-Fri: backup refresh (catches late-reporting bars)

    Args:
        runner: SwingDashboardRunner instance.
        app: FastAPI app (has app.state.broadcast_state_update coroutine).
    """
    scheduler = AsyncIOScheduler(timezone="America/New_York")

    async def _run_refresh():
        logger.info("Scheduled refresh triggered")
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            await loop.run_in_executor(pool, runner.refresh)

        # Notify WebSocket clients
        broadcast = getattr(app.state, "broadcast_state_update", None)
        if broadcast is not None:
            try:
                await broadcast()
            except Exception as e:
                logger.warning("Broadcast failed: %s", e)

        logger.info("Scheduled refresh complete — as_of=%s",
                    runner.last_state.as_of_date if runner.last_state else "?")

    # 4:30 PM ET (market close + 30 min)
    scheduler.add_job(
        _run_refresh,
        CronTrigger(hour=16, minute=30, day_of_week="mon-fri"),
        id="refresh_1630",
        name="Daily refresh 4:30 PM ET",
        replace_existing=True,
        misfire_grace_time=300,  # 5 min grace
    )

    # 5:00 PM ET (backup in case data is delayed)
    scheduler.add_job(
        _run_refresh,
        CronTrigger(hour=17, minute=0, day_of_week="mon-fri"),
        id="refresh_1700",
        name="Daily refresh 5:00 PM ET",
        replace_existing=True,
        misfire_grace_time=300,
    )

    return scheduler
