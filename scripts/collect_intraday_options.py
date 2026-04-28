#!/usr/bin/env python
"""Collect intraday SPX option chain snapshots via Schwab API.

Usage::

    # Single snapshot (testing / manual)
    python scripts/collect_intraday_options.py --once

    # Run continuously during RTH, snapshot every 15 min
    python scripts/collect_intraday_options.py

    # Custom interval and symbol
    python scripts/collect_intraday_options.py --interval 10 --symbol '$SPX'

Designed to be launched as a cron job or systemd timer at market open::

    # crontab: run Mon-Fri at 9:25 ET, process handles its own RTH window
    25 9 * * 1-5 cd /path/to/gybcap && ~/ml-venv/bin/python scripts/collect_intraday_options.py >> logs/intraday_collect.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

import pytz

from gex.intraday_collector import IntradayOptionsCollector
from gex.intraday_storage import IntradayStorage

ET = pytz.timezone("US/Eastern")

# RTH window (inclusive)
RTH_START = dt.time(9, 30)
RTH_END = dt.time(16, 15)  # a few minutes after close for final snapshot


def _now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def _in_rth(now: dt.datetime) -> bool:
    t = now.time()
    wd = now.weekday()
    return wd < 5 and RTH_START <= t <= RTH_END


def run_once(collector: IntradayOptionsCollector, storage: IntradayStorage) -> None:
    """Take one snapshot and save."""
    now = _now_et()
    print(f"[{now:%Y-%m-%d %H:%M:%S ET}] Taking snapshot...")
    try:
        result = collector.snapshot()
        path = storage.save_snapshot(result)
        print(
            f"  Saved {len(result.chain)} contracts, "
            f"underlying={result.underlying_price:.2f}, "
            f"path={path}"
        )
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)


def run_loop(
    collector: IntradayOptionsCollector,
    storage: IntradayStorage,
    interval_min: int = 15,
) -> None:
    """Run continuously, taking snapshots every *interval_min* during RTH."""
    print(f"Starting intraday collection loop (interval={interval_min}min)")
    print(f"RTH window: {RTH_START} - {RTH_END} ET, Mon-Fri")

    while True:
        now = _now_et()

        if not _in_rth(now):
            # If after RTH, done for today
            if now.time() > RTH_END and now.weekday() < 5:
                print(f"[{now:%H:%M ET}] RTH closed. Exiting.")
                break
            # Before RTH or weekend — wait 60s and re-check
            time.sleep(60)
            continue

        run_once(collector, storage)

        # Sleep until next interval
        sleep_sec = interval_min * 60
        next_time = now + dt.timedelta(seconds=sleep_sec)
        print(f"  Next snapshot at ~{next_time:%H:%M ET}")
        time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description="Collect intraday SPX option snapshots")
    parser.add_argument("--once", action="store_true", help="Single snapshot then exit")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between snapshots (default 15)")
    parser.add_argument("--symbol", default="$SPX", help="Underlying symbol (default $SPX)")
    parser.add_argument("--strikes", type=int, default=80, help="Number of strikes per side (default 80)")
    parser.add_argument("--output-dir", default="data/intraday_options", help="Storage root directory")
    args = parser.parse_args()

    collector = IntradayOptionsCollector(
        symbol=args.symbol,
        strike_count=args.strikes,
    )
    storage = IntradayStorage(root=args.output_dir)

    if args.once:
        run_once(collector, storage)
    else:
        run_loop(collector, storage, interval_min=args.interval)


if __name__ == "__main__":
    main()
