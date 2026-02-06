"""CLI entry point: python -m strategies.realtime"""

import argparse
import logging
import sys

from strategies.realtime.config import EngineConfig
from strategies.realtime.engine import RealtimeEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time trading engine",
    )
    parser.add_argument(
        "--no-gex", action="store_true", help="Disable GEX provider",
    )
    parser.add_argument(
        "--no-discord", action="store_true", help="Log-only mode (no Discord)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["breakout", "reversion"],
        default=None,
        help="Only run specific strategies (default: all)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Update interval in seconds (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = EngineConfig.default()
    config.gex_enabled = not args.no_gex
    config.discord_enabled = not args.no_discord
    config.update_interval_sec = args.interval

    # Filter strategies if requested
    if args.strategies:
        config.strategies = [
            s for s in config.strategies if s.strategy_type in args.strategies
        ]

    engine = RealtimeEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
