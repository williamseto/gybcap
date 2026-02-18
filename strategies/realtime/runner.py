"""CLI entry point: python -m strategies.realtime"""

import argparse
import logging
import sys

from strategies.realtime.config import EngineConfig, PlaybackConfig
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

    # Playback mode
    parser.add_argument(
        "--playback", action="store_true",
        help="Run in CSV playback mode (replay historical days)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to CSV file for playback mode",
    )
    parser.add_argument(
        "--playback-days", nargs="+", default=None,
        help="Specific trading days to replay (e.g. 2024-06-10)",
    )
    parser.add_argument(
        "--n-days", type=int, default=2,
        help="Number of recent days to replay (default: 2)",
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

    if args.playback:
        if not args.csv:
            parser.error("--playback requires --csv")

        from strategies.realtime.playback import PlaybackRunner

        # Clear model paths that may not exist — playback runs unfiltered
        for slot in config.strategies:
            slot.model_path = None

        playback_config = PlaybackConfig(
            csv_path=args.csv,
            playback_days=args.playback_days,
            n_days=args.n_days,
        )
        runner = PlaybackRunner(config, playback_config)
        result = runner.run()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Playback complete: {len(result.days_played)} days in {result.elapsed_sec:.1f}s")
        print(f"Total signals: {len(result.signals)}")
        for day in result.days_played:
            n_sig = len(result.day_signals.get(day, []))
            n_bars = result.bars_per_day.get(day, 0)
            print(f"  {day}: {n_bars} bars, {n_sig} signals")
            for sig in result.day_signals.get(day, []):
                print(f"    {sig}")
        print(f"{'='*60}")
    else:
        engine = RealtimeEngine(config)
        engine.run()


if __name__ == "__main__":
    main()
