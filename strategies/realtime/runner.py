"""CLI entry point: python -m strategies.realtime"""

import argparse
import datetime as dt
import logging
from zoneinfo import ZoneInfo

from strategies.data.history_sync import DEFAULT_SCHWAB_MINUTE_CSV
from strategies.realtime.config import (
    EngineConfig,
    PlaybackConfig,
    RealtimeStrategyConfig,
)
from strategies.realtime.engine import RealtimeEngine
from strategies.realtime.strategy_cli import (
    add_footprint_predictor_args,
    add_reversal_predictor_args,
    build_footprint_predictor_config,
    build_reversal_predictor_config,
)

LA_TZ = ZoneInfo("America/Los_Angeles")
logger = logging.getLogger(__name__)


def _batch_strategy_name(cfg: RealtimeStrategyConfig) -> str:
    name = cfg.params.get("strategy_name")
    if isinstance(name, str) and name:
        return name
    if cfg.kind.startswith("batch_"):
        return cfg.kind.replace("batch_", "", 1)
    return cfg.kind


def _maybe_sync_history(args: argparse.Namespace) -> None:
    if args.playback or args.playback_sql_day or args.skip_history_sync:
        return

    try:
        from strategies.data.history_sync import MinuteHistorySyncConfig, sync_minute_history
        from strategies.data.schwab import SchwabAuthConfig, SchwabClient
    except Exception as e:
        logger.warning("History sync unavailable (import error): %s", e)
        return

    try:
        auth = SchwabAuthConfig.from_env(allow_missing=True)
        if auth is None:
            logger.warning(
                "Skipping Schwab history sync: missing SCHWAB_APP_KEY/SCHWAB_APP_SECRET."
            )
            return

        client = SchwabClient(auth)
        sync_cfg = MinuteHistorySyncConfig(
            symbol=args.history_sync_symbol,
            csv_path=args.history_sync_csv,
            frequency_type=args.history_sync_frequency_type,
            frequency=args.history_sync_frequency,
            include_extended_hours=not args.history_sync_no_extended_hours,
            lookback_days_if_missing=args.history_sync_lookback_days,
            stale_after_minutes=args.history_sync_stale_minutes,
        )
        result = sync_minute_history(client, sync_cfg)
        logger.info(
            "History sync [%s]: symbol=%s rows_before=%d rows_after=%d rows_added=%d "
            "candles_fetched=%d latest=%s (%s)",
            result.status,
            result.symbol,
            result.rows_before,
            result.rows_after,
            result.rows_added,
            result.candles_fetched,
            result.latest_dt,
            result.message,
        )
    except Exception as e:
        logger.warning("Schwab history sync failed: %s", e, exc_info=args.verbose)


def _parse_local_datetime_to_ts(raw: str) -> int:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty datetime string")
    formats = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")
    parsed = None
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(
            f"Invalid datetime '{raw}'. Expected 'YYYY-MM-DD HH:MM[:SS]' in America/Los_Angeles."
        )
    local = parsed.replace(tzinfo=LA_TZ)
    return int(local.timestamp())


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
        "--signal-jsonl-path",
        type=str,
        default=None,
        help=(
            "Optional JSONL output path for emitted realtime signals "
            "(used for external execution bridges such as NinjaTrader)."
        ),
    )
    parser.add_argument(
        "--signal-jsonl-truncate-on-start",
        action="store_true",
        help="If set, truncate --signal-jsonl-path at startup.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["breakout", "reversion"],
        default=None,
        help="Opt-in batch strategies to run. If omitted, breakout/reversion are disabled.",
    )
    parser.add_argument(
        "--reversion-model-path",
        type=str,
        default=None,
        help="Override XGBoost filter model path for reversion strategy slot.",
    )
    parser.add_argument(
        "--breakout-model-path",
        type=str,
        default=None,
        help="Override XGBoost filter model path for breakout strategy slot.",
    )
    parser.add_argument(
        "--disable-strategy-model-filters",
        action="store_true",
        help="Run breakout/reversion slots without their XGBoost model filters.",
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
    add_reversal_predictor_args(parser)
    add_footprint_predictor_args(parser)
    parser.add_argument(
        "--skip-history-sync",
        action="store_true",
        help="Disable startup sync of local minute history from Schwab API.",
    )
    parser.add_argument(
        "--history-sync-symbol",
        type=str,
        default="/ES",
        help="Schwab symbol to sync on startup (default: /ES).",
    )
    parser.add_argument(
        "--history-sync-csv",
        type=str,
        default=DEFAULT_SCHWAB_MINUTE_CSV,
        help="Local minute CSV updated during startup sync.",
    )
    parser.add_argument(
        "--history-sync-lookback-days",
        type=int,
        default=365,
        help="Initial backfill days when --history-sync-csv does not exist.",
    )
    parser.add_argument(
        "--history-sync-stale-minutes",
        type=int,
        default=30,
        help="Skip remote fetch when local CSV is newer than this age.",
    )
    parser.add_argument(
        "--history-sync-frequency-type",
        type=str,
        default="minute",
        help="Schwab pricehistory frequencyType for startup sync.",
    )
    parser.add_argument(
        "--history-sync-frequency",
        type=int,
        default=1,
        help="Schwab pricehistory frequency for startup sync.",
    )
    parser.add_argument(
        "--history-sync-no-extended-hours",
        action="store_true",
        help="Disable extended-hours candles in startup history sync.",
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
    parser.add_argument(
        "--playback-sql-day",
        nargs="+",
        default=None,
        help="SQL playback trading_day(s), e.g. 2026-02-20. Uses MySQL second data.",
    )
    parser.add_argument(
        "--playback-step-sec",
        type=int,
        default=60,
        help="Playback step size in seconds for non-fast-forward playback.",
    )
    parser.add_argument(
        "--playback-fast-forward",
        action="store_true",
        help="For SQL playback, run single-timestamp snapshot mode (initialize at end time + one strategy pass).",
    )
    parser.add_argument(
        "--playback-init-local",
        type=str,
        default=None,
        help="Optional SQL playback engine init time in LA timezone: 'YYYY-MM-DD HH:MM[:SS]'.",
    )
    parser.add_argument(
        "--playback-end-local",
        type=str,
        default=None,
        help="Optional SQL playback end time in LA timezone: 'YYYY-MM-DD HH:MM[:SS]'.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    _maybe_sync_history(args)

    config = EngineConfig.default()
    config.gex_enabled = not args.no_gex
    config.discord_enabled = not args.no_discord
    config.signal_jsonl_path = args.signal_jsonl_path
    config.signal_jsonl_truncate_on_start = bool(args.signal_jsonl_truncate_on_start)
    config.update_interval_sec = args.interval

    for slot in config.strategy_configs:
        if slot.kind not in ("batch_reversion", "batch_breakout"):
            continue
        if args.disable_strategy_model_filters:
            slot.params["model_path"] = None
        elif slot.kind == "batch_reversion" and args.reversion_model_path is not None:
            slot.params["model_path"] = args.reversion_model_path
        elif slot.kind == "batch_breakout" and args.breakout_model_path is not None:
            slot.params["model_path"] = args.breakout_model_path

    reversal_cfg = build_reversal_predictor_config(
        args,
        default_history_csv=args.history_sync_csv,
    )
    if reversal_cfg is not None:
        config.strategy_configs.append(reversal_cfg)

    footprint_cfg = build_footprint_predictor_config(
        args,
        default_history_csv=args.history_sync_csv,
    )
    if footprint_cfg is not None:
        config.strategy_configs.append(footprint_cfg)

    allowed = set(args.strategies or [])
    config.strategy_configs = [
        cfg
        for cfg in config.strategy_configs
        if cfg.kind not in ("batch_breakout", "batch_reversion")
        or _batch_strategy_name(cfg) in allowed
    ]

    if not args.strategies:
        logger.info(
            "Batch strategies disabled by default. Use --strategies breakout/reversion to enable them."
        )

    if args.playback_sql_day:
        from strategies.realtime.data_source import MySQLSource, get_session_window_for_trading_day
        from strategies.realtime.playback import CollectingSignalHandler

        for slot in config.strategy_configs:
            if slot.kind in ("batch_breakout", "batch_reversion"):
                slot.params["model_path"] = None

        init_ts = _parse_local_datetime_to_ts(args.playback_init_local) if args.playback_init_local else None
        end_ts = _parse_local_datetime_to_ts(args.playback_end_local) if args.playback_end_local else None

        source = MySQLSource(config.db)
        engine = RealtimeEngine(config, data_source=source)
        collector = CollectingSignalHandler()
        engine.signal_handler = collector

        bars_per_day = {}
        played_days = []
        t0 = dt.datetime.now()

        for day in [str(d) for d in args.playback_sql_day]:
            collector.set_day(day)
            search_start, search_end = get_session_window_for_trading_day(day)
            data_min_ts, data_max_ts = source.fetch_range_bounds(search_start, search_end)
            if data_min_ts <= 0 or data_max_ts <= 0:
                logger.warning("No SQL rows found for trading_day=%s", day)
                continue

            if args.playback_fast_forward:
                snapshot_ts = end_ts if end_ts is not None else init_ts
                run_end_ts = max(data_min_ts, min(snapshot_ts or data_max_ts, data_max_ts))
                run_init_ts = run_end_ts
            else:
                run_init_ts = max(data_min_ts, min(init_ts or data_min_ts, data_max_ts))
                run_end_ts = max(run_init_ts, min(end_ts or data_max_ts, data_max_ts))

            logger.info(
                "SQL playback trading_day=%s init_ts=%d end_ts=%d data_min=%d data_max=%d",
                day, run_init_ts, run_end_ts, data_min_ts, data_max_ts,
            )

            if args.playback_fast_forward:
                # Single-timestamp snapshot mode: initialize as-of end_ts and run one
                # strategy pass over accumulated bars (no stepped replay loop).
                engine.initialize(run_end_ts)
                engine.run_strategy_pass(dispatch=True)
            else:
                engine.initialize(run_init_ts)
                engine.replay_to(
                    run_end_ts,
                    step_sec=args.playback_step_sec,
                    fast_forward=False,
                )

            bars_per_day[day] = len(engine.min_df)
            played_days.append(day)
            logger.info(
                "  %s: %d bars, %d signals",
                day,
                bars_per_day[day],
                len(collector.day_signals.get(day, [])),
            )

            engine.reset_day_state()

        elapsed_sec = (dt.datetime.now() - t0).total_seconds()

        print(f"\n{'='*60}")
        print(f"SQL playback complete: {len(played_days)} days in {elapsed_sec:.1f}s")
        print(f"Total signals: {len(collector.signals)}")
        for day in played_days:
            n_sig = len(collector.day_signals.get(day, []))
            n_bars = bars_per_day.get(day, 0)
            print(f"  {day}: {n_bars} bars, {n_sig} signals")
            for sig in collector.day_signals.get(day, []):
                print(f"    {sig}")
        print(f"{'='*60}")
    elif args.playback:
        if not args.csv:
            parser.error("--playback requires --csv")

        from strategies.realtime.playback import PlaybackRunner

        # Playback defaults to unfiltered enabled batch strategies.
        for slot in config.strategy_configs:
            if slot.kind in ("batch_breakout", "batch_reversion"):
                slot.params["model_path"] = None

        playback_config = PlaybackConfig(
            csv_path=args.csv,
            playback_days=args.playback_days,
            n_days=args.n_days,
        )
        runner = PlaybackRunner(config, playback_config)
        result = runner.run()

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
