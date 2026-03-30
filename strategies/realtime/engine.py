"""Slim real-time engine orchestrator."""

import logging
import time
from collections import deque
from typing import List, Optional

import pandas as pd

from strategies.realtime.bar_aggregator import BarAggregator
from strategies.realtime.config import EngineConfig
from strategies.realtime.data_source import MySQLSource, get_trading_day_start_ts
from strategies.realtime.level_provider import DayPriceLevelProvider
from strategies.realtime.protocol import RealtimeSignal, RealtimeStrategy
from strategies.realtime.signal_handler import (
    CompositeSignalHandler,
    DiscordSignalHandler,
    JsonlFileSignalHandler,
    LoggingSignalHandler,
    SignalHandler,
)
from strategies.realtime.strategy_factory import (
    RealtimeStrategyFactory,
    create_default_strategy_factory,
)

logger = logging.getLogger(__name__)


class RealtimeEngine:
    """
    Orchestrates data ingestion, bar aggregation, strategy execution,
    and signal dispatch in a real-time loop.
    """

    def __init__(
        self,
        config: EngineConfig,
        data_source=None,
        strategy_factory: Optional[RealtimeStrategyFactory] = None,
    ):
        self.config = config

        # Data pipeline
        self.data_source = data_source or MySQLSource(config.db)
        self.aggregator = BarAggregator()
        self.level_provider = DayPriceLevelProvider(self.data_source)
        self.strategy_factory = strategy_factory or create_default_strategy_factory()

        # State
        self.raw_deque: deque = deque(maxlen=config.max_window_sec)
        self.min_df = pd.DataFrame()
        self.last_ts: Optional[int] = None
        self.trading_day_end_ts: int = 0

        # Strategies
        self.strategies: List[RealtimeStrategy] = []

        # Signal handler
        self.signal_handler: SignalHandler = self._build_signal_handler(config)

        # GEX (optional)
        self.gex_provider = None
        if config.gex_enabled:
            try:
                from gex.gex_provider import RealtimeGEXProvider

                self.gex_provider = RealtimeGEXProvider()
            except Exception as e:
                logger.warning("GEX provider unavailable: %s", e)

    # ------------------------------------------------------------------
    # Strategy management
    # ------------------------------------------------------------------

    def register_strategy(self, strategy: RealtimeStrategy) -> None:
        self.strategies.append(strategy)
        logger.info("Registered strategy: %s", strategy.name)

    def remove_strategy(self, name: str) -> None:
        self.strategies = [s for s in self.strategies if s.name != name]
        logger.info("Removed strategy: %s", name)

    def _register_strategies_from_config(self) -> None:
        """Build and register strategies from unified strategy configs."""
        for slot in self.config.iter_enabled_strategy_configs():
            try:
                strategy = self.strategy_factory.build(slot, self)
                self.register_strategy(strategy)
            except Exception as e:
                logger.error(
                    "Failed to register strategy kind=%s name=%s: %s",
                    slot.kind,
                    slot.name or "",
                    e,
                    exc_info=True,
                )
                continue

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, now_ts: Optional[int] = None) -> None:
        """Bootstrap: fetch today's data, build bars, init levels & GEX."""
        if now_ts is None:
            now_ts = int(time.time())

        trading_day_start_ts = get_trading_day_start_ts(now_ts)
        self.trading_day_end_ts = trading_day_start_ts + 22 * 3600

        # Previous-day levels
        self.level_provider.initialize(now_ts)

        # Today's ticks
        sec_df = self.data_source.fetch_range(trading_day_start_ts, now_ts)
        min_df = self.aggregator.build_minute_bars(sec_df) if not sec_df.empty else pd.DataFrame()

        self.raw_deque.extend(sec_df.to_dict('records'))
        self.last_ts = int(sec_df['timestamp'].max()) if not sec_df.empty else trading_day_start_ts
        self.min_df = min_df

        # GEX initialization
        if self.gex_provider is not None and not min_df.empty:
            try:
                self.gex_provider.fetch_chain()
                rth_df = self.min_df.between_time('6:30', '12:59')
                if not rth_df.empty:
                    self.gex_provider.spx_es_spread = (
                        self.gex_provider.spot_price - rth_df['open'].iloc[0]
                    )
                    self.min_df['nearby_gamma_score'] = 0.0
                    self.min_df.loc[rth_df.index, 'nearby_gamma_score'] = (
                        self.gex_provider.compute_gex_score_batch_mapped(rth_df)
                    )
            except Exception as e:
                logger.warning("GEX init warning: %s", e)

        # Register configured strategies
        if not self.strategies:
            self._register_strategies_from_config()
        self._push_ticks_to_strategies(sec_df)

        logger.info(
            "Initialized: %d bars, %d strategies, day_end=%d",
            len(self.min_df), len(self.strategies), self.trading_day_end_ts,
        )

    def update(self, now_ts: int) -> None:
        """Single update cycle: fetch new ticks, update bars, run strategies."""
        # 1) Fetch new ticks
        df_new = self.data_source.fetch_since(self.last_ts, now_ts)
        if df_new.empty:
            return

        # 2) Update deque and last_ts
        for rec in df_new.to_dict('records'):
            self.raw_deque.append(rec)
        self.last_ts = int(df_new['timestamp'].max())

        # 3) Merge into minute bars
        deque_df = pd.DataFrame(self.raw_deque)
        self.min_df = self.aggregator.merge_incremental(
            self.min_df, df_new, deque_df,
        )

        # 4) GEX on new bars
        if self.gex_provider is not None:
            try:
                new_rth = self.min_df.between_time('6:30', '12:59')
                if not new_rth.empty and 'nearby_gamma_score' not in self.min_df.columns:
                    self.min_df['nearby_gamma_score'] = 0.0
                if not new_rth.empty:
                    self.min_df.loc[new_rth.index, 'nearby_gamma_score'] = (
                        self.gex_provider.compute_gex_score_batch_mapped(new_rth)
                    )
            except Exception as e:
                logger.warning("GEX update error: %s", e, exc_info=True)

        self._push_ticks_to_strategies(df_new)

        # 5/6) Run strategies + dispatch
        self.run_strategy_pass(dispatch=True)

    def run(self) -> None:
        """Main loop: initialize then poll until trading day ends."""
        self.initialize()

        while True:
            time.sleep(self.config.update_interval_sec)
            now_ts = int(time.time())

            self.update(now_ts)

            if now_ts >= self.trading_day_end_ts:
                logger.info("Trading day ended.")
                break

    # ------------------------------------------------------------------
    # Replay hooks
    # ------------------------------------------------------------------

    def replay_to(self, end_ts: int, step_sec: int = 60, fast_forward: bool = False) -> None:
        """Run update cycles until *end_ts*.

        This uses the same `update()` path as live mode and is intended for
        deterministic playback/backfill runs.
        """
        if self.last_ts is None:
            raise RuntimeError("Engine must be initialized before replay_to().")

        if end_ts <= self.last_ts:
            return

        step_sec = max(int(step_sec), 1)

        if fast_forward:
            self.update(int(end_ts))
            return

        sim_ts = int(self.last_ts) + step_sec
        while sim_ts <= int(end_ts):
            self.update(sim_ts)
            sim_ts += step_sec

    def run_strategy_pass(self, dispatch: bool = True) -> List[RealtimeSignal]:
        """Run one strategy pass against current bars without fetching new ticks."""
        all_signals: List[RealtimeSignal] = []
        for strategy in self.strategies:
            try:
                signals = strategy.process(self.min_df)
                all_signals.extend(signals)
            except Exception as e:
                logger.error("Strategy %s error: %s", strategy.name, e)

        if dispatch and all_signals:
            self.signal_handler.handle(all_signals)
        return all_signals

    def reset_day_state(self) -> None:
        """Clear intraday engine state and reset strategies for the next session."""
        self.min_df = self.min_df.iloc[0:0]
        self.last_ts = None
        self.raw_deque.clear()

        for strategy in self.strategies:
            reset_day = getattr(strategy, "reset_day", None)
            if callable(reset_day):
                try:
                    reset_day()
                    continue
                except Exception as e:
                    logger.warning("Strategy %s reset_day error: %s", strategy.name, e)
            if hasattr(strategy, "_emitted"):
                try:
                    strategy._emitted.clear()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signal_handler(config: EngineConfig) -> SignalHandler:
        composite = CompositeSignalHandler()
        composite.add(LoggingSignalHandler())
        if config.discord_enabled:
            composite.add(DiscordSignalHandler())
        if config.signal_jsonl_path:
            composite.add(
                JsonlFileSignalHandler(
                    output_path=str(config.signal_jsonl_path),
                    truncate_on_start=bool(config.signal_jsonl_truncate_on_start),
                )
            )
        return composite

    def _push_ticks_to_strategies(self, ticks_df: pd.DataFrame) -> None:
        if ticks_df is None or ticks_df.empty:
            return
        for strategy in self.strategies:
            on_new_ticks = getattr(strategy, "on_new_ticks", None)
            if callable(on_new_ticks):
                try:
                    on_new_ticks(ticks_df)
                except Exception as e:
                    logger.warning(
                        "Strategy %s on_new_ticks error: %s",
                        getattr(strategy, "name", strategy.__class__.__name__),
                        e,
                    )
