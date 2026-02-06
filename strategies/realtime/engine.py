"""Slim real-time engine orchestrator."""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Type

import pandas as pd

from strategies.breakout.strategy import BreakoutRetestStrategy
from strategies.reversion.strategy import ReversionStrategy
from strategies.realtime.config import EngineConfig, StrategySlotConfig
from strategies.realtime.data_source import MySQLSource, get_trading_day_start_ts
from strategies.realtime.bar_aggregator import BarAggregator
from strategies.realtime.level_provider import DayPriceLevelProvider
from strategies.realtime.protocol import RealtimeStrategy, RealtimeSignal, BatchStrategyAdapter
from strategies.realtime.signal_handler import (
    SignalHandler,
    CompositeSignalHandler,
    DiscordSignalHandler,
    LoggingSignalHandler,
)

logger = logging.getLogger(__name__)

# Map strategy type names to batch strategy classes
STRATEGY_REGISTRY: Dict[str, Type] = {
    'breakout': BreakoutRetestStrategy,
    'reversion': ReversionStrategy,
}


class RealtimeEngine:
    """
    Orchestrates data ingestion, bar aggregation, strategy execution,
    and signal dispatch in a real-time loop.
    """

    def __init__(self, config: EngineConfig):
        self.config = config

        # Data pipeline
        self.data_source = MySQLSource(config.db)
        self.aggregator = BarAggregator()
        self.level_provider = DayPriceLevelProvider(self.data_source)

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
        """Create and register BatchStrategyAdapters from EngineConfig slots."""
        for slot in self.config.strategies:
            if not slot.enabled:
                continue
            strategy_cls = STRATEGY_REGISTRY.get(slot.strategy_type)
            if strategy_cls is None:
                logger.warning("Unknown strategy type: %s", slot.strategy_type)
                continue

            adapter = BatchStrategyAdapter(
                strategy_name=slot.strategy_type,
                strategy_cls=strategy_cls,
                level_provider=self.level_provider,
                model_path=slot.model_path,
                level_cols=slot.level_cols,
                threshold_pct=slot.threshold_pct,
                lookahead_bars=slot.lookahead_bars,
                pred_threshold=slot.pred_threshold,
            )
            self.register_strategy(adapter)

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
            except Exception:
                pass

        # 5) Run strategies
        all_signals: List[RealtimeSignal] = []
        for strategy in self.strategies:
            try:
                signals = strategy.process(self.min_df)
                all_signals.extend(signals)
            except Exception as e:
                logger.error("Strategy %s error: %s", strategy.name, e)

        # 6) Dispatch signals
        if all_signals:
            self.signal_handler.handle(all_signals)

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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_signal_handler(config: EngineConfig) -> SignalHandler:
        composite = CompositeSignalHandler()
        composite.add(LoggingSignalHandler())
        if config.discord_enabled:
            composite.add(DiscordSignalHandler())
        return composite
