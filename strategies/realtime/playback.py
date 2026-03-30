"""CSV playback mode: replay historical days through the full engine pipeline."""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

from strategies.realtime.config import EngineConfig, PlaybackConfig
from strategies.realtime.csv_data_source import CSVDataSource
from strategies.realtime.engine import RealtimeEngine
from strategies.realtime.protocol import RealtimeSignal

logger = logging.getLogger(__name__)


class CollectingSignalHandler:
    """Stores signals in a list instead of dispatching externally."""

    def __init__(self):
        self.signals: List[RealtimeSignal] = []
        self.day_signals: Dict[str, List[RealtimeSignal]] = {}
        self._current_day: str = ''

    def set_day(self, day: str) -> None:
        self._current_day = day
        if day not in self.day_signals:
            self.day_signals[day] = []

    def handle(self, signals: List[RealtimeSignal]) -> None:
        self.signals.extend(signals)
        if self._current_day:
            self.day_signals[self._current_day].extend(signals)


@dataclass
class PlaybackResult:
    """Summary of a playback run."""
    signals: List[RealtimeSignal]
    day_signals: Dict[str, List[RealtimeSignal]]
    days_played: List[str]
    bars_per_day: Dict[str, int]
    elapsed_sec: float


class PlaybackRunner:
    """Replay historical trading days through the full realtime engine pipeline."""

    def __init__(self, config: EngineConfig, playback_config: PlaybackConfig):
        self.config = config
        self.playback_config = playback_config

    def run(self) -> PlaybackResult:
        t0 = time.time()

        csv_source = CSVDataSource(
            self.playback_config.csv_path, upsample=True,
        )

        # Determine which days to play
        all_days = csv_source.get_trading_days()
        if not all_days:
            raise ValueError("CSV has no trading_day column or no data")

        if self.playback_config.playback_days:
            playback_days = [
                d for d in self.playback_config.playback_days if d in all_days
            ]
            if not playback_days:
                raise ValueError(
                    f"None of the requested days found in CSV. "
                    f"Available: {all_days[:5]}...{all_days[-5:]}"
                )
        else:
            playback_days = all_days[-self.playback_config.n_days:]

        # Set up engine with CSV source and collecting handler
        engine = RealtimeEngine(self.config, data_source=csv_source)
        collector = CollectingSignalHandler()
        engine.signal_handler = collector

        bars_per_day: Dict[str, int] = {}

        for i, day in enumerate(playback_days):
            logger.info(
                "Playback day %d/%d: %s", i + 1, len(playback_days), day,
            )
            collector.set_day(day)

            day_start_ts, day_end_ts = csv_source.get_day_timestamp_range(day)
            if day_start_ts == 0:
                logger.warning("No data for day %s, skipping", day)
                continue

            # Initialize engine for this day
            engine.initialize(day_start_ts)
            engine.replay_to(day_end_ts, step_sec=60, fast_forward=False)

            bars_per_day[day] = len(engine.min_df)
            n_signals = len(collector.day_signals.get(day, []))
            logger.info(
                "  %s: %d bars, %d signals", day, bars_per_day[day], n_signals,
            )

            engine.reset_day_state()

        elapsed = time.time() - t0

        return PlaybackResult(
            signals=collector.signals,
            day_signals=collector.day_signals,
            days_played=playback_days,
            bars_per_day=bars_per_day,
            elapsed_sec=elapsed,
        )
