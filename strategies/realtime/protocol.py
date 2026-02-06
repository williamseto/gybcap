"""Strategy protocol and adapter for real-time trading."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Protocol, Set, Tuple, runtime_checkable
import pandas as pd
import numpy as np
import xgboost as xgb

from strategies.core.types import Direction, TriggerEvent, TradeSignal
from strategies.core.base import BaseStrategy


@dataclass
class RealtimeSignal:
    """A signal emitted by a real-time strategy."""
    strategy_name: str
    trigger_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    entry_price: float
    direction: str          # 'bull' / 'bear'
    level_name: str
    level_value: float
    pred_proba: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.strategy_name} {self.direction} @ {self.level_name}="
            f"{self.level_value:.2f} entry={self.entry_price:.2f} "
            f"prob={self.pred_proba:.2f} [{self.entry_ts}]"
        )


@runtime_checkable
class RealtimeStrategy(Protocol):
    """Plug-and-play interface for real-time strategies."""

    @property
    def name(self) -> str: ...

    def process(self, bars: pd.DataFrame) -> List[RealtimeSignal]: ...


class BatchStrategyAdapter:
    """
    Wraps a batch BaseStrategy subclass for real-time use.

    On each process() call:
    1. Computes levels on the provided 1-min bars via the level_provider
    2. Instantiates the batch strategy with the resampled bars
    3. Calls detect_triggers() + find_entries()
    4. Applies XGBoost model filtering (pred_proba >= threshold)
    5. De-duplicates against previously-emitted signals
    6. Returns new RealtimeSignal objects
    """

    def __init__(
        self,
        strategy_name: str,
        strategy_cls: type,
        level_provider: Any,
        model_path: Optional[str] = None,
        level_cols: Optional[List[str]] = None,
        threshold_pct: float = 0.0012,
        lookahead_bars: int = 12,
        pred_threshold: float = 0.4,
        t_samp: str = '5Min',
        stop_buffer_pct: float = 0.0025,
        rr: float = 1.5,
    ):
        self._name = strategy_name
        self._strategy_cls = strategy_cls
        self._level_provider = level_provider
        self._level_cols = level_cols
        self._threshold_pct = threshold_pct
        self._lookahead_bars = lookahead_bars
        self._pred_threshold = pred_threshold
        self._t_samp = t_samp
        self._stop_buffer_pct = stop_buffer_pct
        self._rr = rr

        # XGBoost model for filtering
        self._model: Optional[xgb.XGBClassifier] = None
        if model_path:
            self._model = xgb.XGBClassifier()
            self._model.load_model(model_path)

        # Track emitted signals for dedup: set of (level_name, direction, trigger_ts)
        self._emitted: Set[Tuple[str, str, pd.Timestamp]] = set()

    @property
    def name(self) -> str:
        return self._name

    def process(self, bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        """
        Process 1-min bars and return new signals.

        Args:
            bars_1m: 1-minute OHLCV bars with datetime index (LA timezone)
        """
        # Attach levels and resample to strategy timeframe
        bars = self._level_provider.attach_levels_to_bars(bars_1m, t_samp=self._t_samp)

        if bars.empty:
            return []

        # Filter to level_cols that exist in bars
        available_cols = [c for c in (self._level_cols or []) if c in bars.columns]
        if not available_cols:
            return []

        # Instantiate the batch strategy with prepared bars
        strat: BaseStrategy = self._strategy_cls(
            bars=bars,
            level_cols=available_cols,
            threshold_pct=self._threshold_pct,
            lookahead_bars=self._lookahead_bars,
        )

        triggers = strat.detect_triggers()
        if not triggers:
            return []

        signals = strat.find_entries(triggers, self._stop_buffer_pct, self._rr)
        if not signals:
            return []

        # Apply model filtering and dedup
        new_signals: List[RealtimeSignal] = []

        for signal in signals:
            direction_str = str(signal.direction)
            dedup_key = (signal.level_name, direction_str, signal.trigger_event.trigger_ts)

            if dedup_key in self._emitted:
                continue

            # Model filtering
            pred_proba = 1.0
            if self._model is not None:
                feat_cols = self._level_provider.feat_cols
                entry_row = bars.iloc[signal.entry_bar_index]
                bear = 1 if signal.direction == Direction.BEAR else 0
                feature_vec = np.append(entry_row[feat_cols].values, bear)
                pred = self._model.predict_proba(feature_vec.reshape(1, -1))
                pred_proba = float(pred[0][1])

                if pred_proba < self._pred_threshold:
                    continue

            self._emitted.add(dedup_key)

            new_signals.append(RealtimeSignal(
                strategy_name=self._name,
                trigger_ts=signal.trigger_event.trigger_ts,
                entry_ts=signal.entry_ts,
                entry_price=signal.entry_price,
                direction=direction_str,
                level_name=signal.level_name,
                level_value=signal.level_price,
                pred_proba=pred_proba,
            ))

        return new_signals
