"""Configuration dataclasses for the real-time engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


DEFAULT_LEVEL_COLS = [
    "prev_high",
    "prev_low",
    "vwap",
    "ovn_lo",
    "ovn_hi",
    "ib_lo",
    "ib_hi",
]


@dataclass
class DatabaseConfig:
    """MySQL connection parameters."""
    host: str = '172.30.144.1'
    port: int = 3306
    user: str = 'kibblesoup'
    password: str = 'kibblesoup'
    database: str = 'sys'
    table: str = 'price_data'
    symbol: str = 'ES'

    @property
    def connection_string(self) -> str:
        return (
            f"mysql+mysqlconnector://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class RealtimeStrategyConfig:
    """Generic strategy slot that supports strategy-specific params."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    name: Optional[str] = None


@dataclass
class RangePredictorSlotConfig:
    """Configuration for the range predictor integration."""
    model_dir: str = 'models/range_predictor'
    predictions_csv: str = 'data/range_predictions.csv'
    enabled: bool = True


@dataclass
class PlaybackConfig:
    """Configuration for CSV playback mode."""
    csv_path: str = ''
    playback_days: Optional[List[str]] = None
    n_days: int = 2
    warmup_days: int = 0


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    strategy_configs: List[RealtimeStrategyConfig] = field(default_factory=list)
    update_interval_sec: float = 5.0
    gex_enabled: bool = True
    discord_enabled: bool = True
    signal_jsonl_path: Optional[str] = None
    signal_jsonl_truncate_on_start: bool = False
    range_predictions_path: str = 'sandbox/range_predictions.csv'
    max_window_sec: int = 120

    def iter_enabled_strategy_configs(self) -> List[RealtimeStrategyConfig]:
        return [cfg for cfg in self.strategy_configs if cfg.enabled]

    @classmethod
    def default(cls) -> "EngineConfig":
        """Return config matching original default behavior."""
        return cls(
            db=DatabaseConfig(),
            strategy_configs=[
                RealtimeStrategyConfig(
                    kind="batch_breakout",
                    name="breakout",
                    params={
                        "strategy_name": "breakout",
                        "model_path": "bo_retest_model.json",
                        "level_cols": list(DEFAULT_LEVEL_COLS),
                        "threshold_pct": 0.0012,
                        "lookahead_bars": 12,
                        "pred_threshold": 0.4,
                    },
                ),
                RealtimeStrategyConfig(
                    kind="batch_reversion",
                    name="reversion",
                    params={
                        "strategy_name": "reversion",
                        "model_path": "reversion_model.json",
                        "level_cols": list(DEFAULT_LEVEL_COLS),
                        "threshold_pct": 0.0012,
                        "lookahead_bars": 12,
                        "pred_threshold": 0.4,
                    },
                ),
            ],
            update_interval_sec=5.0,
            gex_enabled=True,
            discord_enabled=True,
        )
