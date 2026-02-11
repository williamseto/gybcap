"""Configuration dataclasses for the real-time engine."""

from dataclasses import dataclass, field
from typing import List, Optional


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
class StrategySlotConfig:
    """Configuration for a single strategy slot."""
    strategy_type: str                          # 'breakout' or 'reversion'
    model_path: Optional[str] = None
    level_cols: Optional[List[str]] = None
    threshold_pct: float = 0.0012
    lookahead_bars: int = 12
    pred_threshold: float = 0.4
    enabled: bool = True

    def __post_init__(self):
        if self.level_cols is None:
            self.level_cols = [
                'prev_high', 'prev_low', 'vwap',
                'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi',
            ]


@dataclass
class ReversalPredictorSlotConfig:
    """Configuration for the Phase 3 reversal predictor strategy."""
    model_dir: str = 'models/reversal_phase3'
    pred_threshold: float = 0.50
    proximity_pts: float = 5.0
    historical_csv_path: Optional[str] = None
    warmup_days: int = 60
    enabled: bool = True


@dataclass
class EngineConfig:
    """Top-level engine configuration."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    strategies: List[StrategySlotConfig] = field(default_factory=list)
    reversal_predictor: Optional[ReversalPredictorSlotConfig] = None
    update_interval_sec: float = 5.0
    gex_enabled: bool = True
    discord_enabled: bool = True
    range_predictions_path: str = 'sandbox/range_predictions.csv'
    max_window_sec: int = 120

    @classmethod
    def default(cls) -> "EngineConfig":
        """Return config matching the original hardcoded behavior."""
        return cls(
            db=DatabaseConfig(),
            strategies=[
                StrategySlotConfig(
                    strategy_type='breakout',
                    model_path='bo_retest_model.json',
                    level_cols=[
                        'prev_high', 'prev_low', 'vwap',
                        'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi',
                    ],
                    threshold_pct=0.0012,
                    lookahead_bars=12,
                    pred_threshold=0.4,
                ),
                StrategySlotConfig(
                    strategy_type='reversion',
                    model_path='reversion_model.json',
                    level_cols=[
                        'prev_high', 'prev_low', 'vwap',
                        'ovn_lo', 'ovn_hi', 'ib_lo', 'ib_hi',
                    ],
                    threshold_pct=0.0012,
                    lookahead_bars=12,
                    pred_threshold=0.4,
                ),
            ],
            update_interval_sec=5.0,
            gex_enabled=True,
            discord_enabled=True,
        )
