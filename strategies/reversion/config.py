"""Default configuration for reversion strategy."""

from dataclasses import dataclass
from typing import List


@dataclass
class ReversionConfig:
    """Configuration for reversion strategy."""

    # Price levels to trade
    level_cols: List[str] = None

    # Entry parameters
    threshold_pct: float = 0.0012  # Distance threshold for retest entry
    lookahead_bars: int = 10  # Bars to look ahead for retest

    # Risk parameters
    stop_buffer_pct: float = 0.0025  # Stop buffer as % of price
    reward_risk_ratio: float = 2.0  # Take profit as multiple of risk
    max_risk_points: float = 10.0  # Maximum risk in price points

    # Timeframe
    timeframe: str = "5min"  # Default timeframe for reversion strategy

    # Trade size
    fixed_size: float = 1.0

    def __post_init__(self):
        if self.level_cols is None:
            # Only use levels with positive out-of-sample edge:
            # vwap (+0.30 avg), rth_lo (+0.30 avg), ovn_lo (+0.02 avg)
            # Excluded: prev_high (-1.02), prev_low (-2.50), ovn_hi (-0.78), rth_hi (-0.21)
            self.level_cols = [
                'vwap', 'rth_lo', 'ovn_lo'
            ]

    @classmethod
    def default(cls) -> "ReversionConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> "ReversionConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'level_cols': self.level_cols,
            'threshold_pct': self.threshold_pct,
            'lookahead_bars': self.lookahead_bars,
            'stop_buffer_pct': self.stop_buffer_pct,
            'reward_risk_ratio': self.reward_risk_ratio,
            'max_risk_points': self.max_risk_points,
            'timeframe': self.timeframe,
            'fixed_size': self.fixed_size,
        }
