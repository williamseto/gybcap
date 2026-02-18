"""Configuration for swing regime classification."""
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent.parent


class MacroRegime(IntEnum):
    BEAR = 0
    BALANCE = 1
    BULL = 2


class MicroRegime(IntEnum):
    DOWN = 0
    BALANCE = 1
    UP = 2


@dataclass
class InstrumentConfig:
    symbol: str
    path: Path
    schema: str  # 'combined', 'mnt_standard'
    session_start_hour: int = 18  # 6 PM ET
    tick_size: float = 0.25


@dataclass
class SwingConfig:
    primary: InstrumentConfig = field(default_factory=lambda: INSTRUMENTS["ES"])
    correlation_instruments: List[str] = field(default_factory=lambda: ["NQ", "ZN"])
    swing_lookback: int = 10  # bars each side for swing detection
    micro_threshold_pct: float = 0.003  # 0.3% dead zone for micro labels
    hmm_n_states: int = 3
    n_folds: int = 5
    min_train_days: int = 500
    corr_windows: List[int] = field(default_factory=lambda: [10, 20, 60])
    near_level_pts: float = 4.0  # points proximity for level features
    detect_threshold: float = 0.05   # zigzag reversal detection (5%)
    bull_threshold: float = 0.10     # min rally for BULL classification (10%)
    bear_threshold: float = 0.07     # min drawdown for BEAR classification (7%)


INSTRUMENTS: Dict[str, InstrumentConfig] = {
    "ES": InstrumentConfig(
        symbol="ES",
        path=ROOT / "raw_data" / "es_min_combined.csv",
        schema="combined",
        tick_size=0.25,
    ),
    "NQ": InstrumentConfig(
        symbol="NQ",
        path=Path("/mnt/d/data/nq_min_2011.csv"),
        schema="mnt_standard",
        tick_size=0.25,
    ),
    "ZN": InstrumentConfig(
        symbol="ZN",
        path=Path("/mnt/d/data/zn_min_2011.csv"),
        schema="mnt_standard",
        tick_size=1 / 64,  # 32nds
    ),
}
