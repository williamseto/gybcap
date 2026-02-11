# Project Instructions

## Python Environment
Always activate the virtual environment before running Python commands:
```bash
source ~/ml-venv/bin/activate
```

Or use the full path: `~/ml-venv/bin/python`

## Project Overview
ES futures trading strategy development with ML-based trade filtering.

## Key Components

### strategies/ Package (Refactored Feb 2026)
Modular trading pipeline replacing monolithic `sandbox/test_bo_retest.py`:

```bash
# Train breakout strategy (5min default)
python -m strategies train --strategy breakout

# Train reversion strategy (15min default)
python -m strategies train --strategy reversion

# With Dalton day type features
python -m strategies train --strategy breakout --dalton

# Backtest
python -m strategies backtest --strategy breakout --model breakout_model.json
```

Key modules:
- `strategies/core/trade_simulator.py` - Numba-vectorized trade simulation
- `strategies/features/` - Plugin system for features (price_levels, gamma, dalton)
- `strategies/breakout/` - Breakout retest strategy
- `strategies/reversion/` - Reversion strategy
- `strategies/training/` - XGBoost trainer
- `strategies/realtime/` - Modular real-time trading engine (see below)

### strategies/realtime/ -- Real-Time Engine

Plug-and-play real-time trading engine replacing the monolithic `util/test_client.py`.

```bash
# Run with default config (breakout + reversion, GEX, Discord)
python -m strategies.realtime

# Disable GEX provider
python -m strategies.realtime --no-gex

# Log-only mode (no Discord alerts)
python -m strategies.realtime --no-discord

# Run only specific strategies
python -m strategies.realtime --strategies breakout
python -m strategies.realtime --strategies reversion
python -m strategies.realtime --strategies breakout reversion
```

Key modules:
- `engine.py` - Slim orchestrator (data -> bars -> strategies -> signals)
- `protocol.py` - `RealtimeStrategy` protocol + `BatchStrategyAdapter`
- `config.py` - `EngineConfig`, `StrategySlotConfig`, `DatabaseConfig` dataclasses
- `data_source.py` - `DataSource` protocol + `MySQLSource`
- `bar_aggregator.py` - Tick-to-OHLC bar aggregation
- `level_provider.py` - `DayPriceLevelProvider` (intraday VWAP, IB, OVN, etc.)
- `signal_handler.py` - `SignalHandler` protocol + Discord/Logging handlers
- `runner.py` - CLI entry point

**Adding custom strategies:**
```python
from strategies.realtime import RealtimeEngine, EngineConfig, RealtimeSignal

class MyStrategy:
    @property
    def name(self) -> str:
        return "my_strategy"

    def process(self, bars: pd.DataFrame) -> list[RealtimeSignal]:
        # bars = 1-min OHLCV, datetime-indexed, LA timezone
        ...

config = EngineConfig.default()
engine = RealtimeEngine(config)
engine.register_strategy(MyStrategy())
engine.run()
```

### graveyard/
Archived legacy code (replaced by `strategies/realtime/`):
- `graveyard/util/test_client.py` - Old monolithic real-time client
- `graveyard/util/strategy_util.py` - Old duplicated strategy implementations
- `graveyard/util/client_util.py` - Old data fetch utilities

### Data
- Primary: `raw_data/es_min_3y_clean_td_gamma.csv` (1-min ES futures with gamma)
- 775 trading days of data

### Other Key Directories
- `gex/` - Gamma exposure utilities (IVMapper)
- `vp/` - Volume profile and Dalton day classifier

## Testing
```bash
pytest tests/ -v
```
56 tests covering trade simulation, strategies, and feature providers.

---

## Reversion Strategy Analysis (Feb 2026)

### Problem: Massive Overfitting
The original reversion strategy with price-level features showed:
- **In-sample**: 63% WR
- **Out-of-sample**: 35% WR (28% gap = massive overfitting)
- Feature correlations with y_succ: all < 0.04

**Root Cause**: Price-level z-score features (vwap_z, ovn_lo_z, etc.) describe
"where" price is relative to levels, not "how" the rejection happened.

### Level-Filtering Analysis
Analysis of individual levels showed mixed edge:
- **Positive edge**: `vwap`, `rth_lo`, `ovn_lo` (combined PnL > 0)
- **Negative edge**: `prev_high`, `prev_low`, `rth_hi`, `ovn_hi`

The reversion strategy now defaults to only trading the positive-edge levels.

### New Feature Engineering Approach

#### Volume Microstructure Features (`strategies/features/volume_microstructure.py`)
Hypothesis: Features describing "how" price interacted with levels are more predictive.

Features include:
- **Mini volume profiles**: Volume concentration at level (vol_at_level_pct, vol_concentration_z)
- **Bid-ask delta**: Order flow imbalance at level (delta_at_level, bid_ask_ratio_at_level)
- **Volume dynamics**: Volume behavior approaching rejection (vol_into_level, delta_flip_at_level)
- **Profile shape**: POC distance, VA containment, skewness

#### Reversion Quality Features (`strategies/features/reversion_quality.py`)
Bar-level rejection characteristics:
- `wick_to_body_ratio`: Larger = stronger rejection
- `rejection_penetration`: How far past level before reversing
- `close_position`: Where close is within bar range
- `level_touch_count`: First touch vs repeated tests

### Walk-Forward Validation

Random CV causes look-ahead bias in time series. Use walk-forward instead:

```bash
# Recommended: Walk-forward validation with volume features
python -m strategies train --strategy reversion --walk-forward --volume-features --quality-features
```

The trainer now supports:
- `walk_forward_cv()`: Time-respecting cross-validation
- `analyze_feature_correlations()`: Warns if features have |corr| < 0.05

### Hyperparameter Recommendations
From CV analysis:
```python
{
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_weight': 5,
    'scale_pos_weight': 3.0,
}
```

### V3 Causal Zone Prediction (Feb 2026)

Previous V1/V2 multi-scale models used 1-sec data and were found to be
lookahead-biased (see `strategies/reversal/NONCAUSAL_NOTES.md`). V3 uses
1-min bars only with zone-based labeling.

**Key idea**: Predict P(pre-reversal zone) at every bar near a price level,
learning approach patterns rather than reacting to a precise trigger bar.

```bash
source ~/ml-venv/bin/activate

# Full experiment — XGBoost + V3 neural model
PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py

# XGBoost only (~5 min)
PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py --xgboost-only

# V3 neural only
PYTHONPATH=/home/william/gybcap python -u sandbox/train_causal.py --v3-only
```

Key modules:
- `strategies/labeling/reversal_zones.py` — `LevelAnchoredZoneLabeler`
- `strategies/features/zone_features.py` — Scalar + VP heatmap extraction
- `strategies/reversal/causal_model.py` — V3 model (VPBranchCNN + TCN + scalars)
- `strategies/reversal/causal_trainer.py` — Walk-forward XGBoost + V3 trainer

### Success Criteria
| Metric | Baseline | Target |
|--------|----------|--------|
| Zone detection ROC-AUC | 0.50 | > 0.60 |
| Zone precision at P>0.5 | — | > 40% |
| Trading WR (P>threshold) | base rate | > 45% |
| E[PnL/trade] | ~-0.40pt | > 0.00pt |
