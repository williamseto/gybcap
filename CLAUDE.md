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

### Experimentation Script
Run feature analysis:
```bash
source ~/ml-venv/bin/activate
python sandbox/test_volume_features.py --data raw_data/es_min_3y_clean_td_gamma.csv
```

This compares:
1. Price-only baseline
2. Volume features only
3. Quality features only
4. All features combined

### Success Criteria
| Metric | Baseline | Target |
|--------|----------|--------|
| Feature |corr| | < 0.04 | > 0.05 (at least 3 features) |
| OOS Win Rate | 35% | > 40% |
| Overfitting Gap | 28% | < 15% |
