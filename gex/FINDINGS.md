# IV Surface Features for ES Trading — Empirical Findings

Research summary on what end-of-day options chain features predict for next-day
ES futures trading. Three actionable indicators emerged; two common-sense
applications were tested and **rejected** by the data.

All results below are out-of-sample, walk-forward (5 folds, expanding window,
100-day minimum train, 535-674 prediction days depending on test).

## TL;DR

| Application | Verdict | Key Number |
|-------------|---------|-----------|
| Position sizing via 1/pred_width | **Use it** | Sharpe 1.30 vs 0.69 (1/ATR), +75% P&L at matched std |
| 25-delta put skew as fade-direction filter | **Use it** | Long-fade edge +25.6 (steep skew) vs +10.3 (flat skew) |
| Intraday range consumption as vol forecast | **Use it (sizing only)** | r=+0.588 vs forward 30-min range; not directional |
| ATR-based stop sizing replacement | **Don't bother** | IV stops do not beat ATR stops |
| Predicted-width as strategy selector | **Don't bother** | Marginally better than ATR; no clean trend/reversion split |
| Put/call OI as directional regime filter | **Don't bother** | Frequency effect, no per-trade edge |

## Background

The starting point was the existing `nearby_gamma_score` feature — a single
EOD-derived ratio of local to relative gamma. We extracted ~22 IV surface
features from the historical SPX options chain (`gamma_shares_combined.parquet`)
and added them to the range predictor.

Walk-forward OOS results vs the baseline range predictor:

| Model | Width R² | Width corr |
|-------|---------|-----------|
| Baseline (80 features) | 0.018 | 0.249 |
| GEX-only (92 features) | 0.046 | 0.290 |
| **IV surface (102 features)** | **0.161** | **0.406** |
| GEX + IV surface (114 features) | 0.164 | 0.408 |

IV features account for nearly all the improvement. The strongest individual
IV features (correlation with realized range): `iv_term_slope` (-0.607),
`atm_iv` (+0.604), `iv_butterfly` (-0.453).

But improved range prediction does not automatically yield trading edge.
The fade-at-boundary approach we initially tried looked spectacular (98%
WR) but turned out to be a structural artifact — a random boundary at
roughly the right distance achieves 80% WR with the same trading rules
because of the "fade extreme, hold to close" effect. The real applications
of the prediction are elsewhere.

## Finding #1: Position Sizing — 1/pred_width replaces 1/ATR

A simple long-at-open / short-at-close trade with three sizing schemes,
each globally rescaled to match constant-sizing daily P&L std:

| Sizing | Mean | Sharpe | Total | MaxDD | Skew |
|--------|------|--------|-------|-------|------|
| Constant (baseline) | +2.04 | 0.74 | +1089 | -671 | +2.71 |
| 1/ATR_14 levered | +1.89 | 0.69 | +1013 | **-914** | -0.77 |
| **1/pred_width levered** | **+3.56** | **1.30** | **+1906** | **-573** | +1.94 |

At matched volatility, 1/pred_width sizing produces **+75% more cumulative P&L**
than constant or ATR sizing — and smaller drawdowns and positive skew.
Backward-looking ATR sizing actually performs *worse* than constant.

The mechanism: pred_width incorporates the next-day IV term structure and
OI positioning, so it correctly down-sizes on volatile days that ATR doesn't
recognize as volatile yet (and vice versa).

**Implementation**: `gex.iv_indicators.iv_position_size(pred_width, target_risk_pts=10.0)`

```python
from gex.iv_indicators import iv_position_size

# Replace: position_size = base_size / atr_14
# With:    position_size = base_size * iv_position_size(pred_width)
size_mult = iv_position_size(pred_width=42.5, target_risk_pts=10.0)
```

## Finding #2: 25-delta put skew is the directional fade filter

Bucket fade-at-boundary trades by 25-delta skew tercile and split by direction:

| Skew bucket | Long-fade avg | Long PF | Short-fade avg | Short PF |
|-------------|---------------|---------|----------------|----------|
| T1 (flat skew, complacent) | **+10.3** | 4.90 | +15.1 | 4.56 |
| T2 (mid) | +19.4 | 12.70 | +17.8 | 27.79 |
| **T3 (steep skew, peak fear)** | **+25.6** | 10.61 | +20.0 | 6.53 |

Long-fade edge is **2.5x larger in steep-skew vs flat-skew regimes**
(+25.6 vs +10.3 average P&L per trade). The pattern is monotonic across all
three terciles.

This matches conventional positioning logic: heavy put pricing = peak fear
= dips bounce. Flat put pricing = complacent market = rallies fade. The
data supports the textbook intuition cleanly via skew, but **not via P/C
OI ratio** (which we initially expected to work — see Rejected #2 below).

**Implementation**: `gex.iv_indicators.classify_skew_regime(iv_skew_25d, skew_history)`

```python
from gex.iv_indicators import classify_skew_regime

ctx = classify_skew_regime(
    iv_skew_25d=today_skew,
    skew_history=skew_lookback_60d,
)
# ctx.long_fade_bias  → 1.5 in high regime, 1.0 in mid, 0.6 in low
# ctx.short_fade_bias → 0.7 in high regime, 1.0 in mid, 1.2 in low
```

## Finding #3: Intraday range consumption forecasts forward vol

Sample current intraday range vs predicted daily width at 60/120/180/240
minutes into the session, measure forward 30-min range:

| Source | r vs fwd 30-min range | r vs fwd directional | r vs fwd two-sided |
|--------|----------------------|---------------------|---------------------|
| **IV consumption** (range / pred_width) | **+0.588** | +0.466 | +0.452 |
| ATR consumption (range / ATR_14) | +0.367 | +0.271 | +0.311 |

The IV-based ratio forecasts forward volatility **60% better than ATR-based**.
The correlation is **positive** — high consumption signals MORE forward vol
(volatility persistence, not exhaustion).

**Important caveat**: directional and two-sided components scale equally
(dir/range ratio ≈ 0.50 across all consumption buckets). High consumption
does NOT predict directional continuation or reversal. Naive directional
trades (continuation or fade) at high-consumption + range-extreme conditions
produced trivial edge (avg P&L $0.83-$2.44/trade).

**Use it as a sizing modulator only**: when consumption is high, expect
more vol → size down or tighten stops. When low, expect less vol → can
size up.

**Implementation**: `gex.iv_indicators.forecast_forward_vol(intraday_range, pred_width)`

```python
from gex.iv_indicators import forecast_forward_vol

forecast = forecast_forward_vol(
    intraday_range_so_far=22.0,
    pred_width=42.5,
)
# forecast.size_modulator → multiplier in [0.3, 1.5]
# forecast.regime         → 'low' | 'normal' | 'elevated' | 'extreme'
```

## Rejected: things that didn't work

### Rejected #1: Fade-at-boundary as an entry strategy

98-100% win rates and 50+ profit factor under generous parameters look
amazing but are entirely structural. A random boundary placed at roughly
the right distance from prev close achieves 80% WR with the same rules,
because once a boundary triggers, your entry is near the day's extreme by
definition, and "the close is closer to the center than the extreme" is
nearly tautological.

Per-trade edge of the actual model over random boundaries is ~2 pts/trade —
real but tiny relative to the 16 pts/trade structural baseline.

### Rejected #2: P/C OI ratio as directional regime filter

The intuition "heavy put OI = bullish reversal" shows up in the close-open
data (high P/C OI days drift +17 pts on average vs -17 for low-P/C days),
but this turned out to be a **frequency effect**, not a per-trade edge:
on bullish-drift days you get more long-fade signals, but each long-fade
trade has comparable (or worse) edge per trade than on neutral days.

| P/C OI bucket | long-fade avg | short-fade avg | (long-short) |
|---------------|---------------|----------------|--------------|
| T1 (low) | +21.2 | +18.3 | +3.0 |
| T3 (high) | +13.8 | +16.9 | **-3.2** |

Use `iv_skew_25d` for directional filtering instead.

### Rejected #3: IV-based stop calibration

ATR-scaled stops, IV-width-scaled stops, and fixed-pt stops all produce
near-break-even results on a simple ORB-style breakout strategy. IV-based
stops do not improve over ATR-based stops:

| Config | Avg P&L | PF |
|--------|---------|-----|
| Fixed 8pt | +0.2 | 1.05 |
| ATR 25% | -0.1 | 0.99 |
| **IV 25%** | **-0.3** | **0.95** |

Stop sizing isn't where the IV signal adds value.

### Rejected #4: Predicted-width as a trend-vs-reversion strategy selector

Bucketing days by predicted width tercile does separate days by realized
range (37 / 49 / 76 pts), but does NOT cleanly separate trend-favoring
from reversion-favoring days. IV bucketing is marginally better than ATR
(differential 0.27 vs 0.13 in trend-vs-reversion ratio across narrow→wide)
but neither is dramatic enough to use as a strategy switch.

## How to replicate

The full experiment scripts are under `sandbox/`:

```bash
source ~/ml-venv/bin/activate

# Build IV features and run all 5 application tests
python sandbox/range_predictor_applications.py

# Follow-up tests on position sizing leverage, directional filters, and
# range exhaustion decomposition
python sandbox/range_predictor_followup.py

# Daily OHLCV-level analysis (model comparison, fade-at-boundary, GEX regime)
python sandbox/analyze_iv_surface_model.py

# 1-min intraday simulation (fade-at-boundary with realistic fills)
python sandbox/analyze_iv_intraday.py

# Stress-tested intraday simulation (vs random boundaries)
python sandbox/analyze_iv_intraday_v2.py
```

The IV feature extraction and integration:
- `gex/iv_surface_features.py` — extracts ~22 features from EOD chain
- `strategies/range_predictor/features.py` — passes them through to the
  range predictor's daily feature matrix
- `gex/iv_indicators.py` — production-ready indicator functions backed by
  these findings

## Caveats and open questions

1. **Sample size**: 535-674 trading days (~2.5 years). Patterns are
   statistically significant but a longer window would be more convincing,
   especially for skew regime stability across vol regimes.

2. **Survivorship of edge**: the IV surface features come from a single
   EOD snapshot per day. Intraday IV moves (vanna flows, real-time skew
   shifts) aren't captured. The intraday options collector
   (`gex/intraday_collector.py`) is in place to remedy this; once 2-3
   months of data accumulate the indicators here can be re-validated and
   refined with intraday IV updates.

3. **Strategy coupling**: all the per-trade edge numbers above come from
   the V2 fade-at-boundary simulation framework. The skew regime filter
   should be re-tested on the actual production breakout/reversion
   strategies before sizing up real capital. The position sizing finding
   should generalize since it operates on raw daily P&L.

4. **Regime drift**: 25-delta skew and IV term slope have absolute levels
   that drift over multi-year windows (post-COVID, post-2022). Tercile
   bucketing on a rolling lookback window (e.g. 60-120 days) is more
   robust than fixed thresholds.
