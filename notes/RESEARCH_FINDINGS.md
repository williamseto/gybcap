# ES Intraday/Daily Edge Research — Consolidated Findings

Single source of truth for the range-prediction / volume / options research arc
(early–mid 2026). Distilled from ~20 exploratory scripts that have since been
deleted; this document is the durable record. Read this before re-running any
"does X predict direction" experiment — most have been tested and closed.

Data: `raw_data/es_min_extended.csv` (1-min ES, 2009→2025, ~4,300 RTH days) for
robustness; `raw_data/es_min_3y_clean_td_gamma.csv` (2023–2025) + options chain
(`gex/gamma_shares_combined.parquet`) for options-based work.

---

## TL;DR

| Claim | Verdict |
|-------|---------|
| Volatility / daily range is forecastable | ✅ Robust (R²≈0.40–0.49 over 17yr, simple TA) |
| Vol-targeted position sizing improves risk-adjusted return | ✅ Robust (+34–65% Sharpe vs constant) |
| Daily/intraday **direction** is predictable from our features | ❌ No (IC≈0; every test failed or was sub-cost) |
| Price **levels** (swing, options, volume-profile) produce tradeable reactions | ❌ No (placebo-indistinguishable at every scale) |

**Bottom line:** the only durable edge is *magnitude* (volatility), monetized
through position sizing. Direction at these timescales is, empirically, not
predictable from price/volume/EOD-options features. This matches market-
efficiency priors (returns ≈ martingale; variance is autocorrelated).

---

## Surviving findings

### 1. Volatility / range is forecastable (robust, 17yr)

Out-of-sample walk-forward R² for next-day range width (`width_pct`):

| Predictor | OOS R² (17yr) |
|-----------|---------------|
| Naive "width = yesterday's range" | +0.40 |
| Bollinger-only (3 feats) | +0.30–0.37 |
| 1-day volume-profile shape (5 feats) | +0.35 |
| **Multi-scale VP (1-day + week + month)** | **+0.49** |
| Everything (TA + multi-scale VP) | +0.48 |

- Vol predictability is **regime-dependent**: on the calm 2023–2025 slice,
  baseline width R² was only **0.018** (little variance to explain). On the
  full regime-diverse 17yr it's ~0.40. The 3yr number badly understated how
  forecastable vol is.
- Best parsimonious feature families: lagged range / ATR persistence, Bollinger
  width, and **multi-scale volume-profile value-area widths** (week & month
  composites add real signal; quarter/year overfit and hurt OOS).
- **`price_vs_va`** (position relative to the composite value area) is a strong,
  *asymmetric* vol-regime feature: price below the monthly value area precedes
  next-day ranges ~2× wider than price above it (downside = stress = high vol).
  Corr with next-day width: −0.39 (month) to −0.55 (year).
- Volume itself (raw volume z) is useless for width (R²≈−0.25 alone); it's the
  *profile shape* (value-area width, dispersion/entropy, position vs value) that
  carries the signal.

Reusable infra: `strategies/range_predictor/` (width/center model),
`gex/iv_surface_features.py` (options-chain IV features). Note IV surface
features lifted 3yr width R² 0.018→0.16, but on a regime-diverse sample plain TA
already reaches ~0.40, so IV's *marginal* contribution to vol forecasting is
much smaller than the 3yr slice implied.

### 2. Vol-targeted position sizing (the one monetizable edge)

Sizing a base strategy by `target_vol / forecast_vol` (capped ~0.5×–2.5×),
std-matched to constant sizing, over 17yr:

| Base strategy | Constant Sharpe | 1/ATR-sized Sharpe | Lift |
|---------------|-----------------|--------------------|------|
| Buy-and-hold (open→close) | 0.39 | 0.53 | +34% |
| Trend-following (long > SMA200) | 0.23 | 0.37 | +65% |

- Robust across eras; larger relative lift on directional bases.
- Mechanism is structural (variance control / vol-drag reduction), **not** a
  hidden directional bet — confirmed because the directional `center` prediction
  from the *same* model has zero skill (see dead-ends). Vol-targeting works even
  with zero directional alpha.
- Forecaster quality (corr with next-day |return|): ATR_14 +0.24, range_14
  +0.23, EWMA-94 +0.19. ATR/realized-range are the simplest and best.
- Trade-off: vol-targeting flips return skew from + to slightly − (gives up
  upside tail to control downside). Sizing method matters — naïve
  1/pred_width without careful std-matching can underperform.

---

## Dead ends (tested and closed — do not re-run without new data)

| Idea | Result | Why it failed |
|------|--------|---------------|
| **Unconditional center / direction prediction** | IC≈0.01–0.05; long-short tercile CI crosses zero | Daily direction ≈ martingale |
| **Fade-at-boundary (reversion)** | Looked great (98% WR) → structural artifact | Random boundaries match it; "fade extreme, hold to close" is near-tautological |
| **Envelope mean-reversion to prev_close** | Looked superb (Sharpe 5+) → **look-ahead bug** | `aggregate_to_daily` evening-index leaked same-day close into prev_close (see gotcha memory). Clean version: Sharpe −1 |
| **Volume climax / exhaustion reversal triggers** | Failed permutation test | Shuffled volume features beat the real ones; edge was 2023–2025 curve-fit |
| **Level reactions** — swing (prev_day_low), options (pred boundary), VP (POC/VAH/VAL), multi-scale composites | All placebo-indistinguishable | No level, at any scale or construction, reacts better than a random level at the same distance |
| **LVN/HVN positioning** | No vol-expansion (corr≈0), no directional continuation | "Thin air moves fast" does not hold in ES |
| **Confirmation / momentum breakout** | Gross Sharpe +0.45, **net −0.65 after 1pt cost** | Real intraday continuation exists but is ~arbitraged to the cost of capture. Break-even ≈0.4pt round-trip. Regime-dependent: negative 2009–2023, positive only in 2024–2026 (recency, not deployable) |
| **IV surface as a *directional* signal** | IC bump +0.037 (3yr), not significant | Options positioning leaks a faint directional whisper but far too small at EOD/daily resolution |

---

## Methodology lessons (cost us real time — keep)

1. **Look-ahead via `aggregate_to_daily`**: it indexes days by the first
   (evening) bar's date, so ES days are dated one day early and `daily["close"]`
   is the full-session close. Joining it to RTH-dated intraday data and using
   `.shift(1)` for "prev_close" silently yields the *same* day's close — an
   invisible, catastrophic look-ahead. Always build an RTH-only daily frame
   keyed by the true RTH date. (See memory `gotcha-aggregate-daily-lookahead`.)
2. **3-year backtests overstate edges.** Multiple "edges" (volume climax/
   exhaustion, level confluence, ~57% WR fades) on 2023–2025 evaporated or went
   negative on the full 17yr. The calm recent regime flatters reversion and
   understates vol predictability.
3. **Require all three before believing a signal:** train/test split,
   permutation test (shuffle the feature, edge should die), and placebo (random
   levels/directions). Symmetric long+short both winning, cross-era stability,
   and implausible Sharpe (>3) are red flags for look-ahead, not for a great
   strategy.
4. **Costs decide.** Several directional signals had positive *gross* edge that
   realistic transaction costs fully consumed. Always sweep cost 0→2pt.
5. **Magnitude ≫ direction in tractability.** Variance is autocorrelated and
   forecastable; returns are not. Build strategy around the predictable thing.

---

## Where to go next (if anything)

- **Productionize vol-targeted sizing** with the parsimonious vol forecast
  (ATR/range persistence + Bollinger width + week/month VP value-area width +
  `price_vs_va`). This is the deployable outcome.
- **Directional, if pursued at all:** only two avenues remain unrefuted —
  (a) using the (sub-cost gross-positive) momentum signal as a *tilt on an
  already-held position* (marginal turnover cost, not full round-trip), and
  (b) *live* options/forced-flow at intraday resolution (the intraday options
  collector), which is the only genuinely new information source. Everything
  derivable from historical price/volume has been tested and is closed.
