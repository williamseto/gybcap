# Graveyard Experiment Summary

Last updated: 2026-04-21

This file replaces archived graveyard artifacts/scripts.  
Detailed file-level recovery is available via git history.

## Scope
- Primary focus was intraday ES reversal prediction + execution policy.
- Secondary focus included orderflow/footprint deep models and trend-pullback sidecar ideas.
- Work emphasized causal feature contracts, runtime parity, SQL playback OOS checks, and live-feasible policy behavior.

## What Worked
- Structural near-level labeling (major-move framing) improved stability versus stop/target-coupled labeling.
- Causal feature fixes (notably `rth_hi/rth_lo`, `ovn_hi/ovn_lo`, rolling stats) improved training/runtime contract consistency.
- Frontier-style routing + budget/cooldown controls improved tradeability versus naive per-bar acceptance.
- Blend-style routing (`policy_prob` + selective `q_twohead`) improved recent SQL behavior versus single strict thresholding.
- Sequential execution handling (single-position contract with flatten/ignore controls) is more realistic than independent-trade evaluation.

## What Did Not Hold Up
- Most stacked post-filters (MFE gates, heavy threshold stacking, online threshold hacks) were unstable across slices.
- Many dynamic-budget/override relaxations increased activity but degraded PF or increased noise.
- Orderflow/footprint deep paths did not show robust standalone alpha under tested contracts.
- Trend-pullback sidecar did not demonstrate enough robustness as always-on additive logic.
- Independent-trade evaluator often overstated results relative to live-feasible sequential execution.

## Key Tradeoffs Observed
- Better PF commonly came with worse no-TP day rate or lower coverage.
- More late-day coverage often required relaxed gates and increased false positives.
- Low-lane micro-execution tweaks could improve PF but often hurt no-TP and/or consistency.
- Full-history vs recent SQL OOS behavior differed materially; contract mismatch can dominate apparent gains.

## Recent Live-Oriented Snapshot
- Current live-selected SQL gate on recent 3-week window (`2026-03-09`..`2026-03-27`):
  - trades/day: `~7.0`
  - PF: `~1.29`
  - zero-trade-day rate: `0.0`
  - mean pnl/day: positive
- This is tradable but still below the historical PF target explored in research.

## Archived Families (Now Removed Here)
- Legacy configs under `graveyard/configs/...`
- Legacy sweep/ablation scripts under `graveyard/sandbox/...`
- Legacy result json/csv/parquet dumps under `graveyard/sandbox/results_legacy_...`
- Early monolithic/utility implementations and orphaned tests/scripts

---

## Range Predictor Research (Apr 2026)

### Newsletter Reverse-Engineering

Scripts: `sandbox/reverse_engineer_newsletter.py`, `newsletter_formula.py`,
`newsletter_dynamics.py`, `newsletter_linear_oos.py`, `newsletter_xgb_with_ar.py`

- Newsletter width is a simple GARCH-style AR(1) update:
  `nl_width_t ≈ α + 0.91·nl_width_{t-1} - 480·prev_ret + 218·|prev_ret|`
- Linear formula: R²=0.92, MAE=4.5pts (walk-forward OOS on ~490 days)
- 60-feature XGBoost replica: R²=0.25, MAE=15.8pts — trees can't approximate a
  linear recursive process; they step-function it and add noise
- Newsletter midpoint ≈ prev_close (no model beats the 0-shift baseline)
- Newsletter ≈ ATR×0.7 in information content (rank corr 0.85)
- Production formula: `strategies/range_predictor/newsletter/formula.py`

### Range Prediction Feature Engineering

Scripts: `sandbox/test_volume_features.py`, `sandbox/analyze_rth_model.py`

- Width/center decomposition (width=volatility, center=direction) outperforms
  predicting high/low independently
- Added volatility regime features (ATR percentile, momentum), GARCH proxy
  (squared returns), range dynamics (range/ATR ratio, asymmetry), bar structure
- Fixed causality bug: blanket `shift(1)` was double-shifting already-lagged
  features; replaced with selective shifting
- RTH model adds value for intraday: tighter width, better centering for
  overnight-gap days

### Quantile & Conditional Range

Scripts: `sandbox/quantile_ranges.py`, `sandbox/intraday_conditional_range.py`,
`sandbox/conditional_trading_analysis.py`

- XGB quantile regression poorly calibrated (Q10 acts as Q36, Q90 as Q82),
  25% monotonicity violations — daily features can't separate vol quantiles
- Intraday conditional model (30-min checkpoints) is where real value lives:
  after 30-60min of observed vol, range estimates improve dramatically
- Production module: `strategies/range_predictor/conditional.py`

### Range Trading Impact

Scripts: `sandbox/range_signal_analysis.py`, `range_signal_analysis_v2.py`,
`range_trading_impact.py`, `range_trading_impact_v2.py`

- Proximity fade (enter when price within ~10pts of predicted boundary):
  475 signals, 60% WR, +5.8 PnL with 30-bar hold
- v1 backtest was broken (85% of trades exited on entry bar due to simulator
  starting on boundary-breaking bar with median 20pt overshoot vs 5pt stop)
- v2 fixed this with proper entry timing and retreat/proximity trigger logic
- Breakout signal (close < pred_low → next day up) concentrated in 2024-2025
  bull; not robust

### Dalton Day-Type & Trend Score

Scripts: `sandbox/range_dalton_ablation.py`, `range_dalton_dynamic.py`,
`sandbox/range_score_robustness.py`

- Fade edge comes from range prediction, not from implicitly identifying balance days
- Early trend detection: 30-min heuristic gives ~70% accuracy for EOD day type;
  diminishing returns after 60 min
- Rolling trend score is robust: monotonic PnL gradient across score buckets
  holds across bootstrap resamples and time sub-periods

### Trade Entry Filtering (Apr 2026)

Scripts: `sandbox/train_ptrend.py`, `train_ptrend_v2.py`, `train_ptrend_v3.py`

- Heuristic trend score ≤ 1.5 beats ALL ML entry filters (74% WR, +9.7 PnL)
- Walk-forward threshold selection: ~1.0 (more restrictive than initial =3)
- Stable across all three sub-periods; threshold < 1.5 best in every fold
- Models tested: XGB (stumps/shallow/standard), RF, Logistic, SVM RBF, MLP
- Narrow range is most learnable target (AUC 0.85) but doesn't help fade filtering

### Trade Management & Exit Research (Apr 2026)

Scripts: `sandbox/train_trade_mgmt.py`, `train_trade_mgmt_v2.py`,
`sandbox/train_runner_mgmt.py`, `sandbox/train_small_winner.py`

- **Bar-level exit model**: AUC 0.58, no improvement over 30-bar fixed hold
- **Structural models**: triage AUC 0.80 (bar 10), retest AUC 0.87, regime AUC 0.73
  — combined gives same PnL in 30% less time, but marginal vs complexity
- **Runners decline after bar 30**: incremental PnL negative at every checkpoint
  (45, 60, 90, 120 bars)
- **No heuristic exit rule beats bar 30**: trailing stops, profit targets,
  range-consumed, trend score exits, time limits all tested
- **Small winners (0-5pt at bar 30)**: genuinely indistinguishable — runners and
  reversals identical on all features
- **Only 15pt hard stop adds value** (+0.6 PnL)

### Final Recommended System

```
Range predictor (width/center) → heuristic trend score ≤ 1.5
  → proximity fade entry → 15pt hard stop → 30-bar fixed hold
```

---

## Recovery
- Restore any removed artifact from git:
  - `git log --all --full-history -- graveyard/<path>`
  - `git log --all --full-history -- sandbox/<filename>`
  - `git checkout <commit> -- <path>`
- For higher-level context, also see:
  - `.codex/MEMORY.md`
  - `.codex/RESEARCH_SUMMARY_2026-03-10.md`
  - `strategies/range_predictor/TRADE_FILTERING_RESEARCH.md`

