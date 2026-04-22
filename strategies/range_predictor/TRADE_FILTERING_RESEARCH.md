# Range Predictor: Trade Filtering & Management Research

Date: 2026-04-20

## Context

This document summarizes research into using the range predictor's predicted
boundaries for proximity-fade trading (enter when price approaches predicted
boundary within ~10 pts) and what entry/exit logic works best.

Base dataset: ~475 proximity fade signals over 775 trading days.
Base performance: 60% WR, +5.8 PnL (30-bar fixed hold).

All results use walk-forward OOS (5-fold, min_train_days=100).

---

## 1. Entry Filtering: Heuristic vs ML

### Day-Type Filtering (train_ptrend_v2, train_ptrend_v3)

Tested whether ML models can filter which days are suitable for proximity fading,
using features like OTF ratio, VA width, POC position, OR breakout, move from open.

**Models tested**: XGBoost (stumps/shallow/standard), Random Forest, Logistic
Regression, SVM RBF, MLP (small 64,32 and tiny 16).

**Key findings**:
- Narrow range is most learnable target (AUC 0.85), but doesn't translate to
  better fade filtering
- No ML model beats the simple heuristic trend score
- Walk-forward heuristic threshold selection picks ~1.0 (more restrictive than
  the initial threshold=3), yielding 270 signals, 74% WR, +9.7 PnL [+7.3, +12.6]
- Heuristic is stable: threshold < 1.5 is best in ALL three sub-periods
- Non-tree models (logistic, SVM, MLP) do NOT outperform tree-based models

**Recommendation**: Use heuristic trend score threshold ≤ 1.5 for entry filtering.

---

## 2. Exit Timing: Fixed Hold vs ML Exits

### Bar-Level Exit Model (train_trade_mgmt)

Trained P(should_hold) at each post-entry bar using 17 features (momentum,
volatility, trend score, time).

- AUC: 0.58 (barely above random)
- No improvement over fixed 30-bar hold
- Simple rule baselines (take profit, trailing stop) all underperform fixed hold
- Oracle ceiling: +14.3 PnL vs +5.8 fixed — there IS signal, but it's too noisy
  for bar-level modeling

### Structural Exit Models (train_trade_mgmt_v2)

Three targeted models instead of generic bar-level:

1. **Early Triage (bar 10)**: Classify trade quality from first 10 bars
   - AUC: 0.80
   - P>0.6 threshold: 97% WR, +22.7 PnL (small sample)
   - Top features: current_pnl, mfe, pnl_slope

2. **Boundary Retest Prediction**: P(price retests the boundary level)
   - AUC: 0.87
   - Top feature: dist_to_boundary (very intuitive — close to boundary = likely retest)

3. **Regime Change**: P(calm → trend transition)
   - AUC: 0.73
   - Moderate signal but doesn't add value over simpler approaches

4. **Combined**: Triage → Retest monitoring → Regime cut
   - +7.4 PnL in 21 bars vs +7.1 in 30 bars (same PnL, 30% less time exposure)
   - Marginal improvement doesn't justify model complexity

---

## 3. Runner Management (train_runner_mgmt)

### Do Runners Continue After Bar 30?

Extracted full-day trade paths (no max_hold cap) for profitable trades.

**Key finding: Runners DECLINE after bar 30.** Incremental PnL is negative at
every checkpoint (bars 45, 60, 90, 120). The average trade peaks at ~30 minutes.

### Heuristic Exit Rules Tested

- Trailing stops (3pt, 5pt, 8pt)
- Profit targets (5pt, 8pt, 10pt)
- Range-consumed threshold exits
- Trend score deterioration exits
- Time limits (45, 60, 90 bars)
- Combined rules (trail + time limit)

**No heuristic rule beats exit at bar 30.**

### Big Winners (PnL > 10 at bar 30)

Trail 5pt adds +0.4 (trivial improvement). These trades have already captured
most of their move.

### ML Continuation Prediction

- "Continues to profit" target: AUC 0.50 (useless)
- "MFE continues" target: AUC 0.73 (learnable), but doesn't translate to PnL
  improvement because MFE continuation doesn't mean the exit is wrong

---

## 4. Small Winner Deep Dive (train_small_winner)

### Are Small Winners (0-5pt at bar 30) Distinguishable?

Built enriched feature set across 7 categories:
- Momentum, path shape, volume structure, boundary integrity, VP shape,
  range context, session context

**Finding: Small winners are genuinely indistinguishable at bar 30.**

- Runners and reversals look identical on all measured features
- ML "runner" model (AUC 0.725) just reflects current PnL — no predictive edge
- Feature ablation: no feature category adds OOS value for small winners

### Conventional Wisdom Strategies

Tested the classic approaches:
- Cut losers early → already handled by the 30-bar hold
- Let winners ride → runners decline after bar 30 (see above)
- Trail stops → 15pt hard stop adds +0.6 PnL (+7.7 vs +7.1)
- Profit targets → all underperform fixed hold

**Only the 15pt hard stop adds value.**

---

## Final Recommended System

```
Range predictor (width/center decomposition)
  → Heuristic trend score ≤ 1.5 for entry filtering
  → Proximity fade: enter when price within ~10pts of predicted boundary
  → 15pt hard stop
  → 30-bar fixed hold
```

### What We Ruled Out

| Approach | Why Not |
|----------|---------|
| ML entry filter | Heuristic is more stable and interpretable |
| Bar-level exit model | AUC 0.58, too noisy |
| Structural exit models | Marginal gain, high complexity |
| Extended hold (>30 bars) | Incremental PnL is negative |
| Trailing stops on runners | Trivial improvement |
| Small-winner continuation | Genuinely random at bar 30 |
| Non-tree ML models | Don't outperform XGBoost or heuristic |

### Scripts Archived

Research scripts moved to `graveyard/2026-04-apr/range_trade_research/`:
- `train_ptrend_v3.py` — Non-tree models + walk-forward heuristic comparison
- `train_trade_mgmt.py` — Bar-level exit model
- `train_trade_mgmt_v2.py` — Structural exit models (triage, retest, regime)
- `train_runner_mgmt.py` — Runner management analysis
- `train_small_winner.py` — Small-winner deep dive + conventional wisdom
