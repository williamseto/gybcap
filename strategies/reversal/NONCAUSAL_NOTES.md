# Non-Causal Model Notes (V1/V2)

Archived Feb 2026. This documents the multi-scale rejection model experiments
that relied on 1-second data and were found to be lookahead-biased.

## V1 Architecture (~89K params)
- 3 VP branch CNNs (micro/meso/macro from 1-sec data)
- TCN on 180-bar 1-sec sequence
- Scalar MLP for rejection features
- Binary sigmoid head
- Results: P=59.1%, R=29.1%, F1=39.0%, ROC=0.627 (406 OOS events)

## V2 Architecture (~108K params)
Added: FiLM conditioning, prototype head, attention pooling, BatchNorm on scalars.
- Results: P=60.0%, R=49.3%, F1=54.1%, ROC=0.724

## Causal Analysis (Feb 7, 2026)
Replaced forward-looking windows with backward-only:
- **Causal V2**: P=0%, R=0%, F1=0%, ROC=0.525 (~random)
- **Non-causal V2**: P=56.9%, R=46.3%, F1=51.0%
- Conclusion: V2 relied entirely on post-event (future) data.

## Sliding Lookahead (Feb 8, 2026)

| Lookahead | Prec  | Recall | F1    | ROC   |
|-----------|-------|--------|-------|-------|
| 0s        | 0.0%  | 0.0%   | 0.0%  | 0.540 |
| 60s       | 0.0%  | 0.0%   | 0.0%  | 0.521 |
| 90s       | 41.6% | 27.6%  | 33.2% | 0.539 |
| 120s      | 43.0% | 38.8%  | 40.8% | 0.587 |

Signal appears only with >90s of post-event data. Delayed-entry PnL is
negative at all causal lookaheads.

## Lessons Learned
1. Always validate with causal features before claiming predictive edge.
2. Pre-event microstructure alone (1-sec) carries ~zero signal for this architecture.
3. 1-sec data had only 107 overlapping days with 1-min (major gap in 2024).
4. The trigger-bar framing is inherently sensitive to detection timing.

## V3 Approach
Zone-based labeling using 1-min bars only. Predict P(pre-reversal zone) at
every bar near a level, learning approach patterns rather than reacting to
a precise trigger bar. See `causal_model.py` and `causal_trainer.py`.
