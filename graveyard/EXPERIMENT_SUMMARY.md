# Graveyard Experiment Summary

Last updated: 2026-03-30

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

## Recovery
- Restore any removed artifact from git:
  - `git log --all --full-history -- graveyard/<path>`
  - `git checkout <commit> -- graveyard/<path>`
- For higher-level context, also see:
  - `.codex/MEMORY.md`
  - `.codex/RESEARCH_SUMMARY_2026-03-10.md`

