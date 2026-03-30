#!/usr/bin/env python
"""Run a multi-config SQL playback comparison over a set of trading days."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_reversal_playback_day import (
    _benchmark_old_episodes,
    _benchmark_old_swings,
    _build_strategy,
    _load_causal_history,
    _load_day_bars,
    _load_json,
    _run_diag,
    _summarize_diag,
)


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _config_key(path: str) -> str:
    return Path(path).stem


def _safe_float(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def _mean_ignore_none(vals: List[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reversal SQL playback config matrix")
    parser.add_argument("--days", required=True, help="Comma-separated trading days, e.g. 2026-02-23,2026-02-24")
    parser.add_argument("--configs", required=True, help="Comma-separated config paths")
    parser.add_argument(
        "--output-dir",
        default="sandbox/results/reversal_playback_matrix_latest",
    )
    parser.add_argument(
        "--benchmark-day",
        default="2026-02-26",
    )
    parser.add_argument(
        "--benchmark-episodes-csv",
        default="sandbox/results/day_2026-02-26_stable_v1_top_reversal_episodes.csv",
    )
    parser.add_argument(
        "--benchmark-swings-csv",
        default="sandbox/results/day_2026-02-26_stable_v1_top_swings.csv",
    )
    args = parser.parse_args()

    days = _parse_csv_list(args.days)
    configs = _parse_csv_list(args.configs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    day_cache: Dict[str, Dict[str, Any]] = {}
    results: Dict[str, Any] = {"days_requested": days, "configs": configs, "runs": {}}

    for day in days:
        try:
            bars = _load_day_bars(day)
        except Exception as exc:
            results.setdefault("skipped_days", {})[day] = str(exc)
            continue
        day_cache[day] = {"bars": bars}

    for cfg_path in configs:
        cfg_key = _config_key(cfg_path)
        cfg_obj = _load_json(cfg_path)
        params = dict(cfg_obj.get("params", cfg_obj))
        history_csv = str(params.get("historical_csv_path", "raw_data/schwab/es_minute_history.csv"))
        warmup_days = int(params.get("warmup_days", 60))

        cfg_dir = out_dir / cfg_key
        cfg_dir.mkdir(parents=True, exist_ok=True)

        per_day: Dict[str, Any] = {}
        for day, cached in day_cache.items():
            history_key = (history_csv, warmup_days)
            history_map = cached.setdefault("history_by_key", {})
            if history_key not in history_map:
                history_map[history_key] = _load_causal_history(history_csv, day, warmup_days)

            strategy = _build_strategy(params)
            strategy.set_historical_context(history_map[history_key])

            diag = _run_diag(
                strategy=strategy,
                bars_1m=cached["bars"],
                label_forward_bars=int(strategy._metadata.get("label_forward_bars", 60)),
                label_excursion_ratio=float(strategy._metadata.get("label_excursion_ratio", 1.5)),
                label_breakout_excursion_ratio=float(
                    strategy._metadata.get("label_breakout_excursion_ratio", 1.2)
                ),
                label_terminal_disp_frac=float(
                    strategy._metadata.get("label_terminal_disp_frac", 0.20)
                ),
                label_reversal_vol_mult=float(
                    strategy._metadata.get("label_reversal_vol_mult", 0.10)
                ),
                label_breakout_vol_mult=float(
                    strategy._metadata.get("label_breakout_vol_mult", 0.08)
                ),
            )
            diag["dt"] = pd.to_datetime(diag["dt"])

            prefix = cfg_dir / day
            diag_path = prefix.with_name(prefix.name + "_diag.parquet")
            ep_path = prefix.with_name(prefix.name + "_episodes.csv")
            summary_path = prefix.with_name(prefix.name + "_summary.json")
            bench_ep_compare = prefix.with_name(prefix.name + "_benchmark_old_episodes_compare.csv")
            bench_sw_compare = prefix.with_name(prefix.name + "_benchmark_old_swings_compare.csv")

            summary, episode_df = _summarize_diag(diag)
            old_ep = None
            old_sw = None
            if day == args.benchmark_day:
                old_ep = _benchmark_old_episodes(diag, args.benchmark_episodes_csv, str(bench_ep_compare))
                old_sw = _benchmark_old_swings(diag, args.benchmark_swings_csv, str(bench_sw_compare))

            diag.to_parquet(diag_path, index=False)
            episode_df.to_csv(ep_path, index=False)

            payload = {
                "trading_day": day,
                "config": cfg_path,
                "model_dir": str(params.get("model_dir")),
                "diag_parquet": str(diag_path),
                "episodes_csv": str(ep_path),
                "summary": summary,
                "benchmark_old_episodes": old_ep,
                "benchmark_old_swings": old_sw,
            }
            with open(summary_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            per_day[day] = payload

        day_payloads = list(per_day.values())
        summary_vals = [p["summary"] for p in day_payloads]
        agg = {
            "days_completed": int(len(day_payloads)),
            "zero_trade_days": int(sum(1 for s in summary_vals if int(s["rth_accepted_rows"]) == 0)),
            "zero_trade_rate": (
                float(sum(1 for s in summary_vals if int(s["rth_accepted_rows"]) == 0) / max(len(summary_vals), 1))
            ),
            "mean_rth_accepted_rows": _mean_ignore_none([float(s["rth_accepted_rows"]) for s in summary_vals]),
            "mean_rth_episode_recall": _mean_ignore_none(
                [_safe_float(s.get("rth_episode_recall")) for s in summary_vals]
            ),
            "mean_rth_episode_recall_ge20": _mean_ignore_none(
                [_safe_float(s.get("rth_episode_recall_ge20")) for s in summary_vals]
            ),
            "mean_rth_opening_30m_bar_recall": _mean_ignore_none(
                [_safe_float(s.get("rth_opening_30m_bar_recall_outcome1")) for s in summary_vals]
            ),
            "mean_base_prob_gap": _mean_ignore_none(
                [
                    (
                        _safe_float(s.get("rth_mean_base_prob_outcome1"))
                        - _safe_float(s.get("rth_mean_base_prob_outcome0"))
                    )
                    if s.get("rth_mean_base_prob_outcome1") is not None
                    and s.get("rth_mean_base_prob_outcome0") is not None
                    else None
                    for s in summary_vals
                ]
            ),
        }
        bench_payload = per_day.get(args.benchmark_day)
        if bench_payload is not None:
            agg["benchmark_day"] = args.benchmark_day
            agg["benchmark_old_episode_recall"] = (
                bench_payload.get("benchmark_old_episodes", {}) or {}
            ).get("recall")
            agg["benchmark_old_episode_recall_ge20"] = (
                bench_payload.get("benchmark_old_episodes", {}) or {}
            ).get("recall_ge20")
            agg["benchmark_old_swing_recall_near_level"] = (
                bench_payload.get("benchmark_old_swings", {}) or {}
            ).get("recall_near_level")

        results["runs"][cfg_key] = {
            "config": cfg_path,
            "aggregate": agg,
            "days": per_day,
        }

    summary_path = out_dir / "matrix_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(str(summary_path))


if __name__ == "__main__":
    main()
