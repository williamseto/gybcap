#!/usr/bin/env python
"""Sweep q_twohead runtime router settings under sequential replacement policy.

Fast in-process evaluator to iterate on live-feasible contracts:
- quality source: q_twohead runtime inference
- execution: compare independent reference vs sequential replacement variants
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import sandbox.run_sequential_reconciliation_v1 as sr


def _safe_float(v, d=0.0):
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(d)


def _build_router_grid(base: sr.RouterCfg) -> List[sr.RouterCfg]:
    grid: List[sr.RouterCfg] = []

    # Baseline and high-quality tightening.
    grid.append(replace(base))
    for q_thr in [0.21, 0.22, 0.23]:
        grid.append(replace(base, q_thr=float(q_thr)))

    # Budget/cadence variants.
    grid.append(replace(base, k_total=5, q_slots=5, c_slots=3, r_slots=2))
    grid.append(replace(base, k_total=7, q_slots=6, c_slots=4, r_slots=2))
    grid.append(replace(base, cooldown_min=4))
    grid.append(replace(base, cooldown_min=3, k_total=7, q_slots=6, c_slots=4, r_slots=2))

    # Post-open balance variants.
    grid.append(replace(base, c_start=480, c_slots=4, r_start=630, r_slots=2))
    grid.append(replace(base, c_start=510, c_slots=4, r_start=630, r_slots=2))

    # Slightly stricter rescue/coverage with stronger quality floor.
    grid.append(replace(base, q_thr=0.22, c_thr=0.18, r_thr=0.14))
    grid.append(replace(base, q_thr=0.23, c_thr=0.18, r_thr=0.14, c_start=480))

    # Ensure uniqueness.
    uniq: Dict[str, sr.RouterCfg] = {}
    for cfg in grid:
        key = json.dumps(asdict(cfg), sort_keys=True)
        uniq[key] = cfg
    return list(uniq.values())


def _router_name(cfg: sr.RouterCfg) -> str:
    return (
        f"k{cfg.k_total}_q{cfg.q_thr:.2f}_qs{cfg.q_slots}_"
        f"c{cfg.c_thr:.2f}@{cfg.c_start}_cs{cfg.c_slots}_"
        f"r{cfg.r_thr:.2f}@{cfg.r_start}_rs{cfg.r_slots}_"
        f"cd{cfg.cooldown_min}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows-parquet", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--q-rows-parquet", default="sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet")
    ap.add_argument("--ohlcv-parquet", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--live-config", default="configs/realtime/strategies/reversal_frontier_qtwohead_runtime_v2_cal_d_k6_ll99_v1.jsonc")
    ap.add_argument("--qtwohead-model-dir", default="models/reversal_frontier_qtwohead_runtime_v2")
    ap.add_argument("--output-json", default="sandbox/results/qtwohead_runtime_router_sweep_v1_20260321.json")
    args = ap.parse_args()

    t0 = time.time()

    live = sr._load_live_router_cfg(args.live_config)
    router_base = sr._router_cfg_from_live(live)
    router_base.quality_source = "q_twohead_runtime"

    rows = sr._load_rows(args.rows_parquet, args.q_rows_parquet)
    ohlcv = sr._load_ohlcv(args.ohlcv_parquet)
    rows = sr._add_proximity(rows, ohlcv)
    rows = sr._add_runtime_qtwohead(rows, model_dir=str(args.qtwohead_model_dir), score_col="q_twohead_runtime")

    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)

    day_groups = sr._build_day_groups(rows)
    day_end = sr._build_rth_day_end(ohlcv)
    all_days = [d for d, _ in day_groups]

    variants = [
        sr.ExecVariant(name="independent_ref_live_router", mode="independent", opposite_action="ignore"),
        sr.ExecVariant(
            name="seq_hold_replace_qgain04_px025_v1",
            mode="sequential",
            opposite_action="ignore",
            same_side_action="replace",
            same_side_replace_min_q_gain=0.04,
            same_side_replace_min_price_improve=0.25,
            same_side_min_hold_bars=1,
            consume_budget_on_replace=False,
        ),
        sr.ExecVariant(
            name="seq_hold_replace_qgain03_px000_v1",
            mode="sequential",
            opposite_action="ignore",
            same_side_action="replace",
            same_side_replace_min_q_gain=0.03,
            same_side_replace_min_price_improve=0.0,
            same_side_min_hold_bars=1,
            consume_budget_on_replace=False,
        ),
    ]

    out_rows: List[Dict] = []
    grid = _build_router_grid(router_base)
    for cfg in grid:
        rname = _router_name(cfg)
        for v in variants:
            chosen, tdf, diag = sr._simulate_variant(
                rows=rows,
                day_groups=day_groups,
                day_end=day_end,
                close=close,
                high=high,
                low=low,
                router_cfg=cfg,
                variant=v,
                policy_scorer=None,
            )
            label = sr._label_metrics(rows, chosen, day_groups)
            execm = sr._exec_metrics(tdf, all_days)
            rec = {
                "router_name": rname,
                "variant": v.name,
                "router_cfg": asdict(cfg),
                "variant_cfg": asdict(v),
                "label": label,
                "exec": execm,
                "diag": diag,
            }
            out_rows.append(rec)
            print(
                f"{rname:60s} {v.name:34s} PF={_safe_float(execm.get('profit_factor'),-1):.3f} "
                f"noTP={_safe_float(label.get('rth_no_tp_day_rate'),1):.3f} tr/day={_safe_float(execm.get('mean_trades_per_day'),0):.2f}"
            )

    ranked = sorted(
        out_rows,
        key=lambda r: (
            -_safe_float(r["exec"].get("profit_factor"), -1e9),
            _safe_float(r["label"].get("rth_no_tp_day_rate"), 1e9),
            -_safe_float(r["exec"].get("annualized_daily_sharpe"), -1e9),
            -_safe_float(r["exec"].get("mean_pnl_per_day"), -1e9),
        ),
    )

    feasible = [
        r for r in ranked
        if 5.0 <= _safe_float(r["exec"].get("mean_trades_per_day"), 0.0) <= 10.0
    ]
    target_like = [
        r for r in feasible
        if _safe_float(r["exec"].get("profit_factor"), 0.0) >= 2.0
        and _safe_float(r["label"].get("rth_no_tp_day_rate"), 1.0) <= 0.10
    ]

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "meta": {
            "rows_parquet": str(args.rows_parquet),
            "q_rows_parquet": str(args.q_rows_parquet),
            "ohlcv_parquet": str(args.ohlcv_parquet),
            "live_config": str(args.live_config),
            "qtwohead_model_dir": str(args.qtwohead_model_dir),
            "n_rows": int(len(rows)),
            "n_days": int(len(all_days)),
            "n_router_cfg": int(len(grid)),
            "n_variants": int(len(variants)),
        },
        "ranked": sr._to_jsonable(ranked),
        "feasible_top20": sr._to_jsonable(feasible[:20]),
        "target_like_top20": sr._to_jsonable(target_like[:20]),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sr._to_jsonable(out), indent=2) + "\n", encoding="utf-8")

    rows_csv = []
    for r in ranked:
        rows_csv.append(
            {
                "router_name": r["router_name"],
                "variant": r["variant"],
                "profit_factor": _safe_float(r["exec"].get("profit_factor"), np.nan),
                "annualized_daily_sharpe": _safe_float(r["exec"].get("annualized_daily_sharpe"), np.nan),
                "mean_pnl_per_day": _safe_float(r["exec"].get("mean_pnl_per_day"), np.nan),
                "mean_trades_per_day": _safe_float(r["exec"].get("mean_trades_per_day"), np.nan),
                "target_hit_rate": _safe_float(r["exec"].get("target_hit_rate"), np.nan),
                "stop_hit_rate": _safe_float(r["exec"].get("stop_hit_rate"), np.nan),
                "timed_exit_rate": _safe_float(r["exec"].get("timed_exit_rate"), np.nan),
                "manual_exit_rate": _safe_float(r["exec"].get("manual_exit_rate"), np.nan),
                "rth_no_tp_day_rate": _safe_float(r["label"].get("rth_no_tp_day_rate"), np.nan),
                "rth_no_large_tp_day_rate": _safe_float(r["label"].get("rth_no_large_tp_day_rate"), np.nan),
            }
        )
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
