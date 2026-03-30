#!/usr/bin/env python
"""Validate current frontier configs on walk-forward folds + recent SQL playback days."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MISSING_IMPORT: Optional[Exception] = None
try:
    from sandbox import train_level_models as tlm
    from sandbox.run_eod_exit_exec_search_v1 import _build_rth_day_end, _simulate_mixed_eod_metrics
    from sandbox.run_global_episode_model_experiments import _episode_rows
    from sandbox.run_quality_coverage_router_v1 import _build_day_groups, _simulate_router
    from sandbox.run_no_tp10_policy_grid_v1 import _day_groups, _label_metrics
    from sandbox.run_rth_tp_multitask_contract_v1 import (
        _build_tp_contract_rows,
        _fit_full_base_binary,
        _fit_full_weighted_binary,
    )
    from sandbox.run_sql_policy_context_compare import (
        _build_ohlcv_and_features_for_combined,
        _load_causal_history,
        _load_day_bars,
        _load_training_episodes,
        _normalize_sql_day,
    )
    from sandbox.run_upstream_stop10_separability_v1 import _attach_base_scores, _feature_pool
    from sandbox.run_upstream_twohead_regime_policy_v1 import (
        _build_fail_weights,
        _build_tp_weights,
        _strict_feature_subset,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - compatibility guard
    _MISSING_IMPORT = exc


def _require_deps() -> None:
    if _MISSING_IMPORT is None:
        return
    raise RuntimeError(
        "validate_frontier_folds_sql_v1.py depends on archived sandbox modules that were pruned. "
        "Restore required scripts from graveyard before running this validator."
    ) from _MISSING_IMPORT


def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _mean(vals: List[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _frontier_configs() -> Dict[str, Dict]:
    # From targeted frontier sweeps:
    # - strict-feasible best: PF~1.972, noTP~0.083
    # - high-PF edge case: PF~2.024, noTP~0.120
    return {
        "frontier_feasible": {
            "quality_col": "q_twohead",
            "policy": {
                "k_total": 7,
                "q_thr": 0.58,
                "q_slots": 3,
                "c_thr": 0.48,
                "c_start": 450,
                "c_slots": 2,
                "r_thr": 0.40,
                "r_start": 600,
                "r_slots": 2,
                "cooldown": 5,
                "cap": 2,
            },
            "exec": {
                "gate": 0.62,
                "high_stop": 10.0,
                "high_target": 40.0,
                "low_stop": 4.0,
                "low_target": 8.0,
            },
        },
        "frontier_high_pf": {
            "quality_col": "q_twohead",
            "policy": {
                "k_total": 7,
                "q_thr": 0.58,
                "q_slots": 3,
                "c_thr": 0.48,
                "c_start": 450,
                "c_slots": 2,
                "r_thr": 0.45,
                "r_start": 600,
                "r_slots": 1,
                "cooldown": 5,
                "cap": 2,
            },
            "exec": {
                "gate": 0.62,
                "high_stop": 10.0,
                "high_target": 40.0,
                "low_stop": 4.0,
                "low_target": 8.0,
            },
        },
    }


def _load_rows_with_scores(rows_path: str, score_rows_path: str, scored_rows_path: str) -> pd.DataFrame:
    rows = pd.read_parquet(rows_path).copy().reset_index(drop=True)
    score_rows = pd.read_parquet(score_rows_path).copy().reset_index(drop=True)
    rows = _attach_base_scores(rows, score_rows)
    q = pd.read_parquet(scored_rows_path).copy().reset_index(drop=True)
    keep = [c for c in ["trading_day", "episode_start_idx", "q_tp", "q_twohead", "q_twohead_regime", "q_twohead_stable"] if c in q.columns]
    rows = rows.merge(q[keep], on=["trading_day", "episode_start_idx"], how="left")
    for c in ["q_tp", "q_twohead", "q_twohead_regime", "q_twohead_stable"]:
        if c not in rows.columns:
            rows[c] = 0.5
        rows[c] = pd.to_numeric(rows[c], errors="coerce").fillna(0.5)
    if "event_base_prob" not in rows.columns:
        rows["event_base_prob"] = 0.5
    return rows


def _eval_config_on_rows(
    rows: pd.DataFrame,
    cfg: Dict,
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
) -> Dict[str, float]:
    rg = _build_day_groups(rows)
    chosen = _simulate_router(
        rows,
        rg,
        quality_col=cfg["quality_col"],
        coverage_col="event_base_prob",
        rescue_col="event_base_prob",
        k_total=int(cfg["policy"]["k_total"]),
        q_thr=float(cfg["policy"]["q_thr"]),
        q_slots=int(cfg["policy"]["q_slots"]),
        c_thr=float(cfg["policy"]["c_thr"]),
        c_start=int(cfg["policy"]["c_start"]),
        c_slots=int(cfg["policy"]["c_slots"]),
        r_thr=float(cfg["policy"]["r_thr"]),
        r_start=int(cfg["policy"]["r_start"]),
        r_slots=int(cfg["policy"]["r_slots"]),
        cooldown_min=int(cfg["policy"]["cooldown"]),
        diversity_cap=int(cfg["policy"]["cap"]),
    )
    label = _label_metrics(rows, chosen, _day_groups(rows))
    score_q = pd.to_numeric(rows[cfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
    m_exec = _simulate_mixed_eod_metrics(
        rows,
        chosen,
        score_q=score_q,
        gate=float(cfg["exec"]["gate"]),
        stop_hi=float(cfg["exec"]["high_stop"]),
        target_hi=float(cfg["exec"]["high_target"]),
        stop_lo=float(cfg["exec"]["low_stop"]),
        target_lo=float(cfg["exec"]["low_target"]),
        close=close,
        high=high,
        low=low,
        day_end=day_end,
    )
    return {"label": label, "metrics": m_exec}


def _fold_validation(rows: pd.DataFrame, ohlcv_path: str, cfgs: Dict[str, Dict], *, n_folds: int, min_train_days: int) -> Dict:
    oc = pd.read_parquet(ohlcv_path).copy()
    if "_src_index" in oc.columns:
        oc = oc.sort_values("_src_index").reset_index(drop=True)
    close = pd.to_numeric(oc["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(oc["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(oc["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(oc)

    days = sorted(rows["trading_day"].astype(str).unique().tolist())
    splits = tlm.get_fold_splits(days, int(n_folds), int(min_train_days))
    out: Dict[str, Dict] = {}
    for cfg_name, cfg in cfgs.items():
        folds: List[Dict] = []
        for i, (_, test_days) in enumerate(splits):
            fold_rows = rows.loc[rows["trading_day"].astype(str).isin(set(test_days))].copy().reset_index(drop=True)
            if fold_rows.empty:
                continue
            res = _eval_config_on_rows(fold_rows, cfg, close=close, high=high, low=low, day_end=day_end)
            folds.append(
                {
                    "fold": int(i),
                    "n_days": int(fold_rows["trading_day"].astype(str).nunique()),
                    "label": res["label"],
                    "metrics": res["metrics"],
                }
            )
        agg = {
            "mean_pf": _mean([_safe_float(f["metrics"].get("profit_factor"), np.nan) for f in folds]),
            "mean_sharpe": _mean([_safe_float(f["metrics"].get("annualized_daily_sharpe"), np.nan) for f in folds]),
            "mean_no_tp_day_rate": _mean([_safe_float(f["label"].get("rth_no_tp_day_rate"), np.nan) for f in folds]),
            "mean_selected_per_day": _mean([_safe_float(f["label"].get("mean_selected_per_day"), np.nan) for f in folds]),
            "mean_precision": _mean([_safe_float(f["label"].get("precision_labeled"), np.nan) for f in folds]),
            "worst_fold_pf": min([_safe_float(f["metrics"].get("profit_factor"), np.nan) for f in folds], default=None),
            "worst_fold_no_tp_day_rate": max([_safe_float(f["label"].get("rth_no_tp_day_rate"), np.nan) for f in folds], default=None),
        }
        out[cfg_name] = {"aggregate": agg, "folds": folds}
    return out


def _train_full_frontier_heads(data_path: str, seed: int) -> Dict:
    train_ep, feature_cols = _load_training_episodes(data_path)
    event_model = _fit_full_base_binary(train_ep, feature_cols, "episode_binary_target", seed=int(seed))
    large_model = _fit_full_base_binary(train_ep, feature_cols, "large_move_target", seed=int(seed) + 100)

    X_full = train_ep.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    event_base = event_model.predict_proba(X_full)[:, 1].astype(np.float64)
    large_base = large_model.predict_proba(X_full)[:, 1].astype(np.float64)
    contract_rows, contract_features = _build_tp_contract_rows(train_ep, feature_cols, event_base, large_base)

    y_tp = pd.to_numeric(contract_rows["tp_now_target"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    contract_rows["fail_target"] = np.where(np.isfinite(y_tp), (y_tp == 0.0).astype(np.float64), np.nan)
    contract_rows["w_tp"] = _build_tp_weights(contract_rows)
    contract_rows["w_fail"] = _build_fail_weights(contract_rows)

    strict_features = _strict_feature_subset(contract_rows, _feature_pool(contract_rows))
    tp_model = _fit_full_weighted_binary(
        contract_rows,
        strict_features,
        "tp_now_target",
        "w_tp",
        seed=int(seed) + 210,
        monotone_event_anchor=False,
    )
    fail_model = _fit_full_weighted_binary(
        contract_rows,
        strict_features,
        "fail_target",
        "w_fail",
        seed=int(seed) + 220,
        monotone_event_anchor=False,
    )
    return {
        "event_model": event_model,
        "large_model": large_model,
        "tp_model": tp_model,
        "fail_model": fail_model,
        "feature_cols": feature_cols,
        "strict_features": strict_features,
    }


def _score_sql_day(
    day: str,
    heads: Dict,
    *,
    history_csv: str,
    warmup_days: int,
    cfgs: Dict[str, Dict],
) -> Dict:
    history = _load_causal_history(history_csv, day, int(warmup_days))
    day_bars = _normalize_sql_day(day, _load_day_bars(day))
    combined = pd.concat([history, day_bars], axis=0, ignore_index=True, sort=False)
    ohlcv, base_cols = _build_ohlcv_and_features_for_combined(combined)
    episodes, _ = _episode_rows(ohlcv, base_cols, gap_bars=10)
    ep_day = episodes.loc[episodes["trading_day"].astype(str) == str(day)].copy().reset_index(drop=True)
    if ep_day.empty:
        return {"error": "no_episode_rows"}

    feature_cols = heads["feature_cols"]
    for c in feature_cols:
        if c not in ep_day.columns:
            ep_day[c] = 0.0

    X_day = ep_day.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    event_day = heads["event_model"].predict_proba(X_day)[:, 1].astype(np.float64)
    large_day = heads["large_model"].predict_proba(X_day)[:, 1].astype(np.float64)
    rows_day, _ = _build_tp_contract_rows(ep_day, feature_cols, event_day, large_day)
    if rows_day.empty:
        return {"error": "no_contract_rows"}

    strict_features = heads["strict_features"]
    for c in strict_features:
        if c not in rows_day.columns:
            rows_day[c] = 0.0
    X_tp = rows_day.loc[:, strict_features].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    tp = heads["tp_model"].predict_proba(X_tp)[:, 1].astype(np.float64)
    fail = heads["fail_model"].predict_proba(X_tp)[:, 1].astype(np.float64)
    rows_day["q_twohead"] = np.clip(tp * (1.0 - fail), 0.0, 1.0)

    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)

    out_cfg: Dict[str, Dict] = {}
    for name, cfg in cfgs.items():
        res = _eval_config_on_rows(rows_day, cfg, close=close, high=high, low=low, day_end=day_end)
        out_cfg[name] = res
    return {
        "n_episode_rows": int(len(ep_day)),
        "n_contract_rows": int(len(rows_day)),
        "configs": out_cfg,
    }


def _sql_week_validation(
    cfgs: Dict[str, Dict],
    *,
    data_path: str,
    history_csv: str,
    warmup_days: int,
    sql_days: List[str],
    seed: int,
) -> Dict:
    heads = _train_full_frontier_heads(data_path, seed=int(seed))
    per_day: Dict[str, Dict] = {}
    for d in sql_days:
        try:
            per_day[d] = _score_sql_day(d, heads, history_csv=history_csv, warmup_days=int(warmup_days), cfgs=cfgs)
        except Exception as exc:
            per_day[d] = {"error": str(exc)}

    agg: Dict[str, Dict] = {}
    for name in cfgs.keys():
        rows = [v["configs"][name] for v in per_day.values() if "configs" in v and name in v["configs"]]
        agg[name] = {
            "n_days": int(len(rows)),
            "mean_pf": _mean([_safe_float(r["metrics"].get("profit_factor"), np.nan) for r in rows]),
            "mean_sharpe": _mean([_safe_float(r["metrics"].get("annualized_daily_sharpe"), np.nan) for r in rows]),
            "mean_no_tp_day_rate": _mean([_safe_float(r["label"].get("rth_no_tp_day_rate"), np.nan) for r in rows]),
            "mean_selected_per_day": _mean([_safe_float(r["label"].get("mean_selected_per_day"), np.nan) for r in rows]),
            "mean_precision": _mean([_safe_float(r["label"].get("precision_labeled"), np.nan) for r in rows]),
            "mean_pnl_per_day": _mean([_safe_float(r["metrics"].get("mean_pnl_per_day"), np.nan) for r in rows]),
            "profit_day_rate": _mean([_safe_float(r["metrics"].get("profit_day_rate"), np.nan) for r in rows]),
        }
    return {"aggregate": agg, "per_day": per_day}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--score-rows", default="sandbox/results/upstream_two_stage_constrained_v1_smallgrid_scores.parquet")
    ap.add_argument("--scored-rows", default="sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet")
    ap.add_argument("--ohlcv", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--data", default=tlm.DATA_PATH)
    ap.add_argument("--history-csv", default="raw_data/schwab/es_minute_history.csv")
    ap.add_argument("--warmup-days", type=int, default=60)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=100)
    ap.add_argument("--sql-days", default="2026-02-24,2026-02-25,2026-02-26,2026-02-27,2026-03-02")
    ap.add_argument("--seed", type=int, default=419)
    ap.add_argument("--output-json", default="sandbox/results/frontier_validation_folds_sql_v1.json")
    args = ap.parse_args()
    _require_deps()

    t0 = time.time()
    cfgs = _frontier_configs()
    rows = _load_rows_with_scores(args.rows, args.score_rows, args.scored_rows)

    fold_out = _fold_validation(
        rows,
        args.ohlcv,
        cfgs,
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
    )

    sql_days = [x.strip() for x in str(args.sql_days).split(",") if x.strip()]
    sql_out = _sql_week_validation(
        cfgs,
        data_path=args.data,
        history_csv=args.history_csv,
        warmup_days=int(args.warmup_days),
        sql_days=sql_days,
        seed=int(args.seed),
    )

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "configs": cfgs,
        "fold_validation": fold_out,
        "sql_week_validation": sql_out,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
