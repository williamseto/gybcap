#!/usr/bin/env python
"""Regime-gated trend sidecar experiment (independent book, no policy mixing).

Goal:
- keep reversal policy untouched
- activate trend+pullback sidecar only on likely trend days (early-session gate)
- evaluate OOF/fold-style aggregate and SQL-week behavior
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MISSING_IMPORT: Optional[Exception] = None
try:
    from sandbox import train_level_models as tlm
    from sandbox.run_eod_exit_exec_search_v1 import _build_rth_day_end
    from sandbox.run_global_episode_model_experiments import _build_ohlcv_and_features, _episode_rows
    from sandbox.run_rth_tp_multitask_contract_v1 import _build_tp_contract_rows
    from sandbox.run_sql_policy_context_compare import _build_ohlcv_and_features_for_combined, _load_causal_history, _load_day_bars, _normalize_sql_day
    from sandbox.run_trend_pullback_family_v1 import _barrier_label, _fit_full_model, _walk_forward_scores
    from sandbox.run_trend_pullback_family_v2 import _augment_ohlcv_with_dynamic_pullback_columns
    from sandbox.run_trend_pullback_family_v3 import _pullback_features_v3, _trend_pullback_candidates_v3
    from sandbox.run_trend_pullback_standalone_conjunction_v1 import (
        _aggregate_trade_metrics,
        _choose_reversal_rows,
        _choose_trend_rows,
        _precompute_fixed_outcomes,
        _safe_float,
        _trade_table_fixed,
        _trade_table_mixed,
    )
    from sandbox.validate_frontier_folds_sql_v1 import _frontier_configs, _load_rows_with_scores, _train_full_frontier_heads
except ModuleNotFoundError as exc:  # pragma: no cover - compatibility guard
    _MISSING_IMPORT = exc


def _require_deps() -> None:
    if _MISSING_IMPORT is None:
        return
    raise RuntimeError(
        "run_trend_sidecar_regime_gate_v1.py depends on archived sandbox modules that were pruned. "
        "Restore required scripts from graveyard before running this campaign."
    ) from _MISSING_IMPORT


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _trend_cfgs() -> Dict[str, Dict[str, float]]:
    # Seeded from prior standalone sweep frontier.
    return {
        "trend_top1_f25_g10_s8_t40": {"floor": 0.25, "gap": 10, "stop_pts": 8.0, "target_pts": 40.0},
        "trend_top2_f30_g0_s6_t40": {"floor": 0.30, "gap": 0, "stop_pts": 6.0, "target_pts": 40.0},
        "trend_top3_f30_g0_s8_t40": {"floor": 0.30, "gap": 0, "stop_pts": 8.0, "target_pts": 40.0},
    }


def _trend_gate_day_features(ohlcv: pd.DataFrame, decision_min: int) -> pd.DataFrame:
    m = pd.to_numeric(ohlcv.get("minute_of_day"), errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    mask = (m >= 390) & (m <= int(decision_min))
    if not mask.any():
        return pd.DataFrame(columns=["trading_day", "od_frac_abs", "trend_strength_abs", "move_open_abs"])
    s = ohlcv.loc[mask, ["trading_day", "opening_drive_frac", "session_trend_strength", "dist_from_rth_open_pts"]].copy()
    s["trading_day"] = s["trading_day"].astype(str)
    s["od_frac_abs"] = pd.to_numeric(s["opening_drive_frac"], errors="coerce").abs().fillna(0.0)
    s["trend_strength_abs"] = pd.to_numeric(s["session_trend_strength"], errors="coerce").abs().fillna(0.0)
    s["move_open_abs"] = pd.to_numeric(s["dist_from_rth_open_pts"], errors="coerce").abs().fillna(0.0)
    g = (
        s.groupby("trading_day", sort=False)
        .agg({"od_frac_abs": "max", "trend_strength_abs": "max", "move_open_abs": "max"})
        .reset_index()
    )
    return g


def _gate_specs() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for decision_min in [420, 450]:
        for mode in ["all3", "two_of_three"]:
            for od_thr in [0.8, 1.0, 1.2, 1.4]:
                for ts_thr in [0.08, 0.12, 0.16]:
                    for mv_thr in [8.0, 12.0, 16.0]:
                        out.append(
                            {
                                "decision_min": int(decision_min),
                                "mode": str(mode),
                                "od_thr": float(od_thr),
                                "ts_thr": float(ts_thr),
                                "mv_thr": float(mv_thr),
                                "gate_key": f"d{int(decision_min)}_{mode}_od{int(round(od_thr*100))}_ts{int(round(ts_thr*100))}_mv{int(round(mv_thr))}",
                            }
                        )
    return out


def _active_days_from_gate(day_feat: pd.DataFrame, spec: Dict[str, object]) -> set[str]:
    if day_feat.empty:
        return set()
    c1 = pd.to_numeric(day_feat["od_frac_abs"], errors="coerce").fillna(0.0) >= float(spec["od_thr"])
    c2 = pd.to_numeric(day_feat["trend_strength_abs"], errors="coerce").fillna(0.0) >= float(spec["ts_thr"])
    c3 = pd.to_numeric(day_feat["move_open_abs"], errors="coerce").fillna(0.0) >= float(spec["mv_thr"])
    if str(spec["mode"]) == "all3":
        active = c1 & c2 & c3
    else:
        active = (c1.astype(np.int8) + c2.astype(np.int8) + c3.astype(np.int8)) >= 2
    return set(day_feat.loc[active, "trading_day"].astype(str).tolist())


def _trend_chosen_with_gate(cand: pd.DataFrame, trend_cfg: Dict[str, float], *, k: int, active_days: set[str], decision_min: int) -> np.ndarray:
    chosen = np.zeros(len(cand), dtype=bool)
    if len(cand) == 0 or not active_days:
        return chosen
    sub_mask = cand["trading_day"].astype(str).isin(active_days).to_numpy() & (
        pd.to_numeric(cand["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.float64, copy=False) >= float(decision_min)
    )
    if not sub_mask.any():
        return chosen
    sub = cand.loc[sub_mask].copy().reset_index(drop=True)
    sub_chosen = _choose_trend_rows(sub, trend_cfg, k=int(k))
    chosen[np.flatnonzero(sub_mask)[np.flatnonzero(sub_chosen)]] = True
    return chosen


def _score_sort_key(row: Dict[str, object]) -> Tuple[float, float, float]:
    m = row.get("metrics", {})
    return (
        -_safe_float(m.get("profit_factor"), -1e9),
        _safe_float(m.get("no_target_day_rate"), 1e9),
        -_safe_float(m.get("annualized_daily_sharpe"), -1e9),
    )


def _best_gate_rows(rows: List[Dict[str, object]], *, top_n: int) -> List[Dict[str, object]]:
    # keep rare-but-not-trivial sidecar activation by design.
    filt = [
        r
        for r in rows
        if 0.05 <= _safe_float(r.get("active_day_rate"), -1.0) <= 0.35 and _safe_float(r["metrics"].get("mean_selected_per_day_all"), 0.0) <= 10.0
    ]
    src = filt if filt else rows
    src_sorted = sorted(src, key=_score_sort_key)
    out: List[Dict[str, object]] = []
    seen = set()
    for r in src_sorted:
        g = str(r["gate"]["gate_key"])
        if g in seen:
            continue
        out.append(r)
        seen.add(g)
        if len(out) >= int(top_n):
            break
    return out


def _prepare_sql_day(day: str, *, history_csv: str, warmup_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    history = _load_causal_history(history_csv, day, int(warmup_days))
    day_bars = _normalize_sql_day(day, _load_day_bars(day))
    combined = pd.concat([history, day_bars], axis=0, ignore_index=True, sort=False)
    ohlcv, base_cols = _build_ohlcv_and_features_for_combined(combined)
    ohlcv, dyn_cols = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv)
    all_cols = list(dict.fromkeys(list(base_cols) + list(dyn_cols)))
    episodes, _ = _episode_rows(ohlcv, all_cols, gap_bars=10)
    ep_day = episodes.loc[episodes["trading_day"].astype(str) == str(day)].copy().reset_index(drop=True)
    return ohlcv, ep_day


def _score_sql_reversal_rows(ep_day: pd.DataFrame, heads: Dict[str, object]) -> pd.DataFrame:
    if ep_day.empty:
        return pd.DataFrame()
    fcols = heads["feature_cols"]
    for c in fcols:
        if c not in ep_day.columns:
            ep_day[c] = 0.0
    X = ep_day.loc[:, fcols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    event_p = heads["event_model"].predict_proba(X)[:, 1].astype(np.float64)
    large_p = heads["large_model"].predict_proba(X)[:, 1].astype(np.float64)
    rows, _ = _build_tp_contract_rows(ep_day, fcols, event_p, large_p)
    if rows.empty:
        return rows
    sfeat = heads["strict_features"]
    for c in sfeat:
        if c not in rows.columns:
            rows[c] = 0.0
    Xs = rows.loc[:, sfeat].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    tp = heads["tp_model"].predict_proba(Xs)[:, 1].astype(np.float64)
    fail = heads["fail_model"].predict_proba(Xs)[:, 1].astype(np.float64)
    rows["q_twohead"] = np.clip(tp * (1.0 - fail), 0.0, 1.0)
    return rows


def _score_sql_trend_rows(ep_day: pd.DataFrame, *, full_model, full_feature_cols: List[str]) -> pd.DataFrame:
    day_cand = _trend_pullback_candidates_v3(ep_day)
    if day_cand.empty:
        return day_cand
    for c in full_feature_cols:
        if c not in day_cand.columns:
            day_cand[c] = 0.0
    X = day_cand.loc[:, full_feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    day_cand["score"] = full_model.predict_proba(X)[:, 1].astype(np.float64)
    return day_cand


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=tlm.DATA_PATH)
    ap.add_argument("--history-csv", default="raw_data/schwab/es_minute_history.csv")
    ap.add_argument("--sql-days", default="2026-02-24,2026-02-25,2026-02-26,2026-02-27,2026-03-02")
    ap.add_argument("--warmup-days", type=int, default=60)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=100)
    ap.add_argument("--seed", type=int, default=419)
    ap.add_argument("--k-trend", type=int, default=5)
    ap.add_argument("--trend-train-stop", type=float, default=4.0)
    ap.add_argument("--trend-train-target", type=float, default=10.0)
    ap.add_argument("--forward-bars", type=int, default=60)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--score-rows", default="sandbox/results/upstream_two_stage_constrained_v1_smallgrid_scores.parquet")
    ap.add_argument("--scored-rows", default="sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet")
    ap.add_argument("--reversal-ohlcv", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--top-gates", type=int, default=8)
    ap.add_argument("--output-json", default="sandbox/results/trend_sidecar_regime_gate_v1.json")
    args = ap.parse_args()
    _require_deps()

    t0 = time.time()
    sql_days = [x.strip() for x in str(args.sql_days).split(",") if x.strip()]

    print("Load episodes + trend OOF scoring base...")
    ohlcv, base_cols = _build_ohlcv_and_features(args.data)
    ohlcv, dyn_cols = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv)
    base_cols = list(dict.fromkeys(list(base_cols) + list(dyn_cols)))
    episodes, _ = _episode_rows(ohlcv, base_cols, gap_bars=10)
    trend_cand = _trend_pullback_candidates_v3(episodes).copy().reset_index(drop=True)
    trend_feature_cols = _pullback_features_v3(trend_cand)

    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    trend_cand["cont_target_train"] = _barrier_label(
        trend_cand,
        close,
        high,
        low,
        stop_pts=float(args.trend_train_stop),
        target_pts=float(args.trend_train_target),
        forward_bars=int(args.forward_bars),
    )
    oof_score = _walk_forward_scores(
        trend_cand,
        trend_feature_cols,
        "cont_target_train",
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        seed=int(args.seed),
    )
    valid = np.isfinite(oof_score)
    trend_cand = trend_cand.loc[valid].copy().reset_index(drop=True)
    trend_cand["score"] = oof_score[valid]

    # full model for SQL evaluation later
    full_trend = _trend_pullback_candidates_v3(episodes).copy().reset_index(drop=True)
    full_trend["cont_target_train"] = _barrier_label(
        full_trend,
        close,
        high,
        low,
        stop_pts=float(args.trend_train_stop),
        target_pts=float(args.trend_train_target),
        forward_bars=int(args.forward_bars),
    )
    full_trend_features = _pullback_features_v3(full_trend)
    trend_full_model = _fit_full_model(full_trend, full_trend_features, "cont_target_train", seed=int(args.seed))

    print("Load reversal frontier rows...")
    rev_cfgs = _frontier_configs()
    rev_rows = _load_rows_with_scores(args.rows, args.score_rows, args.scored_rows)
    rev_ohlcv = pd.read_parquet(args.reversal_ohlcv).copy()
    if "_src_index" in rev_ohlcv.columns:
        rev_ohlcv = rev_ohlcv.sort_values("_src_index").reset_index(drop=True)
    rev_close = pd.to_numeric(rev_ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    rev_high = pd.to_numeric(rev_ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    rev_low = pd.to_numeric(rev_ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    rev_day_end = _build_rth_day_end(rev_ohlcv)
    all_days = sorted(rev_ohlcv["trading_day"].astype(str).unique().tolist())

    print("Precompute reversal standalone trades...")
    rev_tables: Dict[str, pd.DataFrame] = {}
    rev_metrics: Dict[str, Dict[str, object]] = {}
    for rname, rcfg in rev_cfgs.items():
        rchosen = _choose_reversal_rows(rev_rows, rcfg)
        rq = pd.to_numeric(rev_rows[rcfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
        rtr = _trade_table_mixed(
            rev_rows,
            rchosen,
            score_q=rq,
            gate=float(rcfg["exec"]["gate"]),
            stop_hi=float(rcfg["exec"]["high_stop"]),
            target_hi=float(rcfg["exec"]["high_target"]),
            stop_lo=float(rcfg["exec"]["low_stop"]),
            target_lo=float(rcfg["exec"]["low_target"]),
            close=rev_close,
            high=rev_high,
            low=rev_low,
            day_end=rev_day_end,
            strategy_name=f"reversal:{rname}",
        )
        rev_tables[rname] = rtr
        rev_metrics[rname] = _aggregate_trade_metrics(rtr, all_days)

    print("Evaluate OOF regime-gated sidecar grid...")
    trend_cfgs = _trend_cfgs()
    day_feats_by_decision = {d: _trend_gate_day_features(ohlcv, d) for d in [420, 450]}
    trend_day_end = _build_rth_day_end(ohlcv)
    trend_exec_pairs = sorted({(float(v["stop_pts"]), float(v["target_pts"])) for v in trend_cfgs.values()})
    precomputed_trend = {
        (stp, tgt): _precompute_fixed_outcomes(
            trend_cand,
            close=close,
            high=high,
            low=low,
            day_end=trend_day_end,
            stop_pts=float(stp),
            target_pts=float(tgt),
        )
        for stp, tgt in trend_exec_pairs
    }

    gate_rows: List[Dict[str, object]] = []
    # Include ungated reference.
    ungated = {
        "decision_min": 390,
        "mode": "ungated",
        "od_thr": 0.0,
        "ts_thr": 0.0,
        "mv_thr": 0.0,
        "gate_key": "ungated",
    }
    all_gate_specs = [ungated] + _gate_specs()
    for spec in all_gate_specs:
        if spec["gate_key"] == "ungated":
            active_days = set(all_days)
        else:
            active_days = _active_days_from_gate(day_feats_by_decision[int(spec["decision_min"])], spec)
        active_rate = float(len(active_days) / len(all_days)) if all_days else 0.0

        for rname in rev_cfgs.keys():
            rev_trades = rev_tables[rname]
            for tname, tcfg in trend_cfgs.items():
                tch = _trend_chosen_with_gate(
                    trend_cand,
                    tcfg,
                    k=int(args.k_trend),
                    active_days=active_days,
                    decision_min=int(spec["decision_min"]),
                )
                pnl_arr, kind_arr = precomputed_trend[(float(tcfg["stop_pts"]), float(tcfg["target_pts"]))]
                ttr = _trade_table_fixed(
                    trend_cand,
                    tch,
                    close=close,
                    high=high,
                    low=low,
                    day_end=trend_day_end,
                    stop_pts=float(tcfg["stop_pts"]),
                    target_pts=float(tcfg["target_pts"]),
                    strategy_name=f"trend:{tname}",
                    precomputed_pnl=pnl_arr,
                    precomputed_kind=kind_arr,
                )
                combo = pd.concat([rev_trades, ttr], axis=0, ignore_index=True, sort=False)
                met = _aggregate_trade_metrics(combo, all_days)
                tmet = _aggregate_trade_metrics(ttr, all_days)
                gate_rows.append(
                    {
                        "reversal_cfg": rname,
                        "trend_cfg": tname,
                        "gate": dict(spec),
                        "active_days": int(len(active_days)),
                        "active_day_rate": float(active_rate),
                        "metrics": met,
                        "trend_sidecar_metrics": tmet,
                    }
                )

    best_gates = _best_gate_rows(gate_rows, top_n=int(args.top_gates))

    print("SQL-week evaluation for shortlisted gates...")
    sql_out = {
        "shortlisted_gate_rows": best_gates,
        "aggregate": {},
        "per_day": {},
    }
    if sql_days and best_gates:
        rev_heads = _train_full_frontier_heads(args.data, seed=int(args.seed))
        # Deduplicate specs by gate key to avoid repeated per-day eval cost.
        shortlist_specs: Dict[str, Dict[str, object]] = {}
        for r in best_gates:
            shortlist_specs[str(r["gate"]["gate_key"])] = dict(r["gate"])

        per_combo_tables: Dict[str, List[pd.DataFrame]] = {}
        for rname in rev_cfgs.keys():
            for tname in trend_cfgs.keys():
                for gkey in shortlist_specs.keys():
                    per_combo_tables[f"{rname}__{tname}__{gkey}"] = []

        for day in sql_days:
            try:
                ohlcv_day, ep_day = _prepare_sql_day(day, history_csv=args.history_csv, warmup_days=int(args.warmup_days))
                if ep_day.empty:
                    sql_out["per_day"][str(day)] = {"note": "no_episode_rows"}
                    continue
                day_close = pd.to_numeric(ohlcv_day["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
                day_high = pd.to_numeric(ohlcv_day["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
                day_low = pd.to_numeric(ohlcv_day["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
                day_end = _build_rth_day_end(ohlcv_day)
                day_feat_420 = _trend_gate_day_features(ohlcv_day, 420)
                day_feat_450 = _trend_gate_day_features(ohlcv_day, 450)

                rev_day = _score_sql_reversal_rows(ep_day.copy(), rev_heads)
                trend_day = _score_sql_trend_rows(ep_day.copy(), full_model=trend_full_model, full_feature_cols=full_trend_features)
                day_out = {
                    "n_episode_rows": int(len(ep_day)),
                    "n_reversal_rows": int(len(rev_day)),
                    "n_trend_rows": int(len(trend_day)),
                    "combos": {},
                }

                # precompute day reversal tables per cfg
                rev_day_tables: Dict[str, pd.DataFrame] = {}
                for rname, rcfg in rev_cfgs.items():
                    if rev_day.empty:
                        rtr = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                    else:
                        rchosen = _choose_reversal_rows(rev_day, rcfg)
                        rq = pd.to_numeric(rev_day[rcfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
                        rtr = _trade_table_mixed(
                            rev_day,
                            rchosen,
                            score_q=rq,
                            gate=float(rcfg["exec"]["gate"]),
                            stop_hi=float(rcfg["exec"]["high_stop"]),
                            target_hi=float(rcfg["exec"]["high_target"]),
                            stop_lo=float(rcfg["exec"]["low_stop"]),
                            target_lo=float(rcfg["exec"]["low_target"]),
                            close=day_close,
                            high=day_high,
                            low=day_low,
                            day_end=day_end,
                            strategy_name=f"reversal:{rname}",
                        )
                    rev_day_tables[rname] = rtr

                for gkey, gspec in shortlist_specs.items():
                    if gspec["gate_key"] == "ungated":
                        active = True
                    else:
                        dfeat = day_feat_420 if int(gspec["decision_min"]) == 420 else day_feat_450
                        adays = _active_days_from_gate(dfeat, gspec)
                        active = str(day) in adays
                    for tname, tcfg in trend_cfgs.items():
                        if trend_day.empty or (not active):
                            ttr = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                        else:
                            tmask = pd.to_numeric(trend_day["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.float64, copy=False) >= float(gspec["decision_min"])
                            tsub = trend_day.loc[tmask].copy().reset_index(drop=True)
                            tchosen = _choose_trend_rows(tsub, tcfg, k=int(args.k_trend))
                            ttr = _trade_table_fixed(
                                tsub,
                                tchosen,
                                close=day_close,
                                high=day_high,
                                low=day_low,
                                day_end=day_end,
                                stop_pts=float(tcfg["stop_pts"]),
                                target_pts=float(tcfg["target_pts"]),
                                strategy_name=f"trend:{tname}",
                            )
                        for rname in rev_cfgs.keys():
                            combo = pd.concat([rev_day_tables[rname], ttr], axis=0, ignore_index=True, sort=False)
                            key = f"{rname}__{tname}__{gkey}"
                            per_combo_tables[key].append(combo)
                            day_out["combos"][key] = _aggregate_trade_metrics(combo, [str(day)])
                sql_out["per_day"][str(day)] = _to_jsonable(day_out)
            except Exception as exc:
                sql_out["per_day"][str(day)] = {"error": str(exc)}

        sql_out["aggregate"] = {
            k: _to_jsonable(_aggregate_trade_metrics(pd.concat(v, axis=0, ignore_index=True, sort=False) if v else pd.DataFrame(), sql_days))
            for k, v in per_combo_tables.items()
        }

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "meta": {
            "data": str(args.data),
            "sql_days": sql_days,
            "k_trend": int(args.k_trend),
            "top_gates": int(args.top_gates),
            "note": "Trend sidecar only; reversal policy unchanged; no mixed routing policies.",
        },
        "reversal_baseline_oof": _to_jsonable(rev_metrics),
        "trend_sidecar_cfgs": _to_jsonable(trend_cfgs),
        "gate_grid_oof": _to_jsonable(gate_rows),
        "best_gates_oof": _to_jsonable(best_gates),
        "sql_shortlist_eval": _to_jsonable(sql_out),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(out), indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
