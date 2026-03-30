#!/usr/bin/env python
"""Standalone trend+pullback v3 campaign + conjunction scorecard vs reversal.

Intent:
- keep trend+pullback as an independent model/strategy lane
- evaluate with the same style of OOF/fold/SQL scorecards used for reversal
- measure conjunction with reversal as two independent books (no policy mixing)
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    from sandbox.run_execution_pf_sweep_v1 import _simulate_diverse_first_k
    from sandbox.run_fp_reduction_followon import _to_jsonable
    from sandbox.run_global_episode_model_experiments import _build_ohlcv_and_features, _episode_rows
    from sandbox.run_quality_coverage_router_v1 import _build_day_groups, _simulate_router
    from sandbox.run_rth_tp_multitask_contract_v1 import _build_tp_contract_rows
    from sandbox.run_sql_policy_context_compare import _build_ohlcv_and_features_for_combined, _load_causal_history, _load_day_bars, _normalize_sql_day
    from sandbox.run_trend_pullback_family_v1 import _barrier_label, _fit_full_model, _walk_forward_scores
    from sandbox.run_trend_pullback_family_v2 import _augment_ohlcv_with_dynamic_pullback_columns
    from sandbox.run_trend_pullback_family_v3 import _pullback_features_v3, _trend_pullback_candidates_v3
    from sandbox.validate_frontier_folds_sql_v1 import _frontier_configs, _load_rows_with_scores, _train_full_frontier_heads
except ModuleNotFoundError as exc:  # pragma: no cover - compatibility guard
    _MISSING_IMPORT = exc


def _require_deps() -> None:
    if _MISSING_IMPORT is None:
        return
    raise RuntimeError(
        "run_trend_pullback_standalone_conjunction_v1.py depends on archived sandbox modules that were pruned. "
        "Restore required scripts from graveyard before running this campaign."
    ) from _MISSING_IMPORT


def _safe_float(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _mean(vals: Iterable[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _variant_parts(variant: str) -> Tuple[float, int]:
    m = re.search(r"_f(\d+)_g(\d+)$", str(variant))
    if not m:
        raise ValueError(f"Cannot parse trend variant: {variant}")
    return float(int(m.group(1)) / 100.0), int(m.group(2))


def _score_sort_key(row: Dict[str, object]) -> Tuple[float, float, float]:
    m = row.get("metrics", {})
    return (
        -_safe_float(m.get("profit_factor"), -1e9),
        -_safe_float(m.get("annualized_daily_sharpe"), -1e9),
        -_safe_float(m.get("mean_pnl_per_day"), -1e9),
    )


def _simulate_trade(entry: float, trade_dir: int, stop_pts: float, target_pts: float, hi: np.ndarray, lo: np.ndarray, close: np.ndarray, i0: int, i1: int) -> Tuple[float, str]:
    if i1 <= i0 + 1:
        return 0.0, "timed"
    done = False
    if trade_dir >= 0:
        stop_px = float(entry - stop_pts)
        target_px = float(entry + target_pts)
        for j in range(i0 + 1, i1):
            h = float(hi[j])
            l = float(lo[j])
            if l <= stop_px and h >= target_px:
                return -float(stop_pts), "stop_tie"
            if l <= stop_px:
                return -float(stop_pts), "stop"
            if h >= target_px:
                return float(target_pts), "target"
        if not done:
            return float(close[i1 - 1] - entry), "timed"
    stop_px = float(entry + stop_pts)
    target_px = float(entry - target_pts)
    for j in range(i0 + 1, i1):
        h = float(hi[j])
        l = float(lo[j])
        if h >= stop_px and l <= target_px:
            return -float(stop_pts), "stop_tie"
        if h >= stop_px:
            return -float(stop_pts), "stop"
        if l <= target_px:
            return float(target_pts), "target"
    return float(entry - close[i1 - 1]), "timed"


def _trade_table_fixed(
    rows: pd.DataFrame,
    chosen: np.ndarray,
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    stop_pts: float,
    target_pts: float,
    strategy_name: str,
    precomputed_pnl: np.ndarray | None = None,
    precomputed_kind: np.ndarray | None = None,
) -> pd.DataFrame:
    pick = np.asarray(chosen, dtype=bool)
    if len(rows) == 0 or not pick.any():
        return pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])

    work = rows.loc[pick, ["trading_day", "episode_start_idx", "trade_direction", "minute_of_day"]].copy().reset_index(drop=True)
    if precomputed_pnl is not None and precomputed_kind is not None:
        idx = np.flatnonzero(pick)
        pnl = np.asarray(precomputed_pnl, dtype=np.float64)[idx]
        kind = np.asarray(precomputed_kind, dtype=object)[idx]
    else:
        pnl = np.zeros(len(work), dtype=np.float64)
        kind = np.empty(len(work), dtype=object)
        for k in range(len(work)):
            day = str(work.at[k, "trading_day"])
            idx = int(pd.to_numeric(work.at[k, "episode_start_idx"], errors="coerce"))
            tdir = int(pd.to_numeric(work.at[k, "trade_direction"], errors="coerce"))
            if idx < 0 or tdir == 0:
                pnl[k] = 0.0
                kind[k] = "skip"
                continue
            end = int(day_end.get(day, len(close) - 1)) + 1
            entry = float(close[idx])
            pnl[k], kind[k] = _simulate_trade(entry, tdir, float(stop_pts), float(target_pts), high, low, close, idx, end)

    work["pnl"] = pnl
    work["kind"] = kind
    work["strategy"] = str(strategy_name)
    work["lane"] = "fixed"
    work["stop_pts"] = float(stop_pts)
    work["target_pts"] = float(target_pts)
    return work


def _precompute_fixed_outcomes(
    rows: pd.DataFrame,
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    stop_pts: float,
    target_pts: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rows)
    pnl = np.zeros(n, dtype=np.float64)
    kind = np.empty(n, dtype=object)
    for k in range(n):
        day = str(rows.at[k, "trading_day"])
        idx = int(pd.to_numeric(rows.at[k, "episode_start_idx"], errors="coerce"))
        tdir = int(pd.to_numeric(rows.at[k, "trade_direction"], errors="coerce"))
        if idx < 0 or tdir == 0:
            pnl[k] = 0.0
            kind[k] = "skip"
            continue
        end = int(day_end.get(day, len(close) - 1)) + 1
        entry = float(close[idx])
        pnl[k], kind[k] = _simulate_trade(entry, tdir, float(stop_pts), float(target_pts), high, low, close, idx, end)
    return pnl, kind


def _trade_table_mixed(
    rows: pd.DataFrame,
    chosen: np.ndarray,
    *,
    score_q: np.ndarray,
    gate: float,
    stop_hi: float,
    target_hi: float,
    stop_lo: float,
    target_lo: float,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    strategy_name: str,
) -> pd.DataFrame:
    pick = np.asarray(chosen, dtype=bool)
    if len(rows) == 0 or not pick.any():
        return pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])

    idx_all = np.flatnonzero(pick)
    hi_lane = np.asarray(score_q[idx_all] >= float(gate), dtype=bool)
    work = rows.loc[pick, ["trading_day", "episode_start_idx", "trade_direction", "minute_of_day"]].copy().reset_index(drop=True)
    pnl = np.zeros(len(work), dtype=np.float64)
    kind = np.empty(len(work), dtype=object)
    lane = np.empty(len(work), dtype=object)
    stop_used = np.zeros(len(work), dtype=np.float64)
    target_used = np.zeros(len(work), dtype=np.float64)

    for k in range(len(work)):
        day = str(work.at[k, "trading_day"])
        idx = int(pd.to_numeric(work.at[k, "episode_start_idx"], errors="coerce"))
        tdir = int(pd.to_numeric(work.at[k, "trade_direction"], errors="coerce"))
        if hi_lane[k]:
            stp = float(stop_hi)
            tgt = float(target_hi)
            lane[k] = "high"
        else:
            stp = float(stop_lo)
            tgt = float(target_lo)
            lane[k] = "low"
        stop_used[k] = stp
        target_used[k] = tgt
        if idx < 0 or tdir == 0:
            pnl[k] = 0.0
            kind[k] = "skip"
            continue
        end = int(day_end.get(day, len(close) - 1)) + 1
        entry = float(close[idx])
        pnl[k], kind[k] = _simulate_trade(entry, tdir, stp, tgt, high, low, close, idx, end)

    work["pnl"] = pnl
    work["kind"] = kind
    work["strategy"] = str(strategy_name)
    work["lane"] = lane
    work["stop_pts"] = stop_used
    work["target_pts"] = target_used
    return work


def _aggregate_trade_metrics(trades: pd.DataFrame, all_days: List[str]) -> Dict[str, object]:
    days = [str(d) for d in all_days]
    if not days:
        days = sorted(trades["trading_day"].astype(str).unique().tolist()) if not trades.empty else []

    if trades.empty:
        n_days = int(len(days))
        return {
            "mean_selected_per_day_all": 0.0,
            "mean_selected_per_active_day": 0.0,
            "mean_pnl_per_day": 0.0,
            "annualized_daily_sharpe": None,
            "profit_factor": None,
            "mean_pnl_per_trade": None,
            "win_rate": None,
            "profit_day_rate": 0.0 if n_days > 0 else None,
            "no_win_day_rate": 1.0 if n_days > 0 else None,
            "target_rate": None,
            "stop_rate": None,
            "timed_rate": None,
            "high_lane_rate": None,
            "low_lane_rate": None,
            "n_trades": 0,
            "n_days": n_days,
            "zero_trade_day_rate": 1.0 if n_days > 0 else None,
            "no_target_day_rate": 1.0 if n_days > 0 else None,
            "total_pnl": 0.0,
        }

    t = trades.copy()
    t["trading_day"] = t["trading_day"].astype(str)
    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce").fillna(0.0)
    if "kind" not in t.columns:
        t["kind"] = "timed"
    if "lane" not in t.columns:
        t["lane"] = "fixed"

    day_g = t.groupby("trading_day", sort=False)
    day_df = pd.DataFrame(
        {
            "n": day_g.size(),
            "pnl": day_g["pnl"].sum(),
            "wins": day_g["pnl"].apply(lambda s: int((s > 0).sum())),
            "targets": day_g["kind"].apply(lambda s: int((s == "target").sum())),
            "gp": day_g["pnl"].apply(lambda s: float(s.clip(lower=0.0).sum())),
            "gl": day_g["pnl"].apply(lambda s: float((-s.clip(upper=0.0)).sum())),
        }
    )

    if days:
        idx = pd.Index(days, dtype=object)
    else:
        idx = day_df.index
    day_df = day_df.reindex(idx, fill_value=0.0)
    day_df["n"] = pd.to_numeric(day_df["n"], errors="coerce").fillna(0).astype(int)
    day_df["wins"] = pd.to_numeric(day_df["wins"], errors="coerce").fillna(0).astype(int)
    day_df["targets"] = pd.to_numeric(day_df["targets"], errors="coerce").fillna(0).astype(int)

    daily = day_df["pnl"].to_numpy(dtype=np.float64, copy=False)
    mean_day = float(np.mean(daily)) if daily.size else 0.0
    std_day = float(np.std(daily, ddof=1)) if daily.size > 1 else 0.0
    sharpe = float((mean_day / std_day) * np.sqrt(252.0)) if std_day > 1e-9 else None
    gp = float(day_df["gp"].sum())
    gl = float(day_df["gl"].sum())
    pf = (gp / gl) if gl > 1e-9 else None

    n_trades = int(len(t))
    total_pnl = float(t["pnl"].sum())
    n_active_days = int((day_df["n"] > 0).sum())
    n_days = int(len(day_df))
    return {
        "mean_selected_per_day_all": float(day_df["n"].mean()) if n_days > 0 else 0.0,
        "mean_selected_per_active_day": float(day_df.loc[day_df["n"] > 0, "n"].mean()) if n_active_days > 0 else 0.0,
        "mean_pnl_per_day": mean_day,
        "annualized_daily_sharpe": sharpe,
        "profit_factor": pf,
        "mean_pnl_per_trade": (total_pnl / n_trades) if n_trades > 0 else None,
        "win_rate": float((t["pnl"] > 0).mean()) if n_trades > 0 else None,
        "profit_day_rate": float((day_df["pnl"] > 0).mean()) if n_days > 0 else None,
        "no_win_day_rate": float((day_df["wins"] == 0).mean()) if n_days > 0 else None,
        "target_rate": float((t["kind"] == "target").mean()) if n_trades > 0 else None,
        "stop_rate": float(np.isin(t["kind"], ["stop", "stop_tie"]).mean()) if n_trades > 0 else None,
        "timed_rate": float((t["kind"] == "timed").mean()) if n_trades > 0 else None,
        "high_lane_rate": float((t["lane"] == "high").mean()) if n_trades > 0 else None,
        "low_lane_rate": float((t["lane"] == "low").mean()) if n_trades > 0 else None,
        "n_trades": n_trades,
        "n_days": n_days,
        "zero_trade_day_rate": float((day_df["n"] == 0).mean()) if n_days > 0 else None,
        "no_target_day_rate": float((day_df["targets"] == 0).mean()) if n_days > 0 else None,
        "total_pnl": total_pnl,
    }


def _trend_policy_grid() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for floor in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for gap in [0, 5, 10]:
            variant = f"trend_pullback_v3_f{int(round(floor * 100))}_g{int(gap)}"
            for stop in [4.0, 5.0, 6.0, 8.0, 10.0]:
                for target in [8.0, 10.0, 12.0, 15.0, 18.0, 24.0, 30.0, 40.0]:
                    if target <= stop:
                        continue
                    out.append(
                        {
                            "variant": variant,
                            "floor": float(floor),
                            "gap": int(gap),
                            "stop_pts": float(stop),
                            "target_pts": float(target),
                        }
                    )
    return out


def _choose_trend_rows(rows: pd.DataFrame, cfg: Dict[str, object], *, k: int) -> np.ndarray:
    return _simulate_diverse_first_k(
        rows,
        "score",
        k=int(k),
        floor=float(cfg["floor"]),
        gap_minutes=int(cfg["gap"]),
    )


def _choose_reversal_rows(rows: pd.DataFrame, cfg: Dict[str, object]) -> np.ndarray:
    return _simulate_router(
        rows,
        _build_day_groups(rows),
        quality_col=str(cfg["quality_col"]),
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


def _trend_sweep_oof(
    cand: pd.DataFrame,
    *,
    ohlcv: pd.DataFrame,
    k: int,
) -> List[Dict[str, object]]:
    all_days = sorted(ohlcv["trading_day"].astype(str).unique().tolist())
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)

    grid = _trend_policy_grid()
    unique_exec = sorted({(float(g["stop_pts"]), float(g["target_pts"])) for g in grid})
    precomputed: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
    for stop, target in unique_exec:
        precomputed[(stop, target)] = _precompute_fixed_outcomes(
            cand,
            close=close,
            high=high,
            low=low,
            day_end=day_end,
            stop_pts=float(stop),
            target_pts=float(target),
        )

    out: List[Dict[str, object]] = []
    for cfg in grid:
        chosen = _choose_trend_rows(cand, cfg, k=int(k))
        pnl_arr, kind_arr = precomputed[(float(cfg["stop_pts"]), float(cfg["target_pts"]))]
        trades = _trade_table_fixed(
            cand,
            chosen,
            close=close,
            high=high,
            low=low,
            day_end=day_end,
            stop_pts=float(cfg["stop_pts"]),
            target_pts=float(cfg["target_pts"]),
            strategy_name="trend_pullback",
            precomputed_pnl=pnl_arr,
            precomputed_kind=kind_arr,
        )
        met = _aggregate_trade_metrics(trades, all_days)
        out.append({**cfg, "metrics": met})
    return out


def _fold_splits(days: List[str], n_folds: int, min_train_days: int) -> List[Tuple[List[str], List[str]]]:
    return tlm.get_fold_splits(days, int(n_folds), int(min_train_days))


def _trend_fold_validation(
    cand: pd.DataFrame,
    cfgs: Dict[str, Dict[str, object]],
    *,
    ohlcv: pd.DataFrame,
    n_folds: int,
    min_train_days: int,
    k: int,
) -> Dict[str, object]:
    all_days = sorted(ohlcv["trading_day"].astype(str).unique().tolist())
    splits = _fold_splits(all_days, int(n_folds), int(min_train_days))
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)

    out: Dict[str, object] = {}
    for name, cfg in cfgs.items():
        folds: List[Dict[str, object]] = []
        for fidx, (_, test_days) in enumerate(splits):
            day_set = set(test_days)
            sub = cand.loc[cand["trading_day"].astype(str).isin(day_set)].copy().reset_index(drop=True)
            chosen = _choose_trend_rows(sub, cfg, k=int(k))
            trades = _trade_table_fixed(
                sub,
                chosen,
                close=close,
                high=high,
                low=low,
                day_end=day_end,
                stop_pts=float(cfg["stop_pts"]),
                target_pts=float(cfg["target_pts"]),
                strategy_name=f"trend_pullback:{name}",
            )
            met = _aggregate_trade_metrics(trades, sorted(day_set))
            folds.append({"fold": int(fidx), "n_days": int(len(day_set)), "metrics": met})
        out[name] = {
            "aggregate": {
                "mean_pf": _mean([_safe_float(f["metrics"].get("profit_factor"), np.nan) for f in folds]),
                "mean_sharpe": _mean([_safe_float(f["metrics"].get("annualized_daily_sharpe"), np.nan) for f in folds]),
                "mean_no_target_day_rate": _mean([_safe_float(f["metrics"].get("no_target_day_rate"), np.nan) for f in folds]),
                "mean_zero_trade_day_rate": _mean([_safe_float(f["metrics"].get("zero_trade_day_rate"), np.nan) for f in folds]),
                "mean_selected_per_day_all": _mean([_safe_float(f["metrics"].get("mean_selected_per_day_all"), np.nan) for f in folds]),
                "mean_target_rate": _mean([_safe_float(f["metrics"].get("target_rate"), np.nan) for f in folds]),
            },
            "folds": folds,
        }
    return out


def _reversal_oof_tables(
    rows: pd.DataFrame,
    cfgs: Dict[str, Dict[str, object]],
    *,
    ohlcv: pd.DataFrame,
) -> Dict[str, Dict[str, object]]:
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)
    all_days = sorted(ohlcv["trading_day"].astype(str).unique().tolist())

    out: Dict[str, Dict[str, object]] = {}
    for name, cfg in cfgs.items():
        chosen = _choose_reversal_rows(rows, cfg)
        score_q = pd.to_numeric(rows[cfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
        trades = _trade_table_mixed(
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
            strategy_name=f"reversal:{name}",
        )
        out[name] = {
            "trades": trades,
            "metrics": _aggregate_trade_metrics(trades, all_days),
        }
    return out


def _conjunction_oof(
    rev_rows: pd.DataFrame,
    rev_cfgs: Dict[str, Dict[str, object]],
    trend_cand: pd.DataFrame,
    trend_cfgs: Dict[str, Dict[str, object]],
    *,
    ohlcv: pd.DataFrame,
    k_trend: int,
) -> Dict[str, object]:
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)
    all_days = sorted(ohlcv["trading_day"].astype(str).unique().tolist())

    out: Dict[str, object] = {}
    for rname, rcfg in rev_cfgs.items():
        rev_chosen = _choose_reversal_rows(rev_rows, rcfg)
        rev_score = pd.to_numeric(rev_rows[rcfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
        rev_trades = _trade_table_mixed(
            rev_rows,
            rev_chosen,
            score_q=rev_score,
            gate=float(rcfg["exec"]["gate"]),
            stop_hi=float(rcfg["exec"]["high_stop"]),
            target_hi=float(rcfg["exec"]["high_target"]),
            stop_lo=float(rcfg["exec"]["low_stop"]),
            target_lo=float(rcfg["exec"]["low_target"]),
            close=close,
            high=high,
            low=low,
            day_end=day_end,
            strategy_name=f"reversal:{rname}",
        )
        for tname, tcfg in trend_cfgs.items():
            trend_chosen = _choose_trend_rows(trend_cand, tcfg, k=int(k_trend))
            trend_trades = _trade_table_fixed(
                trend_cand,
                trend_chosen,
                close=close,
                high=high,
                low=low,
                day_end=day_end,
                stop_pts=float(tcfg["stop_pts"]),
                target_pts=float(tcfg["target_pts"]),
                strategy_name=f"trend:{tname}",
            )
            combo = pd.concat([rev_trades, trend_trades], axis=0, ignore_index=True, sort=False)
            overlap = int(
                pd.merge(
                    rev_trades[["trading_day", "episode_start_idx", "trade_direction"]],
                    trend_trades[["trading_day", "episode_start_idx", "trade_direction"]],
                    on=["trading_day", "episode_start_idx", "trade_direction"],
                    how="inner",
                ).shape[0]
            )
            key = f"{rname}__{tname}"
            out[key] = {
                "reversal_cfg": rname,
                "trend_cfg": tname,
                "metrics": _aggregate_trade_metrics(combo, all_days),
                "component": {
                    "reversal": _aggregate_trade_metrics(rev_trades, all_days),
                    "trend": _aggregate_trade_metrics(trend_trades, all_days),
                },
                "overlap_same_signal_count": overlap,
                "n_combo_trades": int(len(combo)),
            }
    return out


def _conjunction_fold_validation(
    rev_rows: pd.DataFrame,
    rev_cfgs: Dict[str, Dict[str, object]],
    trend_cand: pd.DataFrame,
    trend_cfgs: Dict[str, Dict[str, object]],
    *,
    ohlcv: pd.DataFrame,
    n_folds: int,
    min_train_days: int,
    k_trend: int,
) -> Dict[str, object]:
    all_days = sorted(ohlcv["trading_day"].astype(str).unique().tolist())
    splits = _fold_splits(all_days, int(n_folds), int(min_train_days))
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(ohlcv)

    out: Dict[str, object] = {}
    for rname, rcfg in rev_cfgs.items():
        for tname, tcfg in trend_cfgs.items():
            key = f"{rname}__{tname}"
            folds: List[Dict[str, object]] = []
            for fidx, (_, test_days) in enumerate(splits):
                day_set = set(test_days)
                rev_sub = rev_rows.loc[rev_rows["trading_day"].astype(str).isin(day_set)].copy().reset_index(drop=True)
                trend_sub = trend_cand.loc[trend_cand["trading_day"].astype(str).isin(day_set)].copy().reset_index(drop=True)

                if rev_sub.empty:
                    rev_trades = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                else:
                    rev_chosen = _choose_reversal_rows(rev_sub, rcfg)
                    rev_score = pd.to_numeric(rev_sub[rcfg["quality_col"]], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
                    rev_trades = _trade_table_mixed(
                        rev_sub,
                        rev_chosen,
                        score_q=rev_score,
                        gate=float(rcfg["exec"]["gate"]),
                        stop_hi=float(rcfg["exec"]["high_stop"]),
                        target_hi=float(rcfg["exec"]["high_target"]),
                        stop_lo=float(rcfg["exec"]["low_stop"]),
                        target_lo=float(rcfg["exec"]["low_target"]),
                        close=close,
                        high=high,
                        low=low,
                        day_end=day_end,
                        strategy_name=f"reversal:{rname}",
                    )

                if trend_sub.empty:
                    trend_trades = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                else:
                    trend_chosen = _choose_trend_rows(trend_sub, tcfg, k=int(k_trend))
                    trend_trades = _trade_table_fixed(
                        trend_sub,
                        trend_chosen,
                        close=close,
                        high=high,
                        low=low,
                        day_end=day_end,
                        stop_pts=float(tcfg["stop_pts"]),
                        target_pts=float(tcfg["target_pts"]),
                        strategy_name=f"trend:{tname}",
                    )

                combo = pd.concat([rev_trades, trend_trades], axis=0, ignore_index=True, sort=False)
                met = _aggregate_trade_metrics(combo, sorted(day_set))
                folds.append({"fold": int(fidx), "n_days": int(len(day_set)), "metrics": met})

            out[key] = {
                "aggregate": {
                    "mean_pf": _mean([_safe_float(f["metrics"].get("profit_factor"), np.nan) for f in folds]),
                    "mean_sharpe": _mean([_safe_float(f["metrics"].get("annualized_daily_sharpe"), np.nan) for f in folds]),
                    "mean_no_target_day_rate": _mean([_safe_float(f["metrics"].get("no_target_day_rate"), np.nan) for f in folds]),
                    "mean_zero_trade_day_rate": _mean([_safe_float(f["metrics"].get("zero_trade_day_rate"), np.nan) for f in folds]),
                    "mean_selected_per_day_all": _mean([_safe_float(f["metrics"].get("mean_selected_per_day_all"), np.nan) for f in folds]),
                    "mean_target_rate": _mean([_safe_float(f["metrics"].get("target_rate"), np.nan) for f in folds]),
                },
                "folds": folds,
            }
    return out


def _prepare_sql_day(
    day: str,
    *,
    history_csv: str,
    warmup_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    history = _load_causal_history(history_csv, day, int(warmup_days))
    day_bars = _normalize_sql_day(day, _load_day_bars(day))
    combined = pd.concat([history, day_bars], axis=0, ignore_index=True, sort=False)
    ohlcv, base_cols = _build_ohlcv_and_features_for_combined(combined)
    ohlcv, dyn_cols = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv)
    all_cols = list(dict.fromkeys(list(base_cols) + list(dyn_cols)))
    episodes, _ = _episode_rows(ohlcv, all_cols, gap_bars=10)
    ep_day = episodes.loc[episodes["trading_day"].astype(str) == str(day)].copy().reset_index(drop=True)
    return ohlcv, ep_day, all_cols


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
    for col in full_feature_cols:
        if col not in day_cand.columns:
            day_cand[col] = 0.0
    X = day_cand.loc[:, full_feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    day_cand["score"] = full_model.predict_proba(X)[:, 1].astype(np.float64)
    return day_cand


def _sql_week_eval(
    sql_days: List[str],
    *,
    history_csv: str,
    warmup_days: int,
    rev_cfgs: Dict[str, Dict[str, object]],
    trend_cfgs: Dict[str, Dict[str, object]],
    rev_heads: Dict[str, object],
    trend_model,
    trend_feature_cols: List[str],
    k_trend: int,
) -> Dict[str, object]:
    rev_trades_by_cfg: Dict[str, List[pd.DataFrame]] = {k: [] for k in rev_cfgs.keys()}
    trend_trades_by_cfg: Dict[str, List[pd.DataFrame]] = {k: [] for k in trend_cfgs.keys()}
    combo_trades_by_pair: Dict[str, List[pd.DataFrame]] = {
        f"{rk}__{tk}": [] for rk in rev_cfgs.keys() for tk in trend_cfgs.keys()
    }
    per_day: Dict[str, object] = {}

    for day in sql_days:
        try:
            ohlcv, ep_day, _ = _prepare_sql_day(day, history_csv=history_csv, warmup_days=int(warmup_days))
            if ep_day.empty:
                per_day[str(day)] = {"note": "no_episode_rows"}
                continue
            close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            day_end = _build_rth_day_end(ohlcv)

            rev_rows = _score_sql_reversal_rows(ep_day.copy(), rev_heads)
            trend_rows = _score_sql_trend_rows(ep_day.copy(), full_model=trend_model, full_feature_cols=trend_feature_cols)
            day_meta: Dict[str, object] = {
                "n_episode_rows": int(len(ep_day)),
                "n_reversal_rows": int(len(rev_rows)),
                "n_trend_rows": int(len(trend_rows)),
                "reversal": {},
                "trend": {},
                "conjunction": {},
            }

            for rname, rcfg in rev_cfgs.items():
                if rev_rows.empty:
                    rtr = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                else:
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
                        close=close,
                        high=high,
                        low=low,
                        day_end=day_end,
                        strategy_name=f"reversal:{rname}",
                    )
                rev_trades_by_cfg[rname].append(rtr)
                day_meta["reversal"][rname] = _aggregate_trade_metrics(rtr, [str(day)])

            for tname, tcfg in trend_cfgs.items():
                if trend_rows.empty:
                    ttr = pd.DataFrame(columns=["trading_day", "episode_start_idx", "trade_direction", "minute_of_day", "pnl", "kind", "strategy", "lane", "stop_pts", "target_pts"])
                else:
                    tchosen = _choose_trend_rows(trend_rows, tcfg, k=int(k_trend))
                    ttr = _trade_table_fixed(
                        trend_rows,
                        tchosen,
                        close=close,
                        high=high,
                        low=low,
                        day_end=day_end,
                        stop_pts=float(tcfg["stop_pts"]),
                        target_pts=float(tcfg["target_pts"]),
                        strategy_name=f"trend:{tname}",
                    )
                trend_trades_by_cfg[tname].append(ttr)
                day_meta["trend"][tname] = _aggregate_trade_metrics(ttr, [str(day)])

            for rname in rev_cfgs.keys():
                for tname in trend_cfgs.keys():
                    key = f"{rname}__{tname}"
                    combo = pd.concat([rev_trades_by_cfg[rname][-1], trend_trades_by_cfg[tname][-1]], axis=0, ignore_index=True, sort=False)
                    combo_trades_by_pair[key].append(combo)
                    day_meta["conjunction"][key] = _aggregate_trade_metrics(combo, [str(day)])

            per_day[str(day)] = _to_jsonable(day_meta)
        except Exception as exc:
            per_day[str(day)] = {"error": str(exc)}

    rev_agg = {
        name: _aggregate_trade_metrics(pd.concat(lst, axis=0, ignore_index=True, sort=False) if lst else pd.DataFrame(), sql_days)
        for name, lst in rev_trades_by_cfg.items()
    }
    trend_agg = {
        name: _aggregate_trade_metrics(pd.concat(lst, axis=0, ignore_index=True, sort=False) if lst else pd.DataFrame(), sql_days)
        for name, lst in trend_trades_by_cfg.items()
    }
    combo_agg = {
        name: _aggregate_trade_metrics(pd.concat(lst, axis=0, ignore_index=True, sort=False) if lst else pd.DataFrame(), sql_days)
        for name, lst in combo_trades_by_pair.items()
    }
    return {
        "aggregate": {
            "reversal": _to_jsonable(rev_agg),
            "trend": _to_jsonable(trend_agg),
            "conjunction": _to_jsonable(combo_agg),
        },
        "per_day": _to_jsonable(per_day),
    }


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
    ap.add_argument("--top-trend", type=int, default=3)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--score-rows", default="sandbox/results/upstream_two_stage_constrained_v1_smallgrid_scores.parquet")
    ap.add_argument("--scored-rows", default="sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet")
    ap.add_argument("--reversal-ohlcv", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--output-json", default="sandbox/results/trend_pullback_standalone_conjunction_v1.json")
    args = ap.parse_args()
    _require_deps()

    t0 = time.time()
    sql_days = [x.strip() for x in str(args.sql_days).split(",") if x.strip()]

    print("Loading base dataset + trend candidates...")
    ohlcv, base_cols = _build_ohlcv_and_features(args.data)
    ohlcv, dyn_cols = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv)
    base_cols = list(dict.fromkeys(list(base_cols) + list(dyn_cols)))
    episodes, _ = _episode_rows(ohlcv, base_cols, gap_bars=10)
    cand = _trend_pullback_candidates_v3(episodes)
    trend_feature_cols = _pullback_features_v3(cand)

    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)

    print("Building trend training target + OOF scores...")
    cand["cont_target_train"] = _barrier_label(
        cand,
        close,
        high,
        low,
        stop_pts=float(args.trend_train_stop),
        target_pts=float(args.trend_train_target),
        forward_bars=int(args.forward_bars),
    )
    oof_score = _walk_forward_scores(
        cand,
        trend_feature_cols,
        "cont_target_train",
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        seed=int(args.seed),
    )
    valid = np.isfinite(oof_score)
    cand = cand.loc[valid].copy().reset_index(drop=True)
    cand["score"] = oof_score[valid]

    print("Trend OOF sweep...")
    trend_grid = _trend_sweep_oof(cand, ohlcv=ohlcv, k=int(args.k_trend))
    trend_grid_sorted = sorted(trend_grid, key=_score_sort_key)
    trend_feasible = [
        r
        for r in trend_grid_sorted
        if _safe_float(r["metrics"].get("mean_selected_per_day_all"), 0.0) >= 1.0
        and _safe_float(r["metrics"].get("mean_selected_per_day_all"), 0.0) <= 10.0
    ]
    top_trend_rows = (trend_feasible if trend_feasible else trend_grid_sorted)[: int(args.top_trend)]
    top_trend_cfgs = {
        f"trend_top{i+1}_{row['variant']}_s{int(row['stop_pts'])}_t{int(row['target_pts'])}": {
            "variant": row["variant"],
            "floor": float(row["floor"]),
            "gap": int(row["gap"]),
            "stop_pts": float(row["stop_pts"]),
            "target_pts": float(row["target_pts"]),
        }
        for i, row in enumerate(top_trend_rows)
    }

    print("Trend full model fit...")
    full_cand = _trend_pullback_candidates_v3(episodes)
    full_cand["cont_target_train"] = _barrier_label(
        full_cand,
        close,
        high,
        low,
        stop_pts=float(args.trend_train_stop),
        target_pts=float(args.trend_train_target),
        forward_bars=int(args.forward_bars),
    )
    full_trend_features = _pullback_features_v3(full_cand)
    trend_model = _fit_full_model(full_cand, full_trend_features, "cont_target_train", seed=int(args.seed))

    print("Reversal frontier load...")
    rev_cfgs = _frontier_configs()
    rev_rows = _load_rows_with_scores(args.rows, args.score_rows, args.scored_rows)
    rev_ohlcv = pd.read_parquet(args.reversal_ohlcv).copy()
    if "_src_index" in rev_ohlcv.columns:
        rev_ohlcv = rev_ohlcv.sort_values("_src_index").reset_index(drop=True)

    print("OOF standalone + conjunction...")
    rev_oof = _reversal_oof_tables(rev_rows, rev_cfgs, ohlcv=rev_ohlcv)
    trend_fold = _trend_fold_validation(
        cand,
        top_trend_cfgs,
        ohlcv=ohlcv,
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        k=int(args.k_trend),
    )
    combo_oof = _conjunction_oof(
        rev_rows,
        rev_cfgs,
        cand,
        top_trend_cfgs,
        ohlcv=rev_ohlcv,
        k_trend=int(args.k_trend),
    )
    combo_fold = _conjunction_fold_validation(
        rev_rows,
        rev_cfgs,
        cand,
        top_trend_cfgs,
        ohlcv=rev_ohlcv,
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        k_trend=int(args.k_trend),
    )

    if sql_days:
        print("SQL week scoring (full models)...")
        rev_heads = _train_full_frontier_heads(args.data, seed=int(args.seed))
        sql_out = _sql_week_eval(
            sql_days,
            history_csv=args.history_csv,
            warmup_days=int(args.warmup_days),
            rev_cfgs=rev_cfgs,
            trend_cfgs=top_trend_cfgs,
            rev_heads=rev_heads,
            trend_model=trend_model,
            trend_feature_cols=full_trend_features,
            k_trend=int(args.k_trend),
        )
    else:
        sql_out = {"note": "sql_week_skipped", "aggregate": {"reversal": {}, "trend": {}, "conjunction": {}}, "per_day": {}}

    y = pd.to_numeric(cand["cont_target_train"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    s = pd.to_numeric(cand["score"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    pos = np.isfinite(y) & np.isfinite(s) & (y > 0.5)
    neg = np.isfinite(y) & np.isfinite(s) & (y <= 0.5)
    trend_sep = {
        "n_rows": int(len(cand)),
        "pos_rate": float(np.mean(np.isfinite(y) & (y > 0.5))),
        "score_mean_pos": float(np.mean(s[pos])) if pos.any() else None,
        "score_mean_neg": float(np.mean(s[neg])) if neg.any() else None,
        "score_q10_pos": float(np.quantile(s[pos], 0.10)) if pos.any() else None,
        "score_q90_neg": float(np.quantile(s[neg], 0.90)) if neg.any() else None,
    }

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "meta": {
            "data": str(args.data),
            "history_csv": str(args.history_csv),
            "sql_days": sql_days,
            "k_trend": int(args.k_trend),
            "trend_train_stop": float(args.trend_train_stop),
            "trend_train_target": float(args.trend_train_target),
            "forward_bars": int(args.forward_bars),
            "note": "Conjunction uses additive independent books (no mixed routing policy).",
        },
        "trend_standalone": {
            "oof_separability": _to_jsonable(trend_sep),
            "oof_top_20": _to_jsonable(trend_grid_sorted[:20]),
            "oof_top_feasible_20": _to_jsonable(trend_feasible[:20]),
            "selected_top_trend_cfgs": _to_jsonable(top_trend_cfgs),
            "fold_validation": _to_jsonable(trend_fold),
        },
        "reversal_standalone_oof": _to_jsonable({k: v["metrics"] for k, v in rev_oof.items()}),
        "conjunction_oof": _to_jsonable(combo_oof),
        "conjunction_fold_validation": _to_jsonable(combo_fold),
        "sql_week": _to_jsonable(sql_out),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(out), indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
