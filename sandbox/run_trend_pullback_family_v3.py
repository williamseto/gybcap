#!/usr/bin/env python
"""Trend-pullback standalone family v3 (relaxed candidates + sidecar readiness).

Goal:
- spin out trend+pullback as its own standalone model/strategy artifact
- run the same OOF + SQL-day evaluation pattern used in prior reversal research
"""

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
    from sandbox.run_execution_pf_sweep_v1 import _build_sql_day_rows_and_ohlcv, _qualifies, _score_selected_trades, _simulate_diverse_first_k
    from sandbox.run_fp_reduction_followon import _to_jsonable
    from sandbox.run_global_episode_model_experiments import _build_ohlcv_and_features, _episode_rows
    from sandbox.run_trend_pullback_family_v1 import _barrier_label, _fit_full_model, _rth_mask, _walk_forward_scores
    from sandbox.run_trend_pullback_family_v2 import _augment_ohlcv_with_dynamic_pullback_columns
except ModuleNotFoundError as exc:  # pragma: no cover - compatibility guard
    _MISSING_IMPORT = exc


def _require_deps() -> None:
    if _MISSING_IMPORT is None:
        return
    raise RuntimeError(
        "run_trend_pullback_family_v3.py depends on archived sandbox modules that were pruned. "
        "Restore required scripts from graveyard before running this campaign."
    ) from _MISSING_IMPORT


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _trend_proxy_dir_v3(df: pd.DataFrame) -> pd.Series:
    od = _safe_num(df, "opening_drive_regime_dir")
    strength = _safe_num(df, "session_trend_strength")
    dist_open = _safe_num(df, "dist_from_rth_open_pts")
    proxy = od.copy()
    z = proxy.eq(0.0)
    proxy.loc[z] = np.sign(strength.loc[z])
    z2 = proxy.eq(0.0)
    proxy.loc[z2] = np.sign(dist_open.loc[z2])
    return proxy


def _trend_pullback_candidates_v3(episodes: pd.DataFrame) -> pd.DataFrame:
    df = episodes.loc[_rth_mask(episodes)].copy()
    trend_dir = _trend_proxy_dir_v3(df)
    trdir = _safe_num(df, "trade_direction")
    strength = _safe_num(df, "session_trend_strength")
    minute = _safe_num(df, "minute_of_day", default=-1)
    ext_open = _safe_num(df, "trend_extension_from_open_pts")
    ext_vwap = _safe_num(df, "trend_extension_from_vwap_pts")
    pb_rth = _safe_num(df, "pullback_ratio_rth_leg")
    pb_drive = _safe_num(df, "pullback_ratio_drive_leg")
    dist_anchor = _safe_num(df, "dist_to_dynamic_anchor")
    anchor_cluster = _safe_num(df, "dynamic_anchor_cluster_4pt")
    piv20 = _safe_num(df, "pivot_pull_ratio_20")
    piv60 = _safe_num(df, "pivot_pull_ratio_60")
    dist_p20 = _safe_num(df, "dist_to_pivot20")
    dist_p60 = _safe_num(df, "dist_to_pivot60")
    dist_open = _safe_num(df, "dist_from_rth_open_pts")

    align = trdir.eq(trend_dir)
    mask = (
        trend_dir.ne(0.0)
        & align
        & minute.ge(395)
        & strength.abs().ge(0.03)
        & ext_open.ge(0.5)
        & ext_vwap.between(-8.0, 30.0)
        & pb_rth.between(0.0, 1.20)
        & pb_drive.between(0.0, 1.80)
        & dist_anchor.le(12.0)
        & anchor_cluster.ge(0.0)
        & piv20.between(0.0, 1.50)
        & piv60.between(0.0, 1.80)
        & ((dist_p20.le(12.0)) | (dist_p60.le(14.0)))
    )
    # Avoid extreme late-session chasing.
    mask &= minute.lt(760)
    # Require some directional extension from the open.
    mask &= (dist_open * trend_dir).ge(-2.0)

    out = df.loc[mask].copy().reset_index(drop=True)
    out["trend_proxy_dir_v3"] = trend_dir.loc[mask].to_numpy(dtype=np.float64, copy=False)
    return out


def _pullback_features_v3(df: pd.DataFrame) -> List[str]:
    cols = [
        "trend_proxy_dir_v3",
        "opening_drive_regime_dir",
        "opening_drive_net_pts",
        "opening_drive_frac",
        "session_trend_strength",
        "trend_extension_from_open_pts",
        "trend_extension_from_vwap_pts",
        "pullback_from_extreme_pts",
        "pullback_ratio_rth_leg",
        "pullback_ratio_drive_leg",
        "dist_to_dynamic_anchor",
        "dynamic_anchor_cluster_2pt",
        "dynamic_anchor_cluster_4pt",
        "pivot_pull_ratio_20",
        "pivot_pull_ratio_60",
        "dist_to_pivot20",
        "dist_to_pivot60",
        "dist_to_retr_38",
        "dist_to_retr_50",
        "dist_to_retr_62",
        "dist_to_rth_mid_dyn",
        "dist_from_vwap_pts",
        "dist_from_rth_open_pts",
        "current_run_bars",
        "consecutive_same_dir",
        "bar_momentum",
        "rejection_penetration",
        "vol_ratio_rejection",
        "bars_to_nearest_level",
        "rth_level_firmness",
        "level_side_touch_density_30",
        "mae_estimate",
        "level_side_bb_interaction",
        "minute_of_day",
    ]
    cols.extend(sorted([c for c in df.columns if c.startswith("is_")]))
    return [c for c in cols if c in df.columns]


def _sort_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        rows,
        key=lambda r: (
            -(r["profit_factor"] if r["profit_factor"] is not None else -1e9),
            -(r["annualized_daily_sharpe"] if r["annualized_daily_sharpe"] is not None else -1e9),
            -(r["mean_pnl_per_day"]),
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="raw_data/es_min_3y_clean_td_gamma.csv")
    p.add_argument("--history-csv", default="raw_data/schwab/es_minute_history.csv")
    p.add_argument("--sql-days", default="2026-02-24,2026-02-25,2026-02-26,2026-02-27,2026-03-02")
    p.add_argument("--warmup-days", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train-days", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--forward-bars", type=int, default=60)
    p.add_argument("--train-stop", type=float, default=4.0)
    p.add_argument("--train-target", type=float, default=10.0)
    p.add_argument("--output-json", default="sandbox/results/trend_pullback_family_v3.json")
    args = p.parse_args()
    _require_deps()

    t0 = time.time()
    sql_days = [x.strip() for x in str(args.sql_days).split(",") if x.strip()]

    print("Loading OOF structural dataset...")
    ohlcv, base_feature_cols = _build_ohlcv_and_features(args.data)
    ohlcv, dyn_cols = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv)
    base_feature_cols = list(dict.fromkeys(list(base_feature_cols) + list(dyn_cols)))
    episodes, feature_cols_all = _episode_rows(ohlcv, base_feature_cols, gap_bars=10)
    cand = _trend_pullback_candidates_v3(episodes)
    feature_cols = _pullback_features_v3(cand)

    print("Building continuation target...")
    close = ohlcv["close"].to_numpy(dtype=np.float64, copy=False)
    high = ohlcv["high"].to_numpy(dtype=np.float64, copy=False)
    low = ohlcv["low"].to_numpy(dtype=np.float64, copy=False)
    cand["cont_target_train"] = _barrier_label(
        cand,
        close,
        high,
        low,
        stop_pts=float(args.train_stop),
        target_pts=float(args.train_target),
        forward_bars=int(args.forward_bars),
    )

    print("Training walk-forward continuation model...")
    score_oof = _walk_forward_scores(
        cand,
        feature_cols,
        "cont_target_train",
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        seed=int(args.seed),
    )
    cand = cand.loc[np.isfinite(score_oof)].copy().reset_index(drop=True)
    cand["score"] = score_oof[np.isfinite(score_oof)]

    variants = []
    for floor in [0.25, 0.30, 0.35, 0.40, 0.45]:
        for gap in [0, 5, 10]:
            variants.append((f"trend_pullback_v3_f{int(round(floor*100))}_g{gap}", "score", floor, gap))
    stops = [4.0, 5.0, 6.0, 8.0]
    targets = [8.0, 10.0, 12.0, 15.0, 18.0, 24.0]

    print("Sweeping OOF execution parameters...")
    oof_grid: List[Dict[str, object]] = []
    qualified: List[Dict[str, object]] = []
    for name, score_col, floor, gap in variants:
        chosen = _simulate_diverse_first_k(cand, score_col, k=int(args.k), floor=float(floor), gap_minutes=int(gap))
        for stop in stops:
            for target in targets:
                if target <= stop:
                    continue
                m = _score_selected_trades(
                    cand,
                    chosen,
                    close=close,
                    high=high,
                    low=low,
                    stop_pts=float(stop),
                    target_pts=float(target),
                    forward_bars=int(args.forward_bars),
                )
                row = {"variant": name, "stop_pts": float(stop), "target_pts": float(target), **m}
                oof_grid.append(row)
                if _qualifies(m):
                    qualified.append(row)

    qualified_sorted = _sort_rows(qualified)
    top_pf = _sort_rows(oof_grid)[:20]
    best_oof = qualified_sorted[0] if qualified_sorted else top_pf[0]

    print("Training full continuation model...")
    full_cand = _trend_pullback_candidates_v3(episodes)
    full_cand["cont_target_train"] = _barrier_label(
        full_cand,
        close,
        high,
        low,
        stop_pts=float(args.train_stop),
        target_pts=float(args.train_target),
        forward_bars=int(args.forward_bars),
    )
    full_feature_cols = _pullback_features_v3(full_cand)
    full_model = _fit_full_model(full_cand, full_feature_cols, "cont_target_train", seed=int(args.seed))

    print("Evaluating top configs on external SQL days...")
    top_candidates = qualified_sorted[:10] if qualified_sorted else top_pf[:10]
    per_sql_day: Dict[str, Dict[str, object]] = {}
    for day in sql_days:
        _, ohlcv_day = _build_sql_day_rows_and_ohlcv(
            day,
            history_csv=args.history_csv,
            warmup_days=int(args.warmup_days),
            feature_cols=feature_cols_all,
        )
        if ohlcv_day.empty:
            per_sql_day[str(day)] = {"note": "no_rows"}
            continue
        ohlcv_day, _ = _augment_ohlcv_with_dynamic_pullback_columns(ohlcv_day)
        ep_day, _ = _episode_rows(ohlcv_day, base_feature_cols, gap_bars=10)
        ep_day = ep_day.loc[ep_day["trading_day"].astype(str) == str(day)].copy().reset_index(drop=True)
        day_cand = _trend_pullback_candidates_v3(ep_day)
        if day_cand.empty:
            per_sql_day[str(day)] = {"note": "no_candidates"}
            continue
        for col in full_feature_cols:
            if col not in day_cand.columns:
                day_cand[col] = 0.0
        X_day = day_cand.loc[:, full_feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        day_cand["score"] = full_model.predict_proba(X_day)[:, 1].astype(np.float64)
        close_d = ohlcv_day["close"].to_numpy(dtype=np.float64, copy=False)
        high_d = ohlcv_day["high"].to_numpy(dtype=np.float64, copy=False)
        low_d = ohlcv_day["low"].to_numpy(dtype=np.float64, copy=False)
        day_out: Dict[str, object] = {}
        for cand_cfg in top_candidates:
            key = f"{cand_cfg['variant']}_s{int(cand_cfg['stop_pts'])}_t{int(cand_cfg['target_pts'])}"
            parts = cand_cfg["variant"].split("_")
            floor = float(parts[3][1:]) / 100.0
            gap = int(parts[4][1:])
            chosen = _simulate_diverse_first_k(day_cand, "score", k=int(args.k), floor=floor, gap_minutes=gap)
            m = _score_selected_trades(
                day_cand,
                chosen,
                close=close_d,
                high=high_d,
                low=low_d,
                stop_pts=float(cand_cfg["stop_pts"]),
                target_pts=float(cand_cfg["target_pts"]),
                forward_bars=int(args.forward_bars),
            )
            day_out[key] = m
        per_sql_day[str(day)] = _to_jsonable(day_out)

    result = {
        "meta": {
            "data": str(args.data),
            "history_csv": str(args.history_csv),
            "sql_days": sql_days,
            "k": int(args.k),
            "train_stop": float(args.train_stop),
            "train_target": float(args.train_target),
            "forward_bars": int(args.forward_bars),
            "candidate_count_oof": int(len(cand)),
            "elapsed_sec": float(time.time() - t0),
            "note": "Standalone trend+pullback v3 with relaxed candidate family for sidecar usage.",
        },
        "oof_best": _to_jsonable(best_oof),
        "oof_qualified_top10": _to_jsonable(qualified_sorted[:10]),
        "oof_top_pf_20": _to_jsonable(top_pf),
        "oof_num_qualified_pf_ge_2": int(len(qualified_sorted)),
        "per_sql_day_top_candidates": _to_jsonable(per_sql_day),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_to_jsonable(result), indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
