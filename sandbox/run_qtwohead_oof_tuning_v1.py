#!/usr/bin/env python
"""OOF hyperparameter tuning for q_twohead runtime model.

Goal:
- reduce full-history in-sample inflation
- test whether q_twohead still carries useful signal under strict walk-forward OOF
- evaluate with live-feasible router/execution contract (sequential + independent reference)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sandbox import train_level_models as tlm
import sandbox.run_sequential_reconciliation_v1 as sr
import sandbox.train_frontier_qtwohead_models as tf


def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _try_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        m = np.isfinite(y) & np.isfinite(s)
        if int(m.sum()) < 500:
            return None
        yy = y[m].astype(np.int8, copy=False)
        if np.unique(yy).size < 2:
            return None
        return float(roc_auc_score(yy, s[m]))
    except Exception:
        return None


def _fold_val_days(train_days: Sequence[str]) -> set[str]:
    n = len(train_days)
    n_val = max(1, int(np.ceil(n * 0.2)))
    return set(train_days[-n_val:])


def _fit_binary(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    w_fit: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    w_val: Optional[np.ndarray],
    *,
    params: Dict[str, float | int],
    seed: int,
) -> xgb.XGBClassifier:
    pos = float((y_fit == 1).sum())
    neg = float((y_fit == 0).sum())
    spw = float(max(1.0, neg / max(pos, 1.0)))

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=int(params.get("n_estimators", 360)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 4)),
        min_child_weight=float(params.get("min_child_weight", 3.0)),
        subsample=float(params.get("subsample", 0.85)),
        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
        reg_alpha=float(params.get("reg_alpha", 0.2)),
        reg_lambda=float(params.get("reg_lambda", 2.0)),
        gamma=float(params.get("gamma", 0.0)),
        random_state=int(seed),
        n_jobs=8,
        scale_pos_weight=spw,
        tree_method="hist",
    )

    use_val = (
        X_val is not None
        and y_val is not None
        and w_val is not None
        and len(y_val) >= 100
        and np.unique(y_val).size >= 2
    )
    if use_val:
        model.set_params(early_stopping_rounds=30)
        model.fit(
            X_fit,
            y_fit,
            sample_weight=w_fit,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=False,
        )
    else:
        model.fit(X_fit, y_fit, sample_weight=w_fit, verbose=False)
    return model


def _oof_scores(
    rows: pd.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    weight_col: str,
    n_folds: int,
    min_train_days: int,
    seed: int,
    params: Dict[str, float | int],
) -> np.ndarray:
    day_key = rows["trading_day"].astype(str)
    days = sorted(day_key.unique().tolist())
    splits = tlm.get_fold_splits(days, int(n_folds), int(min_train_days))

    y_all = pd.to_numeric(rows[target_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    w_all = pd.to_numeric(rows[weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    X_all = rows.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

    out = np.full(len(rows), np.nan, dtype=np.float64)
    for fold_idx, (train_days, test_days) in enumerate(splits):
        val_days = _fold_val_days(train_days)
        fit_days = set(train_days) - val_days

        fit_mask = day_key.isin(fit_days).to_numpy() & np.isfinite(y_all)
        val_mask = day_key.isin(val_days).to_numpy() & np.isfinite(y_all)
        test_mask = day_key.isin(set(test_days)).to_numpy()

        if int(fit_mask.sum()) < 500:
            continue
        y_fit = y_all[fit_mask].astype(np.int8, copy=False)
        if np.unique(y_fit).size < 2:
            continue
        X_fit = X_all[fit_mask]
        w_fit = w_all[fit_mask]

        X_val = y_val = w_val = None
        if int(val_mask.sum()) >= 100:
            X_val = X_all[val_mask]
            y_val = y_all[val_mask].astype(np.int8, copy=False)
            w_val = w_all[val_mask]

        model = _fit_binary(
            X_fit,
            y_fit,
            w_fit,
            X_val,
            y_val,
            w_val,
            params=params,
            seed=int(seed) + fold_idx * 101,
        )
        if int(test_mask.sum()) > 0:
            out[test_mask] = model.predict_proba(X_all[test_mask])[:, 1].astype(np.float64)
    return out


def _full_fit_model(
    rows: pd.DataFrame,
    *,
    feature_cols: List[str],
    target_col: str,
    weight_col: str,
    seed: int,
    params: Dict[str, float | int],
) -> xgb.XGBClassifier:
    y = pd.to_numeric(rows[target_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(y)
    X = rows.loc[mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    yy = y[mask].astype(np.int8, copy=False)
    w = pd.to_numeric(rows.loc[mask, weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    if np.unique(yy).size < 2:
        raise RuntimeError(f"{target_col} has <2 classes for full fit")
    return _fit_binary(X, yy, w, None, None, None, params=params, seed=int(seed))


def _build_param_grid() -> List[Dict[str, object]]:
    return [
        {
            "name": "base_ref",
            "params": {
                "n_estimators": 360,
                "learning_rate": 0.05,
                "max_depth": 4,
                "min_child_weight": 3.0,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.2,
                "reg_lambda": 2.0,
                "gamma": 0.0,
            },
        },
        {
            "name": "reg_light",
            "params": {
                "n_estimators": 420,
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 6.0,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.8,
                "reg_lambda": 3.5,
                "gamma": 0.10,
            },
        },
        {
            "name": "reg_medium",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 8.0,
                "subsample": 0.78,
                "colsample_bytree": 0.78,
                "reg_alpha": 1.2,
                "reg_lambda": 4.5,
                "gamma": 0.20,
            },
        },
        {
            "name": "reg_heavy",
            "params": {
                "n_estimators": 550,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 10.0,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 2.0,
                "reg_lambda": 6.0,
                "gamma": 0.40,
            },
        },
        {
            "name": "depth5_bal",
            "params": {
                "n_estimators": 420,
                "learning_rate": 0.04,
                "max_depth": 5,
                "min_child_weight": 8.0,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.8,
                "reg_lambda": 3.5,
                "gamma": 0.15,
            },
        },
        {
            "name": "depth3_fast",
            "params": {
                "n_estimators": 300,
                "learning_rate": 0.06,
                "max_depth": 3,
                "min_child_weight": 6.0,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.6,
                "reg_lambda": 3.0,
                "gamma": 0.10,
            },
        },
    ]


def _router_cfg_from_args(args: argparse.Namespace, base_live_cfg: Dict[str, object]) -> sr.RouterCfg:
    cfg = sr._router_cfg_from_live(base_live_cfg)
    cfg.quality_source = "q_twohead_runtime"
    cfg.k_total = int(args.k_total)
    cfg.q_slots = int(args.q_slots)
    cfg.c_slots = int(args.c_slots)
    cfg.r_slots = int(args.r_slots)
    cfg.q_thr = float(args.q_thr)
    cfg.c_thr = float(args.c_thr)
    cfg.r_thr = float(args.r_thr)
    cfg.c_start = int(args.c_start)
    cfg.r_start = int(args.r_start)
    cfg.cooldown_min = int(args.cooldown_min)
    return cfg


def _variant_by_name(name: str) -> sr.ExecVariant:
    for v in sr._build_variants():
        if v.name == name:
            return v
    raise ValueError(f"variant not found: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--score-rows", default="sandbox/results/upstream_two_stage_constrained_v1_smallgrid_scores.parquet")
    ap.add_argument("--runtime-feature-source", default="")
    ap.add_argument("--ohlcv", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--live-config", default="configs/realtime/strategies/reversal_frontier_qtwohead_runtime_v2_cal_d_k6_ll99_v1.jsonc")
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--min-train-days", type=int, default=100)
    ap.add_argument("--seed", type=int, default=419)
    ap.add_argument("--k-total", type=int, default=6)
    ap.add_argument("--q-slots", type=int, default=6)
    ap.add_argument("--c-slots", type=int, default=4)
    ap.add_argument("--r-slots", type=int, default=2)
    ap.add_argument("--q-thr", type=float, default=0.20)
    ap.add_argument("--c-thr", type=float, default=0.16)
    ap.add_argument("--r-thr", type=float, default=0.12)
    ap.add_argument("--c-start", type=int, default=480)
    ap.add_argument("--r-start", type=int, default=630)
    ap.add_argument("--cooldown-min", type=int, default=5)
    ap.add_argument("--seq-variant", default="seq_hold_replace_qgain03_px000_v1")
    ap.add_argument("--export-best-model-dir", default="")
    ap.add_argument("--output-json", default="sandbox/results/qtwohead_oof_tuning_v1_20260321.json")
    args = ap.parse_args()

    t0 = time.time()

    rows = pd.read_parquet(args.rows).copy().reset_index(drop=True)
    rows = tf._attach_event_prob(rows, args.score_rows)
    rows["event_base_prob"] = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5)

    y_tp = pd.to_numeric(rows.get("tp_now_target"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    rows["fail_target"] = np.where(np.isfinite(y_tp), (y_tp == 0.0).astype(np.float64), np.nan)
    rows["w_tp"] = tf._build_tp_weights(rows)
    rows["w_fail"] = tf._build_fail_weights(rows)

    runtime_allow = tf._load_runtime_feature_allowlist(args.runtime_feature_source)
    feature_cols = tf._strict_feature_subset(rows, list(rows.columns), runtime_allowlist=runtime_allow)
    if not feature_cols:
        raise RuntimeError("No usable features for q_twohead tuning")

    ohlcv = sr._load_ohlcv(args.ohlcv)
    rows = sr._add_proximity(rows, ohlcv)
    close = pd.to_numeric(ohlcv["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(ohlcv["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(ohlcv["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = sr._build_rth_day_end(ohlcv)

    live_cfg = sr._load_live_router_cfg(args.live_config)
    router_cfg = _router_cfg_from_args(args, live_cfg)

    variant_seq = _variant_by_name(str(args.seq_variant))
    variant_ind = _variant_by_name("independent_ref_live_router")

    grid = _build_param_grid()
    all_results: List[Dict[str, object]] = []

    for i, cell in enumerate(grid):
        name = str(cell["name"])
        params = dict(cell["params"])

        tp_oof = _oof_scores(
            rows,
            feature_cols=feature_cols,
            target_col="tp_now_target",
            weight_col="w_tp",
            n_folds=int(args.n_folds),
            min_train_days=int(args.min_train_days),
            seed=int(args.seed) + i * 17 + 11,
            params=params,
        )
        fail_oof = _oof_scores(
            rows,
            feature_cols=feature_cols,
            target_col="fail_target",
            weight_col="w_fail",
            n_folds=int(args.n_folds),
            min_train_days=int(args.min_train_days),
            seed=int(args.seed) + i * 17 + 29,
            params=params,
        )
        q_oof = np.clip(tp_oof * (1.0 - fail_oof), 0.0, 1.0)

        eval_rows = rows.copy()
        eval_rows["q_twohead_runtime"] = q_oof
        # strict OOF view: only days with available OOF q-scores
        has_q = eval_rows.groupby(eval_rows["trading_day"].astype(str), sort=False)["q_twohead_runtime"].transform(
            lambda s: bool(np.isfinite(pd.to_numeric(s, errors="coerce")).any())
        )
        eval_rows = eval_rows.loc[has_q].copy().reset_index(drop=True)

        day_groups = sr._build_day_groups(eval_rows)
        all_days = [d for d, _ in day_groups]

        chosen_ind, tdf_ind, diag_ind = sr._simulate_variant(
            rows=eval_rows,
            day_groups=day_groups,
            day_end=day_end,
            close=close,
            high=high,
            low=low,
            router_cfg=router_cfg,
            variant=variant_ind,
            policy_scorer=None,
        )
        label_ind = sr._label_metrics(eval_rows, chosen_ind, day_groups)
        exec_ind = sr._exec_metrics(tdf_ind, all_days)

        chosen_seq, tdf_seq, diag_seq = sr._simulate_variant(
            rows=eval_rows,
            day_groups=day_groups,
            day_end=day_end,
            close=close,
            high=high,
            low=low,
            router_cfg=router_cfg,
            variant=variant_seq,
            policy_scorer=None,
        )
        label_seq = sr._label_metrics(eval_rows, chosen_seq, day_groups)
        exec_seq = sr._exec_metrics(tdf_seq, all_days)

        rec = {
            "name": name,
            "params": params,
            "n_rows_eval": int(len(eval_rows)),
            "n_days_eval": int(len(all_days)),
            "auc_tp_oof": _try_auc(y_tp, q_oof),
            "independent": {
                "label": label_ind,
                "exec": exec_ind,
                "diag": diag_ind,
            },
            "sequential": {
                "variant": variant_seq.name,
                "label": label_seq,
                "exec": exec_seq,
                "diag": diag_seq,
            },
        }
        all_results.append(rec)
        print(
            f"{name:12s} auc={_safe_float(rec['auc_tp_oof'], float('nan')):.4f} "
            f"seqPF={_safe_float(exec_seq.get('profit_factor'), -1):.3f} "
            f"seqNoTP={_safe_float(label_seq.get('rth_no_tp_day_rate'), 1):.3f} "
            f"seqTr/day={_safe_float(exec_seq.get('mean_trades_per_day'), 0):.2f} "
            f"indPF={_safe_float(exec_ind.get('profit_factor'), -1):.3f}"
        )

    ranked = sorted(
        all_results,
        key=lambda r: (
            -_safe_float(r["sequential"]["exec"].get("profit_factor"), -1e9),
            _safe_float(r["sequential"]["label"].get("rth_no_tp_day_rate"), 1e9),
            -_safe_float(r["sequential"]["exec"].get("annualized_daily_sharpe"), -1e9),
        ),
    )
    feasible = [
        r for r in ranked
        if 5.0 <= _safe_float(r["sequential"]["exec"].get("mean_trades_per_day"), 0.0) <= 10.0
    ]
    target_like = [
        r for r in feasible
        if _safe_float(r["sequential"]["exec"].get("profit_factor"), 0.0) >= 2.0
        and _safe_float(r["sequential"]["label"].get("rth_no_tp_day_rate"), 1.0) <= 0.10
    ]

    export_info = None
    if args.export_best_model_dir and ranked:
        best = ranked[0]
        params = dict(best["params"])
        out_dir = Path(args.export_best_model_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tp_m = _full_fit_model(
            rows,
            feature_cols=feature_cols,
            target_col="tp_now_target",
            weight_col="w_tp",
            seed=int(args.seed) + 701,
            params=params,
        )
        fail_m = _full_fit_model(
            rows,
            feature_cols=feature_cols,
            target_col="fail_target",
            weight_col="w_fail",
            seed=int(args.seed) + 709,
            params=params,
        )
        tp_m.save_model(str(out_dir / "tp_model.json"))
        fail_m.save_model(str(out_dir / "fail_model.json"))
        md = {
            "feature_cols": feature_cols,
            "rows_path": str(args.rows),
            "score_rows_path": str(args.score_rows),
            "best_name": str(best["name"]),
            "best_params": params,
            "n_rows": int(len(rows)),
            "n_features": int(len(feature_cols)),
            "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "metadata.json").write_text(json.dumps(md, indent=2) + "\n", encoding="utf-8")
        export_info = {"model_dir": str(out_dir), "best_name": str(best["name"])}

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "meta": {
            "rows": str(args.rows),
            "score_rows": str(args.score_rows),
            "ohlcv": str(args.ohlcv),
            "live_config": str(args.live_config),
            "n_folds": int(args.n_folds),
            "min_train_days": int(args.min_train_days),
            "seed": int(args.seed),
            "feature_count": int(len(feature_cols)),
            "router_cfg": sr.asdict(router_cfg),
            "seq_variant": str(variant_seq.name),
        },
        "ranked": sr._to_jsonable(ranked),
        "feasible": sr._to_jsonable(feasible),
        "target_like": sr._to_jsonable(target_like),
        "export": export_info,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sr._to_jsonable(out), indent=2) + "\n", encoding="utf-8")

    rows_csv = []
    for r in ranked:
        rows_csv.append(
            {
                "name": r["name"],
                "auc_tp_oof": _safe_float(r.get("auc_tp_oof"), np.nan),
                "seq_pf": _safe_float(r["sequential"]["exec"].get("profit_factor"), np.nan),
                "seq_sharpe": _safe_float(r["sequential"]["exec"].get("annualized_daily_sharpe"), np.nan),
                "seq_mean_pnl_per_day": _safe_float(r["sequential"]["exec"].get("mean_pnl_per_day"), np.nan),
                "seq_trades_per_day": _safe_float(r["sequential"]["exec"].get("mean_trades_per_day"), np.nan),
                "seq_no_tp": _safe_float(r["sequential"]["label"].get("rth_no_tp_day_rate"), np.nan),
                "ind_pf": _safe_float(r["independent"]["exec"].get("profit_factor"), np.nan),
                "ind_sharpe": _safe_float(r["independent"]["exec"].get("annualized_daily_sharpe"), np.nan),
                "ind_mean_pnl_per_day": _safe_float(r["independent"]["exec"].get("mean_pnl_per_day"), np.nan),
                "ind_trades_per_day": _safe_float(r["independent"]["exec"].get("mean_trades_per_day"), np.nan),
                "ind_no_tp": _safe_float(r["independent"]["label"].get("rth_no_tp_day_rate"), np.nan),
            }
        )
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
