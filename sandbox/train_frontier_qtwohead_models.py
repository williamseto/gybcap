#!/usr/bin/env python
"""Train and export frontier q_twohead quality models for runtime use.

Exports:
- tp_model.json
- fail_model.json
- metadata.json (feature_cols + training contract info)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _try_auc(y: np.ndarray, s: np.ndarray) -> float | None:
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


def _load_runtime_feature_allowlist(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text())
        if isinstance(obj, dict):
            cols = obj.get("columns", [])
        else:
            cols = obj
        return {str(c) for c in cols if str(c).strip()}
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        return {str(c) for c in df.columns}
    if p.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(p, nrows=1)
        return {str(c) for c in df.columns}
    return set()


def _strict_feature_subset(
    rows: pd.DataFrame,
    cols: list[str],
    runtime_allowlist: set[str] | None = None,
) -> list[str]:
    ban_sub = (
        "target",
        "prob",
        "weight",
        "hard_negative",
        "episode",
        "strict",
        "clear",
        "three_class",
        "mfe",
        "disp_start",
        "rev_thr",
        "seen_",
        "blend",
        "custom",
        "deep",
        "label",
        "outcome",
        "calibration",
        "tp_",
        "large_",
        "ambiguity_",
        "priority_",
        "acc_",
        "prop_",
    )
    runtime_allow = set(runtime_allowlist or set())
    out: list[str] = []
    for c in cols:
        lc = c.lower()
        if lc in {"trading_day", "ratio_aligned_start"}:
            continue
        if lc.startswith("w_"):
            continue
        if lc != "event_base_prob" and any(b in lc for b in ban_sub):
            continue
        if c in {
            "large_base_prob",
            "tp_base_prob",
            "mae_estimate",
            "w_tp",
            "w_fail",
            "fail_target",
        }:
            continue
        if runtime_allow and c not in runtime_allow and c != "event_base_prob":
            continue
        if pd.api.types.is_numeric_dtype(rows[c]):
            out.append(c)
    return out


def _build_tp_weights(rows: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(rows.get("tp_now_target"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ev = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
    minute = pd.to_numeric(rows.get("minute_of_day"), errors="coerce").fillna(390).to_numpy(dtype=np.int32, copy=False)
    w = np.ones(len(rows), dtype=np.float64)
    pos = np.isfinite(y) & (y == 1.0)
    neg = np.isfinite(y) & (y == 0.0)
    w[pos] *= 1.2
    w[neg] *= 1.5
    w[neg & (ev >= 0.55)] *= 1.8
    w[neg & (minute < 450)] *= 1.35
    return w


def _build_fail_weights(rows: pd.DataFrame) -> np.ndarray:
    y = pd.to_numeric(rows.get("tp_now_target"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ev = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
    minute = pd.to_numeric(rows.get("minute_of_day"), errors="coerce").fillna(390).to_numpy(dtype=np.int32, copy=False)
    flips = pd.to_numeric(rows.get("same_level_side_flip_count_10"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    w = np.ones(len(rows), dtype=np.float64)
    fail = np.isfinite(y) & (y == 0.0)
    succ = np.isfinite(y) & (y == 1.0)
    w[fail] *= 1.8
    w[fail & (ev >= 0.55)] *= 1.7
    w[fail & (minute < 450)] *= 1.4
    w[fail & (flips >= 2.0)] *= 1.25
    w[succ] *= 1.05
    return w


def _attach_event_prob(rows: pd.DataFrame, score_rows_path: str) -> pd.DataFrame:
    if "event_base_prob" in rows.columns:
        rows["event_base_prob"] = pd.to_numeric(rows["event_base_prob"], errors="coerce").fillna(0.5)
        return rows
    p = Path(score_rows_path)
    if not p.exists():
        rows["event_base_prob"] = 0.5
        return rows
    s = pd.read_parquet(p).copy().reset_index(drop=True)
    key_cols = [c for c in ["trading_day", "episode_start_idx"] if c in rows.columns and c in s.columns]
    if len(key_cols) < 2:
        rows["event_base_prob"] = 0.5
        return rows
    src_col = None
    for c in ("event_base_prob", "acc_prop_tcn_base_max", "prop_tcn_base_max"):
        if c in s.columns:
            src_col = c
            break
    if src_col is None:
        rows["event_base_prob"] = 0.5
        return rows
    merged = rows.merge(
        s[key_cols + [src_col]].drop_duplicates(subset=key_cols),
        on=key_cols,
        how="left",
    )
    merged["event_base_prob"] = pd.to_numeric(merged[src_col], errors="coerce").fillna(0.5)
    if src_col != "event_base_prob":
        merged = merged.drop(columns=[src_col], errors="ignore")
    return merged


def _fit_weighted_binary_full(
    rows: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
    *,
    seed: int,
) -> xgb.XGBClassifier:
    y = pd.to_numeric(rows[target_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    mask = np.isfinite(y)
    X = rows.loc[mask, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    y_fit = y[mask].astype(np.int8, copy=False)
    if np.unique(y_fit).size < 2:
        raise RuntimeError(f"{target_col} has <2 classes in full fit")
    w = pd.to_numeric(rows.loc[mask, weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    pos = float((y_fit == 1).sum())
    neg = float((y_fit == 0).sum())
    scale_pos = float(neg / max(pos, 1.0))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=360,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        gamma=0.0,
        random_state=int(seed),
        n_jobs=8,
        scale_pos_weight=scale_pos,
        tree_method="hist",
    )
    model.fit(X, y_fit, sample_weight=w, verbose=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--score-rows", default="sandbox/results/upstream_two_stage_constrained_v1_smallgrid_scores.parquet")
    ap.add_argument(
        "--runtime-feature-source",
        default="",
        help="Optional feature source (json/parquet/csv) to restrict training to runtime-available columns.",
    )
    ap.add_argument(
        "--output-model-dir",
        default="models/reversal_frontier_qtwohead_major_r25_reclaim_v1",
    )
    ap.add_argument("--seed", type=int, default=419)
    args = ap.parse_args()

    t0 = time.time()
    rows = pd.read_parquet(args.rows).copy().reset_index(drop=True)
    rows = _attach_event_prob(rows, args.score_rows)
    rows["event_base_prob"] = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5)

    y_tp = pd.to_numeric(rows.get("tp_now_target"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    rows["fail_target"] = np.where(np.isfinite(y_tp), (y_tp == 0.0).astype(np.float64), np.nan)
    rows["w_tp"] = _build_tp_weights(rows)
    rows["w_fail"] = _build_fail_weights(rows)

    runtime_allow = _load_runtime_feature_allowlist(args.runtime_feature_source)
    feature_cols = _strict_feature_subset(
        rows,
        list(rows.columns),
        runtime_allowlist=runtime_allow,
    )
    if not feature_cols:
        raise RuntimeError("No usable feature columns for q_twohead training")

    tp_model = _fit_weighted_binary_full(
        rows,
        feature_cols,
        "tp_now_target",
        "w_tp",
        seed=int(args.seed) + 11,
    )
    fail_model = _fit_weighted_binary_full(
        rows,
        feature_cols,
        "fail_target",
        "w_fail",
        seed=int(args.seed) + 29,
    )

    X = rows.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    tp = tp_model.predict_proba(X)[:, 1].astype(np.float64)
    fail = fail_model.predict_proba(X)[:, 1].astype(np.float64)
    q = np.clip(tp * (1.0 - fail), 0.0, 1.0)
    auc_tp = _try_auc(y_tp, q)

    out_dir = Path(args.output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tp_model.save_model(str(out_dir / "tp_model.json"))
    fail_model.save_model(str(out_dir / "fail_model.json"))

    metadata = {
        "feature_cols": feature_cols,
        "label_tp_col": "tp_now_target",
        "label_fail_col": "fail_target",
        "rows_path": str(args.rows),
        "score_rows_path": str(args.score_rows),
        "seed": int(args.seed),
        "n_rows": int(len(rows)),
        "n_features": int(len(feature_cols)),
        "train_auc_tp_proxy": auc_tp,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote={out_dir}")
    print(f"n_rows={len(rows)} n_features={len(feature_cols)} auc_tp_proxy={auc_tp}")
    if runtime_allow:
        print(f"runtime_allowlist_features={len(runtime_allow)}")


if __name__ == "__main__":
    main()
