#!/usr/bin/env python
"""Learned meta-router over expert signals (full-history + SQL OOS campaign).

This script builds a meta score that ranks expert-emitted reversal candidates,
then evaluates sequential single-position execution policies.

Data scopes:
- Full history (from cached contract rows + ohlcv parquet)
- Optional SQL OOS replay slice (runtime-emitted expert signals)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sandbox import train_level_models as tlm
from sandbox.analyze_reversal_playback_day import _build_strategy, _load_causal_history, _load_day_bars, _load_json
from sandbox.run_sequential_reconciliation_v1 import _PolicyCtx, _PolicyScorer


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _norm_ts(ts: Any) -> Optional[pd.Timestamp]:
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        return t.tz_convert("America/Los_Angeles").tz_localize(None)
    return t


@dataclass
class ExpertContract:
    name: str
    gate: float
    stop_hi: float
    target_hi: float
    stop_lo: float
    target_lo: float
    min_signal_score: float
    start_minute: int


@dataclass
class PolicyCfg:
    name: str
    experts: List[str]
    score_col: str  # raw_score | meta_score
    k_total: int = 7
    cooldown_min: int = 5
    unresolved_min: int = 10
    entry_thr: float = 0.55
    flatten_thr: float = 0.60
    flip_thr: float = 0.70
    opposite_action: str = "flatten"  # ignore | flatten | flip_or_flatten
    expert_caps: Dict[str, int] = None
    expert_starts: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.expert_caps is None:
            self.expert_caps = {}
        if self.expert_starts is None:
            self.expert_starts = {}


def _build_rth_day_end(oc: pd.DataFrame) -> Dict[str, int]:
    day_end: Dict[str, int] = {}
    td = oc["trading_day"].astype(str).to_numpy()
    mod = pd.to_numeric(oc["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    for day in pd.unique(td):
        m = td == day
        rth = m & (mod >= 390) & (mod < 780)
        idx = np.flatnonzero(rth)
        if idx.size > 0:
            day_end[str(day)] = int(idx[-1])
            continue
        idx_all = np.flatnonzero(m)
        if idx_all.size > 0:
            day_end[str(day)] = int(idx_all[-1])
    return day_end


def _simulate_trade(
    *,
    entry_idx: int,
    entry_price: float,
    direction: int,
    stop_pts: float,
    target_pts: float,
    end_idx: int,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> Tuple[float, int, str]:
    if end_idx <= entry_idx:
        return 0.0, int(entry_idx), "timed"
    if direction > 0:
        stop_px = float(entry_price - stop_pts)
        target_px = float(entry_price + target_pts)
        for j in range(int(entry_idx) + 1, int(end_idx) + 1):
            hj = float(high[j])
            lj = float(low[j])
            if lj <= stop_px and hj >= target_px:
                return -float(stop_pts), int(j), "stop_tie"
            if lj <= stop_px:
                return -float(stop_pts), int(j), "stop"
            if hj >= target_px:
                return float(target_pts), int(j), "target"
        return float(close[int(end_idx)] - entry_price), int(end_idx), "timed"
    stop_px = float(entry_price + stop_pts)
    target_px = float(entry_price - target_pts)
    for j in range(int(entry_idx) + 1, int(end_idx) + 1):
        hj = float(high[j])
        lj = float(low[j])
        if hj >= stop_px and lj <= target_px:
            return -float(stop_pts), int(j), "stop_tie"
        if hj >= stop_px:
            return -float(stop_pts), int(j), "stop"
        if lj <= target_px:
            return float(target_pts), int(j), "target"
    return float(entry_price - close[int(end_idx)]), int(end_idx), "timed"


def _build_policy_prob(rows: pd.DataFrame, model_dir: str) -> np.ndarray:
    scorer = _PolicyScorer(model_dir=model_dir, use_ensemble_mean=True)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    day_key = rows["trading_day"].astype(str)
    for _, g in rows.groupby(day_key, sort=False):
        idx = g.sort_values(["minute_of_day", "episode_start_idx"]).index.to_numpy(dtype=np.int64, copy=False)
        ctx = _PolicyCtx()
        for pos in idx:
            row = rows.loc[int(pos)]
            base_prob = _safe_float(row.get("event_base_prob", 0.5), 0.5)
            p, _ = scorer.predict(row=row, base_prob=base_prob, policy_ctx=ctx)
            out[int(pos)] = float(p)
            ctx.episodes_started_today += 1
            ts = pd.Timestamp(row.get("episode_start_dt")) if pd.notna(row.get("episode_start_dt")) else None
            if ts is not None and not pd.isna(ts):
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("America/Los_Angeles")
                ctx.last_episode_ts = ts
    return out


def _build_q2_score(rows: pd.DataFrame, model_dir: str) -> np.ndarray:
    md = json.loads((Path(model_dir) / "metadata.json").read_text())
    feat = [str(c) for c in md.get("feature_cols", [])]
    if not feat:
        return np.full(len(rows), 0.0, dtype=np.float64)
    work = rows.copy()
    if "trade_direction" not in work.columns and "side" in work.columns:
        work["trade_direction"] = pd.to_numeric(work.get("side"), errors="coerce").fillna(0.0)
    for c in feat:
        if c not in work.columns:
            work[c] = 0.0
    X = work[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    tp = xgb.XGBClassifier()
    tp.load_model(str(Path(model_dir) / "tp_model.json"))
    fail = xgb.XGBClassifier()
    fail.load_model(str(Path(model_dir) / "fail_model.json"))
    tp_p = tp.predict_proba(X)[:, 1].astype(np.float64)
    fail_p = fail.predict_proba(X)[:, 1].astype(np.float64)
    return np.clip(tp_p * (1.0 - fail_p), 0.0, 1.0)


def _build_candidates_full(
    rows: pd.DataFrame,
    *,
    policy_prob: np.ndarray,
    q2_score: np.ndarray,
    experts: Dict[str, ExpertContract],
) -> pd.DataFrame:
    c_rows: List[Dict[str, Any]] = []
    minute = pd.to_numeric(rows["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    direction = pd.to_numeric(rows["trade_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int8, copy=False)
    level_name = rows["nearest_level_name"].astype(str).to_numpy()
    base_prob = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5).to_numpy(dtype=np.float64, copy=False)
    day = rows["trading_day"].astype(str).to_numpy()
    entry_idx = pd.to_numeric(rows["episode_start_idx"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    rth = (minute >= 390) & (minute < 780) & (direction != 0) & (entry_idx >= 0)

    for i in np.flatnonzero(rth):
        lp = float(policy_prob[int(i)])
        qq = float(q2_score[int(i)])
        if np.isfinite(lp) and lp >= float(experts["live"].min_signal_score):
            c_rows.append(
                {
                    "row_idx": int(i),
                    "trading_day": str(day[i]),
                    "entry_idx": int(entry_idx[i]),
                    "minute_of_day": int(minute[i]),
                    "direction": int(direction[i]),
                    "level_name": str(level_name[i]),
                    "expert": "live",
                    "raw_score": float(lp),
                    "base_prob": float(base_prob[i]),
                }
            )
        if np.isfinite(qq) and qq >= float(experts["q2"].min_signal_score):
            c_rows.append(
                {
                    "row_idx": int(i),
                    "trading_day": str(day[i]),
                    "entry_idx": int(entry_idx[i]),
                    "minute_of_day": int(minute[i]),
                    "direction": int(direction[i]),
                    "level_name": str(level_name[i]),
                    "expert": "q2",
                    "raw_score": float(qq),
                    "base_prob": float(base_prob[i]),
                }
            )
    if not c_rows:
        return pd.DataFrame(
            columns=[
                "row_idx",
                "trading_day",
                "entry_idx",
                "minute_of_day",
                "direction",
                "level_name",
                "expert",
                "raw_score",
                "base_prob",
            ]
        )
    return pd.DataFrame(c_rows).sort_values(["trading_day", "entry_idx", "expert"]).reset_index(drop=True)


def _attach_candidate_outcomes(
    cand: pd.DataFrame,
    *,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    experts: Dict[str, ExpertContract],
) -> pd.DataFrame:
    c = cand.copy()
    pnl = np.zeros(len(c), dtype=np.float64)
    exit_idx = np.zeros(len(c), dtype=np.int32)
    exit_reason = np.empty(len(c), dtype=object)
    for i, r in c.iterrows():
        ex = experts[str(r["expert"])]
        score = float(r["raw_score"])
        hi_lane = score >= float(ex.gate)
        stop_pts = float(ex.stop_hi if hi_lane else ex.stop_lo)
        target_pts = float(ex.target_hi if hi_lane else ex.target_lo)
        idx = int(r["entry_idx"])
        if idx < 0 or idx >= len(close):
            pnl[i] = 0.0
            exit_idx[i] = idx
            exit_reason[i] = "skip"
            continue
        end_i = int(day_end.get(str(r["trading_day"]), len(close) - 1))
        p, ex_idx, rsn = _simulate_trade(
            entry_idx=idx,
            entry_price=float(close[idx]),
            direction=int(r["direction"]),
            stop_pts=stop_pts,
            target_pts=target_pts,
            end_idx=end_i,
            close=close,
            high=high,
            low=low,
        )
        pnl[i] = float(p)
        exit_idx[i] = int(ex_idx)
        exit_reason[i] = str(rsn)
    c["pnl"] = pnl
    c["exit_idx"] = exit_idx
    c["exit_reason"] = exit_reason
    c["target_win"] = (c["pnl"] > 0.0).astype(np.int8)
    w = np.ones(len(c), dtype=np.float64)
    neg = c["target_win"].to_numpy(dtype=np.int8, copy=False) == 0
    m = pd.to_numeric(c["minute_of_day"], errors="coerce").fillna(390).to_numpy(dtype=np.int32, copy=False)
    stop_like = c["exit_reason"].astype(str).str.startswith("stop").to_numpy()
    w[neg] *= 1.4
    w[neg & stop_like] *= 1.25
    w[neg & (m < 450)] *= 1.15
    c["meta_weight"] = w
    return c


def _build_meta_features(cand: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    c = cand.copy()
    lvl = c["level_name"].astype(str).str.lower()
    feat = pd.DataFrame(index=c.index)
    feat["score"] = pd.to_numeric(c["raw_score"], errors="coerce").fillna(0.0)
    feat["base_prob"] = pd.to_numeric(c["base_prob"], errors="coerce").fillna(0.5)
    feat["score_minus_base"] = feat["score"] - feat["base_prob"]
    feat["score_x_base"] = feat["score"] * feat["base_prob"]
    m = pd.to_numeric(c["minute_of_day"], errors="coerce").fillna(390)
    feat["minute_norm"] = m / 1440.0
    feat["is_opening_30m"] = (m <= 420).astype(np.float32)
    feat["direction"] = pd.to_numeric(c["direction"], errors="coerce").fillna(0.0)
    feat["expert_live"] = (c["expert"].astype(str) == "live").astype(np.float32)
    feat["expert_q2"] = (c["expert"].astype(str) == "q2").astype(np.float32)
    feat["is_ovn_level"] = lvl.str.startswith("ovn_").astype(np.float32)
    feat["is_ib_level"] = lvl.str.startswith("ib_").astype(np.float32)
    feat["is_rth_level"] = lvl.str.startswith("rth_").astype(np.float32)
    feat["is_prev_level"] = lvl.str.startswith("prev_").astype(np.float32)
    feat["is_vwap_level"] = (lvl == "vwap").astype(np.float32)
    cols = list(feat.columns)
    return feat, cols


def _train_meta_xgb(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    seed: int,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    w_val: Optional[np.ndarray] = None,
) -> xgb.XGBClassifier:
    pos_rate = float(np.mean(y)) if y.size else 0.0
    spw = max(1.0, (1.0 - pos_rate) / max(pos_rate, 1e-6))
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=int(seed),
        verbosity=0,
    )
    if X_val is not None and y_val is not None and len(y_val) > 0 and np.unique(y_val).size >= 2:
        model.set_params(early_stopping_rounds=30)
        fit_kwargs: Dict[str, Any] = {"eval_set": [(X_val, y_val)], "verbose": False}
        if w is not None:
            fit_kwargs["sample_weight"] = w
        if w_val is not None:
            fit_kwargs["sample_weight_eval_set"] = [w_val]
        model.fit(X, y, **fit_kwargs)
    else:
        fit_kwargs = {"verbose": False}
        if w is not None:
            fit_kwargs["sample_weight"] = w
        model.fit(X, y, **fit_kwargs)
    return model


def _walk_forward_meta_scores(
    cand: pd.DataFrame,
    feat: pd.DataFrame,
    feat_cols: List[str],
    *,
    n_folds: int,
    min_train_days: int,
    seed: int,
) -> np.ndarray:
    day_key = cand["trading_day"].astype(str)
    days = sorted(day_key.unique(), key=lambda x: int(x))
    splits = tlm.get_fold_splits(days, n_folds, min_train_days)
    y_all = pd.to_numeric(cand["target_win"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    w_all = pd.to_numeric(cand["meta_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    out = np.full(len(cand), np.nan, dtype=np.float64)

    for fold_idx, (train_days, test_days) in enumerate(splits):
        val_n = max(1, int(round(len(train_days) * 0.2)))
        val_days = set(train_days[-val_n:])
        fit_days = set(train_days[:-val_n]) if len(train_days) > val_n else set(train_days)
        fit_mask = day_key.isin(fit_days).to_numpy()
        val_mask = day_key.isin(val_days).to_numpy()
        test_mask = day_key.isin(set(test_days)).to_numpy()
        if int(fit_mask.sum()) < 500:
            continue
        X_fit = feat.loc[fit_mask, feat_cols].to_numpy(dtype=np.float32, copy=False)
        y_fit = y_all[fit_mask].astype(np.int8, copy=False)
        w_fit = w_all[fit_mask]
        if np.unique(y_fit).size < 2:
            continue
        X_val = None
        y_val = None
        w_val = None
        if int(val_mask.sum()) >= 100:
            X_val = feat.loc[val_mask, feat_cols].to_numpy(dtype=np.float32, copy=False)
            y_val = y_all[val_mask].astype(np.int8, copy=False)
            w_val = w_all[val_mask]

        model = _train_meta_xgb(
            X_fit,
            y_fit,
            w_fit,
            seed=seed + fold_idx * 37,
            X_val=X_val,
            y_val=y_val,
            w_val=w_val,
        )
        X_test = feat.loc[test_mask, feat_cols].to_numpy(dtype=np.float32, copy=False)
        if X_test.shape[0] == 0:
            continue
        out[test_mask] = model.predict_proba(X_test)[:, 1].astype(np.float64)
    return out


def _fit_full_meta_model(
    feat: pd.DataFrame,
    feat_cols: List[str],
    y: np.ndarray,
    w: np.ndarray,
    *,
    seed: int,
) -> xgb.XGBClassifier:
    X = feat[feat_cols].to_numpy(dtype=np.float32, copy=False)
    yy = y.astype(np.int8, copy=False)
    return _train_meta_xgb(X, yy, w, seed=seed)


def _simulate_policy(
    *,
    cand: pd.DataFrame,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    policy: PolicyCfg,
    experts: Dict[str, ExpertContract],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    c = cand[cand["expert"].isin(policy.experts)].copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame()
    c = c.sort_values(["trading_day", "entry_idx", policy.score_col], ascending=[True, True, False]).reset_index(drop=True)

    trades: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []

    for day, g in c.groupby(c["trading_day"].astype(str), sort=False):
        rows = g.sort_values(["entry_idx", policy.score_col], ascending=[True, False]).reset_index(drop=True)
        end_idx = int(day_end.get(str(day), len(close) - 1))

        accepted = 0
        last_entry_minute = -10_000
        expert_counts: Dict[str, int] = {}
        unresolved_until: Dict[Tuple[str, int], int] = {}
        pos: Optional[Dict[str, Any]] = None
        next_scan = 1

        for _, r in rows.iterrows():
            i = int(r["entry_idx"])
            if i < 0 or i > end_idx:
                continue
            m = int(r["minute_of_day"])
            d = int(r["direction"])
            ex_name = str(r["expert"])
            ex = experts[ex_name]
            raw_score = float(r["raw_score"])
            s = float(r[policy.score_col])
            lvl = str(r["level_name"])
            key = (lvl, d)

            if pos is not None:
                t_pnl, t_exit, t_reason = _simulate_trade(
                    entry_idx=int(pos["entry_idx"]),
                    entry_price=float(pos["entry_price"]),
                    direction=int(pos["direction"]),
                    stop_pts=float(pos["stop_pts"]),
                    target_pts=float(pos["target_pts"]),
                    end_idx=min(int(i), int(end_idx)),
                    close=close,
                    high=high,
                    low=low,
                )
                if int(t_exit) <= int(i) and str(t_reason) != "timed":
                    trades.append(
                        {
                            "trading_day": str(day),
                            "entry_idx": int(pos["entry_idx"]),
                            "exit_idx": int(t_exit),
                            "pnl": float(t_pnl),
                            "exit_reason": str(t_reason),
                            "entry_expert": str(pos["entry_expert"]),
                            "lane": str(pos["lane"]),
                        }
                    )
                    pos = None
                    next_scan = int(t_exit) + 1
                else:
                    next_scan = int(i) + 1

            # Opposite management first.
            if pos is not None and int(d) != int(pos["direction"]):
                if s >= float(policy.flatten_thr):
                    pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
                    trades.append(
                        {
                            "trading_day": str(day),
                            "entry_idx": int(pos["entry_idx"]),
                            "exit_idx": int(i),
                            "pnl": float(pnl),
                            "exit_reason": "flatten_signal",
                            "entry_expert": str(pos["entry_expert"]),
                            "lane": str(pos["lane"]),
                        }
                    )
                    pos = None
                    if str(policy.opposite_action).lower() == "flip_or_flatten" and s >= float(policy.flip_thr):
                        if accepted < int(policy.k_total):
                            if (m - last_entry_minute) >= int(policy.cooldown_min):
                                if m >= int(policy.expert_starts.get(ex_name, ex.start_minute)):
                                    if int(expert_counts.get(ex_name, 0)) < int(policy.expert_caps.get(ex_name, 99)):
                                        if int(unresolved_until.get(key, -10_000)) < int(m):
                                            hi_lane = raw_score >= float(ex.gate)
                                            stop_pts = float(ex.stop_hi if hi_lane else ex.stop_lo)
                                            target_pts = float(ex.target_hi if hi_lane else ex.target_lo)
                                            pos = {
                                                "entry_idx": int(i),
                                                "entry_price": float(close[i]),
                                                "direction": int(d),
                                                "stop_pts": float(stop_pts),
                                                "target_pts": float(target_pts),
                                                "entry_expert": str(ex_name),
                                                "lane": ("high" if hi_lane else "low"),
                                            }
                                            accepted += 1
                                            last_entry_minute = int(m)
                                            expert_counts[ex_name] = int(expert_counts.get(ex_name, 0)) + 1
                                            unresolved_until[key] = int(m) + int(policy.unresolved_min)
                                            entries.append(
                                                {
                                                    "trading_day": str(day),
                                                    "entry_idx": int(i),
                                                    "minute_of_day": int(m),
                                                    "expert": str(ex_name),
                                                    "score": float(s),
                                                    "action": "flip",
                                                }
                                            )
                continue

            if pos is not None:
                continue

            # Entry.
            if s < float(policy.entry_thr):
                continue
            if accepted >= int(policy.k_total):
                continue
            if (m - last_entry_minute) < int(policy.cooldown_min):
                continue
            if m < int(policy.expert_starts.get(ex_name, ex.start_minute)):
                continue
            if int(expert_counts.get(ex_name, 0)) >= int(policy.expert_caps.get(ex_name, 99)):
                continue
            if int(unresolved_until.get(key, -10_000)) >= int(m):
                continue

            hi_lane = raw_score >= float(ex.gate)
            stop_pts = float(ex.stop_hi if hi_lane else ex.stop_lo)
            target_pts = float(ex.target_hi if hi_lane else ex.target_lo)
            pos = {
                "entry_idx": int(i),
                "entry_price": float(close[i]),
                "direction": int(d),
                "stop_pts": float(stop_pts),
                "target_pts": float(target_pts),
                "entry_expert": str(ex_name),
                "lane": ("high" if hi_lane else "low"),
            }
            accepted += 1
            last_entry_minute = int(m)
            expert_counts[ex_name] = int(expert_counts.get(ex_name, 0)) + 1
            unresolved_until[key] = int(m) + int(policy.unresolved_min)
            entries.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(i),
                    "minute_of_day": int(m),
                    "expert": str(ex_name),
                    "score": float(s),
                    "action": "entry",
                }
            )

        if pos is not None:
            t_pnl, t_exit, t_reason = _simulate_trade(
                entry_idx=int(pos["entry_idx"]),
                entry_price=float(pos["entry_price"]),
                direction=int(pos["direction"]),
                stop_pts=float(pos["stop_pts"]),
                target_pts=float(pos["target_pts"]),
                end_idx=int(end_idx),
                close=close,
                high=high,
                low=low,
            )
            if str(t_reason) == "timed":
                t_pnl = float((float(close[end_idx]) - float(pos["entry_price"])) * float(pos["direction"]))
                t_exit = int(end_idx)
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(pos["entry_idx"]),
                    "exit_idx": int(t_exit),
                    "pnl": float(t_pnl),
                    "exit_reason": str(t_reason),
                    "entry_expert": str(pos["entry_expert"]),
                    "lane": str(pos["lane"]),
                }
            )

    return pd.DataFrame(trades), pd.DataFrame(entries)


def _metrics(trades: pd.DataFrame, day_ids: Sequence[str]) -> Dict[str, Any]:
    day_pnl = pd.Series(0.0, index=pd.Index([str(d) for d in day_ids], dtype=str))
    if not trades.empty:
        by_day = trades.groupby(trades["trading_day"].astype(str))["pnl"].sum()
        day_pnl.loc[by_day.index] = by_day
    daily = day_pnl.to_numpy(dtype=np.float64, copy=False)
    mean_day = float(np.mean(daily)) if daily.size else 0.0
    std_day = float(np.std(daily, ddof=1)) if daily.size > 1 else 0.0
    sharpe = float((mean_day / std_day) * np.sqrt(252.0)) if std_day > 1e-9 else None
    if trades.empty:
        return {
            "n_trades": 0,
            "mean_trades_per_day": 0.0,
            "mean_pnl_per_day": mean_day,
            "annualized_daily_sharpe": sharpe,
            "profit_factor": None,
            "win_rate": None,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "timed_exit_rate": None,
            "zero_trade_day_rate": 1.0,
            "zero_target_day_rate": 1.0,
        }
    gp = float(trades["pnl"].clip(lower=0.0).sum())
    gl = float((-trades["pnl"].clip(upper=0.0)).sum())
    exits = trades["exit_reason"].astype(str)
    target_days = set(trades.loc[exits == "target", "trading_day"].astype(str).unique().tolist())
    return {
        "n_trades": int(len(trades)),
        "mean_trades_per_day": float(len(trades) / max(len(day_ids), 1)),
        "mean_pnl_per_day": mean_day,
        "annualized_daily_sharpe": sharpe,
        "profit_factor": (gp / gl) if gl > 1e-9 else None,
        "win_rate": float((trades["pnl"] > 0.0).mean()),
        "target_hit_rate": float((exits == "target").mean()),
        "stop_hit_rate": float(exits.str.startswith("stop").mean()),
        "timed_exit_rate": float((exits == "timed").mean()),
        "zero_trade_day_rate": float(np.mean(day_pnl.to_numpy() == 0.0)),
        "zero_target_day_rate": float(np.mean([str(d) not in target_days for d in day_ids])) if day_ids else None,
    }


def _distribution(entries: pd.DataFrame, day_ids: Sequence[str]) -> Dict[str, Any]:
    if entries.empty:
        return {
            "open_60_share": 0.0,
            "post_open_60_share": 0.0,
            "late_morning_120_share": 0.0,
            "afternoon_150_share": 0.0,
            "zero_post_open_60_day_rate": 1.0,
            "zero_after_510_day_rate": 1.0,
        }
    e = entries.copy()
    m = pd.to_numeric(e["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    n = float(len(e))
    w_open = (m >= 390) & (m < 450)
    w_post = (m >= 450) & (m < 510)
    w_late = (m >= 510) & (m < 630)
    w_aft = (m >= 630) & (m < 780)
    d_post = e.loc[w_post, "trading_day"].astype(str).value_counts().to_dict()
    d_after = e.loc[(m >= 510) & (m < 780), "trading_day"].astype(str).value_counts().to_dict()
    day_s = [str(d) for d in day_ids]
    return {
        "open_60_share": float(w_open.sum() / n),
        "post_open_60_share": float(w_post.sum() / n),
        "late_morning_120_share": float(w_late.sum() / n),
        "afternoon_150_share": float(w_aft.sum() / n),
        "zero_post_open_60_day_rate": float(np.mean([int(d_post.get(d, 0)) == 0 for d in day_s])) if day_s else None,
        "zero_after_510_day_rate": float(np.mean([int(d_after.get(d, 0)) == 0 for d in day_s])) if day_s else None,
    }


def _policy_grid() -> List[PolicyCfg]:
    out: List[PolicyCfg] = []
    out.append(
        PolicyCfg(
            name="live_raw_ref",
            experts=["live"],
            score_col="raw_score",
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            entry_thr=0.58,
            flatten_thr=0.60,
            flip_thr=0.72,
            opposite_action="flatten",
            expert_caps={"live": 5},
            expert_starts={"live": 390},
        )
    )
    out.append(
        PolicyCfg(
            name="q2_raw_ref",
            experts=["q2"],
            score_col="raw_score",
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            entry_thr=0.20,
            flatten_thr=0.20,
            flip_thr=0.24,
            opposite_action="flatten",
            expert_caps={"q2": 6},
            expert_starts={"q2": 450},
        )
    )
    out.append(
        PolicyCfg(
            name="mix_raw_ref",
            experts=["live", "q2"],
            score_col="raw_score",
            k_total=8,
            cooldown_min=5,
            unresolved_min=10,
            entry_thr=0.20,
            flatten_thr=0.22,
            flip_thr=0.26,
            opposite_action="flatten",
            expert_caps={"live": 5, "q2": 4},
            expert_starts={"live": 390, "q2": 450},
        )
    )

    for k in [6, 7, 8, 9]:
        for thr in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
            for cap_q2 in [1, 2, 3, 4, 5]:
                out.append(
                    PolicyCfg(
                        name=f"mix_meta_k{k}_t{int(thr*100)}_cq2{cap_q2}_flat_v1",
                        experts=["live", "q2"],
                        score_col="meta_score",
                        k_total=int(k),
                        cooldown_min=5,
                        unresolved_min=10,
                        entry_thr=float(thr),
                        flatten_thr=float(max(thr, 0.52)),
                        flip_thr=0.70,
                        opposite_action="flatten",
                        expert_caps={"live": 5, "q2": int(cap_q2)},
                        expert_starts={"live": 390, "q2": 450},
                    )
                )
                out.append(
                    PolicyCfg(
                        name=f"mix_meta_k{k}_t{int(thr*100)}_cq2{cap_q2}_flip_v1",
                        experts=["live", "q2"],
                        score_col="meta_score",
                        k_total=int(k),
                        cooldown_min=5,
                        unresolved_min=10,
                        entry_thr=float(thr),
                        flatten_thr=float(max(thr, 0.52)),
                        flip_thr=float(max(thr + 0.08, 0.60)),
                        opposite_action="flip_or_flatten",
                        expert_caps={"live": 5, "q2": int(cap_q2)},
                        expert_starts={"live": 390, "q2": 450},
                    )
                )
    return out


def _evaluate_policy_set(
    *,
    cand: pd.DataFrame,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_end: Dict[str, int],
    day_ids: Sequence[str],
    experts: Dict[str, ExpertContract],
    policies: List[PolicyCfg],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pol in policies:
        tdf, entries = _simulate_policy(
            cand=cand,
            close=close,
            high=high,
            low=low,
            day_end=day_end,
            policy=pol,
            experts=experts,
        )
        out.append(
            {
                "policy": asdict(pol),
                "metrics": _metrics(tdf, day_ids),
                "distribution": _distribution(entries, day_ids),
                "entry_expert_split": (
                    entries["expert"].astype(str).value_counts(normalize=True).to_dict() if not entries.empty else {}
                ),
            }
        )
    out = sorted(
        out,
        key=lambda r: (
            _safe_float(r["metrics"].get("profit_factor"), -1.0),
            _safe_float(r["metrics"].get("annualized_daily_sharpe"), -1.0),
            _safe_float(r["metrics"].get("mean_pnl_per_day"), -1e9),
        ),
        reverse=True,
    )
    return out


def _load_sql_candidates(
    *,
    days: Sequence[str],
    expert_cfg: Dict[str, str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[str], Dict[str, str]]:
    cand_rows: List[Dict[str, Any]] = []
    close_all: List[np.ndarray] = []
    high_all: List[np.ndarray] = []
    low_all: List[np.ndarray] = []
    day_end: Dict[str, int] = {}
    ok_days: List[str] = []
    fail_days: Dict[str, str] = {}
    offset = 0

    bars_by_day: Dict[str, pd.DataFrame] = {}
    for day in days:
        try:
            b = _load_day_bars(day).copy()
            if "dt" not in b.columns:
                b = b.reset_index()
            b["dt"] = pd.to_datetime(b["dt"], errors="coerce")
            b = b.sort_values("dt").reset_index(drop=True)
            bars_by_day[str(day)] = b
            ok_days.append(str(day))
            c = pd.to_numeric(b["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            h = pd.to_numeric(b["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            l = pd.to_numeric(b["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            close_all.append(c)
            high_all.append(h)
            low_all.append(l)
            day_end[str(day)] = int(offset + len(b) - 1)
            offset += int(len(b))
        except Exception as e:
            fail_days[str(day)] = str(e)

    # Build candidates from each expert strategy.
    offset_map: Dict[str, int] = {}
    cur = 0
    for d in ok_days:
        offset_map[d] = int(cur)
        cur += int(len(bars_by_day[d]))

    for ex_name, cfg_path in expert_cfg.items():
        raw = _load_json(cfg_path)
        params = dict(raw.get("params", raw))
        history_csv = str(params.get("historical_csv_path", "raw_data/schwab/es_minute_history.csv"))
        warmup_days = int(params.get("warmup_days", 60))

        for day in ok_days:
            try:
                strategy = _build_strategy(params)
                hist = _load_causal_history(history_csv, day, warmup_days)
                strategy.set_historical_context(hist)
                bars = bars_by_day[day]
                sigs = strategy.process(bars)
                idx_map: Dict[pd.Timestamp, int] = {}
                for i, ts in enumerate(bars["dt"]):
                    nt = _norm_ts(ts)
                    if nt is not None:
                        idx_map[nt] = int(i)
                for s in sigs:
                    ts = _norm_ts(getattr(s, "entry_ts", None))
                    if ts is None:
                        continue
                    i_local = idx_map.get(ts)
                    if i_local is None:
                        continue
                    i_global = int(offset_map[day] + i_local)
                    md = dict(getattr(s, "metadata", {}) or {})
                    q = _safe_float(md.get("quality_score", getattr(s, "pred_proba", 0.5)), 0.5)
                    bp = _safe_float(getattr(s, "pred_proba", np.nan), np.nan)
                    d = 1 if str(getattr(s, "direction", "bull")).strip().lower() == "bull" else -1
                    m = int(bars.iloc[int(i_local)]["dt"].hour) * 60 + int(bars.iloc[int(i_local)]["dt"].minute)
                    if m < 390 or m >= 780:
                        continue
                    cand_rows.append(
                        {
                            "row_idx": -1,
                            "trading_day": str(day),
                            "entry_idx": int(i_global),
                            "minute_of_day": int(m),
                            "direction": int(d),
                            "level_name": str(getattr(s, "level_name", "")),
                            "expert": str(ex_name),
                            "raw_score": float(q),
                            "base_prob": float(bp) if np.isfinite(bp) else 0.5,
                        }
                    )
            except Exception as e:
                fail_days[str(day)] = str(e)
                continue

    if cand_rows:
        cand = pd.DataFrame(cand_rows).sort_values(["trading_day", "entry_idx", "expert"]).reset_index(drop=True)
    else:
        cand = pd.DataFrame(
            columns=["row_idx", "trading_day", "entry_idx", "minute_of_day", "direction", "level_name", "expert", "raw_score", "base_prob"]
        )

    close = np.concatenate(close_all) if close_all else np.asarray([], dtype=np.float64)
    high = np.concatenate(high_all) if high_all else np.asarray([], dtype=np.float64)
    low = np.concatenate(low_all) if low_all else np.asarray([], dtype=np.float64)
    return cand, close, high, low, day_end, ok_days, fail_days


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="sandbox/results/upstream_contract_rows_scored_v1.parquet")
    ap.add_argument("--ohlcv", default="sandbox/results/upstream_sep_shared_ohlcv.parquet")
    ap.add_argument("--policy-model-dir", default="models/reversal_policy_episode_meta_major_r25_reclaim_v1")
    ap.add_argument("--q2-model-dir", default="models/reversal_frontier_qtwohead_runtime_v2")
    ap.add_argument(
        "--sql-days",
        default="2026-03-04,2026-03-05,2026-03-06,2026-03-09,2026-03-10,2026-03-11,2026-03-12,2026-03-13,2026-03-16,2026-03-17,2026-03-18,2026-03-19",
    )
    ap.add_argument(
        "--sql-experts",
        default=(
            "live=configs/realtime/strategies/reversal_live_selected.jsonc,"
            "q2=configs/realtime/strategies/reversal_frontier_qtwohead_runtime_v2_cal_d_k6_ll99_v1.jsonc"
        ),
        help="Comma-separated name=path expert configs for SQL replay.",
    )
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--min-train-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=617)
    ap.add_argument("--output-json", default="sandbox/results/meta_mixture_router_v1.json")
    args = ap.parse_args()

    t0 = time.time()
    rows = pd.read_parquet(args.rows).copy().reset_index(drop=True)
    oc = pd.read_parquet(args.ohlcv).copy()
    if "_src_index" in oc.columns:
        oc = oc.sort_values("_src_index").reset_index(drop=True)
    close = pd.to_numeric(oc["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(oc["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(oc["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    day_end = _build_rth_day_end(oc)
    full_days = sorted(rows["trading_day"].astype(str).unique(), key=lambda x: int(x))

    experts = {
        "live": ExpertContract(
            name="live",
            gate=0.62,
            stop_hi=10.0,
            target_hi=40.0,
            stop_lo=4.0,
            target_lo=8.0,
            min_signal_score=0.45,
            start_minute=390,
        ),
        "q2": ExpertContract(
            name="q2",
            gate=0.20,
            stop_hi=10.0,
            target_hi=40.0,
            stop_lo=4.0,
            target_lo=8.0,
            min_signal_score=0.12,
            start_minute=450,
        ),
    }

    print("scoring full-history experts ...")
    policy_prob = _build_policy_prob(rows, args.policy_model_dir)
    q2_score = _build_q2_score(rows, args.q2_model_dir)
    full_cand = _build_candidates_full(rows, policy_prob=policy_prob, q2_score=q2_score, experts=experts)
    full_cand = _attach_candidate_outcomes(full_cand, close=close, high=high, low=low, day_end=day_end, experts=experts)
    feat_df, feat_cols = _build_meta_features(full_cand)

    print("training walk-forward meta scores ...")
    full_cand["meta_score"] = _walk_forward_meta_scores(
        full_cand,
        feat_df,
        feat_cols,
        n_folds=int(args.n_folds),
        min_train_days=int(args.min_train_days),
        seed=int(args.seed),
    )
    full_cand["meta_score"] = pd.to_numeric(full_cand["meta_score"], errors="coerce").fillna(0.5)
    y = pd.to_numeric(full_cand["target_win"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    w = pd.to_numeric(full_cand["meta_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
    full_meta_model = _fit_full_meta_model(feat_df, feat_cols, y=y, w=w, seed=int(args.seed) + 911)

    policies = _policy_grid()
    full_eval = _evaluate_policy_set(
        cand=full_cand,
        close=close,
        high=high,
        low=low,
        day_end=day_end,
        day_ids=full_days,
        experts=experts,
        policies=policies,
    )

    full_live_ref = next((r for r in full_eval if r["policy"]["name"] == "live_raw_ref"), None)
    live_ref_pf = _safe_float((full_live_ref or {}).get("metrics", {}).get("profit_factor"), np.nan)
    full_better_pf = [
        r
        for r in full_eval
        if np.isfinite(live_ref_pf) and _safe_float(r["metrics"].get("profit_factor"), -1.0) > live_ref_pf
    ]
    full_feasible = [
        r
        for r in full_eval
        if 5.0 <= _safe_float(r["metrics"].get("mean_trades_per_day"), 0.0) <= 10.0
        and _safe_float(r["distribution"].get("zero_after_510_day_rate"), 1.0) <= 0.10
    ]

    # SQL OOS evaluation using runtime-emitted expert signals.
    sql_cfg = {}
    for tok in _parse_csv(args.sql_experts):
        name, path = tok.split("=", 1)
        sql_cfg[str(name).strip()] = str(path).strip()
    sql_days = _parse_csv(args.sql_days)
    print("loading sql expert candidates ...")
    sql_cand, sql_close, sql_high, sql_low, sql_day_end, sql_ok_days, sql_fail_days = _load_sql_candidates(
        days=sql_days,
        expert_cfg=sql_cfg,
    )
    sql_eval: List[Dict[str, Any]] = []
    if not sql_cand.empty:
        sql_feat, _ = _build_meta_features(sql_cand)
        sql_X = sql_feat[feat_cols].to_numpy(dtype=np.float32, copy=False)
        sql_cand["meta_score"] = full_meta_model.predict_proba(sql_X)[:, 1].astype(np.float64)
        # Evaluate top policy set from full history plus baseline refs.
        top_names = [r["policy"]["name"] for r in full_eval[:10]]
        pol_sql = [p for p in policies if p.name in set(top_names + ["live_raw_ref", "q2_raw_ref", "mix_raw_ref"])]
        sql_eval = _evaluate_policy_set(
            cand=sql_cand,
            close=sql_close,
            high=sql_high,
            low=sql_low,
            day_end=sql_day_end,
            day_ids=sql_ok_days,
            experts=experts,
            policies=pol_sql,
        )

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "experts": {k: asdict(v) for k, v in experts.items()},
        "full_history": {
            "rows_path": args.rows,
            "ohlcv_path": args.ohlcv,
            "n_rows": int(len(rows)),
            "n_days": int(len(full_days)),
            "n_candidates": int(len(full_cand)),
            "meta_features": feat_cols,
            "live_ref_pf": live_ref_pf if np.isfinite(live_ref_pf) else None,
            "results": full_eval,
            "better_than_live_ref_by_pf": full_better_pf[:40],
            "feasible_5to10_trades_zero_after510_le10": full_feasible[:40],
        },
        "sql_oos": {
            "days_requested": sql_days,
            "days_loaded": sql_ok_days,
            "days_failed": sql_fail_days,
            "n_candidates": int(len(sql_cand)),
            "results": sql_eval,
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(str(out_path))


if __name__ == "__main__":
    main()
