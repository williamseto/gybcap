#!/usr/bin/env python
"""Shared sequential-reconciliation helpers for active router research scripts.

This module is a compatibility shim for the current sandbox stack after prune.
It keeps the q_twohead / policy-router research scripts runnable with a single
live-feasible contract:
- causal per-row quality scoring
- day-local top-K router with cooldown/diversity/unresolved controls
- independent and sequential execution simulators
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sandbox import train_frontier_qtwohead_models as tf
from sandbox.analyze_reversal_playback_day import _load_json
from strategies.reversal.level_utils import level_group as _level_group


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _norm_ts(ts: Any) -> Optional[pd.Timestamp]:
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        return t.tz_convert("America/Los_Angeles").tz_localize(None)
    return t


def _minute_from_ts(ts: Optional[pd.Timestamp]) -> int:
    if ts is None:
        return 0
    return int(ts.hour) * 60 + int(ts.minute)


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


@dataclass
class RouterCfg:
    quality_source: str = "q_twohead"
    k_total: int = 6
    q_thr: float = 0.20
    q_slots: int = 6
    c_thr: float = 0.16
    c_start: int = 450
    c_slots: int = 4
    r_thr: float = 0.12
    r_start: int = 600
    r_slots: int = 2
    cooldown_min: int = 5
    inertia_enabled: bool = False
    inertia_global_minute_gap: int = 0
    inertia_level_group_minute_gap: int = 0
    inertia_override_min_q: Optional[float] = None
    inertia_override_min_q_gap: float = 0.0
    diversity_cap: int = 2
    group_unresolved: bool = True
    unresolved_min: int = 10
    gate: float = 0.62
    stop_hi: float = 10.0
    target_hi: float = 40.0
    stop_lo: float = 4.0
    target_lo: float = 8.0
    # blend_policy_q2 controls
    blend_live_cap: int = 5
    blend_live_start: int = 390
    blend_live_end: int = 780
    blend_live_min_q: float = 0.0
    blend_live_flatten_min_q: float = 0.66
    blend_q2_cap: int = 2
    blend_q2_start: int = 450
    blend_q2_end: int = 780
    blend_q2_min_q: float = 0.16
    blend_q2_flatten_min_q: float = 0.16
    early_loss_lock_enabled: bool = False
    early_loss_lock_start_minute: int = 390
    early_loss_lock_end_minute: int = 450
    early_loss_lock_trigger_diff: int = 1
    early_loss_lock_duration_min: int = 45
    early_loss_lock_once_per_day: bool = True


@dataclass
class ExecVariant:
    name: str
    mode: str = "sequential"  # independent | sequential
    opposite_action: str = "ignore"  # ignore | flatten | flip_or_flatten
    same_side_action: str = "hold"  # hold | replace
    same_side_replace_min_q_gain: float = 0.03
    same_side_replace_min_price_improve: float = 0.0
    same_side_min_hold_bars: int = 1
    consume_budget_on_replace: bool = False


@dataclass
class _PolicyCtx:
    episodes_started_today: int = 0
    last_episode_ts: Optional[pd.Timestamp] = None


class _PolicyScorer:
    """Offline scorer for policy_prob quality source (causal, row-by-row)."""

    def __init__(self, model_dir: str, use_ensemble_mean: bool = True) -> None:
        self._model: Optional[xgb.XGBClassifier] = None
        self._ensemble: List[xgb.XGBClassifier] = []
        self._feature_cols: List[str] = []
        self._use_ensemble_mean = bool(use_ensemble_mean)

        md_path = Path(model_dir) / "policy_metadata.json"
        if not md_path.exists():
            return
        try:
            md = json.loads(md_path.read_text())
            self._feature_cols = [str(c) for c in md.get("feature_cols", []) if str(c).strip()]
            self._use_ensemble_mean = bool(md.get("policy_use_ensemble_mean", self._use_ensemble_mean))
            pm = Path(model_dir) / "policy_model.json"
            if pm.exists():
                m = xgb.XGBClassifier()
                m.load_model(str(pm))
                self._model = m
            for ep in sorted(Path(model_dir).glob("policy_ensemble_*.json")):
                em = xgb.XGBClassifier()
                em.load_model(str(ep))
                self._ensemble.append(em)
        except Exception:
            self._model = None
            self._ensemble = []
            self._feature_cols = []

    def _build_row(
        self,
        *,
        row: pd.Series,
        base_prob: float,
        policy_ctx: _PolicyCtx,
    ) -> Optional[np.ndarray]:
        if self._model is None or not self._feature_cols:
            return None

        ts = _norm_ts(row.get("episode_start_dt", row.get("dt")))
        minute = _minute_from_ts(ts)
        is_rth = 390 <= minute < 780

        side = _safe_float(row.get("side"), np.nan)
        if not np.isfinite(side):
            side = _safe_float(row.get("trade_direction"), 0.0)

        minutes_since_prev = 9999.0
        if ts is not None and policy_ctx.last_episode_ts is not None:
            minutes_since_prev = max(
                (ts - policy_ctx.last_episode_ts).total_seconds() / 60.0,
                0.0,
            )

        data: Dict[str, float] = {
            "base_prob": float(base_prob),
            "side": float(side),
            "is_rth_session": 1.0 if is_rth else 0.0,
            "minute_of_day_norm": float(minute / 1440.0),
            "episode_index_so_far": float(max(int(policy_ctx.episodes_started_today), 0)),
            "episode_index_log": float(np.log1p(max(int(policy_ctx.episodes_started_today), 0))),
            "minutes_since_prev_episode": float(minutes_since_prev),
            "minutes_since_prev_episode_log": float(np.log1p(float(minutes_since_prev))),
            "is_opening_30m": 1.0 if minute <= 420 else 0.0,
        }
        for c in self._feature_cols:
            if c in data:
                continue
            if c in row.index and pd.notna(row[c]):
                data[c] = _safe_float(row[c], 0.0)
            else:
                data[c] = 0.0
        arr = np.asarray([data[c] for c in self._feature_cols], dtype=np.float32)
        return arr.reshape(1, -1)

    def predict(
        self,
        *,
        row: pd.Series,
        base_prob: float,
        policy_ctx: _PolicyCtx,
    ) -> Tuple[float, Optional[float]]:
        x = self._build_row(row=row, base_prob=float(base_prob), policy_ctx=policy_ctx)
        if x is None or self._model is None:
            return float(base_prob), None
        prob = float(self._model.predict_proba(x)[0, 1])
        std: Optional[float] = None
        if self._ensemble:
            ens = np.asarray([m.predict_proba(x)[0, 1] for m in self._ensemble], dtype=np.float64)
            if ens.size:
                ens_mean = float(ens.mean())
                std = float(ens.std(ddof=0))
                if self._use_ensemble_mean:
                    prob = ens_mean
        return float(prob), std


def _load_live_router_cfg(path: str) -> Dict[str, Any]:
    return _load_json(str(path))


def _router_cfg_from_live(live_cfg: Dict[str, Any]) -> RouterCfg:
    inertia_override_raw = (
        _safe_float(live_cfg.get("frontier_inertia_override_min_q"), np.nan)
        if live_cfg.get("frontier_inertia_override_min_q") is not None
        else np.nan
    )
    return RouterCfg(
        quality_source=str(live_cfg.get("frontier_quality_source", "q_twohead")),
        k_total=_safe_int(live_cfg.get("frontier_k_total", 6), 6),
        q_thr=_safe_float(live_cfg.get("frontier_q_thr", 0.20), 0.20),
        q_slots=_safe_int(live_cfg.get("frontier_q_slots", 6), 6),
        c_thr=_safe_float(live_cfg.get("frontier_c_thr", 0.16), 0.16),
        c_start=_safe_int(live_cfg.get("frontier_c_start", 450), 450),
        c_slots=_safe_int(live_cfg.get("frontier_c_slots", 4), 4),
        r_thr=_safe_float(live_cfg.get("frontier_r_thr", 0.12), 0.12),
        r_start=_safe_int(live_cfg.get("frontier_r_start", 600), 600),
        r_slots=_safe_int(live_cfg.get("frontier_r_slots", 2), 2),
        cooldown_min=_safe_int(live_cfg.get("frontier_cooldown_min", 5), 5),
        inertia_enabled=bool(live_cfg.get("frontier_inertia_enabled", False)),
        inertia_global_minute_gap=_safe_int(
            live_cfg.get("frontier_inertia_global_minute_gap", 0),
            0,
        ),
        inertia_level_group_minute_gap=_safe_int(
            live_cfg.get("frontier_inertia_level_group_minute_gap", 0),
            0,
        ),
        inertia_override_min_q=(
            float(inertia_override_raw) if np.isfinite(inertia_override_raw) else None
        ),
        inertia_override_min_q_gap=_safe_float(
            live_cfg.get("frontier_inertia_override_min_q_gap", 0.0),
            0.0,
        ),
        diversity_cap=_safe_int(live_cfg.get("frontier_diversity_cap", 2), 2),
        group_unresolved=bool(live_cfg.get("frontier_group_unresolved_enabled", True)),
        unresolved_min=_safe_int(live_cfg.get("frontier_unresolved_min", 10), 10),
        gate=_safe_float(live_cfg.get("frontier_virtual_gate", 0.62), 0.62),
        stop_hi=_safe_float(live_cfg.get("frontier_virtual_stop_hi", 10.0), 10.0),
        target_hi=_safe_float(live_cfg.get("frontier_virtual_target_hi", 40.0), 40.0),
        stop_lo=_safe_float(live_cfg.get("frontier_virtual_stop_lo", 4.0), 4.0),
        target_lo=_safe_float(live_cfg.get("frontier_virtual_target_lo", 8.0), 8.0),
        blend_live_cap=_safe_int(live_cfg.get("frontier_blend_live_cap", 5), 5),
        blend_live_start=_safe_int(live_cfg.get("frontier_blend_live_start", 390), 390),
        blend_live_end=_safe_int(live_cfg.get("frontier_blend_live_end", 780), 780),
        blend_live_min_q=_safe_float(live_cfg.get("frontier_blend_live_min_q", 0.0), 0.0),
        blend_live_flatten_min_q=_safe_float(live_cfg.get("frontier_blend_live_flatten_min_q", 0.66), 0.66),
        blend_q2_cap=_safe_int(live_cfg.get("frontier_blend_q2_cap", 2), 2),
        blend_q2_start=_safe_int(live_cfg.get("frontier_blend_q2_start", 450), 450),
        blend_q2_end=_safe_int(live_cfg.get("frontier_blend_q2_end", 780), 780),
        blend_q2_min_q=_safe_float(live_cfg.get("frontier_blend_q2_min_q", 0.16), 0.16),
        blend_q2_flatten_min_q=_safe_float(live_cfg.get("frontier_blend_q2_flatten_min_q", 0.16), 0.16),
        early_loss_lock_enabled=bool(live_cfg.get("frontier_early_loss_lock_enabled", False)),
        early_loss_lock_start_minute=_safe_int(
            live_cfg.get("frontier_early_loss_lock_start_minute", 390),
            390,
        ),
        early_loss_lock_end_minute=_safe_int(
            live_cfg.get("frontier_early_loss_lock_end_minute", 450),
            450,
        ),
        early_loss_lock_trigger_diff=_safe_int(
            live_cfg.get("frontier_early_loss_lock_trigger_diff", 1),
            1,
        ),
        early_loss_lock_duration_min=_safe_int(
            live_cfg.get("frontier_early_loss_lock_duration_min", 45),
            45,
        ),
        early_loss_lock_once_per_day=bool(
            live_cfg.get("frontier_early_loss_lock_once_per_day", True)
        ),
    )


def _build_variants() -> List[ExecVariant]:
    return [
        ExecVariant(name="independent_ref_live_router", mode="independent", opposite_action="ignore"),
        ExecVariant(name="seq_hold_flatten_v1", mode="sequential", opposite_action="flatten", same_side_action="hold"),
        ExecVariant(
            name="seq_hold_replace_qgain04_px025_v1",
            mode="sequential",
            opposite_action="ignore",
            same_side_action="replace",
            same_side_replace_min_q_gain=0.04,
            same_side_replace_min_price_improve=0.25,
            same_side_min_hold_bars=1,
            consume_budget_on_replace=False,
        ),
        ExecVariant(
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


def _load_rows(rows_parquet: str, q_rows_parquet: str) -> pd.DataFrame:
    rows = pd.read_parquet(rows_parquet).copy().reset_index(drop=True)
    rows = tf._attach_event_prob(rows, q_rows_parquet)
    rows["event_base_prob"] = pd.to_numeric(rows.get("event_base_prob"), errors="coerce").fillna(0.5)

    q_path = Path(q_rows_parquet)
    if q_path.exists():
        q = pd.read_parquet(q_path).copy().reset_index(drop=True)
        key = [c for c in ["trading_day", "episode_start_idx"] if c in rows.columns and c in q.columns]
        if len(key) == 2:
            keep = [c for c in ["event_base_prob", "q_tp", "q_twohead", "q_twohead_regime", "q_twohead_stable", "q_twohead_ev_blend"] if c in q.columns]
            if keep:
                rows = rows.merge(q[key + keep].drop_duplicates(subset=key), on=key, how="left", suffixes=("", "_q"))
                if "event_base_prob_q" in rows.columns:
                    rows["event_base_prob"] = pd.to_numeric(rows["event_base_prob_q"], errors="coerce").fillna(rows["event_base_prob"])
                    rows = rows.drop(columns=["event_base_prob_q"], errors="ignore")

    for c in ["q_tp", "q_twohead", "q_twohead_regime", "q_twohead_stable", "q_twohead_ev_blend"]:
        if c not in rows.columns:
            rows[c] = np.nan
        rows[c] = pd.to_numeric(rows[c], errors="coerce")

    rows["trading_day"] = rows["trading_day"].astype(str)
    rows["episode_start_idx"] = pd.to_numeric(rows["episode_start_idx"], errors="coerce").fillna(-1).astype(np.int64)
    rows["trade_direction"] = pd.to_numeric(rows.get("trade_direction"), errors="coerce").fillna(0).astype(np.int8)
    if "minute_of_day" not in rows.columns:
        ts = pd.to_datetime(rows.get("episode_start_dt"), errors="coerce")
        rows["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).fillna(-1).astype(np.int32)
    else:
        rows["minute_of_day"] = pd.to_numeric(rows["minute_of_day"], errors="coerce").fillna(-1).astype(np.int32)
    return rows


def _load_ohlcv(path: str) -> pd.DataFrame:
    oc = pd.read_parquet(path).copy()
    if "_src_index" in oc.columns:
        oc = oc.sort_values("_src_index").reset_index(drop=True)
    if "minute_of_day" not in oc.columns and "dt" in oc.columns:
        dt = pd.to_datetime(oc["dt"], errors="coerce")
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
        oc["minute_of_day"] = (dt.dt.hour * 60 + dt.dt.minute).astype(np.int32)
    return oc.reset_index(drop=True)


def _add_proximity(rows: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "proximity_pts" in out.columns:
        out["proximity_pts"] = pd.to_numeric(out["proximity_pts"], errors="coerce").fillna(0.0)
        return out
    if "bars_to_nearest_level" in out.columns:
        out["proximity_pts"] = pd.to_numeric(out["bars_to_nearest_level"], errors="coerce").fillna(0.0).abs()
        return out
    out["proximity_pts"] = 0.0
    return out


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


def _build_day_groups(rows: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    w = rows.copy()
    w["trading_day"] = w["trading_day"].astype(str)
    w["episode_start_idx"] = pd.to_numeric(w["episode_start_idx"], errors="coerce").fillna(-1).astype(np.int64)
    w["minute_of_day"] = pd.to_numeric(w["minute_of_day"], errors="coerce").fillna(-1).astype(np.int32)
    w["trade_direction"] = pd.to_numeric(w["trade_direction"], errors="coerce").fillna(0).astype(np.int8)
    mask = (w["minute_of_day"] >= 390) & (w["minute_of_day"] < 780) & (w["trade_direction"] != 0) & (w["episode_start_idx"] >= 0)
    w = w.loc[mask].copy()
    out: List[Tuple[str, np.ndarray]] = []
    for day, g in w.groupby(w["trading_day"].astype(str), sort=False):
        idx = (
            g.sort_values(["minute_of_day", "episode_start_idx"], ascending=[True, True])
            .index.to_numpy(dtype=np.int64, copy=False)
        )
        out.append((str(day), idx))
    return out


def _add_runtime_qtwohead(rows: pd.DataFrame, *, model_dir: str, score_col: str = "q_twohead_runtime") -> pd.DataFrame:
    out = rows.copy()
    md_path = Path(model_dir) / "metadata.json"
    tp_path = Path(model_dir) / "tp_model.json"
    fail_path = Path(model_dir) / "fail_model.json"
    if not (md_path.exists() and tp_path.exists() and fail_path.exists()):
        out[score_col] = np.nan
        return out
    md = json.loads(md_path.read_text())
    feat = [str(c) for c in md.get("feature_cols", []) if str(c).strip()]
    for c in feat:
        if c not in out.columns:
            out[c] = 0.0
    X = out[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    tp = xgb.XGBClassifier()
    tp.load_model(str(tp_path))
    fail = xgb.XGBClassifier()
    fail.load_model(str(fail_path))
    p_tp = tp.predict_proba(X)[:, 1].astype(np.float64)
    p_fail = fail.predict_proba(X)[:, 1].astype(np.float64)
    out[score_col] = np.clip(p_tp * (1.0 - p_fail), 0.0, 1.0)
    return out


def _add_runtime_policy_prob(
    rows: pd.DataFrame,
    *,
    model_dir: str,
    score_col: str = "policy_prob_runtime",
    use_ensemble_mean: Optional[bool] = None,
) -> pd.DataFrame:
    out = rows.copy()
    md_path = Path(model_dir) / "policy_metadata.json"
    model_path = Path(model_dir) / "policy_model.json"
    if not (md_path.exists() and model_path.exists()):
        out[score_col] = np.nan
        return out

    try:
        md = json.loads(md_path.read_text())
        feat = [str(c) for c in md.get("feature_cols", []) if str(c).strip()]
        if not feat:
            out[score_col] = np.nan
            return out
        use_ens = bool(md.get("policy_use_ensemble_mean", True)) if use_ensemble_mean is None else bool(use_ensemble_mean)
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        ens_models: List[xgb.XGBClassifier] = []
        for ep in sorted(Path(model_dir).glob("policy_ensemble_*.json")):
            em = xgb.XGBClassifier()
            em.load_model(str(ep))
            ens_models.append(em)

        n = int(len(out))
        if n <= 0:
            out[score_col] = np.nan
            return out

        base_prob = pd.to_numeric(out.get("event_base_prob"), errors="coerce").fillna(0.5).to_numpy(dtype=np.float32, copy=False)
        side = pd.to_numeric(out.get("side"), errors="coerce").to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(side).all():
            td = pd.to_numeric(out.get("trade_direction"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            bad = ~np.isfinite(side)
            if np.any(bad):
                side[bad] = td[bad]

        if "minute_of_day" in out.columns:
            minute = pd.to_numeric(out["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
        else:
            ts = pd.to_datetime(out.get("episode_start_dt", out.get("dt")), errors="coerce")
            minute = (ts.dt.hour * 60 + ts.dt.minute).fillna(-1).to_numpy(dtype=np.int32, copy=False)

        is_rth = ((minute >= 390) & (minute < 780)).astype(np.float32, copy=False)
        minute_norm = (minute.astype(np.float32, copy=False) / 1440.0)
        is_opening = (minute <= 420).astype(np.float32, copy=False)
        ep_idx = np.zeros(n, dtype=np.float32)
        ep_log = np.zeros(n, dtype=np.float32)
        mins_prev = np.full(n, 9999.0, dtype=np.float32)
        mins_prev_log = np.full(n, float(np.log1p(9999.0)), dtype=np.float32)

        X = np.zeros((n, len(feat)), dtype=np.float32)
        for j, col in enumerate(feat):
            if col == "base_prob":
                X[:, j] = base_prob
            elif col == "side":
                X[:, j] = side
            elif col == "is_rth_session":
                X[:, j] = is_rth
            elif col == "minute_of_day_norm":
                X[:, j] = minute_norm
            elif col == "episode_index_so_far":
                X[:, j] = ep_idx
            elif col == "episode_index_log":
                X[:, j] = ep_log
            elif col == "minutes_since_prev_episode":
                X[:, j] = mins_prev
            elif col == "minutes_since_prev_episode_log":
                X[:, j] = mins_prev_log
            elif col == "is_opening_30m":
                X[:, j] = is_opening
            elif col in out.columns:
                X[:, j] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
            else:
                X[:, j] = 0.0

        prob = model.predict_proba(X)[:, 1].astype(np.float64)
        if ens_models:
            ens = np.column_stack([m.predict_proba(X)[:, 1] for m in ens_models]).astype(np.float64)
            if use_ens:
                prob = ens.mean(axis=1)
        out[score_col] = np.clip(prob, 0.0, 1.0)
        return out
    except Exception:
        out[score_col] = np.nan
        return out


def _quality_for_row(
    *,
    row: pd.Series,
    router_cfg: RouterCfg,
    policy_scorer: Optional[_PolicyScorer],
    policy_ctx: _PolicyCtx,
) -> Tuple[float, Optional[Dict[str, float]]]:
    src = str(router_cfg.quality_source).strip().lower()
    base_prob = _safe_float(row.get("event_base_prob"), 0.5)
    if src == "policy_prob":
        p_cached = _safe_float(row.get("policy_prob_runtime"), np.nan)
        if np.isfinite(p_cached):
            return float(p_cached), {"policy": float(p_cached)}
        if policy_scorer is None:
            return base_prob, None
        p, _ = policy_scorer.predict(row=row, base_prob=base_prob, policy_ctx=policy_ctx)
        return float(p), {"policy": float(p)}
    if src == "blend_policy_q2":
        minute = _safe_int(row.get("minute_of_day"), -1)
        live_q = _safe_float(row.get("policy_prob_runtime"), np.nan)
        if not np.isfinite(live_q):
            live_q = float(base_prob)
        if policy_scorer is not None and (not np.isfinite(_safe_float(row.get("policy_prob_runtime"), np.nan))):
            p, _ = policy_scorer.predict(row=row, base_prob=base_prob, policy_ctx=policy_ctx)
            live_q = float(p)

        q2_q = _safe_float(row.get("q_twohead_runtime"), np.nan)
        if not np.isfinite(q2_q):
            q2_q = _safe_float(row.get("q_twohead"), np.nan)

        live_ok = (
            minute >= int(router_cfg.blend_live_start)
            and minute < int(router_cfg.blend_live_end)
            and np.isfinite(live_q)
            and float(live_q) >= float(router_cfg.blend_live_min_q)
        )
        q2_ok = (
            minute >= int(router_cfg.blend_q2_start)
            and minute < int(router_cfg.blend_q2_end)
            and np.isfinite(q2_q)
            and float(q2_q) >= float(router_cfg.blend_q2_min_q)
        )
        live_score = float(live_q) if live_ok else float(np.nan)
        q2_score = float(q2_q) if q2_ok else float(np.nan)
        if np.isfinite(q2_score) and ((not np.isfinite(live_score)) or q2_score > live_score):
            return float(q2_score), {"live": live_score, "q2": q2_score}
        if np.isfinite(live_score):
            return float(live_score), {"live": live_score, "q2": q2_score}
        return float(np.nan), {"live": live_score, "q2": q2_score}
    s = _safe_float(row.get(src), np.nan)
    if np.isfinite(s):
        return float(s), None
    if src == "q_twohead_runtime":
        s2 = _safe_float(row.get("q_twohead_runtime"), np.nan)
        if np.isfinite(s2):
            return float(s2), None
    return float(base_prob), None


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


def _simulate_variant(
    *,
    rows: pd.DataFrame,
    day_groups: List[Tuple[str, np.ndarray]],
    day_end: Dict[str, int],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    router_cfg: RouterCfg,
    variant: ExecVariant,
    policy_scorer: Optional[_PolicyScorer],
    return_state: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    quality_src = str(router_cfg.quality_source).strip().lower()
    chosen = np.zeros(len(rows), dtype=bool)
    q_out = np.full(len(rows), np.nan, dtype=np.float64)
    lane_out = np.full(len(rows), "", dtype=object)
    flatten_min_q_out = np.full(len(rows), np.nan, dtype=np.float64)

    lane_counts: Dict[str, int] = {"q": 0, "c": 0, "r": 0}
    action_counts: Dict[str, int] = {}
    trades: List[Dict[str, Any]] = []
    selected_dir_counts: Dict[int, int] = {1: 0, -1: 0}
    selected_level_counts: Dict[str, int] = {}
    day_unique_levels: List[int] = []
    day_level_hhi: List[float] = []
    day_long_share: List[float] = []
    day_short_share: List[float] = []
    tod_bucket_counts: Dict[str, int] = {
        "open_60": 0,
        "post_open_60": 0,
        "late_morning_120": 0,
        "afternoon_150": 0,
    }

    for day, idxs in day_groups:
        q_used = c_used = r_used = 0
        accepted = 0
        last_minute = -10_000
        blend_live_used = 0
        blend_q2_used = 0
        level_counts: Dict[str, int] = {}
        unresolved_until: Dict[Tuple[str, int], int] = {}
        inertia_last_global: Optional[Dict[str, float]] = None
        inertia_last_by_group: Dict[str, Dict[str, float]] = {}
        ctx = _PolicyCtx()
        realized_wins = 0
        realized_losses = 0
        early_loss_lock_until_minute = -1
        early_loss_lock_triggered_today = False
        virtual_open_trades: List[Dict[str, Any]] = []

        day_idxs: List[int] = []
        for pos in idxs:
            row = rows.loc[int(pos)]
            minute = _safe_int(row.get("minute_of_day"), -1)
            if minute < 390 or minute >= 780:
                continue
            bar_num = _safe_int(row.get("episode_start_idx"), -1)
            if 0 <= bar_num < len(high):
                hi_now = _safe_float(high[int(bar_num)], np.nan)
                lo_now = _safe_float(low[int(bar_num)], np.nan)
                for tr in virtual_open_trades:
                    if bool(tr.get("closed", False)):
                        continue
                    entry_bar = _safe_int(tr.get("entry_bar"), -1)
                    if bar_num <= entry_bar:
                        continue
                    direction_v = _safe_int(tr.get("direction"), 0)
                    stop_px = _safe_float(tr.get("stop_px"), np.nan)
                    target_px = _safe_float(tr.get("target_px"), np.nan)
                    if direction_v == 0 or (not np.isfinite(stop_px)) or (not np.isfinite(target_px)):
                        continue
                    hit: Optional[str] = None
                    if direction_v > 0:
                        if lo_now <= stop_px and hi_now >= target_px:
                            hit = "stop_tie"
                        elif lo_now <= stop_px:
                            hit = "stop"
                        elif hi_now >= target_px:
                            hit = "target"
                    else:
                        if hi_now >= stop_px and lo_now <= target_px:
                            hit = "stop_tie"
                        elif hi_now >= stop_px:
                            hit = "stop"
                        elif lo_now <= target_px:
                            hit = "target"
                    if hit is None:
                        continue
                    tr["closed"] = True
                    tr["exit_reason"] = str(hit)
                    if str(hit).startswith("stop"):
                        realized_losses += 1
                    else:
                        realized_wins += 1

            if bool(router_cfg.early_loss_lock_enabled):
                if int(early_loss_lock_until_minute) >= int(minute):
                    continue
                if not (
                    bool(router_cfg.early_loss_lock_once_per_day)
                    and bool(early_loss_lock_triggered_today)
                ):
                    lock_start = _safe_int(router_cfg.early_loss_lock_start_minute, 390)
                    lock_end = _safe_int(router_cfg.early_loss_lock_end_minute, 450)
                    if lock_start > lock_end:
                        lock_start, lock_end = lock_end, lock_start
                    trigger_diff = max(_safe_int(router_cfg.early_loss_lock_trigger_diff, 1), 1)
                    lock_dur = max(_safe_int(router_cfg.early_loss_lock_duration_min, 45), 0)
                    if (
                        int(minute) >= int(lock_start)
                        and int(minute) <= int(lock_end)
                        and lock_dur > 0
                        and (int(realized_losses) - int(realized_wins)) >= int(trigger_diff)
                    ):
                        early_loss_lock_until_minute = min(int(minute) + int(lock_dur), 779)
                        early_loss_lock_triggered_today = True
                        continue
            score, score_meta = _quality_for_row(
                row=row,
                router_cfg=router_cfg,
                policy_scorer=policy_scorer,
                policy_ctx=ctx,
            )
            blend_expert: Optional[str] = None
            if quality_src == "blend_policy_q2":
                live_s = float(np.nan)
                q2_s = float(np.nan)
                if isinstance(score_meta, dict):
                    live_s = _safe_float(score_meta.get("live"), np.nan)
                    q2_s = _safe_float(score_meta.get("q2"), np.nan)
                cand_scores: List[Tuple[str, float]] = []
                if np.isfinite(live_s) and (
                    int(router_cfg.blend_live_cap) <= 0 or int(blend_live_used) < int(router_cfg.blend_live_cap)
                ):
                    cand_scores.append(("live", float(live_s)))
                if np.isfinite(q2_s) and (
                    int(router_cfg.blend_q2_cap) <= 0 or int(blend_q2_used) < int(router_cfg.blend_q2_cap)
                ):
                    cand_scores.append(("q2", float(q2_s)))
                if not cand_scores:
                    continue
                cand_scores = sorted(cand_scores, key=lambda x: float(x[1]), reverse=True)
                blend_expert = str(cand_scores[0][0])
                score = float(cand_scores[0][1])

            if not np.isfinite(score):
                continue
            q_out[int(pos)] = float(score)

            lane: Optional[str] = None
            if score >= float(router_cfg.q_thr) and q_used < int(router_cfg.q_slots):
                lane = "q"
            elif minute >= int(router_cfg.c_start) and score >= float(router_cfg.c_thr) and c_used < int(router_cfg.c_slots):
                lane = "c"
            elif minute >= int(router_cfg.r_start) and score >= float(router_cfg.r_thr) and r_used < int(router_cfg.r_slots):
                lane = "r"
            if lane is None:
                continue
            if accepted >= int(router_cfg.k_total):
                continue
            if (int(minute) - int(last_minute)) < int(router_cfg.cooldown_min):
                continue
            lvl = str(row.get("nearest_level_name", ""))
            direction = _safe_int(row.get("trade_direction"), 0)
            if int(router_cfg.diversity_cap) > 0:
                if int(level_counts.get(lvl, 0)) >= int(router_cfg.diversity_cap):
                    continue
            if bool(router_cfg.group_unresolved):
                key = (lvl, int(direction))
                if int(unresolved_until.get(key, -10_000)) >= int(minute):
                    continue
                unresolved_until[key] = int(minute) + int(router_cfg.unresolved_min)

            if bool(router_cfg.inertia_enabled):
                block = False

                def _override_ok(last_q: float) -> bool:
                    has_override = (
                        router_cfg.inertia_override_min_q is not None
                        or float(router_cfg.inertia_override_min_q_gap) > 0.0
                    )
                    if not has_override:
                        return False
                    if (
                        router_cfg.inertia_override_min_q is not None
                        and float(score) < float(router_cfg.inertia_override_min_q)
                    ):
                        return False
                    if float(router_cfg.inertia_override_min_q_gap) > 0.0 and (
                        float(score) - float(last_q)
                    ) < float(router_cfg.inertia_override_min_q_gap):
                        return False
                    return True

                if int(router_cfg.inertia_level_group_minute_gap) > 0:
                    grp = _level_group(lvl)
                    last_grp = inertia_last_by_group.get(str(grp))
                    if last_grp is not None and int(last_grp.get("direction", 0)) != int(direction):
                        dt = int(minute) - int(last_grp.get("minute", -10_000))
                        if dt < int(router_cfg.inertia_level_group_minute_gap):
                            if not _override_ok(float(last_grp.get("quality", 0.0))):
                                block = True
                if (not block) and int(router_cfg.inertia_global_minute_gap) > 0 and inertia_last_global is not None:
                    if int(inertia_last_global.get("direction", 0)) != int(direction):
                        dt = int(minute) - int(inertia_last_global.get("minute", -10_000))
                        if dt < int(router_cfg.inertia_global_minute_gap):
                            if not _override_ok(float(inertia_last_global.get("quality", 0.0))):
                                block = True
                if block:
                    continue

            chosen[int(pos)] = True
            lane_out[int(pos)] = str(lane)
            accepted += 1
            last_minute = int(minute)
            level_counts[lvl] = int(level_counts.get(lvl, 0)) + 1
            selected_level_counts[str(lvl)] = int(selected_level_counts.get(str(lvl), 0)) + 1
            selected_dir_counts[int(direction)] = int(selected_dir_counts.get(int(direction), 0)) + 1
            if lane == "q":
                q_used += 1
            elif lane == "c":
                c_used += 1
            else:
                r_used += 1
            if blend_expert == "live":
                blend_live_used += 1
            elif blend_expert == "q2":
                blend_q2_used += 1
            lane_counts[str(lane)] = int(lane_counts.get(str(lane), 0)) + 1
            day_idxs.append(int(pos))
            if quality_src == "blend_policy_q2":
                if blend_expert == "q2":
                    flatten_min_q_out[int(pos)] = float(router_cfg.blend_q2_flatten_min_q)
                elif blend_expert == "live":
                    flatten_min_q_out[int(pos)] = float(router_cfg.blend_live_flatten_min_q)

            entry_idx_v = _safe_int(row.get("episode_start_idx"), -1)
            direction_v = _safe_int(row.get("trade_direction"), 0)
            if 0 <= entry_idx_v < len(close) and direction_v != 0:
                entry_px = _safe_float(close[int(entry_idx_v)], np.nan)
                if np.isfinite(entry_px):
                    hi_lane = float(score) >= float(router_cfg.gate)
                    stop_pts = float(router_cfg.stop_hi if hi_lane else router_cfg.stop_lo)
                    target_pts = float(router_cfg.target_hi if hi_lane else router_cfg.target_lo)
                    if stop_pts > 0.0 and target_pts > 0.0:
                        if int(direction_v) > 0:
                            stop_px = float(entry_px - stop_pts)
                            target_px = float(entry_px + target_pts)
                        else:
                            stop_px = float(entry_px + stop_pts)
                            target_px = float(entry_px - target_pts)
                        virtual_open_trades.append(
                            {
                                "entry_bar": int(entry_idx_v),
                                "direction": int(direction_v),
                                "stop_px": float(stop_px),
                                "target_px": float(target_px),
                                "closed": False,
                            }
                        )

            rec = {
                "minute": float(int(minute)),
                "direction": float(int(direction)),
                "quality": float(score),
            }
            inertia_last_global = dict(rec)
            inertia_last_by_group[str(_level_group(lvl))] = dict(rec)

            if 390 <= int(minute) < 450:
                tod_bucket_counts["open_60"] = int(tod_bucket_counts["open_60"]) + 1
            elif 450 <= int(minute) < 510:
                tod_bucket_counts["post_open_60"] = int(tod_bucket_counts["post_open_60"]) + 1
            elif 510 <= int(minute) < 630:
                tod_bucket_counts["late_morning_120"] = int(tod_bucket_counts["late_morning_120"]) + 1
            elif 630 <= int(minute) < 780:
                tod_bucket_counts["afternoon_150"] = int(tod_bucket_counts["afternoon_150"]) + 1

            if quality_src == "policy_prob":
                ts = _norm_ts(row.get("episode_start_dt"))
                ctx.episodes_started_today += 1
                if ts is not None:
                    ctx.last_episode_ts = ts

        if level_counts:
            cnt = np.asarray(list(level_counts.values()), dtype=np.float64)
            p = cnt / max(float(cnt.sum()), 1.0)
            day_level_hhi.append(float(np.sum(np.square(p))))
            day_unique_levels.append(int(len(level_counts)))
            # Direction split from selected events on this day.
            day_rows = rows.loc[day_idxs, "trade_direction"] if day_idxs else pd.Series(dtype=float)
            dvals = pd.to_numeric(day_rows, errors="coerce").fillna(0).astype(int)
            long_n = int((dvals > 0).sum())
            short_n = int((dvals < 0).sum())
            denom = max(long_n + short_n, 1)
            day_long_share.append(float(long_n / denom))
            day_short_share.append(float(short_n / denom))

        if not day_idxs:
            continue

        day_events = (
            rows.loc[day_idxs, ["episode_start_idx", "trade_direction", "trading_day", "nearest_level_name"]]
            .copy()
            .reset_index()
            .rename(columns={"index": "row_idx"})
        )
        day_events["quality"] = q_out[day_events["row_idx"].to_numpy(dtype=np.int64, copy=False)]
        day_events["lane"] = lane_out[day_events["row_idx"].to_numpy(dtype=np.int64, copy=False)]
        day_events["flatten_min_q"] = flatten_min_q_out[
            day_events["row_idx"].to_numpy(dtype=np.int64, copy=False)
        ]
        day_events["minute_of_day"] = pd.to_numeric(
            rows.loc[day_events["row_idx"].to_numpy(dtype=np.int64, copy=False), "minute_of_day"],
            errors="coerce",
        ).fillna(-1).to_numpy(dtype=np.int32, copy=False)
        day_events = day_events.sort_values(["episode_start_idx", "quality"], ascending=[True, False]).reset_index(drop=True)

        if str(variant.mode).lower() == "independent":
            for _, ev in day_events.iterrows():
                entry_idx = _safe_int(ev["episode_start_idx"], -1)
                if entry_idx < 0 or entry_idx >= len(close):
                    continue
                end_idx = int(day_end.get(str(day), len(close) - 1))
                if end_idx < entry_idx:
                    continue
                qv = _safe_float(ev["quality"], 0.5)
                hi_lane = qv >= float(router_cfg.gate)
                stop_pts = float(router_cfg.stop_hi if hi_lane else router_cfg.stop_lo)
                target_pts = float(router_cfg.target_hi if hi_lane else router_cfg.target_lo)
                pnl, ex_idx, reason = _simulate_trade(
                    entry_idx=entry_idx,
                    entry_price=float(close[entry_idx]),
                    direction=_safe_int(ev["trade_direction"], 0),
                    stop_pts=stop_pts,
                    target_pts=target_pts,
                    end_idx=end_idx,
                    close=close,
                    high=high,
                    low=low,
                )
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(entry_idx),
                        "exit_idx": int(ex_idx),
                        "pnl": float(pnl),
                        "exit_reason": str(reason),
                        "lane": "high" if hi_lane else "low",
                    }
                )
            continue

        # Sequential mode.
        pos: Optional[Dict[str, Any]] = None
        next_scan = -1
        end_idx = int(day_end.get(str(day), len(close) - 1))
        for _, ev in day_events.iterrows():
            i = _safe_int(ev["episode_start_idx"], -1)
            if i < 0 or i >= len(close) or i > end_idx:
                continue
            qv = _safe_float(ev["quality"], 0.5)
            direction = _safe_int(ev["trade_direction"], 0)
            lane_hi = qv >= float(router_cfg.gate)
            stop_pts = float(router_cfg.stop_hi if lane_hi else router_cfg.stop_lo)
            target_pts = float(router_cfg.target_hi if lane_hi else router_cfg.target_lo)
            lane_name = "high" if lane_hi else "low"

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
                            "lane": str(pos["lane"]),
                        }
                    )
                    pos = None
                    next_scan = int(t_exit) + 1
                else:
                    next_scan = int(i) + 1

            if pos is None:
                pos = {
                    "entry_idx": int(i),
                    "entry_price": float(close[i]),
                    "direction": int(direction),
                    "stop_pts": float(stop_pts),
                    "target_pts": float(target_pts),
                    "lane": str(lane_name),
                    "quality": float(qv),
                }
                action_counts["entry"] = int(action_counts.get("entry", 0)) + 1
                continue

            same_dir = int(direction) == int(pos["direction"])
            if same_dir:
                if str(variant.same_side_action).lower() != "replace":
                    continue
                held = int(i) - int(pos["entry_idx"])
                if held < int(variant.same_side_min_hold_bars):
                    continue
                q_gain = float(qv) - float(pos["quality"])
                if q_gain < float(variant.same_side_replace_min_q_gain):
                    continue
                price_improve = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
                if price_improve < float(variant.same_side_replace_min_price_improve):
                    continue
                pnl = float(price_improve)
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(pos["entry_idx"]),
                        "exit_idx": int(i),
                        "pnl": float(pnl),
                        "exit_reason": "replace_signal",
                        "lane": str(pos["lane"]),
                    }
                )
                pos = {
                    "entry_idx": int(i),
                    "entry_price": float(close[i]),
                    "direction": int(direction),
                    "stop_pts": float(stop_pts),
                    "target_pts": float(target_pts),
                    "lane": str(lane_name),
                    "quality": float(qv),
                }
                action_counts["replace"] = int(action_counts.get("replace", 0)) + 1
                continue

            # Opposite direction.
            mode = str(variant.opposite_action).lower()
            if mode == "ignore":
                continue
            flatten_min_q = _safe_float(ev.get("flatten_min_q"), np.nan)
            if np.isfinite(flatten_min_q) and float(qv) < float(flatten_min_q):
                continue
            pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
            exit_reason = "flip_signal" if mode == "flip_or_flatten" else "flatten_signal"
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(pos["entry_idx"]),
                    "exit_idx": int(i),
                    "pnl": float(pnl),
                    "exit_reason": str(exit_reason),
                    "lane": str(pos["lane"]),
                }
            )
            action_counts["flatten"] = int(action_counts.get("flatten", 0)) + 1
            if mode == "flip_or_flatten":
                pos = {
                    "entry_idx": int(i),
                    "entry_price": float(close[i]),
                    "direction": int(direction),
                    "stop_pts": float(stop_pts),
                    "target_pts": float(target_pts),
                    "lane": str(lane_name),
                    "quality": float(qv),
                }
                action_counts["flip"] = int(action_counts.get("flip", 0)) + 1
            else:
                pos = None

        if pos is not None and end_idx >= 0:
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
                pnl = float((float(close[end_idx]) - float(pos["entry_price"])) * float(pos["direction"]))
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(pos["entry_idx"]),
                        "exit_idx": int(end_idx),
                        "pnl": float(pnl),
                        "exit_reason": "timed",
                        "lane": str(pos["lane"]),
                    }
                )
            else:
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(pos["entry_idx"]),
                        "exit_idx": int(t_exit),
                        "pnl": float(t_pnl),
                        "exit_reason": str(t_reason),
                        "lane": str(pos["lane"]),
                    }
                )

    tdf = pd.DataFrame(trades)
    total_selected = int(chosen.sum())
    long_sel = int(selected_dir_counts.get(1, 0))
    short_sel = int(selected_dir_counts.get(-1, 0))
    denom_dir = max(long_sel + short_sel, 1)
    denom_tod = max(sum(int(v) for v in tod_bucket_counts.values()), 1)
    diag = {
        "selected": total_selected,
        "lane_counts": lane_counts,
        "action_counts": action_counts,
        "diversity": {
            "unique_levels_total": int(len(selected_level_counts)),
            "mean_unique_levels_per_day": float(np.mean(day_unique_levels)) if day_unique_levels else 0.0,
            "mean_level_hhi_per_day": float(np.mean(day_level_hhi)) if day_level_hhi else 0.0,
            "long_selected_share": float(long_sel / denom_dir),
            "short_selected_share": float(short_sel / denom_dir),
            "mean_long_share_per_day": float(np.mean(day_long_share)) if day_long_share else 0.0,
            "mean_short_share_per_day": float(np.mean(day_short_share)) if day_short_share else 0.0,
            "open_60_share": float(int(tod_bucket_counts["open_60"]) / denom_tod),
            "post_open_60_share": float(int(tod_bucket_counts["post_open_60"]) / denom_tod),
            "late_morning_120_share": float(int(tod_bucket_counts["late_morning_120"]) / denom_tod),
            "afternoon_150_share": float(int(tod_bucket_counts["afternoon_150"]) / denom_tod),
        },
    }
    if bool(return_state):
        diag["state"] = {
            "quality_by_row": q_out.copy(),
            "router_lane_by_row": lane_out.copy(),
        }
    return chosen, tdf, diag


def _label_metrics(
    rows: pd.DataFrame,
    chosen: np.ndarray,
    day_groups: List[Tuple[str, np.ndarray]],
) -> Dict[str, Any]:
    pick = np.asarray(chosen, dtype=bool)
    all_days = [str(d) for d, _ in day_groups]
    if len(rows) == 0 or len(all_days) == 0:
        return {
            "n_selected": 0,
            "mean_selected_per_day": 0.0,
            "precision_labeled": None,
            "recall_labeled": None,
            "rth_no_tp_day_rate": None,
            "rth_no_large_tp_day_rate": None,
        }

    w = rows.copy()
    w["trading_day"] = w["trading_day"].astype(str)
    tp = pd.to_numeric(w.get("tp_now_target"), errors="coerce")
    large = pd.to_numeric(w.get("large_tp_now_target"), errors="coerce")

    selected = w.loc[pick].copy()
    precision = None
    if not selected.empty and tp is not None:
        precision = float((pd.to_numeric(selected.get("tp_now_target"), errors="coerce") == 1.0).mean())

    recall = None
    if tp is not None:
        pos_total = int((tp == 1.0).sum())
        pos_sel = int((pd.to_numeric(selected.get("tp_now_target"), errors="coerce") == 1.0).sum()) if not selected.empty else 0
        recall = float(pos_sel / pos_total) if pos_total > 0 else None

    sel_per_day = selected.groupby(selected["trading_day"].astype(str)).size().to_dict() if not selected.empty else {}

    tp_days = set()
    large_days = set()
    if not selected.empty:
        stp = pd.to_numeric(selected.get("tp_now_target"), errors="coerce")
        tp_days = set(selected.loc[stp == 1.0, "trading_day"].astype(str).tolist())
        if large is not None and "large_tp_now_target" in selected.columns:
            slg = pd.to_numeric(selected.get("large_tp_now_target"), errors="coerce")
            large_days = set(selected.loc[slg == 1.0, "trading_day"].astype(str).tolist())
    if not large_days:
        large_days = set(tp_days)

    return {
        "n_selected": int(pick.sum()),
        "mean_selected_per_day": float(np.mean([int(sel_per_day.get(d, 0)) for d in all_days])),
        "precision_labeled": precision,
        "recall_labeled": recall,
        "rth_no_tp_day_rate": float(np.mean([d not in tp_days for d in all_days])),
        "rth_no_large_tp_day_rate": float(np.mean([d not in large_days for d in all_days])),
    }


def _exec_metrics(trades: pd.DataFrame, day_ids: List[str]) -> Dict[str, Any]:
    day_pnl = pd.Series(0.0, index=pd.Index(day_ids, dtype=str))
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
            "mean_pnl_per_trade": None,
            "win_rate": None,
            "zero_trade_day_rate": 1.0 if day_ids else None,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "timed_exit_rate": None,
            "manual_exit_rate": None,
        }
    gp = float(trades["pnl"].clip(lower=0.0).sum())
    gl = float((-trades["pnl"].clip(upper=0.0)).sum())
    exits = trades["exit_reason"].astype(str)
    return {
        "n_trades": int(len(trades)),
        "mean_trades_per_day": float(len(trades) / max(len(day_ids), 1)),
        "mean_pnl_per_day": mean_day,
        "annualized_daily_sharpe": sharpe,
        "profit_factor": (gp / gl) if gl > 1e-9 else None,
        "mean_pnl_per_trade": float(trades["pnl"].mean()),
        "win_rate": float((trades["pnl"] > 0.0).mean()),
        "zero_trade_day_rate": float(np.mean(day_pnl.to_numpy() == 0.0)) if day_ids else None,
        "target_hit_rate": float((exits == "target").mean()),
        "stop_hit_rate": float(exits.str.startswith("stop").mean()),
        "timed_exit_rate": float((exits == "timed").mean()),
        "manual_exit_rate": float(exits.isin(["flip_signal", "flatten_signal", "replace_signal"]).mean()),
    }
