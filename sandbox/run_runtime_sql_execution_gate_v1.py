#!/usr/bin/env python
"""Run runtime SQL playback days and evaluate execution metrics from emitted signals.

This gate runs the real runtime strategy path (`strategy.process`) on SQL day bars,
then applies an execution simulator directly to emitted actions:

- independent mode: each entry signal is an independent trade
- sequential mode: single open position with entry/flip/flatten actions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sandbox.analyze_reversal_playback_day import (
    _build_strategy,
    _load_causal_history,
    _load_day_bars,
    _load_json,
)


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _simulate_trade_independent(
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


def _metrics(trades: pd.DataFrame, day_ids: List[str]) -> Dict[str, Any]:
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
            "zero_trade_day_rate": 1.0,
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
        "zero_trade_day_rate": float(np.mean(day_pnl.to_numpy() == 0.0)),
        "target_hit_rate": float((exits == "target").mean()),
        "stop_hit_rate": float(exits.str.startswith("stop").mean()),
        "timed_exit_rate": float((exits == "timed").mean()),
        "manual_exit_rate": float(exits.isin(["flip_signal", "flatten_signal", "replace_signal"]).mean()),
    }


def _contract_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "gate": _safe_float(cfg.get("frontier_virtual_gate", 0.62), 0.62),
        "stop_hi": _safe_float(cfg.get("frontier_virtual_stop_hi", 10.0), 10.0),
        "target_hi": _safe_float(cfg.get("frontier_virtual_target_hi", 40.0), 40.0),
        "stop_lo": _safe_float(cfg.get("frontier_virtual_stop_lo", 4.0), 4.0),
        "target_lo": _safe_float(cfg.get("frontier_virtual_target_lo", 8.0), 8.0),
    }


def _simulate_day_from_signals(
    *,
    day: str,
    bars: pd.DataFrame,
    signals: List[Any],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    def _norm_ts(ts: Any) -> Optional[pd.Timestamp]:
        t = pd.Timestamp(ts)
        if pd.isna(t):
            return None
        if t.tzinfo is not None:
            return t.tz_convert("America/Los_Angeles").tz_localize(None)
        return t

    b = bars.copy()
    if "dt" not in b.columns:
        b = b.reset_index()
    b["dt"] = pd.to_datetime(b["dt"], errors="coerce")
    b = b.sort_values("dt").reset_index(drop=True)
    if b.empty:
        return pd.DataFrame(), {}

    close = pd.to_numeric(b["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(b["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(b["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    idx_map: Dict[pd.Timestamp, int] = {}
    for i, ts in enumerate(b["dt"]):
        nt = _norm_ts(ts)
        if nt is not None:
            idx_map[nt] = int(i)
    end_idx = int(len(b) - 1)

    contract = _contract_from_cfg(cfg)
    exec_mode = str(cfg.get("frontier_execution_mode", "independent")).strip().lower() or "independent"
    replace_enabled = bool(cfg.get("frontier_same_side_replace_enabled", False))
    replace_min_q_gain = _safe_float(cfg.get("frontier_same_side_replace_min_q_gain", 0.03), 0.03)
    replace_min_price_improve = _safe_float(cfg.get("frontier_same_side_replace_min_price_improve", 0.25), 0.25)
    replace_min_hold_bars = int(cfg.get("frontier_same_side_replace_min_hold_bars", 1))

    events: List[Dict[str, Any]] = []
    action_counts: Dict[str, int] = {}
    for s in signals:
        md = dict(getattr(s, "metadata", {}) or {})
        ts = _norm_ts(getattr(s, "entry_ts"))
        i = idx_map.get(ts) if ts is not None else None
        if i is None:
            continue
        direction_str = str(getattr(s, "direction", "bull")).strip().lower()
        d = 1 if direction_str == "bull" else -1
        action = str(md.get("execution_action", "entry")).strip().lower() or "entry"
        action_counts[action] = int(action_counts.get(action, 0)) + 1
        q = _safe_float(md.get("quality_score", getattr(s, "pred_proba", 0.5)), 0.5)
        lane = "high" if q >= float(contract["gate"]) else "low"
        stop_pts = float(contract["stop_hi"] if lane == "high" else contract["stop_lo"])
        target_pts = float(contract["target_hi"] if lane == "high" else contract["target_lo"])
        events.append(
            {
                "idx": int(i),
                "direction": int(d),
                "action": action,
                "quality": float(q),
                "lane": lane,
                "stop_pts": float(stop_pts),
                "target_pts": float(target_pts),
                "level_name": str(getattr(s, "level_name", "")),
            }
        )

    events = sorted(events, key=lambda x: (int(x["idx"]), 0 if x["action"] in {"flatten", "flip"} else 1))
    trades: List[Dict[str, Any]] = []

    if exec_mode != "sequential":
        for ev in events:
            if ev["action"] == "flatten":
                continue
            pnl, exit_idx, exit_reason = _simulate_trade_independent(
                entry_idx=int(ev["idx"]),
                entry_price=float(close[int(ev["idx"])]),
                direction=int(ev["direction"]),
                stop_pts=float(ev["stop_pts"]),
                target_pts=float(ev["target_pts"]),
                end_idx=end_idx,
                close=close,
                high=high,
                low=low,
            )
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(ev["idx"]),
                    "exit_idx": int(exit_idx),
                    "pnl": float(pnl),
                    "exit_reason": str(exit_reason),
                    "lane": str(ev["lane"]),
                }
            )
        return pd.DataFrame(trades), action_counts

    pos: Optional[Dict[str, Any]] = None
    for ev in events:
        i = int(ev["idx"])
        if pos is not None:
            for j in range(int(pos["next_check_idx"]), i + 1):
                if j <= int(pos["entry_idx"]):
                    continue
                hit = None
                if int(pos["direction"]) > 0:
                    if low[j] <= float(pos["stop_px"]) and high[j] >= float(pos["target_px"]):
                        hit = "stop_tie"
                    elif low[j] <= float(pos["stop_px"]):
                        hit = "stop"
                    elif high[j] >= float(pos["target_px"]):
                        hit = "target"
                else:
                    if high[j] >= float(pos["stop_px"]) and low[j] <= float(pos["target_px"]):
                        hit = "stop_tie"
                    elif high[j] >= float(pos["stop_px"]):
                        hit = "stop"
                    elif low[j] <= float(pos["target_px"]):
                        hit = "target"
                if hit is None:
                    continue
                exit_px = float(pos["stop_px"]) if str(hit).startswith("stop") else float(pos["target_px"])
                pnl = float((exit_px - float(pos["entry_price"])) * float(pos["direction"]))
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(pos["entry_idx"]),
                        "exit_idx": int(j),
                        "pnl": float(pnl),
                        "exit_reason": str(hit),
                        "lane": str(pos["lane"]),
                    }
                )
                pos = None
                break
            if pos is not None:
                pos["next_check_idx"] = int(i + 1)

        action = str(ev["action"])
        if action == "flatten":
            if pos is not None:
                pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
                trades.append(
                    {
                        "trading_day": str(day),
                        "entry_idx": int(pos["entry_idx"]),
                        "exit_idx": int(i),
                        "pnl": float(pnl),
                        "exit_reason": "flatten_signal",
                        "lane": str(pos["lane"]),
                    }
                )
                pos = None
            continue

        if action == "flip" and pos is not None:
            pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(pos["entry_idx"]),
                    "exit_idx": int(i),
                    "pnl": float(pnl),
                    "exit_reason": "flip_signal",
                    "lane": str(pos["lane"]),
                }
            )
            pos = None

        if pos is not None and int(ev["direction"]) == int(pos["direction"]) and bool(replace_enabled):
            hold_bars = max(int(i) - int(pos["entry_idx"]), 0)
            q_gain = float(ev["quality"]) - float(pos.get("entry_quality", ev["quality"]))
            improves_q = q_gain >= float(replace_min_q_gain)
            improves_px = False
            if int(pos["direction"]) > 0:
                improves_px = float(close[i]) <= float(pos["entry_price"]) - float(replace_min_price_improve)
            else:
                improves_px = float(close[i]) >= float(pos["entry_price"]) + float(replace_min_price_improve)
            if hold_bars >= int(replace_min_hold_bars) and (improves_q or improves_px):
                pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
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
                action_counts["replace"] = int(action_counts.get("replace", 0)) + 1
                pos = None
            else:
                continue

        if pos is not None:
            continue

        ep = float(close[i])
        if int(ev["direction"]) > 0:
            stop_px = ep - float(ev["stop_pts"])
            target_px = ep + float(ev["target_pts"])
        else:
            stop_px = ep + float(ev["stop_pts"])
            target_px = ep - float(ev["target_pts"])
        pos = {
            "entry_idx": int(i),
            "next_check_idx": int(i + 1),
            "entry_price": float(ep),
            "direction": int(ev["direction"]),
            "stop_px": float(stop_px),
            "target_px": float(target_px),
            "lane": str(ev["lane"]),
            "entry_quality": float(ev["quality"]),
        }

    if pos is not None:
        for j in range(int(pos["next_check_idx"]), end_idx + 1):
            if j <= int(pos["entry_idx"]):
                continue
            hit = None
            if int(pos["direction"]) > 0:
                if low[j] <= float(pos["stop_px"]) and high[j] >= float(pos["target_px"]):
                    hit = "stop_tie"
                elif low[j] <= float(pos["stop_px"]):
                    hit = "stop"
                elif high[j] >= float(pos["target_px"]):
                    hit = "target"
            else:
                if high[j] >= float(pos["stop_px"]) and low[j] <= float(pos["target_px"]):
                    hit = "stop_tie"
                elif high[j] >= float(pos["stop_px"]):
                    hit = "stop"
                elif low[j] <= float(pos["target_px"]):
                    hit = "target"
            if hit is None:
                continue
            exit_px = float(pos["stop_px"]) if str(hit).startswith("stop") else float(pos["target_px"])
            pnl = float((exit_px - float(pos["entry_price"])) * float(pos["direction"]))
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(pos["entry_idx"]),
                    "exit_idx": int(j),
                    "pnl": float(pnl),
                    "exit_reason": str(hit),
                    "lane": str(pos["lane"]),
                }
            )
            pos = None
            break
        if pos is not None:
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
    return pd.DataFrame(trades), action_counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", required=True, help="Comma-separated SQL trading days.")
    ap.add_argument("--configs", required=True, help="Comma-separated realtime strategy config paths.")
    ap.add_argument("--output-json", default="sandbox/results/runtime_sql_execution_gate_v1.json")
    args = ap.parse_args()

    days = _parse_csv(args.days)
    cfg_paths = _parse_csv(args.configs)
    out: Dict[str, Any] = {"days": days, "configs": cfg_paths, "runs": {}}

    for cfg_path in cfg_paths:
        raw = _load_json(cfg_path)
        params = dict(raw.get("params", raw))
        history_csv = str(params.get("historical_csv_path", "raw_data/schwab/es_minute_history.csv"))
        warmup_days = int(params.get("warmup_days", 60))
        all_trades: List[pd.DataFrame] = []
        action_totals: Dict[str, int] = {}
        day_ok: List[str] = []
        day_errors: Dict[str, str] = {}

        for day in days:
            try:
                strategy = _build_strategy(params)
                hist = _load_causal_history(history_csv, day, warmup_days)
                strategy.set_historical_context(hist)
                bars = _load_day_bars(day)
                sigs = strategy.process(bars)
                tdf, actions = _simulate_day_from_signals(day=day, bars=bars, signals=sigs, cfg=params)
                all_trades.append(tdf)
                for k, v in actions.items():
                    action_totals[k] = int(action_totals.get(k, 0)) + int(v)
                day_ok.append(day)
            except Exception as e:
                day_errors[day] = str(e)

        trades = pd.concat(all_trades, axis=0, ignore_index=True) if all_trades else pd.DataFrame()
        out["runs"][Path(cfg_path).stem] = {
            "config_path": cfg_path,
            "days_completed": day_ok,
            "days_failed": day_errors,
            "action_counts": action_totals,
            "exec": _metrics(trades, day_ok),
        }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(str(out_path))


if __name__ == "__main__":
    main()
