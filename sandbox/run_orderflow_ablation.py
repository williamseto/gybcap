#!/usr/bin/env python
"""Run order-flow ablations with paired day-bootstrap confidence intervals.

This version is expectancy-first:
- custom execution parameters (stop/target/max-bars/cost)
- paired day-bootstrap CIs for incremental WR and net expectancy
- trade concentration gates
- level/time-of-day breakdowns for enhanced model

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/run_orderflow_ablation.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_level_models_orderflow as ofexp


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _tod_bucket(ts: pd.Timestamp) -> str:
    minute_of_day = ts.hour * 60 + ts.minute
    if minute_of_day < (8 * 60):
        return "open_0630_0759"
    if minute_of_day < (10 * 60):
        return "mid_0800_0959"
    if minute_of_day < (11 * 60 + 30):
        return "late_1000_1129"
    return "close_1130_1315"


def _simulate_trade_records(
    ohlcv: pd.DataFrame,
    predicted_indices: np.ndarray,
    stop_pts: float = ofexp.STOP_PTS,
    target_pts: float = ofexp.TARGET_PTS,
    max_bars: int = ofexp.MAX_BARS,
    cost_per_trade: float = 0.0,
) -> pd.DataFrame:
    """Simulate trade outcomes and retain per-trade metadata."""
    if len(predicted_indices) == 0:
        return pd.DataFrame(
            columns=[
                "bar_index",
                "trading_day",
                "entry_dt",
                "nearest_level_name",
                "tod_bucket",
                "pnl_gross",
                "pnl_net",
                "win",
                "outcome",
            ]
        )

    close = ohlcv["close"].values.astype(np.float64)
    high = ohlcv["high"].values.astype(np.float64)
    low = ohlcv["low"].values.astype(np.float64)
    trade_dir = ohlcv["trade_direction"].values
    trading_day = ohlcv["trading_day"].astype(str).values
    dt_vals = pd.to_datetime(ohlcv["dt"]) if "dt" in ohlcv.columns else None
    level_vals = (
        ohlcv["nearest_level_name"].astype(str).values
        if "nearest_level_name" in ohlcv.columns
        else np.array(["unknown"] * len(ohlcv))
    )
    n = len(close)

    rows = []
    for idx in predicted_indices:
        i = int(idx)
        if i >= n:
            continue

        direction = int(trade_dir[i])
        if direction == 0:
            continue

        entry_price = close[i]
        trade_pnl = 0.0
        outcome = "timeout"

        for j in range(i + 1, min(i + max_bars + 1, n)):
            if direction == 1:
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    outcome = "stop"
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    outcome = "target"
                    break
            else:
                if high[j] >= entry_price + stop_pts:
                    trade_pnl = -stop_pts
                    outcome = "stop"
                    break
                if low[j] <= entry_price - target_pts:
                    trade_pnl = target_pts
                    outcome = "target"
                    break

        net_pnl = float(trade_pnl - cost_per_trade)
        entry_dt = dt_vals.iloc[i] if dt_vals is not None else pd.NaT

        rows.append(
            {
                "bar_index": i,
                "trading_day": trading_day[i],
                "entry_dt": entry_dt,
                "nearest_level_name": level_vals[i],
                "tod_bucket": _tod_bucket(entry_dt) if pd.notna(entry_dt) else "unknown",
                "pnl_gross": float(trade_pnl),
                "pnl_net": net_pnl,
                "win": int(net_pnl > 0),
                "outcome": outcome,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "bar_index",
                "trading_day",
                "entry_dt",
                "nearest_level_name",
                "tod_bucket",
                "pnl_gross",
                "pnl_net",
                "win",
                "outcome",
            ]
        )

    return pd.DataFrame(rows)


def _trade_metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "mean_pnl_gross": 0.0,
            "mean_pnl_net": 0.0,
            "total_pnl_net": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "max_day_share": 0.0,
            "n_days_traded": 0,
        }

    net = trades["pnl_net"].values
    wins = net[net > 0]
    losses = net[net < 0]

    day_counts = trades.groupby("trading_day").size()
    max_day_share = float(day_counts.max() / max(len(trades), 1))

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    if len(losses) > 0 and abs(avg_loss) > 1e-12:
        payoff_ratio = avg_win / abs(avg_loss)
    else:
        payoff_ratio = float("inf") if len(wins) > 0 else 0.0

    sum_wins = float(wins.sum()) if len(wins) > 0 else 0.0
    sum_losses = float(losses.sum()) if len(losses) > 0 else 0.0
    if abs(sum_losses) > 1e-12:
        profit_factor = sum_wins / abs(sum_losses)
    else:
        profit_factor = float("inf") if sum_wins > 0 else 0.0

    return {
        "n_trades": int(len(trades)),
        "win_rate": float((net > 0).mean()),
        "mean_pnl_gross": float(trades["pnl_gross"].mean()),
        "mean_pnl_net": float(net.mean()),
        "total_pnl_net": float(net.sum()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "max_day_share": max_day_share,
        "n_days_traded": int(trades["trading_day"].nunique()),
    }


def _group_metrics(trades: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
    if trades.empty or group_col not in trades.columns:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for key, grp in trades.groupby(group_col):
        gm = _trade_metrics(grp)
        out[str(key)] = {
            "n_trades": gm["n_trades"],
            "win_rate": gm["win_rate"],
            "mean_pnl_net": gm["mean_pnl_net"],
            "payoff_ratio": gm["payoff_ratio"],
        }
    return out


def _daily_trade_stats(trades: pd.DataFrame) -> Dict[str, Tuple[int, int, float]]:
    """Map day -> (n_trades, wins, total_net_pnl)."""
    if trades.empty:
        return {}

    grouped = (
        trades.groupby("trading_day")
        .agg(n=("pnl_net", "count"), wins=("win", "sum"), pnl=("pnl_net", "sum"))
        .reset_index()
    )
    return {
        str(row.trading_day): (int(row.n), int(row.wins), float(row.pnl))
        for row in grouped.itertuples(index=False)
    }


def _paired_day_bootstrap(
    baseline_stats: Dict[str, Tuple[int, int, float]],
    enhanced_stats: Dict[str, Tuple[int, int, float]],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Paired block bootstrap by trading day for WR and mean net-PnL deltas."""
    days = sorted(set(baseline_stats.keys()) | set(enhanced_stats.keys()))
    if not days:
        return {
            "delta_wr_mean": 0.0,
            "delta_wr_ci_low": 0.0,
            "delta_wr_ci_high": 0.0,
            "delta_pnl_mean": 0.0,
            "delta_pnl_ci_low": 0.0,
            "delta_pnl_ci_high": 0.0,
        }

    rng = np.random.default_rng(seed)
    wr_deltas = []
    pnl_deltas = []

    for _ in range(n_bootstrap):
        sampled_days = rng.choice(days, size=len(days), replace=True)

        b_n = b_w = 0
        b_pnl = 0.0
        e_n = e_w = 0
        e_pnl = 0.0

        for day in sampled_days:
            if day in baseline_stats:
                n, w, p = baseline_stats[day]
                b_n += n
                b_w += w
                b_pnl += p
            if day in enhanced_stats:
                n, w, p = enhanced_stats[day]
                e_n += n
                e_w += w
                e_pnl += p

        if b_n == 0 or e_n == 0:
            continue

        b_wr = b_w / b_n
        e_wr = e_w / e_n
        b_mp = b_pnl / b_n
        e_mp = e_pnl / e_n
        wr_deltas.append(e_wr - b_wr)
        pnl_deltas.append(e_mp - b_mp)

    if not wr_deltas:
        return {
            "delta_wr_mean": 0.0,
            "delta_wr_ci_low": 0.0,
            "delta_wr_ci_high": 0.0,
            "delta_pnl_mean": 0.0,
            "delta_pnl_ci_low": 0.0,
            "delta_pnl_ci_high": 0.0,
        }

    wr_arr = np.array(wr_deltas)
    pnl_arr = np.array(pnl_deltas)
    return {
        "delta_wr_mean": float(wr_arr.mean()),
        "delta_wr_ci_low": float(np.percentile(wr_arr, 2.5)),
        "delta_wr_ci_high": float(np.percentile(wr_arr, 97.5)),
        "delta_pnl_mean": float(pnl_arr.mean()),
        "delta_pnl_ci_low": float(np.percentile(pnl_arr, 2.5)),
        "delta_pnl_ci_high": float(np.percentile(pnl_arr, 97.5)),
    }


def _paired_auc_bootstrap(
    baseline_result: Dict,
    enhanced_result: Dict,
    day_by_index: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Paired day bootstrap for delta AUC."""
    b_prob = pd.Series(
        baseline_result["all_y_prob"],
        index=baseline_result["all_test_indices"],
    )
    e_prob = pd.Series(
        enhanced_result["all_y_prob"],
        index=enhanced_result["all_test_indices"],
    )
    y_true = pd.Series(
        baseline_result["all_y_true"],
        index=baseline_result["all_test_indices"],
    )

    common_idx = b_prob.index.intersection(e_prob.index).intersection(y_true.index)
    if len(common_idx) == 0:
        return {
            "delta_auc": 0.0,
            "delta_auc_ci_low": 0.0,
            "delta_auc_ci_high": 0.0,
        }

    y_arr = y_true.loc[common_idx].astype(int).values
    b_arr = b_prob.loc[common_idx].values
    e_arr = e_prob.loc[common_idx].values
    day_arr = day_by_index.loc[common_idx].astype(str).values

    if len(np.unique(y_arr)) < 2:
        return {
            "delta_auc": 0.0,
            "delta_auc_ci_low": 0.0,
            "delta_auc_ci_high": 0.0,
        }

    b_auc = roc_auc_score(y_arr, b_arr)
    e_auc = roc_auc_score(y_arr, e_arr)

    day_to_positions = {}
    for pos, day in enumerate(day_arr):
        day_to_positions.setdefault(day, []).append(pos)
    days = sorted(day_to_positions.keys())
    day_pos_arrays = [np.asarray(day_to_positions[d], dtype=np.int64) for d in days]

    rng = np.random.default_rng(seed)
    deltas = []

    for _ in range(n_bootstrap):
        sampled_day_idx = rng.integers(0, len(days), size=len(days))
        boot_positions = np.concatenate([day_pos_arrays[i] for i in sampled_day_idx])
        y_boot = y_arr[boot_positions]
        if len(np.unique(y_boot)) < 2:
            continue
        b_auc_boot = roc_auc_score(y_boot, b_arr[boot_positions])
        e_auc_boot = roc_auc_score(y_boot, e_arr[boot_positions])
        deltas.append(e_auc_boot - b_auc_boot)

    if not deltas:
        return {
            "delta_auc": float(e_auc - b_auc),
            "delta_auc_ci_low": 0.0,
            "delta_auc_ci_high": 0.0,
        }

    arr = np.array(deltas)
    return {
        "delta_auc": float(e_auc - b_auc),
        "delta_auc_ci_low": float(np.percentile(arr, 2.5)),
        "delta_auc_ci_high": float(np.percentile(arr, 97.5)),
    }


def _gate_result(
    enhanced_metrics: Dict[str, float],
    trade_ci: Dict[str, float],
    auc_delta: float,
    min_trades: int,
    max_day_share: float,
    min_payoff_ratio: float,
    min_expectancy: float,
    auc_degradation_guard: float,
) -> Dict[str, float | bool]:
    pass_min_trades = enhanced_metrics["n_trades"] >= min_trades
    pass_day_concentration = enhanced_metrics["max_day_share"] <= max_day_share
    pass_payoff = enhanced_metrics["payoff_ratio"] >= min_payoff_ratio
    pass_expectancy = enhanced_metrics["mean_pnl_net"] >= min_expectancy
    pass_incremental_ci = trade_ci["delta_pnl_ci_low"] > 0.0
    pass_auc_guard = auc_delta >= auc_degradation_guard

    pass_all = (
        pass_min_trades
        and pass_day_concentration
        and pass_payoff
        and pass_expectancy
        and pass_incremental_ci
        and pass_auc_guard
    )

    return {
        "pass_all": pass_all,
        "pass_min_trades": pass_min_trades,
        "pass_day_concentration": pass_day_concentration,
        "pass_payoff_ratio": pass_payoff,
        "pass_expectancy": pass_expectancy,
        "pass_incremental_ci": pass_incremental_ci,
        "pass_auc_guard": pass_auc_guard,
        "min_trades_required": int(min_trades),
        "max_day_share_allowed": float(max_day_share),
        "min_payoff_ratio_required": float(min_payoff_ratio),
        "min_expectancy_required": float(min_expectancy),
        "auc_degradation_guard": float(auc_degradation_guard),
    }


def run_slice(
    name: str,
    samples_df: pd.DataFrame,
    baseline_cols: list[str],
    enhanced_cols: list[str],
    ohlcv: pd.DataFrame,
    threshold: float,
    n_folds: int,
    min_train_days: int,
    n_bootstrap: int,
    stop_pts: float,
    target_pts: float,
    max_bars: int,
    cost_per_trade: float,
    min_trades_gate: int,
    max_day_share_gate: float,
    min_payoff_ratio: float,
    min_expectancy: float,
    auc_degradation_guard: float,
    seed: int,
) -> Dict:
    print("\n" + "=" * 90)
    print(f"ABLATION SLICE: {name}")
    print("=" * 90)
    print(
        f"Samples={len(samples_df):,}, days={samples_df['trading_day'].nunique()}, "
        f"features(base={len(baseline_cols)}, enh={len(enhanced_cols)})"
    )

    baseline_result = ofexp.train_walk_forward(
        samples_df=samples_df,
        feature_cols=baseline_cols,
        ohlcv=ohlcv,
        n_folds=n_folds,
        min_train_days=min_train_days,
        label=f"{name} | baseline",
    )
    enhanced_result = ofexp.train_walk_forward(
        samples_df=samples_df,
        feature_cols=enhanced_cols,
        ohlcv=ohlcv,
        n_folds=n_folds,
        min_train_days=min_train_days,
        label=f"{name} | +orderflow",
    )

    b_pred_indices = baseline_result["all_test_indices"][
        baseline_result["all_y_prob"] >= threshold
    ]
    e_pred_indices = enhanced_result["all_test_indices"][
        enhanced_result["all_y_prob"] >= threshold
    ]

    b_trades = _simulate_trade_records(
        ohlcv=ohlcv,
        predicted_indices=b_pred_indices,
        stop_pts=stop_pts,
        target_pts=target_pts,
        max_bars=max_bars,
        cost_per_trade=cost_per_trade,
    )
    e_trades = _simulate_trade_records(
        ohlcv=ohlcv,
        predicted_indices=e_pred_indices,
        stop_pts=stop_pts,
        target_pts=target_pts,
        max_bars=max_bars,
        cost_per_trade=cost_per_trade,
    )

    b_metrics = _trade_metrics(b_trades)
    e_metrics = _trade_metrics(e_trades)

    boot_trade = _paired_day_bootstrap(
        _daily_trade_stats(b_trades),
        _daily_trade_stats(e_trades),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    day_by_index = ohlcv["trading_day"].astype(str)
    boot_auc = _paired_auc_bootstrap(
        baseline_result,
        enhanced_result,
        day_by_index=day_by_index,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    gate = _gate_result(
        enhanced_metrics=e_metrics,
        trade_ci=boot_trade,
        auc_delta=float(enhanced_result["overall_auc"] - baseline_result["overall_auc"]),
        min_trades=min_trades_gate,
        max_day_share=max_day_share_gate,
        min_payoff_ratio=min_payoff_ratio,
        min_expectancy=min_expectancy,
        auc_degradation_guard=auc_degradation_guard,
    )

    result = {
        "slice": name,
        "n_samples": int(len(samples_df)),
        "n_days": int(samples_df["trading_day"].nunique()),
        "threshold": float(threshold),
        "execution": {
            "stop_pts": float(stop_pts),
            "target_pts": float(target_pts),
            "max_bars": int(max_bars),
            "cost_per_trade": float(cost_per_trade),
        },
        "baseline_auc": float(baseline_result["overall_auc"]),
        "enhanced_auc": float(enhanced_result["overall_auc"]),
        "delta_auc": float(enhanced_result["overall_auc"] - baseline_result["overall_auc"]),
        "delta_auc_ci_low": boot_auc["delta_auc_ci_low"],
        "delta_auc_ci_high": boot_auc["delta_auc_ci_high"],
        "baseline": b_metrics,
        "enhanced": e_metrics,
        "delta_wr": float(e_metrics["win_rate"] - b_metrics["win_rate"]),
        "delta_wr_ci_low": boot_trade["delta_wr_ci_low"],
        "delta_wr_ci_high": boot_trade["delta_wr_ci_high"],
        "delta_mean_pnl_net": float(e_metrics["mean_pnl_net"] - b_metrics["mean_pnl_net"]),
        "delta_mean_pnl_net_ci_low": boot_trade["delta_pnl_ci_low"],
        "delta_mean_pnl_net_ci_high": boot_trade["delta_pnl_ci_high"],
        "gate": gate,
        "enhanced_breakdown": {
            "by_level": _group_metrics(e_trades, "nearest_level_name"),
            "by_tod": _group_metrics(e_trades, "tod_bucket"),
        },
    }

    print("\nSlice Summary:")
    print(
        f"  AUC: {result['baseline_auc']:.4f} -> {result['enhanced_auc']:.4f} "
        f"(delta={result['delta_auc']:+.4f}, "
        f"95% CI [{result['delta_auc_ci_low']:+.4f}, {result['delta_auc_ci_high']:+.4f}])"
    )
    print(
        f"  WR(net): {b_metrics['win_rate']:.2%} -> {e_metrics['win_rate']:.2%} "
        f"(95% CI delta [{result['delta_wr_ci_low']:+.2%}, {result['delta_wr_ci_high']:+.2%}])"
    )
    print(
        f"  E[PnL_net]: {b_metrics['mean_pnl_net']:+.3f} -> {e_metrics['mean_pnl_net']:+.3f} "
        f"(95% CI delta [{result['delta_mean_pnl_net_ci_low']:+.3f}, {result['delta_mean_pnl_net_ci_high']:+.3f}])"
    )
    print(
        f"  Payoff ratio: {b_metrics['payoff_ratio']:.3f} -> {e_metrics['payoff_ratio']:.3f}; "
        f"Max day share(enh): {e_metrics['max_day_share']:.2%}; Gate pass={gate['pass_all']}"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Order-flow ablation matrix with expectancy-first gating"
    )
    parser.add_argument("--data", default=ofexp.DATA_PATH)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-pts", type=float, default=ofexp.STOP_PTS)
    parser.add_argument("--target-pts", type=float, default=ofexp.TARGET_PTS)
    parser.add_argument("--max-bars", type=int, default=ofexp.MAX_BARS)
    parser.add_argument("--cost-per-trade", type=float, default=0.0)

    # Edge gates
    parser.add_argument("--min-trades-gate", type=int, default=1000)
    parser.add_argument("--max-day-share-gate", type=float, default=0.20)
    parser.add_argument("--min-payoff-ratio", type=float, default=1.20)
    parser.add_argument("--min-enhanced-expectancy", type=float, default=0.0)
    parser.add_argument(
        "--auc-degradation-guard",
        type=float,
        default=-0.005,
        help="Enhanced AUC must be >= baseline AUC + this value.",
    )

    parser.add_argument(
        "--output-json",
        default="sandbox/results/orderflow_ablation_results.json",
    )
    args = parser.parse_args()

    t0 = time.time()
    set_seed(args.seed)

    print("=" * 90)
    print("ORDER-FLOW ABLATION MATRIX (EXPECTANCY-FIRST)")
    print("=" * 90)

    print("\nPreparing base dataset...")
    ohlcv = ofexp.load_data(args.data)
    ohlcv = ofexp.compute_levels(ohlcv)
    ohlcv, feature_cols = ofexp.compute_all_features(ohlcv)
    ohlcv = ofexp.label_reversals_breakouts(ohlcv)
    ohlcv, level_encoding_cols = ofexp.compute_level_encoding_features(ohlcv)
    baseline_cols = list(dict.fromkeys(feature_cols + level_encoding_cols))

    ohlcv_no_miss, of_cols = ofexp.merge_orderflow_features(
        ohlcv.copy(), add_missing_indicators=False
    )
    ohlcv_with_miss, of_cols_with_miss = ofexp.merge_orderflow_features(
        ohlcv.copy(), add_missing_indicators=True
    )

    sample_mask_no_miss = ohlcv_no_miss["outcome"].isin([0, 1])
    samples_no_miss = ohlcv_no_miss.loc[sample_mask_no_miss].copy()
    sample_mask_with_miss = ohlcv_with_miss["outcome"].isin([0, 1])
    samples_with_miss = ohlcv_with_miss.loc[sample_mask_with_miss].copy()

    of_cov_mask = samples_no_miss[of_cols[0]].notna()
    samples_cov = samples_no_miss.loc[of_cov_mask].copy()

    if "ovn" in samples_with_miss.columns:
        rth_mask = samples_with_miss["ovn"] == 0
    else:
        rth_mask = np.ones(len(samples_with_miss), dtype=bool)
    samples_rth = samples_with_miss.loc[rth_mask].copy()

    enhanced_raw_cols = list(dict.fromkeys(baseline_cols + of_cols))
    enhanced_miss_cols = list(dict.fromkeys(baseline_cols + of_cols_with_miss))

    run_kwargs = {
        "threshold": args.threshold,
        "n_folds": args.n_folds,
        "min_train_days": args.min_train_days,
        "n_bootstrap": args.bootstrap_iters,
        "stop_pts": args.stop_pts,
        "target_pts": args.target_pts,
        "max_bars": args.max_bars,
        "cost_per_trade": args.cost_per_trade,
        "min_trades_gate": args.min_trades_gate,
        "max_day_share_gate": args.max_day_share_gate,
        "min_payoff_ratio": args.min_payoff_ratio,
        "min_expectancy": args.min_enhanced_expectancy,
        "auc_degradation_guard": args.auc_degradation_guard,
    }

    results = []
    results.append(
        run_slice(
            name="all_samples_raw_of",
            samples_df=samples_no_miss,
            baseline_cols=baseline_cols,
            enhanced_cols=enhanced_raw_cols,
            ohlcv=ohlcv_no_miss,
            seed=args.seed,
            **run_kwargs,
        )
    )
    results.append(
        run_slice(
            name="all_samples_with_missing_indicators",
            samples_df=samples_with_miss,
            baseline_cols=baseline_cols,
            enhanced_cols=enhanced_miss_cols,
            ohlcv=ohlcv_with_miss,
            seed=args.seed + 1,
            **run_kwargs,
        )
    )
    results.append(
        run_slice(
            name="of_covered_only",
            samples_df=samples_cov,
            baseline_cols=baseline_cols,
            enhanced_cols=enhanced_raw_cols,
            ohlcv=ohlcv_no_miss,
            seed=args.seed + 2,
            **run_kwargs,
        )
    )
    results.append(
        run_slice(
            name="rth_only_with_missing_indicators",
            samples_df=samples_rth,
            baseline_cols=baseline_cols,
            enhanced_cols=enhanced_miss_cols,
            ohlcv=ohlcv_with_miss,
            seed=args.seed + 3,
            **run_kwargs,
        )
    )

    positive_slices = [r["slice"] for r in results if r["gate"]["pass_all"]]
    overall_verdict = "EDGE_FOUND" if positive_slices else "NO_EDGE_CONFIRMED"

    payload = {
        "config": vars(args),
        "results": results,
        "overall_verdict": overall_verdict,
        "positive_slices": positive_slices,
        "elapsed_sec": float(time.time() - t0),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 90)
    print("OVERALL VERDICT")
    print("=" * 90)
    print(f"Verdict: {overall_verdict}")
    if positive_slices:
        print(f"Slices with positive incremental edge: {positive_slices}")
    print(f"Result JSON: {args.output_json}")
    print(f"Elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
