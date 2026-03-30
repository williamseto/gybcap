#!/usr/bin/env python
"""Run a causal SQL playback snapshot for one day and emit reversal diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.train_level_models import label_reversals_major_move
from strategies.realtime.bar_aggregator import BarAggregator
from strategies.realtime.config import DatabaseConfig
from strategies.realtime.csv_data_source import CSVDataSource
from strategies.realtime.data_source import MySQLSource, get_session_window_for_trading_day
from strategies.reversal.realtime_strategy import ReversalPredictorStrategy


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def _build_strategy(params: Dict[str, Any]) -> ReversalPredictorStrategy:
    return ReversalPredictorStrategy(
        model_dir=str(params["model_dir"]),
        pred_threshold=float(params.get("pred_threshold", 0.5)),
        pred_threshold_rth=(
            float(params["pred_threshold_rth"])
            if params.get("pred_threshold_rth") is not None
            else None
        ),
        pred_threshold_ovn=(
            float(params["pred_threshold_ovn"])
            if params.get("pred_threshold_ovn") is not None
            else None
        ),
        proximity_pts=float(params.get("proximity_pts", 5.0)),
        side_hysteresis_pts=float(params.get("side_hysteresis_pts", 0.0)),
        side_flip_confirm_pts=(
            float(params["side_flip_confirm_pts"])
            if params.get("side_flip_confirm_pts") is not None
            else None
        ),
        same_level_min_confirm_bars=int(params.get("same_level_min_confirm_bars", 1)),
        level_signal_cooldown_bars=int(params.get("level_signal_cooldown_bars", 0)),
        use_episode_gating=bool(params.get("use_episode_gating", False)),
        episode_gap_bars=int(params.get("episode_gap_bars", 1)),
        trade_budget_per_day=int(params.get("trade_budget_per_day", 0)),
        trade_budget_rth_per_day=int(params.get("trade_budget_rth_per_day", 0)),
        trade_budget_ovn_per_day=int(params.get("trade_budget_ovn_per_day", 0)),
        budget_adapt_step=float(params.get("budget_adapt_step", 0.01)),
        budget_adapt_tolerance=float(params.get("budget_adapt_tolerance", 0.5)),
        budget_expected_init=float(params.get("budget_expected_init", 20.0)),
        budget_ewma_alpha=float(params.get("budget_ewma_alpha", 0.20)),
        budget_max_threshold=float(params.get("budget_max_threshold", 0.95)),
        policy_model_dir=(
            str(params["policy_model_dir"])
            if params.get("policy_model_dir")
            else None
        ),
        policy_threshold=(
            float(params["policy_threshold"])
            if params.get("policy_threshold") is not None
            else None
        ),
        policy_threshold_rth=(
            float(params["policy_threshold_rth"])
            if params.get("policy_threshold_rth") is not None
            else None
        ),
        policy_threshold_ovn=(
            float(params["policy_threshold_ovn"])
            if params.get("policy_threshold_ovn") is not None
            else None
        ),
        frontier_router_enabled=bool(params.get("frontier_router_enabled", False)),
        frontier_quality_source=str(params.get("frontier_quality_source", "base_prob")),
        frontier_quality_model_dir=(
            str(params["frontier_quality_model_dir"])
            if params.get("frontier_quality_model_dir")
            else None
        ),
        frontier_quality_lookup_path=(
            str(params["frontier_quality_lookup_path"])
            if params.get("frontier_quality_lookup_path")
            else None
        ),
        frontier_blend_live_cap=int(params.get("frontier_blend_live_cap", 5)),
        frontier_blend_live_start=int(params.get("frontier_blend_live_start", 390)),
        frontier_blend_live_end=int(params.get("frontier_blend_live_end", 780)),
        frontier_blend_live_min_q=float(params.get("frontier_blend_live_min_q", 0.0)),
        frontier_blend_live_flatten_min_q=float(
            params.get("frontier_blend_live_flatten_min_q", 0.66)
        ),
        frontier_blend_q2_cap=int(params.get("frontier_blend_q2_cap", 2)),
        frontier_blend_q2_start=int(params.get("frontier_blend_q2_start", 450)),
        frontier_blend_q2_end=int(params.get("frontier_blend_q2_end", 780)),
        frontier_blend_q2_min_q=float(params.get("frontier_blend_q2_min_q", 0.16)),
        frontier_blend_q2_flatten_min_q=float(
            params.get("frontier_blend_q2_flatten_min_q", 0.16)
        ),
        frontier_blend_q2_override_prob=(
            float(params["frontier_blend_q2_override_prob"])
            if params.get("frontier_blend_q2_override_prob") is not None
            else None
        ),
        frontier_execution_mode=str(params.get("frontier_execution_mode", "independent")),
        frontier_opposite_action=str(params.get("frontier_opposite_action", "ignore")),
        frontier_opposite_flip_min_q=float(params.get("frontier_opposite_flip_min_q", 0.72)),
        frontier_opposite_flatten_min_q=float(params.get("frontier_opposite_flatten_min_q", 0.66)),
        frontier_opposite_min_q_gap_vs_open=float(
            params.get("frontier_opposite_min_q_gap_vs_open", 0.0)
        ),
        frontier_opposite_flip_requires_open_loss=bool(
            params.get("frontier_opposite_flip_requires_open_loss", False)
        ),
        frontier_consume_budget_on_flatten=bool(params.get("frontier_consume_budget_on_flatten", False)),
        frontier_emit_flatten_signal=bool(params.get("frontier_emit_flatten_signal", False)),
        frontier_k_total=int(params.get("frontier_k_total", 7)),
        frontier_q_thr=float(params.get("frontier_q_thr", 0.58)),
        frontier_q_slots=int(params.get("frontier_q_slots", 3)),
        frontier_c_thr=float(params.get("frontier_c_thr", 0.48)),
        frontier_c_start=int(params.get("frontier_c_start", 450)),
        frontier_c_slots=int(params.get("frontier_c_slots", 2)),
        frontier_r_thr=float(params.get("frontier_r_thr", 0.45)),
        frontier_r_start=int(params.get("frontier_r_start", 600)),
        frontier_r_slots=int(params.get("frontier_r_slots", 1)),
        frontier_cooldown_min=int(params.get("frontier_cooldown_min", 5)),
        frontier_diversity_cap=int(params.get("frontier_diversity_cap", 2)),
        frontier_high_override_prob=(
            float(params["frontier_high_override_prob"])
            if params.get("frontier_high_override_prob") is not None
            else None
        ),
        frontier_high_override_cap=int(params.get("frontier_high_override_cap", 0)),
        frontier_high_override_start=int(params.get("frontier_high_override_start", 390)),
        frontier_high_override_window_min=int(params.get("frontier_high_override_window_min", 0)),
        frontier_high_override_window_cap=int(params.get("frontier_high_override_window_cap", 0)),
        frontier_group_unresolved_enabled=bool(
            params.get("frontier_group_unresolved_enabled", False)
        ),
        frontier_dynamic_budget_enabled=bool(params.get("frontier_dynamic_budget_enabled", False)),
        frontier_dynamic_extra_cap=int(params.get("frontier_dynamic_extra_cap", 0)),
        frontier_dynamic_extra_every=int(params.get("frontier_dynamic_extra_every", 1)),
        frontier_dynamic_start_minute=int(params.get("frontier_dynamic_start_minute", 390)),
        frontier_loss_lock_diff=int(params.get("frontier_loss_lock_diff", 0)),
        frontier_loss_lock_after_nominal_only=bool(
            params.get("frontier_loss_lock_after_nominal_only", True)
        ),
        frontier_virtual_gate=float(params.get("frontier_virtual_gate", 0.62)),
        frontier_virtual_stop_hi=float(params.get("frontier_virtual_stop_hi", 10.0)),
        frontier_virtual_target_hi=float(params.get("frontier_virtual_target_hi", 40.0)),
        frontier_virtual_stop_lo=float(params.get("frontier_virtual_stop_lo", 4.0)),
        frontier_virtual_target_lo=float(params.get("frontier_virtual_target_lo", 8.0)),
        frontier_virtual_gate_calibration_enabled=bool(
            params.get("frontier_virtual_gate_calibration_enabled", True)
        ),
        frontier_virtual_gate_calibration_days=int(
            params.get("frontier_virtual_gate_calibration_days", 20)
        ),
        frontier_virtual_gate_calibration_min_samples=int(
            params.get("frontier_virtual_gate_calibration_min_samples", 120)
        ),
        frontier_virtual_gate_calibration_decay=float(
            params.get("frontier_virtual_gate_calibration_decay", 0.97)
        ),
        frontier_virtual_gate_calibration_grid_size=int(
            params.get("frontier_virtual_gate_calibration_grid_size", 41)
        ),
        frontier_virtual_gate_calibration_min=float(
            params.get("frontier_virtual_gate_calibration_min", 0.10)
        ),
        frontier_virtual_gate_calibration_max=float(
            params.get("frontier_virtual_gate_calibration_max", 0.80)
        ),
        policy_mode=(
            str(params["policy_mode"])
            if params.get("policy_mode") is not None
            else None
        ),
        policy_episode_gap_bars=(
            int(params["policy_episode_gap_bars"])
            if params.get("policy_episode_gap_bars") is not None
            else None
        ),
        policy_use_ensemble_mean=(
            bool(params["policy_use_ensemble_mean"])
            if params.get("policy_use_ensemble_mean") is not None
            else None
        ),
        policy_uncertainty_std_max=(
            float(params["policy_uncertainty_std_max"])
            if params.get("policy_uncertainty_std_max") is not None
            else None
        ),
        large_move_model_dir=(
            str(params["large_move_model_dir"])
            if params.get("large_move_model_dir")
            else None
        ),
        mfe_model_dir=(
            str(params["mfe_model_dir"])
            if params.get("mfe_model_dir")
            else None
        ),
        mfe_min_base_prob=float(params.get("mfe_min_base_prob", 0.0)),
        score_mode=str(params.get("score_mode", "base_prob")),
        event_large_blend_alpha=float(params.get("event_large_blend_alpha", 0.15)),
        use_base_prob_prefilter=bool(params.get("use_base_prob_prefilter", True)),
        split_primary_accepts=int(params.get("split_primary_accepts", 1)),
        dual_lane_enabled=bool(params.get("dual_lane_enabled", False)),
        high_score_threshold=(
            float(params["high_score_threshold"])
            if params.get("high_score_threshold") is not None
            else None
        ),
        high_score_threshold_rth=(
            float(params["high_score_threshold_rth"])
            if params.get("high_score_threshold_rth") is not None
            else None
        ),
        high_score_threshold_ovn=(
            float(params["high_score_threshold_ovn"])
            if params.get("high_score_threshold_ovn") is not None
            else None
        ),
        low_score_threshold=(
            float(params["low_score_threshold"])
            if params.get("low_score_threshold") is not None
            else None
        ),
        low_score_threshold_rth=(
            float(params["low_score_threshold_rth"])
            if params.get("low_score_threshold_rth") is not None
            else None
        ),
        low_score_threshold_ovn=(
            float(params["low_score_threshold_ovn"])
            if params.get("low_score_threshold_ovn") is not None
            else None
        ),
        fallback_start_minute=int(params.get("fallback_start_minute", 390)),
        max_low_per_day=int(params.get("max_low_per_day", 1)),
        allow_low_before_high=bool(params.get("allow_low_before_high", False)),
        low_lane_scope=str(params.get("low_lane_scope", "all")),
        strong_levels=list(params.get("strong_levels", [])),
        dual_max_trades_per_day=int(params.get("dual_max_trades_per_day", 0)),
        mfe_regime_gate_enabled=bool(params.get("mfe_regime_gate_enabled", False)),
        mfe_regime_drive_move_pts=float(params.get("mfe_regime_drive_move_pts", 8.0)),
        mfe_regime_drive_frac=float(params.get("mfe_regime_drive_frac", 0.55)),
        mfe_regime_countertrend_prob_floor=float(
            params.get("mfe_regime_countertrend_prob_floor", 0.75)
        ),
        mfe_regime_end_minute=int(params.get("mfe_regime_end_minute", 690)),
    )


def _load_causal_history(csv_path: str, trading_day: str, warmup_days: int) -> pd.DataFrame:
    csv_source = CSVDataSource(csv_path)
    start_ts, _ = get_session_window_for_trading_day(trading_day)
    return csv_source.fetch_history_bars(start_ts - 1, n_days=int(warmup_days))


def _load_day_bars(trading_day: str) -> pd.DataFrame:
    source = MySQLSource(DatabaseConfig())
    start_ts, end_ts = get_session_window_for_trading_day(trading_day)
    sec_df = source.fetch_range(start_ts, end_ts)
    if sec_df.empty:
        raise ValueError(f"No SQL second data found for trading_day={trading_day}")
    return BarAggregator.build_minute_bars(sec_df)


def _compute_realized_mfe(df: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    out = df.copy()
    close_arr = out["close"].to_numpy(dtype=np.float64, copy=False)
    high_arr = out["high"].to_numpy(dtype=np.float64, copy=False)
    low_arr = out["low"].to_numpy(dtype=np.float64, copy=False)
    side_arr = out.get("side", pd.Series(0, index=out.index)).to_numpy(dtype=np.int8, copy=False)
    mfe_pts = np.full(len(out), np.nan, dtype=np.float64)
    for i in range(len(out)):
        end = min(len(out), i + 1 + int(forward_bars))
        if end <= i + 1:
            continue
        if side_arr[i] > 0:
            mfe_pts[i] = float(np.nanmax(high_arr[i + 1:end] - close_arr[i]))
        elif side_arr[i] < 0:
            mfe_pts[i] = float(np.nanmax(close_arr[i] - low_arr[i + 1:end]))
    out["mfe_pts"] = mfe_pts
    out["mfe_r"] = out["mfe_pts"] / 4.0
    return out


def _run_diag(
    strategy: ReversalPredictorStrategy,
    bars_1m: pd.DataFrame,
    label_forward_bars: int,
    label_excursion_ratio: float,
    label_breakout_excursion_ratio: float,
    label_terminal_disp_frac: float,
    label_reversal_vol_mult: float,
    label_breakout_vol_mult: float,
) -> pd.DataFrame:
    strategy.reset_day()

    combined = strategy._build_combined_df(bars_1m)
    if combined is None or len(combined) < 100:
        raise ValueError("Insufficient combined data for diagnostics")

    featured = strategy._compute_features(combined)
    if featured is None:
        raise ValueError("Feature computation failed")

    current_day_start = len(combined) - len(bars_1m)
    current = featured.iloc[current_day_start:].copy()
    current["__bar_num"] = np.arange(len(current), dtype=np.int32)
    current = strategy._find_nearest_levels(current)
    current = strategy._compute_level_encoding(current, featured)
    current = strategy._append_opening_drive_regime(current)

    labeled = label_reversals_major_move(
        current.copy(),
        proximity_pts=float(strategy._proximity_pts),
        forward_bars=int(label_forward_bars),
        reversal_pts=6.0,
        breakout_pts=4.0,
        tracked_levels=list(strategy._tracked_levels),
        side_hysteresis_pts=float(strategy._side_hysteresis_pts),
        side_flip_confirm_pts=(
            float(strategy._side_flip_confirm_pts)
            if strategy._side_flip_confirm_pts is not None
            else None
        ),
        excursion_ratio=float(label_excursion_ratio),
        breakout_excursion_ratio=float(label_breakout_excursion_ratio),
        terminal_disp_frac=float(label_terminal_disp_frac),
        use_volatility_scaling=True,
        volatility_col="daily_atr_14",
        reversal_vol_mult=float(label_reversal_vol_mult),
        breakout_vol_mult=float(label_breakout_vol_mult),
    )
    current["outcome"] = labeled["outcome"].values
    current = _compute_realized_mfe(current, forward_bars=label_forward_bars)

    near_mask = ~current["nearest_level_price"].isna()
    near_df = current.loc[near_mask].copy()
    near_df = near_df.sort_values("__bar_num")
    if near_df.empty:
        return current.assign(base_prob=np.nan, score=np.nan, mfe_pred_r=np.nan, accepted=False, gate_reason="no_near_level")

    X_df = near_df.reindex(columns=strategy._feature_cols, fill_value=0.0)
    X = X_df.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    y_prob = strategy._model.predict_proba(X)[:, 1]
    large_prob_arr: Optional[np.ndarray] = None
    if strategy._large_move_model is not None and strategy._large_move_feature_cols:
        X_large_df = near_df.reindex(columns=strategy._large_move_feature_cols, fill_value=0.0)
        X_large = X_large_df.fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        large_prob_arr = strategy._large_move_model.predict_proba(X_large)[:, 1]

    rows = []
    for i, (idx, row) in enumerate(near_df.iterrows()):
        row = row.copy()
        base_prob = float(y_prob[i])
        large_prob = float(large_prob_arr[i]) if large_prob_arr is not None else None
        session_key = strategy._session_key(row)
        base_threshold = strategy._session_threshold(session_key)
        score_value = float(base_prob)
        mfe_pred_r = None
        mfe_score_allowed = True
        mfe_gate_reason = "na"
        policy_prob: Optional[float] = None
        policy_std: Optional[float] = None
        frontier_quality_score = float(base_prob)
        frontier_lane: Optional[str] = None
        lane = "high"
        gate_reason = "accepted"
        accepted = False
        level_name = str(row.get("nearest_level_name", ""))
        side = int(row.get("side", 0))
        direction = "bull" if side == 1 else "bear"
        bar_num = int(row.get("__bar_num", i))
        level_dir = (level_name, direction)

        if strategy._mfe_min_base_prob > 0.0 and base_prob < strategy._mfe_min_base_prob:
            gate_reason = "min_base_prob"
        elif strategy._use_base_prob_prefilter and (not strategy._dual_lane_enabled) and base_prob < base_threshold:
            gate_reason = "base_threshold"
        else:
            mfe_score_allowed, mfe_gate_reason = strategy._mfe_regime_score_allowed(
                row=row,
                base_prob=base_prob,
                session_key=session_key,
            )
            score_value, mfe_pred_r = strategy._compute_score(
                row=row,
                base_prob=base_prob,
                session_key=session_key,
                large_prob=large_prob,
            )

            if strategy._frontier_router_enabled:
                minute_of_day = strategy._minute_of_day(pd.Timestamp(row["dt"]))
                if (
                    strategy._frontier_dynamic_budget_enabled
                    or strategy._frontier_group_unresolved_enabled
                    or getattr(strategy, "_frontier_execution_mode", "independent") == "sequential"
                ):
                    strategy._frontier_update_virtual_outcomes(
                        bar_num=int(bar_num),
                        bar_high=float(row.get("high", np.nan)),
                        bar_low=float(row.get("low", np.nan)),
                    )
                if (
                    strategy._frontier_quality_source == "policy_prob"
                    and strategy._policy_model is not None
                ):
                    policy_prob, policy_std = strategy._policy_predict_score(
                        row=row,
                        base_prob=base_prob,
                        session_key=session_key,
                        bar_num=bar_num,
                    )
                    if policy_prob is not None:
                        frontier_quality_score = float(policy_prob)
                elif strategy._frontier_quality_source == "q_twohead":
                    q_score = strategy._frontier_predict_qtwohead(row=row)
                    if q_score is not None:
                        frontier_quality_score = float(q_score)
                elif strategy._frontier_quality_source == "q_lookup":
                    q_score = strategy._frontier_predict_qlookup(row=row)
                    if q_score is not None:
                        frontier_quality_score = float(q_score)
                score_value = float(frontier_quality_score)
                accepted_router, frontier_lane = strategy._frontier_router_accept(
                    row=row,
                    minute_of_day=int(minute_of_day),
                    direction=(1 if side == 1 else -1),
                    level_name=level_name,
                    quality_score=float(frontier_quality_score),
                    coverage_score=float(base_prob),
                    rescue_score=float(base_prob),
                    session_key=session_key,
                )
                if not accepted_router:
                    gate_reason = str(frontier_lane or "router_reject")
                else:
                    lane = str(frontier_lane or "quality")
            elif strategy._dual_lane_enabled:
                lane_high_thr = strategy._lane_high_threshold(session_key)
                lane_low_thr = strategy._lane_low_threshold(session_key)
                if score_value >= lane_high_thr:
                    lane = "high"
                else:
                    allow_low = (
                        lane_low_thr is not None
                        and score_value >= float(lane_low_thr)
                        and score_value < lane_high_thr
                        and strategy._max_low_per_day > 0
                        and strategy._dual_low_taken_today < strategy._max_low_per_day
                    )
                    if allow_low:
                        minute_of_day = strategy._minute_of_day(pd.Timestamp(row["dt"]))
                        if minute_of_day < strategy._fallback_start_minute:
                            allow_low = False
                            gate_reason = "fallback_time"
                        elif (not strategy._allow_low_before_high) and strategy._dual_high_taken_today > 0:
                            allow_low = False
                            gate_reason = "low_after_high_disallowed"
                        elif strategy._low_lane_scope == "strong_levels" and level_name not in strategy._strong_levels:
                            allow_low = False
                            gate_reason = "low_lane_scope"
                    if not allow_low:
                        if gate_reason == "accepted":
                            gate_reason = "lane_threshold"
                    else:
                        lane = "low"

            if gate_reason == "accepted":
                if side == 0:
                    gate_reason = "side_zero"
                else:
                    prev_candidate_bar = strategy._last_candidate_bar_by_level_dir.get(level_dir)
                    prev_streak = strategy._candidate_streak_by_level_dir.get(level_dir, 0)
                    if prev_candidate_bar is not None and bar_num == (prev_candidate_bar + 1):
                        candidate_streak = prev_streak + 1
                    else:
                        candidate_streak = 1
                    strategy._last_candidate_bar_by_level_dir[level_dir] = bar_num
                    strategy._candidate_streak_by_level_dir[level_dir] = candidate_streak
                    if candidate_streak < strategy._same_level_min_confirm_bars:
                        gate_reason = "confirm_bars"

            if gate_reason == "accepted" and strategy._use_episode_gating:
                prev_episode_bar = strategy._episode_last_bar_by_level.get(level_name)
                if prev_episode_bar is not None and (bar_num - prev_episode_bar) <= strategy._episode_gap_bars:
                    gate_reason = "episode_gating"

            if gate_reason == "accepted" and strategy._level_signal_cooldown_bars > 0:
                prev_emit_bar = strategy._last_emit_bar_by_level_dir.get(level_dir)
                if prev_emit_bar is not None and (bar_num - prev_emit_bar) < strategy._level_signal_cooldown_bars:
                    gate_reason = "cooldown"

            if gate_reason == "accepted" and strategy._dual_max_trades_per_day > 0:
                if strategy._dual_taken_today >= strategy._dual_max_trades_per_day:
                    gate_reason = "max_trades_per_day"

            if (
                gate_reason == "accepted"
                and (not strategy._frontier_router_enabled)
                and strategy._policy_model is not None
            ):
                policy_allowed, policy_prob, policy_gate_reason, policy_std = strategy._evaluate_policy_gate(
                    row=row,
                    base_prob=base_prob,
                    session_key=session_key,
                    bar_num=bar_num,
                )
                if not policy_allowed:
                    gate_reason = str(policy_gate_reason or "policy_gate")

            if gate_reason == "accepted" and (not strategy._frontier_router_enabled):
                score_for_budget = score_value if strategy._score_mode != "base_prob" else base_prob
                if policy_prob is not None:
                    score_for_budget = policy_prob
                if not strategy._accept_under_budget(
                    score=score_for_budget,
                    session_key=session_key,
                    base_threshold=base_threshold,
                ):
                    gate_reason = "budget_gate"

            if gate_reason == "accepted":
                bar_ts = str(row["dt"]) if "dt" in row.index and pd.notna(row["dt"]) else str(idx)
                dedup_key = (level_name, direction, bar_ts)
                if dedup_key in strategy._emitted:
                    gate_reason = "dedup"
                else:
                    accepted = True
                    strategy._emitted.add(dedup_key)
                    strategy._last_emit_bar_by_level_dir[level_dir] = bar_num
                    if strategy._use_episode_gating:
                        strategy._episode_last_bar_by_level[level_name] = bar_num
                    if (
                        strategy._frontier_router_enabled
                        and strategy._frontier_quality_source == "policy_prob"
                        and strategy._policy_model is not None
                    ):
                        strategy._policy_mark_episode_start(bar_num=bar_num, row=row)
                    if strategy._frontier_router_enabled and (
                        strategy._frontier_dynamic_budget_enabled
                        or strategy._frontier_group_unresolved_enabled
                        or getattr(strategy, "_frontier_execution_mode", "independent") == "sequential"
                    ):
                        strategy._frontier_register_virtual_trade(
                            bar_num=int(bar_num),
                            entry_price=float(row.get("close", np.nan)),
                            level_name=str(level_name),
                            direction=(1 if side == 1 else -1),
                            quality_score=float(frontier_quality_score),
                        )
                    if strategy._dual_lane_enabled:
                        strategy._dual_taken_today += 1
                        if lane == "high":
                            strategy._dual_high_taken_today += 1
                        else:
                            strategy._dual_low_taken_today += 1

        rows.append({
            "dt": row.get("dt"),
            "trading_day": row.get("trading_day"),
            "close": float(row.get("close", np.nan)),
            "nearest_level_name": level_name,
            "nearest_level_price": float(row.get("nearest_level_price", np.nan)),
            "side": int(side),
            "session": session_key,
            "base_prob": base_prob,
            "large_prob": float(large_prob) if large_prob is not None else np.nan,
            "score": float(score_value),
            "quality_score": float(frontier_quality_score),
            "frontier_router_lane": frontier_lane,
            "mfe_pred_r": (
                float(mfe_pred_r) if mfe_pred_r is not None else np.nan
            ),
            "mfe_score_allowed": bool(mfe_score_allowed),
            "mfe_gate_reason": str(mfe_gate_reason),
            "policy_prob": float(policy_prob) if policy_prob is not None else np.nan,
            "policy_uncertainty_std": (
                float(policy_std) if policy_std is not None else np.nan
            ),
            "accepted": bool(accepted),
            "gate_reason": str(gate_reason),
            "lane": lane,
            "outcome": int(row.get("outcome", -1)),
            "mfe_pts": float(row.get("mfe_pts", np.nan)),
            "mfe_r": float(row.get("mfe_r", np.nan)),
            "level_test_sequence": float(row.get("level_test_sequence", np.nan)),
            "level_touch_count": float(row.get("level_touch_count", np.nan)),
            "level_side_touch_density_30": float(row.get("level_side_touch_density_30", np.nan)),
            "bars_since_last_level_side_touch": float(row.get("bars_since_last_level_side_touch", np.nan)),
            "rth_level_firmness": float(row.get("rth_level_firmness", np.nan)),
            "approach_velocity_10bar": float(row.get("approach_velocity_10bar", np.nan)),
            "approach_deceleration": float(row.get("approach_deceleration", np.nan)),
            "vol_ratio_rejection": float(row.get("vol_ratio_rejection", np.nan)),
            "delta_flip_at_level": float(row.get("delta_flip_at_level", np.nan)),
        })

    diag = current.copy()
    diag["base_prob"] = np.nan
    diag["score"] = np.nan
    diag["mfe_pred_r"] = np.nan
    diag["accepted"] = False
    diag["gate_reason"] = "no_near_level"
    diag["lane"] = ""
    row_df = pd.DataFrame(rows).set_index("dt")
    diag = diag.set_index("dt")
    overlap = [c for c in row_df.columns if c in diag.columns]
    if overlap:
        diag = diag.drop(columns=overlap)
    diag = diag.join(row_df, how="left")
    if "accepted" in diag.columns:
        accepted_vals = diag["accepted"].to_numpy(dtype=object, copy=True)
        accepted_vals[pd.isna(accepted_vals)] = False
        diag["accepted"] = accepted_vals.astype(bool)
    if "gate_reason" in diag.columns:
        diag["gate_reason"] = diag["gate_reason"].fillna("no_near_level").astype(str)
    if "lane" in diag.columns:
        diag["lane"] = diag["lane"].fillna("").astype(str)
    if "mfe_score_allowed" in diag.columns:
        mfe_allowed_vals = diag["mfe_score_allowed"].to_numpy(dtype=object, copy=True)
        mfe_allowed_vals[pd.isna(mfe_allowed_vals)] = True
        diag["mfe_score_allowed"] = mfe_allowed_vals.astype(bool)
    diag = diag.reset_index()
    return diag


def _benchmark_old_episodes(
    diag: pd.DataFrame,
    path: str,
    compare_csv_path: str,
) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    bench = pd.read_csv(p)
    if bench.empty:
        return None
    bench["start_dt"] = pd.to_datetime(bench["start_dt"])
    bench["end_dt"] = pd.to_datetime(bench["end_dt"])
    day_diag = diag.copy()
    day_diag["dt"] = pd.to_datetime(day_diag["dt"])
    accepted = []
    max_prob = []
    dom_reason = []
    for _, row in bench.iterrows():
        m = (day_diag["dt"] >= row["start_dt"]) & (day_diag["dt"] <= row["end_dt"])
        sub = day_diag.loc[m]
        accepted.append(bool(sub["accepted"].any()))
        max_prob.append(float(sub["base_prob"].max()) if not sub.empty else np.nan)
        if sub.empty:
            dom_reason.append("no_rows")
        else:
            reasons = sub["gate_reason"].astype(str)
            dom_reason.append(reasons.value_counts().index[0])
    bench["accepted_new"] = accepted
    bench["max_base_prob_new"] = max_prob
    bench["dom_reason_new"] = dom_reason
    out_csv = Path(compare_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bench.to_csv(out_csv, index=False)
    ge10 = bench["max_mfe_pts"] >= 10.0
    ge20 = bench["max_mfe_pts"] >= 20.0
    return {
        "source_csv": str(p),
        "compare_csv": str(out_csv),
        "accepted": int(bench["accepted_new"].sum()),
        "total": int(len(bench)),
        "recall": float(bench["accepted_new"].mean()),
        "accepted_ge10": int(bench.loc[ge10, "accepted_new"].sum()),
        "total_ge10": int(ge10.sum()),
        "recall_ge10": float(bench.loc[ge10, "accepted_new"].mean()) if ge10.any() else None,
        "accepted_ge20": int(bench.loc[ge20, "accepted_new"].sum()),
        "total_ge20": int(ge20.sum()),
        "recall_ge20": float(bench.loc[ge20, "accepted_new"].mean()) if ge20.any() else None,
        "reject_reasons": bench.loc[~bench["accepted_new"], "dom_reason_new"].value_counts().to_dict(),
    }


def _benchmark_old_swings(
    diag: pd.DataFrame,
    path: str,
    compare_csv_path: str,
) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    bench = pd.read_csv(p)
    if bench.empty:
        return None
    bench["dt"] = pd.to_datetime(bench["dt"])
    day_diag = diag.copy()
    day_diag["dt"] = pd.to_datetime(day_diag["dt"])
    merged = bench.merge(
        day_diag[["dt", "nearest_level_name", "base_prob", "score", "accepted", "gate_reason", "outcome"]],
        on="dt",
        how="left",
        suffixes=("", "_new"),
    )
    out_csv = Path(compare_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    near = merged["near_level"].notna()
    near_and_found = near & merged["nearest_level_name"].notna()
    return {
        "source_csv": str(p),
        "compare_csv": str(out_csv),
        "accepted_near_level": int(merged.loc[near, "accepted"].fillna(False).sum()),
        "total_near_level": int(near.sum()),
        "recall_near_level": float(merged.loc[near, "accepted"].fillna(False).mean()) if near.any() else None,
        "accepted_near_and_found": int(merged.loc[near_and_found, "accepted"].fillna(False).sum()),
        "total_near_and_found": int(near_and_found.sum()),
        "no_near_level_count": int((~near).sum()),
        "reject_reasons_near": merged.loc[near & ~merged["accepted"].fillna(False), "gate_reason"].astype(str).value_counts().to_dict(),
    }


def _summarize_diag(diag: pd.DataFrame) -> Dict[str, Any]:
    rth_mask = diag["session"] == "rth"
    pos_mask = diag["outcome"] == 1
    rth_pos = diag.loc[rth_mask & pos_mask].copy()
    if not rth_pos.empty:
        rth_pos = rth_pos.sort_values("dt")
        new_ep = [True]
        for i in range(1, len(rth_pos)):
            prev = rth_pos.iloc[i - 1]
            cur = rth_pos.iloc[i]
            same_level = cur["nearest_level_name"] == prev["nearest_level_name"]
            same_side = int(cur["side"]) == int(prev["side"])
            gap_min = (pd.Timestamp(cur["dt"]) - pd.Timestamp(prev["dt"])).total_seconds() / 60.0
            new_ep.append(not (same_level and same_side and gap_min <= 1.01))
        rth_pos["episode_id"] = pd.Series(new_ep, index=rth_pos.index).cumsum()
        ep = rth_pos.groupby("episode_id").agg(
            start=("dt", "min"),
            end=("dt", "max"),
            level=("nearest_level_name", "first"),
            side=("side", "first"),
            n_rows=("dt", "size"),
            accepted_any=("accepted", "max"),
            dom_reason=("gate_reason", lambda s: s.astype(str).value_counts().index[0]),
            max_base_prob=("base_prob", "max"),
            mean_base_prob=("base_prob", "mean"),
            max_score=("score", "max"),
            mean_score=("score", "mean"),
            max_mfe_pts=("mfe_pts", "max"),
            mean_mfe_pts=("mfe_pts", "mean"),
            mean_level_test_sequence=("level_test_sequence", "mean"),
        ).reset_index(drop=True)
    else:
        ep = pd.DataFrame()

    out = {
        "rth_rows": int(rth_mask.sum()),
        "rth_near_rows": int((rth_mask & diag["nearest_level_name"].notna()).sum()),
        "rth_accepted_rows": int(diag.loc[rth_mask, "accepted"].sum()),
        "rth_outcome1_rows": int((rth_mask & pos_mask).sum()),
        "rth_bar_recall_outcome1": (
            float(diag.loc[rth_mask & pos_mask, "accepted"].mean())
            if (rth_mask & pos_mask).any()
            else None
        ),
        "rth_gate_reasons_outcome1": (
            diag.loc[rth_mask & pos_mask, "gate_reason"].astype(str).value_counts().to_dict()
        ),
        "rth_mean_base_prob_outcome1": (
            float(diag.loc[rth_mask & pos_mask, "base_prob"].mean())
            if (rth_mask & pos_mask).any()
            else None
        ),
        "rth_mean_base_prob_outcome0": (
            float(diag.loc[rth_mask & (diag["outcome"] == 0), "base_prob"].mean())
            if (rth_mask & (diag["outcome"] == 0)).any()
            else None
        ),
        "rth_mean_level_test_seq_accepted": (
            float(diag.loc[rth_mask & diag["accepted"], "level_test_sequence"].mean())
            if (rth_mask & diag["accepted"]).any()
            else None
        ),
        "rth_mean_level_test_seq_missed_outcome1": (
            float(diag.loc[rth_mask & pos_mask & ~diag["accepted"], "level_test_sequence"].mean())
            if (rth_mask & pos_mask & ~diag["accepted"]).any()
            else None
        ),
        "rth_opening_30m_outcome1_rows": int(((rth_mask & pos_mask) & (diag["dt"].dt.hour * 60 + diag["dt"].dt.minute < 420)).sum()),
        "rth_opening_30m_accepted_rows": int(((rth_mask & diag["accepted"]) & (diag["dt"].dt.hour * 60 + diag["dt"].dt.minute < 420)).sum()),
        "rth_opening_30m_bar_recall_outcome1": (
            float(diag.loc[(rth_mask & pos_mask) & (diag["dt"].dt.hour * 60 + diag["dt"].dt.minute < 420), "accepted"].mean())
            if ((rth_mask & pos_mask) & (diag["dt"].dt.hour * 60 + diag["dt"].dt.minute < 420)).any()
            else None
        ),
        "rth_episode_recall": float(ep["accepted_any"].mean()) if not ep.empty else None,
        "rth_episode_count": int(len(ep)),
        "rth_episode_recall_ge10": (
            float(ep.loc[ep["max_mfe_pts"] >= 10.0, "accepted_any"].mean())
            if (not ep.empty) and (ep["max_mfe_pts"] >= 10.0).any()
            else None
        ),
        "rth_episode_count_ge10": int((ep["max_mfe_pts"] >= 10.0).sum()) if not ep.empty else 0,
        "rth_episode_recall_ge20": (
            float(ep.loc[ep["max_mfe_pts"] >= 20.0, "accepted_any"].mean())
            if (not ep.empty) and (ep["max_mfe_pts"] >= 20.0).any()
            else None
        ),
        "rth_episode_count_ge20": int((ep["max_mfe_pts"] >= 20.0).sum()) if not ep.empty else 0,
    }
    return out, ep


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one-day reversal playback diagnostics")
    parser.add_argument("--trading-day", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--benchmark-episodes-csv", default="sandbox/results/day_2026-02-26_stable_v1_top_reversal_episodes.csv")
    parser.add_argument("--benchmark-swings-csv", default="sandbox/results/day_2026-02-26_stable_v1_top_swings.csv")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    params = dict(cfg.get("params", cfg))

    strategy = _build_strategy(params)
    history_csv = str(params.get("historical_csv_path", "raw_data/schwab/es_minute_history.csv"))
    warmup_days = int(params.get("warmup_days", 60))
    history = _load_causal_history(history_csv, args.trading_day, warmup_days)
    strategy.set_historical_context(history)

    bars_1m = _load_day_bars(args.trading_day)
    diag = _run_diag(
        strategy=strategy,
        bars_1m=bars_1m,
        label_forward_bars=int(strategy._metadata.get("label_forward_bars", 60)),
        label_excursion_ratio=float(strategy._metadata.get("label_excursion_ratio", 1.5)),
        label_breakout_excursion_ratio=float(strategy._metadata.get("label_breakout_excursion_ratio", 1.2)),
        label_terminal_disp_frac=float(strategy._metadata.get("label_terminal_disp_frac", 0.20)),
        label_reversal_vol_mult=float(strategy._metadata.get("label_reversal_vol_mult", 0.10)),
        label_breakout_vol_mult=float(strategy._metadata.get("label_breakout_vol_mult", 0.08)),
    )
    diag["dt"] = pd.to_datetime(diag["dt"])

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    diag_path = prefix.with_name(prefix.name + "_diag.parquet")
    episodes_path = prefix.with_name(prefix.name + "_episodes.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    bench_ep_compare = prefix.with_name(prefix.name + "_benchmark_old_episodes_compare.csv")
    bench_swings_compare = prefix.with_name(prefix.name + "_benchmark_old_swings_compare.csv")

    summary, episode_df = _summarize_diag(diag)
    old_ep = _benchmark_old_episodes(
        diag,
        args.benchmark_episodes_csv,
        str(bench_ep_compare),
    )
    old_swings = _benchmark_old_swings(
        diag,
        args.benchmark_swings_csv,
        str(bench_swings_compare),
    )
    diag.to_parquet(diag_path, index=False)
    episode_df.to_csv(episodes_path, index=False)

    payload = {
        "trading_day": args.trading_day,
        "config": args.config,
        "model_dir": str(params.get("model_dir")),
        "diag_parquet": str(diag_path),
        "episodes_csv": str(episodes_path),
        "summary": summary,
        "benchmark_old_episodes": old_ep,
        "benchmark_old_swings": old_swings,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
