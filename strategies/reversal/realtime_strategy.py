"""
RealtimeStrategy implementation for Phase 3 reversal prediction.

Predicts P(reversal) at near-level bars using the Phase 3 XGBoost model.
Replicates the exact feature computation pipeline from train_level_models.py
to ensure consistency between offline training and realtime inference.

Requires historical context (50-60 days of 1-min bars) for feature warm-up.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from strategies.labeling.reversal_zones import TRACKED_LEVELS
from strategies.labeling.near_level import (
    assign_level_side,
    compute_nearest_level_arrays,
)
from strategies.reversal.level_utils import level_group
from strategies.realtime.orderflow_columns import normalize_orderflow_columns
from strategies.realtime.protocol import RealtimeSignal

logger = logging.getLogger(__name__)


class ReversalPredictorStrategy:
    """Predicts P(reversal) at near-level bars using Phase 3 model."""

    def __init__(
        self,
        model_dir: str,
        pred_threshold: float = 0.50,
        pred_threshold_rth: Optional[float] = None,
        pred_threshold_ovn: Optional[float] = None,
        proximity_pts: float = 5.0,
        side_hysteresis_pts: float = 0.0,
        side_flip_confirm_pts: Optional[float] = None,
        same_level_min_confirm_bars: int = 1,
        level_signal_cooldown_bars: int = 0,
        use_episode_gating: bool = False,
        episode_gap_bars: int = 1,
        trade_budget_per_day: int = 0,
        trade_budget_rth_per_day: int = 0,
        trade_budget_ovn_per_day: int = 0,
        budget_adapt_step: float = 0.01,
        budget_adapt_tolerance: float = 0.5,
        budget_expected_init: float = 20.0,
        budget_ewma_alpha: float = 0.20,
        budget_max_threshold: float = 0.95,
        policy_model_dir: Optional[str] = None,
        policy_threshold: Optional[float] = None,
        policy_threshold_rth: Optional[float] = None,
        policy_threshold_ovn: Optional[float] = None,
        policy_mode: Optional[str] = None,
        policy_episode_gap_bars: Optional[int] = None,
        policy_use_ensemble_mean: Optional[bool] = None,
        policy_uncertainty_std_max: Optional[float] = None,
        large_move_model_dir: Optional[str] = None,
        mfe_model_dir: Optional[str] = None,
        mfe_min_base_prob: float = 0.0,
        score_mode: str = "base_prob",
        event_large_blend_alpha: float = 0.15,
        use_base_prob_prefilter: bool = True,
        split_primary_accepts: int = 1,
        dual_lane_enabled: bool = False,
        high_score_threshold: Optional[float] = None,
        high_score_threshold_rth: Optional[float] = None,
        high_score_threshold_ovn: Optional[float] = None,
        low_score_threshold: Optional[float] = None,
        low_score_threshold_rth: Optional[float] = None,
        low_score_threshold_ovn: Optional[float] = None,
        fallback_start_minute: int = 390,
        max_low_per_day: int = 1,
        allow_low_before_high: bool = False,
        low_lane_scope: str = "all",
        strong_levels: Optional[List[str]] = None,
        dual_max_trades_per_day: int = 0,
        frontier_router_enabled: bool = False,
        frontier_quality_source: str = "base_prob",
        frontier_quality_model_dir: Optional[str] = None,
        frontier_quality_lookup_path: Optional[str] = None,
        frontier_blend_live_cap: int = 5,
        frontier_blend_live_start: int = 390,
        frontier_blend_live_end: int = 780,
        frontier_blend_live_min_q: float = 0.0,
        frontier_blend_live_flatten_min_q: float = 0.66,
        frontier_blend_q2_cap: int = 2,
        frontier_blend_q2_start: int = 450,
        frontier_blend_q2_end: int = 780,
        frontier_blend_q2_min_q: float = 0.16,
        frontier_blend_q2_flatten_min_q: float = 0.16,
        frontier_blend_q2_override_prob: Optional[float] = 0.24,
        frontier_execution_mode: str = "independent",
        frontier_opposite_action: str = "ignore",
        frontier_opposite_flip_min_q: float = 0.72,
        frontier_opposite_flatten_min_q: float = 0.66,
        frontier_opposite_min_q_gap_vs_open: float = 0.0,
        frontier_opposite_flip_requires_open_loss: bool = False,
        frontier_consume_budget_on_flatten: bool = False,
        frontier_emit_flatten_signal: bool = False,
        frontier_k_total: int = 7,
        frontier_q_thr: float = 0.58,
        frontier_q_slots: int = 3,
        frontier_c_thr: float = 0.48,
        frontier_c_start: int = 450,
        frontier_c_slots: int = 2,
        frontier_r_thr: float = 0.45,
        frontier_r_start: int = 600,
        frontier_r_slots: int = 1,
        frontier_cooldown_min: int = 5,
        frontier_inertia_enabled: bool = False,
        frontier_inertia_global_minute_gap: int = 0,
        frontier_inertia_level_group_minute_gap: int = 0,
        frontier_inertia_override_min_q: Optional[float] = None,
        frontier_inertia_override_min_q_gap: float = 0.0,
        frontier_diversity_cap: int = 2,
        frontier_high_override_prob: Optional[float] = None,
        frontier_high_override_cap: int = 0,
        frontier_high_override_start: int = 390,
        frontier_high_override_window_min: int = 0,
        frontier_high_override_window_cap: int = 0,
        frontier_group_unresolved_enabled: bool = False,
        frontier_dynamic_budget_enabled: bool = False,
        frontier_dynamic_extra_cap: int = 0,
        frontier_dynamic_extra_every: int = 1,
        frontier_dynamic_start_minute: int = 390,
        frontier_loss_lock_diff: int = 0,
        frontier_loss_lock_after_nominal_only: bool = True,
        frontier_early_loss_lock_enabled: bool = False,
        frontier_early_loss_lock_start_minute: int = 390,
        frontier_early_loss_lock_end_minute: int = 450,
        frontier_early_loss_lock_trigger_diff: int = 1,
        frontier_early_loss_lock_duration_min: int = 45,
        frontier_early_loss_lock_once_per_day: bool = True,
        frontier_virtual_gate: float = 0.62,
        frontier_virtual_stop_hi: float = 10.0,
        frontier_virtual_target_hi: float = 40.0,
        frontier_virtual_stop_lo: float = 4.0,
        frontier_virtual_target_lo: float = 8.0,
        frontier_virtual_gate_calibration_enabled: bool = True,
        frontier_virtual_gate_calibration_days: int = 20,
        frontier_virtual_gate_calibration_min_samples: int = 120,
        frontier_virtual_gate_calibration_decay: float = 0.97,
        frontier_virtual_gate_calibration_grid_size: int = 41,
        frontier_virtual_gate_calibration_min: float = 0.10,
        frontier_virtual_gate_calibration_max: float = 0.80,
        mfe_regime_gate_enabled: bool = False,
        mfe_regime_drive_move_pts: float = 8.0,
        mfe_regime_drive_frac: float = 0.55,
        mfe_regime_countertrend_prob_floor: float = 0.75,
        mfe_regime_end_minute: int = 690,
    ):
        self._model_dir = model_dir
        self._pred_threshold = float(pred_threshold)
        self._pred_threshold_rth = (
            float(pred_threshold_rth) if pred_threshold_rth is not None else self._pred_threshold
        )
        self._pred_threshold_ovn = (
            float(pred_threshold_ovn) if pred_threshold_ovn is not None else self._pred_threshold
        )
        self._proximity_pts = proximity_pts
        self._side_hysteresis_pts = max(float(side_hysteresis_pts), 0.0)
        self._side_flip_confirm_pts = (
            float(side_flip_confirm_pts) if side_flip_confirm_pts is not None else None
        )
        self._same_level_min_confirm_bars = max(int(same_level_min_confirm_bars), 1)
        self._level_signal_cooldown_bars = max(int(level_signal_cooldown_bars), 0)
        self._use_episode_gating = bool(use_episode_gating)
        self._episode_gap_bars = max(int(episode_gap_bars), 0)
        self._trade_budget_per_day = max(int(trade_budget_per_day), 0)
        self._trade_budget_rth_per_day = max(int(trade_budget_rth_per_day), 0)
        self._trade_budget_ovn_per_day = max(int(trade_budget_ovn_per_day), 0)
        self._budget_adapt_step = max(float(budget_adapt_step), 0.0)
        self._budget_adapt_tolerance = max(float(budget_adapt_tolerance), 0.0)
        self._budget_ewma_alpha = min(max(float(budget_ewma_alpha), 0.0), 1.0)
        self._budget_max_threshold = max(float(budget_max_threshold), min(self._pred_threshold_rth, self._pred_threshold_ovn))
        self._budget_expected_episodes = max(float(budget_expected_init), 1.0)
        self._budget_dynamic_threshold = float(self._pred_threshold)
        self._budget_seen_today = 0
        self._budget_accepted_today = 0
        self._use_session_budget = (
            self._trade_budget_rth_per_day > 0 or self._trade_budget_ovn_per_day > 0
        )
        self._budget_expected_episodes_by_session = {
            "rth": max(float(budget_expected_init), 1.0),
            "ovn": max(float(budget_expected_init), 1.0),
        }
        self._budget_dynamic_threshold_by_session = {
            "rth": float(self._pred_threshold_rth),
            "ovn": float(self._pred_threshold_ovn),
        }
        self._budget_seen_by_session = {"rth": 0, "ovn": 0}
        self._budget_accepted_by_session = {"rth": 0, "ovn": 0}

        # Load model and metadata
        self._model = xgb.XGBClassifier()
        self._model.load_model(os.path.join(model_dir, 'model.json'))

        with open(os.path.join(model_dir, 'metadata.json')) as f:
            self._metadata = json.load(f)

        self._feature_cols: List[str] = self._metadata['feature_cols']
        md_tracked = self._metadata.get("tracked_levels", list(TRACKED_LEVELS))
        if not isinstance(md_tracked, list) or len(md_tracked) == 0:
            md_tracked = list(TRACKED_LEVELS)
        self._tracked_levels: List[str] = [str(x) for x in md_tracked]
        self._same_day_bidask_only = bool(self._metadata.get('same_day_bidask_only', False))
        self._policy_model: Optional[xgb.XGBClassifier] = None
        self._policy_feature_cols: List[str] = []
        self._policy_threshold = float(policy_threshold) if policy_threshold is not None else None
        self._policy_threshold_rth = (
            float(policy_threshold_rth) if policy_threshold_rth is not None else None
        )
        self._policy_threshold_ovn = (
            float(policy_threshold_ovn) if policy_threshold_ovn is not None else None
        )
        self._policy_mode = "row"
        self._policy_episode_gap_bars = 0
        self._policy_use_ensemble_mean = False
        self._policy_uncertainty_std_max: Optional[float] = None
        self._policy_ensemble_models: List[xgb.XGBClassifier] = []
        if policy_model_dir:
            self._load_policy_model(policy_model_dir)
        if policy_mode is not None:
            self._policy_mode = str(policy_mode).strip().lower() or "row"
        if policy_episode_gap_bars is not None:
            self._policy_episode_gap_bars = max(int(policy_episode_gap_bars), 0)
        if policy_use_ensemble_mean is not None:
            self._policy_use_ensemble_mean = bool(policy_use_ensemble_mean)
        if policy_uncertainty_std_max is not None:
            self._policy_uncertainty_std_max = max(float(policy_uncertainty_std_max), 0.0)
        if self._policy_mode not in {"row", "episode_start"}:
            raise ValueError(
                f"Unsupported policy_mode={self._policy_mode}. Expected row or episode_start."
            )

        self._mfe_model: Optional[xgb.XGBRegressor] = None
        self._mfe_feature_cols: List[str] = []
        self._large_move_model: Optional[xgb.XGBClassifier] = None
        self._large_move_feature_cols: List[str] = []
        self._mfe_min_base_prob = max(float(mfe_min_base_prob), 0.0)
        self._score_mode = "base_prob"
        self._event_large_blend_alpha = float(event_large_blend_alpha)
        self._use_base_prob_prefilter = bool(use_base_prob_prefilter)
        self._split_primary_accepts = max(int(split_primary_accepts), 1)
        self._dual_lane_enabled = bool(dual_lane_enabled)
        self._high_score_threshold = (
            float(high_score_threshold) if high_score_threshold is not None else None
        )
        self._high_score_threshold_rth = (
            float(high_score_threshold_rth) if high_score_threshold_rth is not None else None
        )
        self._high_score_threshold_ovn = (
            float(high_score_threshold_ovn) if high_score_threshold_ovn is not None else None
        )
        self._low_score_threshold = (
            float(low_score_threshold) if low_score_threshold is not None else None
        )
        self._low_score_threshold_rth = (
            float(low_score_threshold_rth) if low_score_threshold_rth is not None else None
        )
        self._low_score_threshold_ovn = (
            float(low_score_threshold_ovn) if low_score_threshold_ovn is not None else None
        )
        self._fallback_start_minute = int(fallback_start_minute)
        self._max_low_per_day = max(int(max_low_per_day), 0)
        self._allow_low_before_high = bool(allow_low_before_high)
        self._low_lane_scope = str(low_lane_scope).strip().lower() or "all"
        self._strong_levels = {
            str(x).strip()
            for x in (strong_levels or [])
            if str(x).strip()
        }
        self._dual_max_trades_per_day = max(int(dual_max_trades_per_day), 0)
        self._frontier_router_enabled = bool(frontier_router_enabled)
        self._frontier_quality_source = str(frontier_quality_source).strip().lower() or "base_prob"
        self._frontier_quality_model_dir = (
            str(frontier_quality_model_dir).strip() if frontier_quality_model_dir else ""
        )
        self._frontier_quality_lookup_path = (
            str(frontier_quality_lookup_path).strip() if frontier_quality_lookup_path else ""
        )
        self._frontier_blend_live_cap = max(int(frontier_blend_live_cap), 0)
        self._frontier_blend_live_start = int(frontier_blend_live_start)
        self._frontier_blend_live_end = int(frontier_blend_live_end)
        self._frontier_blend_live_min_q = float(frontier_blend_live_min_q)
        self._frontier_blend_live_flatten_min_q = float(frontier_blend_live_flatten_min_q)
        self._frontier_blend_q2_cap = max(int(frontier_blend_q2_cap), 0)
        self._frontier_blend_q2_start = int(frontier_blend_q2_start)
        self._frontier_blend_q2_end = int(frontier_blend_q2_end)
        self._frontier_blend_q2_min_q = float(frontier_blend_q2_min_q)
        self._frontier_blend_q2_flatten_min_q = float(frontier_blend_q2_flatten_min_q)
        self._frontier_blend_q2_override_prob = (
            float(frontier_blend_q2_override_prob)
            if frontier_blend_q2_override_prob is not None
            else None
        )
        self._frontier_execution_mode = str(frontier_execution_mode).strip().lower() or "independent"
        self._frontier_opposite_action = str(frontier_opposite_action).strip().lower() or "ignore"
        self._frontier_opposite_flip_min_q = float(frontier_opposite_flip_min_q)
        self._frontier_opposite_flatten_min_q = float(frontier_opposite_flatten_min_q)
        self._frontier_opposite_min_q_gap_vs_open = max(
            float(frontier_opposite_min_q_gap_vs_open), 0.0
        )
        self._frontier_opposite_flip_requires_open_loss = bool(
            frontier_opposite_flip_requires_open_loss
        )
        self._frontier_consume_budget_on_flatten = bool(frontier_consume_budget_on_flatten)
        self._frontier_emit_flatten_signal = bool(frontier_emit_flatten_signal)
        self._frontier_qtp_model: Optional[xgb.XGBClassifier] = None
        self._frontier_qfail_model: Optional[xgb.XGBClassifier] = None
        self._frontier_q_feature_cols: List[str] = []
        self._frontier_q_lookup: Dict[Tuple[str, int], float] = {}
        self._frontier_warned_quality_missing = False
        self._frontier_k_total = max(int(frontier_k_total), 0)
        self._frontier_q_thr = float(frontier_q_thr)
        self._frontier_q_slots = max(int(frontier_q_slots), 0)
        self._frontier_c_thr = float(frontier_c_thr)
        self._frontier_c_start = int(frontier_c_start)
        self._frontier_c_slots = max(int(frontier_c_slots), 0)
        self._frontier_r_thr = float(frontier_r_thr)
        self._frontier_r_start = int(frontier_r_start)
        self._frontier_r_slots = max(int(frontier_r_slots), 0)
        self._frontier_cooldown_min = max(int(frontier_cooldown_min), 0)
        self._frontier_inertia_enabled = bool(frontier_inertia_enabled)
        self._frontier_inertia_global_minute_gap = max(int(frontier_inertia_global_minute_gap), 0)
        self._frontier_inertia_level_group_minute_gap = max(
            int(frontier_inertia_level_group_minute_gap),
            0,
        )
        self._frontier_inertia_override_min_q = (
            float(frontier_inertia_override_min_q)
            if frontier_inertia_override_min_q is not None
            else None
        )
        self._frontier_inertia_override_min_q_gap = max(
            float(frontier_inertia_override_min_q_gap),
            0.0,
        )
        self._frontier_diversity_cap = max(int(frontier_diversity_cap), 0)
        self._frontier_high_override_prob = (
            float(frontier_high_override_prob) if frontier_high_override_prob is not None else None
        )
        # < 0 means unbounded daily overrides.
        self._frontier_high_override_cap = int(frontier_high_override_cap)
        self._frontier_high_override_start = int(frontier_high_override_start)
        self._frontier_high_override_window_min = max(int(frontier_high_override_window_min), 0)
        self._frontier_high_override_window_cap = max(int(frontier_high_override_window_cap), 0)
        self._frontier_group_unresolved_enabled = bool(frontier_group_unresolved_enabled)
        self._frontier_dynamic_budget_enabled = bool(frontier_dynamic_budget_enabled)
        self._frontier_dynamic_extra_cap = max(int(frontier_dynamic_extra_cap), 0)
        self._frontier_dynamic_extra_every = max(int(frontier_dynamic_extra_every), 1)
        self._frontier_dynamic_start_minute = int(frontier_dynamic_start_minute)
        self._frontier_loss_lock_diff = max(int(frontier_loss_lock_diff), 0)
        self._frontier_loss_lock_after_nominal_only = bool(frontier_loss_lock_after_nominal_only)
        self._frontier_early_loss_lock_enabled = bool(frontier_early_loss_lock_enabled)
        self._frontier_early_loss_lock_start_minute = int(frontier_early_loss_lock_start_minute)
        self._frontier_early_loss_lock_end_minute = int(frontier_early_loss_lock_end_minute)
        self._frontier_early_loss_lock_trigger_diff = max(int(frontier_early_loss_lock_trigger_diff), 1)
        self._frontier_early_loss_lock_duration_min = max(int(frontier_early_loss_lock_duration_min), 0)
        self._frontier_early_loss_lock_once_per_day = bool(frontier_early_loss_lock_once_per_day)
        if self._frontier_early_loss_lock_start_minute > self._frontier_early_loss_lock_end_minute:
            self._frontier_early_loss_lock_start_minute, self._frontier_early_loss_lock_end_minute = (
                self._frontier_early_loss_lock_end_minute,
                self._frontier_early_loss_lock_start_minute,
            )
        self._frontier_virtual_gate = float(frontier_virtual_gate)
        self._frontier_virtual_gate_default = float(frontier_virtual_gate)
        self._frontier_virtual_stop_hi = max(float(frontier_virtual_stop_hi), 0.0)
        self._frontier_virtual_target_hi = max(float(frontier_virtual_target_hi), 0.0)
        self._frontier_virtual_stop_lo = max(float(frontier_virtual_stop_lo), 0.0)
        self._frontier_virtual_target_lo = max(float(frontier_virtual_target_lo), 0.0)
        self._frontier_virtual_gate_calibration_enabled = bool(frontier_virtual_gate_calibration_enabled)
        self._frontier_virtual_gate_calibration_days = max(int(frontier_virtual_gate_calibration_days), 5)
        self._frontier_virtual_gate_calibration_min_samples = max(int(frontier_virtual_gate_calibration_min_samples), 20)
        self._frontier_virtual_gate_calibration_decay = min(
            max(float(frontier_virtual_gate_calibration_decay), 0.0), 1.0
        )
        self._frontier_virtual_gate_calibration_grid_size = max(
            int(frontier_virtual_gate_calibration_grid_size), 11
        )
        self._frontier_virtual_gate_calibration_min = float(frontier_virtual_gate_calibration_min)
        self._frontier_virtual_gate_calibration_max = float(frontier_virtual_gate_calibration_max)
        if self._frontier_virtual_gate_calibration_min > self._frontier_virtual_gate_calibration_max:
            self._frontier_virtual_gate_calibration_min, self._frontier_virtual_gate_calibration_max = (
                self._frontier_virtual_gate_calibration_max,
                self._frontier_virtual_gate_calibration_min,
            )
        self._frontier_router_day: Optional[str] = None
        self._frontier_router_accepted_today = 0
        self._frontier_router_q_used_today = 0
        self._frontier_router_c_used_today = 0
        self._frontier_router_r_used_today = 0
        self._frontier_router_override_used_today = 0
        self._frontier_router_override_window_counts: Dict[int, int] = {}
        self._frontier_realized_wins_today = 0
        self._frontier_realized_losses_today = 0
        self._frontier_virtual_open_trades: List[Dict[str, object]] = []
        self._frontier_seq_position: Optional[Dict[str, object]] = None
        self._frontier_router_last_accept_minute = -10_000
        self._frontier_router_side_counts: Dict[str, int] = {}
        self._frontier_inertia_last_global: Optional[Dict[str, float]] = None
        self._frontier_inertia_last_by_group: Dict[str, Dict[str, float]] = {}
        self._frontier_router_warned_policy_missing = False
        self._frontier_blend_live_used_today = 0
        self._frontier_blend_q2_used_today = 0
        self._frontier_early_loss_lock_until_minute = -1
        self._frontier_early_loss_lock_triggered_today = False
        self._dual_taken_today = 0
        self._dual_high_taken_today = 0
        self._dual_low_taken_today = 0
        self._mfe_regime_gate_enabled = bool(mfe_regime_gate_enabled)
        self._mfe_regime_drive_move_pts = max(float(mfe_regime_drive_move_pts), 0.0)
        self._mfe_regime_drive_frac = max(float(mfe_regime_drive_frac), 0.0)
        self._mfe_regime_countertrend_prob_floor = max(
            float(mfe_regime_countertrend_prob_floor), 0.0
        )
        self._mfe_regime_end_minute = int(mfe_regime_end_minute)
        # Phase-2 runtime prune:
        # keep only base_prob scoring for production path; older hybrid/ranking
        # score modes are retained only as archived research artifacts.
        requested_score_mode = str(score_mode).strip().lower() or "base_prob"
        deprecated_opts: List[str] = []
        if requested_score_mode != "base_prob":
            deprecated_opts.append(f"score_mode={requested_score_mode}")
        if large_move_model_dir:
            deprecated_opts.append("large_move_model_dir")
        if mfe_model_dir:
            deprecated_opts.append("mfe_model_dir")
        if self._mfe_min_base_prob > 0.0:
            deprecated_opts.append("mfe_min_base_prob")
        if self._mfe_regime_gate_enabled:
            deprecated_opts.append("mfe_regime_gate_enabled")
        if deprecated_opts:
            logger.warning(
                "Deprecated reversal scoring options ignored in runtime prune: %s",
                ", ".join(deprecated_opts),
            )
        # Enforce single active scoring contract at runtime.
        self._mfe_min_base_prob = 0.0
        self._mfe_regime_gate_enabled = False
        if self._low_lane_scope not in {"all", "strong_levels"}:
            raise ValueError(
                f"Unsupported low_lane_scope={self._low_lane_scope}. Expected all or strong_levels."
            )
        if self._frontier_quality_source not in {
            "base_prob",
            "policy_prob",
            "q_twohead",
            "q_lookup",
            "blend_policy_q2",
        }:
            raise ValueError(
                f"Unsupported frontier_quality_source={self._frontier_quality_source}. "
                "Expected base_prob, policy_prob, q_twohead, q_lookup, or blend_policy_q2."
            )
        if self._frontier_execution_mode not in {"independent", "sequential"}:
            raise ValueError(
                f"Unsupported frontier_execution_mode={self._frontier_execution_mode}. "
                "Expected independent or sequential."
            )
        if self._frontier_opposite_action not in {"ignore", "flatten", "flip", "flip_or_flatten"}:
            raise ValueError(
                f"Unsupported frontier_opposite_action={self._frontier_opposite_action}. "
                "Expected ignore, flatten, flip, or flip_or_flatten."
            )
        if self._frontier_router_enabled and self._frontier_k_total <= 0:
            raise ValueError(
                "frontier_router_enabled requires frontier_k_total > 0"
            )
        if self._frontier_quality_source == "q_twohead" and self._frontier_quality_model_dir:
            self._load_frontier_quality_models(self._frontier_quality_model_dir)
        if self._frontier_quality_source == "blend_policy_q2" and self._frontier_quality_model_dir:
            self._load_frontier_quality_models(self._frontier_quality_model_dir)
        if self._frontier_quality_source == "q_lookup" and self._frontier_quality_lookup_path:
            self._load_frontier_quality_lookup(self._frontier_quality_lookup_path)
        self._score_feature_cols_required = set(self._feature_cols)
        # Enable provider-level optional features when policy or q-twohead heads
        # depend on them (e.g., intraday regime + direction-stability features).
        self._score_feature_cols_required.update(self._policy_feature_cols)
        self._score_feature_cols_required.update(self._frontier_q_feature_cols)

        # Historical context for feature warm-up
        self._historical_ohlcv: Optional[pd.DataFrame] = None

        # Dedup: track emitted signals by (level_name, direction, bar_timestamp)
        self._emitted: Set[Tuple[str, str, str]] = set()
        self._last_emit_bar_by_level_dir: Dict[Tuple[str, str], int] = {}
        self._last_candidate_bar_by_level_dir: Dict[Tuple[str, str], int] = {}
        self._candidate_streak_by_level_dir: Dict[Tuple[str, str], int] = {}
        self._episode_last_bar_by_level: Dict[str, int] = {}
        self._policy_last_episode_bar: Optional[int] = None
        self._policy_last_episode_dt: Optional[pd.Timestamp] = None
        self._policy_episodes_started_today: int = 0

        # Cache last processed bar count to avoid reprocessing
        self._last_processed_count: int = 0

        logger.info(
            "ReversalPredictorStrategy loaded: %d features, threshold=%.2f, "
            "hysteresis=%.2f, flip_confirm=%s, confirm_bars=%d, cooldown_bars=%d, "
            "episode_gating=%s(gap=%d), budget=%d/day, session_budget(rth=%d,ovn=%d), "
            "thresholds(rth=%.2f,ovn=%.2f), policy=%s(mode=%s,ens=%d), "
            "base_prefilter=%s, dual_lane=%s, frontier_router=%s(qsrc=%s,k=%d,unresolved=%s,exec=%s,opp=%s), "
            "inertia=%s(g=%d,grp=%d,ovr_q=%s,ovr_gap=%.2f), "
            "early_loss_lock=%s(start=%d,end=%d,diff=%d,dur=%d,once=%s), "
            "same_day_bidask_only=%s, tracked_levels=%d",
            len(self._feature_cols),
            self._pred_threshold,
            self._side_hysteresis_pts,
            "auto" if self._side_flip_confirm_pts is None else f"{self._side_flip_confirm_pts:.2f}",
            self._same_level_min_confirm_bars,
            self._level_signal_cooldown_bars,
            self._use_episode_gating,
            self._episode_gap_bars,
            self._trade_budget_per_day,
            self._trade_budget_rth_per_day,
            self._trade_budget_ovn_per_day,
            self._pred_threshold_rth,
            self._pred_threshold_ovn,
            bool(self._policy_model is not None),
            self._policy_mode,
            len(self._policy_ensemble_models),
            self._use_base_prob_prefilter,
            self._dual_lane_enabled,
            self._frontier_router_enabled,
            self._frontier_quality_source,
            self._frontier_k_total,
            self._frontier_group_unresolved_enabled,
            self._frontier_execution_mode,
            self._frontier_opposite_action,
            self._frontier_inertia_enabled,
            self._frontier_inertia_global_minute_gap,
            self._frontier_inertia_level_group_minute_gap,
            "off" if self._frontier_inertia_override_min_q is None else f"{self._frontier_inertia_override_min_q:.2f}",
            self._frontier_inertia_override_min_q_gap,
            self._frontier_early_loss_lock_enabled,
            self._frontier_early_loss_lock_start_minute,
            self._frontier_early_loss_lock_end_minute,
            self._frontier_early_loss_lock_trigger_diff,
            self._frontier_early_loss_lock_duration_min,
            self._frontier_early_loss_lock_once_per_day,
            self._same_day_bidask_only,
            len(self._tracked_levels),
        )

    @property
    def name(self) -> str:
        return "reversal_predictor"

    # ------------------------------------------------------------------
    # Historical context
    # ------------------------------------------------------------------

    def set_historical_context(self, ohlcv_history: pd.DataFrame) -> None:
        """Provide multi-day history for feature warm-up.

        Called once during initialization with 50-60 days of 1-min bars.
        Must have columns: open, high, low, close, volume, trading_day.
        """
        prepared = self._prepare_ohlcv_schema(ohlcv_history)
        self._historical_ohlcv = prepared
        n_days = prepared['trading_day'].nunique() if 'trading_day' in prepared.columns else 0
        logger.info("Historical context set: %d bars, %d days", len(prepared), n_days)
        if self._frontier_router_enabled and self._frontier_virtual_gate_calibration_enabled:
            self._calibrate_frontier_virtual_gate_startup(prepared)

    @staticmethod
    def _simulate_contract_pnl(
        *,
        entry_idx: int,
        end_idx: int,
        direction: int,
        entry_price: float,
        stop_pts: float,
        target_pts: float,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        if end_idx <= entry_idx or direction == 0:
            return 0.0
        if direction > 0:
            stop_px = float(entry_price - stop_pts)
            target_px = float(entry_price + target_pts)
            for j in range(int(entry_idx) + 1, int(end_idx) + 1):
                hj = float(high[j])
                lj = float(low[j])
                if lj <= stop_px and hj >= target_px:
                    return -float(stop_pts)
                if lj <= stop_px:
                    return -float(stop_pts)
                if hj >= target_px:
                    return float(target_pts)
            return float(close[int(end_idx)] - entry_price)
        stop_px = float(entry_price + stop_pts)
        target_px = float(entry_price - target_pts)
        for j in range(int(entry_idx) + 1, int(end_idx) + 1):
            hj = float(high[j])
            lj = float(low[j])
            if hj >= stop_px and lj <= target_px:
                return -float(stop_pts)
            if hj >= stop_px:
                return -float(stop_pts)
            if lj <= target_px:
                return float(target_pts)
        return float(entry_price - close[int(end_idx)])

    def _compute_batch_policy_scores(
        self,
        cand: pd.DataFrame,
        base_prob: np.ndarray,
    ) -> np.ndarray:
        out = np.full(len(cand), np.nan, dtype=np.float64)
        if self._policy_model is None or len(cand) == 0:
            return out
        work = cand.copy()
        work["__row_i"] = np.arange(len(work), dtype=np.int64)
        for _, g in work.groupby(work["trading_day"].astype(str), sort=False):
            g2 = g.sort_values(["minute_of_day", "__src_idx"], ascending=[True, True])
            episode_idx = 0
            last_ts: Optional[pd.Timestamp] = None
            row_i: List[int] = []
            feat_rows: List[np.ndarray] = []
            for _, rr in g2.iterrows():
                i = int(rr["__row_i"])
                ts = None
                if "dt" in rr.index and pd.notna(rr["dt"]):
                    ts = pd.Timestamp(rr["dt"])
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert("America/Los_Angeles")
                minutes_since_prev = 9999.0
                if ts is not None and last_ts is not None:
                    minutes_since_prev = max(
                        float((ts - last_ts).total_seconds() / 60.0), 0.0
                    )
                x = self._build_policy_feature_row(
                    row=rr,
                    base_prob=float(base_prob[i]),
                    session_key="rth",
                    episode_index_so_far=int(episode_idx),
                    minutes_since_prev_episode=float(minutes_since_prev),
                )
                if x is None:
                    continue
                row_i.append(int(i))
                feat_rows.append(x.reshape(-1))
                episode_idx += 1
                if ts is not None:
                    last_ts = ts
            if not feat_rows:
                continue
            X = np.vstack(feat_rows).astype(np.float32, copy=False)
            pred = self._policy_model.predict_proba(X)[:, 1].astype(np.float64)
            if self._policy_ensemble_models and self._policy_use_ensemble_mean:
                ens_mat = np.vstack(
                    [m.predict_proba(X)[:, 1].astype(np.float64) for m in self._policy_ensemble_models]
                )
                if ens_mat.size:
                    pred = ens_mat.mean(axis=0)
            out[np.asarray(row_i, dtype=np.int64)] = pred
        return out

    def _compute_batch_qtwohead_scores(
        self,
        cand: pd.DataFrame,
        base_prob: np.ndarray,
    ) -> np.ndarray:
        out = np.full(len(cand), np.nan, dtype=np.float64)
        if (
            self._frontier_qtp_model is None
            or self._frontier_qfail_model is None
            or not self._frontier_q_feature_cols
            or len(cand) == 0
        ):
            return out
        X = np.zeros((len(cand), len(self._frontier_q_feature_cols)), dtype=np.float32)
        for j, col in enumerate(self._frontier_q_feature_cols):
            if col == "event_base_prob":
                X[:, j] = base_prob.astype(np.float32, copy=False)
                continue
            if col not in cand.columns:
                X[:, j] = 0.0
                continue
            X[:, j] = (
                pd.to_numeric(cand[col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )
        tp = self._frontier_qtp_model.predict_proba(X)[:, 1].astype(np.float64)
        fail = self._frontier_qfail_model.predict_proba(X)[:, 1].astype(np.float64)
        out[:] = np.clip(tp * (1.0 - fail), 0.0, 1.0)
        return out

    def _compute_batch_quality_scores(
        self,
        cand: pd.DataFrame,
        base_prob: np.ndarray,
    ) -> np.ndarray:
        src = str(self._frontier_quality_source).strip().lower()
        if src == "base_prob":
            return base_prob.astype(np.float64, copy=False)

        if src == "policy_prob":
            policy_prob = self._compute_batch_policy_scores(cand, base_prob)
            out = policy_prob.copy()
            out[~np.isfinite(out)] = base_prob[~np.isfinite(out)]
            return out
        if src == "q_twohead":
            q2_prob = self._compute_batch_qtwohead_scores(cand, base_prob)
            out = q2_prob.copy()
            out[~np.isfinite(out)] = base_prob[~np.isfinite(out)]
            return out
        if src == "q_lookup":
            out = np.full(len(cand), np.nan, dtype=np.float64)
            if self._frontier_q_lookup:
                td = cand["trading_day"].astype(str).to_numpy()
                epi = cand["__src_idx"].to_numpy(dtype=np.int64, copy=False)
                for i in range(len(cand)):
                    qv = self._frontier_q_lookup.get((str(td[i]), int(epi[i])))
                    try:
                        fv = float(qv)
                        out[i] = fv if np.isfinite(fv) else np.nan
                    except Exception:
                        out[i] = np.nan
            out[~np.isfinite(out)] = base_prob[~np.isfinite(out)]
            return out
        if src == "blend_policy_q2":
            policy_prob = self._compute_batch_policy_scores(cand, base_prob)
            q2_prob = self._compute_batch_qtwohead_scores(cand, base_prob)
            minute = pd.to_numeric(cand["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
            live_q = policy_prob.copy()
            live_q[~np.isfinite(live_q)] = base_prob[~np.isfinite(live_q)]
            out = np.full(len(cand), np.nan, dtype=np.float64)
            live_mask = (
                (minute >= int(self._frontier_blend_live_start))
                & (minute < int(self._frontier_blend_live_end))
                & (live_q >= float(self._frontier_blend_live_min_q))
            )
            out[live_mask] = live_q[live_mask]
            q2_mask = (
                (minute >= int(self._frontier_blend_q2_start))
                & (minute < int(self._frontier_blend_q2_end))
                & np.isfinite(q2_prob)
                & (q2_prob >= float(self._frontier_blend_q2_min_q))
            )
            overwrite = q2_mask & ((~np.isfinite(out)) | (q2_prob > out))
            out[overwrite] = q2_prob[overwrite]
            return out

        # Unknown source fallback.
        return base_prob.astype(np.float64, copy=False)

    def _calibrate_frontier_virtual_gate_startup(self, prepared: pd.DataFrame) -> None:
        """Batch recalibrate lane split gate from trailing resolved history.

        Falls back to configured default gate on any failure/insufficient data.
        """
        t0 = time.perf_counter()
        self._frontier_virtual_gate = float(self._frontier_virtual_gate_default)
        try:
            if prepared.empty or "trading_day" not in prepared.columns:
                logger.warning(
                    "Frontier gate calibration skipped: no historical bars; using default gate=%.3f",
                    float(self._frontier_virtual_gate_default),
                )
                return

            day_vals = prepared["trading_day"].astype(str).dropna().unique().tolist()
            if not day_vals:
                logger.warning(
                    "Frontier gate calibration skipped: no trading_day values; using default gate=%.3f",
                    float(self._frontier_virtual_gate_default),
                )
                return
            parsed = pd.to_datetime(pd.Series(day_vals), errors="coerce")
            if parsed.notna().all():
                order = np.argsort(parsed.to_numpy())
                sorted_days = [str(day_vals[i]) for i in order]
            else:
                sorted_days = sorted(day_vals, key=lambda x: int(x) if str(x).isdigit() else str(x))
            use_days = set(sorted_days[-int(self._frontier_virtual_gate_calibration_days):])
            hist = prepared.loc[prepared["trading_day"].astype(str).isin(use_days)].copy().reset_index(drop=True)
            if hist.empty:
                logger.warning(
                    "Frontier gate calibration skipped: trailing window empty; using default gate=%.3f",
                    float(self._frontier_virtual_gate_default),
                )
                return

            feat = self._compute_features(hist)
            if feat is None or feat.empty:
                logger.warning(
                    "Frontier gate calibration skipped: feature build failed; using default gate=%.3f",
                    float(self._frontier_virtual_gate_default),
                )
                return
            feat = self._find_nearest_levels(feat)
            feat = self._compute_level_encoding(feat, feat)
            feat = self._append_opening_drive_regime(feat)
            feat = feat.reset_index(drop=True)
            feat["__src_idx"] = np.arange(len(feat), dtype=np.int64)

            if "minute_of_day" not in feat.columns:
                dt = pd.to_datetime(feat.get("dt"), errors="coerce")
                feat["minute_of_day"] = (dt.dt.hour * 60 + dt.dt.minute).fillna(-1).astype(np.int32)
            minute = pd.to_numeric(feat["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
            near = feat["nearest_level_name"].astype(str)
            direction = pd.to_numeric(feat.get("trade_direction"), errors="coerce").fillna(0.0).to_numpy(dtype=np.int8, copy=False)
            mask = (
                near.ne("")
                & pd.notna(feat["nearest_level_price"])
                & (minute >= 390)
                & (minute < 780)
                & (direction != 0)
            )
            cand = feat.loc[mask].copy().reset_index(drop=True)
            if len(cand) < int(self._frontier_virtual_gate_calibration_min_samples):
                logger.warning(
                    "Frontier gate calibration skipped: samples=%d < min=%d; using default gate=%.3f",
                    int(len(cand)),
                    int(self._frontier_virtual_gate_calibration_min_samples),
                    float(self._frontier_virtual_gate_default),
                )
                return

            for c in self._feature_cols:
                if c not in cand.columns:
                    cand[c] = 0.0
            X = (
                cand.loc[:, self._feature_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float32, copy=False)
            )
            base_prob = self._model.predict_proba(X)[:, 1].astype(np.float64)
            q = self._compute_batch_quality_scores(cand, base_prob)
            keep = np.isfinite(q)
            cand = cand.loc[keep].copy().reset_index(drop=True)
            q = q[keep]
            if len(cand) < int(self._frontier_virtual_gate_calibration_min_samples):
                logger.warning(
                    "Frontier gate calibration skipped: valid quality samples=%d < min=%d; using default gate=%.3f",
                    int(len(cand)),
                    int(self._frontier_virtual_gate_calibration_min_samples),
                    float(self._frontier_virtual_gate_default),
                )
                return

            # Restrict calibration to a router-feasible subset (causal cadence + budget),
            # so gate optimization matches live emitted-signal distribution better than
            # using every near-level candidate row.
            sel = np.zeros(len(cand), dtype=bool)
            for _, g in cand.groupby(cand["trading_day"].astype(str), sort=False):
                order = g.sort_values(["minute_of_day", "__src_idx"], ascending=[True, True]).index.to_numpy(dtype=np.int64, copy=False)
                accepted = 0
                last_minute = -10_000
                side_counts: Dict[str, int] = {}
                unresolved_until: Dict[Tuple[str, int], int] = {}
                for ii in order:
                    m = int(pd.to_numeric(cand.at[int(ii), "minute_of_day"], errors="coerce"))
                    if accepted >= int(self._frontier_k_total):
                        break
                    if (m - int(last_minute)) < int(self._frontier_cooldown_min):
                        continue
                    lvl = str(cand.at[int(ii), "nearest_level_name"])
                    d = int(pd.to_numeric(cand.at[int(ii), "trade_direction"], errors="coerce"))
                    key = f"{lvl}:{int(d)}"
                    if (
                        int(self._frontier_diversity_cap) > 0
                        and int(side_counts.get(key, 0)) >= int(self._frontier_diversity_cap)
                    ):
                        continue
                    if bool(self._frontier_group_unresolved_enabled):
                        unr_key = (str(lvl), int(d))
                        if int(unresolved_until.get(unr_key, -10_000)) >= int(m):
                            continue
                        unresolved_until[unr_key] = int(m) + int(max(self._frontier_cooldown_min, 5))
                    sel[int(ii)] = True
                    accepted += 1
                    last_minute = int(m)
                    side_counts[key] = int(side_counts.get(key, 0)) + 1
            cand = cand.loc[sel].copy().reset_index(drop=True)
            q = q[sel]
            if len(cand) < int(self._frontier_virtual_gate_calibration_min_samples):
                logger.warning(
                    "Frontier gate calibration skipped after router-feasible selection: samples=%d < min=%d; using default gate=%.3f",
                    int(len(cand)),
                    int(self._frontier_virtual_gate_calibration_min_samples),
                    float(self._frontier_virtual_gate_default),
                )
                return

            close = pd.to_numeric(feat["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            high = pd.to_numeric(feat["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            low = pd.to_numeric(feat["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
            td = feat["trading_day"].astype(str)
            day_end: Dict[str, int] = {}
            for day, g in feat.groupby(td, sort=False):
                gm = pd.to_numeric(g["minute_of_day"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
                gi = g.index.to_numpy(dtype=np.int64, copy=False)
                rth_idx = gi[(gm >= 390) & (gm < 780)]
                if rth_idx.size > 0:
                    day_end[str(day)] = int(rth_idx[-1])
                elif gi.size > 0:
                    day_end[str(day)] = int(gi[-1])

            n = len(cand)
            pnl_hi = np.zeros(n, dtype=np.float64)
            pnl_lo = np.zeros(n, dtype=np.float64)
            td_c = cand["trading_day"].astype(str).to_numpy()
            idx_c = cand["__src_idx"].to_numpy(dtype=np.int64, copy=False)
            dir_c = pd.to_numeric(cand["trade_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int8, copy=False)
            for i in range(n):
                idx0 = int(idx_c[i])
                end = int(day_end.get(str(td_c[i]), idx0))
                if end <= idx0:
                    continue
                ep = float(close[idx0])
                d = int(dir_c[i])
                pnl_hi[i] = self._simulate_contract_pnl(
                    entry_idx=idx0,
                    end_idx=end,
                    direction=d,
                    entry_price=ep,
                    stop_pts=float(self._frontier_virtual_stop_hi),
                    target_pts=float(self._frontier_virtual_target_hi),
                    high=high,
                    low=low,
                    close=close,
                )
                pnl_lo[i] = self._simulate_contract_pnl(
                    entry_idx=idx0,
                    end_idx=end,
                    direction=d,
                    entry_price=ep,
                    stop_pts=float(self._frontier_virtual_stop_lo),
                    target_pts=float(self._frontier_virtual_target_lo),
                    high=high,
                    low=low,
                    close=close,
                )

            day_order = {d: i for i, d in enumerate(sorted_days[-int(self._frontier_virtual_gate_calibration_days):])}
            age = np.asarray(
                [max(len(day_order) - 1 - int(day_order.get(str(d), 0)), 0) for d in td_c],
                dtype=np.int32,
            )
            decay = float(self._frontier_virtual_gate_calibration_decay)
            if decay <= 0.0:
                w = np.ones(n, dtype=np.float64)
            else:
                w = np.power(decay, age.astype(np.float64, copy=False))
            w_sum = float(w.sum())
            if w_sum <= 1e-9:
                w = np.ones(n, dtype=np.float64)
                w_sum = float(w.sum())

            gate_min = max(float(self._frontier_virtual_gate_calibration_min), float(np.nanmin(q)))
            gate_max = min(float(self._frontier_virtual_gate_calibration_max), float(np.nanmax(q)))
            if gate_max <= gate_min:
                gate = float(np.clip(np.nanmedian(q), gate_min, gate_max))
                self._frontier_virtual_gate = float(gate)
                return
            grid = np.linspace(
                gate_min,
                gate_max,
                int(self._frontier_virtual_gate_calibration_grid_size),
                dtype=np.float64,
            )
            best_gate = float(self._frontier_virtual_gate_default)
            best_ev = -np.inf
            for g in grid:
                pnl = np.where(q >= float(g), pnl_hi, pnl_lo)
                ev = float(np.dot(pnl, w) / w_sum)
                if ev > best_ev:
                    best_ev = ev
                    best_gate = float(g)

            self._frontier_virtual_gate = float(
                np.clip(
                    best_gate,
                    float(self._frontier_virtual_gate_calibration_min),
                    float(self._frontier_virtual_gate_calibration_max),
                )
            )
            elapsed = float(time.perf_counter() - t0)
            logger.info(
                "Frontier gate calibrated at startup: gate %.3f -> %.3f (samples=%d, days=%d, ev=%.4f, elapsed=%.2fs)",
                float(self._frontier_virtual_gate_default),
                float(self._frontier_virtual_gate),
                int(len(cand)),
                int(len(use_days)),
                float(best_ev),
                float(elapsed),
            )
        except Exception:
            self._frontier_virtual_gate = float(self._frontier_virtual_gate_default)
            logger.exception(
                "Frontier gate calibration failed; using default gate=%.3f",
                float(self._frontier_virtual_gate_default),
            )

    def _load_policy_model(self, policy_model_dir: str) -> None:
        """Load optional stage-2 policy gate model."""
        model_path_candidates = [
            os.path.join(policy_model_dir, "policy_model.json"),
            os.path.join(policy_model_dir, "model.json"),
        ]
        model_path = next((p for p in model_path_candidates if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(
                f"No policy model found in {policy_model_dir} (expected policy_model.json or model.json)"
            )

        md_path_candidates = [
            os.path.join(policy_model_dir, "policy_metadata.json"),
            os.path.join(policy_model_dir, "metadata.json"),
        ]
        md_path = next((p for p in md_path_candidates if os.path.exists(p)), None)
        if md_path is None:
            raise FileNotFoundError(
                f"No policy metadata found in {policy_model_dir} (expected policy_metadata.json or metadata.json)"
            )

        with open(md_path) as f:
            md = json.load(f)

        feature_cols = md.get("feature_cols", [])
        if not isinstance(feature_cols, list) or not feature_cols:
            raise ValueError(f"Policy metadata missing non-empty feature_cols: {md_path}")

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        self._policy_model = model
        self._policy_feature_cols = [str(c) for c in feature_cols]
        self._policy_mode = str(md.get("policy_mode", self._policy_mode)).strip().lower() or "row"
        self._policy_episode_gap_bars = max(
            int(md.get("policy_episode_gap_bars", self._policy_episode_gap_bars)),
            0,
        )
        self._policy_use_ensemble_mean = bool(
            md.get("policy_use_ensemble_mean", self._policy_use_ensemble_mean)
        )
        if md.get("policy_uncertainty_std_max") is not None:
            self._policy_uncertainty_std_max = max(
                float(md["policy_uncertainty_std_max"]),
                0.0,
            )

        ensemble_paths = md.get("ensemble_model_paths", [])
        self._policy_ensemble_models = []
        if isinstance(ensemble_paths, list):
            for rel_path in ensemble_paths:
                if not str(rel_path).strip():
                    continue
                full_path = (
                    str(rel_path)
                    if os.path.isabs(str(rel_path))
                    else os.path.join(policy_model_dir, str(rel_path))
                )
                if not os.path.exists(full_path):
                    raise FileNotFoundError(
                        f"Policy ensemble member missing: {full_path}"
                    )
                ens_model = xgb.XGBClassifier()
                ens_model.load_model(full_path)
                self._policy_ensemble_models.append(ens_model)

        if self._policy_threshold is None:
            self._policy_threshold = float(md.get("threshold", 0.5))
        if self._policy_threshold_rth is None:
            self._policy_threshold_rth = float(self._policy_threshold)
        if self._policy_threshold_ovn is None:
            self._policy_threshold_ovn = float(self._policy_threshold)
        logger.info(
            "Loaded policy model: %s (features=%d, threshold=%.3f, mode=%s, ensemble=%d)",
            model_path,
            len(self._policy_feature_cols),
            float(self._policy_threshold),
            self._policy_mode,
            len(self._policy_ensemble_models),
        )

    def _load_mfe_model(self, mfe_model_dir: str) -> None:
        """Deprecated in runtime prune; kept only for backwards compatibility."""
        logger.warning(
            "Ignoring deprecated MFE model path in runtime prune: %s",
            mfe_model_dir,
        )
        self._mfe_model = None
        self._mfe_feature_cols = []

    def _load_large_move_model(self, large_move_model_dir: str) -> None:
        """Deprecated in runtime prune; kept only for backwards compatibility."""
        logger.warning(
            "Ignoring deprecated large-move model path in runtime prune: %s",
            large_move_model_dir,
        )
        self._large_move_model = None
        self._large_move_feature_cols = []

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def process(self, bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        """Process current day's 1-min bars. Returns new signals.

        Args:
            bars_1m: 1-minute OHLCV bars with datetime index (LA timezone).
                     Must have columns: open, high, low, close, volume.
                     Orderflow columns should use canonical naming:
                     ask=`askvolume`, bid=`bidvolume`.
        """
        bars_1m = normalize_orderflow_columns(
            bars_1m,
            copy=True,
        )
        if bars_1m.empty or len(bars_1m) <= self._last_processed_count:
            return []

        # Combine historical + current day into a single DataFrame
        combined = self._build_combined_df(bars_1m)
        if combined is None or len(combined) < 100:
            return []

        # Compute all features on the combined data
        featured = self._compute_features(combined)
        if featured is None:
            return []

        # Extract only current-day bars for prediction
        current_day_start = len(combined) - len(bars_1m)
        current_day_df = featured.iloc[current_day_start:].copy()
        current_day_df['__bar_num'] = np.arange(len(current_day_df), dtype=np.int32)

        # Find nearest level and label for each bar
        current_day_df = self._find_nearest_levels(current_day_df)

        # Compute level-encoding features
        current_day_df = self._compute_level_encoding(current_day_df, featured)
        current_day_df = self._append_opening_drive_regime(current_day_df)

        # Predict on new near-level bars
        new_signals = self._predict_and_emit(current_day_df, bars_1m)

        self._last_processed_count = len(bars_1m)
        return new_signals

    def reset_day(self) -> None:
        """Reset per-day state for a new trading day."""
        if self._trade_budget_per_day > 0 and self._budget_seen_today > 0:
            alpha = self._budget_ewma_alpha
            self._budget_expected_episodes = (
                (1.0 - alpha) * self._budget_expected_episodes
                + alpha * float(self._budget_seen_today)
            )
        if self._use_session_budget:
            alpha = self._budget_ewma_alpha
            for sess in ("rth", "ovn"):
                seen = int(self._budget_seen_by_session.get(sess, 0))
                if seen > 0:
                    self._budget_expected_episodes_by_session[sess] = (
                        (1.0 - alpha) * float(self._budget_expected_episodes_by_session[sess])
                        + alpha * float(seen)
                    )

        self._emitted.clear()
        self._last_emit_bar_by_level_dir.clear()
        self._last_candidate_bar_by_level_dir.clear()
        self._candidate_streak_by_level_dir.clear()
        self._episode_last_bar_by_level.clear()
        self._policy_last_episode_bar = None
        self._policy_last_episode_dt = None
        self._policy_episodes_started_today = 0
        self._budget_dynamic_threshold = float(self._pred_threshold)
        self._budget_seen_today = 0
        self._budget_accepted_today = 0
        self._budget_dynamic_threshold_by_session = {
            "rth": float(self._pred_threshold_rth),
            "ovn": float(self._pred_threshold_ovn),
        }
        self._budget_seen_by_session = {"rth": 0, "ovn": 0}
        self._budget_accepted_by_session = {"rth": 0, "ovn": 0}
        self._dual_taken_today = 0
        self._dual_high_taken_today = 0
        self._dual_low_taken_today = 0
        self._reset_frontier_router_state()
        self._last_processed_count = 0

    # ------------------------------------------------------------------
    # Feature computation (replicates train_level_models.py pipeline)
    # ------------------------------------------------------------------

    def _build_combined_df(self, bars_1m: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Combine historical context with current day bars."""
        # Convert bars_1m from datetime-indexed to flat DataFrame
        current = bars_1m.copy()
        if current.index.name == 'dt' or isinstance(current.index, pd.DatetimeIndex):
            current = current.reset_index()
            if 'dt' not in current.columns:
                current.rename(columns={current.columns[0]: 'dt'}, inplace=True)

        # Ensure dt column exists
        if 'dt' not in current.columns and isinstance(bars_1m.index, pd.DatetimeIndex):
            current['dt'] = bars_1m.index
        current = self._prepare_ohlcv_schema(current)

        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in current.columns:
                logger.warning("Missing column %s in bars_1m", col)
                return None

        if self._historical_ohlcv is not None and len(self._historical_ohlcv) > 0:
            historical = self._historical_ohlcv.copy()
            if 'ovn' in current.columns and 'ovn' not in historical.columns and 'dt' in historical.columns:
                historical['ovn'] = self._compute_ovn_flag(historical['dt'])
            if 'nearby_gamma_score' in current.columns and 'nearby_gamma_score' not in historical.columns:
                historical['nearby_gamma_score'] = 0.0
            # Ensure consistent columns
            common_cols = list(set(historical.columns) & set(current.columns))
            combined = pd.concat([historical[common_cols], current[common_cols]], ignore_index=True)
        else:
            combined = current

        return combined

    def _prepare_ohlcv_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize schema needed by feature providers."""
        out = normalize_orderflow_columns(df, copy=True)
        out.columns = out.columns.str.lower()

        if 'dt' not in out.columns and {'date', 'time'}.issubset(out.columns):
            out['dt'] = pd.to_datetime(
                out['date'].astype(str) + ' ' + out['time'].astype(str),
                errors='coerce',
            )
        elif 'dt' in out.columns:
            out['dt'] = pd.to_datetime(out['dt'], errors='coerce')

        if 'dt' in out.columns:
            out = out.dropna(subset=['dt'])

        if 'trading_day' not in out.columns and 'dt' in out.columns:
            out['trading_day'] = self._compute_trading_day(out['dt'])
        if 'ovn' not in out.columns and 'dt' in out.columns:
            out['ovn'] = self._compute_ovn_flag(out['dt'])
        if 'nearby_gamma_score' not in out.columns:
            out['nearby_gamma_score'] = 0.0

        return out

    @staticmethod
    def _compute_trading_day(dt_series: pd.Series) -> pd.Series:
        """Compute trading_day from datetime series (6 PM ET boundary)."""
        # Convert to naive if tz-aware
        if hasattr(dt_series.dt, 'tz') and dt_series.dt.tz is not None:
            dt_naive = dt_series.dt.tz_localize(None)
        else:
            dt_naive = dt_series

        # Bars after 3 PM LA (6 PM ET) belong to next day
        trading_day = dt_naive.dt.date.astype(str)
        after_boundary = dt_naive.dt.hour >= 15
        next_day = (dt_naive + pd.Timedelta(days=1)).dt.date.astype(str)
        trading_day = trading_day.where(~after_boundary, next_day)
        return trading_day

    @staticmethod
    def _compute_ovn_flag(dt_series: pd.Series) -> pd.Series:
        """Compute overnight session flag matching training schema."""
        if hasattr(dt_series.dt, 'tz') and dt_series.dt.tz is not None:
            dt_local = dt_series.dt.tz_convert('America/Los_Angeles')
        else:
            dt_local = dt_series

        hour = dt_local.dt.hour
        minute = dt_local.dt.minute
        ovn = (hour < 6) | (hour > 13) | ((hour == 6) & (minute < 30))
        return ovn.astype(np.int8)

    def _compute_features(self, ohlcv: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Replicate the exact feature pipeline from train_level_models.py."""
        try:
            # 1. Compute price levels
            ohlcv = self._compute_levels(ohlcv)

            # 2. Compute all feature providers
            ohlcv = self._compute_all_providers(ohlcv)

            return ohlcv
        except Exception as e:
            logger.error("Feature computation failed: %s", e, exc_info=True)
            return None

    def _compute_levels(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute price levels (mirrors train_level_models.compute_levels)."""
        from strategies.features.price_levels import PriceLevelProvider

        plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
        feat_df = plp._compute_impl(ohlcv)

        level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']
        if 'level_1' in feat_df.columns:
            # Robust against duplicate dt values (e.g., warmup + current-day overlap).
            feat_by_row = feat_df.set_index('level_1')
            for col in level_cols:
                if col in feat_by_row.columns:
                    ohlcv[col] = feat_by_row[col].reindex(ohlcv.index).values
        else:
            if 'dt' in feat_df.columns:
                feat_df = feat_df.set_index('dt')
            ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv
            for col in level_cols:
                if col in feat_df.columns:
                    src = feat_df[col]
                    if src.index.has_duplicates:
                        src = src.groupby(level=0).last()
                    ohlcv[col] = src.reindex(ohlcv_dt.index).values

        levels = plp.prev_day_levels(ohlcv)
        ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
        ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])

        tracked_set = set(self._tracked_levels)
        needs_intraday_levels = bool(
            tracked_set
            & {
                "ib_mid",
                "ib_hi_ext_025",
                "ib_hi_ext_050",
                "ib_hi_ext_100",
                "ib_lo_ext_025",
                "ib_lo_ext_050",
                "ib_lo_ext_100",
                "or5_hi",
                "or5_lo",
                "or5_mid",
                "or15_hi",
                "or15_lo",
                "or15_mid",
                "vwap_up_1s",
                "vwap_dn_1s",
                "vwap_up_15s",
                "vwap_dn_15s",
                "vwap_up_2s",
                "vwap_dn_2s",
            }
        )
        if needs_intraday_levels and "dt" in ohlcv.columns:
            dt = pd.to_datetime(ohlcv["dt"], errors="coerce")
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt_local = dt.dt.tz_convert("America/Los_Angeles")
            else:
                dt_local = dt
            ohlcv["minute_of_day"] = (
                dt_local.dt.hour * 60 + dt_local.dt.minute
            ).astype(np.int32)

        # Prior RTH open/close references (previous trading day constants).
        if tracked_set & {"prior_rth_open", "prior_rth_close"}:
            if "minute_of_day" not in ohlcv.columns and "dt" in ohlcv.columns:
                dt = pd.to_datetime(ohlcv["dt"], errors="coerce")
                if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                    dt_local = dt.dt.tz_convert("America/Los_Angeles")
                else:
                    dt_local = dt
                ohlcv["minute_of_day"] = (
                    dt_local.dt.hour * 60 + dt_local.dt.minute
                ).astype(np.int32)
            rth_mask = (ohlcv["minute_of_day"] >= 390) & (ohlcv["minute_of_day"] < 780)
            rth = ohlcv.loc[rth_mask, ["trading_day", "open", "close"]].copy()
            daily_rth_open = rth.groupby("trading_day")["open"].first() if not rth.empty else pd.Series(dtype=np.float64)
            daily_rth_close = rth.groupby("trading_day")["close"].last() if not rth.empty else pd.Series(dtype=np.float64)
            if "prior_rth_open" in tracked_set:
                ohlcv["prior_rth_open"] = ohlcv["trading_day"].map(daily_rth_open.shift(1).to_dict())
            if "prior_rth_close" in tracked_set:
                ohlcv["prior_rth_close"] = ohlcv["trading_day"].map(daily_rth_close.shift(1).to_dict())

        # Opening-range references.
        if tracked_set & {"or5_hi", "or5_lo", "or5_mid", "or15_hi", "or15_lo", "or15_mid"}:
            for col in ("or5_hi", "or5_lo", "or5_mid", "or15_hi", "or15_lo", "or15_mid"):
                if col in tracked_set and col not in ohlcv.columns:
                    ohlcv[col] = np.nan

            for _, g in ohlcv.groupby("trading_day", sort=False):
                gi = g.index
                mod = ohlcv.loc[gi, "minute_of_day"].to_numpy(dtype=np.int32, copy=False)
                highs = ohlcv.loc[gi, "high"].to_numpy(dtype=np.float64, copy=False)
                lows = ohlcv.loc[gi, "low"].to_numpy(dtype=np.float64, copy=False)

                if tracked_set & {"or5_hi", "or5_lo", "or5_mid"}:
                    m5 = (mod >= 390) & (mod <= 394)
                    if np.any(m5):
                        or5_hi = float(np.nanmax(highs[m5]))
                        or5_lo = float(np.nanmin(lows[m5]))
                        valid = gi[mod >= 395]
                        if len(valid) > 0:
                            if "or5_hi" in tracked_set:
                                ohlcv.loc[valid, "or5_hi"] = or5_hi
                            if "or5_lo" in tracked_set:
                                ohlcv.loc[valid, "or5_lo"] = or5_lo
                            if "or5_mid" in tracked_set:
                                ohlcv.loc[valid, "or5_mid"] = (or5_hi + or5_lo) / 2.0

                if tracked_set & {"or15_hi", "or15_lo", "or15_mid"}:
                    m15 = (mod >= 390) & (mod <= 404)
                    if np.any(m15):
                        or15_hi = float(np.nanmax(highs[m15]))
                        or15_lo = float(np.nanmin(lows[m15]))
                        valid = gi[mod >= 405]
                        if len(valid) > 0:
                            if "or15_hi" in tracked_set:
                                ohlcv.loc[valid, "or15_hi"] = or15_hi
                            if "or15_lo" in tracked_set:
                                ohlcv.loc[valid, "or15_lo"] = or15_lo
                            if "or15_mid" in tracked_set:
                                ohlcv.loc[valid, "or15_mid"] = (or15_hi + or15_lo) / 2.0

        # VWAP sigma bands (running, causal).
        vwap_sigma_mult = {
            "vwap_up_1s": +1.0,
            "vwap_dn_1s": -1.0,
            "vwap_up_15s": +1.5,
            "vwap_dn_15s": -1.5,
            "vwap_up_2s": +2.0,
            "vwap_dn_2s": -2.0,
        }
        if tracked_set & set(vwap_sigma_mult.keys()):
            for col in vwap_sigma_mult:
                if col in tracked_set and col not in ohlcv.columns:
                    ohlcv[col] = np.nan
            for _, g in ohlcv.groupby("trading_day", sort=False):
                gi = g.index
                vwap = ohlcv.loc[gi, "vwap"].to_numpy(dtype=np.float64, copy=False)
                tp = (
                    ohlcv.loc[gi, "open"].to_numpy(dtype=np.float64, copy=False)
                    + ohlcv.loc[gi, "high"].to_numpy(dtype=np.float64, copy=False)
                    + ohlcv.loc[gi, "low"].to_numpy(dtype=np.float64, copy=False)
                    + ohlcv.loc[gi, "close"].to_numpy(dtype=np.float64, copy=False)
                ) / 4.0
                dev = tp - vwap
                sigma = pd.Series(dev).expanding(min_periods=10).std().to_numpy(dtype=np.float64, copy=False)
                valid = np.isfinite(vwap) & np.isfinite(sigma)
                if not np.any(valid):
                    continue
                for col, mult in vwap_sigma_mult.items():
                    if col not in tracked_set:
                        continue
                    out = np.full(len(gi), np.nan, dtype=np.float64)
                    out[valid] = vwap[valid] + float(mult) * sigma[valid]
                    ohlcv.loc[gi, col] = out

        ib_relative_levels = {
            "ib_mid",
            "ib_hi_ext_025",
            "ib_hi_ext_050",
            "ib_hi_ext_100",
            "ib_lo_ext_025",
            "ib_lo_ext_050",
            "ib_lo_ext_100",
        }
        needs_ib_relative = any(lvl in ib_relative_levels for lvl in self._tracked_levels)
        if needs_ib_relative and "dt" in ohlcv.columns:
            if "minute_of_day" not in ohlcv.columns:
                dt = pd.to_datetime(ohlcv["dt"], errors="coerce")
                if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                    dt_local = dt.dt.tz_convert("America/Los_Angeles")
                else:
                    dt_local = dt
                minute_of_day = (dt_local.dt.hour * 60 + dt_local.dt.minute).astype(np.int32)
                ohlcv["minute_of_day"] = minute_of_day
            else:
                minute_of_day = ohlcv["minute_of_day"]
            ib_lo = ohlcv["ib_lo"].to_numpy(dtype=np.float64, copy=False)
            ib_hi = ohlcv["ib_hi"].to_numpy(dtype=np.float64, copy=False)
            mod = minute_of_day.to_numpy(dtype=np.int32, copy=False)
            valid_ib = (
                (mod >= 450)
                & np.isfinite(ib_lo)
                & np.isfinite(ib_hi)
                & (ib_hi > ib_lo)
            )
            ib_range = ib_hi - ib_lo
            if "ib_mid" in self._tracked_levels:
                ohlcv["ib_mid"] = np.where(valid_ib, (ib_hi + ib_lo) / 2.0, np.nan)
            if "ib_hi_ext_025" in self._tracked_levels:
                ohlcv["ib_hi_ext_025"] = np.where(valid_ib, ib_hi + 0.25 * ib_range, np.nan)
            if "ib_hi_ext_050" in self._tracked_levels:
                ohlcv["ib_hi_ext_050"] = np.where(valid_ib, ib_hi + 0.50 * ib_range, np.nan)
            if "ib_hi_ext_100" in self._tracked_levels:
                ohlcv["ib_hi_ext_100"] = np.where(valid_ib, ib_hi + 1.00 * ib_range, np.nan)
            if "ib_lo_ext_025" in self._tracked_levels:
                ohlcv["ib_lo_ext_025"] = np.where(valid_ib, ib_lo - 0.25 * ib_range, np.nan)
            if "ib_lo_ext_050" in self._tracked_levels:
                ohlcv["ib_lo_ext_050"] = np.where(valid_ib, ib_lo - 0.50 * ib_range, np.nan)
            if "ib_lo_ext_100" in self._tracked_levels:
                ohlcv["ib_lo_ext_100"] = np.where(valid_ib, ib_lo - 1.00 * ib_range, np.nan)
        return ohlcv

    def _compute_all_providers(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Run all feature providers (mirrors train_level_models.compute_all_features)."""
        from strategies.features.higher_timeframe import HigherTimeframeProvider
        from strategies.features.intraday_regime import (
            INTRADAY_REGIME_FEATURES,
            compute_intraday_regime_features,
        )
        from strategies.features.volume_microstructure import VolumeMicrostructureProvider
        from strategies.features.reversion_quality import ReversionQualityProvider
        from strategies.features.temporal_interactions import TemporalInteractionProvider

        feature_frames = []

        htf = HigherTimeframeProvider()
        htf_df = htf._compute_impl(ohlcv)
        htf_cols = [col for col in htf.feature_names if col in htf_df.columns]
        if htf_cols:
            feature_frames.append(htf_df[htf_cols].copy())

        has_bidask = 'bidvolume' in ohlcv.columns
        vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
        vol_df = vmp._compute_impl(ohlcv)
        vol_cols = [col for col in vmp.feature_names if col in vol_df.columns]
        if vol_cols:
            feature_frames.append(vol_df[vol_cols].copy())

        rqp = ReversionQualityProvider(same_day_bidask_only=self._same_day_bidask_only)
        qual_df = rqp._compute_impl(ohlcv)
        qual_cols = [col for col in rqp.feature_names if col in qual_df.columns]
        if qual_cols:
            feature_frames.append(qual_df[qual_cols].copy())

        tip = TemporalInteractionProvider()
        temp_df = tip._compute_impl(ohlcv)
        temp_cols = [col for col in tip.feature_names if col in temp_df.columns]
        if temp_cols:
            feature_frames.append(temp_df[temp_cols].copy())

        # Parity with training: RTH running-level stability features.
        if "rth_lo" in ohlcv.columns and "rth_hi" in ohlcv.columns:
            stability = pd.DataFrame(index=ohlcv.index)
            stability["rth_lo_age_bars"] = 0.0
            stability["rth_hi_age_bars"] = 0.0
            stability["rth_lo_updates_30"] = 0.0
            stability["rth_hi_updates_30"] = 0.0
            stability["rth_level_firmness"] = 0.0

            def _age_since_change(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
                out = np.zeros_like(vals, dtype=np.float32)
                last_change = -1
                prev = np.nan
                for i in range(len(vals)):
                    if not valid[i] or not np.isfinite(vals[i]):
                        out[i] = 0.0
                        continue
                    cur = float(vals[i])
                    if not np.isfinite(prev) or abs(cur - prev) > 1e-9:
                        last_change = i
                    out[i] = float(i - last_change) if last_change >= 0 else 0.0
                    prev = cur
                return out

            for _, g in ohlcv.groupby("trading_day", sort=False):
                gi = g.index
                lo = ohlcv.loc[gi, "rth_lo"].to_numpy(dtype=np.float64, copy=False)
                hi = ohlcv.loc[gi, "rth_hi"].to_numpy(dtype=np.float64, copy=False)
                valid = np.isfinite(lo) & np.isfinite(hi) & (lo != 0.0) & (hi != 0.0)
                if not np.any(valid):
                    continue

                lo_chg = np.zeros(len(gi), dtype=np.float32)
                hi_chg = np.zeros(len(gi), dtype=np.float32)
                lo_chg[1:] = (
                    valid[1:]
                    & valid[:-1]
                    & (np.abs(lo[1:] - lo[:-1]) > 1e-9)
                ).astype(np.float32)
                hi_chg[1:] = (
                    valid[1:]
                    & valid[:-1]
                    & (np.abs(hi[1:] - hi[:-1]) > 1e-9)
                ).astype(np.float32)

                lo_updates_30 = pd.Series(lo_chg).rolling(30, min_periods=1).sum().to_numpy(dtype=np.float32)
                hi_updates_30 = pd.Series(hi_chg).rolling(30, min_periods=1).sum().to_numpy(dtype=np.float32)
                lo_age = _age_since_change(lo, valid)
                hi_age = _age_since_change(hi, valid)
                firmness = 1.0 / (1.0 + lo_updates_30 + hi_updates_30)

                stability.loc[gi, "rth_lo_age_bars"] = lo_age
                stability.loc[gi, "rth_hi_age_bars"] = hi_age
                stability.loc[gi, "rth_lo_updates_30"] = lo_updates_30
                stability.loc[gi, "rth_hi_updates_30"] = hi_updates_30
                stability.loc[gi, "rth_level_firmness"] = firmness

            feature_frames.append(stability)

        if any(col in self._score_feature_cols_required for col in INTRADAY_REGIME_FEATURES):
            regime_df = compute_intraday_regime_features(ohlcv)
            regime_cols = [col for col in INTRADAY_REGIME_FEATURES if col in regime_df.columns]
            if regime_cols:
                feature_frames.append(regime_df[regime_cols].copy())

        if feature_frames:
            ohlcv = pd.concat([ohlcv] + feature_frames, axis=1)
            ohlcv = ohlcv.loc[:, ~ohlcv.columns.duplicated(keep='last')]

        return ohlcv

    def _find_nearest_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find nearest tracked level for each bar (mirrors label_reversals_breakouts logic)."""
        close_arr = df['close'].values.astype(np.float64)

        level_arrs = {}
        for lvl_name in self._tracked_levels:
            if lvl_name in df.columns:
                level_arrs[lvl_name] = df[lvl_name].values.astype(np.float64)

        nearest_level_name, nearest_level_price = compute_nearest_level_arrays(
            close_arr,
            level_arrs,
            proximity_pts=float(self._proximity_pts),
        )

        df['nearest_level_name'] = nearest_level_name
        df['nearest_level_price'] = nearest_level_price
        side_arr = assign_level_side(
            close_arr,
            nearest_level_name,
            nearest_level_price,
            side_hysteresis_pts=float(self._side_hysteresis_pts),
            side_flip_confirm_pts=(
                float(self._side_flip_confirm_pts) if self._side_flip_confirm_pts is not None else None
            ),
            trading_day=df['trading_day'].values if 'trading_day' in df.columns else None,
        )
        df['side'] = side_arr

        return df

    def _compute_level_encoding(self, current_df: pd.DataFrame,
                                 full_df: pd.DataFrame) -> pd.DataFrame:
        """Compute level-encoding features (mirrors compute_level_encoding_features).

        Uses full_df for trailing reversal rates (needs historical outcomes).
        current_df gets the features added.
        """
        nearest = current_df['nearest_level_name'].values

        # 1. One-hot encoding
        for lvl in self._tracked_levels:
            current_df[f'is_{lvl}'] = (nearest == lvl).astype(np.float32)

        # 2. level_is_support
        current_df['level_is_support'] = (current_df['side'] == 1).astype(np.float32)
        # Runtime parity with offline contract rows: trade_direction matches side.
        current_df['trade_direction'] = pd.to_numeric(
            current_df['side'], errors='coerce'
        ).fillna(0.0).astype(np.float32)

        # 3. Trailing reversal rates — set to 0.5 (neutral) since we don't have
        # outcomes during realtime. In playback mode, these come from historical data.
        current_df['level_trailing_rev_rate_20d'] = 0.5
        current_df['level_trailing_rev_rate_50d'] = 0.5

        # If full_df has outcome data (playback mode), compute actual rates
        if 'outcome' in full_df.columns:
            self._compute_trailing_rates(current_df, full_df)

        # 4. BB interaction
        if 'daily_bb_pct_b' in current_df.columns:
            current_df['level_side_bb_interaction'] = (
                current_df['level_is_support'] * current_df['daily_bb_pct_b']
            )
        else:
            current_df['level_side_bb_interaction'] = 0.0

        # 5. Approach direction
        close_vals = current_df['close'].values.astype(np.float64)
        lvl_price_vals = current_df['nearest_level_price'].values.astype(np.float64)
        n = len(current_df)
        approach_dir = np.zeros(n, dtype=np.float32)
        for i in range(5, n):
            if np.isnan(lvl_price_vals[i]):
                continue
            price_change = close_vals[i] - close_vals[i - 5]
            if close_vals[i] >= lvl_price_vals[i]:
                approach_dir[i] = -1.0 if price_change < 0 else 1.0
            else:
                approach_dir[i] = 1.0 if price_change > 0 else -1.0
        current_df['approach_dir_vs_level'] = approach_dir

        # 6. Intraday episode/reclaim-state features (current-close causal)
        from strategies.features.episode_state import compute_episode_state_features

        ep_df = compute_episode_state_features(
            current_df,
            level_col='nearest_level_name',
            side_col='side',
            trading_day_col='trading_day',
            density_window_bars=30,
        )
        for col in ep_df.columns:
            current_df[col] = ep_df[col].values.astype(np.float32)

        needed_dir_cols = {
            "same_level_side_run_bars",
            "bars_since_same_level_side_flip",
            "same_level_side_flip_count_10",
            "same_level_cross_count_10",
            "same_level_side_stability_10",
            "aligned_return_frac_5",
        }
        if self._score_feature_cols_required & needed_dir_cols:
            current_df = self._append_direction_stability_features(current_df)

        return current_df

    def _append_direction_stability_features(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """Add causal same-level direction-stability features used by ranking heads."""
        feature_cols = [
            "same_level_side_run_bars",
            "bars_since_same_level_side_flip",
            "same_level_side_flip_count_10",
            "same_level_cross_count_10",
            "same_level_side_stability_10",
            "aligned_return_frac_5",
        ]
        for col in feature_cols:
            if col not in current_df.columns:
                current_df[col] = 0.0

        if current_df.empty:
            return current_df

        grouped = current_df.groupby(
            ["trading_day", "nearest_level_name"],
            sort=False,
            dropna=False,
        )
        for _, g in grouped:
            idx = g.index.to_numpy(dtype=np.int64, copy=False)
            if idx.size == 0:
                continue

            sides = (
                pd.Series(current_df.loc[idx, "side"])
                .fillna(0)
                .to_numpy(dtype=np.int8, copy=False)
            )
            close = pd.to_numeric(
                current_df.loc[idx, "close"],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=False)
            level = pd.to_numeric(
                current_df.loc[idx, "nearest_level_price"],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(level) & (sides != 0)

            signed_dist = close - level
            dist_sign = np.sign(signed_dist)
            flip_events = np.zeros(idx.size, dtype=np.float32)
            cross_events = np.zeros(idx.size, dtype=np.float32)
            run_bars = np.zeros(idx.size, dtype=np.float32)
            bars_since_flip = np.zeros(idx.size, dtype=np.float32)

            last_flip = -1
            run = 0
            prev_side = 0
            prev_sign = 0.0
            for j in range(idx.size):
                if not valid[j]:
                    run = 0
                    prev_side = 0
                    prev_sign = 0.0
                    continue

                cur_side = int(sides[j])
                cur_sign = float(dist_sign[j]) if np.isfinite(dist_sign[j]) else 0.0
                if prev_side != 0 and cur_side != prev_side:
                    flip_events[j] = 1.0
                    last_flip = j
                    run = 1
                else:
                    run = run + 1 if run > 0 else 1

                if prev_sign != 0.0 and cur_sign != 0.0 and cur_sign != prev_sign:
                    cross_events[j] = 1.0

                run_bars[j] = float(run)
                bars_since_flip[j] = (
                    float(j - last_flip) if last_flip >= 0 else run_bars[j]
                )

                prev_side = cur_side
                prev_sign = cur_sign if cur_sign != 0.0 else prev_sign

            flip_10 = pd.Series(flip_events).rolling(10, min_periods=1).sum().to_numpy(dtype=np.float32)
            cross_10 = pd.Series(cross_events).rolling(10, min_periods=1).sum().to_numpy(dtype=np.float32)
            close_diff = np.diff(close, prepend=close[0]).astype(np.float32)
            aligned_good = np.zeros(idx.size, dtype=np.float32)
            for j in range(idx.size):
                if sides[j] == 0:
                    continue
                aligned_good[j] = 1.0 if close_diff[j] * float(sides[j]) > 0.0 else 0.0
            aligned_frac_5 = (
                pd.Series(aligned_good).rolling(5, min_periods=1).mean().to_numpy(dtype=np.float32)
            )
            stability = (1.0 / (1.0 + flip_10 + cross_10)).astype(np.float32)

            current_df.loc[idx, "same_level_side_run_bars"] = run_bars
            current_df.loc[idx, "bars_since_same_level_side_flip"] = bars_since_flip
            current_df.loc[idx, "same_level_side_flip_count_10"] = flip_10
            current_df.loc[idx, "same_level_cross_count_10"] = cross_10
            current_df.loc[idx, "same_level_side_stability_10"] = stability
            current_df.loc[idx, "aligned_return_frac_5"] = aligned_frac_5

        return current_df

    def _compute_trailing_rates(self, current_df: pd.DataFrame,
                                 full_df: pd.DataFrame) -> None:
        """Compute trailing reversal rates from historical outcomes."""
        if 'outcome' not in full_df.columns or 'trading_day' not in full_df.columns:
            return

        near_mask = full_df['outcome'].isin([0, 1])
        if not near_mask.any():
            return

        near_df = full_df.loc[near_mask, ['trading_day', 'nearest_level_name', 'outcome']].copy()
        days = sorted(full_df['trading_day'].unique())

        daily_stats = near_df.groupby(['trading_day', 'nearest_level_name']).agg(
            n_rev=('outcome', lambda x: (x == 1).sum()),
            n_total=('outcome', 'count'),
        ).reset_index()

        for lvl in self._tracked_levels:
            lvl_stats = daily_stats[daily_stats['nearest_level_name'] == lvl].copy()
            lvl_stats = lvl_stats.set_index('trading_day').reindex(days).fillna(0)

            lvl_stats['rev_20d'] = lvl_stats['n_rev'].rolling(20, min_periods=1).sum()
            lvl_stats['total_20d'] = lvl_stats['n_total'].rolling(20, min_periods=1).sum()
            lvl_stats['rev_rate_20d'] = lvl_stats['rev_20d'] / lvl_stats['total_20d'].clip(lower=1)

            lvl_stats['rev_50d'] = lvl_stats['n_rev'].rolling(50, min_periods=1).sum()
            lvl_stats['total_50d'] = lvl_stats['n_total'].rolling(50, min_periods=1).sum()
            lvl_stats['rev_rate_50d'] = lvl_stats['rev_50d'] / lvl_stats['total_50d'].clip(lower=1)

            # Shift by 1 for causality
            lvl_stats['rev_rate_20d'] = lvl_stats['rev_rate_20d'].shift(1)
            lvl_stats['rev_rate_50d'] = lvl_stats['rev_rate_50d'].shift(1)

            rate_20d_map = lvl_stats['rev_rate_20d'].to_dict()
            rate_50d_map = lvl_stats['rev_rate_50d'].to_dict()

            lvl_mask = current_df['nearest_level_name'] == lvl
            if lvl_mask.any() and 'trading_day' in current_df.columns:
                mapped_20d = current_df.loc[lvl_mask, 'trading_day'].map(rate_20d_map)
                mapped_50d = current_df.loc[lvl_mask, 'trading_day'].map(rate_50d_map)
                current_df.loc[lvl_mask, 'level_trailing_rev_rate_20d'] = mapped_20d.fillna(0.5)
                current_df.loc[lvl_mask, 'level_trailing_rev_rate_50d'] = mapped_50d.fillna(0.5)

    # ------------------------------------------------------------------
    # Prediction and signal emission
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rth_ts(ts: pd.Timestamp) -> bool:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert('America/Los_Angeles')
        minute = int(t.hour) * 60 + int(t.minute)
        return 390 <= minute < 780

    def _session_key(self, row: pd.Series) -> str:
        if 'dt' in row.index and pd.notna(row['dt']):
            return "rth" if self._is_rth_ts(pd.Timestamp(row['dt'])) else "ovn"
        return "ovn"

    def _session_threshold(self, session_key: str) -> float:
        return float(self._pred_threshold_rth if session_key == "rth" else self._pred_threshold_ovn)

    @staticmethod
    def _minute_of_day(ts: pd.Timestamp) -> int:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert('America/Los_Angeles')
        return int(t.hour) * 60 + int(t.minute)

    def _lane_high_threshold(self, session_key: str) -> float:
        if session_key == "rth" and self._high_score_threshold_rth is not None:
            return float(self._high_score_threshold_rth)
        if session_key == "ovn" and self._high_score_threshold_ovn is not None:
            return float(self._high_score_threshold_ovn)
        if self._high_score_threshold is not None:
            return float(self._high_score_threshold)
        return float(self._session_threshold(session_key))

    def _lane_low_threshold(self, session_key: str) -> Optional[float]:
        if session_key == "rth" and self._low_score_threshold_rth is not None:
            return float(self._low_score_threshold_rth)
        if session_key == "ovn" and self._low_score_threshold_ovn is not None:
            return float(self._low_score_threshold_ovn)
        if self._low_score_threshold is not None:
            return float(self._low_score_threshold)
        return None

    def _policy_session_threshold(self, session_key: str) -> float:
        if session_key == "rth":
            if self._policy_threshold_rth is not None:
                return float(self._policy_threshold_rth)
        else:
            if self._policy_threshold_ovn is not None:
                return float(self._policy_threshold_ovn)
        return float(self._policy_threshold if self._policy_threshold is not None else 0.5)

    def _policy_predict_score(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
        bar_num: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        if self._policy_model is None:
            return None, None
        minutes_since_prev_episode = 9999.0
        if "dt" in row.index and pd.notna(row["dt"]) and self._policy_last_episode_dt is not None:
            ts = pd.Timestamp(row["dt"])
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/Los_Angeles")
            minutes_since_prev_episode = max(
                float((ts - self._policy_last_episode_dt).total_seconds() / 60.0),
                0.0,
            )
        policy_x = self._build_policy_feature_row(
            row=row,
            base_prob=base_prob,
            session_key=session_key,
            episode_index_so_far=self._policy_episodes_started_today,
            minutes_since_prev_episode=minutes_since_prev_episode,
        )
        if policy_x is None:
            return None, None

        policy_prob = float(self._policy_model.predict_proba(policy_x)[0, 1])
        policy_std: Optional[float] = None
        if self._policy_ensemble_models:
            ens_preds = np.asarray(
                [m.predict_proba(policy_x)[0, 1] for m in self._policy_ensemble_models],
                dtype=np.float64,
            )
            if ens_preds.size:
                ens_mean = float(ens_preds.mean())
                policy_std = float(ens_preds.std(ddof=0))
                if self._policy_use_ensemble_mean:
                    policy_prob = ens_mean
        return policy_prob, policy_std

    def _load_frontier_quality_models(self, model_dir: str) -> None:
        """Load two-head frontier quality models (tp + fail risk)."""
        md_path = os.path.join(model_dir, "metadata.json")
        tp_path = os.path.join(model_dir, "tp_model.json")
        fail_path = os.path.join(model_dir, "fail_model.json")
        if not (os.path.exists(md_path) and os.path.exists(tp_path) and os.path.exists(fail_path)):
            logger.warning(
                "Frontier q_twohead model artifacts missing in %s; expected metadata.json/tp_model.json/fail_model.json",
                model_dir,
            )
            return
        try:
            with open(md_path) as f:
                md = json.load(f)
            cols = md.get("feature_cols", [])
            if not isinstance(cols, list) or not cols:
                logger.warning("Frontier q_twohead metadata has empty feature_cols: %s", md_path)
                return
            tp_model = xgb.XGBClassifier()
            fail_model = xgb.XGBClassifier()
            tp_model.load_model(tp_path)
            fail_model.load_model(fail_path)
            self._frontier_qtp_model = tp_model
            self._frontier_qfail_model = fail_model
            self._frontier_q_feature_cols = [str(c) for c in cols]
            logger.info(
                "Loaded frontier q_twohead models from %s (n_features=%d)",
                model_dir,
                len(self._frontier_q_feature_cols),
            )
        except Exception:
            logger.exception("Failed loading frontier q_twohead models from %s", model_dir)

    def _load_frontier_quality_lookup(self, parquet_path: str) -> None:
        """Load optional per-episode quality lookup for parity/offline replay.

        Expects columns:
        - trading_day (str)
        - episode_start_idx (int)
        - q_twohead (float)
        """
        try:
            df = pd.read_parquet(parquet_path).copy()
        except Exception:
            logger.exception("Failed reading frontier q_lookup parquet: %s", parquet_path)
            return
        req = {"trading_day", "episode_start_idx", "q_twohead"}
        if not req.issubset(set(df.columns)):
            logger.warning(
                "Frontier q_lookup parquet missing required columns %s: %s",
                sorted(req),
                parquet_path,
            )
            return
        td = df["trading_day"].astype(str)
        ep = pd.to_numeric(df["episode_start_idx"], errors="coerce").fillna(-1).astype(np.int64)
        q = pd.to_numeric(df["q_twohead"], errors="coerce")
        lookup: Dict[Tuple[str, int], float] = {}
        for day, epi, qq in zip(td.to_numpy(), ep.to_numpy(), q.to_numpy()):
            if int(epi) < 0 or not np.isfinite(qq):
                continue
            lookup[(str(day), int(epi))] = float(qq)
        self._frontier_q_lookup = lookup
        logger.info(
            "Loaded frontier q_lookup from %s (keys=%d)",
            parquet_path,
            len(self._frontier_q_lookup),
        )

    def _frontier_predict_qtwohead(
        self,
        row: pd.Series,
        event_prob: Optional[float] = None,
    ) -> Optional[float]:
        if (
            self._frontier_qtp_model is None
            or self._frontier_qfail_model is None
            or not self._frontier_q_feature_cols
        ):
            return None
        data: Dict[str, float] = {}
        for col in self._frontier_q_feature_cols:
            if col == "event_base_prob" and event_prob is not None and np.isfinite(float(event_prob)):
                data[col] = float(event_prob)
                continue
            if col not in row.index or pd.isna(row[col]):
                data[col] = 0.0
                continue
            try:
                vv = float(row[col])
                if np.isfinite(vv):
                    data[col] = float(vv)
                    continue
            except Exception:
                pass
            vv2 = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            data[col] = float(vv2) if pd.notna(vv2) else 0.0
        x = np.asarray([data[c] for c in self._frontier_q_feature_cols], dtype=np.float32).reshape(1, -1)
        tp = float(self._frontier_qtp_model.predict_proba(x)[0, 1])
        fail = float(self._frontier_qfail_model.predict_proba(x)[0, 1])
        return float(np.clip(tp * (1.0 - fail), 0.0, 1.0))

    def _frontier_predict_qlookup(self, row: pd.Series) -> Optional[float]:
        if not self._frontier_q_lookup:
            return None
        day = None
        epi = None
        if "trading_day" in row.index and pd.notna(row["trading_day"]):
            day = str(row["trading_day"])
        if "episode_start_idx" in row.index and pd.notna(row["episode_start_idx"]):
            try:
                epi = int(row["episode_start_idx"])
            except Exception:
                epi = None
        if day is None or epi is None or epi < 0:
            return None
        return self._frontier_q_lookup.get((day, epi))

    def _frontier_update_virtual_outcomes(
        self,
        *,
        bar_num: int,
        bar_high: float,
        bar_low: float,
    ) -> None:
        if not self._frontier_virtual_open_trades:
            return
        bn = int(bar_num)
        hi = float(bar_high)
        lo = float(bar_low)
        for tr in self._frontier_virtual_open_trades:
            if bool(tr.get("closed", False)):
                continue
            entry_bar = int(tr.get("entry_bar", -1))
            if bn <= entry_bar:
                continue
            direction = int(tr.get("direction", 0))
            stop_px = float(tr.get("stop_px", np.nan))
            target_px = float(tr.get("target_px", np.nan))
            if not np.isfinite(stop_px) or not np.isfinite(target_px) or direction == 0:
                continue
            hit = None
            if direction > 0:
                if lo <= stop_px and hi >= target_px:
                    hit = "stop_tie"
                elif lo <= stop_px:
                    hit = "stop"
                elif hi >= target_px:
                    hit = "target"
            else:
                if hi >= stop_px and lo <= target_px:
                    hit = "stop_tie"
                elif hi >= stop_px:
                    hit = "stop"
                elif lo <= target_px:
                    hit = "target"
            if hit is None:
                continue
            tr["closed"] = True
            tr["exit_bar"] = bn
            tr["exit_reason"] = hit
            ep = float(tr.get("entry_price", np.nan))
            pnl = 0.0
            if np.isfinite(ep):
                if direction > 0:
                    exit_px = (
                        float(stop_px)
                        if str(hit).startswith("stop")
                        else float(target_px) if hit == "target" else float(ep)
                    )
                else:
                    exit_px = (
                        float(stop_px)
                        if str(hit).startswith("stop")
                        else float(target_px) if hit == "target" else float(ep)
                    )
                pnl = float((exit_px - ep) * float(direction))
            self._frontier_apply_realized_outcome(pnl=float(pnl), reason=str(hit))
            seq = self._frontier_seq_position
            if seq is not None:
                same_dir = int(seq.get("direction", 0)) == int(direction)
                same_entry = int(seq.get("entry_bar", -1)) == int(entry_bar)
                if same_dir and same_entry:
                    self._frontier_seq_position = None

    def _frontier_apply_realized_outcome(
        self,
        *,
        pnl: float,
        reason: str,
    ) -> None:
        rs = str(reason)
        if rs.startswith("stop"):
            self._frontier_realized_losses_today += 1
            return
        if rs == "target":
            self._frontier_realized_wins_today += 1
            return
        if float(pnl) > 0.0:
            self._frontier_realized_wins_today += 1
        elif float(pnl) < 0.0:
            self._frontier_realized_losses_today += 1

    def _frontier_early_loss_lock_reject_reason(
        self,
        *,
        minute_of_day: int,
    ) -> Optional[str]:
        if not self._frontier_early_loss_lock_enabled:
            return None
        m = int(minute_of_day)
        # Active lock window always blocks new accepts.
        if self._frontier_early_loss_lock_until_minute >= m:
            return "router_early_loss_lock"
        # Optionally allow only one trigger per day.
        if (
            self._frontier_early_loss_lock_once_per_day
            and self._frontier_early_loss_lock_triggered_today
        ):
            return None
        if (
            m < int(self._frontier_early_loss_lock_start_minute)
            or m > int(self._frontier_early_loss_lock_end_minute)
        ):
            return None
        diff = int(self._frontier_realized_losses_today) - int(self._frontier_realized_wins_today)
        if diff < int(self._frontier_early_loss_lock_trigger_diff):
            return None
        dur = int(self._frontier_early_loss_lock_duration_min)
        if dur <= 0:
            return None
        self._frontier_early_loss_lock_until_minute = min(int(m) + int(dur), 779)
        self._frontier_early_loss_lock_triggered_today = True
        return "router_early_loss_lock"

    def _frontier_decide_sequential_action(
        self,
        *,
        direction: int,
        quality_score: float,
        open_unrealized: float,
        flatten_min_q_override: Optional[float] = None,
    ) -> str:
        if self._frontier_execution_mode != "sequential":
            return "entry"
        pos = self._frontier_seq_position
        if pos is None:
            return "entry"
        if int(direction) == int(pos.get("direction", 0)):
            return "ignore"

        oa = str(self._frontier_opposite_action)
        if oa == "ignore":
            return "ignore"
        if oa == "flatten":
            flatten_min_q = (
                float(flatten_min_q_override)
                if flatten_min_q_override is not None
                else float(self._frontier_opposite_flatten_min_q)
            )
            return (
                "flatten"
                if float(quality_score) >= float(flatten_min_q)
                else "ignore"
            )

        flip_ok = True
        if float(quality_score) < float(self._frontier_opposite_flip_min_q):
            flip_ok = False
        pos_q = float(pos.get("quality_at_entry", 0.0))
        if float(quality_score) - float(pos_q) < float(self._frontier_opposite_min_q_gap_vs_open):
            flip_ok = False
        if self._frontier_opposite_flip_requires_open_loss and float(open_unrealized) > 0.0:
            flip_ok = False

        if oa == "flip":
            return "flip" if flip_ok else "ignore"
        if oa == "flip_or_flatten":
            flatten_min_q = (
                float(flatten_min_q_override)
                if flatten_min_q_override is not None
                else float(self._frontier_opposite_flatten_min_q)
            )
            if flip_ok:
                return "flip"
            return (
                "flatten"
                if float(quality_score) >= float(flatten_min_q)
                else "ignore"
            )
        return "ignore"

    def _frontier_close_sequential_position(
        self,
        *,
        exit_price: float,
        reason: str,
    ) -> Optional[float]:
        pos = self._frontier_seq_position
        if pos is None:
            return None
        direction = int(pos.get("direction", 0))
        entry_price = float(pos.get("entry_price", np.nan))
        entry_bar = int(pos.get("entry_bar", -1))
        if direction == 0 or not np.isfinite(entry_price):
            self._frontier_seq_position = None
            return None
        pnl = float((float(exit_price) - float(entry_price)) * float(direction))
        self._frontier_apply_realized_outcome(pnl=float(pnl), reason=str(reason))
        for tr in self._frontier_virtual_open_trades:
            if bool(tr.get("closed", False)):
                continue
            if int(tr.get("direction", 0)) != int(direction):
                continue
            if int(tr.get("entry_bar", -1)) != int(entry_bar):
                continue
            tr["closed"] = True
            tr["exit_reason"] = str(reason)
        self._frontier_seq_position = None
        return float(pnl)

    def _frontier_has_unresolved_trade(
        self,
        *,
        level_name: str,
        direction: int,
    ) -> bool:
        key = f"{str(level_name)}:{int(direction)}"
        for tr in self._frontier_virtual_open_trades:
            if bool(tr.get("closed", False)):
                continue
            if str(tr.get("level_key", "")) == key:
                return True
        return False

    @staticmethod
    def _frontier_level_group(level_name: str) -> str:
        return level_group(level_name)

    def _frontier_inertia_override_ok(
        self,
        *,
        quality_score: float,
        last_quality: float,
    ) -> bool:
        has_override_cfg = (
            self._frontier_inertia_override_min_q is not None
            or self._frontier_inertia_override_min_q_gap > 0.0
        )
        if not has_override_cfg:
            return False
        if (
            self._frontier_inertia_override_min_q is not None
            and float(quality_score) < float(self._frontier_inertia_override_min_q)
        ):
            return False
        if self._frontier_inertia_override_min_q_gap > 0.0 and (
            float(quality_score) - float(last_quality)
        ) < float(self._frontier_inertia_override_min_q_gap):
            return False
        return True

    def _frontier_inertia_reject_reason(
        self,
        *,
        minute_of_day: int,
        direction: int,
        level_name: str,
        quality_score: float,
    ) -> Optional[str]:
        if not self._frontier_inertia_enabled:
            return None

        m = int(minute_of_day)
        d = int(direction)
        if d == 0:
            return None

        if self._frontier_inertia_level_group_minute_gap > 0:
            group = self._frontier_level_group(level_name)
            last_group = self._frontier_inertia_last_by_group.get(str(group))
            if last_group is not None and int(last_group.get("direction", 0)) != d:
                dt = m - int(last_group.get("minute", -10_000))
                if dt < int(self._frontier_inertia_level_group_minute_gap):
                    if not self._frontier_inertia_override_ok(
                        quality_score=float(quality_score),
                        last_quality=float(last_group.get("quality", 0.0)),
                    ):
                        return "router_inertia_group"

        if self._frontier_inertia_global_minute_gap > 0 and self._frontier_inertia_last_global is not None:
            last_global = self._frontier_inertia_last_global
            if int(last_global.get("direction", 0)) != d:
                dt = m - int(last_global.get("minute", -10_000))
                if dt < int(self._frontier_inertia_global_minute_gap):
                    if not self._frontier_inertia_override_ok(
                        quality_score=float(quality_score),
                        last_quality=float(last_global.get("quality", 0.0)),
                    ):
                        return "router_inertia_global"
        return None

    def _frontier_inertia_record_accept(
        self,
        *,
        minute_of_day: int,
        direction: int,
        level_name: str,
        quality_score: float,
    ) -> None:
        if not self._frontier_inertia_enabled:
            return
        rec = {
            "minute": float(int(minute_of_day)),
            "direction": float(int(direction)),
            "quality": float(quality_score),
        }
        self._frontier_inertia_last_global = dict(rec)
        group = self._frontier_level_group(level_name)
        self._frontier_inertia_last_by_group[str(group)] = dict(rec)

    def _frontier_register_virtual_trade(
        self,
        *,
        bar_num: int,
        entry_price: float,
        level_name: str,
        direction: int,
        quality_score: float,
    ) -> None:
        if not (
            self._frontier_dynamic_budget_enabled
            or self._frontier_group_unresolved_enabled
            or self._frontier_execution_mode == "sequential"
        ):
            return
        exec_lane = "high" if float(quality_score) >= float(self._frontier_virtual_gate) else "low"
        stop_pts = float(self._frontier_virtual_stop_hi if exec_lane == "high" else self._frontier_virtual_stop_lo)
        target_pts = float(self._frontier_virtual_target_hi if exec_lane == "high" else self._frontier_virtual_target_lo)
        if stop_pts <= 0.0 or target_pts <= 0.0:
            return
        ep = float(entry_price)
        if int(direction) > 0:
            stop_px = ep - stop_pts
            target_px = ep + target_pts
        else:
            stop_px = ep + stop_pts
            target_px = ep - target_pts
        self._frontier_virtual_open_trades.append(
            {
                "entry_bar": int(bar_num),
                "direction": int(direction),
                "level_key": f"{str(level_name)}:{int(direction)}",
                "entry_price": float(ep),
                "stop_px": float(stop_px),
                "target_px": float(target_px),
                "closed": False,
            }
        )

    def _reset_frontier_router_state(self) -> None:
        self._frontier_router_day = None
        self._frontier_router_accepted_today = 0
        self._frontier_router_q_used_today = 0
        self._frontier_router_c_used_today = 0
        self._frontier_router_r_used_today = 0
        self._frontier_router_override_used_today = 0
        self._frontier_router_override_window_counts.clear()
        self._frontier_realized_wins_today = 0
        self._frontier_realized_losses_today = 0
        self._frontier_virtual_open_trades.clear()
        self._frontier_seq_position = None
        self._frontier_router_last_accept_minute = -10_000
        self._frontier_router_side_counts.clear()
        self._frontier_inertia_last_global = None
        self._frontier_inertia_last_by_group.clear()
        self._frontier_blend_live_used_today = 0
        self._frontier_blend_q2_used_today = 0
        self._frontier_early_loss_lock_until_minute = -1
        self._frontier_early_loss_lock_triggered_today = False

    def _sync_frontier_router_day(self, row: pd.Series) -> None:
        day: Optional[str] = None
        if "trading_day" in row.index and pd.notna(row["trading_day"]):
            day = str(row["trading_day"])
        elif "dt" in row.index and pd.notna(row["dt"]):
            td = self._compute_trading_day(pd.Series([pd.Timestamp(row["dt"])]))
            if not td.empty:
                day = str(td.iloc[0])
        if not day:
            return
        if self._frontier_router_day != day:
            self._reset_frontier_router_state()
            self._frontier_router_day = day

    def _frontier_router_accept(
        self,
        *,
        row: pd.Series,
        minute_of_day: int,
        direction: int,
        level_name: str,
        quality_score: float,
        coverage_score: float,
        rescue_score: float,
        session_key: str,
        unresolved_block: Optional[bool] = None,
    ) -> Tuple[bool, Optional[str]]:
        self._sync_frontier_router_day(row)
        if session_key != "rth":
            return False, "router_session"
        m = int(minute_of_day)
        if m < 390 or m >= 780:
            return False, "router_time"
        early_loss_lock_reason = self._frontier_early_loss_lock_reject_reason(
            minute_of_day=int(m)
        )
        if early_loss_lock_reason is not None:
            return False, str(early_loss_lock_reason)
        unresolved = bool(unresolved_block) if unresolved_block is not None else False
        if unresolved_block is None and self._frontier_group_unresolved_enabled:
            unresolved = self._frontier_has_unresolved_trade(
                level_name=str(level_name),
                direction=int(direction),
            )
        if unresolved:
            return False, "router_unresolved"
        if self._frontier_cooldown_min > 0 and (
            m - int(self._frontier_router_last_accept_minute)
        ) < int(self._frontier_cooldown_min):
            return False, "router_cooldown"
        key = f"{level_name}:{int(direction)}"
        if (
            self._frontier_diversity_cap > 0
            and self._frontier_router_side_counts.get(key, 0) >= int(self._frontier_diversity_cap)
        ):
            return False, "router_diversity"
        inertia_reason = self._frontier_inertia_reject_reason(
            minute_of_day=int(m),
            direction=int(direction),
            level_name=str(level_name),
            quality_score=float(quality_score),
        )
        if inertia_reason is not None:
            return False, str(inertia_reason)

        lane: Optional[str] = None
        if self._frontier_router_q_used_today < int(self._frontier_q_slots):
            if float(quality_score) >= float(self._frontier_q_thr):
                lane = "quality"
        if lane is None and self._frontier_router_c_used_today < int(self._frontier_c_slots):
            if m >= int(self._frontier_c_start) and float(coverage_score) >= float(self._frontier_c_thr):
                lane = "coverage"
        if lane is None and self._frontier_router_r_used_today < int(self._frontier_r_slots):
            if m >= int(self._frontier_r_start) and float(rescue_score) >= float(self._frontier_r_thr):
                lane = "rescue"
        if lane is None:
            return False, "router_threshold"

        eff_k_total = int(self._frontier_k_total)
        if self._frontier_dynamic_budget_enabled and m >= int(self._frontier_dynamic_start_minute):
            diff = max(
                int(self._frontier_realized_wins_today) - int(self._frontier_realized_losses_today),
                0,
            )
            extra = diff // max(int(self._frontier_dynamic_extra_every), 1)
            eff_k_total = int(self._frontier_k_total) + min(extra, int(self._frontier_dynamic_extra_cap))

        loss_lock_active = True
        if self._frontier_loss_lock_after_nominal_only:
            loss_lock_active = self._frontier_router_accepted_today >= int(self._frontier_k_total)
        loss_locked = (
            bool(loss_lock_active)
            and int(self._frontier_loss_lock_diff) > 0
            and (int(self._frontier_realized_losses_today) - int(self._frontier_realized_wins_today))
            >= int(self._frontier_loss_lock_diff)
        )
        within_budget = (
            self._frontier_router_accepted_today < int(eff_k_total)
            and (not loss_locked)
        )
        can_override = False
        override_bucket: Optional[int] = None
        if not within_budget:
            if (
                self._frontier_high_override_prob is None
                or m < int(self._frontier_high_override_start)
                or float(quality_score) < float(self._frontier_high_override_prob)
            ):
                return False, "router_budget"

            if loss_locked:
                return False, "router_loss_lock"

            if (
                int(self._frontier_high_override_cap) >= 0
                and self._frontier_router_override_used_today >= int(self._frontier_high_override_cap)
            ):
                return False, "router_override_cap"

            if (
                int(self._frontier_high_override_window_min) > 0
                and int(self._frontier_high_override_window_cap) > 0
            ):
                window_min = int(self._frontier_high_override_window_min)
                override_bucket = max(int((m - 390) // window_min), 0)
                used = int(self._frontier_router_override_window_counts.get(override_bucket, 0))
                if used >= int(self._frontier_high_override_window_cap):
                    return False, "router_override_window"
            can_override = True

        if not (within_budget or can_override):
            return False, "router_budget"

        if lane == "quality":
            self._frontier_router_q_used_today += 1
        elif lane == "coverage":
            self._frontier_router_c_used_today += 1
        elif lane == "rescue":
            self._frontier_router_r_used_today += 1
        self._frontier_router_accepted_today += 1
        if can_override:
            self._frontier_router_override_used_today += 1
            if override_bucket is not None:
                self._frontier_router_override_window_counts[override_bucket] = (
                    int(self._frontier_router_override_window_counts.get(override_bucket, 0)) + 1
                )
        self._frontier_router_last_accept_minute = m
        self._frontier_router_side_counts[key] = self._frontier_router_side_counts.get(key, 0) + 1
        self._frontier_inertia_record_accept(
            minute_of_day=int(m),
            direction=int(direction),
            level_name=str(level_name),
            quality_score=float(quality_score),
        )
        return True, lane

    def _frontier_blend_q2_accept(
        self,
        *,
        row: pd.Series,
        minute_of_day: int,
        direction: int,
        level_name: str,
        quality_score: float,
        session_key: str,
        unresolved_block: Optional[bool] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Supplemental q2 acceptance path for blend_policy_q2 quality source."""
        self._sync_frontier_router_day(row)
        if session_key != "rth":
            return False, "router_session"
        m = int(minute_of_day)
        if m < 390 or m >= 780:
            return False, "router_time"
        early_loss_lock_reason = self._frontier_early_loss_lock_reject_reason(
            minute_of_day=int(m)
        )
        if early_loss_lock_reason is not None:
            return False, str(early_loss_lock_reason)
        if m < int(self._frontier_blend_q2_start) or m >= int(self._frontier_blend_q2_end):
            return False, "router_q2_time"
        if float(quality_score) < float(self._frontier_blend_q2_min_q):
            return False, "router_q2_threshold"
        if self._frontier_blend_q2_cap > 0 and self._frontier_blend_q2_used_today >= int(self._frontier_blend_q2_cap):
            return False, "router_q2_cap"

        unresolved = bool(unresolved_block) if unresolved_block is not None else False
        if unresolved_block is None and self._frontier_group_unresolved_enabled:
            unresolved = self._frontier_has_unresolved_trade(
                level_name=str(level_name),
                direction=int(direction),
            )
        if unresolved:
            return False, "router_unresolved"
        if self._frontier_cooldown_min > 0 and (
            m - int(self._frontier_router_last_accept_minute)
        ) < int(self._frontier_cooldown_min):
            return False, "router_cooldown"

        key = f"{level_name}:{int(direction)}"
        if (
            self._frontier_diversity_cap > 0
            and self._frontier_router_side_counts.get(key, 0) >= int(self._frontier_diversity_cap)
        ):
            return False, "router_diversity"
        inertia_reason = self._frontier_inertia_reject_reason(
            minute_of_day=int(m),
            direction=int(direction),
            level_name=str(level_name),
            quality_score=float(quality_score),
        )
        if inertia_reason is not None:
            return False, str(inertia_reason)

        eff_k_total = int(self._frontier_k_total)
        if self._frontier_dynamic_budget_enabled and m >= int(self._frontier_dynamic_start_minute):
            diff = max(
                int(self._frontier_realized_wins_today) - int(self._frontier_realized_losses_today),
                0,
            )
            extra = diff // max(int(self._frontier_dynamic_extra_every), 1)
            eff_k_total = int(self._frontier_k_total) + min(extra, int(self._frontier_dynamic_extra_cap))

        loss_lock_active = True
        if self._frontier_loss_lock_after_nominal_only:
            loss_lock_active = self._frontier_router_accepted_today >= int(self._frontier_k_total)
        loss_locked = (
            bool(loss_lock_active)
            and int(self._frontier_loss_lock_diff) > 0
            and (int(self._frontier_realized_losses_today) - int(self._frontier_realized_wins_today))
            >= int(self._frontier_loss_lock_diff)
        )
        within_budget = (
            self._frontier_router_accepted_today < int(eff_k_total)
            and (not loss_locked)
        )
        can_override = False
        override_bucket: Optional[int] = None
        if not within_budget:
            override_prob = self._frontier_blend_q2_override_prob
            if override_prob is None:
                return False, "router_budget"
            if m < int(self._frontier_high_override_start) or float(quality_score) < float(override_prob):
                return False, "router_budget"
            if loss_locked:
                return False, "router_loss_lock"
            if (
                int(self._frontier_high_override_cap) >= 0
                and self._frontier_router_override_used_today >= int(self._frontier_high_override_cap)
            ):
                return False, "router_override_cap"
            if (
                int(self._frontier_high_override_window_min) > 0
                and int(self._frontier_high_override_window_cap) > 0
            ):
                window_min = int(self._frontier_high_override_window_min)
                override_bucket = max(int((m - 390) // window_min), 0)
                used = int(self._frontier_router_override_window_counts.get(override_bucket, 0))
                if used >= int(self._frontier_high_override_window_cap):
                    return False, "router_override_window"
            can_override = True

        if not (within_budget or can_override):
            return False, "router_budget"

        self._frontier_router_accepted_today += 1
        self._frontier_blend_q2_used_today += 1
        if can_override:
            self._frontier_router_override_used_today += 1
            if override_bucket is not None:
                self._frontier_router_override_window_counts[override_bucket] = (
                    int(self._frontier_router_override_window_counts.get(override_bucket, 0)) + 1
                )
        self._frontier_router_last_accept_minute = m
        self._frontier_router_side_counts[key] = self._frontier_router_side_counts.get(key, 0) + 1
        self._frontier_inertia_record_accept(
            minute_of_day=int(m),
            direction=int(direction),
            level_name=str(level_name),
            quality_score=float(quality_score),
        )
        return True, "q2"

    def _build_mfe_feature_row(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
    ) -> Optional[np.ndarray]:
        # Deprecated in runtime prune; base-prob scoring only.
        return None

    def _append_opening_drive_regime(self, current_df: pd.DataFrame) -> pd.DataFrame:
        out = current_df.copy()
        if out.empty or "dt" not in out.columns:
            out["opening_drive_regime_dir"] = 0
            out["opening_drive_net_pts"] = 0.0
            out["opening_drive_frac"] = 0.0
            return out

        dt = pd.to_datetime(out["dt"], errors="coerce")
        minute = dt.map(lambda ts: self._minute_of_day(ts) if pd.notna(ts) else -1).astype(np.int32)
        out["opening_drive_regime_dir"] = 0
        out["opening_drive_net_pts"] = 0.0
        out["opening_drive_frac"] = 0.0

        rth_open_mask = minute >= 390
        or_mask = (minute >= 390) & (minute < 420)
        if not or_mask.any():
            return out

        or_slice = out.loc[or_mask].sort_values("dt")
        open_px = float(or_slice["open"].iloc[0]) if "open" in or_slice.columns else float(or_slice["close"].iloc[0])
        close_30 = float(or_slice["close"].iloc[-1])
        high_30 = float(or_slice["high"].max())
        low_30 = float(or_slice["low"].min())
        net = float(close_30 - open_px)
        rng = max(float(high_30 - low_30), 0.25)
        drive_frac = float(abs(net) / rng)
        regime_dir = 0
        if abs(net) >= self._mfe_regime_drive_move_pts and drive_frac >= self._mfe_regime_drive_frac:
            pos_in_range = float((close_30 - low_30) / rng)
            if net > 0.0 and pos_in_range >= 0.7:
                regime_dir = 1
            elif net < 0.0 and pos_in_range <= 0.3:
                regime_dir = -1

        apply_mask = rth_open_mask & (minute >= 420)
        if apply_mask.any():
            out.loc[apply_mask, "opening_drive_regime_dir"] = regime_dir
            out.loc[apply_mask, "opening_drive_net_pts"] = net
            out.loc[apply_mask, "opening_drive_frac"] = drive_frac
        return out

    def _mfe_regime_score_allowed(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
    ) -> Tuple[bool, str]:
        # Deprecated in runtime prune; kept for diagnostics compatibility.
        return True, "disabled"

    def _compute_score(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
        large_prob: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        # Runtime prune keeps a single score contract to match selected live path.
        return float(base_prob), None

    def _accepted_count_for_score(self, session_key: str) -> int:
        if self._use_session_budget:
            return int(self._budget_accepted_by_session.get(session_key, 0))
        if self._dual_lane_enabled:
            return int(self._dual_taken_today)
        return int(self._budget_accepted_today)

    def _accept_under_budget(self, score: float, session_key: str, base_threshold: float) -> bool:
        """Causal intraday budget controller with optional session split."""
        if self._use_session_budget:
            budget = (
                int(self._trade_budget_rth_per_day)
                if session_key == "rth"
                else int(self._trade_budget_ovn_per_day)
            )
            if budget <= 0:
                return True

            self._budget_seen_by_session[session_key] += 1
            seen = float(self._budget_seen_by_session[session_key])
            expected = max(float(self._budget_expected_episodes_by_session[session_key]), 1.0)
            expected_so_far = float(budget) * min(1.0, seen / expected)
            gap = float(self._budget_accepted_by_session[session_key]) - expected_so_far

            dyn_thr = float(self._budget_dynamic_threshold_by_session[session_key])
            if gap > self._budget_adapt_tolerance:
                dyn_thr = min(self._budget_max_threshold, dyn_thr + self._budget_adapt_step)
            elif gap < -self._budget_adapt_tolerance:
                dyn_thr = max(base_threshold, dyn_thr - self._budget_adapt_step)
            self._budget_dynamic_threshold_by_session[session_key] = dyn_thr

            if self._budget_accepted_by_session[session_key] >= budget:
                return False
            if score < dyn_thr:
                return False
            self._budget_accepted_by_session[session_key] += 1
            return True

        if self._trade_budget_per_day <= 0:
            return True

        self._budget_seen_today += 1
        expected_so_far = float(self._trade_budget_per_day) * min(
            1.0,
            float(self._budget_seen_today) / max(self._budget_expected_episodes, 1.0),
        )
        gap = float(self._budget_accepted_today) - expected_so_far
        if gap > self._budget_adapt_tolerance:
            self._budget_dynamic_threshold = min(
                self._budget_max_threshold,
                self._budget_dynamic_threshold + self._budget_adapt_step,
            )
        elif gap < -self._budget_adapt_tolerance:
            self._budget_dynamic_threshold = max(
                base_threshold,
                self._budget_dynamic_threshold - self._budget_adapt_step,
            )

        if self._budget_accepted_today >= self._trade_budget_per_day:
            return False
        if score < self._budget_dynamic_threshold:
            return False

        self._budget_accepted_today += 1
        return True

    def _build_policy_feature_row(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
        episode_index_so_far: int = 0,
        minutes_since_prev_episode: float = 9999.0,
    ) -> Optional[np.ndarray]:
        if self._policy_model is None or not self._policy_feature_cols:
            return None

        data: Dict[str, float] = {}
        if 'dt' in row.index and pd.notna(row['dt']):
            ts = pd.Timestamp(row['dt'])
            if ts.tzinfo is not None:
                ts = ts.tz_convert('America/Los_Angeles')
            minute = int(ts.hour) * 60 + int(ts.minute)
        else:
            minute = 0

        data['base_prob'] = float(base_prob)
        data['is_rth_session'] = float(1.0 if session_key == "rth" else 0.0)
        data['minute_of_day_norm'] = float(minute / 1440.0)
        data['side'] = float(row.get('side', 0.0))
        data['episode_index_so_far'] = float(max(int(episode_index_so_far), 0))
        data['episode_index_log'] = float(np.log1p(max(int(episode_index_so_far), 0)))
        data['minutes_since_prev_episode'] = float(max(float(minutes_since_prev_episode), 0.0))
        data['minutes_since_prev_episode_log'] = float(
            np.log1p(max(float(minutes_since_prev_episode), 0.0))
        )
        data['is_opening_30m'] = float(1.0 if minute <= 420 else 0.0)

        for col in self._policy_feature_cols:
            if col in data:
                continue
            if col in row.index and pd.notna(row[col]):
                data[col] = float(row[col])
            else:
                data[col] = 0.0

        arr = np.asarray([data[c] for c in self._policy_feature_cols], dtype=np.float32)
        return arr.reshape(1, -1)

    def _policy_mark_episode_start(
        self,
        bar_num: int,
        row: pd.Series,
    ) -> None:
        self._policy_last_episode_bar = int(bar_num)
        self._policy_episodes_started_today += 1
        self._policy_last_episode_dt = None
        if "dt" in row.index and pd.notna(row["dt"]):
            ts = pd.Timestamp(row["dt"])
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/Los_Angeles")
            self._policy_last_episode_dt = ts

    def _evaluate_policy_gate(
        self,
        row: pd.Series,
        base_prob: float,
        session_key: str,
        bar_num: int,
    ) -> Tuple[bool, Optional[float], Optional[str], Optional[float]]:
        if self._policy_model is None:
            return True, None, None, None

        if self._policy_mode == "episode_start":
            prev_bar = self._policy_last_episode_bar
            if prev_bar is not None and (int(bar_num) - int(prev_bar)) <= self._policy_episode_gap_bars:
                return False, None, "policy_episode_window", None

        minutes_since_prev_episode = 9999.0
        if "dt" in row.index and pd.notna(row["dt"]) and self._policy_last_episode_dt is not None:
            ts = pd.Timestamp(row["dt"])
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/Los_Angeles")
            minutes_since_prev_episode = max(
                float((ts - self._policy_last_episode_dt).total_seconds() / 60.0),
                0.0,
            )

        policy_x = self._build_policy_feature_row(
            row=row,
            base_prob=base_prob,
            session_key=session_key,
            episode_index_so_far=self._policy_episodes_started_today,
            minutes_since_prev_episode=minutes_since_prev_episode,
        )
        if policy_x is None:
            return False, None, "policy_input", None

        policy_prob = float(self._policy_model.predict_proba(policy_x)[0, 1])
        policy_std: Optional[float] = None
        if self._policy_ensemble_models:
            ens_preds = np.asarray(
                [m.predict_proba(policy_x)[0, 1] for m in self._policy_ensemble_models],
                dtype=np.float64,
            )
            if ens_preds.size:
                ens_mean = float(ens_preds.mean())
                policy_std = float(ens_preds.std(ddof=0))
                if self._policy_use_ensemble_mean:
                    policy_prob = ens_mean

        if self._policy_mode == "episode_start":
            self._policy_mark_episode_start(bar_num=bar_num, row=row)

        if (
            self._policy_uncertainty_std_max is not None
            and policy_std is not None
            and policy_std > self._policy_uncertainty_std_max
        ):
            return False, policy_prob, "policy_uncertainty", policy_std

        if policy_prob < self._policy_session_threshold(session_key):
            return False, policy_prob, "policy_threshold", policy_std
        return True, policy_prob, None, policy_std

    def _predict_and_emit(self, df: pd.DataFrame,
                          bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        """Run XGBoost on near-level bars and emit signals above threshold."""
        near_mask = ~df['nearest_level_price'].isna()
        if not near_mask.any():
            return []

        near_df = df.loc[near_mask].copy()
        if '__bar_num' in near_df.columns:
            near_df = near_df.loc[near_df['__bar_num'] >= self._last_processed_count].copy()
            near_df = near_df.sort_values('__bar_num')
        else:
            near_df = near_df.sort_index()
        if near_df.empty:
            return []

        signals = []

        missing_cols = [c for c in self._feature_cols if c not in near_df.columns]
        if missing_cols:
            logger.debug("Missing %d feature cols (filling with 0): %s",
                         len(missing_cols), missing_cols[:5])

        # Build feature matrix with correct column ordering
        X_df = near_df.reindex(columns=self._feature_cols, fill_value=0.0)
        X = X_df.fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        # Predict
        y_prob = self._model.predict_proba(X)[:, 1]

        # Emit signals for high-probability predictions
        for i, (idx, row) in enumerate(near_df.iterrows()):
            base_prob = float(y_prob[i])
            session_key = self._session_key(row)
            base_threshold = self._session_threshold(session_key)
            if self._mfe_min_base_prob > 0.0 and base_prob < self._mfe_min_base_prob:
                continue
            if (
                (not self._frontier_router_enabled)
                and self._use_base_prob_prefilter
                and (not self._dual_lane_enabled)
                and base_prob < base_threshold
            ):
                continue

            level_name = str(row['nearest_level_name'])
            side = int(row['side'])
            if side == 0:
                continue
            direction = "bull" if side == 1 else "bear"
            direction_int = 1 if direction == "bull" else -1
            level_dir = (level_name, direction)
            bar_num = int(row['__bar_num']) if '__bar_num' in row.index else i
            minute_of_day = 0
            if "dt" in row.index and pd.notna(row["dt"]):
                minute_of_day = self._minute_of_day(pd.Timestamp(row["dt"]))

            policy_prob: Optional[float] = None
            policy_std: Optional[float] = None
            frontier_q2_score: Optional[float] = None
            frontier_quality_score = float(base_prob)
            blend_expert = "live"
            if (
                self._frontier_router_enabled
                and self._frontier_quality_source in {"policy_prob", "blend_policy_q2"}
            ):
                policy_prob, policy_std = self._policy_predict_score(
                    row=row,
                    base_prob=base_prob,
                    session_key=session_key,
                    bar_num=bar_num,
                )
                if policy_prob is not None:
                    frontier_quality_score = float(policy_prob)
                elif self._frontier_quality_source == "policy_prob" and (not self._frontier_router_warned_policy_missing):
                    logger.warning(
                        "Frontier router requested policy_prob quality source, but policy score unavailable; "
                        "falling back to base_prob quality score."
                    )
                    self._frontier_router_warned_policy_missing = True
            if self._frontier_router_enabled and self._frontier_quality_source in {"q_twohead", "blend_policy_q2"}:
                event_prob = float(base_prob)
                if "event_base_prob" in self._frontier_q_feature_cols and self._policy_model is not None:
                    pp = policy_prob
                    if pp is None:
                        pp, _ = self._policy_predict_score(
                            row=row,
                            base_prob=base_prob,
                            session_key=session_key,
                            bar_num=bar_num,
                        )
                    if pp is not None and np.isfinite(pp):
                        event_prob = float(pp)
                q_score = self._frontier_predict_qtwohead(row=row, event_prob=event_prob)
                frontier_q2_score = q_score
                if q_score is not None:
                    if self._frontier_quality_source == "q_twohead":
                        frontier_quality_score = float(q_score)
                elif self._frontier_quality_source == "q_twohead" and (not self._frontier_warned_quality_missing):
                    logger.warning(
                        "Frontier router requested q_twohead quality source, but quality model is unavailable; "
                        "falling back to base_prob quality score."
                    )
                    self._frontier_warned_quality_missing = True
            elif self._frontier_router_enabled and self._frontier_quality_source == "q_lookup":
                q_score = self._frontier_predict_qlookup(row=row)
                if q_score is not None:
                    frontier_quality_score = float(q_score)
                elif not self._frontier_warned_quality_missing:
                    logger.warning(
                        "Frontier router requested q_lookup quality source, but lookup value is unavailable; "
                        "falling back to base_prob quality score."
                    )
                    self._frontier_warned_quality_missing = True

            score_value = float(frontier_quality_score if self._frontier_router_enabled else base_prob)

            lane = "high"
            frontier_lane: Optional[str] = None
            lane_high_thr: Optional[float] = None
            lane_low_thr: Optional[float] = None
            sequential_action = "entry"
            open_unrealized = 0.0
            if self._frontier_router_enabled:
                if (
                    self._frontier_dynamic_budget_enabled
                    or self._frontier_group_unresolved_enabled
                    or self._frontier_execution_mode == "sequential"
                ):
                    self._frontier_update_virtual_outcomes(
                        bar_num=int(bar_num),
                        bar_high=float(row.get("high", np.nan)),
                        bar_low=float(row.get("low", np.nan)),
                    )
                if self._frontier_execution_mode == "sequential":
                    if self._frontier_seq_position is not None:
                        pos = self._frontier_seq_position
                        open_unrealized = float(
                            (float(row["close"]) - float(pos.get("entry_price", row["close"])))
                            * float(pos.get("direction", 0))
                        )

                unresolved_block = False
                if self._frontier_group_unresolved_enabled:
                    if self._frontier_execution_mode == "sequential":
                        pos = self._frontier_seq_position
                        unresolved_block = bool(
                            pos is not None
                            and int(pos.get("direction", 0)) == int(direction_int)
                            and str(pos.get("level_name", "")) == str(level_name)
                        )
                    else:
                        unresolved_block = self._frontier_has_unresolved_trade(
                            level_name=str(level_name),
                            direction=int(direction_int),
                        )

                if self._frontier_quality_source == "blend_policy_q2":
                    cand_rows: List[Dict[str, object]] = []
                    live_q = float(policy_prob) if policy_prob is not None else float(base_prob)
                    if (
                        minute_of_day >= int(self._frontier_blend_live_start)
                        and minute_of_day < int(self._frontier_blend_live_end)
                        and live_q >= float(self._frontier_blend_live_min_q)
                        and (
                            self._frontier_blend_live_cap <= 0
                            or self._frontier_blend_live_used_today < int(self._frontier_blend_live_cap)
                        )
                    ):
                        cand_rows.append(
                            {
                                "expert": "live",
                                "quality": float(live_q),
                                "flatten_min_q": float(self._frontier_blend_live_flatten_min_q),
                            }
                        )
                    q2_q = float(frontier_q2_score) if frontier_q2_score is not None else np.nan
                    if (
                        np.isfinite(q2_q)
                        and minute_of_day >= int(self._frontier_blend_q2_start)
                        and minute_of_day < int(self._frontier_blend_q2_end)
                        and q2_q >= float(self._frontier_blend_q2_min_q)
                        and (
                            self._frontier_blend_q2_cap <= 0
                            or self._frontier_blend_q2_used_today < int(self._frontier_blend_q2_cap)
                        )
                    ):
                        cand_rows.append(
                            {
                                "expert": "q2",
                                "quality": float(q2_q),
                                "flatten_min_q": float(self._frontier_blend_q2_flatten_min_q),
                            }
                        )
                    if not cand_rows:
                        continue
                    cand_rows = sorted(
                        cand_rows,
                        key=lambda x: float(x.get("quality", 0.0)),
                        reverse=True,
                    )
                    selected = None
                    for cand in cand_rows:
                        cand_quality = float(cand["quality"])
                        cand_flatten_min_q = float(cand["flatten_min_q"])
                        cand_expert = str(cand["expert"])
                        cand_action = "entry"
                        if self._frontier_execution_mode == "sequential":
                            cand_action = self._frontier_decide_sequential_action(
                                direction=int(direction_int),
                                quality_score=float(cand_quality),
                                open_unrealized=float(open_unrealized),
                                flatten_min_q_override=float(cand_flatten_min_q),
                            )
                            if cand_action == "ignore":
                                continue

                        needs_entry_accept = (
                            self._frontier_execution_mode != "sequential"
                            or cand_action in {"entry", "flip"}
                        )
                        needs_flatten_only = (
                            self._frontier_execution_mode == "sequential"
                            and cand_action == "flatten"
                        )

                        accepted_router = True
                        cand_lane: Optional[str] = None
                        if needs_entry_accept or (
                            needs_flatten_only and self._frontier_consume_budget_on_flatten
                        ):
                            if cand_expert == "live":
                                accepted_router, cand_lane = self._frontier_router_accept(
                                    row=row,
                                    minute_of_day=int(minute_of_day),
                                    direction=direction_int,
                                    level_name=level_name,
                                    quality_score=float(cand_quality),
                                    coverage_score=float(base_prob),
                                    rescue_score=float(base_prob),
                                    session_key=session_key,
                                    unresolved_block=bool(unresolved_block),
                                )
                                if accepted_router and needs_entry_accept:
                                    self._frontier_blend_live_used_today += 1
                            else:
                                accepted_router, cand_lane = self._frontier_blend_q2_accept(
                                    row=row,
                                    minute_of_day=int(minute_of_day),
                                    direction=direction_int,
                                    level_name=level_name,
                                    quality_score=float(cand_quality),
                                    session_key=session_key,
                                    unresolved_block=bool(unresolved_block),
                                )
                            if not accepted_router:
                                continue
                        else:
                            if unresolved_block:
                                continue
                            cand_lane = "flatten"

                        selected = (cand_expert, cand_quality, cand_action, cand_lane)
                        break

                    if selected is None:
                        continue
                    blend_expert = str(selected[0])
                    frontier_quality_score = float(selected[1])
                    sequential_action = str(selected[2])
                    frontier_lane = str(selected[3] or blend_expert)
                    lane = str(frontier_lane)
                else:
                    sequential_action = "entry"
                    if self._frontier_execution_mode == "sequential":
                        sequential_action = self._frontier_decide_sequential_action(
                            direction=int(direction_int),
                            quality_score=float(frontier_quality_score),
                            open_unrealized=float(open_unrealized),
                        )
                        if sequential_action == "ignore":
                            continue

                    needs_entry_accept = (
                        self._frontier_execution_mode != "sequential"
                        or sequential_action in {"entry", "flip"}
                    )
                    needs_flatten_only = (
                        self._frontier_execution_mode == "sequential"
                        and sequential_action == "flatten"
                    )

                    if needs_entry_accept or (needs_flatten_only and self._frontier_consume_budget_on_flatten):
                        accepted_router, frontier_lane = self._frontier_router_accept(
                            row=row,
                            minute_of_day=int(minute_of_day),
                            direction=direction_int,
                            level_name=level_name,
                            quality_score=float(frontier_quality_score),
                            coverage_score=float(base_prob),
                            rescue_score=float(base_prob),
                            session_key=session_key,
                            unresolved_block=bool(unresolved_block),
                        )
                        if not accepted_router:
                            continue
                        lane = str(frontier_lane or "quality")
                    else:
                        # flatten-only action without slot consumption
                        if unresolved_block:
                            continue
                        lane = "flatten"
            else:
                lane_high_thr = self._lane_high_threshold(session_key)
                lane_low_thr = self._lane_low_threshold(session_key)
                if self._dual_lane_enabled:
                    if score_value >= float(lane_high_thr):
                        lane = "high"
                    else:
                        allow_low = (
                            lane_low_thr is not None
                            and score_value >= float(lane_low_thr)
                            and score_value < float(lane_high_thr)
                            and self._max_low_per_day > 0
                            and self._dual_low_taken_today < self._max_low_per_day
                        )
                        if allow_low:
                            if minute_of_day < self._fallback_start_minute:
                                allow_low = False
                            if (not self._allow_low_before_high) and self._dual_high_taken_today > 0:
                                allow_low = False
                            if allow_low and self._low_lane_scope == "strong_levels":
                                lvl_name = str(row.get("nearest_level_name", ""))
                                if lvl_name not in self._strong_levels:
                                    allow_low = False
                        if not allow_low:
                            continue
                        lane = "low"

            if not (
                self._frontier_router_enabled
                and self._frontier_execution_mode == "sequential"
                and sequential_action == "flatten"
            ):
                # Optional persistence gate: require N consecutive candidate bars.
                prev_candidate_bar = self._last_candidate_bar_by_level_dir.get(level_dir)
                prev_streak = self._candidate_streak_by_level_dir.get(level_dir, 0)
                if prev_candidate_bar is not None and bar_num == (prev_candidate_bar + 1):
                    candidate_streak = prev_streak + 1
                else:
                    candidate_streak = 1
                self._last_candidate_bar_by_level_dir[level_dir] = bar_num
                self._candidate_streak_by_level_dir[level_dir] = candidate_streak
                if candidate_streak < self._same_level_min_confirm_bars:
                    continue

                # Optional episode gate: collapse clustered same-level touches into one decision.
                if self._use_episode_gating:
                    prev_episode_bar = self._episode_last_bar_by_level.get(level_name)
                    if prev_episode_bar is not None and (bar_num - prev_episode_bar) <= self._episode_gap_bars:
                        continue

                # Optional cooldown to reduce clustered repeated entries.
                if self._level_signal_cooldown_bars > 0:
                    prev_emit_bar = self._last_emit_bar_by_level_dir.get(level_dir)
                    if prev_emit_bar is not None and (bar_num - prev_emit_bar) < self._level_signal_cooldown_bars:
                        continue

                if self._dual_max_trades_per_day > 0 and self._dual_taken_today >= self._dual_max_trades_per_day:
                    continue

            score_for_budget = base_prob
            if (not self._frontier_router_enabled) and self._policy_model is not None:
                policy_allowed, policy_prob, policy_gate_reason, policy_std = self._evaluate_policy_gate(
                    row=row,
                    base_prob=base_prob,
                    session_key=session_key,
                    bar_num=bar_num,
                )
                if not policy_allowed:
                    continue
                if policy_prob is None:
                    continue
                score_for_budget = policy_prob

            # Optional day trade budget controller.
            if not self._frontier_router_enabled:
                if not self._accept_under_budget(
                    score=score_for_budget,
                    session_key=session_key,
                    base_threshold=base_threshold,
                ):
                    continue

            # Determine bar timestamp for dedup
            if 'dt' in row.index and pd.notna(row['dt']):
                bar_ts = str(row['dt'])
            else:
                bar_ts = str(idx)

            action_tag = direction
            if (
                self._frontier_router_enabled
                and self._frontier_execution_mode == "sequential"
                and sequential_action == "flatten"
            ):
                action_tag = f"flatten_{direction}"
            dedup_key = (level_name, action_tag, bar_ts)
            if dedup_key in self._emitted:
                continue

            emit_signal = True
            if self._frontier_router_enabled and self._frontier_execution_mode == "sequential":
                if sequential_action in {"flatten", "flip"} and self._frontier_seq_position is not None:
                    _ = self._frontier_close_sequential_position(
                        exit_price=float(row["close"]),
                        reason="flip_signal" if sequential_action == "flip" else "flatten_signal",
                    )
                if sequential_action == "flatten":
                    emit_signal = bool(self._frontier_emit_flatten_signal)

            entry_action = True
            if self._frontier_router_enabled and self._frontier_execution_mode == "sequential":
                entry_action = sequential_action in {"entry", "flip"}

            self._emitted.add(dedup_key)
            if entry_action:
                self._last_emit_bar_by_level_dir[level_dir] = bar_num
                if self._use_episode_gating:
                    self._episode_last_bar_by_level[level_name] = bar_num
                if (
                    self._frontier_router_enabled
                    and self._frontier_quality_source == "policy_prob"
                    and self._policy_model is not None
                ):
                    self._policy_mark_episode_start(bar_num=bar_num, row=row)
                if self._frontier_router_enabled and (
                    self._frontier_dynamic_budget_enabled
                    or self._frontier_group_unresolved_enabled
                    or self._frontier_execution_mode == "sequential"
                ):
                    self._frontier_register_virtual_trade(
                        bar_num=int(bar_num),
                        entry_price=float(row["close"]),
                        level_name=str(level_name),
                        direction=int(direction_int),
                        quality_score=float(frontier_quality_score),
                    )
                if self._frontier_router_enabled and self._frontier_execution_mode == "sequential":
                    self._frontier_seq_position = {
                        "entry_bar": int(bar_num),
                        "direction": int(direction_int),
                        "entry_price": float(row["close"]),
                        "quality_at_entry": float(frontier_quality_score),
                        "level_name": str(level_name),
                    }
                if self._dual_lane_enabled and (not self._frontier_router_enabled):
                    self._dual_taken_today += 1
                    if lane == "high":
                        self._dual_high_taken_today += 1
                    else:
                        self._dual_low_taken_today += 1

            if not emit_signal:
                continue

            # Build entry timestamp from bars_1m index
            if isinstance(bars_1m.index, pd.DatetimeIndex) and idx < len(bars_1m):
                entry_ts = bars_1m.index[idx] if isinstance(idx, int) else pd.Timestamp(bar_ts)
            else:
                entry_ts = pd.Timestamp(bar_ts) if bar_ts != str(idx) else pd.Timestamp.now()

            signals.append(RealtimeSignal(
                strategy_name=self.name,
                trigger_ts=entry_ts,
                entry_ts=entry_ts,
                entry_price=float(row['close']),
                direction=direction,
                level_name=level_name,
                level_value=float(row['nearest_level_price']),
                pred_proba=base_prob,
                metadata={
                    'proximity': float(abs(row['close'] - row['nearest_level_price'])),
                    'session': session_key,
                    'threshold_session': float(base_threshold),
                    'score_value': float(score_value),
                    'base_prob': float(base_prob),
                    'quality_score': float(frontier_quality_score),
                    'lane': lane,
                    'lane_high_threshold': (
                        float(lane_high_thr) if lane_high_thr is not None else None
                    ),
                    'lane_low_threshold': (
                        float(lane_low_thr) if lane_low_thr is not None else None
                    ),
                    'frontier_router_enabled': bool(self._frontier_router_enabled),
                    'frontier_router_lane': frontier_lane,
                    'frontier_blend_expert': blend_expert if self._frontier_quality_source == "blend_policy_q2" else None,
                    'execution_action': sequential_action if self._frontier_router_enabled else "entry",
                    'frontier_execution_mode': self._frontier_execution_mode,
                    'frontier_router_accepted_today': int(self._frontier_router_accepted_today),
                    'frontier_router_q_used_today': int(self._frontier_router_q_used_today),
                    'frontier_router_c_used_today': int(self._frontier_router_c_used_today),
                    'frontier_router_r_used_today': int(self._frontier_router_r_used_today),
                    'frontier_router_override_used_today': int(self._frontier_router_override_used_today),
                    'frontier_realized_wins_today': int(self._frontier_realized_wins_today),
                    'frontier_realized_losses_today': int(self._frontier_realized_losses_today),
                    'policy_prob': float(policy_prob) if policy_prob is not None else None,
                    'policy_mode': self._policy_mode,
                    'policy_uncertainty_std': (
                        float(policy_std) if policy_std is not None else None
                    ),
                    'budget_dynamic_threshold': (
                        float(self._budget_dynamic_threshold_by_session[session_key])
                        if self._use_session_budget else float(self._budget_dynamic_threshold)
                    ),
                    'budget_seen_today': int(self._budget_seen_today),
                    'budget_accepted_today': int(self._budget_accepted_today),
                    'budget_seen_session': int(self._budget_seen_by_session.get(session_key, 0)),
                    'budget_accepted_session': int(self._budget_accepted_by_session.get(session_key, 0)),
                    'dual_taken_today': int(self._dual_taken_today),
                    'dual_high_taken_today': int(self._dual_high_taken_today),
                    'dual_low_taken_today': int(self._dual_low_taken_today),
                },
            ))

        return signals
