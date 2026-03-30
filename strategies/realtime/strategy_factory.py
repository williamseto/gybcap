"""Registry/factory for constructing realtime strategies from config."""

from __future__ import annotations

import time
from typing import Callable, Dict, TYPE_CHECKING

from strategies.breakout.strategy import BreakoutRetestStrategy
from strategies.reversion.strategy import ReversionStrategy
from strategies.realtime.config import DEFAULT_LEVEL_COLS, RealtimeStrategyConfig
from strategies.realtime.protocol import BatchStrategyAdapter, RealtimeStrategy

if TYPE_CHECKING:
    from strategies.realtime.engine import RealtimeEngine


StrategyBuilder = Callable[[RealtimeStrategyConfig, "RealtimeEngine"], RealtimeStrategy]


class RealtimeStrategyFactory:
    """Strategy registry with pluggable builders keyed by config kind."""

    def __init__(self):
        self._builders: Dict[str, StrategyBuilder] = {}

    def register(self, kind: str, builder: StrategyBuilder) -> None:
        self._builders[kind] = builder

    def build(self, cfg: RealtimeStrategyConfig, engine: "RealtimeEngine") -> RealtimeStrategy:
        if cfg.kind not in self._builders:
            known = ", ".join(sorted(self._builders))
            raise KeyError(f"Unknown strategy kind '{cfg.kind}'. Known: {known}")
        return self._builders[cfg.kind](cfg, engine)


def _build_batch_adapter(
    cfg: RealtimeStrategyConfig,
    engine: "RealtimeEngine",
    strategy_cls: type,
    default_name: str,
) -> RealtimeStrategy:
    p = cfg.params
    return BatchStrategyAdapter(
        strategy_name=str(p.get("strategy_name", cfg.name or default_name)),
        strategy_cls=strategy_cls,
        level_provider=engine.level_provider,
        model_path=p.get("model_path"),
        level_cols=list(p.get("level_cols", DEFAULT_LEVEL_COLS)),
        threshold_pct=float(p.get("threshold_pct", 0.0012)),
        lookahead_bars=int(p.get("lookahead_bars", 12)),
        pred_threshold=float(p.get("pred_threshold", 0.4)),
        t_samp=str(p.get("t_samp", "5Min")),
        stop_buffer_pct=float(p.get("stop_buffer_pct", 0.0025)),
        rr=float(p.get("rr", 1.5)),
    )


def _build_batch_breakout(cfg: RealtimeStrategyConfig, engine: "RealtimeEngine") -> RealtimeStrategy:
    return _build_batch_adapter(cfg, engine, BreakoutRetestStrategy, "breakout")


def _build_batch_reversion(cfg: RealtimeStrategyConfig, engine: "RealtimeEngine") -> RealtimeStrategy:
    return _build_batch_adapter(cfg, engine, ReversionStrategy, "reversion")


def _build_reversal_predictor(cfg: RealtimeStrategyConfig, engine: "RealtimeEngine") -> RealtimeStrategy:
    from strategies.reversal.realtime_strategy import ReversalPredictorStrategy
    from strategies.realtime.csv_data_source import CSVDataSource

    p = cfg.params
    strategy = ReversalPredictorStrategy(
        model_dir=str(p["model_dir"]),
        pred_threshold=float(p.get("pred_threshold", 0.5)),
        pred_threshold_rth=(
            float(p["pred_threshold_rth"])
            if p.get("pred_threshold_rth") is not None
            else None
        ),
        pred_threshold_ovn=(
            float(p["pred_threshold_ovn"])
            if p.get("pred_threshold_ovn") is not None
            else None
        ),
        proximity_pts=float(p.get("proximity_pts", 5.0)),
        side_hysteresis_pts=float(p.get("side_hysteresis_pts", 0.0)),
        side_flip_confirm_pts=(
            float(p["side_flip_confirm_pts"])
            if p.get("side_flip_confirm_pts") is not None
            else None
        ),
        same_level_min_confirm_bars=int(p.get("same_level_min_confirm_bars", 1)),
        level_signal_cooldown_bars=int(p.get("level_signal_cooldown_bars", 0)),
        use_episode_gating=bool(p.get("use_episode_gating", False)),
        episode_gap_bars=int(p.get("episode_gap_bars", 1)),
        trade_budget_per_day=int(p.get("trade_budget_per_day", 0)),
        trade_budget_rth_per_day=int(p.get("trade_budget_rth_per_day", 0)),
        trade_budget_ovn_per_day=int(p.get("trade_budget_ovn_per_day", 0)),
        budget_adapt_step=float(p.get("budget_adapt_step", 0.01)),
        budget_adapt_tolerance=float(p.get("budget_adapt_tolerance", 0.5)),
        budget_expected_init=float(p.get("budget_expected_init", 20.0)),
        budget_ewma_alpha=float(p.get("budget_ewma_alpha", 0.20)),
        budget_max_threshold=float(p.get("budget_max_threshold", 0.95)),
        policy_model_dir=(
            str(p["policy_model_dir"])
            if p.get("policy_model_dir")
            else None
        ),
        policy_threshold=(
            float(p["policy_threshold"])
            if p.get("policy_threshold") is not None
            else None
        ),
        policy_threshold_rth=(
            float(p["policy_threshold_rth"])
            if p.get("policy_threshold_rth") is not None
            else None
        ),
        policy_threshold_ovn=(
            float(p["policy_threshold_ovn"])
            if p.get("policy_threshold_ovn") is not None
            else None
        ),
        policy_mode=(
            str(p["policy_mode"])
            if p.get("policy_mode") is not None
            else None
        ),
        policy_episode_gap_bars=(
            int(p["policy_episode_gap_bars"])
            if p.get("policy_episode_gap_bars") is not None
            else None
        ),
        policy_use_ensemble_mean=(
            bool(p["policy_use_ensemble_mean"])
            if p.get("policy_use_ensemble_mean") is not None
            else None
        ),
        policy_uncertainty_std_max=(
            float(p["policy_uncertainty_std_max"])
            if p.get("policy_uncertainty_std_max") is not None
            else None
        ),
        use_base_prob_prefilter=bool(p.get("use_base_prob_prefilter", True)),
        dual_lane_enabled=bool(p.get("dual_lane_enabled", False)),
        high_score_threshold=(
            float(p["high_score_threshold"])
            if p.get("high_score_threshold") is not None
            else None
        ),
        high_score_threshold_rth=(
            float(p["high_score_threshold_rth"])
            if p.get("high_score_threshold_rth") is not None
            else None
        ),
        high_score_threshold_ovn=(
            float(p["high_score_threshold_ovn"])
            if p.get("high_score_threshold_ovn") is not None
            else None
        ),
        low_score_threshold=(
            float(p["low_score_threshold"])
            if p.get("low_score_threshold") is not None
            else None
        ),
        low_score_threshold_rth=(
            float(p["low_score_threshold_rth"])
            if p.get("low_score_threshold_rth") is not None
            else None
        ),
        low_score_threshold_ovn=(
            float(p["low_score_threshold_ovn"])
            if p.get("low_score_threshold_ovn") is not None
            else None
        ),
        fallback_start_minute=int(p.get("fallback_start_minute", 390)),
        max_low_per_day=int(p.get("max_low_per_day", 1)),
        allow_low_before_high=bool(p.get("allow_low_before_high", False)),
        low_lane_scope=str(p.get("low_lane_scope", "all")),
        strong_levels=(
            [str(x).strip() for x in p.get("strong_levels", []) if str(x).strip()]
            if isinstance(p.get("strong_levels", []), list)
            else [
                s.strip()
                for s in str(p.get("strong_levels", "")).split(",")
                if s.strip()
            ]
        ),
        dual_max_trades_per_day=int(p.get("dual_max_trades_per_day", 0)),
        frontier_router_enabled=bool(p.get("frontier_router_enabled", False)),
        frontier_quality_source=str(p.get("frontier_quality_source", "base_prob")),
        frontier_quality_model_dir=(
            str(p["frontier_quality_model_dir"])
            if p.get("frontier_quality_model_dir")
            else None
        ),
        frontier_quality_lookup_path=(
            str(p["frontier_quality_lookup_path"])
            if p.get("frontier_quality_lookup_path")
            else None
        ),
        frontier_blend_live_cap=int(p.get("frontier_blend_live_cap", 5)),
        frontier_blend_live_start=int(p.get("frontier_blend_live_start", 390)),
        frontier_blend_live_end=int(p.get("frontier_blend_live_end", 780)),
        frontier_blend_live_min_q=float(p.get("frontier_blend_live_min_q", 0.0)),
        frontier_blend_live_flatten_min_q=float(
            p.get("frontier_blend_live_flatten_min_q", 0.66)
        ),
        frontier_blend_q2_cap=int(p.get("frontier_blend_q2_cap", 2)),
        frontier_blend_q2_start=int(p.get("frontier_blend_q2_start", 450)),
        frontier_blend_q2_end=int(p.get("frontier_blend_q2_end", 780)),
        frontier_blend_q2_min_q=float(p.get("frontier_blend_q2_min_q", 0.16)),
        frontier_blend_q2_flatten_min_q=float(
            p.get("frontier_blend_q2_flatten_min_q", 0.16)
        ),
        frontier_blend_q2_override_prob=(
            float(p["frontier_blend_q2_override_prob"])
            if p.get("frontier_blend_q2_override_prob") is not None
            else None
        ),
        frontier_execution_mode=str(p.get("frontier_execution_mode", "independent")),
        frontier_opposite_action=str(p.get("frontier_opposite_action", "ignore")),
        frontier_opposite_flip_min_q=float(p.get("frontier_opposite_flip_min_q", 0.72)),
        frontier_opposite_flatten_min_q=float(p.get("frontier_opposite_flatten_min_q", 0.66)),
        frontier_opposite_min_q_gap_vs_open=float(
            p.get("frontier_opposite_min_q_gap_vs_open", 0.0)
        ),
        frontier_opposite_flip_requires_open_loss=bool(
            p.get("frontier_opposite_flip_requires_open_loss", False)
        ),
        frontier_consume_budget_on_flatten=bool(p.get("frontier_consume_budget_on_flatten", False)),
        frontier_emit_flatten_signal=bool(p.get("frontier_emit_flatten_signal", False)),
        frontier_k_total=int(p.get("frontier_k_total", 7)),
        frontier_q_thr=float(p.get("frontier_q_thr", 0.58)),
        frontier_q_slots=int(p.get("frontier_q_slots", 3)),
        frontier_c_thr=float(p.get("frontier_c_thr", 0.48)),
        frontier_c_start=int(p.get("frontier_c_start", 450)),
        frontier_c_slots=int(p.get("frontier_c_slots", 2)),
        frontier_r_thr=float(p.get("frontier_r_thr", 0.45)),
        frontier_r_start=int(p.get("frontier_r_start", 600)),
        frontier_r_slots=int(p.get("frontier_r_slots", 1)),
        frontier_cooldown_min=int(p.get("frontier_cooldown_min", 5)),
        frontier_diversity_cap=int(p.get("frontier_diversity_cap", 2)),
        frontier_high_override_prob=(
            float(p["frontier_high_override_prob"])
            if p.get("frontier_high_override_prob") is not None
            else None
        ),
        frontier_high_override_cap=int(p.get("frontier_high_override_cap", 0)),
        frontier_high_override_start=int(p.get("frontier_high_override_start", 390)),
        frontier_high_override_window_min=int(p.get("frontier_high_override_window_min", 0)),
        frontier_high_override_window_cap=int(p.get("frontier_high_override_window_cap", 0)),
        frontier_group_unresolved_enabled=bool(
            p.get("frontier_group_unresolved_enabled", False)
        ),
        frontier_dynamic_budget_enabled=bool(p.get("frontier_dynamic_budget_enabled", False)),
        frontier_dynamic_extra_cap=int(p.get("frontier_dynamic_extra_cap", 0)),
        frontier_dynamic_extra_every=int(p.get("frontier_dynamic_extra_every", 1)),
        frontier_dynamic_start_minute=int(p.get("frontier_dynamic_start_minute", 390)),
        frontier_loss_lock_diff=int(p.get("frontier_loss_lock_diff", 0)),
        frontier_loss_lock_after_nominal_only=bool(
            p.get("frontier_loss_lock_after_nominal_only", True)
        ),
        frontier_virtual_gate=float(p.get("frontier_virtual_gate", 0.62)),
        frontier_virtual_stop_hi=float(p.get("frontier_virtual_stop_hi", 10.0)),
        frontier_virtual_target_hi=float(p.get("frontier_virtual_target_hi", 40.0)),
        frontier_virtual_stop_lo=float(p.get("frontier_virtual_stop_lo", 4.0)),
        frontier_virtual_target_lo=float(p.get("frontier_virtual_target_lo", 8.0)),
        frontier_virtual_gate_calibration_enabled=bool(
            p.get("frontier_virtual_gate_calibration_enabled", True)
        ),
        frontier_virtual_gate_calibration_days=int(
            p.get("frontier_virtual_gate_calibration_days", 20)
        ),
        frontier_virtual_gate_calibration_min_samples=int(
            p.get("frontier_virtual_gate_calibration_min_samples", 120)
        ),
        frontier_virtual_gate_calibration_decay=float(
            p.get("frontier_virtual_gate_calibration_decay", 0.97)
        ),
        frontier_virtual_gate_calibration_grid_size=int(
            p.get("frontier_virtual_gate_calibration_grid_size", 41)
        ),
        frontier_virtual_gate_calibration_min=float(
            p.get("frontier_virtual_gate_calibration_min", 0.10)
        ),
        frontier_virtual_gate_calibration_max=float(
            p.get("frontier_virtual_gate_calibration_max", 0.80)
        ),
    )

    history_csv = p.get("historical_csv_path")
    if history_csv:
        csv_source = CSVDataSource(str(history_csv))
        history = csv_source.fetch_history_bars(
            int(time.time()),
            n_days=int(p.get("warmup_days", 60)),
        )
        strategy.set_historical_context(history)

    return strategy


def _build_footprint_predictor(cfg: RealtimeStrategyConfig, engine: "RealtimeEngine") -> RealtimeStrategy:
    from strategies.reversal.realtime_footprint_strategy import FootprintRealtimeStrategy
    from strategies.realtime.csv_data_source import CSVDataSource

    p = cfg.params
    strategy = FootprintRealtimeStrategy(
        model_dir=str(p["model_dir"]),
        pred_threshold=p.get("pred_threshold"),
        proximity_pts=p.get("proximity_pts"),
        sec_window_sec=int(p.get("sec_window_sec", 900)),
        device=str(p.get("device", "auto")),
    )

    history_csv = p.get("historical_csv_path")
    if history_csv:
        csv_source = CSVDataSource(str(history_csv))
        history = csv_source.fetch_history_bars(
            int(time.time()),
            n_days=int(p.get("warmup_days", 60)),
        )
        strategy.set_historical_context(history)

    return strategy


def create_default_strategy_factory() -> RealtimeStrategyFactory:
    factory = RealtimeStrategyFactory()
    factory.register("batch_breakout", _build_batch_breakout)
    factory.register("batch_reversion", _build_batch_reversion)
    factory.register("reversal_predictor", _build_reversal_predictor)
    factory.register("footprint_predictor", _build_footprint_predictor)
    return factory
