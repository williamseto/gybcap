"""Strategy-specific CLI argument groups and config resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from strategies.realtime.config import RealtimeStrategyConfig


def _load_params_from_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    params = data.get("params")
    if isinstance(params, dict):
        return params
    return data


def _resolve(cli_value: Any, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in cfg:
        return cfg[key]
    return default


def add_reversal_predictor_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Reversal Predictor Strategy")
    group.add_argument(
        "--reversal-config",
        type=str,
        default=None,
        help="JSON config for reversal predictor params (keys mirror strategy params).",
    )
    group.add_argument(
        "--reversal-model-dir",
        type=str,
        default=None,
        help="Enable Phase-3 reversal predictor with model artifacts in this directory.",
    )
    group.add_argument(
        "--reversal-threshold",
        type=float,
        default=None,
        help="Probability threshold for reversal predictor (default: 0.50).",
    )
    group.add_argument(
        "--reversal-proximity",
        type=float,
        default=None,
        help="Near-level proximity (points) for reversal predictor (default: 5.0).",
    )
    group.add_argument(
        "--reversal-side-hysteresis",
        type=float,
        default=None,
        help="Side hysteresis band in points to reduce bull/bear flips near a level.",
    )
    group.add_argument(
        "--reversal-side-flip-confirm",
        type=float,
        default=None,
        help="Optional stronger opposite-side cross (points) required before side can flip.",
    )
    group.add_argument(
        "--reversal-confirm-bars",
        type=int,
        default=None,
        help="Require N consecutive qualifying bars for same level+direction before signaling.",
    )
    group.add_argument(
        "--reversal-cooldown-bars",
        type=int,
        default=None,
        help="Minimum bars between repeated signals for same level+direction.",
    )
    group.add_argument(
        "--reversal-use-episode-gating",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Collapse clustered same-level threshold hits into one decision episode.",
    )
    group.add_argument(
        "--reversal-episode-gap-bars",
        type=int,
        default=None,
        help="Max bar gap for candidates to remain in the same level episode (default: 1).",
    )
    group.add_argument(
        "--reversal-trade-budget-per-day",
        type=int,
        default=None,
        help="Optional max reversal entries per day (default: 0 disables budgeting).",
    )
    group.add_argument(
        "--reversal-budget-adapt-step",
        type=float,
        default=None,
        help="Adaptive budget threshold step size per episode (default: 0.01).",
    )
    group.add_argument(
        "--reversal-budget-adapt-tolerance",
        type=float,
        default=None,
        help="Adaptive budget tolerated gap before threshold adjusts (default: 0.5).",
    )
    group.add_argument(
        "--reversal-budget-expected-init",
        type=float,
        default=None,
        help="Initial expected episodes/day for budget calibration (default: 20).",
    )
    group.add_argument(
        "--reversal-budget-ewma-alpha",
        type=float,
        default=None,
        help="EWMA update alpha for expected episodes/day (default: 0.20).",
    )
    group.add_argument(
        "--reversal-budget-max-threshold",
        type=float,
        default=None,
        help="Upper bound for adaptive budget threshold (default: 0.95).",
    )
    group.add_argument(
        "--reversal-threshold-rth",
        type=float,
        default=None,
        help="Optional RTH-specific base threshold override.",
    )
    group.add_argument(
        "--reversal-threshold-ovn",
        type=float,
        default=None,
        help="Optional overnight-specific base threshold override.",
    )
    group.add_argument(
        "--reversal-budget-rth-per-day",
        type=int,
        default=None,
        help="Optional RTH-specific trade budget per day.",
    )
    group.add_argument(
        "--reversal-budget-ovn-per-day",
        type=int,
        default=None,
        help="Optional overnight-specific trade budget per day.",
    )
    group.add_argument(
        "--reversal-policy-model-dir",
        type=str,
        default=None,
        help="Optional stage-2 policy model directory for accept/skip gating.",
    )
    group.add_argument(
        "--reversal-policy-threshold",
        type=float,
        default=None,
        help="Optional decision threshold for stage-2 policy model.",
    )
    group.add_argument(
        "--reversal-policy-threshold-rth",
        type=float,
        default=None,
        help="Optional RTH-specific stage-2 policy threshold.",
    )
    group.add_argument(
        "--reversal-policy-threshold-ovn",
        type=float,
        default=None,
        help="Optional overnight-specific stage-2 policy threshold.",
    )
    group.add_argument(
        "--reversal-frontier-router",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable frontier-style quality/coverage/rescue slot router.",
    )
    group.add_argument(
        "--reversal-frontier-quality-source",
        type=str,
        choices=["base_prob", "policy_prob", "q_twohead", "q_lookup", "blend_policy_q2"],
        default=None,
        help=(
            "Quality score source for frontier router "
            "(base_prob, policy_prob, q_twohead, q_lookup, or blend_policy_q2)."
        ),
    )
    group.add_argument(
        "--reversal-frontier-quality-model-dir",
        type=str,
        default=None,
        help="Optional model dir for q_twohead quality scoring (metadata.json + tp_model.json + fail_model.json).",
    )
    group.add_argument(
        "--reversal-frontier-quality-lookup-path",
        type=str,
        default=None,
        help=(
            "Optional parquet lookup for q_lookup quality source "
            "(expects trading_day, episode_start_idx, q_twohead)."
        ),
    )
    group.add_argument(
        "--reversal-frontier-execution-mode",
        type=str,
        choices=["independent", "sequential"],
        default=None,
        help="Frontier execution mode: independent entries or sequential position-aware decisions.",
    )
    group.add_argument(
        "--reversal-frontier-opposite-action",
        type=str,
        choices=["ignore", "flatten", "flip", "flip_or_flatten"],
        default=None,
        help="Sequential mode behavior when an opposite-direction signal is observed.",
    )
    group.add_argument(
        "--reversal-frontier-opposite-flip-min-q",
        type=float,
        default=None,
        help="Minimum quality required to flip on opposite signal in sequential mode.",
    )
    group.add_argument(
        "--reversal-frontier-opposite-flatten-min-q",
        type=float,
        default=None,
        help="Minimum quality required to flatten on opposite signal in sequential mode.",
    )
    group.add_argument(
        "--reversal-frontier-opposite-min-q-gap-vs-open",
        type=float,
        default=None,
        help="Required quality improvement over current open position quality before flip is allowed.",
    )
    group.add_argument(
        "--reversal-frontier-opposite-flip-requires-open-loss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, flips are only allowed when current open position is in unrealized loss.",
    )
    group.add_argument(
        "--reversal-frontier-consume-budget-on-flatten",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, flatten-only actions still consume router slot/budget.",
    )
    group.add_argument(
        "--reversal-frontier-emit-flatten-signal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit explicit flatten action signals (default false).",
    )
    group.add_argument(
        "--reversal-frontier-k-total",
        type=int,
        default=None,
        help="Frontier router max accepted signals per day (K total).",
    )
    group.add_argument(
        "--reversal-frontier-q-thr",
        type=float,
        default=None,
        help="Frontier router quality-lane threshold.",
    )
    group.add_argument(
        "--reversal-frontier-q-slots",
        type=int,
        default=None,
        help="Frontier router quality-lane slot count.",
    )
    group.add_argument(
        "--reversal-frontier-c-thr",
        type=float,
        default=None,
        help="Frontier router coverage-lane threshold.",
    )
    group.add_argument(
        "--reversal-frontier-c-start",
        type=int,
        default=None,
        help="Frontier router coverage-lane start minute_of_day.",
    )
    group.add_argument(
        "--reversal-frontier-c-slots",
        type=int,
        default=None,
        help="Frontier router coverage-lane slot count.",
    )
    group.add_argument(
        "--reversal-frontier-r-thr",
        type=float,
        default=None,
        help="Frontier router rescue-lane threshold.",
    )
    group.add_argument(
        "--reversal-frontier-r-start",
        type=int,
        default=None,
        help="Frontier router rescue-lane start minute_of_day.",
    )
    group.add_argument(
        "--reversal-frontier-r-slots",
        type=int,
        default=None,
        help="Frontier router rescue-lane slot count.",
    )
    group.add_argument(
        "--reversal-frontier-cooldown-min",
        type=int,
        default=None,
        help="Frontier router cooldown (minutes between accepts).",
    )
    group.add_argument(
        "--reversal-frontier-inertia-enabled",
        action="store_true",
        help="Enable opposite-side inertia gate for frontier accepts.",
    )
    group.add_argument(
        "--reversal-frontier-inertia-global-minute-gap",
        type=int,
        default=None,
        help="Global opposite-side minimum gap in minutes before next accept.",
    )
    group.add_argument(
        "--reversal-frontier-inertia-level-group-minute-gap",
        type=int,
        default=None,
        help="Level-group opposite-side minimum gap in minutes before next accept.",
    )
    group.add_argument(
        "--reversal-frontier-inertia-override-min-q",
        type=float,
        default=None,
        help="Optional absolute quality threshold to override inertia blocks.",
    )
    group.add_argument(
        "--reversal-frontier-inertia-override-min-q-gap",
        type=float,
        default=None,
        help="Optional quality improvement vs last blocked side needed to override inertia.",
    )
    group.add_argument(
        "--reversal-frontier-diversity-cap",
        type=int,
        default=None,
        help="Frontier router per level+direction cap per day.",
    )
    group.add_argument(
        "--reversal-frontier-high-override-prob",
        type=float,
        default=None,
        help="Allow extra accepts beyond K when quality >= this threshold.",
    )
    group.add_argument(
        "--reversal-frontier-high-override-cap",
        type=int,
        default=None,
        help="Max number of high-probability overrides per day.",
    )
    group.add_argument(
        "--reversal-frontier-high-override-start",
        type=int,
        default=None,
        help="Minute-of-day after which high-probability overrides are allowed.",
    )
    group.add_argument(
        "--reversal-frontier-high-override-window-min",
        type=int,
        default=None,
        help=(
            "Optional override window size in minutes (post-budget only). "
            "If >0 with --reversal-frontier-high-override-window-cap, caps overrides per window."
        ),
    )
    group.add_argument(
        "--reversal-frontier-high-override-window-cap",
        type=int,
        default=None,
        help=(
            "Optional per-window override cap (used with --reversal-frontier-high-override-window-min). "
            "Set with daily cap < 0 for unbounded-day but time-bounded overrides."
        ),
    )
    group.add_argument(
        "--reversal-frontier-group-unresolved",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Block repeated accepts for same level+direction while prior virtual trade remains unresolved.",
    )
    group.add_argument(
        "--reversal-frontier-dynamic-budget",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable wins-losses-aware dynamic budget extension for frontier router.",
    )
    group.add_argument(
        "--reversal-frontier-dynamic-extra-cap",
        type=int,
        default=None,
        help="Max dynamic budget extension beyond K_total when dynamic budget is enabled.",
    )
    group.add_argument(
        "--reversal-frontier-dynamic-extra-every",
        type=int,
        default=None,
        help="Add one dynamic slot for each N net wins (wins-losses).",
    )
    group.add_argument(
        "--reversal-frontier-dynamic-start",
        type=int,
        default=None,
        help="Minute-of-day when dynamic budget extension can start.",
    )
    group.add_argument(
        "--reversal-frontier-loss-lock-diff",
        type=int,
        default=None,
        help="Lock further accepts when losses-wins >= this diff (0 disables).",
    )
    group.add_argument(
        "--reversal-frontier-loss-lock-after-nominal-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply loss-lock only after nominal K budget is reached.",
    )
    group.add_argument(
        "--reversal-frontier-virtual-gate",
        type=float,
        default=None,
        help="Gate used for virtual outcome lane assignment in dynamic-budget accounting.",
    )
    group.add_argument(
        "--reversal-skip-startup-calibration",
        action="store_true",
        help="Skip startup virtual-gate calibration (enabled by default).",
    )
    group.add_argument(
        "--reversal-frontier-virtual-high-stop",
        type=float,
        default=None,
        help="Virtual high-lane stop used by dynamic-budget accounting.",
    )
    group.add_argument(
        "--reversal-frontier-virtual-high-target",
        type=float,
        default=None,
        help="Virtual high-lane target used by dynamic-budget accounting.",
    )
    group.add_argument(
        "--reversal-frontier-virtual-low-stop",
        type=float,
        default=None,
        help="Virtual low-lane stop used by dynamic-budget accounting.",
    )
    group.add_argument(
        "--reversal-frontier-virtual-low-target",
        type=float,
        default=None,
        help="Virtual low-lane target used by dynamic-budget accounting.",
    )
    group.add_argument(
        "--reversal-history-csv",
        type=str,
        default=None,
        help="Optional CSV source for reversal predictor warmup history.",
    )
    group.add_argument(
        "--reversal-warmup-days",
        type=int,
        default=None,
        help="Warmup days for --reversal-history-csv (default: 60).",
    )


def build_reversal_predictor_config(
    args: argparse.Namespace,
    default_history_csv: Optional[str],
) -> Optional[RealtimeStrategyConfig]:
    cfg = _load_params_from_json(getattr(args, "reversal_config", None))
    model_dir = _resolve(getattr(args, "reversal_model_dir", None), cfg, "model_dir", None)
    if not model_dir:
        return None

    gate_calibration_enabled = bool(
        _resolve(
            None,
            cfg,
            "frontier_virtual_gate_calibration_enabled",
            True,
        )
    )
    if bool(getattr(args, "reversal_skip_startup_calibration", False)):
        gate_calibration_enabled = False

    history_csv = _resolve(
        getattr(args, "reversal_history_csv", None),
        cfg,
        "historical_csv_path",
        default_history_csv,
    )
    params = {
        "model_dir": str(model_dir),
        "pred_threshold": float(_resolve(args.reversal_threshold, cfg, "pred_threshold", 0.50)),
        "proximity_pts": float(_resolve(args.reversal_proximity, cfg, "proximity_pts", 5.0)),
        "side_hysteresis_pts": float(_resolve(args.reversal_side_hysteresis, cfg, "side_hysteresis_pts", 0.0)),
        "side_flip_confirm_pts": _resolve(
            args.reversal_side_flip_confirm, cfg, "side_flip_confirm_pts", None
        ),
        "same_level_min_confirm_bars": int(
            _resolve(args.reversal_confirm_bars, cfg, "same_level_min_confirm_bars", 1)
        ),
        "level_signal_cooldown_bars": int(
            _resolve(args.reversal_cooldown_bars, cfg, "level_signal_cooldown_bars", 0)
        ),
        "use_episode_gating": bool(
            _resolve(args.reversal_use_episode_gating, cfg, "use_episode_gating", False)
        ),
        "episode_gap_bars": int(_resolve(args.reversal_episode_gap_bars, cfg, "episode_gap_bars", 1)),
        "trade_budget_per_day": int(
            _resolve(args.reversal_trade_budget_per_day, cfg, "trade_budget_per_day", 0)
        ),
        "budget_adapt_step": float(_resolve(args.reversal_budget_adapt_step, cfg, "budget_adapt_step", 0.01)),
        "budget_adapt_tolerance": float(
            _resolve(args.reversal_budget_adapt_tolerance, cfg, "budget_adapt_tolerance", 0.5)
        ),
        "budget_expected_init": float(
            _resolve(args.reversal_budget_expected_init, cfg, "budget_expected_init", 20.0)
        ),
        "budget_ewma_alpha": float(_resolve(args.reversal_budget_ewma_alpha, cfg, "budget_ewma_alpha", 0.20)),
        "budget_max_threshold": float(
            _resolve(args.reversal_budget_max_threshold, cfg, "budget_max_threshold", 0.95)
        ),
        "pred_threshold_rth": _resolve(args.reversal_threshold_rth, cfg, "pred_threshold_rth", None),
        "pred_threshold_ovn": _resolve(args.reversal_threshold_ovn, cfg, "pred_threshold_ovn", None),
        "trade_budget_rth_per_day": int(
            _resolve(args.reversal_budget_rth_per_day, cfg, "trade_budget_rth_per_day", 0)
        ),
        "trade_budget_ovn_per_day": int(
            _resolve(args.reversal_budget_ovn_per_day, cfg, "trade_budget_ovn_per_day", 0)
        ),
        "policy_model_dir": _resolve(args.reversal_policy_model_dir, cfg, "policy_model_dir", None),
        "policy_threshold": _resolve(args.reversal_policy_threshold, cfg, "policy_threshold", None),
        "policy_threshold_rth": _resolve(
            args.reversal_policy_threshold_rth, cfg, "policy_threshold_rth", None
        ),
        "policy_threshold_ovn": _resolve(
            args.reversal_policy_threshold_ovn, cfg, "policy_threshold_ovn", None
        ),
        "frontier_router_enabled": bool(
            _resolve(
                getattr(args, "reversal_frontier_router", None),
                cfg,
                "frontier_router_enabled",
                False,
            )
        ),
        "frontier_quality_source": str(
            _resolve(
                getattr(args, "reversal_frontier_quality_source", None),
                cfg,
                "frontier_quality_source",
                "base_prob",
            )
        ),
        "frontier_quality_model_dir": _resolve(
            getattr(args, "reversal_frontier_quality_model_dir", None),
            cfg,
            "frontier_quality_model_dir",
            None,
        ),
        "frontier_quality_lookup_path": _resolve(
            getattr(args, "reversal_frontier_quality_lookup_path", None),
            cfg,
            "frontier_quality_lookup_path",
            None,
        ),
        # Blend-router knobs are currently config-only (no CLI flags).
        "frontier_blend_live_cap": int(_resolve(None, cfg, "frontier_blend_live_cap", 5)),
        "frontier_blend_live_start": int(_resolve(None, cfg, "frontier_blend_live_start", 390)),
        "frontier_blend_live_end": int(_resolve(None, cfg, "frontier_blend_live_end", 780)),
        "frontier_blend_live_min_q": float(_resolve(None, cfg, "frontier_blend_live_min_q", 0.0)),
        "frontier_blend_live_flatten_min_q": float(
            _resolve(None, cfg, "frontier_blend_live_flatten_min_q", 0.66)
        ),
        "frontier_blend_q2_cap": int(_resolve(None, cfg, "frontier_blend_q2_cap", 2)),
        "frontier_blend_q2_start": int(_resolve(None, cfg, "frontier_blend_q2_start", 450)),
        "frontier_blend_q2_end": int(_resolve(None, cfg, "frontier_blend_q2_end", 780)),
        "frontier_blend_q2_min_q": float(_resolve(None, cfg, "frontier_blend_q2_min_q", 0.16)),
        "frontier_blend_q2_flatten_min_q": float(
            _resolve(None, cfg, "frontier_blend_q2_flatten_min_q", 0.16)
        ),
        "frontier_blend_q2_override_prob": _resolve(
            None,
            cfg,
            "frontier_blend_q2_override_prob",
            0.24,
        ),
        "frontier_execution_mode": str(
            _resolve(
                getattr(args, "reversal_frontier_execution_mode", None),
                cfg,
                "frontier_execution_mode",
                "independent",
            )
        ),
        "frontier_opposite_action": str(
            _resolve(
                getattr(args, "reversal_frontier_opposite_action", None),
                cfg,
                "frontier_opposite_action",
                "ignore",
            )
        ),
        "frontier_opposite_flip_min_q": float(
            _resolve(
                getattr(args, "reversal_frontier_opposite_flip_min_q", None),
                cfg,
                "frontier_opposite_flip_min_q",
                0.72,
            )
        ),
        "frontier_opposite_flatten_min_q": float(
            _resolve(
                getattr(args, "reversal_frontier_opposite_flatten_min_q", None),
                cfg,
                "frontier_opposite_flatten_min_q",
                0.66,
            )
        ),
        "frontier_opposite_min_q_gap_vs_open": float(
            _resolve(
                getattr(args, "reversal_frontier_opposite_min_q_gap_vs_open", None),
                cfg,
                "frontier_opposite_min_q_gap_vs_open",
                0.0,
            )
        ),
        "frontier_opposite_flip_requires_open_loss": bool(
            _resolve(
                getattr(args, "reversal_frontier_opposite_flip_requires_open_loss", None),
                cfg,
                "frontier_opposite_flip_requires_open_loss",
                False,
            )
        ),
        "frontier_consume_budget_on_flatten": bool(
            _resolve(
                getattr(args, "reversal_frontier_consume_budget_on_flatten", None),
                cfg,
                "frontier_consume_budget_on_flatten",
                False,
            )
        ),
        "frontier_emit_flatten_signal": bool(
            _resolve(
                getattr(args, "reversal_frontier_emit_flatten_signal", None),
                cfg,
                "frontier_emit_flatten_signal",
                False,
            )
        ),
        "frontier_k_total": int(
            _resolve(
                getattr(args, "reversal_frontier_k_total", None),
                cfg,
                "frontier_k_total",
                7,
            )
        ),
        "frontier_q_thr": float(
            _resolve(
                getattr(args, "reversal_frontier_q_thr", None),
                cfg,
                "frontier_q_thr",
                0.58,
            )
        ),
        "frontier_q_slots": int(
            _resolve(
                getattr(args, "reversal_frontier_q_slots", None),
                cfg,
                "frontier_q_slots",
                3,
            )
        ),
        "frontier_c_thr": float(
            _resolve(
                getattr(args, "reversal_frontier_c_thr", None),
                cfg,
                "frontier_c_thr",
                0.48,
            )
        ),
        "frontier_c_start": int(
            _resolve(
                getattr(args, "reversal_frontier_c_start", None),
                cfg,
                "frontier_c_start",
                450,
            )
        ),
        "frontier_c_slots": int(
            _resolve(
                getattr(args, "reversal_frontier_c_slots", None),
                cfg,
                "frontier_c_slots",
                2,
            )
        ),
        "frontier_r_thr": float(
            _resolve(
                getattr(args, "reversal_frontier_r_thr", None),
                cfg,
                "frontier_r_thr",
                0.45,
            )
        ),
        "frontier_r_start": int(
            _resolve(
                getattr(args, "reversal_frontier_r_start", None),
                cfg,
                "frontier_r_start",
                600,
            )
        ),
        "frontier_r_slots": int(
            _resolve(
                getattr(args, "reversal_frontier_r_slots", None),
                cfg,
                "frontier_r_slots",
                1,
            )
        ),
        "frontier_cooldown_min": int(
            _resolve(
                getattr(args, "reversal_frontier_cooldown_min", None),
                cfg,
                "frontier_cooldown_min",
                5,
            )
        ),
        "frontier_inertia_enabled": bool(
            _resolve(
                True if getattr(args, "reversal_frontier_inertia_enabled", False) else None,
                cfg,
                "frontier_inertia_enabled",
                False,
            )
        ),
        "frontier_inertia_global_minute_gap": int(
            _resolve(
                getattr(args, "reversal_frontier_inertia_global_minute_gap", None),
                cfg,
                "frontier_inertia_global_minute_gap",
                0,
            )
        ),
        "frontier_inertia_level_group_minute_gap": int(
            _resolve(
                getattr(args, "reversal_frontier_inertia_level_group_minute_gap", None),
                cfg,
                "frontier_inertia_level_group_minute_gap",
                0,
            )
        ),
        "frontier_inertia_override_min_q": _resolve(
            getattr(args, "reversal_frontier_inertia_override_min_q", None),
            cfg,
            "frontier_inertia_override_min_q",
            None,
        ),
        "frontier_inertia_override_min_q_gap": float(
            _resolve(
                getattr(args, "reversal_frontier_inertia_override_min_q_gap", None),
                cfg,
                "frontier_inertia_override_min_q_gap",
                0.0,
            )
        ),
        "frontier_diversity_cap": int(
            _resolve(
                getattr(args, "reversal_frontier_diversity_cap", None),
                cfg,
                "frontier_diversity_cap",
                2,
            )
        ),
        "frontier_high_override_prob": _resolve(
            getattr(args, "reversal_frontier_high_override_prob", None),
            cfg,
            "frontier_high_override_prob",
            None,
        ),
        "frontier_high_override_cap": int(
            _resolve(
                getattr(args, "reversal_frontier_high_override_cap", None),
                cfg,
                "frontier_high_override_cap",
                0,
            )
        ),
        "frontier_high_override_start": int(
            _resolve(
                getattr(args, "reversal_frontier_high_override_start", None),
                cfg,
                "frontier_high_override_start",
                390,
            )
        ),
        "frontier_high_override_window_min": int(
            _resolve(
                getattr(args, "reversal_frontier_high_override_window_min", None),
                cfg,
                "frontier_high_override_window_min",
                0,
            )
        ),
        "frontier_high_override_window_cap": int(
            _resolve(
                getattr(args, "reversal_frontier_high_override_window_cap", None),
                cfg,
                "frontier_high_override_window_cap",
                0,
            )
        ),
        "frontier_group_unresolved_enabled": bool(
            _resolve(
                getattr(args, "reversal_frontier_group_unresolved", None),
                cfg,
                "frontier_group_unresolved_enabled",
                False,
            )
        ),
        "frontier_dynamic_budget_enabled": bool(
            _resolve(
                getattr(args, "reversal_frontier_dynamic_budget", None),
                cfg,
                "frontier_dynamic_budget_enabled",
                False,
            )
        ),
        "frontier_dynamic_extra_cap": int(
            _resolve(
                getattr(args, "reversal_frontier_dynamic_extra_cap", None),
                cfg,
                "frontier_dynamic_extra_cap",
                0,
            )
        ),
        "frontier_dynamic_extra_every": int(
            _resolve(
                getattr(args, "reversal_frontier_dynamic_extra_every", None),
                cfg,
                "frontier_dynamic_extra_every",
                1,
            )
        ),
        "frontier_dynamic_start_minute": int(
            _resolve(
                getattr(args, "reversal_frontier_dynamic_start", None),
                cfg,
                "frontier_dynamic_start_minute",
                390,
            )
        ),
        "frontier_loss_lock_diff": int(
            _resolve(
                getattr(args, "reversal_frontier_loss_lock_diff", None),
                cfg,
                "frontier_loss_lock_diff",
                0,
            )
        ),
        "frontier_loss_lock_after_nominal_only": bool(
            _resolve(
                getattr(args, "reversal_frontier_loss_lock_after_nominal_only", None),
                cfg,
                "frontier_loss_lock_after_nominal_only",
                True,
            )
        ),
        "frontier_virtual_gate": float(
            _resolve(
                getattr(args, "reversal_frontier_virtual_gate", None),
                cfg,
                "frontier_virtual_gate",
                0.62,
            )
        ),
        "frontier_virtual_gate_calibration_enabled": bool(
            gate_calibration_enabled
        ),
        "frontier_virtual_gate_calibration_days": int(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_days",
                20,
            )
        ),
        "frontier_virtual_gate_calibration_min_samples": int(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_min_samples",
                120,
            )
        ),
        "frontier_virtual_gate_calibration_decay": float(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_decay",
                0.97,
            )
        ),
        "frontier_virtual_gate_calibration_grid_size": int(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_grid_size",
                41,
            )
        ),
        "frontier_virtual_gate_calibration_min": float(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_min",
                0.10,
            )
        ),
        "frontier_virtual_gate_calibration_max": float(
            _resolve(
                None,
                cfg,
                "frontier_virtual_gate_calibration_max",
                0.80,
            )
        ),
        "frontier_virtual_stop_hi": float(
            _resolve(
                getattr(args, "reversal_frontier_virtual_high_stop", None),
                cfg,
                "frontier_virtual_stop_hi",
                10.0,
            )
        ),
        "frontier_virtual_target_hi": float(
            _resolve(
                getattr(args, "reversal_frontier_virtual_high_target", None),
                cfg,
                "frontier_virtual_target_hi",
                40.0,
            )
        ),
        "frontier_virtual_stop_lo": float(
            _resolve(
                getattr(args, "reversal_frontier_virtual_low_stop", None),
                cfg,
                "frontier_virtual_stop_lo",
                4.0,
            )
        ),
        "frontier_virtual_target_lo": float(
            _resolve(
                getattr(args, "reversal_frontier_virtual_low_target", None),
                cfg,
                "frontier_virtual_target_lo",
                8.0,
            )
        ),
        "policy_mode": _resolve(
            getattr(args, "reversal_policy_mode", None), cfg, "policy_mode", None
        ),
        "policy_episode_gap_bars": _resolve(
            getattr(args, "reversal_policy_episode_gap_bars", None),
            cfg,
            "policy_episode_gap_bars",
            None,
        ),
        "policy_use_ensemble_mean": _resolve(
            getattr(args, "reversal_policy_use_ensemble_mean", None),
            cfg,
            "policy_use_ensemble_mean",
            None,
        ),
        "policy_uncertainty_std_max": _resolve(
            getattr(args, "reversal_policy_uncertainty_std_max", None),
            cfg,
            "policy_uncertainty_std_max",
            None,
        ),
        "use_base_prob_prefilter": bool(
            _resolve(
                getattr(args, "reversal_use_base_prob_prefilter", None),
                cfg,
                "use_base_prob_prefilter",
                True,
            )
        ),
        "dual_lane_enabled": bool(
            _resolve(getattr(args, "reversal_dual_lane_enabled", None), cfg, "dual_lane_enabled", False)
        ),
        "high_score_threshold": _resolve(
            getattr(args, "reversal_high_score_threshold", None), cfg, "high_score_threshold", None
        ),
        "high_score_threshold_rth": _resolve(
            getattr(args, "reversal_high_score_threshold_rth", None), cfg, "high_score_threshold_rth", None
        ),
        "high_score_threshold_ovn": _resolve(
            getattr(args, "reversal_high_score_threshold_ovn", None), cfg, "high_score_threshold_ovn", None
        ),
        "low_score_threshold": _resolve(
            getattr(args, "reversal_low_score_threshold", None), cfg, "low_score_threshold", None
        ),
        "low_score_threshold_rth": _resolve(
            getattr(args, "reversal_low_score_threshold_rth", None), cfg, "low_score_threshold_rth", None
        ),
        "low_score_threshold_ovn": _resolve(
            getattr(args, "reversal_low_score_threshold_ovn", None), cfg, "low_score_threshold_ovn", None
        ),
        "fallback_start_minute": int(
            _resolve(getattr(args, "reversal_fallback_start_minute", None), cfg, "fallback_start_minute", 390)
        ),
        "max_low_per_day": int(
            _resolve(getattr(args, "reversal_max_low_per_day", None), cfg, "max_low_per_day", 1)
        ),
        "allow_low_before_high": bool(
            _resolve(getattr(args, "reversal_allow_low_before_high", None), cfg, "allow_low_before_high", False)
        ),
        "low_lane_scope": str(
            _resolve(getattr(args, "reversal_low_lane_scope", None), cfg, "low_lane_scope", "all")
        ),
        "strong_levels": _resolve(
            getattr(args, "reversal_strong_levels", None), cfg, "strong_levels", []
        ),
        "dual_max_trades_per_day": int(
            _resolve(
                getattr(args, "reversal_dual_max_trades_per_day", None),
                cfg,
                "dual_max_trades_per_day",
                0,
            )
        ),
        "historical_csv_path": history_csv,
        "warmup_days": int(_resolve(args.reversal_warmup_days, cfg, "warmup_days", 60)),
    }
    return RealtimeStrategyConfig(
        kind="reversal_predictor",
        name="reversal_predictor",
        params=params,
    )


def add_footprint_predictor_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Footprint Predictor Strategy")
    group.add_argument(
        "--footprint-config",
        type=str,
        default=None,
        help="JSON config for footprint predictor params (keys mirror strategy params).",
    )
    group.add_argument(
        "--footprint-model-dir",
        type=str,
        default=None,
        help="Enable realtime footprint predictor with deployable bundle in this directory.",
    )
    group.add_argument(
        "--footprint-threshold",
        type=float,
        default=None,
        help="Optional override threshold for footprint predictor.",
    )
    group.add_argument(
        "--footprint-proximity",
        type=float,
        default=None,
        help="Optional override proximity for footprint predictor.",
    )
    group.add_argument(
        "--footprint-sec-window",
        type=int,
        default=None,
        help="Seconds of raw tick context retained for footprint extraction (default: 900).",
    )
    group.add_argument(
        "--footprint-device",
        type=str,
        choices=["auto", "cpu"],
        default=None,
        help="Device preference for footprint model inference (default: auto).",
    )
    group.add_argument(
        "--footprint-history-csv",
        type=str,
        default=None,
        help="Optional CSV source for footprint predictor warmup history.",
    )
    group.add_argument(
        "--footprint-warmup-days",
        type=int,
        default=None,
        help="Warmup days for --footprint-history-csv (default: 60).",
    )


def build_footprint_predictor_config(
    args: argparse.Namespace,
    default_history_csv: Optional[str],
) -> Optional[RealtimeStrategyConfig]:
    cfg = _load_params_from_json(getattr(args, "footprint_config", None))
    model_dir = _resolve(getattr(args, "footprint_model_dir", None), cfg, "model_dir", None)
    if not model_dir:
        return None

    history_csv = _resolve(
        getattr(args, "footprint_history_csv", None),
        cfg,
        "historical_csv_path",
        default_history_csv,
    )
    params = {
        "model_dir": str(model_dir),
        "pred_threshold": _resolve(args.footprint_threshold, cfg, "pred_threshold", None),
        "proximity_pts": _resolve(args.footprint_proximity, cfg, "proximity_pts", None),
        "sec_window_sec": int(_resolve(args.footprint_sec_window, cfg, "sec_window_sec", 900)),
        "historical_csv_path": history_csv,
        "warmup_days": int(_resolve(args.footprint_warmup_days, cfg, "warmup_days", 60)),
        "device": str(_resolve(args.footprint_device, cfg, "device", "auto")),
    }
    return RealtimeStrategyConfig(
        kind="footprint_predictor",
        name="reversal_footprint",
        params=params,
    )
