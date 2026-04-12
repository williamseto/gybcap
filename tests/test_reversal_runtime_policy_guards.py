import numpy as np
import pandas as pd

from sandbox import run_sequential_reconciliation_v1 as sr
from strategies.reversal.level_utils import level_group
from strategies.reversal.realtime_strategy import ReversalPredictorStrategy


def test_level_group_shared_mapping() -> None:
    assert level_group("vwap") == "vwap"
    assert level_group("vwap_lo") == "vwap"
    assert level_group("rth_hi") == "rth"
    assert level_group("ovn_lo") == "ovn"
    assert level_group("ib_hi") == "ib"
    assert level_group("prev_low") == "prev"
    assert level_group("custom_level") == "custom_level"


def test_router_cfg_coerces_nan_inertia_override_to_none() -> None:
    cfg = {"frontier_inertia_override_min_q": "nan"}
    router = sr._router_cfg_from_live(cfg)
    assert router.inertia_override_min_q is None

    cfg_valid = {"frontier_inertia_override_min_q": "0.72"}
    router_valid = sr._router_cfg_from_live(cfg_valid)
    assert router_valid.inertia_override_min_q == 0.72


def test_blend_flatten_threshold_blocks_low_quality_opposite_signal() -> None:
    rows = pd.DataFrame(
        {
            "trading_day": ["2026-03-30", "2026-03-30"],
            "episode_start_idx": [0, 1],
            "minute_of_day": [400, 410],
            "trade_direction": [1, -1],
            "nearest_level_name": ["vwap", "vwap"],
            "event_base_prob": [0.50, 0.50],
            "policy_prob_runtime": [0.60, 0.62],
            "q_twohead_runtime": [np.nan, np.nan],
            "q_twohead": [np.nan, np.nan],
        }
    )
    day_groups = [("2026-03-30", np.asarray([0, 1], dtype=np.int64))]
    day_end = {"2026-03-30": 4}
    close = np.asarray([100.0, 100.2, 100.1, 100.15, 100.05], dtype=np.float64)
    high = close + 0.20
    low = close - 0.20

    base_cfg = dict(
        quality_source="blend_policy_q2",
        k_total=8,
        q_thr=0.0,
        q_slots=8,
        c_slots=0,
        r_slots=0,
        cooldown_min=0,
        gate=0.62,
        stop_hi=10.0,
        target_hi=40.0,
        stop_lo=4.0,
        target_lo=8.0,
        blend_live_cap=8,
        blend_live_start=390,
        blend_live_end=780,
        blend_live_min_q=0.0,
        blend_q2_cap=0,
        blend_q2_start=390,
        blend_q2_end=780,
        blend_q2_min_q=0.0,
    )
    variant = sr.ExecVariant(
        name="seq_flip_or_flatten",
        mode="sequential",
        opposite_action="flip_or_flatten",
        same_side_action="hold",
    )

    cfg_block = sr.RouterCfg(
        **base_cfg,
        blend_live_flatten_min_q=0.70,
        blend_q2_flatten_min_q=0.70,
    )
    _, trades_block, diag_block = sr._simulate_variant(
        rows=rows,
        day_groups=day_groups,
        day_end=day_end,
        close=close,
        high=high,
        low=low,
        router_cfg=cfg_block,
        variant=variant,
        policy_scorer=None,
        return_state=False,
    )
    assert int(diag_block["action_counts"].get("flatten", 0)) == 0
    assert int(diag_block["action_counts"].get("flip", 0)) == 0
    assert len(trades_block) == 1

    cfg_allow = sr.RouterCfg(
        **base_cfg,
        blend_live_flatten_min_q=0.50,
        blend_q2_flatten_min_q=0.50,
    )
    _, _trades_allow, diag_allow = sr._simulate_variant(
        rows=rows,
        day_groups=day_groups,
        day_end=day_end,
        close=close,
        high=high,
        low=low,
        router_cfg=cfg_allow,
        variant=variant,
        policy_scorer=None,
        return_state=False,
    )
    assert int(diag_allow["action_counts"].get("flatten", 0)) == 1
    assert int(diag_allow["action_counts"].get("flip", 0)) == 1


def test_realtime_early_loss_lock_and_inertia_rejectors() -> None:
    s = ReversalPredictorStrategy.__new__(ReversalPredictorStrategy)

    s._frontier_early_loss_lock_enabled = True
    s._frontier_early_loss_lock_until_minute = -1
    s._frontier_early_loss_lock_once_per_day = True
    s._frontier_early_loss_lock_triggered_today = False
    s._frontier_early_loss_lock_start_minute = 390
    s._frontier_early_loss_lock_end_minute = 450
    s._frontier_realized_losses_today = 2
    s._frontier_realized_wins_today = 0
    s._frontier_early_loss_lock_trigger_diff = 2
    s._frontier_early_loss_lock_duration_min = 20

    reason = s._frontier_early_loss_lock_reject_reason(minute_of_day=400)
    assert reason == "router_early_loss_lock"
    assert s._frontier_early_loss_lock_until_minute == 420
    assert s._frontier_early_loss_lock_triggered_today is True
    assert s._frontier_early_loss_lock_reject_reason(minute_of_day=410) == "router_early_loss_lock"
    assert s._frontier_early_loss_lock_reject_reason(minute_of_day=430) is None

    s._frontier_inertia_enabled = True
    s._frontier_inertia_level_group_minute_gap = 12
    s._frontier_inertia_global_minute_gap = 10
    s._frontier_inertia_override_min_q = None
    s._frontier_inertia_override_min_q_gap = 0.0
    s._frontier_inertia_last_by_group = {"vwap": {"minute": 405.0, "direction": 1.0, "quality": 0.60}}
    s._frontier_inertia_last_global = {"minute": 405.0, "direction": 1.0, "quality": 0.60}

    block_reason = s._frontier_inertia_reject_reason(
        minute_of_day=410,
        direction=-1,
        level_name="vwap",
        quality_score=0.61,
    )
    assert block_reason == "router_inertia_group"

    s._frontier_inertia_override_min_q = 0.70
    s._frontier_inertia_override_min_q_gap = 0.10
    allow_reason = s._frontier_inertia_reject_reason(
        minute_of_day=410,
        direction=-1,
        level_name="vwap",
        quality_score=0.75,
    )
    assert allow_reason is None
