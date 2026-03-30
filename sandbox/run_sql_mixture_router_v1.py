#!/usr/bin/env python
"""Research: multi-expert SQL mixture router with single-position reconciliation.

This study combines emitted signals from multiple expert strategy configs and
evaluates a causal, sequential execution contract:
- single open position at a time
- opposite signal can flatten or flip
- shared global budget plus per-expert budgets
- optional early-session cap to reduce front-loading

The design is intentionally config-driven so adding a new expert or policy
requires editing only small dictionaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return float(default)


def _norm_ts(ts: Any) -> Optional[pd.Timestamp]:
    t = pd.Timestamp(ts)
    if pd.isna(t):
        return None
    if t.tzinfo is not None:
        return t.tz_convert("America/Los_Angeles").tz_localize(None)
    return t


@dataclass
class ExpertSpec:
    name: str
    config_path: str
    start_minute: int = 390
    cap_per_day: int = 99
    flatten_min_q: float = 0.0
    flip_min_q: float = 1.0
    override_prob: Optional[float] = None
    weight: float = 1.0


@dataclass
class MixPolicy:
    name: str
    experts: List[str]
    k_total: int = 5
    cooldown_min: int = 5
    unresolved_min: int = 10
    opposite_action: str = "flatten"  # ignore | flatten | flip_or_flatten
    opposite_flip_min_q_gap_vs_open: float = 0.0
    early_end_minute: int = 510
    early_cap: int = 0
    override_start_minute: int = 570
    override_window_min: int = 0
    override_window_cap: int = 0
    gate: float = 0.62
    stop_hi: float = 10.0
    target_hi: float = 40.0
    stop_lo: float = 4.0
    target_lo: float = 8.0
    expert_caps: Dict[str, int] = None
    expert_start_minutes: Dict[str, int] = None
    expert_end_minutes: Dict[str, int] = None
    expert_override_prob: Dict[str, float] = None
    expert_entry_min_q: Dict[str, float] = None
    expert_flatten_min_q: Dict[str, float] = None
    expert_flip_min_q: Dict[str, float] = None
    expert_weights: Dict[str, float] = None
    # Optional guarded-mixture controls.
    q2_expert: str = "q2"
    anchor_expert: str = "live"
    regime_gate_enabled: bool = False
    regime_q2_start_minute: int = 450
    regime_min_first60_range: float = 0.0
    regime_min_drive_frac: float = 0.0
    regime_q2_quality_floor: float = 0.0
    regime_q2_adv_vs_anchor: float = 0.0
    # Optional online (day-over-day) expert reweighting.
    online_day_weight_enabled: bool = False
    online_alpha: float = 0.20
    online_beta: float = 0.05
    online_weight_min: float = 0.50
    online_weight_max: float = 2.00

    def __post_init__(self) -> None:
        if self.expert_caps is None:
            self.expert_caps = {}
        if self.expert_start_minutes is None:
            self.expert_start_minutes = {}
        if self.expert_end_minutes is None:
            self.expert_end_minutes = {}
        if self.expert_override_prob is None:
            self.expert_override_prob = {}
        if self.expert_entry_min_q is None:
            self.expert_entry_min_q = {}
        if self.expert_flatten_min_q is None:
            self.expert_flatten_min_q = {}
        if self.expert_flip_min_q is None:
            self.expert_flip_min_q = {}
        if self.expert_weights is None:
            self.expert_weights = {}


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_experts(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in _parse_csv(raw):
        if "=" not in tok:
            raise ValueError(f"Invalid expert spec '{tok}', expected name=path")
        name, path = tok.split("=", 1)
        n = str(name).strip()
        p = str(path).strip()
        if not n or not p:
            raise ValueError(f"Invalid expert spec '{tok}', expected name=path")
        out[n] = p
    if not out:
        raise ValueError("No experts parsed from --experts")
    return out


def _build_default_expert_specs(experts: Dict[str, str]) -> Dict[str, ExpertSpec]:
    out: Dict[str, ExpertSpec] = {}
    for name, path in experts.items():
        lname = str(name).lower()
        if "qtwohead" in lname or lname in {"q2", "q_twohead"}:
            out[name] = ExpertSpec(
                name=name,
                config_path=path,
                start_minute=450,
                cap_per_day=99,
                flatten_min_q=0.16,
                flip_min_q=0.22,
                override_prob=0.24,
                weight=1.0,
            )
        else:
            out[name] = ExpertSpec(
                name=name,
                config_path=path,
                start_minute=390,
                cap_per_day=99,
                flatten_min_q=0.66,
                flip_min_q=0.75,
                override_prob=0.75,
                weight=1.0,
            )
    return out


def _policy_grid(experts: Dict[str, ExpertSpec], base_contract: Dict[str, float]) -> List[MixPolicy]:
    names = list(experts.keys())
    if len(names) < 2:
        # Keep a minimal policy if only one expert was provided.
        n0 = names[0]
        e0 = experts[n0]
        return [
            MixPolicy(
                name=f"{n0}_solo_ref",
                experts=[n0],
                k_total=5,
                cooldown_min=5,
                unresolved_min=10,
                opposite_action="flatten",
                gate=base_contract["gate"],
                stop_hi=base_contract["stop_hi"],
                target_hi=base_contract["target_hi"],
                stop_lo=base_contract["stop_lo"],
                target_lo=base_contract["target_lo"],
                expert_caps={n0: 99},
                expert_start_minutes={n0: int(e0.start_minute)},
                expert_override_prob={n0: _safe_float(e0.override_prob, 0.0)},
                expert_entry_min_q={n0: 0.0},
                expert_flatten_min_q={n0: float(e0.flatten_min_q)},
                expert_flip_min_q={n0: float(e0.flip_min_q)},
                expert_weights={n0: float(e0.weight)},
            )
        ]

    # First two experts are primary for this campaign.
    a, b = names[0], names[1]
    ea = experts[a]
    eb = experts[b]
    common = dict(
        gate=base_contract["gate"],
        stop_hi=base_contract["stop_hi"],
        target_hi=base_contract["target_hi"],
        stop_lo=base_contract["stop_lo"],
        target_lo=base_contract["target_lo"],
    )

    pol: List[MixPolicy] = []
    pol.append(
        MixPolicy(
            name=f"{a}_solo_flatten_ref",
            experts=[a],
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={a: 99},
            expert_start_minutes={a: int(ea.start_minute)},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q)},
            expert_weights={a: float(ea.weight)},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name=f"{b}_solo_flatten_ref",
            experts=[b],
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={b: 99},
            expert_start_minutes={b: int(eb.start_minute)},
            expert_override_prob={b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={b: 0.0},
            expert_flatten_min_q={b: float(eb.flatten_min_q)},
            expert_flip_min_q={b: float(eb.flip_min_q)},
            expert_weights={b: float(eb.weight)},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name="mix_q2_veto_flattenonly_v1",
            experts=[a, b],
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            override_start_minute=570,
            override_window_min=60,
            override_window_cap=1,
            expert_caps={a: 5, b: 0},
            expert_start_minutes={a: int(ea.start_minute), b: 450},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0, b: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
            expert_weights={a: float(ea.weight), b: float(eb.weight)},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name="mix_q2_veto_flip_v1",
            experts=[a, b],
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flip_or_flatten",
            opposite_flip_min_q_gap_vs_open=0.00,
            override_start_minute=570,
            override_window_min=60,
            override_window_cap=1,
            expert_caps={a: 5, b: 0},
            expert_start_minutes={a: int(ea.start_minute), b: 450},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0, b: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
            expert_weights={a: float(ea.weight), b: float(eb.weight)},
            **common,
        )
    )

    # Mixture family.
    combos = [
        ("mix_flat_k7_v1", 7, 5, 10, 0, 450, "flatten", 0.00, 5, 3),
        ("mix_flat_k8_v1", 8, 5, 10, 0, 450, "flatten", 0.00, 5, 4),
        ("mix_flat_k9_v1", 9, 5, 10, 0, 450, "flatten", 0.00, 6, 4),
        ("mix_flat_k8_earlycap4_v1", 8, 5, 10, 4, 450, "flatten", 0.00, 5, 4),
        ("mix_flat_k9_earlycap4_v1", 9, 5, 10, 4, 450, "flatten", 0.00, 6, 5),
        ("mix_flat_k8_q2late_v1", 8, 5, 10, 0, 510, "flatten", 0.00, 5, 4),
        ("mix_flip_k7_v1", 7, 5, 10, 0, 450, "flip_or_flatten", 0.00, 5, 3),
        ("mix_flip_k8_v1", 8, 5, 10, 0, 450, "flip_or_flatten", 0.00, 5, 4),
        ("mix_flip_k8_gap05_v1", 8, 5, 10, 0, 450, "flip_or_flatten", 0.05, 5, 4),
        ("mix_flip_k9_earlycap4_v1", 9, 5, 10, 4, 450, "flip_or_flatten", 0.00, 6, 5),
        ("mix_flat_cd3_v1", 8, 3, 8, 0, 450, "flatten", 0.00, 5, 4),
        ("mix_flip_cd3_v1", 8, 3, 8, 0, 450, "flip_or_flatten", 0.00, 5, 4),
    ]
    for name, k, cd, unresolved, early_cap, b_start, opp, gap, cap_a, cap_b in combos:
        pol.append(
            MixPolicy(
                name=name,
                experts=[a, b],
                k_total=int(k),
                cooldown_min=int(cd),
                unresolved_min=int(unresolved),
                opposite_action=str(opp),
                opposite_flip_min_q_gap_vs_open=float(gap),
                early_cap=int(early_cap),
                override_start_minute=570,
                override_window_min=60,
                override_window_cap=1,
                expert_caps={a: int(cap_a), b: int(cap_b)},
                expert_start_minutes={a: int(ea.start_minute), b: int(b_start)},
                expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
                expert_entry_min_q={a: 0.0, b: 0.0},
                expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                expert_weights={a: float(ea.weight), b: float(eb.weight)},
                **common,
            )
        )

    # Entry-quality gate sweep for expert-b (q_twohead-like expert).
    for qmin in [0.18, 0.20, 0.22, 0.24]:
        tag = str(int(round(qmin * 100)))
        pol.append(
            MixPolicy(
                name=f"mix_flat_k8_q2min{tag}_v1",
                experts=[a, b],
                k_total=8,
                cooldown_min=5,
                unresolved_min=10,
                opposite_action="flatten",
                override_start_minute=570,
                override_window_min=60,
                override_window_cap=1,
                expert_caps={a: 5, b: 4},
                expert_start_minutes={a: int(ea.start_minute), b: 450},
                expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
                expert_entry_min_q={a: 0.0, b: float(qmin)},
                expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                expert_weights={a: float(ea.weight), b: float(eb.weight)},
                **common,
            )
        )
    for qmin, cap_b, b_start, k_total in [
        (0.18, 1, 510, 6),
        (0.20, 1, 510, 6),
        (0.18, 2, 510, 7),
        (0.20, 2, 510, 7),
        (0.18, 1, 450, 6),
        (0.20, 1, 450, 6),
    ]:
        tag = f"{int(round(qmin * 100))}_c{cap_b}_s{b_start}_k{k_total}"
        pol.append(
            MixPolicy(
                name=f"mix_light_q2min{tag}_v1",
                experts=[a, b],
                k_total=int(k_total),
                cooldown_min=5,
                unresolved_min=10,
                opposite_action="flatten",
                override_start_minute=570,
                override_window_min=60,
                override_window_cap=1,
                expert_caps={a: 5, b: int(cap_b)},
                expert_start_minutes={a: int(ea.start_minute), b: int(b_start)},
                expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
                expert_entry_min_q={a: 0.0, b: float(qmin)},
                expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                expert_weights={a: float(ea.weight), b: float(eb.weight)},
                **common,
            )
        )
    return pol


def _guarded_policy_grid(experts: Dict[str, ExpertSpec], base_contract: Dict[str, float]) -> List[MixPolicy]:
    names = list(experts.keys())
    if len(names) < 2:
        return _policy_grid(experts, base_contract)
    a, b = names[0], names[1]
    ea = experts[a]
    eb = experts[b]
    common = dict(
        gate=base_contract["gate"],
        stop_hi=base_contract["stop_hi"],
        target_hi=base_contract["target_hi"],
        stop_lo=base_contract["stop_lo"],
        target_lo=base_contract["target_lo"],
    )
    pol: List[MixPolicy] = []
    pol.append(
        MixPolicy(
            name=f"{a}_solo_flatten_ref",
            experts=[a],
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={a: 99},
            expert_start_minutes={a: int(ea.start_minute)},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q)},
            expert_weights={a: 1.0},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name=f"{b}_solo_flatten_ref",
            experts=[b],
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={b: 99},
            expert_start_minutes={b: int(eb.start_minute)},
            expert_override_prob={b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={b: 0.0},
            expert_flatten_min_q={b: float(eb.flatten_min_q)},
            expert_flip_min_q={b: float(eb.flip_min_q)},
            expert_weights={b: 1.0},
            **common,
        )
    )

    # Guarded mixtures: policy expert as anchor, q2 expert admitted only in
    # trend/drive regimes and with score advantage over anchor at same bar.
    pol.append(
        MixPolicy(
            name="guarded_mix_static_v1",
            experts=[a, b],
            k_total=7,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            override_start_minute=570,
            override_window_min=60,
            override_window_cap=1,
            expert_caps={a: 6, b: 3},
            expert_start_minutes={a: int(ea.start_minute), b: 450},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0, b: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
            expert_weights={a: 1.0, b: 1.0},
            q2_expert=b,
            anchor_expert=a,
            regime_gate_enabled=True,
            regime_q2_start_minute=450,
            regime_min_first60_range=18.0,
            regime_min_drive_frac=0.08,
            regime_q2_quality_floor=0.18,
            regime_q2_adv_vs_anchor=0.02,
            online_day_weight_enabled=False,
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name="guarded_mix_online_v1",
            experts=[a, b],
            k_total=7,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            override_start_minute=570,
            override_window_min=60,
            override_window_cap=1,
            expert_caps={a: 6, b: 3},
            expert_start_minutes={a: int(ea.start_minute), b: 450},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0, b: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
            expert_weights={a: 1.0, b: 1.0},
            q2_expert=b,
            anchor_expert=a,
            regime_gate_enabled=True,
            regime_q2_start_minute=450,
            regime_min_first60_range=18.0,
            regime_min_drive_frac=0.08,
            regime_q2_quality_floor=0.18,
            regime_q2_adv_vs_anchor=0.02,
            online_day_weight_enabled=True,
            online_alpha=0.20,
            online_beta=0.06,
            online_weight_min=0.50,
            online_weight_max=1.80,
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name="guarded_mix_online_loose_v1",
            experts=[a, b],
            k_total=8,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            override_start_minute=570,
            override_window_min=60,
            override_window_cap=1,
            expert_caps={a: 6, b: 4},
            expert_start_minutes={a: int(ea.start_minute), b: 450},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0, b: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
            expert_weights={a: 1.0, b: 1.0},
            q2_expert=b,
            anchor_expert=a,
            regime_gate_enabled=True,
            regime_q2_start_minute=450,
            regime_min_first60_range=14.0,
            regime_min_drive_frac=0.06,
            regime_q2_quality_floor=0.16,
            regime_q2_adv_vs_anchor=0.00,
            online_day_weight_enabled=True,
            online_alpha=0.25,
            online_beta=0.07,
            online_weight_min=0.40,
            online_weight_max=2.00,
            **common,
        )
    )
    # Compact guarded sweep around regime/advantage thresholds and q2 cap.
    for cap_b in [1, 2, 3, 4]:
        for f60_min in [14.0, 18.0, 22.0]:
            for dfrac_min in [0.06, 0.09]:
                for adv in [0.00, 0.02]:
                    for online in [False, True]:
                        tag = (
                            f"guarded_mix_sweep_c{cap_b}"
                            f"_f{int(round(f60_min))}"
                            f"_d{int(round(dfrac_min*100))}"
                            f"_a{int(round(adv*100))}"
                            f"_{'onl' if online else 'static'}"
                        )
                        pol.append(
                            MixPolicy(
                                name=tag,
                                experts=[a, b],
                                k_total=8 if cap_b >= 3 else 7,
                                cooldown_min=5,
                                unresolved_min=10,
                                opposite_action="flatten",
                                override_start_minute=570,
                                override_window_min=60,
                                override_window_cap=1,
                                expert_caps={a: 6, b: int(cap_b)},
                                expert_start_minutes={a: int(ea.start_minute), b: 450},
                                expert_override_prob={
                                    a: _safe_float(ea.override_prob, 0.0),
                                    b: _safe_float(eb.override_prob, 0.0),
                                },
                                expert_entry_min_q={a: 0.0, b: 0.0},
                                expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                                expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                                expert_weights={a: 1.0, b: 1.0},
                                q2_expert=b,
                                anchor_expert=a,
                                regime_gate_enabled=True,
                                regime_q2_start_minute=450,
                                regime_min_first60_range=float(f60_min),
                                regime_min_drive_frac=float(dfrac_min),
                                regime_q2_quality_floor=0.16,
                                regime_q2_adv_vs_anchor=float(adv),
                                online_day_weight_enabled=bool(online),
                                online_alpha=0.20,
                                online_beta=0.06,
                                online_weight_min=0.50,
                                online_weight_max=1.80,
                                **common,
                            )
                        )
    # Coverage-focused override sweep: allow post-budget windows earlier in day,
    # while keeping guarded q2 admission and bounded q2 specialist capacity.
    for cap_b in [1, 2, 3]:
        for ov_start in [570, 450, 390]:
            for ov_cap in [1, 2]:
                k_total = 7 if cap_b <= 2 else 8
                tag = f"guarded_cov_c{cap_b}_os{ov_start}_wc{ov_cap}"
                pol.append(
                    MixPolicy(
                        name=tag,
                        experts=[a, b],
                        k_total=int(k_total),
                        cooldown_min=5,
                        unresolved_min=10,
                        opposite_action="flatten",
                        override_start_minute=int(ov_start),
                        override_window_min=60,
                        override_window_cap=int(ov_cap),
                        expert_caps={a: 6, b: int(cap_b)},
                        expert_start_minutes={a: int(ea.start_minute), b: 450},
                        expert_override_prob={
                            a: _safe_float(ea.override_prob, 0.0),
                            b: _safe_float(eb.override_prob, 0.0),
                        },
                        expert_entry_min_q={a: 0.0, b: 0.0},
                        expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                        expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                        expert_weights={a: 1.0, b: 1.0},
                        q2_expert=b,
                        anchor_expert=a,
                        regime_gate_enabled=True,
                        regime_q2_start_minute=450,
                        regime_min_first60_range=14.0,
                        regime_min_drive_frac=0.09,
                        regime_q2_quality_floor=0.16,
                        regime_q2_adv_vs_anchor=0.0,
                        online_day_weight_enabled=False,
                        **common,
                    )
                )
    return pol


def _split_policy_grid(experts: Dict[str, ExpertSpec], base_contract: Dict[str, float]) -> List[MixPolicy]:
    names = list(experts.keys())
    if len(names) < 2:
        return _policy_grid(experts, base_contract)
    a, b = names[0], names[1]
    ea = experts[a]
    eb = experts[b]
    common = dict(
        gate=base_contract["gate"],
        stop_hi=base_contract["stop_hi"],
        target_hi=base_contract["target_hi"],
        stop_lo=base_contract["stop_lo"],
        target_lo=base_contract["target_lo"],
    )
    pol: List[MixPolicy] = []
    pol.append(
        MixPolicy(
            name=f"{a}_solo_flatten_ref",
            experts=[a],
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={a: 99},
            expert_start_minutes={a: int(ea.start_minute)},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q)},
            expert_weights={a: 1.0},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name=f"{b}_solo_flatten_ref",
            experts=[b],
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={b: 99},
            expert_start_minutes={b: int(eb.start_minute)},
            expert_override_prob={b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={b: 0.0},
            expert_flatten_min_q={b: float(eb.flatten_min_q)},
            expert_flip_min_q={b: float(eb.flip_min_q)},
            expert_weights={b: 1.0},
            **common,
        )
    )
    # Regime split: live expert only in opening hour; q2 expert from 07:30 onward.
    for k_total in [7, 8, 9]:
        for live_cap in [1, 2, 3]:
            for live_qmin in [0.50, 0.55, 0.60]:
                tag = f"split_open_livecap{live_cap}_q{int(round(live_qmin*100))}_k{k_total}"
                pol.append(
                    MixPolicy(
                        name=tag,
                        experts=[a, b],
                        k_total=int(k_total),
                        cooldown_min=5,
                        unresolved_min=10,
                        opposite_action="flatten",
                        override_start_minute=450,
                        override_window_min=60,
                        override_window_cap=1,
                        expert_caps={a: int(live_cap), b: 99},
                        expert_start_minutes={a: int(ea.start_minute), b: 450},
                        expert_end_minutes={a: 450},
                        expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
                        expert_entry_min_q={a: float(live_qmin), b: 0.0},
                        expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                        expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                        expert_weights={a: 1.0, b: 1.0},
                        **common,
                    )
                )
    # Slightly more permissive live window to include 07:30-08:00.
    for k_total in [8, 9]:
        for live_cap in [1, 2]:
            tag = f"split_open90_livecap{live_cap}_k{k_total}"
            pol.append(
                MixPolicy(
                    name=tag,
                    experts=[a, b],
                    k_total=int(k_total),
                    cooldown_min=5,
                    unresolved_min=10,
                    opposite_action="flatten",
                    override_start_minute=510,
                    override_window_min=60,
                    override_window_cap=1,
                    expert_caps={a: int(live_cap), b: 99},
                    expert_start_minutes={a: int(ea.start_minute), b: 450},
                    expert_end_minutes={a: 510},
                    expert_override_prob={a: _safe_float(ea.override_prob, 0.0), b: _safe_float(eb.override_prob, 0.0)},
                    expert_entry_min_q={a: 0.55, b: 0.0},
                    expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                    expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                    expert_weights={a: 1.0, b: 1.0},
                    **common,
                )
            )
    return pol


def _blend_policy_grid(experts: Dict[str, ExpertSpec], base_contract: Dict[str, float]) -> List[MixPolicy]:
    names = list(experts.keys())
    if len(names) < 2:
        return _policy_grid(experts, base_contract)
    a, b = names[0], names[1]
    ea = experts[a]
    eb = experts[b]
    common = dict(
        gate=base_contract["gate"],
        stop_hi=base_contract["stop_hi"],
        target_hi=base_contract["target_hi"],
        stop_lo=base_contract["stop_lo"],
        target_lo=base_contract["target_lo"],
    )
    pol: List[MixPolicy] = []
    pol.append(
        MixPolicy(
            name=f"{a}_solo_flatten_ref",
            experts=[a],
            k_total=5,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={a: 99},
            expert_start_minutes={a: int(ea.start_minute)},
            expert_override_prob={a: _safe_float(ea.override_prob, 0.0)},
            expert_entry_min_q={a: 0.0},
            expert_flatten_min_q={a: float(ea.flatten_min_q)},
            expert_flip_min_q={a: float(ea.flip_min_q)},
            expert_weights={a: 1.0},
            **common,
        )
    )
    pol.append(
        MixPolicy(
            name=f"{b}_solo_flatten_ref",
            experts=[b],
            k_total=6,
            cooldown_min=5,
            unresolved_min=10,
            opposite_action="flatten",
            expert_caps={b: 99},
            expert_start_minutes={b: int(eb.start_minute)},
            expert_override_prob={b: _safe_float(eb.override_prob, 0.0)},
            expert_entry_min_q={b: 0.0},
            expert_flatten_min_q={b: float(eb.flatten_min_q)},
            expert_flip_min_q={b: float(eb.flip_min_q)},
            expert_weights={b: 1.0},
            **common,
        )
    )
    # Blend: live anchor throughout day + bounded q2 specialist after opening.
    for k_total in [6, 7, 8]:
        for q2_cap in [1, 2, 3]:
            for q2_min in [0.16, 0.18, 0.20, 0.22]:
                for early_cap in [0, 4]:
                    tag = f"blend_k{k_total}_q2c{q2_cap}_q2m{int(round(q2_min*100))}_ec{early_cap}"
                    pol.append(
                        MixPolicy(
                            name=tag,
                            experts=[a, b],
                            k_total=int(k_total),
                            cooldown_min=5,
                            unresolved_min=10,
                            opposite_action="flatten",
                            early_cap=int(early_cap),
                            early_end_minute=510,
                            override_start_minute=450,
                            override_window_min=60,
                            override_window_cap=1,
                            expert_caps={a: 5, b: int(q2_cap)},
                            expert_start_minutes={a: int(ea.start_minute), b: 450},
                            expert_override_prob={
                                a: _safe_float(ea.override_prob, 0.0),
                                b: _safe_float(eb.override_prob, 0.0),
                            },
                            expert_entry_min_q={a: 0.0, b: float(q2_min)},
                            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                            expert_weights={a: 1.0, b: 1.0},
                            **common,
                        )
                    )
    # Guarded supplement: allow q2 only in stronger first-hour regimes.
    for q2_cap in [1, 2]:
        for q2_min in [0.18, 0.20]:
            for f60_min in [14.0, 18.0]:
                for drive_min in [0.06, 0.09]:
                    tag = (
                        f"blend_guard_q2c{q2_cap}_m{int(round(q2_min*100))}"
                        f"_f{int(round(f60_min))}_d{int(round(drive_min*100))}"
                    )
                    pol.append(
                        MixPolicy(
                            name=tag,
                            experts=[a, b],
                            k_total=7,
                            cooldown_min=5,
                            unresolved_min=10,
                            opposite_action="flatten",
                            override_start_minute=450,
                            override_window_min=60,
                            override_window_cap=1,
                            expert_caps={a: 5, b: int(q2_cap)},
                            expert_start_minutes={a: int(ea.start_minute), b: 450},
                            expert_override_prob={
                                a: _safe_float(ea.override_prob, 0.0),
                                b: _safe_float(eb.override_prob, 0.0),
                            },
                            expert_entry_min_q={a: 0.0, b: float(q2_min)},
                            expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                            expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                            expert_weights={a: 1.0, b: 1.0},
                            q2_expert=b,
                            anchor_expert=a,
                            regime_gate_enabled=True,
                            regime_q2_start_minute=450,
                            regime_min_first60_range=float(f60_min),
                            regime_min_drive_frac=float(drive_min),
                            regime_q2_quality_floor=float(q2_min),
                            regime_q2_adv_vs_anchor=0.0,
                            **common,
                    )
                )
    # Focused neighborhood around current best blend families.
    for opp in ["flatten", "flip_or_flatten"]:
        for q2_w in [1.0, 1.1, 1.2]:
            for cd in [4, 5, 6]:
                tag = f"blend_focus_{opp}_q2w{int(round(q2_w*100))}_cd{cd}"
                pol.append(
                    MixPolicy(
                        name=tag,
                        experts=[a, b],
                        k_total=7,
                        cooldown_min=int(cd),
                        unresolved_min=10,
                        opposite_action=str(opp),
                        opposite_flip_min_q_gap_vs_open=0.02,
                        override_start_minute=450,
                        override_window_min=60,
                        override_window_cap=1,
                        expert_caps={a: 5, b: 2},
                        expert_start_minutes={a: int(ea.start_minute), b: 450},
                        expert_override_prob={
                            a: _safe_float(ea.override_prob, 0.0),
                            b: _safe_float(eb.override_prob, 0.0),
                        },
                        expert_entry_min_q={a: 0.0, b: 0.16},
                        expert_flatten_min_q={a: float(ea.flatten_min_q), b: float(eb.flatten_min_q)},
                        expert_flip_min_q={a: float(ea.flip_min_q), b: float(eb.flip_min_q)},
                        expert_weights={a: 1.0, b: float(q2_w)},
                        **common,
                    )
                )
    return pol


def _load_expert_candidates(
    *,
    days: List[str],
    specs: Dict[str, ExpertSpec],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict[str, float]], List[str], Dict[str, str]]:
    rows: List[Dict[str, Any]] = []
    bars_by_day: Dict[str, pd.DataFrame] = {}
    contract_by_expert: Dict[str, Dict[str, float]] = {}
    day_errors: Dict[str, str] = {}

    for day in days:
        try:
            bars = _load_day_bars(day).copy()
            if "dt" not in bars.columns:
                bars = bars.reset_index()
            bars["dt"] = pd.to_datetime(bars["dt"], errors="coerce")
            bars = bars.sort_values("dt").reset_index(drop=True)
            bars_by_day[day] = bars
        except Exception as e:
            day_errors[str(day)] = str(e)
            continue

    for name, spec in specs.items():
        raw = _load_json(spec.config_path)
        params = dict(raw.get("params", raw))
        history_csv = str(params.get("historical_csv_path", "raw_data/schwab/es_minute_history.csv"))
        warmup_days = int(params.get("warmup_days", 60))
        contract_by_expert[name] = {
            "gate": _safe_float(params.get("frontier_virtual_gate", 0.62), 0.62),
            "stop_hi": _safe_float(params.get("frontier_virtual_stop_hi", 10.0), 10.0),
            "target_hi": _safe_float(params.get("frontier_virtual_target_hi", 40.0), 40.0),
            "stop_lo": _safe_float(params.get("frontier_virtual_stop_lo", 4.0), 4.0),
            "target_lo": _safe_float(params.get("frontier_virtual_target_lo", 8.0), 8.0),
        }

        for day in list(bars_by_day.keys()):
            strategy = _build_strategy(params)
            hist = _load_causal_history(history_csv, day, warmup_days)
            strategy.set_historical_context(hist)
            bars = bars_by_day[day]
            try:
                sigs = strategy.process(bars)
            except Exception as e:
                day_errors[str(day)] = str(e)
                continue

            idx_map: Dict[pd.Timestamp, int] = {}
            for i, ts in enumerate(bars["dt"]):
                nt = _norm_ts(ts)
                if nt is not None:
                    idx_map[nt] = int(i)

            for s in sigs:
                md = dict(getattr(s, "metadata", {}) or {})
                action = str(md.get("execution_action", "entry")).strip().lower() or "entry"
                ts = _norm_ts(getattr(s, "entry_ts", None))
                if ts is None:
                    continue
                i = idx_map.get(ts)
                if i is None:
                    continue
                m = int(bars.iloc[int(i)]["dt"].hour) * 60 + int(bars.iloc[int(i)]["dt"].minute)
                q = _safe_float(md.get("quality_score", getattr(s, "pred_proba", 0.5)), 0.5)
                base_prob = _safe_float(getattr(s, "pred_proba", np.nan), np.nan)
                d = 1 if str(getattr(s, "direction", "bull")).strip().lower() == "bull" else -1
                rows.append(
                    {
                        "trading_day": str(day),
                        "expert": str(name),
                        "idx": int(i),
                        "minute_of_day": int(m),
                        "entry_ts": ts,
                        "direction": int(d),
                        "quality": float(q),
                        "base_prob": float(base_prob),
                        "level_name": str(getattr(s, "level_name", "")),
                        "action": str(action),
                    }
                )

    if rows:
        cand = pd.DataFrame(rows).sort_values(
            ["trading_day", "idx", "entry_ts", "quality"],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)
    else:
        cand = pd.DataFrame(
            columns=[
                "trading_day",
                "expert",
                "idx",
                "minute_of_day",
                "entry_ts",
                "direction",
                "quality",
                "base_prob",
                "level_name",
                "action",
            ]
        )
    return cand, bars_by_day, contract_by_expert, sorted(list(bars_by_day.keys())), day_errors


def _simulate_trade_hits(
    *,
    pos: Dict[str, Any],
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if pos is None:
        return None, None
    for j in range(int(start_idx), int(end_idx) + 1):
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
        trade = {
            "entry_idx": int(pos["entry_idx"]),
            "exit_idx": int(j),
            "entry_expert": str(pos["entry_expert"]),
            "lane": str(pos["lane"]),
            "pnl": float(pnl),
            "exit_reason": str(hit),
            "entry_quality": float(pos["entry_quality"]),
            "direction": int(pos["direction"]),
            "level_name": str(pos["level_name"]),
        }
        return trade, None
    return None, pos


def _simulate_policy_day(
    *,
    day: str,
    bars: pd.DataFrame,
    day_cand: pd.DataFrame,
    policy: MixPolicy,
    expert_specs: Dict[str, ExpertSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    b = bars.sort_values("dt").reset_index(drop=True)
    close = pd.to_numeric(b["close"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(b["high"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    low = pd.to_numeric(b["low"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    end_idx = int(len(b) - 1)

    c = day_cand.copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    c = c[c["expert"].isin(policy.experts)].copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Optional regime context for guarded q2 admission.
    dt = pd.to_datetime(b["dt"], errors="coerce")
    hh = pd.to_numeric(dt.dt.hour, errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    mm = pd.to_numeric(dt.dt.minute, errors="coerce").fillna(-1).to_numpy(dtype=np.int32, copy=False)
    minute_arr = hh * 60 + mm
    rth_idx = np.flatnonzero((minute_arr >= 390) & (minute_arr < 780))
    open_idx = int(rth_idx[0]) if rth_idx.size > 0 else 0
    open_px = float(close[open_idx]) if len(close) > 0 else 0.0
    run_hi = np.maximum.accumulate(high[open_idx:]) if len(high) > open_idx else np.array([], dtype=np.float64)
    run_lo = np.minimum.accumulate(low[open_idx:]) if len(low) > open_idx else np.array([], dtype=np.float64)
    run_rng = np.maximum(run_hi - run_lo, 0.25) if run_hi.size > 0 else np.array([], dtype=np.float64)
    drive_frac = np.full(len(close), np.nan, dtype=np.float64)
    if run_rng.size > 0:
        drive_frac[open_idx:] = np.abs(close[open_idx:] - float(open_px)) / run_rng
    first60_mask = (minute_arr >= 390) & (minute_arr < 450)
    first60_range = np.nan
    if np.any(first60_mask):
        first60_range = float(np.max(high[first60_mask]) - np.min(low[first60_mask]))

    q2_name = str(policy.q2_expert)
    anchor_name = str(policy.anchor_expert)
    anchor_q_by_idx: Dict[int, float] = {}
    if policy.regime_gate_enabled and anchor_name:
        tmp_anchor = c[c["expert"].astype(str) == anchor_name]
        if not tmp_anchor.empty:
            grp = tmp_anchor.groupby(tmp_anchor["idx"].astype(int), sort=False)["quality"].max()
            anchor_q_by_idx = {int(k): float(v) for k, v in grp.items()}

    c["score"] = [
        float(_safe_float(q, 0.0) * _safe_float(policy.expert_weights.get(str(e), 1.0), 1.0))
        for q, e in zip(c["quality"].to_numpy(), c["expert"].astype(str).to_numpy())
    ]
    c = c.sort_values(["idx", "score"], ascending=[True, False]).reset_index(drop=True)

    trades: List[Dict[str, Any]] = []
    accepted_rows: List[Dict[str, Any]] = []
    action_counts: Dict[str, int] = {}
    override_window_counts: Dict[int, int] = {}
    unresolved_until: Dict[Tuple[str, int], int] = {}
    expert_entries: Dict[str, int] = {e: 0 for e in policy.experts}

    entries_total = 0
    early_entries = 0
    last_entry_minute = -10_000
    pos: Optional[Dict[str, Any]] = None
    next_scan_idx = 1

    def _entry_contract(quality: float) -> Tuple[str, float, float]:
        lane = "high" if float(quality) >= float(policy.gate) else "low"
        if lane == "high":
            return lane, float(policy.stop_hi), float(policy.target_hi)
        return lane, float(policy.stop_lo), float(policy.target_lo)

    def _try_budget(minute_of_day: int, quality: float, expert: str) -> bool:
        nonlocal entries_total
        within = int(entries_total) < int(policy.k_total)
        if within:
            return True
        p = policy.expert_override_prob.get(expert)
        if p is None:
            p = _safe_float(expert_specs[expert].override_prob, np.nan)
        if not np.isfinite(_safe_float(p, np.nan)):
            return False
        if int(minute_of_day) < int(policy.override_start_minute):
            return False
        if float(quality) < float(p):
            return False
        if int(policy.override_window_min) > 0 and int(policy.override_window_cap) > 0:
            bucket = max(int((int(minute_of_day) - 390) // int(policy.override_window_min)), 0)
            used = int(override_window_counts.get(bucket, 0))
            if used >= int(policy.override_window_cap):
                return False
            override_window_counts[bucket] = used + 1
        return True

    for _, r in c.iterrows():
        i = int(r["idx"])
        m = int(r["minute_of_day"])
        expert = str(r["expert"])
        direction = int(r["direction"])
        quality = float(_safe_float(r["quality"], 0.0))
        level = str(r.get("level_name", ""))
        key = (level, direction)

        # Advance open position to current candidate index first.
        if pos is not None:
            t_hit, pos = _simulate_trade_hits(
                pos=pos,
                close=close,
                high=high,
                low=low,
                start_idx=int(next_scan_idx),
                end_idx=int(i),
            )
            if t_hit is not None:
                t_hit["trading_day"] = str(day)
                trades.append(t_hit)
                next_scan_idx = int(t_hit["exit_idx"]) + 1
            else:
                next_scan_idx = int(i) + 1

        if int(m) < 390 or int(m) >= 780:
            continue
        if int(m) < int(policy.expert_start_minutes.get(expert, expert_specs[expert].start_minute)):
            continue
        end_m = int(policy.expert_end_minutes.get(expert, -1))
        if end_m > 0 and int(m) >= int(end_m):
            continue
        if policy.regime_gate_enabled and expert == q2_name:
            if int(m) < int(policy.regime_q2_start_minute):
                continue
            if np.isfinite(float(policy.regime_min_first60_range)) and float(policy.regime_min_first60_range) > 0.0:
                if not np.isfinite(float(first60_range)) or float(first60_range) < float(policy.regime_min_first60_range):
                    continue
            if np.isfinite(float(policy.regime_min_drive_frac)) and float(policy.regime_min_drive_frac) > 0.0:
                if i < 0 or i >= len(drive_frac) or not np.isfinite(float(drive_frac[i])):
                    continue
                if float(drive_frac[i]) < float(policy.regime_min_drive_frac):
                    continue
            if np.isfinite(float(policy.regime_q2_quality_floor)) and float(policy.regime_q2_quality_floor) > 0.0:
                if float(quality) < float(policy.regime_q2_quality_floor):
                    continue
            adv_min = float(policy.regime_q2_adv_vs_anchor)
            if np.isfinite(adv_min) and adv_min > 0.0:
                anchor_q = _safe_float(anchor_q_by_idx.get(int(i), np.nan), np.nan)
                if np.isfinite(anchor_q) and float(quality) < float(anchor_q + adv_min):
                    continue

        # If position is open, opposite signals can flatten even when entry-gating
        # constraints (budget/cooldown/caps) are exhausted.
        if pos is not None and int(direction) != int(pos["direction"]):
            opp_mode = str(policy.opposite_action).lower()
            if opp_mode != "ignore":
                flatten_min_q = float(policy.expert_flatten_min_q.get(expert, expert_specs[expert].flatten_min_q))
                if float(quality) >= float(flatten_min_q):
                    pnl = float((float(close[i]) - float(pos["entry_price"])) * float(pos["direction"]))
                    trades.append(
                        {
                            "trading_day": str(day),
                            "entry_idx": int(pos["entry_idx"]),
                            "exit_idx": int(i),
                            "entry_expert": str(pos["entry_expert"]),
                            "lane": str(pos["lane"]),
                            "pnl": float(pnl),
                            "exit_reason": "flatten_signal",
                            "entry_quality": float(pos["entry_quality"]),
                            "direction": int(pos["direction"]),
                            "level_name": str(pos["level_name"]),
                        }
                    )
                    pos = None
                    action_counts["flatten"] = int(action_counts.get("flatten", 0)) + 1

                    if opp_mode == "flip_or_flatten":
                        flip_min_q = float(policy.expert_flip_min_q.get(expert, expert_specs[expert].flip_min_q))
                        if float(quality) >= float(flip_min_q) and (
                            float(quality) - float(flatten_min_q) >= float(policy.opposite_flip_min_q_gap_vs_open)
                        ):
                            can_entry = True
                            if (int(m) - int(last_entry_minute)) < int(policy.cooldown_min):
                                can_entry = False
                            if int(expert_entries.get(expert, 0)) >= int(
                                policy.expert_caps.get(expert, expert_specs[expert].cap_per_day)
                            ):
                                can_entry = False
                            if int(unresolved_until.get(key, -10_000)) >= int(m):
                                can_entry = False
                            if (
                                int(policy.early_cap) > 0
                                and int(m) < int(policy.early_end_minute)
                                and int(early_entries) >= int(policy.early_cap)
                            ):
                                p = policy.expert_override_prob.get(expert)
                                if p is None:
                                    p = _safe_float(expert_specs[expert].override_prob, np.nan)
                                if not np.isfinite(_safe_float(p, np.nan)) or float(quality) < float(p):
                                    can_entry = False
                            if float(quality) < float(policy.expert_entry_min_q.get(expert, 0.0)):
                                can_entry = False
                            if can_entry and _try_budget(m, quality, expert):
                                lane, stop_pts, target_pts = _entry_contract(quality)
                                ep = float(close[i])
                                if direction > 0:
                                    stop_px = ep - float(stop_pts)
                                    target_px = ep + float(target_pts)
                                else:
                                    stop_px = ep + float(stop_pts)
                                    target_px = ep - float(target_pts)
                                pos = {
                                    "entry_idx": int(i),
                                    "entry_price": float(ep),
                                    "direction": int(direction),
                                    "stop_px": float(stop_px),
                                    "target_px": float(target_px),
                                    "entry_expert": str(expert),
                                    "entry_quality": float(quality),
                                    "lane": str(lane),
                                    "level_name": str(level),
                                }
                                entries_total += 1
                                expert_entries[expert] = int(expert_entries.get(expert, 0)) + 1
                                last_entry_minute = int(m)
                                if int(m) < int(policy.early_end_minute):
                                    early_entries += 1
                                unresolved_until[key] = int(m) + int(policy.unresolved_min)
                                action_counts["flip"] = int(action_counts.get("flip", 0)) + 1
                                accepted_rows.append(
                                    {
                                        "trading_day": str(day),
                                        "idx": int(i),
                                        "minute_of_day": int(m),
                                        "expert": str(expert),
                                        "quality": float(quality),
                                        "direction": int(direction),
                                        "action": "flip",
                                    }
                                )
                # Opposite processed; never fall through to plain entry on same row.
                continue

        if pos is None:
            if (int(m) - int(last_entry_minute)) < int(policy.cooldown_min):
                continue
            if int(expert_entries.get(expert, 0)) >= int(policy.expert_caps.get(expert, expert_specs[expert].cap_per_day)):
                continue
            if int(unresolved_until.get(key, -10_000)) >= int(m):
                continue
            if int(policy.early_cap) > 0 and int(m) < int(policy.early_end_minute) and int(early_entries) >= int(policy.early_cap):
                p = policy.expert_override_prob.get(expert)
                if p is None:
                    p = _safe_float(expert_specs[expert].override_prob, np.nan)
                if not np.isfinite(_safe_float(p, np.nan)) or float(quality) < float(p):
                    continue
            if float(quality) < float(policy.expert_entry_min_q.get(expert, 0.0)):
                continue
            if not _try_budget(m, quality, expert):
                continue
            lane, stop_pts, target_pts = _entry_contract(quality)
            ep = float(close[i])
            if direction > 0:
                stop_px = ep - float(stop_pts)
                target_px = ep + float(target_pts)
            else:
                stop_px = ep + float(stop_pts)
                target_px = ep - float(target_pts)
            pos = {
                "entry_idx": int(i),
                "entry_price": float(ep),
                "direction": int(direction),
                "stop_px": float(stop_px),
                "target_px": float(target_px),
                "entry_expert": str(expert),
                "entry_quality": float(quality),
                "lane": str(lane),
                "level_name": str(level),
            }
            entries_total += 1
            expert_entries[expert] = int(expert_entries.get(expert, 0)) + 1
            last_entry_minute = int(m)
            if int(m) < int(policy.early_end_minute):
                early_entries += 1
            unresolved_until[key] = int(m) + int(policy.unresolved_min)
            action_counts["entry"] = int(action_counts.get("entry", 0)) + 1
            accepted_rows.append(
                {
                    "trading_day": str(day),
                    "idx": int(i),
                    "minute_of_day": int(m),
                    "expert": str(expert),
                    "quality": float(quality),
                    "direction": int(direction),
                    "action": "entry",
                }
            )
            continue
        # Position open and same direction -> ignore.

    if pos is not None:
        t_hit, pos = _simulate_trade_hits(
            pos=pos,
            close=close,
            high=high,
            low=low,
            start_idx=int(next_scan_idx),
            end_idx=end_idx,
        )
        if t_hit is not None:
            t_hit["trading_day"] = str(day)
            trades.append(t_hit)
            pos = None
        if pos is not None:
            pnl = float((float(close[end_idx]) - float(pos["entry_price"])) * float(pos["direction"]))
            trades.append(
                {
                    "trading_day": str(day),
                    "entry_idx": int(pos["entry_idx"]),
                    "exit_idx": int(end_idx),
                    "entry_expert": str(pos["entry_expert"]),
                    "lane": str(pos["lane"]),
                    "pnl": float(pnl),
                    "exit_reason": "timed",
                    "entry_quality": float(pos["entry_quality"]),
                    "direction": int(pos["direction"]),
                    "level_name": str(pos["level_name"]),
                }
            )
    return pd.DataFrame(trades), pd.DataFrame(accepted_rows), action_counts


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
            "zero_target_day_rate": 1.0,
        }
    gp = float(trades["pnl"].clip(lower=0.0).sum())
    gl = float((-trades["pnl"].clip(upper=0.0)).sum())
    exits = trades["exit_reason"].astype(str)
    hit_target_days = set(
        trades.loc[exits == "target", "trading_day"].astype(str).unique().tolist()
    )
    zero_target_rate = float(np.mean([d not in hit_target_days for d in day_ids])) if day_ids else None
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
        "manual_exit_rate": float(exits.isin(["flatten_signal"]).mean()),
        "zero_target_day_rate": zero_target_rate,
    }


def _distribution(entries: pd.DataFrame, day_ids: List[str]) -> Dict[str, Any]:
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
    w_open = (m >= 390) & (m < 450)
    w_post = (m >= 450) & (m < 510)
    w_late = (m >= 510) & (m < 630)
    w_aft = (m >= 630) & (m < 780)
    n = float(len(e))
    by_day_post = e.loc[w_post, "trading_day"].astype(str).value_counts().to_dict()
    by_day_after510 = e.loc[(m >= 510) & (m < 780), "trading_day"].astype(str).value_counts().to_dict()
    zero_post = float(np.mean([int(by_day_post.get(d, 0)) == 0 for d in day_ids])) if day_ids else None
    zero_after = float(np.mean([int(by_day_after510.get(d, 0)) == 0 for d in day_ids])) if day_ids else None
    return {
        "open_60_share": float(w_open.sum() / n),
        "post_open_60_share": float(w_post.sum() / n),
        "late_morning_120_share": float(w_late.sum() / n),
        "afternoon_150_share": float(w_aft.sum() / n),
        "zero_post_open_60_day_rate": zero_post,
        "zero_after_510_day_rate": zero_after,
    }


def _expert_breakdown(trades: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if trades.empty:
        return out
    for expert, g in trades.groupby(trades["entry_expert"].astype(str), sort=False):
        exits = g["exit_reason"].astype(str)
        gp = float(g["pnl"].clip(lower=0.0).sum())
        gl = float((-g["pnl"].clip(upper=0.0)).sum())
        out[str(expert)] = {
            "n_trades": int(len(g)),
            "mean_pnl_per_trade": float(g["pnl"].mean()),
            "win_rate": float((g["pnl"] > 0.0).mean()),
            "profit_factor": (gp / gl) if gl > 1e-9 else None,
            "target_hit_rate": float((exits == "target").mean()),
            "stop_hit_rate": float(exits.str.startswith("stop").mean()),
            "timed_exit_rate": float((exits == "timed").mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", required=True, help="Comma-separated SQL trading days.")
    ap.add_argument(
        "--experts",
        default=(
            "live=configs/realtime/strategies/reversal_live_selected.jsonc,"
            "q2=configs/realtime/strategies/reversal_frontier_qtwohead_runtime_v2_cal_d_k6_ll99_v1.jsonc"
        ),
        help="Comma-separated expert specs: name=path",
    )
    ap.add_argument(
        "--policy-mode",
        choices=["full", "guarded", "split", "blend"],
        default="guarded",
        help="Policy set to evaluate: full legacy sweep or guarded-mixture sweep.",
    )
    ap.add_argument("--output-json", default="sandbox/results/sql_mixture_router_v1.json")
    ap.add_argument("--actions-csv", default="sandbox/results/sql_mixture_router_actions_v1.csv")
    args = ap.parse_args()

    t0 = time.time()
    days = _parse_csv(args.days)
    experts_raw = _parse_experts(args.experts)
    specs = _build_default_expert_specs(experts_raw)

    print("loading expert candidates ...")
    cand, bars_by_day, contracts, loaded_days, day_errors = _load_expert_candidates(days=days, specs=specs)
    if cand.empty:
        raise RuntimeError("No expert signals emitted for requested days.")

    # Use first expert contract as baseline execution contract for fair comparison.
    base_contract = next(iter(contracts.values()))
    if str(args.policy_mode) == "full":
        policies = _policy_grid(specs, base_contract=base_contract)
    elif str(args.policy_mode) == "split":
        policies = _split_policy_grid(specs, base_contract=base_contract)
    elif str(args.policy_mode) == "blend":
        policies = _blend_policy_grid(specs, base_contract=base_contract)
    else:
        policies = _guarded_policy_grid(specs, base_contract=base_contract)

    rows_out: List[Dict[str, Any]] = []
    all_actions: List[pd.DataFrame] = []
    for pol in policies:
        trades_parts: List[pd.DataFrame] = []
        entry_parts: List[pd.DataFrame] = []
        action_totals: Dict[str, int] = {}
        base_weights: Dict[str, float] = {
            str(e): float(_safe_float(pol.expert_weights.get(str(e), 1.0), 1.0))
            for e in pol.experts
        }
        runtime_weights: Dict[str, float] = dict(base_weights)
        ewma_by_expert: Dict[str, float] = {str(e): 0.0 for e in pol.experts}

        for day in loaded_days:
            b = bars_by_day.get(day)
            if b is None or b.empty:
                continue
            d_cand = cand[cand["trading_day"].astype(str) == str(day)]
            pol.expert_weights = dict(runtime_weights)
            tdf, entries, a_counts = _simulate_policy_day(
                day=day,
                bars=b,
                day_cand=d_cand,
                policy=pol,
                expert_specs=specs,
            )
            if not tdf.empty:
                trades_parts.append(tdf)
            if not entries.empty:
                entry_parts.append(entries)
            for k, v in a_counts.items():
                action_totals[k] = int(action_totals.get(k, 0)) + int(v)

            if bool(pol.online_day_weight_enabled):
                alpha = min(max(float(_safe_float(pol.online_alpha, 0.20)), 0.0), 1.0)
                beta = max(float(_safe_float(pol.online_beta, 0.05)), 0.0)
                w_min = max(float(_safe_float(pol.online_weight_min, 0.50)), 0.01)
                w_max = max(float(_safe_float(pol.online_weight_max, 2.00)), w_min)
                day_by_exp = (
                    tdf.groupby(tdf["entry_expert"].astype(str))["pnl"].mean().to_dict()
                    if not tdf.empty and "entry_expert" in tdf.columns
                    else {}
                )
                for e in pol.experts:
                    ee = str(e)
                    signal = float(_safe_float(day_by_exp.get(ee, 0.0), 0.0))
                    prev = float(_safe_float(ewma_by_expert.get(ee, 0.0), 0.0))
                    ewma_by_expert[ee] = (1.0 - alpha) * prev + alpha * signal
                anchor = str(pol.anchor_expert) if str(pol.anchor_expert) in runtime_weights else str(pol.experts[0])
                for e in pol.experts:
                    ee = str(e)
                    if ee == anchor:
                        runtime_weights[ee] = float(base_weights.get(ee, 1.0))
                        continue
                    base_w = float(base_weights.get(ee, 1.0))
                    x = float(np.clip(ewma_by_expert.get(ee, 0.0), -50.0, 50.0))
                    mult = float(np.exp(beta * x))
                    runtime_weights[ee] = float(base_w * np.clip(mult, w_min, w_max))

        trades = pd.concat(trades_parts, axis=0, ignore_index=True) if trades_parts else pd.DataFrame()
        entries = pd.concat(entry_parts, axis=0, ignore_index=True) if entry_parts else pd.DataFrame()
        metrics = _metrics(trades, loaded_days)
        dist = _distribution(entries, loaded_days)
        by_expert = _expert_breakdown(trades)
        rows_out.append(
            {
                "policy": asdict(pol),
                "metrics": metrics,
                "distribution": dist,
                "action_counts": action_totals,
                "expert_breakdown": by_expert,
                "final_runtime_weights": {k: float(v) for k, v in runtime_weights.items()},
                "ewma_by_expert": {k: float(v) for k, v in ewma_by_expert.items()},
            }
        )
        if not entries.empty:
            x = entries.copy()
            x["policy_name"] = str(pol.name)
            all_actions.append(x)

    def _rank_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
        m = r.get("metrics", {})
        pf = _safe_float(m.get("profit_factor"), -1.0)
        sh = _safe_float(m.get("annualized_daily_sharpe"), -1.0)
        md = _safe_float(m.get("mean_pnl_per_day"), -1e9)
        return (pf, sh, md)

    rows_out = sorted(rows_out, key=_rank_key, reverse=True)
    baseline = next((r for r in rows_out if str(r.get("policy", {}).get("name", "")) == "live_solo_flatten_ref"), None)
    if baseline is None:
        baseline = next((r for r in rows_out if str(r.get("policy", {}).get("name", "")).startswith("live_")), None)
    baseline_pf = _safe_float((baseline or {}).get("metrics", {}).get("profit_factor"), np.nan)
    better_than_baseline = [
        r for r in rows_out if np.isfinite(baseline_pf) and _safe_float(r["metrics"].get("profit_factor"), -1.0) > baseline_pf
    ]

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": float(time.time() - t0),
        "days_requested": days,
        "days_loaded": loaded_days,
        "days_failed": day_errors,
        "experts": {k: asdict(v) for k, v in specs.items()},
        "baseline_pf_ref": baseline_pf if np.isfinite(baseline_pf) else None,
        "n_candidates": int(len(cand)),
        "candidates_per_day": (
            cand.groupby(cand["trading_day"].astype(str)).size().sort_index().to_dict()
            if not cand.empty
            else {}
        ),
        "results": rows_out,
        "better_than_live_solo_by_pf": better_than_baseline[:40],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(str(out_path))

    if str(args.actions_csv).strip():
        act_path = Path(args.actions_csv)
        act_path.parent.mkdir(parents=True, exist_ok=True)
        if all_actions:
            pd.concat(all_actions, axis=0, ignore_index=True).to_csv(act_path, index=False)
        else:
            pd.DataFrame(columns=["policy_name", "trading_day", "idx", "minute_of_day", "expert", "quality", "direction", "action"]).to_csv(act_path, index=False)
        print(str(act_path))


if __name__ == "__main__":
    main()
