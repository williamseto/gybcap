#!/usr/bin/env python
"""Gate runtime-policy parity against research metrics for the same contract.

This is intended as a pre-live QA check when porting research policies into
runtime configs. It compares selected metrics from:
- research policy output (frontier_policy_hypotheses style JSON), and
- runtime reconciliation output (apples_to_apples JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


def _iter_research_records(obj: Dict) -> Iterable[Dict]:
    for key in ("stage1_ranked", "stage2_ranked", "overall_top50", "best_pf_feasible_top20", "best_pf_under10_top20"):
        arr = obj.get(key, [])
        if isinstance(arr, list):
            for rec in arr:
                if isinstance(rec, dict):
                    yield rec


def _find_research_policy(obj: Dict, policy_name: str) -> Optional[Dict]:
    target = str(policy_name).strip()
    for rec in _iter_research_records(obj):
        if str(rec.get("policy_name", "")).strip() == target:
            return rec
    return None


def _safe(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-json", required=True)
    ap.add_argument("--policy-name", required=True)
    ap.add_argument("--runtime-json", required=True)
    ap.add_argument("--runtime-config-key", required=True)
    ap.add_argument("--contract", default="hi_10_40__lo_4_8__g_0.62")
    ap.add_argument("--tol-pf", type=float, default=0.15)
    ap.add_argument("--tol-no-tp", type=float, default=0.03)
    ap.add_argument("--tol-trades-per-day", type=float, default=0.50)
    ap.add_argument("--tol-sharpe", type=float, default=1.0)
    args = ap.parse_args()

    research = json.loads(Path(args.research_json).read_text())
    runtime = json.loads(Path(args.runtime_json).read_text())

    rec = _find_research_policy(research, args.policy_name)
    if rec is None:
        raise SystemExit(f"policy_name not found in research json: {args.policy_name}")

    contracts = runtime.get("contracts", {})
    if args.contract not in contracts:
        raise SystemExit(f"contract not found in runtime json: {args.contract}")
    full = contracts[args.contract].get("full_history", {})
    if args.runtime_config_key not in full:
        raise SystemExit(
            f"runtime config key not found in full_history: {args.runtime_config_key}"
        )
    rt = full[args.runtime_config_key]

    # Research schema: rec["exec"], rec["label"].
    # Runtime schema: rt["exec"], rt["label"].
    rp_pf = _safe(rec.get("exec", {}).get("profit_factor"))
    rp_sh = _safe(rec.get("exec", {}).get("annualized_daily_sharpe"))
    rp_tpd = _safe(rec.get("exec", {}).get("mean_trades_per_day"))
    rp_no_tp = _safe(rec.get("label", {}).get("rth_no_tp_day_rate"))

    rt_pf = _safe(rt.get("exec", {}).get("profit_factor"))
    rt_sh = _safe(rt.get("exec", {}).get("annualized_daily_sharpe"))
    rt_tpd = _safe(rt.get("exec", {}).get("mean_trades_per_day"))
    rt_no_tp = _safe(rt.get("label", {}).get("rth_no_tp_day_rate"))

    pairs = [
        ("profit_factor", rp_pf, rt_pf, float(args.tol_pf)),
        ("sharpe", rp_sh, rt_sh, float(args.tol_sharpe)),
        ("trades_per_day", rp_tpd, rt_tpd, float(args.tol_trades_per_day)),
        ("no_tp_day_rate", rp_no_tp, rt_no_tp, float(args.tol_no_tp)),
    ]

    failed = []
    print("metric,research,runtime,abs_delta,tolerance,status")
    for name, a, b, tol in pairs:
        if a is None or b is None:
            print(f"{name},{a},{b},nan,{tol},missing")
            failed.append(name)
            continue
        d = abs(float(a) - float(b))
        status = "pass" if d <= tol else "fail"
        print(f"{name},{a:.6f},{b:.6f},{d:.6f},{tol:.6f},{status}")
        if status != "pass":
            failed.append(name)

    if failed:
        raise SystemExit(f"parity_gate_failed: {', '.join(failed)}")
    print("parity_gate_passed")


if __name__ == "__main__":
    main()
