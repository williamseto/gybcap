#!/usr/bin/env python
"""Log research run outcomes into project memory and research index.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/log_research_run.py \
      --run-json sandbox/results/execution_param_sweep.json
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import research_utils as ru


def _infer_decision(payload: dict[str, Any]) -> str:
    if payload.get("overall_verdict") == "EDGE_FOUND":
        return "promote"

    sweep_results = payload.get("sweep_results")
    if isinstance(sweep_results, dict):
        decisions = [v.get("decision") for v in sweep_results.values() if isinstance(v, dict)]
        if "promote" in decisions:
            return "promote"
        if "park" in decisions:
            return "park"
        if decisions:
            return "kill"

    return "unknown"


def _extract_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "overall_verdict" in payload:
        summary["overall_verdict"] = payload.get("overall_verdict")
        summary["positive_slices"] = payload.get("positive_slices", [])

    if "sweep_results" in payload and isinstance(payload["sweep_results"], dict):
        best = {}
        for name, result in payload["sweep_results"].items():
            if not isinstance(result, dict):
                continue
            cfg = result.get("best_config")
            if isinstance(cfg, dict):
                best[name] = {
                    "decision": result.get("decision", "unknown"),
                    "mean_pnl_net": cfg.get("mean_pnl_net"),
                    "win_rate": cfg.get("win_rate"),
                    "payoff_ratio": cfg.get("payoff_ratio"),
                    "n_trades": cfg.get("n_trades"),
                    "threshold": cfg.get("threshold"),
                    "stop_pts": cfg.get("stop_pts"),
                    "target_pts": cfg.get("target_pts"),
                    "max_bars": cfg.get("max_bars"),
                    "gate_pass": cfg.get("gate_pass"),
                }
        summary["sweep_best"] = best

    return summary


def _build_memory_note(
    run_json: str,
    decision: str,
    summary: dict[str, Any],
    notes: str,
) -> str:
    lines = [
        f"### {ru.now_iso()} | {decision.upper()}",
        f"- run_json: `{run_json}`",
    ]

    if "overall_verdict" in summary:
        lines.append(f"- overall_verdict: `{summary['overall_verdict']}`")
        lines.append(f"- positive_slices: {summary.get('positive_slices', [])}")

    sweep_best = summary.get("sweep_best", {})
    if sweep_best:
        for model_name, row in sweep_best.items():
            lines.append(
                "- "
                f"{model_name}: decision={row.get('decision')} "
                f"gate_pass={row.get('gate_pass')} "
                f"E[PnL_net]={row.get('mean_pnl_net')} WR={row.get('win_rate')} "
                f"RR={row.get('payoff_ratio')} trades={row.get('n_trades')} "
                f"thr={row.get('threshold')} stop={row.get('stop_pts')} "
                f"target={row.get('target_pts')} max_bars={row.get('max_bars')}"
            )

    if notes.strip():
        lines.append(f"- notes: {notes.strip()}")

    return "\n".join(lines)


def _should_cleanup(cleanup_on: str, decision: str) -> bool:
    if cleanup_on == "always":
        return True
    if cleanup_on == "kill" and decision == "kill":
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Log research run outcomes")
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--decision", default="", choices=["", "promote", "park", "kill", "unknown"])
    parser.add_argument("--notes", default="")
    parser.add_argument("--index-json", default=".codex/research_runs.json")
    parser.add_argument("--memory-md", default=".codex/MEMORY.md")
    parser.add_argument("--cleanup-on", default="none", choices=["none", "kill", "always"])
    parser.add_argument("--cleanup-path", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = ru.load_json(args.run_json, {})
    if not payload:
        raise ValueError(f"Failed to load run JSON or empty payload: {args.run_json}")

    decision = args.decision or _infer_decision(payload)
    summary = _extract_summary(payload)

    entry = {
        "ts": ru.now_iso(),
        "git_commit": ru.get_git_commit(),
        "decision": decision,
        "run_json": args.run_json,
        "summary": summary,
    }

    note_block = _build_memory_note(
        run_json=args.run_json,
        decision=decision,
        summary=summary,
        notes=args.notes,
    )

    print(f"Decision: {decision}")
    print(f"Index: {args.index_json}")
    print(f"Memory: {args.memory_md}")

    if not args.dry_run:
        ru.append_research_index(args.index_json, entry)
        ru.append_memory_note(args.memory_md, note_block)

    if _should_cleanup(args.cleanup_on, decision) and args.cleanup_path:
        print("Cleanup requested.")
        allowed = [os.getcwd(), "/tmp"]
        for p in args.cleanup_path:
            ok = True
            if not args.dry_run:
                ok = ru.safe_remove_path(p, allowed_roots=allowed)
            print(f"  remove {p}: {'OK' if ok else 'SKIPPED'}")

    print("Done.")


if __name__ == "__main__":
    main()
