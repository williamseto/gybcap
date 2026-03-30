#!/usr/bin/env python
"""Shared utilities for alpha research loop scripts."""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """UTC timestamp in ISO format."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_git_commit(cwd: str | None = None) -> str:
    """Best-effort git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_research_index(index_path: str, entry: dict) -> None:
    data = load_json(index_path, {"runs": []})
    if "runs" not in data or not isinstance(data["runs"], list):
        data = {"runs": []}
    data["runs"].append(entry)
    save_json(index_path, data)


def append_memory_note(memory_path: str, note_block: str) -> None:
    """Append an experiment note under a dedicated auto section."""
    header = "## Research Loop Notes (Auto)"
    if os.path.exists(memory_path):
        with open(memory_path, "r") as f:
            text = f.read()
    else:
        text = "# Project Memory\n\n"

    if header not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n{header}\n"
    if not text.endswith("\n"):
        text += "\n"
    text += f"\n{note_block.strip()}\n"

    os.makedirs(os.path.dirname(memory_path), exist_ok=True)
    with open(memory_path, "w") as f:
        f.write(text)


def rss_mb() -> float:
    """Process RSS in MB (Linux)."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return -1.0


def safe_remove_path(path: str, allowed_roots: list[str] | None = None) -> bool:
    """Remove file/dir only if within allowed roots."""
    p = Path(path).resolve()
    roots = [Path(r).resolve() for r in (allowed_roots or [])]
    if roots and not any(str(p).startswith(str(r)) for r in roots):
        return False

    if not p.exists():
        return True

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    return True
