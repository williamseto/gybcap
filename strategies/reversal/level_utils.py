"""Shared helpers for canonical level grouping in reversal router logic."""

from __future__ import annotations

from typing import Any


def level_group(level_name: Any) -> str:
    """Map concrete level names to stable family buckets used by inertia logic."""
    name = str(level_name).strip().lower()
    if name.startswith("vwap"):
        return "vwap"
    if name.startswith("rth_"):
        return "rth"
    if name.startswith("ovn_"):
        return "ovn"
    if name.startswith("ib_"):
        return "ib"
    if name.startswith("prev_"):
        return "prev"
    return name
