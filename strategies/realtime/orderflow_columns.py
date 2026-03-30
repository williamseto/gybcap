"""Helpers for canonical orderflow column names in realtime paths."""

from __future__ import annotations

import pandas as pd


def normalize_orderflow_columns(
    df: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    """Normalize to canonical `bidvolume`/`askvolume`.

    Legacy stream semantics:
    - `buys`  == ask-side volume
    - `sells` == bid-side volume
    """
    out = df.copy() if copy else df

    if "askvolume" not in out.columns and "buys" in out.columns:
        out["askvolume"] = pd.to_numeric(out["buys"], errors="coerce").fillna(0.0)
    if "bidvolume" not in out.columns and "sells" in out.columns:
        out["bidvolume"] = pd.to_numeric(out["sells"], errors="coerce").fillna(0.0)

    return out
