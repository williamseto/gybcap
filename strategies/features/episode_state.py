"""Causal intraday episode/reclaim-state features for repeated level interactions."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable

import numpy as np
import pandas as pd


BASE_EPISODE_STATE_FEATURES = [
    "level_side_touch_count_day",
    "bars_since_last_level_side_touch",
    "level_side_touch_density_30",
    "level_side_flip_count_day",
    "level_reclaim_count_day",
    "bars_since_last_reclaim",
    "last_reclaim_dwell_bars",
    "last_reclaim_penetration_pts",
    "is_reclaim_bar",
]

EXTENDED_EPISODE_STATE_FEATURES = [
    "level_touch_count_any_side_day",
    "bars_since_last_level_touch_any_side",
    "level_touch_density_any_30",
    "current_run_bars",
    "current_run_max_penetration_pts",
]

EPISODE_STATE_FEATURES = BASE_EPISODE_STATE_FEATURES + EXTENDED_EPISODE_STATE_FEATURES


def _is_valid_level(v: object) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return len(v) > 0
    if isinstance(v, float):
        return not np.isnan(v)
    return True


def compute_episode_state_features(
    df: pd.DataFrame,
    level_col: str = "nearest_level_name",
    side_col: str = "side",
    trading_day_col: str = "trading_day",
    density_window_bars: int = 30,
    close_col: str = "close",
    level_price_col: str = "nearest_level_price",
) -> pd.DataFrame:
    """Compute causal state features around repeated level tests.

    The base touch-count features use prior bars only. Reclaim features are
    available at the current bar close and only depend on the current bar plus
    prior bars for the same day/level.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)
    out["level_side_touch_count_day"] = np.zeros(n, dtype=np.float32)
    out["bars_since_last_level_side_touch"] = np.full(n, 9999.0, dtype=np.float32)
    out["level_side_touch_density_30"] = np.zeros(n, dtype=np.float32)
    out["level_side_flip_count_day"] = np.zeros(n, dtype=np.float32)
    out["level_reclaim_count_day"] = np.zeros(n, dtype=np.float32)
    out["bars_since_last_reclaim"] = np.full(n, 9999.0, dtype=np.float32)
    out["last_reclaim_dwell_bars"] = np.zeros(n, dtype=np.float32)
    out["last_reclaim_penetration_pts"] = np.zeros(n, dtype=np.float32)
    out["is_reclaim_bar"] = np.zeros(n, dtype=np.float32)
    out["level_touch_count_any_side_day"] = np.zeros(n, dtype=np.float32)
    out["bars_since_last_level_touch_any_side"] = np.full(n, 9999.0, dtype=np.float32)
    out["level_touch_density_any_30"] = np.zeros(n, dtype=np.float32)
    out["current_run_bars"] = np.zeros(n, dtype=np.float32)
    out["current_run_max_penetration_pts"] = np.zeros(n, dtype=np.float32)

    if n == 0:
        return out

    if level_col not in df.columns or side_col not in df.columns:
        return out

    levels = df[level_col].values
    sides = pd.Series(df[side_col]).fillna(0).values.astype(np.int8)
    if close_col in df.columns:
        close_arr = pd.to_numeric(df[close_col], errors="coerce").values.astype(np.float64)
    else:
        close_arr = np.full(n, np.nan, dtype=np.float64)
    if level_price_col in df.columns:
        level_price_arr = pd.to_numeric(df[level_price_col], errors="coerce").values.astype(np.float64)
    else:
        level_price_arr = np.full(n, np.nan, dtype=np.float64)
    if trading_day_col in df.columns:
        days = df[trading_day_col].astype(str).values
    else:
        days = np.array(["_all"] * n, dtype=object)

    touch_count: Dict[tuple, int] = defaultdict(int)
    last_touch_idx: Dict[tuple, int] = {}
    recent_touch_idx: Dict[tuple, deque] = defaultdict(deque)
    touch_count_any: Dict[tuple, int] = defaultdict(int)
    last_touch_any_idx: Dict[tuple, int] = {}
    recent_touch_any_idx: Dict[tuple, deque] = defaultdict(deque)

    last_side_by_level: Dict[tuple, int] = {}
    flip_count_by_level: Dict[tuple, int] = defaultdict(int)
    active_run_by_level: Dict[tuple, tuple] = {}
    reclaim_count_by_key: Dict[tuple, int] = defaultdict(int)
    last_reclaim_idx_by_key: Dict[tuple, int] = {}
    last_reclaim_dwell_by_key: Dict[tuple, float] = defaultdict(float)
    last_reclaim_pen_by_key: Dict[tuple, float] = defaultdict(float)

    window = max(int(density_window_bars), 1)

    for i in range(n):
        side = int(sides[i])
        level = levels[i]
        if side == 0 or not _is_valid_level(level):
            continue

        day = str(days[i])
        level_name = str(level)
        key = (day, level_name, side)
        level_key = (day, level_name)
        level_close = float(close_arr[i]) if i < len(close_arr) else np.nan
        level_price = float(level_price_arr[i]) if i < len(level_price_arr) else np.nan
        penetration = 0.0
        if np.isfinite(level_close) and np.isfinite(level_price):
            penetration = float(abs(level_close - level_price))

        prev_count = touch_count[key]
        out.iat[i, 0] = float(prev_count)

        prev_idx = last_touch_idx.get(key)
        if prev_idx is not None:
            out.iat[i, 1] = float(i - prev_idx)

        dq = recent_touch_idx[key]
        while dq and dq[0] < (i - window):
            dq.popleft()
        out.iat[i, 2] = float(len(dq))

        prev_any_count = touch_count_any[level_key]
        out.iat[i, 9] = float(prev_any_count)
        prev_any_idx = last_touch_any_idx.get(level_key)
        if prev_any_idx is not None:
            out.iat[i, 10] = float(i - prev_any_idx)
        dq_any = recent_touch_any_idx[level_key]
        while dq_any and dq_any[0] < (i - window):
            dq_any.popleft()
        out.iat[i, 11] = float(len(dq_any))

        prev_side = last_side_by_level.get(level_key)
        flip_count = flip_count_by_level[level_key]
        if prev_side is not None and prev_side != side:
            flip_count += 1
            flip_count_by_level[level_key] = flip_count
        out.iat[i, 3] = float(flip_count)

        reclaim_count = reclaim_count_by_key[key]
        out.iat[i, 4] = float(reclaim_count)
        prev_reclaim_idx = last_reclaim_idx_by_key.get(key)
        if prev_reclaim_idx is not None:
            out.iat[i, 5] = float(i - prev_reclaim_idx)
        out.iat[i, 6] = float(last_reclaim_dwell_by_key[key])
        out.iat[i, 7] = float(last_reclaim_pen_by_key[key])

        active_run = active_run_by_level.get(level_key)
        if active_run is None:
            active_run_by_level[level_key] = (side, i, penetration)
            out.iat[i, 12] = 0.0
            out.iat[i, 13] = float(penetration)
        else:
            run_side, run_start, run_max_pen = active_run
            if int(run_side) == side:
                next_pen = max(float(run_max_pen), float(penetration))
                active_run_by_level[level_key] = (run_side, run_start, next_pen)
                out.iat[i, 12] = float(i - int(run_start))
                out.iat[i, 13] = float(next_pen)
            else:
                # A side flip closes the prior run and creates a reclaim into
                # the current side at the current bar close.
                reclaim_count += 1
                reclaim_count_by_key[key] = reclaim_count
                last_reclaim_idx_by_key[key] = i
                last_reclaim_dwell_by_key[key] = float(i - int(run_start))
                last_reclaim_pen_by_key[key] = float(run_max_pen)
                out.iat[i, 4] = float(reclaim_count)
                out.iat[i, 5] = 0.0
                out.iat[i, 6] = float(i - int(run_start))
                out.iat[i, 7] = float(run_max_pen)
                out.iat[i, 8] = 1.0
                active_run_by_level[level_key] = (side, i, penetration)
                out.iat[i, 12] = 0.0
                out.iat[i, 13] = float(penetration)

        touch_count[key] = prev_count + 1
        last_touch_idx[key] = i
        dq.append(i)
        touch_count_any[level_key] = prev_any_count + 1
        last_touch_any_idx[level_key] = i
        dq_any.append(i)
        last_side_by_level[level_key] = side

    return out
