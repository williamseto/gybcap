"""Shared helpers for nearest-level detection and side assignment."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def compute_nearest_level_arrays(
    close_arr: np.ndarray,
    level_arrs: Dict[str, np.ndarray],
    proximity_pts: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest level name/price per bar within proximity."""
    n = len(close_arr)
    nearest_level_name = np.full(n, None, dtype=object)
    nearest_level_price = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or not level_arrs:
        return nearest_level_name, nearest_level_price

    close_vals = np.asarray(close_arr, dtype=np.float64)
    proximity = float(proximity_pts)

    for lvl_name, lvl_vals_raw in level_arrs.items():
        lvl_vals = np.asarray(lvl_vals_raw, dtype=np.float64)
        if len(lvl_vals) != n:
            continue
        dist = np.abs(close_vals - lvl_vals)
        valid = np.isfinite(dist) & (dist <= proximity)
        if not np.any(valid):
            continue
        prev_dist = np.abs(close_vals - nearest_level_price)
        update = valid & (~np.isfinite(nearest_level_price) | (dist < prev_dist))
        if not np.any(update):
            continue
        nearest_level_price[update] = lvl_vals[update]
        nearest_level_name[update] = lvl_name

    return nearest_level_name, nearest_level_price


def assign_level_side(
    close_arr: np.ndarray,
    nearest_level_name: np.ndarray,
    nearest_level_price: np.ndarray,
    *,
    side_hysteresis_pts: float = 0.0,
    side_flip_confirm_pts: Optional[float] = None,
    trading_day: Optional[Iterable] = None,
) -> np.ndarray:
    """
    Assign side for each near-level bar.

    Side convention:
    -  1 = above level (support-side, long reversal direction)
    - -1 = below level (resistance-side, short reversal direction)

    If ``side_hysteresis_pts > 0``, side is sticky within a deadband around
    each level and tracked independently per level (resets each trading day).
    ``side_flip_confirm_pts`` can require a deeper opposite-side cross before
    flipping (defaults to ``side_hysteresis_pts``).
    """
    n = len(close_arr)
    side = np.zeros(n, dtype=np.int8)
    if n == 0:
        return side

    close_vals = np.asarray(close_arr, dtype=np.float64)
    lvl_names = np.asarray(nearest_level_name, dtype=object)
    lvl_prices = np.asarray(nearest_level_price, dtype=np.float64)
    near_mask = np.isfinite(lvl_prices)
    if not np.any(near_mask):
        return side

    if side_hysteresis_pts <= 0.0:
        side[near_mask & (close_vals >= lvl_prices)] = 1
        side[near_mask & (close_vals < lvl_prices)] = -1
        return side

    hyst = float(side_hysteresis_pts)
    flip_confirm = float(side_flip_confirm_pts) if side_flip_confirm_pts is not None else hyst
    if flip_confirm < hyst:
        flip_confirm = hyst
    day_vals = None if trading_day is None else np.asarray(list(trading_day), dtype=object)
    if day_vals is not None and len(day_vals) != n:
        day_vals = None

    last_side_by_level: Dict[str, int] = {}
    last_day = day_vals[0] if day_vals is not None and n > 0 else None

    for i in range(n):
        lvl_price = lvl_prices[i]
        if not np.isfinite(lvl_price):
            continue
        lvl_name = lvl_names[i]
        if lvl_name is None:
            continue

        if day_vals is not None:
            day_i = day_vals[i]
            if day_i != last_day:
                last_side_by_level.clear()
                last_day = day_i

        signed_dist = close_vals[i] - lvl_price
        base_side = 1 if signed_dist >= 0.0 else -1
        key = str(lvl_name)
        prev_side = int(last_side_by_level.get(key, 0))
        if prev_side == 1:
            # Keep bullish side unless opposite cross is strong enough.
            s = 1 if signed_dist > -flip_confirm else -1
        elif prev_side == -1:
            # Keep bearish side unless opposite cross is strong enough.
            s = -1 if signed_dist < flip_confirm else 1
        else:
            s = base_side
        side[i] = s
        last_side_by_level[key] = s

    return side
