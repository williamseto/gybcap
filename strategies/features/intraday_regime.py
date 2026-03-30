"""Causal intraday regime/context features for reversal models.

These features summarize the session context at each bar using only information
known at the current bar close. They are intended to help the base reversal
classifier distinguish trend-day continuation from rotational fade setups.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


INTRADAY_REGIME_FEATURES = [
    "gap_from_prev_close_pts",
    "gap_from_prev_close_atr",
    "overnight_range_pts",
    "overnight_open_loc",
    "opening_drive_regime_dir",
    "opening_drive_net_pts",
    "opening_drive_frac",
    "or15_range_pts",
    "or15_close_pos",
    "day_progress_norm",
    "dist_from_rth_open_pts",
    "dist_from_rth_open_in_rth_range",
    "dist_from_vwap_pts",
    "session_trend_strength",
]


def _minute_of_day(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/Los_Angeles")
    return int(t.hour) * 60 + int(t.minute)


def compute_intraday_regime_features(
    df: pd.DataFrame,
    dt_col: str = "dt",
    trading_day_col: str = "trading_day",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    vwap_col: str = "vwap",
    atr_col: str = "daily_atr_14",
    rth_open_minute: int = 390,
    opening_end_minute: int = 420,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in INTRADAY_REGIME_FEATURES:
        out[col] = 0.0

    if df.empty or dt_col not in df.columns:
        return out
    if trading_day_col not in df.columns:
        return out

    dt = pd.to_datetime(df[dt_col], errors="coerce")
    minute = dt.map(lambda ts: _minute_of_day(ts) if pd.notna(ts) else -1).astype(np.int32)
    out["day_progress_norm"] = np.where(
        minute >= rth_open_minute,
        np.clip((minute - rth_open_minute) / max(780 - rth_open_minute, 1), 0.0, 1.0),
        0.0,
    ).astype(np.float32)

    if close_col not in df.columns:
        return out

    prev_close_map: dict[str, float] = {}
    day_close = (
        df.groupby(trading_day_col, sort=True)[close_col].last().astype(np.float64)
        if close_col in df.columns else pd.Series(dtype=np.float64)
    )
    if not day_close.empty:
        prev_close_map = day_close.shift(1).to_dict()

    for day, g in df.groupby(trading_day_col, sort=False):
        gi = g.index
        mod = minute.loc[gi].to_numpy(dtype=np.int32, copy=False)
        opens = pd.to_numeric(g.get(open_col, g[close_col]), errors="coerce").to_numpy(dtype=np.float64, copy=False)
        highs = pd.to_numeric(g.get(high_col, g[close_col]), errors="coerce").to_numpy(dtype=np.float64, copy=False)
        lows = pd.to_numeric(g.get(low_col, g[close_col]), errors="coerce").to_numpy(dtype=np.float64, copy=False)
        closes = pd.to_numeric(g[close_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        vwap = (
            pd.to_numeric(g[vwap_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            if vwap_col in g.columns else np.full(len(g), np.nan, dtype=np.float64)
        )
        atr = (
            pd.to_numeric(g[atr_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            if atr_col in g.columns else np.full(len(g), np.nan, dtype=np.float64)
        )

        prev_close = prev_close_map.get(str(day), np.nan)
        if np.isfinite(prev_close):
            gap = opens[0] - float(prev_close)
            out.loc[gi, "gap_from_prev_close_pts"] = float(gap)
            if np.isfinite(atr[0]) and float(atr[0]) > 0:
                out.loc[gi, "gap_from_prev_close_atr"] = float(gap / atr[0])

        ovn_mask = mod < rth_open_minute
        ovn_range_running = np.zeros(len(g), dtype=np.float32)
        if np.any(ovn_mask):
            ovn_high_run = np.maximum.accumulate(np.where(ovn_mask, highs, np.nan))
            ovn_low_run = np.minimum.accumulate(np.where(ovn_mask, lows, np.nan))
            valid = np.isfinite(ovn_high_run) & np.isfinite(ovn_low_run)
            ovn_range_running[valid] = (ovn_high_run[valid] - ovn_low_run[valid]).astype(np.float32)
            ovn_low = float(np.nanmin(lows[ovn_mask]))
            ovn_hi = float(np.nanmax(highs[ovn_mask]))
            ovn_rng = max(ovn_hi - ovn_low, 0.25)
        else:
            ovn_low = np.nan
            ovn_hi = np.nan
            ovn_rng = 0.25

        out.loc[gi, "overnight_range_pts"] = ovn_range_running

        rth_idx = np.where(mod >= rth_open_minute)[0]
        if rth_idx.size == 0:
            continue
        first_rth = int(rth_idx[0])
        rth_open = float(opens[first_rth])

        if np.isfinite(ovn_low) and np.isfinite(ovn_hi):
            out.loc[gi[first_rth:], "overnight_range_pts"] = float(ovn_hi - ovn_low)
            out.loc[gi[first_rth:], "overnight_open_loc"] = float((rth_open - ovn_low) / ovn_rng)
        else:
            out.loc[gi[first_rth:], "overnight_open_loc"] = 0.5

        rth_highs = np.maximum.accumulate(highs[first_rth:])
        rth_lows = np.minimum.accumulate(lows[first_rth:])
        rth_range = np.maximum(rth_highs - rth_lows, 0.25)

        out.loc[gi[first_rth:], "dist_from_rth_open_pts"] = (closes[first_rth:] - rth_open).astype(np.float32)
        out.loc[gi[first_rth:], "dist_from_rth_open_in_rth_range"] = (
            (closes[first_rth:] - rth_open) / rth_range
        ).astype(np.float32)
        out.loc[gi[first_rth:], "session_trend_strength"] = (
            np.abs(closes[first_rth:] - rth_open) / rth_range
        ).astype(np.float32)

        if np.any(np.isfinite(vwap[first_rth:])):
            dist_vwap = np.zeros(len(g) - first_rth, dtype=np.float32)
            valid = np.isfinite(vwap[first_rth:])
            dist_vwap[valid] = (closes[first_rth:][valid] - vwap[first_rth:][valid]).astype(np.float32)
            out.loc[gi[first_rth:], "dist_from_vwap_pts"] = dist_vwap

        or15_mask = (mod >= rth_open_minute) & (mod < (rth_open_minute + 15))
        if np.any(or15_mask):
            or15_hi = float(np.nanmax(highs[or15_mask]))
            or15_lo = float(np.nanmin(lows[or15_mask]))
            or15_rng = max(or15_hi - or15_lo, 0.25)
            or15_close = float(closes[np.where(or15_mask)[0][-1]])
            active = mod >= (rth_open_minute + 15)
            if np.any(active):
                out.loc[gi[active], "or15_range_pts"] = float(or15_hi - or15_lo)
                out.loc[gi[active], "or15_close_pos"] = float((or15_close - or15_lo) / or15_rng)

        open30_mask = (mod >= rth_open_minute) & (mod < opening_end_minute)
        if np.any(open30_mask):
            open_px = float(opens[first_rth])
            close_30 = float(closes[np.where(open30_mask)[0][-1]])
            high_30 = float(np.nanmax(highs[open30_mask]))
            low_30 = float(np.nanmin(lows[open30_mask]))
            net = float(close_30 - open_px)
            rng = max(float(high_30 - low_30), 0.25)
            drive_frac = float(abs(net) / rng)
            regime_dir = 0
            pos_in_range = float((close_30 - low_30) / rng)
            if abs(net) >= 8.0 and drive_frac >= 0.55:
                if net > 0.0 and pos_in_range >= 0.7:
                    regime_dir = 1
                elif net < 0.0 and pos_in_range <= 0.3:
                    regime_dir = -1
            active = mod >= opening_end_minute
            if np.any(active):
                out.loc[gi[active], "opening_drive_regime_dir"] = float(regime_dir)
                out.loc[gi[active], "opening_drive_net_pts"] = float(net)
                out.loc[gi[active], "opening_drive_frac"] = float(drive_frac)

    return out
