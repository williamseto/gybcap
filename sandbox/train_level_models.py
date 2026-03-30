#!/usr/bin/env python
"""
Phase 3: Single-Model Training with Level-Quality Encoding.

Trains a single XGBoost model that generalizes across all level types by
encoding level identity and quality as features. This lets the model learn
level-specific patterns (e.g. "ovn_hi with high volume → reversal") within
a unified framework with more training data than per-level models.

New features added:
  - One-hot level type (7 features)
  - Trailing 20d/50d reversal rate per level type (causal)
  - Level side (support/resistance)
  - Level-side × BB interaction
  - Approach direction relative to level

Walk-forward CV with honest early stopping (temporal train/val split).

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_level_models.py
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = "raw_data/es_min_3y_clean_td_gamma.csv"
FIGURE_DIR = "sandbox/figures/level_models"

from strategies.labeling.reversal_zones import TRACKED_LEVELS, ReversalBreakoutLabeler
from strategies.labeling.near_level import (
    assign_level_side,
    compute_nearest_level_arrays,
)

STOP_PTS = 4.0
TARGET_PTS = 6.0
MAX_BARS = 45
BASE_TRACKED_LEVELS = list(TRACKED_LEVELS)
IB_RELATIVE_LEVELS_CORE = [
    "ib_mid",
    "ib_hi_ext_025",
    "ib_hi_ext_050",
    "ib_lo_ext_025",
    "ib_lo_ext_050",
]
IB_RELATIVE_LEVELS_FULL = IB_RELATIVE_LEVELS_CORE + [
    "ib_hi_ext_100",
    "ib_lo_ext_100",
]
OR_LEVELS = [
    "or5_hi",
    "or5_lo",
    "or5_mid",
    "or15_hi",
    "or15_lo",
    "or15_mid",
]
PRIOR_RTH_LEVELS = [
    "prior_rth_open",
    "prior_rth_close",
]
VWAP_SIGMA_LEVEL_MULTS = {
    "vwap_up_1s": +1.0,
    "vwap_dn_1s": -1.0,
    "vwap_up_15s": +1.5,
    "vwap_dn_15s": -1.5,
    "vwap_up_2s": +2.0,
    "vwap_dn_2s": -2.0,
}


def _minute_of_day(ts: pd.Timestamp) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("America/Los_Angeles")
    return int(t.hour) * 60 + int(t.minute)


def resolve_tracked_levels(
    tracked_levels_override: Optional[List[str]] = None,
    include_ib_relative: bool = False,
    ib_relative_mode: str = "core",
) -> List[str]:
    """Resolve effective tracked levels for labeling/feature encoding."""
    if tracked_levels_override:
        return list(dict.fromkeys([str(x) for x in tracked_levels_override if str(x).strip()]))

    levels = list(BASE_TRACKED_LEVELS)
    if include_ib_relative:
        ib_levels = IB_RELATIVE_LEVELS_CORE if ib_relative_mode == "core" else IB_RELATIVE_LEVELS_FULL
        levels.extend(ib_levels)
    return list(dict.fromkeys(levels))


# ── Data loading (reused from signal_detection.py) ────────────────────────

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    if 'Date' in df.columns and 'Time' in df.columns:
        df['dt'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S'
        )
    df.columns = df.columns.str.lower()
    print(f"  {len(df):,} bars, {df['trading_day'].nunique()} trading days")
    return df


def compute_levels(ohlcv: pd.DataFrame, tracked_levels: Optional[List[str]] = None) -> pd.DataFrame:
    from strategies.features.price_levels import PriceLevelProvider
    print("\nComputing price levels...")
    plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
    feat_df = plp._compute_impl(ohlcv)

    level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']
    if 'level_1' in feat_df.columns:
        feat_by_row = feat_df.set_index('level_1')
        for col in level_cols:
            if col in feat_by_row.columns:
                ohlcv[col] = feat_by_row[col].reindex(ohlcv.index).values
    else:
        if 'dt' in feat_df.columns:
            feat_df = feat_df.set_index('dt')
        ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv
        for col in level_cols:
            if col in feat_df.columns:
                src = feat_df[col]
                if src.index.has_duplicates:
                    src = src.groupby(level=0).last()
                ohlcv[col] = src.reindex(ohlcv_dt.index).values

    levels = plp.prev_day_levels(ohlcv)
    ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
    ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])

    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)
    tracked_level_set = set(tracked_levels)

    # Ensure minute_of_day exists for all intraday-derived level families.
    needs_minute_of_day = bool(
        tracked_level_set
        & (
            set(IB_RELATIVE_LEVELS_FULL)
            | set(OR_LEVELS)
            | set(VWAP_SIGMA_LEVEL_MULTS.keys())
        )
    )
    if needs_minute_of_day:
        if "dt" not in ohlcv.columns:
            raise ValueError("Missing dt column required to compute intraday levels.")
        minute_of_day = pd.to_datetime(ohlcv["dt"], errors="coerce").map(_minute_of_day).astype(np.int32)
        ohlcv["minute_of_day"] = minute_of_day

    # Prior RTH open/close references (previous trading day constants).
    if tracked_level_set & set(PRIOR_RTH_LEVELS):
        if "minute_of_day" not in ohlcv.columns:
            minute_of_day = pd.to_datetime(ohlcv["dt"], errors="coerce").map(_minute_of_day).astype(np.int32)
            ohlcv["minute_of_day"] = minute_of_day
        rth_mask = (ohlcv["minute_of_day"] >= 390) & (ohlcv["minute_of_day"] < 780)
        rth = ohlcv.loc[rth_mask, ["trading_day", "open", "close"]].copy()
        daily_rth_open = rth.groupby("trading_day")["open"].first() if not rth.empty else pd.Series(dtype=np.float64)
        daily_rth_close = rth.groupby("trading_day")["close"].last() if not rth.empty else pd.Series(dtype=np.float64)
        prior_rth_open = daily_rth_open.shift(1).to_dict()
        prior_rth_close = daily_rth_close.shift(1).to_dict()
        if "prior_rth_open" in tracked_level_set:
            ohlcv["prior_rth_open"] = ohlcv["trading_day"].map(prior_rth_open)
        if "prior_rth_close" in tracked_level_set:
            ohlcv["prior_rth_close"] = ohlcv["trading_day"].map(prior_rth_close)

    # Opening-range references (causal: available only after window completes).
    if tracked_level_set & set(OR_LEVELS):
        if "minute_of_day" not in ohlcv.columns:
            minute_of_day = pd.to_datetime(ohlcv["dt"], errors="coerce").map(_minute_of_day).astype(np.int32)
            ohlcv["minute_of_day"] = minute_of_day
        for col in OR_LEVELS:
            if col in tracked_level_set and col not in ohlcv.columns:
                ohlcv[col] = np.nan

        for _, g in ohlcv.groupby("trading_day", sort=False):
            gi = g.index
            mod = ohlcv.loc[gi, "minute_of_day"].to_numpy(dtype=np.int32, copy=False)
            highs = ohlcv.loc[gi, "high"].to_numpy(dtype=np.float64, copy=False)
            lows = ohlcv.loc[gi, "low"].to_numpy(dtype=np.float64, copy=False)

            # OR5: 6:30-6:34 (inclusive), active from 6:35 onward.
            if tracked_level_set & {"or5_hi", "or5_lo", "or5_mid"}:
                or5_mask = (mod >= 390) & (mod <= 394)
                if np.any(or5_mask):
                    or5_hi = float(np.nanmax(highs[or5_mask]))
                    or5_lo = float(np.nanmin(lows[or5_mask]))
                    valid = gi[mod >= 395]
                    if len(valid) > 0:
                        if "or5_hi" in tracked_level_set:
                            ohlcv.loc[valid, "or5_hi"] = or5_hi
                        if "or5_lo" in tracked_level_set:
                            ohlcv.loc[valid, "or5_lo"] = or5_lo
                        if "or5_mid" in tracked_level_set:
                            ohlcv.loc[valid, "or5_mid"] = (or5_hi + or5_lo) / 2.0

            # OR15: 6:30-6:44 (inclusive), active from 6:45 onward.
            if tracked_level_set & {"or15_hi", "or15_lo", "or15_mid"}:
                or15_mask = (mod >= 390) & (mod <= 404)
                if np.any(or15_mask):
                    or15_hi = float(np.nanmax(highs[or15_mask]))
                    or15_lo = float(np.nanmin(lows[or15_mask]))
                    valid = gi[mod >= 405]
                    if len(valid) > 0:
                        if "or15_hi" in tracked_level_set:
                            ohlcv.loc[valid, "or15_hi"] = or15_hi
                        if "or15_lo" in tracked_level_set:
                            ohlcv.loc[valid, "or15_lo"] = or15_lo
                        if "or15_mid" in tracked_level_set:
                            ohlcv.loc[valid, "or15_mid"] = (or15_hi + or15_lo) / 2.0

    # VWAP sigma bands (causal running dispersion around VWAP).
    if tracked_level_set & set(VWAP_SIGMA_LEVEL_MULTS.keys()):
        for col in VWAP_SIGMA_LEVEL_MULTS:
            if col in tracked_level_set and col not in ohlcv.columns:
                ohlcv[col] = np.nan
        for _, g in ohlcv.groupby("trading_day", sort=False):
            gi = g.index
            vwap = ohlcv.loc[gi, "vwap"].to_numpy(dtype=np.float64, copy=False)
            tp = (
                ohlcv.loc[gi, "open"].to_numpy(dtype=np.float64, copy=False)
                + ohlcv.loc[gi, "high"].to_numpy(dtype=np.float64, copy=False)
                + ohlcv.loc[gi, "low"].to_numpy(dtype=np.float64, copy=False)
                + ohlcv.loc[gi, "close"].to_numpy(dtype=np.float64, copy=False)
            ) / 4.0
            dev = tp - vwap
            sigma = pd.Series(dev).expanding(min_periods=10).std().to_numpy(dtype=np.float64, copy=False)
            valid_sigma = np.isfinite(sigma) & np.isfinite(vwap)
            if not np.any(valid_sigma):
                continue
            for col, mult in VWAP_SIGMA_LEVEL_MULTS.items():
                if col not in tracked_level_set:
                    continue
                band = vwap + float(mult) * sigma
                out = np.full(len(gi), np.nan, dtype=np.float64)
                out[valid_sigma] = band[valid_sigma]
                ohlcv.loc[gi, col] = out

    needs_ib_relative = any(
        lvl in set(IB_RELATIVE_LEVELS_FULL) for lvl in tracked_levels
    )
    if needs_ib_relative:
        # IB-derived references are causal only after IB completion (7:30 PT).
        if "minute_of_day" not in ohlcv.columns:
            if "dt" not in ohlcv.columns:
                raise ValueError("Missing dt column required to compute IB-relative levels.")
            minute_of_day = pd.to_datetime(ohlcv["dt"], errors="coerce").map(_minute_of_day).astype(np.int32)
            ohlcv["minute_of_day"] = minute_of_day
        else:
            minute_of_day = ohlcv["minute_of_day"]
        ib_lo = ohlcv["ib_lo"].to_numpy(dtype=np.float64, copy=False)
        ib_hi = ohlcv["ib_hi"].to_numpy(dtype=np.float64, copy=False)
        valid_ib = (
            (minute_of_day.to_numpy(dtype=np.int32, copy=False) >= 450)
            & np.isfinite(ib_lo)
            & np.isfinite(ib_hi)
            & (ib_hi > ib_lo)
        )
        ib_range = ib_hi - ib_lo
        if "ib_mid" in tracked_levels:
            ohlcv["ib_mid"] = np.where(valid_ib, (ib_hi + ib_lo) / 2.0, np.nan)
        if "ib_hi_ext_025" in tracked_levels:
            ohlcv["ib_hi_ext_025"] = np.where(valid_ib, ib_hi + 0.25 * ib_range, np.nan)
        if "ib_hi_ext_050" in tracked_levels:
            ohlcv["ib_hi_ext_050"] = np.where(valid_ib, ib_hi + 0.50 * ib_range, np.nan)
        if "ib_hi_ext_100" in tracked_levels:
            ohlcv["ib_hi_ext_100"] = np.where(valid_ib, ib_hi + 1.00 * ib_range, np.nan)
        if "ib_lo_ext_025" in tracked_levels:
            ohlcv["ib_lo_ext_025"] = np.where(valid_ib, ib_lo - 0.25 * ib_range, np.nan)
        if "ib_lo_ext_050" in tracked_levels:
            ohlcv["ib_lo_ext_050"] = np.where(valid_ib, ib_lo - 0.50 * ib_range, np.nan)
        if "ib_lo_ext_100" in tracked_levels:
            ohlcv["ib_lo_ext_100"] = np.where(valid_ib, ib_lo - 1.00 * ib_range, np.nan)

    return ohlcv


def compute_all_features(
    ohlcv: pd.DataFrame,
    same_day_bidask_only: bool = False,
    include_intraday_regime_features: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute all feature providers on the data."""
    feature_cols = []

    print("\nComputing higher timeframe features...")
    from strategies.features.higher_timeframe import HigherTimeframeProvider
    htf = HigherTimeframeProvider()
    htf_df = htf._compute_impl(ohlcv)
    for col in htf.feature_names:
        if col in htf_df.columns:
            ohlcv[col] = htf_df[col].values
            feature_cols.append(col)

    print("Computing volume microstructure features...")
    from strategies.features.volume_microstructure import VolumeMicrostructureProvider
    has_bidask = 'bidvolume' in ohlcv.columns
    vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
    vol_df = vmp._compute_impl(ohlcv)
    for col in vmp.feature_names:
        if col in vol_df.columns:
            ohlcv[col] = vol_df[col].values
            feature_cols.append(col)

    print("Computing reversion quality features...")
    from strategies.features.reversion_quality import ReversionQualityProvider
    rqp = ReversionQualityProvider(same_day_bidask_only=same_day_bidask_only)
    qual_df = rqp._compute_impl(ohlcv)
    for col in rqp.feature_names:
        if col in qual_df.columns:
            ohlcv[col] = qual_df[col].values
            feature_cols.append(col)

    print("Computing temporal interaction features...")
    from strategies.features.temporal_interactions import TemporalInteractionProvider
    tip = TemporalInteractionProvider()
    temp_df = tip._compute_impl(ohlcv)
    for col in tip.feature_names:
        if col in temp_df.columns:
            ohlcv[col] = temp_df[col].values
            feature_cols.append(col)

    # RTH running-level stability features: quantify how quickly rth_lo/rth_hi
    # are moving, which helps separate stable references from churny volatility.
    if "rth_lo" in ohlcv.columns and "rth_hi" in ohlcv.columns:
        print("Computing RTH level-stability features...")

        def _age_since_change(vals: np.ndarray, valid: np.ndarray) -> np.ndarray:
            out = np.zeros_like(vals, dtype=np.float32)
            last_change = -1
            prev = np.nan
            for i in range(len(vals)):
                if not valid[i] or not np.isfinite(vals[i]):
                    out[i] = 0.0
                    continue
                cur = float(vals[i])
                if not np.isfinite(prev) or abs(cur - prev) > 1e-9:
                    last_change = i
                out[i] = float(i - last_change) if last_change >= 0 else 0.0
                prev = cur
            return out

        ohlcv["rth_lo_age_bars"] = 0.0
        ohlcv["rth_hi_age_bars"] = 0.0
        ohlcv["rth_lo_updates_30"] = 0.0
        ohlcv["rth_hi_updates_30"] = 0.0
        ohlcv["rth_level_firmness"] = 0.0

        for _, g in ohlcv.groupby("trading_day", sort=False):
            gi = g.index
            lo = ohlcv.loc[gi, "rth_lo"].to_numpy(dtype=np.float64, copy=False)
            hi = ohlcv.loc[gi, "rth_hi"].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(lo) & np.isfinite(hi) & (lo != 0.0) & (hi != 0.0)
            if not np.any(valid):
                continue

            lo_chg = np.zeros(len(gi), dtype=np.float32)
            hi_chg = np.zeros(len(gi), dtype=np.float32)
            lo_chg[1:] = (
                valid[1:]
                & valid[:-1]
                & (np.abs(lo[1:] - lo[:-1]) > 1e-9)
            ).astype(np.float32)
            hi_chg[1:] = (
                valid[1:]
                & valid[:-1]
                & (np.abs(hi[1:] - hi[:-1]) > 1e-9)
            ).astype(np.float32)

            lo_updates_30 = pd.Series(lo_chg).rolling(30, min_periods=1).sum().to_numpy(dtype=np.float32)
            hi_updates_30 = pd.Series(hi_chg).rolling(30, min_periods=1).sum().to_numpy(dtype=np.float32)
            lo_age = _age_since_change(lo, valid)
            hi_age = _age_since_change(hi, valid)
            firmness = 1.0 / (1.0 + lo_updates_30 + hi_updates_30)

            ohlcv.loc[gi, "rth_lo_age_bars"] = lo_age
            ohlcv.loc[gi, "rth_hi_age_bars"] = hi_age
            ohlcv.loc[gi, "rth_lo_updates_30"] = lo_updates_30
            ohlcv.loc[gi, "rth_hi_updates_30"] = hi_updates_30
            ohlcv.loc[gi, "rth_level_firmness"] = firmness

        feature_cols.extend([
            "rth_lo_age_bars",
            "rth_hi_age_bars",
            "rth_lo_updates_30",
            "rth_hi_updates_30",
            "rth_level_firmness",
        ])

    if include_intraday_regime_features:
        print("Computing intraday regime/context features...")
        from strategies.features.intraday_regime import (
            INTRADAY_REGIME_FEATURES,
            compute_intraday_regime_features,
        )
        regime_df = compute_intraday_regime_features(ohlcv)
        for col in INTRADAY_REGIME_FEATURES:
            if col in regime_df.columns:
                ohlcv[col] = regime_df[col].values.astype(np.float32)
                feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"  Total base features: {len(feature_cols)}")
    return ohlcv, feature_cols


# ── Labeling (reused from signal_detection.py) ───────────────────────────

def label_reversals_breakouts(
    ohlcv: pd.DataFrame,
    proximity_pts: float = 5.0,
    forward_bars: int = 45,
    reversal_pts: float = 6.0,
    breakout_pts: float = 4.0,
    tracked_levels: Optional[List[str]] = None,
    side_hysteresis_pts: float = 0.0,
    side_flip_confirm_pts: Optional[float] = None,
    drop_ambiguous_direction: bool = False,
    ambiguity_bar_tolerance: int = 2,
) -> pd.DataFrame:
    """Label near-level bars as reversal (1), breakout (0), or inconclusive (2)."""
    print(f"\nLabeling reversals/breakouts (proximity={proximity_pts}pt, "
          f"reversal={reversal_pts}pt, breakout={breakout_pts}pt)...")

    close_arr = ohlcv['close'].values.astype(np.float64)
    high_arr = ohlcv['high'].values.astype(np.float64)
    low_arr = ohlcv['low'].values.astype(np.float64)
    n = len(ohlcv)

    labels = np.full(n, -1, dtype=np.int32)
    nearest_level_name = np.full(n, None, dtype=object)
    nearest_level_price = np.full(n, np.nan)
    side_arr = np.zeros(n, dtype=np.int8)  # 1=above(support), -1=below(resistance)
    trade_dir_arr = np.zeros(n, dtype=np.int8)

    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)

    # Collect level arrays
    level_arrs = {}
    for lvl_name in tracked_levels:
        if lvl_name in ohlcv.columns:
            level_arrs[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

    nearest_level_name, nearest_level_price = compute_nearest_level_arrays(
        close_arr,
        level_arrs,
        proximity_pts=float(proximity_pts),
    )

    near_level_mask = ~np.isnan(nearest_level_price)
    n_near = near_level_mask.sum()
    print(f"  {n_near:,} near-level bars out of {n:,} ({100*n_near/n:.1f}%)")

    side_arr = assign_level_side(
        close_arr,
        nearest_level_name,
        nearest_level_price,
        side_hysteresis_pts=float(side_hysteresis_pts),
        side_flip_confirm_pts=(
            float(side_flip_confirm_pts) if side_flip_confirm_pts is not None else None
        ),
        trading_day=ohlcv['trading_day'].values if 'trading_day' in ohlcv.columns else None,
    )

    # Label each near-level bar
    for i in range(n):
        if not near_level_mask[i]:
            continue

        level = nearest_level_price[i]
        price = close_arr[i]
        is_above = side_arr[i] >= 1
        trade_dir_arr[i] = side_arr[i]  # long above support, short below resistance

        end = min(i + 1 + forward_bars, n)
        future_highs = high_arr[i+1:end]
        future_lows = low_arr[i+1:end]

        if len(future_highs) == 0:
            labels[i] = 2
            continue

        if is_above:
            # Level is support → reversal=UP, breakout=DOWN through level
            rev_hits = np.where(future_highs >= price + reversal_pts)[0]
            brk_hits = np.where(future_lows <= level - breakout_pts)[0]
        else:
            # Level is resistance → reversal=DOWN, breakout=UP through level
            rev_hits = np.where(future_lows <= price - reversal_pts)[0]
            brk_hits = np.where(future_highs >= level + breakout_pts)[0]

        first_rev = (rev_hits[0]) if len(rev_hits) > 0 else 9999
        first_brk = (brk_hits[0]) if len(brk_hits) > 0 else 9999

        assigned = 2
        assigned_bar = 9999
        if first_rev < first_brk and first_rev < 9999:
            assigned = 1  # reversal
            assigned_bar = int(first_rev)
        elif first_brk < first_rev and first_brk < 9999:
            assigned = 0  # breakout
            assigned_bar = int(first_brk)

        if drop_ambiguous_direction and assigned in (0, 1):
            if is_above:
                # Opposite side assumption (below level)
                opp_rev_hits = np.where(future_lows <= price - reversal_pts)[0]
                opp_brk_hits = np.where(future_highs >= level + breakout_pts)[0]
            else:
                # Opposite side assumption (above level)
                opp_rev_hits = np.where(future_highs >= price + reversal_pts)[0]
                opp_brk_hits = np.where(future_lows <= level - breakout_pts)[0]

            opp_first_rev = (opp_rev_hits[0]) if len(opp_rev_hits) > 0 else 9999
            opp_first_brk = (opp_brk_hits[0]) if len(opp_brk_hits) > 0 else 9999
            if opp_first_rev < opp_first_brk and opp_first_rev < 9999:
                opp = 1
                opp_bar = int(opp_first_rev)
            elif opp_first_brk < opp_first_rev and opp_first_brk < 9999:
                opp = 0
                opp_bar = int(opp_first_brk)
            else:
                opp = 2
                opp_bar = 9999
            if (
                opp in (0, 1)
                and opp != assigned
                and opp_bar <= (assigned_bar + max(int(ambiguity_bar_tolerance), 0))
            ):
                assigned = 2

        labels[i] = assigned  # 1=reversal, 0=breakout, 2=inconclusive

    ohlcv['outcome'] = labels
    ohlcv['nearest_level_name'] = nearest_level_name
    ohlcv['nearest_level_price'] = nearest_level_price
    ohlcv['side'] = side_arr
    ohlcv['trade_direction'] = trade_dir_arr

    n_rev = (labels == 1).sum()
    n_bo = (labels == 0).sum()
    n_inc = (labels == 2).sum()
    print(f"  Reversal: {n_rev:,} ({100*n_rev/max(n_near,1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100*n_bo/max(n_near,1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100*n_inc/max(n_near,1):.1f}%)")

    return ohlcv


def label_reversals_breakouts_structural(
    ohlcv: pd.DataFrame,
    proximity_pts: float = 5.0,
    forward_bars: int = 60,
    reversal_pts: float = 12.0,
    breakout_pts: float = 6.0,
    tracked_levels: Optional[List[str]] = None,
    side_hysteresis_pts: float = 0.5,
    side_flip_confirm_pts: Optional[float] = None,
    drop_ambiguous_direction: bool = False,
    ambiguity_bar_tolerance: int = 2,
) -> pd.DataFrame:
    """
    Structural label path reused from ReversalBreakoutLabeler.

    Returns same output columns as ``label_reversals_breakouts``:
    - outcome: 1 reversal, 0 breakout, 2 inconclusive
    - nearest_level_name, nearest_level_price, side, trade_direction
    """
    levels = tracked_levels or list(BASE_TRACKED_LEVELS)
    labeler = ReversalBreakoutLabeler(
        proximity_pts=float(proximity_pts),
        forward_window=int(forward_bars),
        reversal_threshold_pts=float(reversal_pts),
        breakout_threshold_pts=float(breakout_pts),
        decay_alpha=0.3,
        side_hysteresis_pts=float(side_hysteresis_pts),
        side_flip_confirm_pts=(
            float(side_flip_confirm_pts) if side_flip_confirm_pts is not None else None
        ),
        drop_ambiguous_direction=bool(drop_ambiguous_direction),
        ambiguity_bar_tolerance=int(ambiguity_bar_tolerance),
        tracked_levels=levels,
    )
    labeled = labeler.fit(ohlcv)

    out = ohlcv.copy()
    out["nearest_level_name"] = labeled["nearest_level"].astype(object)
    out["nearest_level_price"] = np.nan
    near_mask = labeled["near_level"].fillna(False).values
    if near_mask.any():
        for lvl in levels:
            if lvl not in out.columns:
                continue
            m = near_mask & (out["nearest_level_name"].values == lvl)
            if not np.any(m):
                continue
            out.loc[m, "nearest_level_price"] = out.loc[m, lvl].values

    out["side"] = labeled["side"].astype(np.int8).values
    out["trade_direction"] = labeled["trade_direction"].astype(np.int8).values

    outcome = np.full(len(out), -1, dtype=np.int32)
    near_outcome = labeled["outcome"].astype(str).values
    outcome[(near_outcome == "reversal")] = 1
    outcome[(near_outcome == "breakout")] = 0
    outcome[(near_outcome == "inconclusive")] = 2
    out["outcome"] = outcome

    n_near = int(near_mask.sum())
    n_rev = int((outcome == 1).sum())
    n_bo = int((outcome == 0).sum())
    n_inc = int((outcome == 2).sum())
    print(
        f"\nStructural labeling (proximity={proximity_pts}pt, "
        f"reversal={reversal_pts}pt, breakout={breakout_pts}pt, "
        f"forward={forward_bars}, side_hyst={side_hysteresis_pts})..."
    )
    print(f"  {n_near:,} near-level bars out of {len(out):,} ({100*n_near/max(len(out),1):.1f}%)")
    print(f"  Reversal: {n_rev:,} ({100*n_rev/max(n_near,1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100*n_bo/max(n_near,1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100*n_inc/max(n_near,1):.1f}%)")

    return out


def label_reversals_major_move(
    ohlcv: pd.DataFrame,
    proximity_pts: float = 5.0,
    forward_bars: int = 60,
    reversal_pts: float = 6.0,
    breakout_pts: float = 4.0,
    tracked_levels: Optional[List[str]] = None,
    side_hysteresis_pts: float = 0.5,
    side_flip_confirm_pts: Optional[float] = None,
    excursion_ratio: float = 1.4,
    breakout_excursion_ratio: float = 1.2,
    terminal_disp_frac: float = 0.20,
    use_volatility_scaling: bool = True,
    volatility_col: str = "daily_atr_14",
    reversal_vol_mult: float = 0.10,
    breakout_vol_mult: float = 0.08,
    drop_near_threshold_ambiguity: bool = False,
    ambiguity_margin_frac: float = 0.0,
    ambiguity_ratio_buffer: float = 0.0,
) -> pd.DataFrame:
    """
    Label near-level bars by excursion quality, decoupled from stop-first ordering.

    Positive (reversal):
      - aligned MFE exceeds threshold, and
      - aligned/adverse excursion ratio exceeds `excursion_ratio`, and
      - terminal displacement is aligned.
    Negative (breakout/failure):
      - adverse excursion exceeds threshold, and
      - adverse/aligned ratio exceeds `breakout_excursion_ratio`, and
      - terminal displacement is adverse.

    Everything else is marked inconclusive to avoid ambiguous supervision.
    """
    print(
        f"\nLabeling major-move outcomes (proximity={proximity_pts}pt, "
        f"forward={forward_bars}, excursion_ratio={excursion_ratio}, "
        f"use_vol_scaling={use_volatility_scaling})..."
    )

    close_arr = ohlcv["close"].values.astype(np.float64)
    high_arr = ohlcv["high"].values.astype(np.float64)
    low_arr = ohlcv["low"].values.astype(np.float64)
    n = len(ohlcv)

    labels = np.full(n, -1, dtype=np.int32)
    nearest_level_name = np.full(n, None, dtype=object)
    nearest_level_price = np.full(n, np.nan)
    side_arr = np.zeros(n, dtype=np.int8)
    trade_dir_arr = np.zeros(n, dtype=np.int8)

    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)
    level_arrs = {}
    for lvl_name in tracked_levels:
        if lvl_name in ohlcv.columns:
            level_arrs[lvl_name] = ohlcv[lvl_name].values.astype(np.float64)

    nearest_level_name, nearest_level_price = compute_nearest_level_arrays(
        close_arr,
        level_arrs,
        proximity_pts=float(proximity_pts),
    )
    near_level_mask = ~np.isnan(nearest_level_price)
    n_near = int(near_level_mask.sum())
    print(f"  {n_near:,} near-level bars out of {n:,} ({100*n_near/max(n,1):.1f}%)")

    side_arr = assign_level_side(
        close_arr,
        nearest_level_name,
        nearest_level_price,
        side_hysteresis_pts=float(side_hysteresis_pts),
        side_flip_confirm_pts=(
            float(side_flip_confirm_pts) if side_flip_confirm_pts is not None else None
        ),
        trading_day=ohlcv["trading_day"].values if "trading_day" in ohlcv.columns else None,
    )

    vol_arr = None
    if use_volatility_scaling and volatility_col in ohlcv.columns:
        vol_arr = ohlcv[volatility_col].values.astype(np.float64)

    eps = 1e-6
    for i in range(n):
        if not near_level_mask[i]:
            continue
        level = nearest_level_price[i]
        price = close_arr[i]
        side = int(side_arr[i])
        if side == 0:
            labels[i] = 2
            continue
        trade_dir_arr[i] = side

        end = min(i + 1 + int(forward_bars), n)
        future_highs = high_arr[i + 1:end]
        future_lows = low_arr[i + 1:end]
        future_closes = close_arr[i + 1:end]
        if len(future_highs) == 0:
            labels[i] = 2
            continue

        if side == 1:
            aligned_mfe = float(np.nanmax(future_highs - price))
            adverse_move = float(np.nanmax(price - future_lows))
            terminal_disp = float(future_closes[-1] - price)
            breakout_level_move = float(np.nanmax(level - future_lows))
            adverse_ref = max(adverse_move, breakout_level_move)
        else:
            aligned_mfe = float(np.nanmax(price - future_lows))
            adverse_move = float(np.nanmax(future_highs - price))
            terminal_disp = float(price - future_closes[-1])
            breakout_level_move = float(np.nanmax(future_highs - level))
            adverse_ref = max(adverse_move, breakout_level_move)

        rev_thr = float(reversal_pts)
        brk_thr = float(breakout_pts)
        if vol_arr is not None:
            vol = float(vol_arr[i])
            if np.isfinite(vol) and vol > 0:
                rev_thr = max(rev_thr, float(reversal_vol_mult) * vol)
                brk_thr = max(brk_thr, float(breakout_vol_mult) * vol)

        ratio_aligned = (aligned_mfe + eps) / (adverse_ref + eps)
        ratio_adverse = (adverse_ref + eps) / (aligned_mfe + eps)

        is_pos = (
            aligned_mfe >= rev_thr
            and ratio_aligned >= float(excursion_ratio)
            and terminal_disp >= float(terminal_disp_frac) * rev_thr
        )
        is_neg = (
            adverse_ref >= brk_thr
            and ratio_adverse >= float(breakout_excursion_ratio)
            and terminal_disp <= -float(terminal_disp_frac) * brk_thr
        )

        if is_pos and not is_neg:
            label = 1
            if bool(drop_near_threshold_ambiguity):
                aligned_margin = (aligned_mfe - rev_thr) / max(rev_thr, eps)
                ratio_margin = ratio_aligned - float(excursion_ratio)
                terminal_margin = (terminal_disp / max(rev_thr, eps)) - float(terminal_disp_frac)
                if (
                    aligned_margin <= float(ambiguity_margin_frac)
                    or ratio_margin <= float(ambiguity_ratio_buffer)
                    or terminal_margin <= float(ambiguity_margin_frac)
                ):
                    label = 2
            labels[i] = label
        elif is_neg and not is_pos:
            label = 0
            if bool(drop_near_threshold_ambiguity):
                adverse_margin = (adverse_ref - brk_thr) / max(brk_thr, eps)
                ratio_margin = ratio_adverse - float(breakout_excursion_ratio)
                terminal_margin = ((-terminal_disp) / max(brk_thr, eps)) - float(terminal_disp_frac)
                if (
                    adverse_margin <= float(ambiguity_margin_frac)
                    or ratio_margin <= float(ambiguity_ratio_buffer)
                    or terminal_margin <= float(ambiguity_margin_frac)
                ):
                    label = 2
            labels[i] = label
        else:
            labels[i] = 2

    ohlcv["outcome"] = labels
    ohlcv["nearest_level_name"] = nearest_level_name
    ohlcv["nearest_level_price"] = nearest_level_price
    ohlcv["side"] = side_arr
    ohlcv["trade_direction"] = trade_dir_arr

    n_rev = int((labels == 1).sum())
    n_bo = int((labels == 0).sum())
    n_inc = int((labels == 2).sum())
    print(f"  Reversal: {n_rev:,} ({100*n_rev/max(n_near,1):.1f}%)")
    print(f"  Breakout: {n_bo:,} ({100*n_bo/max(n_near,1):.1f}%)")
    print(f"  Inconclusive: {n_inc:,} ({100*n_inc/max(n_near,1):.1f}%)")
    return ohlcv


def reduce_to_episode_first_samples(
    samples_df: pd.DataFrame,
    gap_bars: int = 10,
) -> pd.DataFrame:
    """
    Keep only the first bar of each level-side episode.

    Episodes are grouped by trading_day + nearest_level_name + side and split when
    bar-index gap exceeds `gap_bars`.
    """
    if samples_df.empty:
        return samples_df
    gap = max(int(gap_bars), 0)
    keep_idx: List[int] = []

    grouped = samples_df.groupby(
        ["trading_day", "nearest_level_name", "side"],
        sort=False,
    )
    for _, g in grouped:
        idx = np.sort(g.index.to_numpy(dtype=np.int64, copy=False))
        if idx.size == 0:
            continue
        keep_idx.append(int(idx[0]))
        if idx.size == 1:
            continue
        cut = np.where(np.diff(idx) > gap)[0] + 1
        for c in cut:
            keep_idx.append(int(idx[c]))

    keep_idx = sorted(set(keep_idx))
    return samples_df.loc[keep_idx].copy()


# ── Step 1: Level-aware feature engineering ──────────────────────────────

def compute_level_encoding_features(
    ohlcv: pd.DataFrame,
    tracked_levels: Optional[List[str]] = None,
    include_extended_episode_features: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add ~12 features encoding level identity and quality.

    Returns updated ohlcv and list of new feature column names.
    """
    print("\nComputing level-encoding features...")
    new_cols = []
    n = len(ohlcv)
    nearest = ohlcv['nearest_level_name'].values
    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)

    # 1. One-hot encoding of nearest level type (7 features)
    for lvl in tracked_levels:
        col = f'is_{lvl}'
        ohlcv[col] = (nearest == lvl).astype(np.float32)
        new_cols.append(col)

    # 2. level_is_support: price above level → 1, below → 0
    ohlcv['level_is_support'] = (ohlcv['side'] == 1).astype(np.float32)
    new_cols.append('level_is_support')

    # 3. Trailing reversal rates per level type (causal — uses only past outcomes)
    print("  Computing trailing reversal rates (20d, 50d)...")
    ohlcv['level_trailing_rev_rate_20d'] = np.nan
    ohlcv['level_trailing_rev_rate_50d'] = np.nan

    # We need outcome and trading_day columns
    days = sorted(ohlcv['trading_day'].unique())
    day_to_idx = {d: i for i, d in enumerate(days)}

    # Pre-compute per-day per-level reversal counts
    # outcome: 1=reversal, 0=breakout, 2=inconclusive, -1=not near level
    near_mask = ohlcv['outcome'].isin([0, 1])  # only rev/bo bars count
    near_df = ohlcv.loc[near_mask, ['trading_day', 'nearest_level_name', 'outcome']].copy()

    # Daily reversal stats per level
    daily_stats = near_df.groupby(['trading_day', 'nearest_level_name']).agg(
        n_rev=('outcome', lambda x: (x == 1).sum()),
        n_total=('outcome', 'count'),
    ).reset_index()

    # Build cumulative lookup: for each (day, level) → trailing rev rate
    for lvl in tracked_levels:
        lvl_stats = daily_stats[daily_stats['nearest_level_name'] == lvl].copy()
        lvl_stats = lvl_stats.set_index('trading_day').reindex(days).fillna(0)
        lvl_stats['cum_rev'] = lvl_stats['n_rev'].cumsum()
        lvl_stats['cum_total'] = lvl_stats['n_total'].cumsum()

        # 20d trailing (rolling sum)
        lvl_stats['rev_20d'] = lvl_stats['n_rev'].rolling(20, min_periods=1).sum()
        lvl_stats['total_20d'] = lvl_stats['n_total'].rolling(20, min_periods=1).sum()
        lvl_stats['rev_rate_20d'] = lvl_stats['rev_20d'] / lvl_stats['total_20d'].clip(lower=1)

        # 50d trailing
        lvl_stats['rev_50d'] = lvl_stats['n_rev'].rolling(50, min_periods=1).sum()
        lvl_stats['total_50d'] = lvl_stats['n_total'].rolling(50, min_periods=1).sum()
        lvl_stats['rev_rate_50d'] = lvl_stats['rev_50d'] / lvl_stats['total_50d'].clip(lower=1)

        # Shift by 1 to make strictly causal (today uses data up to yesterday)
        lvl_stats['rev_rate_20d'] = lvl_stats['rev_rate_20d'].shift(1)
        lvl_stats['rev_rate_50d'] = lvl_stats['rev_rate_50d'].shift(1)

        # Map back to bars
        rate_20d_map = lvl_stats['rev_rate_20d'].to_dict()
        rate_50d_map = lvl_stats['rev_rate_50d'].to_dict()

        lvl_mask = ohlcv['nearest_level_name'] == lvl
        if lvl_mask.any():
            ohlcv.loc[lvl_mask, 'level_trailing_rev_rate_20d'] = (
                ohlcv.loc[lvl_mask, 'trading_day'].map(rate_20d_map)
            )
            ohlcv.loc[lvl_mask, 'level_trailing_rev_rate_50d'] = (
                ohlcv.loc[lvl_mask, 'trading_day'].map(rate_50d_map)
            )

    new_cols.extend(['level_trailing_rev_rate_20d', 'level_trailing_rev_rate_50d'])

    # 4. level_side_bb_interaction: level_is_support × daily_bb_pct_b
    if 'daily_bb_pct_b' in ohlcv.columns:
        ohlcv['level_side_bb_interaction'] = ohlcv['level_is_support'] * ohlcv['daily_bb_pct_b']
        new_cols.append('level_side_bb_interaction')
    else:
        print("  WARNING: daily_bb_pct_b not found, skipping BB interaction feature")

    # 5. approach_dir_vs_level: +1 approaching from above, -1 from below
    # Use 5-bar close change direction relative to level position
    close_vals = ohlcv['close'].values.astype(np.float64)
    lvl_price_vals = ohlcv['nearest_level_price'].values.astype(np.float64)
    approach_dir = np.zeros(n, dtype=np.float32)
    for i in range(5, n):
        if np.isnan(lvl_price_vals[i]):
            continue
        price_change = close_vals[i] - close_vals[i - 5]
        if close_vals[i] >= lvl_price_vals[i]:
            # Above level: approaching from above if price was falling
            approach_dir[i] = -1.0 if price_change < 0 else 1.0
        else:
            # Below level: approaching from below if price was rising
            approach_dir[i] = 1.0 if price_change > 0 else -1.0
    ohlcv['approach_dir_vs_level'] = approach_dir
    new_cols.append('approach_dir_vs_level')

    # 6. Intraday episode/reclaim-state features (current-close causal)
    print("  Computing episode-state features...")
    from strategies.features.episode_state import (
        BASE_EPISODE_STATE_FEATURES,
        EPISODE_STATE_FEATURES,
        compute_episode_state_features,
    )
    ep_df = compute_episode_state_features(
        ohlcv,
        level_col='nearest_level_name',
        side_col='side',
        trading_day_col='trading_day',
        density_window_bars=30,
    )
    episode_feature_cols = (
        EPISODE_STATE_FEATURES
        if include_extended_episode_features else BASE_EPISODE_STATE_FEATURES
    )
    for col in episode_feature_cols:
        if col in ep_df.columns:
            ohlcv[col] = ep_df[col].values.astype(np.float32)
            new_cols.append(col)

    print(f"  Added {len(new_cols)} level-encoding features")
    return ohlcv, new_cols


# ── Walk-forward splits ──────────────────────────────────────────────────

def get_fold_splits(days: List, n_folds: int, min_train_days: int) -> List[Tuple[List, List]]:
    """Walk-forward expanding-window splits."""
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds
    splits = []
    for fold in range(n_folds):
        train_end_idx = min_train_days + fold * test_days_per_fold
        test_end_idx = train_end_idx + test_days_per_fold
        if fold == n_folds - 1:
            test_end_idx = n_days
        train_d = days[:train_end_idx]
        test_d = days[train_end_idx:test_end_idx]
        if len(test_d) > 0:
            splits.append((train_d, test_d))
    return splits


# ── Trading simulation ──────────────────────────────────────────────────

def simulate_trades(ohlcv: pd.DataFrame, predicted_indices: np.ndarray,
                    stop_pts: float = STOP_PTS, target_pts: float = TARGET_PTS,
                    max_bars: int = MAX_BARS) -> Tuple[int, float, float, float, List[float]]:
    """
    Simulate fixed-stop/target trades at predicted reversal bars.

    Returns (n_trades, win_rate, mean_pnl, total_pnl, pnl_list).
    """
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0, []

    close = ohlcv['close'].values.astype(np.float64)
    high = ohlcv['high'].values.astype(np.float64)
    low = ohlcv['low'].values.astype(np.float64)
    trade_dir = ohlcv['trade_direction'].values
    n = len(close)

    wins = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n:
            continue
        direction = trade_dir[idx]
        if direction == 0:
            continue

        entry_price = close[idx]
        trade_pnl = 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:  # long
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    break
            else:  # short
                if high[j] >= entry_price + stop_pts:
                    trade_pnl = -stop_pts
                    break
                if low[j] <= entry_price - target_pts:
                    trade_pnl = target_pts
                    break

        pnl_list.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1

    n_trades = len(pnl_list)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0, []

    wr = wins / n_trades
    mean_pnl = np.mean(pnl_list)
    total_pnl = np.sum(pnl_list)
    return n_trades, wr, mean_pnl, total_pnl, pnl_list


# ── Step 2: Walk-Forward XGBoost Training ────────────────────────────────

def train_walk_forward(samples_df: pd.DataFrame, feature_cols: List[str],
                       ohlcv: pd.DataFrame, n_folds: int = 5,
                       min_train_days: int = 100,
                       thresholds: List[float] = None) -> Dict:
    """
    Train XGBoost with walk-forward CV and honest early stopping.

    Returns dict with fold results, feature importance, predictions.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 70)
    print("WALK-FORWARD XGBOOST TRAINING")
    print("=" * 70)

    days = sorted(samples_df['trading_day'].unique())
    splits = get_fold_splits(days, n_folds, min_train_days)

    y = (samples_df['outcome'] == 1).astype(int).values  # reversal=1, breakout=0
    pos_rate = y.mean()
    print(f"Samples: {len(samples_df):,}, positive rate (reversal): {pos_rate:.2%}")
    print(f"Features: {len(feature_cols)}")
    print(f"Folds: {len(splits)}, thresholds: {thresholds}")

    fold_results = []
    all_y_true = []
    all_y_prob = []
    all_test_indices = []
    all_feature_imp = {}

    for fold, (train_days, test_days) in enumerate(splits):
        print(f"\nFold {fold + 1}/{len(splits)}: "
              f"{len(train_days)} train days, {len(test_days)} test days")

        train_mask = samples_df['trading_day'].isin(train_days)
        test_mask = samples_df['trading_day'].isin(test_days)

        X_train_full = samples_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        X_test = samples_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_train_full = y[train_mask.values]
        y_test = y[test_mask.values]
        test_indices = samples_df.index[test_mask].values

        if len(y_test) == 0 or y_train_full.sum() < 5:
            print("  Skipping fold (insufficient data)")
            continue

        # Honest early stopping: temporal 80/20 split within training days
        n_train_days = len(train_days)
        val_split_idx = int(n_train_days * 0.8)
        val_days = set(train_days[val_split_idx:])
        train_day_values = samples_df.loc[train_mask, 'trading_day'].values
        val_within_train = np.array([d in val_days for d in train_day_values])
        train_within_train = ~val_within_train

        X_tr = X_train_full[train_within_train]
        y_tr = y_train_full[train_within_train]
        X_val = X_train_full[val_within_train]
        y_val = y_train_full[val_within_train]

        if len(y_val) == 0 or y_tr.sum() < 5:
            # Fallback: use full train without early stopping
            X_tr = X_train_full
            y_tr = y_train_full
            X_val = X_test
            y_val = y_test

        spw = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            min_child_weight=5,
            scale_pos_weight=spw,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            tree_method='hist',
            early_stopping_rounds=30,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        print(f"  AUC={auc:.3f}, best_iteration={model.best_iteration}")

        # Accumulate
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_test_indices.extend(test_indices.tolist())

        fold_results.append({
            'fold': fold,
            'train_days': len(train_days),
            'test_days': len(test_days),
            'train_samples': len(X_tr),
            'test_samples': len(X_test),
            'auc': auc,
            'n_positive': int(y_test.sum()),
            'n_total': len(y_test),
            'best_iteration': model.best_iteration,
        })

        # Feature importance
        imp = model.get_booster().get_score(importance_type='gain')
        for fname, score in imp.items():
            if fname.startswith('f'):
                fidx = int(fname[1:])
                if fidx < len(feature_cols):
                    real_name = feature_cols[fidx]
                    all_feature_imp[real_name] = all_feature_imp.get(real_name, 0) + score

    # Normalize feature importance
    if all_feature_imp:
        max_imp = max(all_feature_imp.values())
        if max_imp > 0:
            all_feature_imp = {k: v / max_imp for k, v in all_feature_imp.items()}

    # Overall AUC
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_test_indices = np.array(all_test_indices)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except ValueError:
        overall_auc = 0.5

    print(f"\nOverall OOS AUC: {overall_auc:.4f}")

    return {
        'fold_results': fold_results,
        'feature_importance': all_feature_imp,
        'all_y_true': all_y_true,
        'all_y_prob': all_y_prob,
        'all_test_indices': all_test_indices,
        'overall_auc': overall_auc,
        'feature_cols': feature_cols,
    }


# ── Step 3: Trading simulation at multiple thresholds ────────────────────

def evaluate_thresholds(train_result: Dict, ohlcv: pd.DataFrame,
                        thresholds: List[float] = None) -> Dict:
    """Evaluate trading performance at multiple probability thresholds."""
    if thresholds is None:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS — TRADING SIMULATION")
    print("=" * 70)

    all_y_true = train_result['all_y_true']
    all_y_prob = train_result['all_y_prob']
    all_test_indices = train_result['all_test_indices']

    threshold_results = {}

    print(f"\n{'Thresh':>7} {'N pred':>8} {'N trades':>9} {'WR':>7} {'E[PnL]':>8} "
          f"{'Total PnL':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 75)

    for thresh in thresholds:
        pred_mask = all_y_prob >= thresh
        pred_indices = all_test_indices[pred_mask]

        # Classification metrics
        y_pred = pred_mask.astype(int)
        tp = ((y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((y_pred == 1) & (all_y_true == 0)).sum()
        fn = ((y_pred == 0) & (all_y_true == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        # Trading simulation
        n_trades, wr, mean_pnl, total_pnl, pnl_list = simulate_trades(
            ohlcv, pred_indices
        )

        threshold_results[thresh] = {
            'n_predicted': int(pred_mask.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'n_trades': n_trades,
            'win_rate': wr,
            'mean_pnl': mean_pnl,
            'total_pnl': total_pnl,
            'pnl_list': pnl_list,
            'pred_indices': pred_indices,
        }

        print(f"{thresh:>7.2f} {pred_mask.sum():>8,} {n_trades:>9,} "
              f"{wr:>7.1%} {mean_pnl:>8.2f} {total_pnl:>10.1f} {precision:>10.1%} {recall:>8.1%}")

    return threshold_results


# ── Step 4: Post-analysis ────────────────────────────────────────────────

def run_post_analysis(train_result: Dict, threshold_results: Dict,
                      ohlcv: pd.DataFrame, samples_df: pd.DataFrame,
                      level_encoding_cols: List[str],
                      tracked_levels: Optional[List[str]] = None):
    """Feature importance, per-level breakdown, baselines, fold stability."""

    print("\n" + "=" * 70)
    print("POST-ANALYSIS")
    print("=" * 70)

    # ── Feature importance ──
    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    print("\n── Top 25 Features by XGB Gain ──")
    print(f"{'Rank':>4} {'Feature':<40} {'Importance':>10} {'Level-Enc':>10}")
    print("-" * 68)
    for i, (feat, imp) in enumerate(fi_sorted[:25]):
        is_level = "***" if feat in level_encoding_cols else ""
        print(f"{i+1:>4} {feat:<40} {imp:>10.4f} {is_level:>10}")

    # Check: do level-encoding features rank in top 10?
    top10_names = [f[0] for f in fi_sorted[:10]]
    level_in_top10 = [f for f in top10_names if f in level_encoding_cols]
    print(f"\nLevel-encoding features in top 10: {level_in_top10 if level_in_top10 else 'NONE'}")

    # ── Per-level WR breakdown ──
    # Use best threshold (highest total PnL)
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    best_res = threshold_results[best_thresh]
    pred_indices = best_res['pred_indices']

    print(f"\n── Per-Level Breakdown (threshold={best_thresh:.2f}) ──")
    print(f"{'Level':<15} {'N trades':>9} {'WR':>7} {'E[PnL]':>8} {'Total PnL':>10}")
    print("-" * 55)

    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)
    for lvl in tracked_levels:
        lvl_mask = ohlcv.loc[pred_indices, 'nearest_level_name'].values == lvl
        lvl_indices = pred_indices[lvl_mask]
        if len(lvl_indices) == 0:
            print(f"{lvl:<15} {'—':>9}")
            continue
        nt, wr, mp, tp, _ = simulate_trades(ohlcv, lvl_indices)
        print(f"{lvl:<15} {nt:>9,} {wr:>7.1%} {mp:>8.2f} {tp:>10.1f}")

    # ── Baselines ──
    print(f"\n── Baseline Comparisons ──")
    near_mask = samples_df['outcome'].isin([0, 1])
    all_near_indices = samples_df.index[near_mask].values

    # Baseline 1: trade every near-level bar
    nt, wr, mp, tp, _ = simulate_trades(ohlcv, all_near_indices)
    print(f"  All near-level bars:  N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # Baseline 2: trade only ovn_hi/ovn_lo bars
    ovn_mask = near_mask & samples_df['nearest_level_name'].isin(['ovn_hi', 'ovn_lo'])
    ovn_indices = samples_df.index[ovn_mask].values
    nt, wr, mp, tp, _ = simulate_trades(ohlcv, ovn_indices)
    print(f"  OVN levels only:      N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # Model at best threshold
    nt, wr, mp, tp = best_res['n_trades'], best_res['win_rate'], best_res['mean_pnl'], best_res['total_pnl']
    print(f"  Model (t={best_thresh:.2f}):      N={nt:>6,}  WR={wr:.1%}  E[PnL]={mp:>+.2f}  Total={tp:>+.1f}")

    # ── Per-fold stability ──
    print(f"\n── Per-Fold Stability ──")
    print(f"{'Fold':>4} {'Train':>6} {'Test':>5} {'AUC':>7} {'N pos':>6}")
    print("-" * 32)
    for fr in train_result['fold_results']:
        print(f"{fr['fold']+1:>4} {fr['train_days']:>6} {fr['test_days']:>5} "
              f"{fr['auc']:>7.3f} {fr['n_positive']:>6}")

    auc_values = [fr['auc'] for fr in train_result['fold_results']]
    print(f"  AUC mean={np.mean(auc_values):.3f}, std={np.std(auc_values):.3f}")


# ── Step 5: Figures ──────────────────────────────────────────────────────

def save_figures(train_result: Dict, threshold_results: Dict,
                 ohlcv: pd.DataFrame, samples_df: pd.DataFrame,
                 level_encoding_cols: List[str],
                 tracked_levels: Optional[List[str]] = None):
    """Save all analysis figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(FIGURE_DIR, exist_ok=True)

    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:25]

    # ── 1. Feature importance bar chart ──
    fig, ax = plt.subplots(figsize=(12, 9))
    names = [f[0] for f in fi_sorted]
    values = [f[1] for f in fi_sorted]
    colors = ['#e74c3c' if n in level_encoding_cols else '#3498db' for n in names]
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Normalized Gain')
    ax.set_title('Feature Importance (red = level-encoding features)')
    ax.invert_yaxis()
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Level-encoding'),
                       Patch(facecolor='#3498db', label='Base features')]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

    # ── 2. Per-fold AUC bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5))
    folds = [fr['fold'] + 1 for fr in train_result['fold_results']]
    aucs = [fr['auc'] for fr in train_result['fold_results']]
    ax.bar(folds, aucs, color='#2ecc71')
    ax.axhline(y=np.mean(aucs), color='black', linestyle='--', label=f'Mean AUC={np.mean(aucs):.3f}')
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Per-Fold AUC')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'per_fold_auc.png'), dpi=150)
    plt.close()

    # ── 3. Per-level-type WR comparison (model vs no-model) ──
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    pred_indices = threshold_results[best_thresh]['pred_indices']

    level_wr_model = {}
    level_wr_baseline = {}
    near_mask = samples_df['outcome'].isin([0, 1])

    tracked_levels = tracked_levels or list(BASE_TRACKED_LEVELS)
    for lvl in tracked_levels:
        # Model
        lvl_pred_mask = ohlcv.loc[pred_indices, 'nearest_level_name'].values == lvl
        lvl_pred = pred_indices[lvl_pred_mask]
        if len(lvl_pred) > 0:
            nt, wr, _, _, _ = simulate_trades(ohlcv, lvl_pred)
            level_wr_model[lvl] = wr if nt > 0 else np.nan
        else:
            level_wr_model[lvl] = np.nan

        # Baseline
        lvl_base_mask = near_mask & (samples_df['nearest_level_name'] == lvl)
        lvl_base = samples_df.index[lvl_base_mask].values
        if len(lvl_base) > 0:
            nt, wr, _, _, _ = simulate_trades(ohlcv, lvl_base)
            level_wr_baseline[lvl] = wr if nt > 0 else np.nan
        else:
            level_wr_baseline[lvl] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tracked_levels))
    width = 0.35
    model_wrs = [level_wr_model.get(l, 0) for l in tracked_levels]
    base_wrs = [level_wr_baseline.get(l, 0) for l in tracked_levels]
    ax.bar(x - width/2, base_wrs, width, label='Baseline (all bars)', color='#95a5a6')
    ax.bar(x + width/2, model_wrs, width, label=f'Model (t={best_thresh:.2f})', color='#e74c3c')
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Level Type')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate by Level Type: Model vs Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(tracked_levels, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'per_level_wr.png'), dpi=150)
    plt.close()

    # ── 4. Equity curve ──
    pnl_list = threshold_results[best_thresh]['pnl_list']
    if pnl_list:
        fig, ax = plt.subplots(figsize=(12, 5))
        cum_pnl = np.cumsum(pnl_list)
        ax.plot(cum_pnl, color='#2ecc71', linewidth=1.0)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative PnL (points)')
        ax.set_title(f'Equity Curve (threshold={best_thresh:.2f}, N={len(pnl_list)} trades)')
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=np.array(cum_pnl) >= 0, alpha=0.1, color='green')
        ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                         where=np.array(cum_pnl) < 0, alpha=0.1, color='red')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, 'equity_curve.png'), dpi=150)
        plt.close()

    # ── 5. Threshold analysis ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    thresholds_sorted = sorted(threshold_results.keys())
    wrs = [threshold_results[t]['win_rate'] for t in thresholds_sorted]
    nts = [threshold_results[t]['n_trades'] for t in thresholds_sorted]
    mpnls = [threshold_results[t]['mean_pnl'] for t in thresholds_sorted]

    ax1.plot(thresholds_sorted, wrs, 'o-', color='#e74c3c', label='Win Rate')
    ax1.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rate vs Threshold')
    ax1.legend()

    ax1b = ax1.twinx()
    ax1b.bar(thresholds_sorted, nts, width=0.02, alpha=0.3, color='#3498db', label='N trades')
    ax1b.set_ylabel('N trades')
    ax1b.legend(loc='lower right')

    ax2.plot(thresholds_sorted, mpnls, 's-', color='#2ecc71')
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('E[PnL] per trade (pts)')
    ax2.set_title('Expected PnL vs Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'threshold_analysis.png'), dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGURE_DIR}/")


# ── Model saving ─────────────────────────────────────────────────────────

def save_model(samples_df: pd.DataFrame, feature_cols: List[str],
               ohlcv: pd.DataFrame, train_result: Dict,
               model_dir: str = 'models/reversal_phase3',
               threshold: float = 0.50,
               same_day_bidask_only: bool = False,
               tracked_levels: Optional[List[str]] = None,
               label_side_hysteresis_pts: float = 0.0,
               label_side_flip_confirm_pts: Optional[float] = None,
               label_drop_ambiguous_direction: bool = False,
               label_ambiguity_bar_tolerance: int = 2,
               labeling_scheme: str = "execution",
               label_forward_bars: int = 45,
               label_reversal_pts: float = 6.0,
               label_breakout_pts: float = 4.0,
               extra_metadata: Optional[Dict[str, object]] = None) -> None:
    """
    Train a final model on all data and save artifacts for realtime use.

    Saves:
      - model.json: XGBoost model
      - metadata.json: feature columns, levels, config
      - reference_predictions.npz: OOS predictions from walk-forward for playback comparison
    """
    import xgboost as xgb

    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 70)

    # Train final model on ALL data
    y = (samples_df['outcome'] == 1).astype(int).values
    X = samples_df[feature_cols].fillna(0).values.astype(np.float32)

    # Use last 20% for honest early stopping
    days = sorted(samples_df['trading_day'].unique())
    val_split_idx = int(len(days) * 0.8)
    val_days = set(days[val_split_idx:])
    day_values = samples_df['trading_day'].values
    val_mask = np.array([d in val_days for d in day_values])
    train_mask = ~val_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    spw = max(1.0, (1 - y_tr.mean()) / max(y_tr.mean(), 1e-6))

    model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=5,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        tree_method='hist',
        early_stopping_rounds=30,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Final model trained: {len(X_tr):,} train, {len(X_val):,} val, "
          f"best_iteration={model.best_iteration}")

    # Save model
    model_path = os.path.join(model_dir, 'model.json')
    model.save_model(model_path)
    print(f"  Model saved to {model_path}")

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'tracked_levels': list(tracked_levels or BASE_TRACKED_LEVELS),
        'threshold': threshold,
        'same_day_bidask_only': bool(same_day_bidask_only),
        'stop_pts': STOP_PTS,
        'target_pts': TARGET_PTS,
        'max_bars': MAX_BARS,
        'proximity_pts': 5.0,
        'label_side_hysteresis_pts': float(label_side_hysteresis_pts),
        'label_side_flip_confirm_pts': (
            float(label_side_flip_confirm_pts) if label_side_flip_confirm_pts is not None else None
        ),
        'label_drop_ambiguous_direction': bool(label_drop_ambiguous_direction),
        'label_ambiguity_bar_tolerance': int(label_ambiguity_bar_tolerance),
        'labeling_scheme': str(labeling_scheme),
        'label_forward_bars': int(label_forward_bars),
        'label_reversal_pts': float(label_reversal_pts),
        'label_breakout_pts': float(label_breakout_pts),
        'n_features': len(feature_cols),
        'n_train_samples': int(len(X_tr)),
        'n_val_samples': int(len(X_val)),
        'best_iteration': int(model.best_iteration),
        'oos_auc': float(train_result['overall_auc']),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {metadata_path}")

    # Save reference predictions (from walk-forward OOS)
    ref_path = os.path.join(model_dir, 'reference_predictions.npz')
    test_indices = train_result['all_test_indices']
    test_trading_days = samples_df.loc[test_indices, 'trading_day'].values
    test_nearest_level = samples_df.loc[test_indices, 'nearest_level_name'].values

    np.savez(
        ref_path,
        y_true=train_result['all_y_true'],
        y_prob=train_result['all_y_prob'],
        test_indices=test_indices,
        trading_days=test_trading_days,
        nearest_level=test_nearest_level,
    )
    print(f"  Reference predictions saved to {ref_path} "
          f"({len(test_indices):,} samples)")


def _compute_probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.float64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    if y_true.size == 0:
        return {
            "pos_mean_prob": 0.0,
            "neg_mean_prob": 0.0,
            "prob_cohens_d": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "pr_auc": 0.0,
        }

    pos = y_prob[y_true >= 0.5]
    neg = y_prob[y_true < 0.5]
    pos_mean = float(np.mean(pos)) if pos.size else 0.0
    neg_mean = float(np.mean(neg)) if neg.size else 0.0

    if pos.size > 1 and neg.size > 1:
        pos_var = float(np.var(pos, ddof=1))
        neg_var = float(np.var(neg, ddof=1))
        pooled = np.sqrt(max((pos_var + neg_var) / 2.0, 1e-12))
        d = float((pos_mean - neg_mean) / pooled)
    else:
        d = 0.0

    brier = float(np.mean((y_prob - y_true) ** 2))
    try:
        from sklearn.metrics import average_precision_score
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        if b1 >= 1.0:
            m = (y_prob >= b0) & (y_prob <= b1)
        else:
            m = (y_prob >= b0) & (y_prob < b1)
        if not np.any(m):
            continue
        conf = float(np.mean(y_prob[m]))
        acc = float(np.mean(y_true[m]))
        ece += float(np.mean(m)) * abs(acc - conf)

    return {
        "pos_mean_prob": pos_mean,
        "neg_mean_prob": neg_mean,
        "prob_cohens_d": d,
        "brier": brier,
        "ece": float(ece),
        "pr_auc": pr_auc,
    }


def _build_summary_payload(
    args: argparse.Namespace,
    tracked_levels: List[str],
    all_feature_cols: List[str],
    train_result: Dict,
    threshold_results: Dict,
) -> Dict:
    threshold_rows = {}
    for thr, row in threshold_results.items():
        threshold_rows[f"{float(thr):.2f}"] = {
            "n_predicted": int(row["n_predicted"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "n_trades": int(row["n_trades"]),
            "win_rate": float(row["win_rate"]),
            "mean_pnl": float(row["mean_pnl"]),
            "total_pnl": float(row["total_pnl"]),
        }

    by_total = max(threshold_results.keys(), key=lambda t: threshold_results[t]["total_pnl"])
    by_mean = max(
        threshold_results.keys(),
        key=lambda t: threshold_results[t]["mean_pnl"],
    )
    top_features = sorted(
        train_result["feature_importance"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )[:20]

    return {
        "config": {
            "data": args.data,
            "n_folds": int(args.n_folds),
            "min_train_days": int(args.min_train_days),
            "same_day_bidask_only": bool(args.same_day_bidask_only),
            "include_ib_relative_levels": bool(args.include_ib_relative_levels),
            "ib_relative_mode": str(args.ib_relative_mode),
            "tracked_levels": list(tracked_levels),
            "n_tracked_levels": int(len(tracked_levels)),
            "n_features": int(len(all_feature_cols)),
            "labeling_scheme": str(args.labeling_scheme),
            "sample_mode": str(args.sample_mode),
            "sample_episode_gap_bars": int(args.sample_episode_gap_bars),
            "label_forward_bars": (
                int(args.label_forward_bars) if args.label_forward_bars is not None else None
            ),
            "label_reversal_pts": (
                float(args.label_reversal_pts) if args.label_reversal_pts is not None else None
            ),
            "label_breakout_pts": (
                float(args.label_breakout_pts) if args.label_breakout_pts is not None else None
            ),
            "label_excursion_ratio": float(args.label_excursion_ratio),
            "label_breakout_excursion_ratio": float(args.label_breakout_excursion_ratio),
            "label_terminal_disp_frac": float(args.label_terminal_disp_frac),
            "label_use_volatility_scaling": bool(args.label_use_volatility_scaling),
            "label_volatility_col": str(args.label_volatility_col),
            "label_reversal_vol_mult": float(args.label_reversal_vol_mult),
            "label_breakout_vol_mult": float(args.label_breakout_vol_mult),
            "include_intraday_regime_features": bool(args.include_intraday_regime_features),
            "include_extended_episode_features": bool(args.include_extended_episode_features),
            "label_drop_near_threshold_ambiguity": bool(args.label_drop_near_threshold_ambiguity),
            "label_ambiguity_margin_frac": float(args.label_ambiguity_margin_frac),
            "label_ambiguity_ratio_buffer": float(args.label_ambiguity_ratio_buffer),
        },
        "overall_auc": float(train_result["overall_auc"]),
        "probability_metrics": _compute_probability_metrics(
            y_true=train_result["all_y_true"],
            y_prob=train_result["all_y_prob"],
        ),
        "best_threshold_by_total_pnl": {
            "threshold": float(by_total),
            **threshold_rows[f"{float(by_total):.2f}"],
        },
        "best_threshold_by_mean_pnl": {
            "threshold": float(by_mean),
            **threshold_rows[f"{float(by_mean):.2f}"],
        },
        "threshold_results": threshold_rows,
        "top_features": [{"feature": str(k), "importance": float(v)} for k, v in top_features],
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Single-model training with level-quality encoding")
    parser.add_argument('--data', default=DATA_PATH)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-train-days', type=int, default=100)
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model artifacts to models/reversal_phase3/')
    parser.add_argument('--model-dir', default='models/reversal_phase3',
                        help='Directory to save model artifacts')
    parser.add_argument(
        '--same-day-bidask-only',
        action='store_true',
        help='Use same-trading-day-only normalization for bid/ask-derived delta features.',
    )
    parser.add_argument(
        '--include-intraday-regime-features',
        action='store_true',
        help='Add causal intraday regime/context features (opening drive, gap, session trend).',
    )
    parser.add_argument(
        '--include-extended-episode-features',
        action='store_true',
        help='Add richer causal episode-state features for repeated level tests.',
    )
    parser.add_argument(
        '--include-ib-relative-levels',
        action='store_true',
        help='Promote IB-relative references into tracked levels for labeling/encoding.',
    )
    parser.add_argument(
        '--ib-relative-mode',
        choices=['core', 'full'],
        default='core',
        help='IB-relative level set when --include-ib-relative-levels is enabled.',
    )
    parser.add_argument(
        '--tracked-levels',
        nargs='*',
        default=None,
        help='Explicit tracked levels override (space separated).',
    )
    parser.add_argument(
        '--labeling-scheme',
        choices=['execution', 'structural', 'major_move'],
        default='execution',
        help='Labeling target: execution, structural, or excursion-quality major_move.',
    )
    parser.add_argument(
        '--label-forward-bars',
        type=int,
        default=None,
        help='Forward bars for label outcome (default: 45 execution, 60 structural).',
    )
    parser.add_argument(
        '--label-reversal-pts',
        type=float,
        default=None,
        help='Reversal move threshold in points (default: 6 execution, 12 structural).',
    )
    parser.add_argument(
        '--label-breakout-pts',
        type=float,
        default=None,
        help='Breakout move threshold in points (default: 4 execution, 6 structural).',
    )
    parser.add_argument(
        '--label-side-hysteresis',
        type=float,
        default=0.0,
        help='Deadband (points) for sticky per-level side assignment during labeling.',
    )
    parser.add_argument(
        '--label-side-flip-confirm',
        type=float,
        default=None,
        help='Optional stronger opposite-side cross (points) required before label side flips.',
    )
    parser.add_argument(
        '--label-excursion-ratio',
        type=float,
        default=1.4,
        help='For major_move labeling: min aligned/adverse excursion ratio for reversal label.',
    )
    parser.add_argument(
        '--label-breakout-excursion-ratio',
        type=float,
        default=1.2,
        help='For major_move labeling: min adverse/aligned excursion ratio for breakout label.',
    )
    parser.add_argument(
        '--label-terminal-disp-frac',
        type=float,
        default=0.20,
        help='For major_move labeling: required terminal displacement as fraction of threshold.',
    )
    parser.add_argument(
        '--label-use-volatility-scaling',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='For major_move labeling: scale min excursion thresholds by volatility floor.',
    )
    parser.add_argument(
        '--label-volatility-col',
        default='daily_atr_14',
        help='Volatility column used for major_move threshold scaling.',
    )
    parser.add_argument(
        '--label-reversal-vol-mult',
        type=float,
        default=0.10,
        help='For major_move labeling: reversal threshold floor = max(pts, vol_mult * volatility_col).',
    )
    parser.add_argument(
        '--label-breakout-vol-mult',
        type=float,
        default=0.08,
        help='For major_move labeling: breakout threshold floor = max(pts, vol_mult * volatility_col).',
    )
    parser.add_argument(
        '--label-drop-ambiguous-direction',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Mark bars as inconclusive when opposite-side interpretation gives conflicting outcome.',
    )
    parser.add_argument(
        '--label-ambiguity-bar-tolerance',
        type=int,
        default=2,
        help='When dropping ambiguous labels, require opposite outcome within this many bars of assigned outcome.',
    )
    parser.add_argument(
        '--label-drop-near-threshold-ambiguity',
        action='store_true',
        help='For major_move labeling: drop labels that only barely clear the excursion/terminal thresholds.',
    )
    parser.add_argument(
        '--label-ambiguity-margin-frac',
        type=float,
        default=0.10,
        help='For major_move ambiguity filtering: minimum excess over move/terminal threshold, as fraction of threshold.',
    )
    parser.add_argument(
        '--label-ambiguity-ratio-buffer',
        type=float,
        default=0.10,
        help='For major_move ambiguity filtering: minimum excess over excursion-ratio threshold.',
    )
    parser.add_argument(
        '--output-json',
        default='',
        help='Optional path to save compact training summary JSON.',
    )
    parser.add_argument(
        '--sample-mode',
        choices=['all', 'episode_first'],
        default='all',
        help='Sample selection mode before model training.',
    )
    parser.add_argument(
        '--sample-episode-gap-bars',
        type=int,
        default=10,
        help='When sample-mode=episode_first, gap (bars) that starts a new episode.',
    )
    args = parser.parse_args()

    t0 = time.time()
    tracked_levels = resolve_tracked_levels(
        tracked_levels_override=args.tracked_levels,
        include_ib_relative=bool(args.include_ib_relative_levels),
        ib_relative_mode=str(args.ib_relative_mode),
    )
    print(f"Tracked levels ({len(tracked_levels)}): {tracked_levels}")

    # ── Load data and compute features ──
    ohlcv = load_data(args.data)
    ohlcv = compute_levels(ohlcv, tracked_levels=tracked_levels)
    ohlcv, feature_cols = compute_all_features(
        ohlcv,
        same_day_bidask_only=args.same_day_bidask_only,
        include_intraday_regime_features=bool(args.include_intraday_regime_features),
    )

    # ── Label reversals/breakouts ──
    if args.labeling_scheme == 'structural':
        label_forward_bars = int(args.label_forward_bars) if args.label_forward_bars is not None else 60
        label_reversal_pts = float(args.label_reversal_pts) if args.label_reversal_pts is not None else 12.0
        label_breakout_pts = float(args.label_breakout_pts) if args.label_breakout_pts is not None else 6.0
        label_side_hysteresis = float(args.label_side_hysteresis) if args.label_side_hysteresis > 0 else 0.5
        ohlcv = label_reversals_breakouts_structural(
            ohlcv,
            tracked_levels=tracked_levels,
            proximity_pts=5.0,
            forward_bars=label_forward_bars,
            reversal_pts=label_reversal_pts,
            breakout_pts=label_breakout_pts,
            side_hysteresis_pts=label_side_hysteresis,
            side_flip_confirm_pts=(
                float(args.label_side_flip_confirm)
                if args.label_side_flip_confirm is not None else None
            ),
            drop_ambiguous_direction=(
                bool(args.label_drop_ambiguous_direction)
                if args.label_drop_ambiguous_direction is not None else False
            ),
            ambiguity_bar_tolerance=int(args.label_ambiguity_bar_tolerance),
        )
    elif args.labeling_scheme == 'major_move':
        label_forward_bars = int(args.label_forward_bars) if args.label_forward_bars is not None else 60
        label_reversal_pts = float(args.label_reversal_pts) if args.label_reversal_pts is not None else 6.0
        label_breakout_pts = float(args.label_breakout_pts) if args.label_breakout_pts is not None else 4.0
        label_side_hysteresis = float(args.label_side_hysteresis) if args.label_side_hysteresis > 0 else 0.5
        ohlcv = label_reversals_major_move(
            ohlcv,
            tracked_levels=tracked_levels,
            proximity_pts=5.0,
            forward_bars=label_forward_bars,
            reversal_pts=label_reversal_pts,
            breakout_pts=label_breakout_pts,
            side_hysteresis_pts=label_side_hysteresis,
            side_flip_confirm_pts=(
                float(args.label_side_flip_confirm)
                if args.label_side_flip_confirm is not None else None
            ),
            excursion_ratio=float(args.label_excursion_ratio),
            breakout_excursion_ratio=float(args.label_breakout_excursion_ratio),
            terminal_disp_frac=float(args.label_terminal_disp_frac),
            use_volatility_scaling=bool(args.label_use_volatility_scaling),
            volatility_col=str(args.label_volatility_col),
            reversal_vol_mult=float(args.label_reversal_vol_mult),
            breakout_vol_mult=float(args.label_breakout_vol_mult),
            drop_near_threshold_ambiguity=bool(args.label_drop_near_threshold_ambiguity),
            ambiguity_margin_frac=float(args.label_ambiguity_margin_frac),
            ambiguity_ratio_buffer=float(args.label_ambiguity_ratio_buffer),
        )
    else:
        label_forward_bars = int(args.label_forward_bars) if args.label_forward_bars is not None else 45
        label_reversal_pts = float(args.label_reversal_pts) if args.label_reversal_pts is not None else 6.0
        label_breakout_pts = float(args.label_breakout_pts) if args.label_breakout_pts is not None else 4.0
        label_side_hysteresis = float(args.label_side_hysteresis)
        ohlcv = label_reversals_breakouts(
            ohlcv,
            tracked_levels=tracked_levels,
            forward_bars=label_forward_bars,
            reversal_pts=label_reversal_pts,
            breakout_pts=label_breakout_pts,
            side_hysteresis_pts=label_side_hysteresis,
            side_flip_confirm_pts=(
                float(args.label_side_flip_confirm)
                if args.label_side_flip_confirm is not None else None
            ),
            drop_ambiguous_direction=(
                bool(args.label_drop_ambiguous_direction)
                if args.label_drop_ambiguous_direction is not None else False
            ),
            ambiguity_bar_tolerance=int(args.label_ambiguity_bar_tolerance),
        )

    # ── Add level-encoding features ──
    ohlcv, level_encoding_cols = compute_level_encoding_features(
        ohlcv,
        tracked_levels=tracked_levels,
        include_extended_episode_features=bool(args.include_extended_episode_features),
    )
    all_feature_cols = feature_cols + level_encoding_cols
    all_feature_cols = list(dict.fromkeys(all_feature_cols))

    # ── Filter to reversal/breakout samples only ──
    samples_mask = ohlcv['outcome'].isin([0, 1])
    samples_df = ohlcv.loc[samples_mask].copy()
    if args.sample_mode == 'episode_first':
        before = len(samples_df)
        samples_df = reduce_to_episode_first_samples(
            samples_df,
            gap_bars=int(args.sample_episode_gap_bars),
        )
        print(
            f"Episode-first sample reduction: {before:,} -> {len(samples_df):,} "
            f"({100*len(samples_df)/max(before,1):.1f}% kept)"
        )
    print(f"\nTraining samples: {len(samples_df):,} "
          f"(reversal={( samples_df['outcome']==1).sum():,}, "
          f"breakout={(samples_df['outcome']==0).sum():,})")

    # ── Train walk-forward XGBoost ──
    train_result = train_walk_forward(
        samples_df, all_feature_cols, ohlcv,
        n_folds=args.n_folds,
        min_train_days=args.min_train_days,
    )

    # ── Evaluate thresholds ──
    threshold_results = evaluate_thresholds(train_result, ohlcv)

    # ── Save model ──
    if args.save_model:
        extra_md = {
            "sample_mode": str(args.sample_mode),
            "sample_episode_gap_bars": int(args.sample_episode_gap_bars),
            "label_excursion_ratio": float(args.label_excursion_ratio),
            "label_breakout_excursion_ratio": float(args.label_breakout_excursion_ratio),
            "label_terminal_disp_frac": float(args.label_terminal_disp_frac),
            "label_use_volatility_scaling": bool(args.label_use_volatility_scaling),
            "label_volatility_col": str(args.label_volatility_col),
            "label_reversal_vol_mult": float(args.label_reversal_vol_mult),
            "label_breakout_vol_mult": float(args.label_breakout_vol_mult),
            "include_intraday_regime_features": bool(args.include_intraday_regime_features),
            "include_extended_episode_features": bool(args.include_extended_episode_features),
            "label_drop_near_threshold_ambiguity": bool(args.label_drop_near_threshold_ambiguity),
            "label_ambiguity_margin_frac": float(args.label_ambiguity_margin_frac),
            "label_ambiguity_ratio_buffer": float(args.label_ambiguity_ratio_buffer),
        }
        save_model(samples_df, all_feature_cols, ohlcv, train_result,
                   model_dir=args.model_dir,
                   same_day_bidask_only=args.same_day_bidask_only,
                   tracked_levels=tracked_levels,
                   label_side_hysteresis_pts=float(label_side_hysteresis),
                   label_side_flip_confirm_pts=(
                       float(args.label_side_flip_confirm)
                       if args.label_side_flip_confirm is not None else None
                   ),
                   label_drop_ambiguous_direction=(
                       bool(args.label_drop_ambiguous_direction)
                       if args.label_drop_ambiguous_direction is not None
                       else False
                   ),
                   label_ambiguity_bar_tolerance=int(args.label_ambiguity_bar_tolerance),
                   labeling_scheme=str(args.labeling_scheme),
                   label_forward_bars=int(label_forward_bars),
                   label_reversal_pts=float(label_reversal_pts),
                   label_breakout_pts=float(label_breakout_pts),
                   extra_metadata=extra_md)

    # ── Post-analysis ──
    run_post_analysis(train_result, threshold_results, ohlcv, samples_df,
                      level_encoding_cols,
                      tracked_levels=tracked_levels)

    # ── Save figures ──
    if not args.skip_plots:
        save_figures(train_result, threshold_results, ohlcv, samples_df,
                     level_encoding_cols,
                     tracked_levels=tracked_levels)

    # ── Summary ──
    best_thresh = max(threshold_results.keys(),
                      key=lambda t: threshold_results[t]['total_pnl'])
    best = threshold_results[best_thresh]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall OOS AUC:      {train_result['overall_auc']:.4f}")
    print(f"Best threshold:       {best_thresh:.2f}")
    print(f"  N trades:           {best['n_trades']:,}")
    print(f"  Win rate:           {best['win_rate']:.1%}")
    print(f"  E[PnL/trade]:       {best['mean_pnl']:+.2f} pts")
    print(f"  Total PnL:          {best['total_pnl']:+.1f} pts")
    print(f"  Precision:          {best['precision']:.1%}")
    print(f"  Recall:             {best['recall']:.1%}")
    prob_metrics = _compute_probability_metrics(
        y_true=train_result["all_y_true"],
        y_prob=train_result["all_y_prob"],
    )
    print(
        f"  Prob sep (d):       {prob_metrics['prob_cohens_d']:+.3f} "
        f"(pos_mean={prob_metrics['pos_mean_prob']:.3f}, "
        f"neg_mean={prob_metrics['neg_mean_prob']:.3f})"
    )
    print(f"  Calibration:        Brier={prob_metrics['brier']:.4f}, ECE={prob_metrics['ece']:.4f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Success criteria check
    print(f"\n── Success Criteria ──")
    wr_pass = best['win_rate'] > 0.52
    pnl_pass = best['mean_pnl'] > 0.3
    print(f"  WR > 52%:           {'PASS' if wr_pass else 'FAIL'} ({best['win_rate']:.1%})")
    print(f"  E[PnL] > 0.3pt:    {'PASS' if pnl_pass else 'FAIL'} ({best['mean_pnl']:+.2f})")

    fi = train_result['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    top10_names = [f[0] for f in fi_sorted[:10]]
    level_in_top10 = [f for f in top10_names if f in level_encoding_cols]
    fi_pass = len(level_in_top10) > 0
    print(f"  Level features in top 10: {'PASS' if fi_pass else 'FAIL'} ({level_in_top10})")

    if args.output_json:
        payload = _build_summary_payload(
            args=args,
            tracked_levels=tracked_levels,
            all_feature_cols=all_feature_cols,
            train_result=train_result,
            threshold_results=threshold_results,
        )
        out_path = args.output_json
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved summary JSON: {out_path}")


if __name__ == '__main__':
    main()
