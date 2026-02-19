#!/usr/bin/env python
"""
Phase 0-1: Per-Second Order Flow Data Pipeline + Feature Engineering

Loads per-second ES futures data, validates against existing 1-min data,
computes ~40 causal order flow features aggregated to 1-min resolution,
and caches results to parquet for downstream analysis.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/explore_sec_data.py
"""

import os
import sys
import time
import glob as globlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ── Config ──────────────────────────────────────────────────────────────
SEC_DATA_DIR = "/mnt/d/data/rtrader_sec_data_of"
MIN_DATA_PATH = "raw_data/es_min_3y_clean_td_gamma.csv"
CACHE_DIR = "strategies/data/sec_cache"
SESSION_BOUNDARY_HOUR = 15  # 3 PM PT = 6 PM ET (next trading day starts)

# Columns in seconds CSVs
SEC_COLS = {
    "BarEndingTime": "dt",
    "Series.Open": "open",
    "Series.High": "high",
    "Series.Low": "low",
    "Series.Close": "close",
    "Series.Trade": "trade",
    "Series.Volume": "volume",
    "BidVolume": "bidvolume",
    "AskVolume": "askvolume",
}


# ── Data Loading ────────────────────────────────────────────────────────
def list_quarter_files() -> list[Path]:
    """List all quarterly CSV files sorted by name."""
    pattern = os.path.join(SEC_DATA_DIR, "es_sec_*.csv")
    files = sorted(globlib.glob(pattern))
    return [Path(f) for f in files]


def load_quarter(filepath: Path) -> pd.DataFrame:
    """Load one quarter of per-second data, normalize columns, parse timestamps."""
    df = pd.read_csv(filepath)
    df = df.rename(columns=SEC_COLS)

    # Parse timestamps — format: MM/DD/YYYY HH:MM:SS AM/PM
    # Timestamps are already in Pacific Time (verified by cross-referencing
    # against 1-min data: sec 6:30 AM matches 1-min 06:30 PT, not 03:30 PT)
    df["dt"] = pd.to_datetime(df["dt"], format="%m/%d/%Y %I:%M:%S %p")

    # Drop trade column (not needed)
    df = df.drop(columns=["trade"], errors="ignore")

    # Basic validation
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] >= 0].copy()

    df = df.sort_values("dt").reset_index(drop=True)
    return df


def assign_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    """Assign trading_day based on session boundary (3 PM PT = 6 PM ET).

    Bars at or after 3:00 PM PT belong to the next trading day's session.
    """
    dates = df["dt"].dt.date
    hours = df["dt"].dt.hour

    # After session boundary → next calendar date as trading day
    next_day_mask = hours >= SESSION_BOUNDARY_HOUR
    trading_dates = dates.copy()
    trading_dates = pd.Series(
        np.where(
            next_day_mask,
            pd.to_datetime(dates) + pd.Timedelta(days=1),
            pd.to_datetime(dates),
        )
    ).dt.date.values

    df["trading_day"] = trading_dates
    return df


def load_1min_data() -> pd.DataFrame:
    """Load existing 1-min data for validation."""
    df = pd.read_csv(MIN_DATA_PATH)
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df.columns = df.columns.str.lower()
    return df


# ── Feature Aggregation ────────────────────────────────────────────────
def aggregate_to_minute(sec_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~32 order flow features from per-second data, aggregated to 1-min.

    Groups A-D are computed purely from seconds data.
    Group E (level-relative) is computed downstream after merging with level data.
    """
    # Floor to minute for grouping
    sec_df = sec_df.copy()
    sec_df["minute"] = sec_df["dt"].dt.floor("min")

    # Pre-compute per-second delta
    sec_df["delta"] = sec_df["bidvolume"] - sec_df["askvolume"]

    results = []
    grouped = sec_df.groupby("minute")
    total_groups = len(grouped)

    print(f"  Aggregating {total_groups:,} minutes from {len(sec_df):,} seconds...")

    for i, (minute_ts, grp) in enumerate(grouped):
        if i % 50000 == 0 and i > 0:
            print(f"    {i:,}/{total_groups:,} minutes processed...")

        row = {"minute": minute_ts}
        n = len(grp)

        vol = grp["volume"].values.astype(np.float64)
        bid = grp["bidvolume"].values.astype(np.float64)
        ask = grp["askvolume"].values.astype(np.float64)
        delta = grp["delta"].values.astype(np.float64)
        close_px = grp["close"].values.astype(np.float64)
        high_px = grp["high"].values.astype(np.float64)
        low_px = grp["low"].values.astype(np.float64)
        total_vol = vol.sum()

        if total_vol == 0 or n < 2:
            # Skip bars with no volume
            continue

        # ── Group A: Intra-Bar Footprint ──
        net_delta = delta.sum()
        row["fp_delta_imbalance"] = net_delta / total_vol

        # Delta at high/low prices
        bar_high = high_px.max()
        bar_low = low_px.min()
        at_high = close_px == bar_high
        at_low = close_px == bar_low
        row["fp_delta_at_high"] = delta[at_high].sum() if at_high.any() else 0.0
        row["fp_delta_at_low"] = delta[at_low].sum() if at_low.any() else 0.0

        denom = abs(row["fp_delta_at_low"]) if row["fp_delta_at_low"] != 0 else 1.0
        row["fp_high_low_delta_ratio"] = row["fp_delta_at_high"] / denom

        # POC (price of control) — price with most volume
        unique_prices = np.unique(close_px)
        if len(unique_prices) > 1:
            vol_by_price = {}
            delta_by_price = {}
            for j in range(n):
                p = close_px[j]
                vol_by_price[p] = vol_by_price.get(p, 0.0) + vol[j]
                delta_by_price[p] = delta_by_price.get(p, 0.0) + delta[j]
            poc_price = max(vol_by_price, key=vol_by_price.get)
            price_range = bar_high - bar_low
            row["fp_poc_position"] = (
                (poc_price - bar_low) / price_range if price_range > 0 else 0.5
            )
            row["fp_volume_at_poc_pct"] = vol_by_price[poc_price] / total_vol
            row["fp_delta_at_poc"] = delta_by_price.get(poc_price, 0.0)
        else:
            row["fp_poc_position"] = 0.5
            row["fp_volume_at_poc_pct"] = 1.0
            row["fp_delta_at_poc"] = net_delta

        row["fp_n_prices_traded"] = len(unique_prices)

        # ── Group B: Volume Dynamics ──
        half = n // 2
        vol_first = vol[:half].sum()
        vol_last = vol[half:].sum()
        row["vol_front_back_ratio"] = (
            vol_first / vol_last if vol_last > 0 else 2.0
        )

        # Volume acceleration (slope of per-second volume)
        if n >= 3:
            x = np.arange(n, dtype=np.float64)
            slope, _, _, _, _ = linregress(x, vol)
            row["vol_acceleration"] = slope
        else:
            row["vol_acceleration"] = 0.0

        med_vol = np.median(vol)
        row["vol_spike_ratio"] = vol.max() / med_vol if med_vol > 0 else 1.0
        row["vol_active_pct"] = (vol > 0).sum() / n

        # Taper score: last 10s vs first 10s
        first_10 = vol[:10].sum() if n >= 10 else vol[:max(1, n // 6)].sum()
        last_10 = vol[-10:].sum() if n >= 10 else vol[-max(1, n // 6):].sum()
        row["vol_taper_score"] = last_10 / first_10 if first_10 > 0 else 1.0

        # Max burst in any 5-second window
        if n >= 5:
            cumvol = np.cumsum(vol)
            burst_5s = np.max(cumvol[4:] - np.concatenate([[0], cumvol[:-5]]))
            row["vol_5s_max_burst"] = burst_5s
        else:
            row["vol_5s_max_burst"] = total_vol

        # Climax position (normalized 0-1)
        max_idx = np.argmax(vol)
        row["vol_climax_position"] = max_idx / max(n - 1, 1)

        # Last 10s volume percentage
        row["vol_last_10s_pct"] = last_10 / total_vol if total_vol > 0 else 0.0

        # ── Group C: Delta Dynamics ──
        row["delta_first_half"] = delta[:half].sum()
        row["delta_second_half"] = delta[half:].sum()

        # Running delta sign changes
        cum_delta = np.cumsum(delta)
        signs = np.sign(cum_delta)
        sign_changes = np.sum(signs[1:] != signs[:-1])
        row["delta_flip_within_bar"] = 1 if sign_changes > 0 else 0
        row["delta_flip_count"] = int(sign_changes)

        # Price direction vs delta direction
        price_dir = close_px[-1] - close_px[0]
        row["delta_divergence"] = (
            -1 if (price_dir > 0 and net_delta < 0) or (price_dir < 0 and net_delta > 0) else 1
        )

        # Absorption score
        row["delta_absorption_score"] = (
            total_vol / abs(net_delta) if abs(net_delta) > 0 else total_vol
        )

        # Store raw values for multi-bar features computed after aggregation
        row["_raw_delta"] = net_delta
        row["_raw_total_vol"] = total_vol

        # OHLCV for validation
        row["sec_open"] = grp["open"].iloc[0]
        row["sec_high"] = bar_high
        row["sec_low"] = bar_low
        row["sec_close"] = grp["close"].iloc[-1]
        row["sec_volume"] = int(total_vol)

        results.append(row)

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    df = df.sort_values("minute").reset_index(drop=True)

    # ── Multi-bar features (Group C continued + Group D) ──
    df = _compute_multi_bar_features(df)

    return df


def _compute_multi_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling/multi-bar features from aggregated minute data."""
    raw_delta = df["_raw_delta"].values
    raw_vol = df["_raw_total_vol"].values

    # Delta z-score over 60-bar window
    delta_roll_mean = pd.Series(raw_delta).rolling(60, min_periods=10).mean().values
    delta_roll_std = pd.Series(raw_delta).rolling(60, min_periods=10).std().values
    delta_roll_std = np.where(delta_roll_std == 0, 1.0, delta_roll_std)
    df["delta_zscore_60bar"] = (raw_delta - delta_roll_mean) / delta_roll_std

    # 3-bar delta momentum
    df["delta_3bar_momentum"] = pd.Series(raw_delta).rolling(3, min_periods=1).sum().values

    # ── Group D: Multi-Bar Approach ──
    # Volume slope over last 3 bars
    vol_series = pd.Series(raw_vol)
    df["approach_vol_slope_3bar"] = (
        vol_series.rolling(3, min_periods=2)
        .apply(lambda x: linregress(np.arange(len(x)), x).slope if len(x) >= 2 else 0.0, raw=False)
        .values
    )

    # Delta slope over last 3 bars
    delta_series = pd.Series(raw_delta)
    df["approach_delta_slope_3bar"] = (
        delta_series.rolling(3, min_periods=2)
        .apply(lambda x: linregress(np.arange(len(x)), x).slope if len(x) >= 2 else 0.0, raw=False)
        .values
    )

    # Price velocity / volume (speed vs vol)
    price_change = df["sec_close"].diff().abs().values
    df["approach_speed_vs_vol"] = np.where(
        raw_vol > 0, price_change / raw_vol * 1000, 0.0  # scale for readability
    )

    # Absorption over 3 bars
    abs_scores = df["delta_absorption_score"].values
    df["approach_absorption_3bar"] = (
        pd.Series(abs_scores).rolling(3, min_periods=1).mean().values
    )

    # Delta divergence from price over 3 bars
    price_3bar = df["sec_close"].diff(3).values
    delta_3bar = pd.Series(raw_delta).rolling(3, min_periods=1).sum().values
    df["approach_delta_diverge_3"] = np.where(
        (price_3bar > 0) & (delta_3bar < 0) | (price_3bar < 0) & (delta_3bar > 0),
        1, 0
    )

    # Volume climax in last 5 bars (vol > 2σ above rolling mean)
    vol_mean_20 = pd.Series(raw_vol).rolling(20, min_periods=5).mean().values
    vol_std_20 = pd.Series(raw_vol).rolling(20, min_periods=5).std().values
    vol_std_20 = np.where(vol_std_20 == 0, 1.0, vol_std_20)
    vol_z = (raw_vol - vol_mean_20) / vol_std_20
    climax_mask = vol_z > 2.0
    df["approach_vol_climax_5bar"] = (
        pd.Series(climax_mask.astype(float)).rolling(5, min_periods=1).max().values
    )

    # Exhaustion composite: declining vol + declining delta + declining speed
    vol_declining = (df["approach_vol_slope_3bar"] < 0).astype(float)
    delta_declining = (df["approach_delta_slope_3bar"].abs() < df["approach_delta_slope_3bar"].shift(1).abs()).astype(float)
    speed_declining = (df["approach_speed_vs_vol"] < pd.Series(df["approach_speed_vs_vol"]).shift(1)).astype(float)
    df["approach_exhaustion"] = (vol_declining + delta_declining + speed_declining) / 3.0

    # Aggressive volume % in last 3 bars
    # Aggressive = volume hitting bid (for sells) or lifting offer (for buys)
    # With bid/ask data, aggressive sell = ask volume, aggressive buy = bid volume
    # Total aggressive = max(bid, ask) per bar
    # Not available at minute level from aggregated data — use |delta|/volume as proxy
    df["approach_aggressive_pct"] = np.where(
        raw_vol > 0,
        pd.Series(np.abs(raw_delta) / raw_vol).rolling(3, min_periods=1).mean().values,
        0.0,
    )

    # Drop raw helper columns
    df = df.drop(columns=["_raw_delta", "_raw_total_vol"])

    return df


# ── Validation ──────────────────────────────────────────────────────────
def validate_against_1min(sec_agg: pd.DataFrame, min_df: pd.DataFrame) -> dict:
    """Cross-validate aggregated seconds data against existing 1-min data.

    Returns dict with match statistics.
    """
    # Align on minute timestamp
    sec_agg = sec_agg.set_index("minute")
    min_df = min_df.set_index("dt")

    common_idx = sec_agg.index.intersection(min_df.index)
    n_common = len(common_idx)

    if n_common == 0:
        return {"overlap_minutes": 0, "note": "No overlapping timestamps found"}

    sec_sub = sec_agg.loc[common_idx]
    min_sub = min_df.loc[common_idx]

    # OHLC comparison
    results = {
        "overlap_minutes": n_common,
        "overlap_days": len(set(common_idx.date)),
    }

    for col_sec, col_min in [
        ("sec_open", "open"),
        ("sec_high", "high"),
        ("sec_low", "low"),
        ("sec_close", "close"),
    ]:
        diff = (sec_sub[col_sec].values - min_sub[col_min].values).astype(float)
        results[f"{col_min}_exact_match_pct"] = (np.abs(diff) < 0.01).mean() * 100
        results[f"{col_min}_mean_abs_diff"] = np.abs(diff).mean()
        results[f"{col_min}_max_abs_diff"] = np.abs(diff).max()

    # Volume comparison
    if "volume" in min_sub.columns:
        vol_diff = sec_sub["sec_volume"].values - min_sub["volume"].values.astype(float)
        results["volume_exact_match_pct"] = (np.abs(vol_diff) < 1).mean() * 100
        results["volume_mean_abs_diff"] = np.abs(vol_diff).mean()

    # Bid/ask convention check
    # If sec bidvol correlates with 1min bidvol, convention is correct
    # If it correlates with askvol, they're swapped
    if "bidvolume" in min_sub.columns and "askvolume" in min_sub.columns:
        # We don't have per-minute bid/ask in aggregated form directly
        # but we can check delta direction vs price direction consistency
        pass

    return results


# ── Caching ─────────────────────────────────────────────────────────────
def cache_quarter(features_df: pd.DataFrame, quarter_name: str):
    """Save aggregated features to parquet."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"sec_features_{quarter_name}.parquet")
    features_df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Cached: {path} ({size_mb:.1f} MB, {len(features_df):,} rows)")


def load_cached_features() -> pd.DataFrame:
    """Load all cached quarter parquets into a single DataFrame."""
    pattern = os.path.join(CACHE_DIR, "sec_features_*.parquet")
    files = sorted(globlib.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cached features found in {CACHE_DIR}")
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("minute").reset_index(drop=True)
    print(f"Loaded {len(combined):,} minutes from {len(files)} cached quarters")
    return combined


# ── Feature Column Names ───────────────────────────────────────────────
ORDERFLOW_FEATURE_COLS = [
    # Group A: Intra-Bar Footprint
    "fp_delta_imbalance",
    "fp_delta_at_high",
    "fp_delta_at_low",
    "fp_high_low_delta_ratio",
    "fp_poc_position",
    "fp_volume_at_poc_pct",
    "fp_n_prices_traded",
    "fp_delta_at_poc",
    # Group B: Volume Dynamics
    "vol_front_back_ratio",
    "vol_acceleration",
    "vol_spike_ratio",
    "vol_taper_score",
    "vol_5s_max_burst",
    "vol_climax_position",
    "vol_last_10s_pct",
    # Group C: Delta Dynamics
    "delta_first_half",
    "delta_second_half",
    "delta_flip_within_bar",
    "delta_flip_count",
    "delta_divergence",
    "delta_absorption_score",
    "delta_zscore_60bar",
    "delta_3bar_momentum",
    # Group D: Multi-Bar Approach
    "approach_vol_slope_3bar",
    "approach_delta_slope_3bar",
    "approach_speed_vs_vol",
    "approach_absorption_3bar",
    "approach_delta_diverge_3",
    "approach_vol_climax_5bar",
    "approach_exhaustion",
    "approach_aggressive_pct",
]


# ── Level-Relative Features (Group E) ──────────────────────────────────
def compute_level_relative_features(
    sec_df: pd.DataFrame,
    min_df: pd.DataFrame,
    level_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Group E features that require knowing level prices.

    This is called at analysis time, not during caching.

    Args:
        sec_df: Raw per-second data with trading_day assigned
        min_df: 1-min data with level columns computed
        level_prices: DataFrame with (trading_day, minute) → nearest_level_price

    Returns:
        DataFrame indexed by minute with Group E features
    """
    from strategies.labeling.reversal_zones import TRACKED_LEVELS

    sec_df = sec_df.copy()
    sec_df["minute"] = sec_df["dt"].dt.floor("min")

    results = []

    for minute_ts, grp in sec_df.groupby("minute"):
        row = {"minute": minute_ts}

        # Get nearest level price for this minute from level_prices
        if minute_ts not in level_prices.index:
            continue

        level_price = level_prices.loc[minute_ts]
        if pd.isna(level_price):
            continue

        close_px = grp["close"].values
        vol = grp["volume"].values.astype(np.float64)
        bid = grp["bidvolume"].values.astype(np.float64)
        ask = grp["askvolume"].values.astype(np.float64)
        delta = bid - ask
        total_vol = vol.sum()

        # Near-level mask (within 1 point)
        near_level = np.abs(close_px - level_price) <= 1.0
        n_near = near_level.sum()

        if n_near == 0:
            # No seconds near the level
            row["level_sec_delta_at_touch"] = 0.0
            row["level_sec_vol_at_touch"] = 0.0
            row["level_sec_rejection_speed"] = 0.0
            row["level_sec_time_at_level"] = 0
            row["level_sec_retests"] = 0
            row["level_sec_bid_ask_at_level"] = 0.0
            row["level_sec_delta_before_touch"] = 0.0
            row["level_sec_touch_vol_ratio"] = 0.0
        else:
            # First touch index
            first_touch = np.argmax(near_level)

            # Delta/volume in 10s window around first touch
            window_start = max(0, first_touch - 5)
            window_end = min(len(grp), first_touch + 5)
            row["level_sec_delta_at_touch"] = delta[window_start:window_end].sum()
            row["level_sec_vol_at_touch"] = vol[window_start:window_end].sum()

            # Rejection speed: pts/sec in first 5s after touch
            after_touch = close_px[first_touch : min(first_touch + 5, len(close_px))]
            if len(after_touch) >= 2:
                row["level_sec_rejection_speed"] = abs(after_touch[-1] - after_touch[0]) / len(after_touch)
            else:
                row["level_sec_rejection_speed"] = 0.0

            # Time at level
            row["level_sec_time_at_level"] = int(n_near)

            # Retests (separate touches)
            touch_groups = np.diff(np.where(near_level)[0])
            row["level_sec_retests"] = 1 + (touch_groups > 3).sum()  # gap of >3s = new test

            # Bid/ask near level
            bid_at_lvl = bid[near_level].sum()
            ask_at_lvl = ask[near_level].sum()
            total_at_lvl = bid_at_lvl + ask_at_lvl
            row["level_sec_bid_ask_at_level"] = (
                (bid_at_lvl - ask_at_lvl) / total_at_lvl if total_at_lvl > 0 else 0.0
            )

            # Delta before touch
            pre_start = max(0, first_touch - 10)
            row["level_sec_delta_before_touch"] = delta[pre_start:first_touch].sum()

            # Touch volume ratio
            avg_vol_per_sec = total_vol / len(grp) if len(grp) > 0 else 1.0
            avg_touch_vol = vol[near_level].mean()
            row["level_sec_touch_vol_ratio"] = (
                avg_touch_vol / avg_vol_per_sec if avg_vol_per_sec > 0 else 1.0
            )

        results.append(row)

    return pd.DataFrame(results)


LEVEL_FEATURE_COLS = [
    "level_sec_delta_at_touch",
    "level_sec_vol_at_touch",
    "level_sec_rejection_speed",
    "level_sec_time_at_level",
    "level_sec_retests",
    "level_sec_bid_ask_at_level",
    "level_sec_delta_before_touch",
    "level_sec_touch_vol_ratio",
]

ALL_ORDERFLOW_COLS = ORDERFLOW_FEATURE_COLS + LEVEL_FEATURE_COLS


# ── Main Pipeline ──────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("=" * 70)
    print("Phase 0-1: Per-Second Order Flow Data Pipeline")
    print("=" * 70)

    # ── Step 1: Discover files ──
    files = list_quarter_files()
    print(f"\nFound {len(files)} quarterly files:")
    for f in files:
        print(f"  {f.name}")

    # ── Step 2: Load 1-min data for validation ──
    print(f"\nLoading 1-min reference data...")
    min_df = load_1min_data()
    print(f"  {len(min_df):,} bars, {min_df['trading_day'].nunique()} trading days")
    print(f"  Date range: {min_df['dt'].min()} → {min_df['dt'].max()}")

    # ── Step 3: Process each quarter ──
    all_validation = []

    for filepath in files:
        quarter_name = filepath.stem.replace("es_sec_", "")  # e.g., "2023q1"
        cache_path = os.path.join(CACHE_DIR, f"sec_features_{quarter_name}.parquet")

        # Skip if already cached
        if os.path.exists(cache_path):
            size_mb = os.path.getsize(cache_path) / 1024 / 1024
            print(f"\n[{quarter_name}] Already cached ({size_mb:.1f} MB) — skipping")
            continue

        print(f"\n{'─' * 50}")
        print(f"Processing {quarter_name} ({filepath.name})...")

        # Load
        t1 = time.time()
        sec_df = load_quarter(filepath)
        print(f"  Loaded: {len(sec_df):,} seconds ({time.time() - t1:.1f}s)")
        print(f"  Date range: {sec_df['dt'].min()} → {sec_df['dt'].max()}")

        # Assign trading day
        sec_df = assign_trading_day(sec_df)
        n_days = len(set(sec_df["trading_day"]))
        print(f"  Trading days: {n_days}")

        # Aggregate to minute
        t2 = time.time()
        features_df = aggregate_to_minute(sec_df)
        print(f"  Aggregated: {len(features_df):,} minutes ({time.time() - t2:.1f}s)")

        # Validate against 1-min data
        val_results = validate_against_1min(features_df.copy(), min_df.copy())
        val_results["quarter"] = quarter_name
        all_validation.append(val_results)

        if val_results.get("overlap_minutes", 0) > 0:
            print(f"  Validation ({val_results['overlap_minutes']:,} overlapping minutes):")
            for col in ["open", "high", "low", "close"]:
                match_pct = val_results.get(f"{col}_exact_match_pct", 0)
                mean_diff = val_results.get(f"{col}_mean_abs_diff", 0)
                print(f"    {col}: {match_pct:.1f}% exact, mean_diff={mean_diff:.4f}")
            vol_match = val_results.get("volume_exact_match_pct", 0)
            print(f"    volume: {vol_match:.1f}% exact match")
        else:
            print(f"  No overlap with 1-min data for this quarter")

        # Cache
        cache_quarter(features_df, quarter_name)

        # Free memory
        del sec_df, features_df

    # ── Step 4: Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Load all cached to verify
    try:
        all_features = load_cached_features()
        print(f"\nTotal cached: {len(all_features):,} minutes")
        print(f"Date range: {all_features['minute'].min()} → {all_features['minute'].max()}")
        print(f"Feature columns: {len(ORDERFLOW_FEATURE_COLS)}")
        print(f"\nFeature statistics:")
        for col in ORDERFLOW_FEATURE_COLS:
            if col in all_features.columns:
                vals = all_features[col].dropna()
                print(
                    f"  {col:35s} mean={vals.mean():>10.3f}  "
                    f"std={vals.std():>10.3f}  "
                    f"min={vals.min():>10.3f}  max={vals.max():>10.3f}"
                )
    except FileNotFoundError:
        print("No cached features found.")

    # Validation summary
    if all_validation:
        print(f"\nValidation across {len(all_validation)} quarters:")
        for v in all_validation:
            q = v.get("quarter", "?")
            n = v.get("overlap_minutes", 0)
            if n > 0:
                open_match = v.get("open_exact_match_pct", 0)
                print(f"  {q}: {n:,} overlap min, open_match={open_match:.1f}%")
            else:
                print(f"  {q}: no overlap")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")


if __name__ == "__main__":
    main()
