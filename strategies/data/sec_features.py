"""Utilities for per-second order-flow caches and feature alignment."""

from __future__ import annotations

import glob as globlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


SEC_DATA_DIR = "/mnt/d/data/rtrader_sec_data_of"
CACHE_DIR = "strategies/data/sec_cache"

# Source columns used in raw quarterly CSVs.
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


ORDERFLOW_FEATURE_COLS = [
    # Group A: Intra-bar footprint
    "fp_delta_imbalance",
    "fp_delta_at_high",
    "fp_delta_at_low",
    "fp_high_low_delta_ratio",
    "fp_poc_position",
    "fp_volume_at_poc_pct",
    "fp_n_prices_traded",
    "fp_delta_at_poc",
    # Group B: Volume dynamics
    "vol_front_back_ratio",
    "vol_acceleration",
    "vol_spike_ratio",
    "vol_taper_score",
    "vol_5s_max_burst",
    "vol_climax_position",
    "vol_last_10s_pct",
    # Group C: Delta dynamics
    "delta_first_half",
    "delta_second_half",
    "delta_flip_within_bar",
    "delta_flip_count",
    "delta_divergence",
    "delta_absorption_score",
    "delta_zscore_60bar",
    "delta_3bar_momentum",
    # Group D: Multi-bar approach
    "approach_vol_slope_3bar",
    "approach_delta_slope_3bar",
    "approach_speed_vs_vol",
    "approach_absorption_3bar",
    "approach_delta_diverge_3",
    "approach_vol_climax_5bar",
    "approach_exhaustion",
    "approach_aggressive_pct",
]


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


def list_quarter_files(sec_data_dir: str = SEC_DATA_DIR) -> list[Path]:
    """List quarterly second-level CSV files."""
    pattern = os.path.join(sec_data_dir, "es_sec_*.csv")
    files = sorted(globlib.glob(pattern))
    return [Path(f) for f in files]


def load_quarter(filepath: Path) -> pd.DataFrame:
    """Load one quarter of per-second data and normalize columns."""
    df = pd.read_csv(filepath)
    df = df.rename(columns=SEC_COLS)

    # Source timestamps are Pacific time wall-clock values.
    df["dt"] = pd.to_datetime(df["dt"], format="%m/%d/%Y %I:%M:%S %p")

    df = df.drop(columns=["trade"], errors="ignore")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] >= 0].copy()
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def load_cached_features(cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """Load all cached minute-level order-flow parquet files."""
    pattern = os.path.join(cache_dir, "sec_features_*.parquet")
    files = sorted(globlib.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No cached features found in {cache_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("minute").reset_index(drop=True)
    return combined


def merge_cached_orderflow_features(
    ohlcv: pd.DataFrame,
    dt_col: str = "dt",
    cache_dir: str = CACHE_DIR,
    feature_cols: Optional[list[str]] = None,
    add_missing_indicators: bool = False,
) -> tuple[pd.DataFrame, list[str], dict]:
    """Merge cached minute-level order-flow features into 1-min bars.

    Returns:
        (merged_df, merged_feature_cols, stats)
    """
    if dt_col not in ohlcv.columns:
        raise KeyError(f"Missing `{dt_col}` column in ohlcv DataFrame")

    of_df = load_cached_features(cache_dir=cache_dir).set_index("minute")
    base_cols = feature_cols or ORDERFLOW_FEATURE_COLS
    of_cols = [c for c in base_cols if c in of_df.columns]
    if not of_cols:
        raise ValueError("No order-flow feature columns found in cached data")

    result = ohlcv.copy()
    dt_vals = pd.to_datetime(result[dt_col]).values
    for col in of_cols:
        result[col] = of_df[col].reindex(dt_vals).values

    n_total = len(result)
    n_matched = int(result[of_cols[0]].notna().sum())
    merged_cols = list(of_cols)

    if add_missing_indicators:
        for col in of_cols:
            miss_col = f"{col}_is_missing"
            result[miss_col] = result[col].isna().astype(np.float32)
            merged_cols.append(miss_col)

    stats = {
        "n_total_bars": n_total,
        "n_matched_bars": n_matched,
        "coverage": n_matched / max(n_total, 1),
        "n_orderflow_cols": len(of_cols),
    }
    return result, merged_cols, stats


def compute_level_relative_features(
    sec_df: pd.DataFrame,
    level_prices: pd.Series,
) -> pd.DataFrame:
    """Compute per-minute second-level features relative to nearest level price."""
    sec_df = sec_df.copy()
    sec_df["minute"] = sec_df["dt"].dt.floor("min")
    results = []

    for minute_ts, grp in sec_df.groupby("minute"):
        if minute_ts not in level_prices.index:
            continue
        level_price = level_prices.loc[minute_ts]
        if pd.isna(level_price):
            continue

        row = {"minute": minute_ts}
        close_px = grp["close"].values
        vol = grp["volume"].values.astype(np.float64)
        bid = grp["bidvolume"].values.astype(np.float64)
        ask = grp["askvolume"].values.astype(np.float64)
        delta = bid - ask
        total_vol = vol.sum()

        near_level = np.abs(close_px - level_price) <= 1.0
        n_near = int(near_level.sum())

        if n_near == 0:
            row["level_sec_delta_at_touch"] = 0.0
            row["level_sec_vol_at_touch"] = 0.0
            row["level_sec_rejection_speed"] = 0.0
            row["level_sec_time_at_level"] = 0
            row["level_sec_retests"] = 0
            row["level_sec_bid_ask_at_level"] = 0.0
            row["level_sec_delta_before_touch"] = 0.0
            row["level_sec_touch_vol_ratio"] = 0.0
        else:
            first_touch = int(np.argmax(near_level))
            window_start = max(0, first_touch - 5)
            window_end = min(len(grp), first_touch + 5)
            row["level_sec_delta_at_touch"] = float(
                delta[window_start:window_end].sum()
            )
            row["level_sec_vol_at_touch"] = float(vol[window_start:window_end].sum())

            after_touch = close_px[first_touch : min(first_touch + 5, len(close_px))]
            if len(after_touch) >= 2:
                row["level_sec_rejection_speed"] = float(
                    abs(after_touch[-1] - after_touch[0]) / len(after_touch)
                )
            else:
                row["level_sec_rejection_speed"] = 0.0

            row["level_sec_time_at_level"] = n_near
            touch_groups = np.diff(np.where(near_level)[0])
            row["level_sec_retests"] = int(1 + (touch_groups > 3).sum())

            bid_at_lvl = bid[near_level].sum()
            ask_at_lvl = ask[near_level].sum()
            total_at_lvl = bid_at_lvl + ask_at_lvl
            row["level_sec_bid_ask_at_level"] = (
                (bid_at_lvl - ask_at_lvl) / total_at_lvl if total_at_lvl > 0 else 0.0
            )

            pre_start = max(0, first_touch - 10)
            row["level_sec_delta_before_touch"] = float(delta[pre_start:first_touch].sum())

            avg_vol_per_sec = total_vol / len(grp) if len(grp) > 0 else 1.0
            avg_touch_vol = vol[near_level].mean()
            row["level_sec_touch_vol_ratio"] = (
                avg_touch_vol / avg_vol_per_sec if avg_vol_per_sec > 0 else 1.0
            )

        results.append(row)

    if not results:
        return pd.DataFrame(columns=["minute"] + LEVEL_FEATURE_COLS)
    return pd.DataFrame(results)
