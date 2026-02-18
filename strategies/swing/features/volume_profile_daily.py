"""Daily volume profile evolution features (~12 features).

Captures how the VP shape changes over days — POC shifts, VA expansion/contraction,
profile shape evolution. Built on top of the VP metrics computed by DailyAggregator.
"""
import pandas as pd
import numpy as np


def compute_vp_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute VP evolution features from daily bars with VP metrics.

    Expects daily DataFrame to have columns from DailyAggregator:
        vp_poc_rel, vp_va_width_rel, vp_skew, vp_entropy, vp_n_peaks

    Args:
        daily: Daily DataFrame with VP metrics columns

    Returns:
        DataFrame with ~12 VP evolution features
    """
    feat = pd.DataFrame(index=daily.index)

    # Base VP metrics (already computed by aggregator)
    feat["vp_poc_rel"] = daily.get("vp_poc_rel", 0.5)
    feat["vp_va_width_rel"] = daily.get("vp_va_width_rel", 1.0)
    feat["vp_skew"] = daily.get("vp_skew", 0.0)
    feat["vp_entropy"] = daily.get("vp_entropy", 0.0)
    feat["vp_n_peaks"] = daily.get("vp_n_peaks", 1)

    # POC vs close distance (in range units)
    c = daily["close"]
    h = daily["high"]
    l = daily["low"]
    day_range = (h - l).replace(0, np.nan)
    poc_price = l + daily.get("vp_poc_rel", 0.5) * (h - l)
    feat["vp_poc_vs_close"] = (c - poc_price) / day_range

    # RTH range fraction (using close_position as proxy if no RTH data)
    # This is an approximation — in minute data we'd compute RTH vs total range

    # Evolution over days
    poc_rel = feat["vp_poc_rel"]
    va_width = feat["vp_va_width_rel"]

    feat["vp_poc_shift_5d"] = poc_rel - poc_rel.shift(5)
    feat["vp_va_width_trend_5d"] = va_width - va_width.shift(5)
    feat["vp_entropy_change_5d"] = feat["vp_entropy"] - feat["vp_entropy"].shift(5)

    # Rolling averages for regime detection
    feat["vp_va_width_10d_avg"] = va_width.rolling(10, min_periods=1).mean()
    feat["vp_n_peaks_5d_avg"] = feat["vp_n_peaks"].rolling(5, min_periods=1).mean()

    return feat.fillna(0.0)


FEATURE_NAMES = [
    "vp_poc_rel", "vp_va_width_rel", "vp_skew", "vp_entropy", "vp_n_peaks",
    "vp_poc_vs_close",
    "vp_poc_shift_5d", "vp_va_width_trend_5d", "vp_entropy_change_5d",
    "vp_va_width_10d_avg", "vp_n_peaks_5d_avg",
]
