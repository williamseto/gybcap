"""Daily technical features for ES (~35 features).

Reuses RSI, BB, ATR logic from strategies/features/higher_timeframe.py.
All features are strictly causal (computed from data up to day t only).
"""
import pandas as pd
import numpy as np


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean().fillna(tr)


def compute_daily_technical(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute ~35 daily technical features from OHLCV daily bars.

    Args:
        daily: DataFrame with columns [open, high, low, close, volume], DatetimeIndex

    Returns:
        DataFrame with feature columns, same index
    """
    c = daily["close"]
    h = daily["high"]
    l = daily["low"]
    o = daily["open"]
    v = daily["volume"]

    feat = pd.DataFrame(index=daily.index)

    # --- Returns ---
    feat["return_1d"] = c.pct_change()
    feat["return_5d"] = c.pct_change(5)
    feat["return_20d"] = c.pct_change(20)
    feat["return_60d"] = c.pct_change(60)

    # --- Momentum ---
    feat["rsi_5"] = _compute_rsi(c, 5)
    feat["rsi_14"] = _compute_rsi(c, 14)

    # --- Trend: close vs SMAs ---
    for w in [20, 50, 200]:
        sma = c.rolling(w, min_periods=1).mean()
        std = c.rolling(w, min_periods=1).std().replace(0, np.nan)
        feat[f"close_vs_sma{w}_z"] = (c - sma) / std

    # SMA slopes (pct change of SMA over 5 days)
    for w in [20, 50]:
        sma = c.rolling(w, min_periods=1).mean()
        feat[f"sma{w}_slope_5d"] = sma.pct_change(5)

    # --- Volatility ---
    feat["atr_5"] = _compute_atr(h, l, c, 5)
    feat["atr_14"] = _compute_atr(h, l, c, 14)
    feat["atr_ratio_5_20"] = feat["atr_5"] / _compute_atr(h, l, c, 20).replace(0, np.nan)

    # Realized vol
    log_ret = np.log(c / c.shift(1))
    feat["rvol_10d"] = log_ret.rolling(10).std() * np.sqrt(252)
    feat["rvol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)

    # Vol-of-vol
    feat["vol_of_vol"] = feat["rvol_10d"].rolling(20).std()

    # --- Range features ---
    day_range = h - l
    feat["range_pct"] = day_range / c
    feat["range_vs_atr"] = day_range / feat["atr_14"].replace(0, np.nan)
    feat["close_position"] = (c - l) / day_range.replace(0, np.nan)

    # Wick ratios
    body = (c - o).abs()
    upper_wick = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_wick = pd.concat([c, o], axis=1).min(axis=1) - l
    feat["upper_wick_ratio"] = upper_wick / day_range.replace(0, np.nan)
    feat["lower_wick_ratio"] = lower_wick / day_range.replace(0, np.nan)

    # --- Gap ---
    feat["gap_pct"] = (o - c.shift(1)) / c.shift(1)

    # --- Consecutive days ---
    up_day = (c > c.shift(1)).astype(int)
    down_day = (c < c.shift(1)).astype(int)

    # Consecutive up/down using cumsum trick
    up_groups = (up_day != up_day.shift(1)).cumsum()
    feat["consec_up"] = up_day.groupby(up_groups).cumsum()
    down_groups = (down_day != down_day.shift(1)).cumsum()
    feat["consec_down"] = down_day.groupby(down_groups).cumsum()

    # --- Drawdown ---
    rolling_high_20 = c.rolling(20, min_periods=1).max()
    rolling_high_60 = c.rolling(60, min_periods=1).max()
    feat["drawdown_20d"] = (c - rolling_high_20) / rolling_high_20
    feat["drawdown_60d"] = (c - rolling_high_60) / rolling_high_60

    # --- Volume ---
    vol_mean = v.rolling(20, min_periods=1).mean()
    vol_std = v.rolling(20, min_periods=1).std().replace(0, np.nan)
    feat["volume_z"] = (v - vol_mean) / vol_std

    # --- Calendar ---
    feat["day_of_week"] = daily.index.dayofweek
    # OPEX week: 3rd Friday of the month
    month_start = daily.index.to_period("M").to_timestamp()
    # Day of month, check if within 3rd Friday week
    dom = daily.index.day
    dow = daily.index.dayofweek
    # 3rd Friday falls on days 15-21
    feat["opex_week"] = ((dom >= 15) & (dom <= 21)).astype(int)

    # --- Bollinger Band features ---
    sma20 = c.rolling(20, min_periods=1).mean()
    std20 = c.rolling(20, min_periods=1).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_bandwidth = bb_upper - bb_lower

    feat["bb_pct_b"] = (c - bb_lower) / bb_bandwidth.replace(0, np.nan)
    feat["bb_width"] = bb_bandwidth / sma20.replace(0, np.nan)
    bb_width_mean = feat["bb_width"].rolling(60, min_periods=10).mean()
    bb_width_std = feat["bb_width"].rolling(60, min_periods=10).std().replace(0, np.nan)
    feat["bb_width_z"] = (feat["bb_width"] - bb_width_mean) / bb_width_std
    atr14 = feat["atr_14"].replace(0, np.nan)
    feat["bb_upper_dist"] = (bb_upper - c) / atr14
    feat["bb_lower_dist"] = (c - bb_lower) / atr14

    # --- Extended MA + crossover features ---
    sma100 = c.rolling(100, min_periods=1).mean()
    sma200 = c.rolling(200, min_periods=1).mean()
    sma50 = c.rolling(50, min_periods=1).mean()
    std100 = c.rolling(100, min_periods=1).std().replace(0, np.nan)

    feat["close_vs_sma100_z"] = (c - sma100) / std100
    feat["sma100_slope_5d"] = sma100.pct_change(5)
    feat["sma200_slope_5d"] = sma200.pct_change(5)
    feat["sma200_slope_20d"] = sma200.pct_change(20)

    # MA alignment score: +1 if 20>50>100>200, -1 if reversed
    ma_pairs = [(sma20, sma50), (sma50, sma100), (sma100, sma200)]
    alignment = sum((short > long_ma).astype(float) for short, long_ma in ma_pairs) / 3.0
    feat["ma_alignment"] = alignment * 2 - 1  # map [0,1] → [-1,+1]

    # Days since MA crosses (positive = golden, negative = death)
    cross_20_50 = (sma20 > sma50).astype(int)
    cross_20_50_change = cross_20_50.diff().fillna(0)
    days_since_20_50 = np.zeros(len(c))
    for i in range(1, len(c)):
        if cross_20_50_change.iloc[i] != 0:
            days_since_20_50[i] = 1 if cross_20_50.iloc[i] == 1 else -1
        else:
            prev = days_since_20_50[i - 1]
            days_since_20_50[i] = (abs(prev) + 1) * np.sign(prev) if prev != 0 else 0
    feat["ma_20_50_cross"] = days_since_20_50

    cross_50_200 = (sma50 > sma200).astype(int)
    cross_50_200_change = cross_50_200.diff().fillna(0)
    days_since_50_200 = np.zeros(len(c))
    for i in range(1, len(c)):
        if cross_50_200_change.iloc[i] != 0:
            days_since_50_200[i] = 1 if cross_50_200.iloc[i] == 1 else -1
        else:
            prev = days_since_50_200[i - 1]
            days_since_50_200[i] = (abs(prev) + 1) * np.sign(prev) if prev != 0 else 0
    feat["ma_50_200_cross"] = days_since_50_200

    # Weekly RSI (using 5-day returns)
    weekly_ret = c.pct_change(5)
    feat["rsi_weekly"] = _compute_rsi(weekly_ret.fillna(0), 14)

    # --- Mean-reversion / trend exhaustion features ---
    # Trend exhaustion: RSI_norm × BB_%B_norm (both mapped to [-1, +1])
    rsi_norm = (feat["rsi_14"] - 50) / 50  # [-1, +1]
    bb_b_norm = feat["bb_pct_b"].clip(0, 1) * 2 - 1  # [0,1] → [-1,+1]
    feat["trend_exhaustion"] = rsi_norm * bb_b_norm

    # Price extension from MAs (ATR-normalized)
    feat["price_extension_20d"] = (c - sma20) / atr14
    feat["price_extension_50d"] = (c - sma50) / atr14

    # Return vs range
    ret_20d = c.pct_change(20).fillna(0)
    atr_20d_sum = feat["atr_14"].rolling(20, min_periods=1).mean()
    feat["return_vs_range_20d"] = ret_20d / (atr_20d_sum / c).replace(0, np.nan)

    # Up-capture ratio
    daily_rets = c.pct_change().fillna(0)
    pos_rets = daily_rets.where(daily_rets > 0, 0)
    neg_rets = daily_rets.where(daily_rets < 0, 0).abs()
    sum_pos = pos_rets.rolling(20, min_periods=5).sum()
    sum_neg = neg_rets.rolling(20, min_periods=5).sum()
    feat["up_capture_ratio_20d"] = sum_pos / sum_neg.replace(0, np.nan)

    # Recovery ratio: where price is within 20d range
    rolling_high_20_rr = c.rolling(20, min_periods=1).max()
    rolling_low_20 = c.rolling(20, min_periods=1).min()
    rr_range = (rolling_high_20_rr - rolling_low_20).replace(0, np.nan)
    feat["recovery_ratio"] = (c - rolling_low_20) / rr_range

    # Down-vol ratio: std of negative returns / std of all returns
    neg_only = daily_rets.where(daily_rets < 0, np.nan)
    down_std = neg_only.rolling(20, min_periods=5).std()
    all_std = daily_rets.rolling(20, min_periods=5).std().replace(0, np.nan)
    feat["down_vol_ratio"] = down_std / all_std

    return feat.fillna(0.0)


FEATURE_NAMES = [
    "return_1d", "return_5d", "return_20d", "return_60d",
    "rsi_5", "rsi_14",
    "close_vs_sma20_z", "close_vs_sma50_z", "close_vs_sma200_z",
    "sma20_slope_5d", "sma50_slope_5d",
    "atr_5", "atr_14", "atr_ratio_5_20",
    "rvol_10d", "rvol_20d", "vol_of_vol",
    "range_pct", "range_vs_atr", "close_position",
    "upper_wick_ratio", "lower_wick_ratio",
    "gap_pct",
    "consec_up", "consec_down",
    "drawdown_20d", "drawdown_60d",
    "volume_z",
    "day_of_week", "opex_week",
    # Bollinger Band features
    "bb_pct_b", "bb_width", "bb_width_z", "bb_upper_dist", "bb_lower_dist",
    # Extended MA + crossover features
    "close_vs_sma100_z", "sma100_slope_5d", "sma200_slope_5d", "sma200_slope_20d",
    "ma_alignment", "ma_20_50_cross", "ma_50_200_cross", "rsi_weekly",
    # Mean-reversion / trend exhaustion features
    "trend_exhaustion", "price_extension_20d", "price_extension_50d",
    "return_vs_range_20d", "up_capture_ratio_20d", "recovery_ratio", "down_vol_ratio",
]
