"""Multi-timeframe range features (16 features).

Computes range position, width, breakout, and compression at 4 timeframes.
All features are strictly causal (use data up to day t only).
"""
import pandas as pd
import numpy as np


TIMEFRAMES = {"5d": 5, "20d": 20, "63d": 63, "252d": 252}

FEATURE_NAMES = []
for tf in TIMEFRAMES:
    FEATURE_NAMES.extend([
        f"range_pos_{tf}",
        f"range_width_{tf}",
        f"range_breakout_{tf}",
        f"range_compression_{tf}",
    ])


def compute_range_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-timeframe range features from daily OHLCV bars.

    Args:
        daily: DataFrame with columns [open, high, low, close, volume], DatetimeIndex.

    Returns:
        DataFrame with 16 range features (4 per timeframe), same index as input.
    """
    close = daily["close"]
    out = pd.DataFrame(index=daily.index)

    for tf_name, tf_days in TIMEFRAMES.items():
        roll_high = close.rolling(tf_days, min_periods=tf_days).max()
        roll_low = close.rolling(tf_days, min_periods=tf_days).min()
        roll_range = roll_high - roll_low

        # Range position: where is close within the rolling range?
        # 0 = at low, 1 = at high, can be >1 or <0 on breakout
        out[f"range_pos_{tf_name}"] = (close - roll_low) / roll_range.replace(0, np.nan)

        # Range width: rolling range as % of price
        out[f"range_width_{tf_name}"] = roll_range / close

        # Breakout: +1 if close > previous day's rolling high, -1 if < previous low
        prev_high = roll_high.shift(1)
        prev_low = roll_low.shift(1)
        breakout = pd.Series(0.0, index=daily.index)
        breakout[close > prev_high] = 1.0
        breakout[close < prev_low] = -1.0
        out[f"range_breakout_{tf_name}"] = breakout

        # Range compression: current width vs longer-term average width
        # Use 3x timeframe for the baseline average
        baseline_window = min(tf_days * 3, len(close))
        avg_width = (roll_range / close).rolling(baseline_window, min_periods=tf_days).mean()
        current_width = roll_range / close
        out[f"range_compression_{tf_name}"] = current_width / avg_width.replace(0, np.nan)

    out = out.fillna(0.0)
    return out
