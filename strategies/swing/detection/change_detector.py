"""Online change detection using CUSUM and EWMA.

No external dependencies beyond pandas/numpy. Detects sustained departures
from normal behavior in returns and anomaly score time series.
"""
import pandas as pd
import numpy as np


def compute_cusum_score(
    series: pd.Series,
    drift: float = 0.5,
    lookback: int = 60,
    reset_threshold: float = 5.0,
) -> pd.DataFrame:
    """One-sided upward/downward CUSUM on a series.

    Detects sustained departures from rolling mean. Implemented from scratch
    (no external dependency).

    Args:
        series: Input time series (e.g., daily returns).
        drift: Allowable drift in std units before accumulation starts.
        lookback: Rolling window for mean/std estimation.
        reset_threshold: CUSUM value at which accumulators reset.

    Returns:
        DataFrame with columns:
          - cusum_pos: cumulative positive departure
          - cusum_neg: cumulative negative departure
          - cusum_score: max(pos, neg), normalized by threshold
    """
    n = len(series)
    vals = series.values.astype(float)

    roll_mean = series.rolling(lookback, min_periods=lookback // 2).mean().values
    roll_std = series.rolling(lookback, min_periods=lookback // 2).std().values

    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    cusum_score = np.zeros(n)

    s_plus = 0.0
    s_minus = 0.0

    for i in range(1, n):
        if np.isnan(roll_mean[i]) or np.isnan(roll_std[i]) or roll_std[i] < 1e-10:
            cusum_pos[i] = 0.0
            cusum_neg[i] = 0.0
            cusum_score[i] = 0.0
            continue

        z = (vals[i] - roll_mean[i]) / roll_std[i]

        s_plus = max(0.0, s_plus + z - drift)
        s_minus = max(0.0, s_minus - z - drift)

        cusum_pos[i] = s_plus
        cusum_neg[i] = s_minus
        cusum_score[i] = max(s_plus, s_minus)

        # Reset after sustained signal
        if cusum_score[i] > reset_threshold:
            s_plus = 0.0
            s_minus = 0.0

    # Normalize by threshold so score ~1.0 means at threshold
    cusum_score = cusum_score / reset_threshold

    out = pd.DataFrame(index=series.index)
    out["cusum_pos"] = cusum_pos
    out["cusum_neg"] = cusum_neg
    out["cusum_score"] = cusum_score
    return out


def compute_ewma_score(
    series: pd.Series,
    span: int = 20,
    baseline: int = 252,
) -> pd.DataFrame:
    """EWMA trend detection on a series.

    Computes exponentially weighted moving average and z-scores it against
    a longer baseline distribution.

    Args:
        series: Input time series (e.g., anomaly scores).
        span: EWMA span (decay factor).
        baseline: Lookback for baseline mean/std estimation.

    Returns:
        DataFrame with columns:
          - ewma_value: exponentially weighted moving average
          - ewma_z: z-score of EWMA vs baseline distribution
    """
    ewma = series.ewm(span=span, min_periods=span // 2).mean()
    baseline_mean = ewma.rolling(baseline, min_periods=baseline // 2).mean()
    baseline_std = ewma.rolling(baseline, min_periods=baseline // 2).std()

    ewma_z = (ewma - baseline_mean) / baseline_std.replace(0, np.nan)

    out = pd.DataFrame(index=series.index)
    out["ewma_value"] = ewma
    out["ewma_z"] = ewma_z.fillna(0)
    return out


def compute_change_features(
    daily_returns: pd.Series,
    anomaly_scores: pd.Series,
) -> pd.DataFrame:
    """Compute change detection features on returns and anomaly scores.

    Combines CUSUM on returns with EWMA on anomaly scores to detect
    regime transitions.

    Args:
        daily_returns: Daily log or simple returns.
        anomaly_scores: Anomaly scores from Isolation Forest (0-1).

    Returns:
        DataFrame with 6 features:
          - return_cusum_score: CUSUM on daily returns (0-1 normalized)
          - return_cusum_direction: +1 if positive departure dominates, -1 if negative
          - anomaly_ewma: EWMA of anomaly scores (smoothed trend)
          - anomaly_ewma_z: z-score of anomaly EWMA vs 252d baseline
          - anomaly_acceleration: 5d change in anomaly_ewma
          - combined_change_score: max(return_cusum_score, anomaly_ewma_z_norm) clipped to [0,1]
    """
    # CUSUM on returns
    cusum = compute_cusum_score(daily_returns)

    # Direction: which accumulator is larger?
    direction = np.where(cusum["cusum_pos"] >= cusum["cusum_neg"], 1.0, -1.0)

    # EWMA on anomaly scores
    ewma = compute_ewma_score(anomaly_scores)

    # Acceleration: 5d change in EWMA
    acceleration = ewma["ewma_value"].diff(5).fillna(0)

    # Combined change score: take max of normalized signals
    # Normalize EWMA z-score to [0, 1] using sigmoid-like clipping
    ewma_z_norm = ewma["ewma_z"].clip(-3, 3) / 3.0  # now in [-1, 1]
    ewma_z_norm = (ewma_z_norm + 1) / 2  # now in [0, 1]

    combined = np.maximum(cusum["cusum_score"].values, ewma_z_norm.values)
    combined = np.clip(combined, 0, 1)

    out = pd.DataFrame(index=daily_returns.index)
    out["return_cusum_score"] = cusum["cusum_score"].values
    out["return_cusum_direction"] = direction
    out["anomaly_ewma"] = ewma["ewma_value"].values
    out["anomaly_ewma_z"] = ewma["ewma_z"].values
    out["anomaly_acceleration"] = acceleration.values
    out["combined_change_score"] = combined
    return out
