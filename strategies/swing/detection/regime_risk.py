"""Composite risk score combining range, anomaly, and change signals.

Produces a single 0-1 risk score that drives adaptive position sizing
and drawdown exit thresholds.
"""
import pandas as pd
import numpy as np


DEFAULT_WEIGHTS = {"range": 0.25, "anomaly": 0.40, "change": 0.35}
RISK_THRESHOLDS = [0.25, 0.50, 0.75]  # low, elevated, high, extreme


def compute_range_stress(range_features: pd.DataFrame) -> pd.Series:
    """Compute range stress score from multi-timeframe range features.

    Combines breakout activity and range position extremity across timeframes.

    Returns:
        Series with values in [0, 1], 0=calm, 1=maximum stress.
    """
    # Find timeframe suffixes in column names
    breakout_cols = [c for c in range_features.columns if c.startswith("range_breakout_")]
    pos_cols = [c for c in range_features.columns if c.startswith("range_pos_")]

    if not breakout_cols or not pos_cols:
        return pd.Series(0.0, index=range_features.index)

    # Count timeframes with active breakouts (either direction)
    breakout_count = range_features[breakout_cols].abs().sum(axis=1)
    breakout_frac = breakout_count / len(breakout_cols)

    # Average distance from center (0.5) across timeframes
    # 0 = centered in range, 0.5 = at edge
    pos_extremity = range_features[pos_cols].sub(0.5).abs().mean(axis=1)

    # Combine: weight breakouts and position extremity equally
    stress = (breakout_frac + pos_extremity) / 2.0
    return stress.clip(0, 1)


def compute_regime_risk_score(
    range_features: pd.DataFrame,
    anomaly_features: pd.DataFrame,
    change_features: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Combine range, anomaly, and change signals into composite risk score.

    Args:
        range_features: From compute_range_features() — 16 range columns.
        anomaly_features: From RollingAnomalyDetector — must have 'anomaly_score'.
        change_features: From compute_change_features() — must have 'combined_change_score'.
        weights: Component weights dict with keys 'range', 'anomaly', 'change'.
                 Default: range=0.25, anomaly=0.40, change=0.35.

    Returns:
        DataFrame with columns:
          - range_stress: normalized range stress (0-1)
          - anomaly_intensity: normalized anomaly score (0-1)
          - change_momentum: normalized change detection score (0-1)
          - risk_score: weighted combination (0-1), 0=calm, 1=maximum stress
          - risk_regime: categorical (0=low, 1=elevated, 2=high, 3=extreme)
    """
    w = weights or DEFAULT_WEIGHTS
    w_range = w.get("range", 0.25)
    w_anomaly = w.get("anomaly", 0.40)
    w_change = w.get("change", 0.35)

    # Normalize weights to sum to 1
    w_total = w_range + w_anomaly + w_change
    w_range /= w_total
    w_anomaly /= w_total
    w_change /= w_total

    # Component scores (all 0-1)
    range_stress = compute_range_stress(range_features)
    anomaly_intensity = anomaly_features["anomaly_score"].clip(0, 1)
    change_momentum = change_features["combined_change_score"].clip(0, 1)

    # Align indices
    idx = range_stress.index
    anomaly_intensity = anomaly_intensity.reindex(idx).fillna(0)
    change_momentum = change_momentum.reindex(idx).fillna(0)

    # Weighted composite
    risk_score = (
        w_range * range_stress
        + w_anomaly * anomaly_intensity
        + w_change * change_momentum
    ).clip(0, 1)

    # Categorical regime
    risk_regime = pd.Series(0, index=idx, dtype=int)
    for i, threshold in enumerate(RISK_THRESHOLDS):
        risk_regime[risk_score > threshold] = i + 1

    out = pd.DataFrame(index=idx)
    out["range_stress"] = range_stress
    out["anomaly_intensity"] = anomaly_intensity
    out["change_momentum"] = change_momentum
    out["risk_score"] = risk_score
    out["risk_regime"] = risk_regime
    return out
