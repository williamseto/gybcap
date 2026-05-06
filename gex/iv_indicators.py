"""IV-based actionable indicators for ES futures trading.

Three production-ready indicators derived from EOD options chain features.
See FINDINGS.md for the empirical research backing each indicator.

Quick reference::

    from gex.iv_indicators import (
        iv_position_size, classify_skew_regime, forecast_forward_vol,
    )

    # 1) Replace ATR sizing with predicted-width sizing
    size_mult = iv_position_size(pred_width=42.5, target_risk_pts=10.0)

    # 2) Get directional fade bias from put skew
    regime = classify_skew_regime(iv_skew_25d=0.018,
                                   skew_history=skew_lookback_series)
    # → "high_skew" → favor LONG fades, skip SHORT fades

    # 3) Intraday vol forecast for live sizing/stops
    fwd_vol = forecast_forward_vol(intraday_range_so_far=22.0,
                                    pred_width=42.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ── 1. Position sizing ────────────────────────────────────────────────


def iv_position_size(
    pred_width: float,
    target_risk_pts: float = 10.0,
    min_mult: float = 0.3,
    max_mult: float = 3.0,
) -> float:
    """Compute position size multiplier from predicted daily range width.

    Replaces ATR-based volatility sizing. Empirically produces 75% more
    cumulative P&L than 1/ATR sizing at matched portfolio std (Sharpe 1.30
    vs 0.69 over 535 days; positive skew vs negative skew).

    The mechanism: pred_width is forward-looking (uses prior-day IV surface,
    OI structure, etc.) and correctly down-sizes on truly volatile days that
    realized vol hasn't caught up to yet.

    Args:
        pred_width: Predicted daily range in price points (from the
            range_predictor model with IV surface features).
        target_risk_pts: Notional daily risk budget in points.
            Multiplier = target_risk_pts / pred_width.
        min_mult, max_mult: Bounds to prevent extreme sizing on
            mis-predicted days.

    Returns:
        Multiplier to apply to a baseline position size.
        Example: pred_width=40, target=10 → mult=0.25
                 pred_width=20, target=10 → mult=0.50
    """
    if pred_width is None or pred_width <= 0 or np.isnan(pred_width):
        return 1.0
    mult = target_risk_pts / pred_width
    return float(np.clip(mult, min_mult, max_mult))


# ── 2. Skew regime classifier ────────────────────────────────────────


SkewRegime = Literal["low_skew", "mid_skew", "high_skew"]


@dataclass
class SkewRegimeContext:
    """Output of classify_skew_regime().

    Attributes:
        regime: 'low_skew' | 'mid_skew' | 'high_skew'
        long_fade_bias: Bias multiplier for buying dips (long fades).
            ≈1.0 in mid regime, >1.0 in high regime, <1.0 in low regime.
        short_fade_bias: Bias multiplier for fading rallies (short fades).
            ≈1.0 in mid, <1.0 in high regime, >1.0 in low regime.
        skew_value: The raw skew value evaluated.
        percentile: Position of skew_value within the lookback distribution
            (0..1).
    """

    regime: SkewRegime
    long_fade_bias: float
    short_fade_bias: float
    skew_value: float
    percentile: float


def classify_skew_regime(
    iv_skew_25d: float,
    skew_history: pd.Series,
    long_bias_high: float = 1.5,
    long_bias_low: float = 0.6,
    short_bias_high: float = 0.7,
    short_bias_low: float = 1.2,
) -> SkewRegimeContext:
    """Classify the 25-delta put skew regime for fade-direction bias.

    Empirical edge (535 days, fade-at-boundary trades, walk-forward OOS):
        Skew T1 (flat):  long-fade avg=+10.3, short-fade avg=+15.1 → favor SHORT
        Skew T2 (mid):   long-fade avg=+19.4, short-fade avg=+17.8 → neutral
        Skew T3 (steep): long-fade avg=+25.6, short-fade avg=+20.0 → favor LONG

    Long-fade edge is **2.5x larger** in high-skew (steep puts = peak fear)
    vs low-skew (flat puts = complacent market). This matches the conventional
    logic: heavy put pricing → bottoms; flat skew → tops.

    Args:
        iv_skew_25d: Today's 25-delta put skew (put_25d_iv - call_25d_iv).
            From gex.iv_surface_features.extract_iv_surface_features().
        skew_history: Rolling window of recent skew values (e.g. last 60-120
            trading days). Used to bucket skew into terciles.
        long_bias_high, long_bias_low: Long-fade size multipliers in
            high/low regimes.
        short_bias_high, short_bias_low: Short-fade size multipliers in
            high/low regimes.

    Returns:
        SkewRegimeContext with regime label and bias multipliers.
    """
    if pd.isna(iv_skew_25d) or len(skew_history) < 20:
        return SkewRegimeContext(
            regime="mid_skew", long_fade_bias=1.0, short_fade_bias=1.0,
            skew_value=float(iv_skew_25d) if not pd.isna(iv_skew_25d) else 0.0,
            percentile=0.5,
        )

    history = skew_history.dropna()
    t1, t2 = history.quantile([1 / 3, 2 / 3])
    pct = float((history < iv_skew_25d).mean())

    if iv_skew_25d <= t1:
        regime: SkewRegime = "low_skew"
        long_bias = long_bias_low
        short_bias = short_bias_low
    elif iv_skew_25d >= t2:
        regime = "high_skew"
        long_bias = long_bias_high
        short_bias = short_bias_high
    else:
        regime = "mid_skew"
        long_bias = 1.0
        short_bias = 1.0

    return SkewRegimeContext(
        regime=regime,
        long_fade_bias=long_bias,
        short_fade_bias=short_bias,
        skew_value=float(iv_skew_25d),
        percentile=pct,
    )


# ── 3. Intraday volatility forecast ───────────────────────────────────


@dataclass
class ForwardVolForecast:
    """Output of forecast_forward_vol().

    Attributes:
        consumption_ratio: current_intraday_range / pred_width.
        size_modulator: Multiplier to apply to baseline position sizing
            (≤1 when high consumption forecasts more vol).
        regime: 'low' | 'normal' | 'elevated' | 'extreme'
        expected_forward_30min_range: Empirical forward-30min range estimate.
    """

    consumption_ratio: float
    size_modulator: float
    regime: str
    expected_forward_30min_range: float


# Empirical forward-30min range by IV consumption bucket (from 2103 samples)
_FORWARD_RANGE_BY_CONSUMPTION = {
    "low":      (0.0, 0.3,   8.7),
    "normal":   (0.3, 0.7,  12.4),
    "elevated": (0.7, 0.9,  15.0),
    "extreme":  (0.9, 999.,  26.2),
}


def forecast_forward_vol(
    intraday_range_so_far: float,
    pred_width: float,
    base_size_modulator: float = 1.0,
) -> ForwardVolForecast:
    """Estimate forward 30-min volatility from intraday range consumption.

    Empirical correlation (2103 mid-session samples):
        IV consumption ratio vs forward-30min range: r = +0.588
        ATR consumption ratio vs forward-30min range: r = +0.367

    The IV-based ratio (current_range / pred_width) forecasts forward
    volatility 60% better than the ATR-based ratio. Note that the
    correlation is POSITIVE — high consumption signals MORE forward vol
    (volatility persistence), not exhaustion. The relationship is two-sided
    (both directional and whipsaw components scale together), so this is
    NOT a directional trade signal — it's a sizing/stop modulator.

    Args:
        intraday_range_so_far: (current_high - current_low) of the session,
            in price points.
        pred_width: Predicted daily range from the range predictor.
        base_size_modulator: Optional baseline modulator to scale.

    Returns:
        ForwardVolForecast. Use ``size_modulator`` to scale positions
        intraday: e.g. multiply your normal position size by this value
        when entering trades mid-session.
    """
    if pred_width is None or pred_width <= 0:
        return ForwardVolForecast(0.0, 1.0, "unknown", 12.0)

    cons = intraday_range_so_far / pred_width

    regime = "normal"
    fwd_range = 12.4
    for label, (lo, hi, expected) in _FORWARD_RANGE_BY_CONSUMPTION.items():
        if lo <= cons < hi:
            regime = label
            fwd_range = expected
            break

    # Modulator: scale inversely with expected forward range (relative to
    # 'normal' baseline of 12.4 pts). Bounded for safety.
    modulator = float(np.clip(12.4 / fwd_range * base_size_modulator, 0.3, 1.5))

    return ForwardVolForecast(
        consumption_ratio=float(cons),
        size_modulator=modulator,
        regime=regime,
        expected_forward_30min_range=fwd_range,
    )


# ── Combined helper ──────────────────────────────────────────────────


def daily_iv_context(
    pred_width: float,
    iv_skew_25d: float,
    skew_history: pd.Series,
    target_risk_pts: float = 10.0,
) -> dict:
    """Compute all daily IV-derived sizing/regime context in one call.

    Returns a dict with:
        - 'base_size_mult': from iv_position_size()
        - 'skew_regime': from classify_skew_regime()
        - 'long_size_mult': base_size_mult * skew.long_fade_bias
        - 'short_size_mult': base_size_mult * skew.short_fade_bias
    """
    base = iv_position_size(pred_width, target_risk_pts=target_risk_pts)
    skew = classify_skew_regime(iv_skew_25d, skew_history)
    return {
        "base_size_mult": base,
        "skew_regime": skew.regime,
        "skew_percentile": skew.percentile,
        "long_size_mult": base * skew.long_fade_bias,
        "short_size_mult": base * skew.short_fade_bias,
    }
