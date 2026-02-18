"""Macro context features (~10 features).

Higher-level regime indicators derived from moving averages, drawdowns,
volatility regimes, and breadth proxies.
"""
import pandas as pd
import numpy as np


def compute_macro_context(
    es_daily: pd.DataFrame,
    other_dailys: list[tuple[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Compute macro context features from ES daily + optional cross-instruments.

    Args:
        es_daily: ES daily DataFrame with columns [close, ...]
        other_dailys: Optional list of (symbol, daily_df) for breadth proxy

    Returns:
        DataFrame with ~10 macro context features
    """
    c = es_daily["close"]
    feat = pd.DataFrame(index=es_daily.index)

    # --- MA regime states ---
    sma20 = c.rolling(20, min_periods=1).mean()
    sma50 = c.rolling(50, min_periods=1).mean()
    sma200 = c.rolling(200, min_periods=1).mean()

    feat["ma_20_vs_50"] = ((sma20 > sma50).astype(int) * 2 - 1).astype(float)  # +1 or -1
    feat["ma_50_vs_200"] = ((sma50 > sma200).astype(int) * 2 - 1).astype(float)

    # --- Drawdown regime ---
    rolling_high = c.expanding().max()
    drawdown = (c - rolling_high) / rolling_high

    # Categorical: 0=normal (>-5%), 1=correction (-5% to -15%), 2=bear (<-15%)
    feat["drawdown_regime"] = pd.cut(
        drawdown,
        bins=[-np.inf, -0.15, -0.05, np.inf],
        labels=[2, 1, 0],
    ).astype(float)

    # --- Volatility regime ---
    log_ret = np.log(c / c.shift(1))
    atr_14 = _compute_atr(es_daily["high"], es_daily["low"], c, 14)
    atr_rank = atr_14.rolling(252, min_periods=50).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
    )
    feat["vol_regime_pctile"] = atr_rank

    # --- Trend strength (Sharpe-like) ---
    ret_20d = c.pct_change(20)
    rvol_20d = log_ret.rolling(20).std() * np.sqrt(252)
    feat["trend_strength_20d"] = ret_20d / rvol_20d.replace(0, np.nan)

    # --- Range squeeze ---
    daily_range = es_daily["high"] - es_daily["low"]
    feat["range_squeeze"] = (
        daily_range.rolling(5, min_periods=1).mean()
        / daily_range.rolling(20, min_periods=1).mean().replace(0, np.nan)
    )

    # --- Breadth proxy ---
    if other_dailys:
        pos_returns = []
        for sym, other_df in other_dailys:
            ret_20 = other_df["close"].pct_change(20)
            pos_returns.append((ret_20 > 0).astype(float))

        if pos_returns:
            breadth = pd.concat(pos_returns, axis=1).mean(axis=1)
            feat["breadth_proxy"] = breadth
        else:
            feat["breadth_proxy"] = 0.5
    else:
        feat["breadth_proxy"] = 0.5

    # --- Distance from 52-week high/low ---
    high_252 = c.rolling(252, min_periods=50).max()
    low_252 = c.rolling(252, min_periods=50).min()
    range_252 = (high_252 - low_252).replace(0, np.nan)
    feat["dist_52w_high"] = (c - high_252) / range_252
    feat["dist_52w_low"] = (c - low_252) / range_252

    return feat.fillna(0.0)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean().fillna(tr)


FEATURE_NAMES = [
    "ma_20_vs_50", "ma_50_vs_200",
    "drawdown_regime",
    "vol_regime_pctile",
    "trend_strength_20d",
    "range_squeeze",
    "breadth_proxy",
    "dist_52w_high", "dist_52w_low",
]
