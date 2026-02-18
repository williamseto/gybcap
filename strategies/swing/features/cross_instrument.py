"""Cross-instrument correlation and relative strength features.

Generates ~15 features per instrument pair (ES vs X), plus 4 correlation
regime features per instrument. Adding new instruments is just a matter
of passing additional DataFrames.
"""
import pandas as pd
import numpy as np

from strategies.swing.features.external_daily import compute_correlation_regime_features


def compute_cross_features(
    es_daily: pd.DataFrame,
    other_dailys: list[tuple[str, pd.DataFrame]],
    corr_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute cross-instrument features for ES vs each other instrument.

    Args:
        es_daily: ES daily DataFrame with columns [open, high, low, close, volume]
        other_dailys: List of (symbol, daily_df) pairs
        corr_windows: Rolling windows for correlation. Default [10, 20, 60].

    Returns:
        DataFrame with ~15 features per instrument pair
    """
    if corr_windows is None:
        corr_windows = [10, 20, 60]

    feat = pd.DataFrame(index=es_daily.index)
    es_ret = es_daily["close"].pct_change()

    for sym, other_df in other_dailys:
        other_ret = other_df["close"].pct_change()
        prefix = sym.lower()

        # Rolling correlations
        for w in corr_windows:
            feat[f"corr_{prefix}_{w}d"] = es_ret.rolling(w, min_periods=w // 2).corr(other_ret)

        # Correlation regime shift (short vs long)
        if len(corr_windows) >= 2:
            short_w, long_w = corr_windows[0], corr_windows[-1]
            corr_short = feat[f"corr_{prefix}_{short_w}d"]
            corr_long = feat[f"corr_{prefix}_{long_w}d"]
            feat[f"corr_{prefix}_shift"] = corr_short - corr_long

            # Z-score of short-term correlation vs 60d rolling stats
            corr_mean = corr_short.rolling(60, min_periods=10).mean()
            corr_std = corr_short.rolling(60, min_periods=10).std().replace(0, np.nan)
            feat[f"corr_{prefix}_z"] = (corr_short - corr_mean) / corr_std

        # Relative strength
        for w in [5, 20]:
            es_ret_w = es_daily["close"].pct_change(w)
            other_ret_w = other_df["close"].pct_change(w)
            feat[f"relstr_{prefix}_{w}d"] = es_ret_w - other_ret_w

        # Rolling beta (20d)
        cov_20 = es_ret.rolling(20, min_periods=10).cov(other_ret)
        var_20 = other_ret.rolling(20, min_periods=10).var().replace(0, np.nan)
        beta = cov_20 / var_20
        feat[f"beta_{prefix}_20d"] = beta
        feat[f"beta_{prefix}_change"] = beta - beta.shift(5)

        # Lead-lag: correlation of ES_t with other_t-1
        other_ret_lag1 = other_ret.shift(1)
        feat[f"leadlag_{prefix}"] = es_ret.rolling(20, min_periods=10).corr(other_ret_lag1)

        # Instrument-specific features
        if sym == "ZN":
            # Stock-bond correlation sign (risk appetite proxy)
            feat["stockbond_corr_sign"] = np.sign(feat.get(f"corr_{prefix}_20d", 0))
            # Bond trend
            feat["zn_trend_20d"] = other_df["close"].pct_change(20)
        elif sym == "NQ":
            # Tech leadership: NQ outperforming = risk-on
            feat["nq_leadership_20d"] = feat.get(f"relstr_{prefix}_20d", 0) * -1  # positive = NQ leading

        # Correlation regime features (4 per instrument)
        corr_regime = compute_correlation_regime_features(
            es_ret, other_ret, f"corr_{prefix}"
        )
        feat = pd.concat([feat, corr_regime], axis=1)

    return feat.fillna(0.0)


def get_feature_names(symbols: list[str]) -> list[str]:
    """Return expected feature names for given cross-instrument symbols."""
    names = []
    for sym in symbols:
        prefix = sym.lower()
        for w in [10, 20, 60]:
            names.append(f"corr_{prefix}_{w}d")
        names.extend([
            f"corr_{prefix}_shift", f"corr_{prefix}_z",
            f"relstr_{prefix}_5d", f"relstr_{prefix}_20d",
            f"beta_{prefix}_20d", f"beta_{prefix}_change",
            f"leadlag_{prefix}",
        ])
        if sym == "ZN":
            names.extend(["stockbond_corr_sign", "zn_trend_20d"])
        elif sym == "NQ":
            names.append("nq_leadership_20d")
        # Correlation regime features (4 per instrument)
        names.extend([
            f"corr_{prefix}_sign_flip_10d",
            f"corr_{prefix}_momentum_10d",
            f"corr_{prefix}_pos_pct_60d",
            f"corr_{prefix}_regime_age",
        ])
    return names
