"""External daily data features: VIX, DXY, leading indicators, and correlation regimes.

Self-contained module — handles its own CSV loading and alignment to ES dates.
Graceful fallback if CSV files are missing.

VIX features (12 base + 5 term structure):
    vix_close, vix_percentile_252d, vix_vs_sma20, vix_change_1d/5d/20d,
    vix_regime, vix_spike, vix_rv_spread, vix_es_corr_20d/60d, vix_es_corr_shift,
    vix_term_structure, vix_term_spread, vix_term_z, vix_term_regime, vix3m_slope_5d

DXY features (9):
    dxy_return_5d/20d, dxy_vs_sma50, dxy_trend, dxy_es_corr_20d/60d,
    dxy_es_corr_shift, dxy_es_corr_z, dxy_momentum_20d

Leading indicator features (10):
    yield_curve_slope, yield_curve_slope_20d_chg, yield_curve_z,
    skew_index, skew_z, skew_change_5d,
    credit_stress, credit_stress_z, credit_momentum, hy_ig_spread

Correlation regime features (8): 4 per pair (VIX-ES, DXY-ES)
"""
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
VIX_PATH = ROOT / "raw_data" / "VIX_History.csv"
VIX3M_PATH = ROOT / "raw_data" / "VIX3M_daily.csv"
DXY_PATH = ROOT / "raw_data" / "DXY_daily.csv"
TNX_PATH = ROOT / "raw_data" / "TNX_daily.csv"
IRX_PATH = ROOT / "raw_data" / "IRX_daily.csv"
SKEW_PATH = ROOT / "raw_data" / "SKEW_daily.csv"
HYG_PATH = ROOT / "raw_data" / "HYG_daily.csv"
LQD_PATH = ROOT / "raw_data" / "LQD_daily.csv"


def load_vix() -> pd.DataFrame | None:
    """Load VIX daily data from raw_data/VIX_History.csv (MM/DD/YYYY format)."""
    if not VIX_PATH.exists():
        warnings.warn(f"VIX data not found at {VIX_PATH}")
        return None
    df = pd.read_csv(VIX_PATH, parse_dates=["DATE"], dayfirst=False)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})
    df = df.set_index("date").sort_index()
    # Ensure numeric
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def load_vix3m() -> pd.DataFrame | None:
    """Load VIX3M daily data from raw_data/VIX3M_daily.csv."""
    if not VIX3M_PATH.exists():
        warnings.warn(f"VIX3M data not found at {VIX3M_PATH}")
        return None
    df = pd.read_csv(VIX3M_PATH, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def load_dxy() -> pd.DataFrame | None:
    """Load DXY daily data from raw_data/DXY_daily.csv."""
    if not DXY_PATH.exists():
        warnings.warn(f"DXY data not found at {DXY_PATH}")
        return None
    df = pd.read_csv(DXY_PATH, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def compute_correlation_regime_features(
    es_ret: pd.Series,
    inst_ret: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """Compute correlation regime features for any ES vs instrument pair.

    4 features:
      - {prefix}_sign_flip_10d: did 20d correlation flip sign in last 10 days?
      - {prefix}_momentum_10d: rate of change of 20d correlation over 10 days
      - {prefix}_pos_pct_60d: % of last 60 days with positive 20d correlation
      - {prefix}_regime_age: days since last sign change of 20d correlation

    Args:
        es_ret: ES daily returns series (aligned index with inst_ret).
        inst_ret: Instrument daily returns series.
        prefix: Feature name prefix (e.g., 'corr_vix', 'corr_dxy').
    """
    feat = pd.DataFrame(index=es_ret.index)
    corr_20d = es_ret.rolling(20, min_periods=10).corr(inst_ret)

    # Sign flip in last 10 days
    corr_sign = np.sign(corr_20d)
    sign_changed = (corr_sign != corr_sign.shift(1)).astype(float)
    feat[f"{prefix}_sign_flip_10d"] = sign_changed.rolling(10, min_periods=1).max()

    # Momentum: 10-day change in correlation
    feat[f"{prefix}_momentum_10d"] = corr_20d - corr_20d.shift(10)

    # Positive percentage over 60 days
    is_pos = (corr_20d > 0).astype(float)
    feat[f"{prefix}_pos_pct_60d"] = is_pos.rolling(60, min_periods=10).mean()

    # Regime age: days since last sign change
    regime_age = np.zeros(len(corr_sign))
    for i in range(1, len(corr_sign)):
        s = corr_sign.iloc[i]
        s_prev = corr_sign.iloc[i - 1]
        if pd.isna(s) or pd.isna(s_prev):
            regime_age[i] = 0
        elif s != s_prev and s != 0:
            regime_age[i] = 1
        else:
            regime_age[i] = regime_age[i - 1] + 1
    feat[f"{prefix}_regime_age"] = regime_age

    return feat


def compute_vix_features(es_daily: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Compute VIX-based features aligned to ES dates.

    Returns DataFrame with 12 VIX features + 4 correlation regime features.
    """
    feat = pd.DataFrame(index=es_daily.index)
    vix_close = vix_df["close"].reindex(es_daily.index, method="ffill")

    # Basic VIX
    feat["vix_close"] = vix_close
    feat["vix_percentile_252d"] = vix_close.rolling(252, min_periods=60).apply(
        lambda x: (x[-1] >= x).mean(), raw=True
    )
    sma20 = vix_close.rolling(20, min_periods=5).mean()
    feat["vix_vs_sma20"] = (vix_close - sma20) / sma20.replace(0, np.nan)

    # VIX changes
    feat["vix_change_1d"] = vix_close.pct_change(1)
    feat["vix_change_5d"] = vix_close.pct_change(5)
    feat["vix_change_20d"] = vix_close.pct_change(20)

    # VIX regime (categorical 0-3 based on level)
    regime = np.zeros(len(vix_close), dtype=int)
    v = vix_close.values
    regime[v >= 30] = 3  # crisis
    regime[(v >= 20) & (v < 30)] = 2  # elevated
    regime[(v >= 15) & (v < 20)] = 1  # normal
    # regime 0 = low vol (< 15)
    feat["vix_regime"] = regime

    # VIX spike: >2 std move in 1 day
    vix_ret = vix_close.pct_change()
    vix_ret_std = vix_ret.rolling(60, min_periods=20).std()
    feat["vix_spike"] = (vix_ret.abs() > 2 * vix_ret_std).astype(float)

    # VIX term structure proxy: VIX - 20d realized vol of ES
    es_ret = es_daily["close"].pct_change()
    rvol_20d = es_ret.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
    feat["vix_rv_spread"] = vix_close - rvol_20d

    # VIX-ES correlations
    vix_ret_aligned = vix_close.pct_change()
    feat["vix_es_corr_20d"] = es_ret.rolling(20, min_periods=10).corr(vix_ret_aligned)
    feat["vix_es_corr_60d"] = es_ret.rolling(60, min_periods=20).corr(vix_ret_aligned)
    feat["vix_es_corr_shift"] = feat["vix_es_corr_20d"] - feat["vix_es_corr_60d"]

    # VIX term structure features (VIX vs VIX3M)
    vix3m_df = load_vix3m()
    if vix3m_df is not None:
        vix3m_close = vix3m_df["close"].reindex(es_daily.index, method="ffill")
        # Ratio: >1 = backwardation (fear), <1 = contango (normal)
        feat["vix_term_structure"] = vix_close / vix3m_close.replace(0, np.nan)
        # Raw spread
        feat["vix_term_spread"] = vix3m_close - vix_close
        # Z-score of term structure ratio
        ts_mean = feat["vix_term_structure"].rolling(60, min_periods=10).mean()
        ts_std = feat["vix_term_structure"].rolling(60, min_periods=10).std().replace(0, np.nan)
        feat["vix_term_z"] = (feat["vix_term_structure"] - ts_mean) / ts_std
        # Persistent backwardation indicator
        feat["vix_term_regime"] = (
            feat["vix_term_structure"].rolling(20, min_periods=5).mean() > 1.0
        ).astype(float)
        # VIX3M slope
        feat["vix3m_slope_5d"] = vix3m_close.pct_change(5)

    # Correlation regime features
    corr_regime = compute_correlation_regime_features(es_ret, vix_ret_aligned, "corr_vix")
    feat = pd.concat([feat, corr_regime], axis=1)

    return feat


def compute_dxy_features(es_daily: pd.DataFrame, dxy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute DXY-based features aligned to ES dates.

    Returns DataFrame with 9 DXY features + 4 correlation regime features.
    """
    feat = pd.DataFrame(index=es_daily.index)
    dxy_close = dxy_df["close"].reindex(es_daily.index, method="ffill")

    # DXY returns
    feat["dxy_return_5d"] = dxy_close.pct_change(5)
    feat["dxy_return_20d"] = dxy_close.pct_change(20)

    # DXY vs SMA50
    sma50 = dxy_close.rolling(50, min_periods=20).mean()
    feat["dxy_vs_sma50"] = (dxy_close - sma50) / sma50.replace(0, np.nan)

    # DXY trend: sign of 20d return
    feat["dxy_trend"] = np.sign(dxy_close.pct_change(20))

    # DXY-ES correlations
    es_ret = es_daily["close"].pct_change()
    dxy_ret = dxy_close.pct_change()
    feat["dxy_es_corr_20d"] = es_ret.rolling(20, min_periods=10).corr(dxy_ret)
    feat["dxy_es_corr_60d"] = es_ret.rolling(60, min_periods=20).corr(dxy_ret)
    feat["dxy_es_corr_shift"] = feat["dxy_es_corr_20d"] - feat["dxy_es_corr_60d"]

    # Z-score of short-term correlation
    corr_mean = feat["dxy_es_corr_20d"].rolling(60, min_periods=10).mean()
    corr_std = feat["dxy_es_corr_20d"].rolling(60, min_periods=10).std().replace(0, np.nan)
    feat["dxy_es_corr_z"] = (feat["dxy_es_corr_20d"] - corr_mean) / corr_std

    # DXY momentum
    feat["dxy_momentum_20d"] = dxy_close.pct_change(20)

    # Correlation regime features
    corr_regime = compute_correlation_regime_features(es_ret, dxy_ret, "corr_dxy")
    feat = pd.concat([feat, corr_regime], axis=1)

    return feat


def _load_yf_csv(path: Path) -> pd.DataFrame | None:
    """Load a yfinance-downloaded CSV with date index."""
    if not path.exists():
        warnings.warn(f"Data not found at {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df


def compute_leading_indicator_features(es_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute yield curve, SKEW, and credit stress features.

    10 features from 4 data sources (TNX, IRX, SKEW, HYG, LQD):
      - yield_curve_slope: 10Y - 13w yield
      - yield_curve_slope_20d_chg: 20d change in slope
      - yield_curve_z: Z-score of slope vs 60d
      - skew_index: CBOE SKEW level
      - skew_z: Z-score vs 60d
      - skew_change_5d: 5d change in SKEW
      - credit_stress: HYG return - ES return (20d), credit vs equity spread
      - credit_stress_z: Z-score of credit_stress
      - credit_momentum: 5d change in credit_stress
      - hy_ig_spread: HYG return - LQD return (20d), quality flight
    """
    feat = pd.DataFrame(index=es_daily.index)

    # Yield curve: TNX (10Y) - IRX (13-week)
    tnx_df = _load_yf_csv(TNX_PATH)
    irx_df = _load_yf_csv(IRX_PATH)
    if tnx_df is not None and irx_df is not None:
        tnx = tnx_df["close"].reindex(es_daily.index, method="ffill")
        irx = irx_df["close"].reindex(es_daily.index, method="ffill")
        slope = tnx - irx
        feat["yield_curve_slope"] = slope
        feat["yield_curve_slope_20d_chg"] = slope - slope.shift(20)
        slope_mean = slope.rolling(60, min_periods=20).mean()
        slope_std = slope.rolling(60, min_periods=20).std().replace(0, np.nan)
        feat["yield_curve_z"] = (slope - slope_mean) / slope_std

    # SKEW index
    skew_df = _load_yf_csv(SKEW_PATH)
    if skew_df is not None:
        skew = skew_df["close"].reindex(es_daily.index, method="ffill")
        feat["skew_index"] = skew
        skew_mean = skew.rolling(60, min_periods=20).mean()
        skew_std = skew.rolling(60, min_periods=20).std().replace(0, np.nan)
        feat["skew_z"] = (skew - skew_mean) / skew_std
        feat["skew_change_5d"] = skew - skew.shift(5)

    # Credit stress: HYG (high-yield) and LQD (investment-grade)
    hyg_df = _load_yf_csv(HYG_PATH)
    lqd_df = _load_yf_csv(LQD_PATH)
    es_ret_20d = es_daily["close"].pct_change(20)

    if hyg_df is not None:
        hyg_close = hyg_df["close"].reindex(es_daily.index, method="ffill")
        hyg_ret_20d = hyg_close.pct_change(20)
        # Credit stress = HYG 20d return - ES 20d return
        # Negative = credit underperforming equity = stress
        credit_stress = hyg_ret_20d - es_ret_20d
        feat["credit_stress"] = credit_stress
        cs_mean = credit_stress.rolling(60, min_periods=20).mean()
        cs_std = credit_stress.rolling(60, min_periods=20).std().replace(0, np.nan)
        feat["credit_stress_z"] = (credit_stress - cs_mean) / cs_std
        feat["credit_momentum"] = credit_stress - credit_stress.shift(5)

        if lqd_df is not None:
            lqd_close = lqd_df["close"].reindex(es_daily.index, method="ffill")
            lqd_ret_20d = lqd_close.pct_change(20)
            # HYG - LQD: negative = flight to quality
            feat["hy_ig_spread"] = hyg_ret_20d - lqd_ret_20d

    return feat


LEADING_FEATURE_NAMES = [
    "yield_curve_slope", "yield_curve_slope_20d_chg", "yield_curve_z",
    "skew_index", "skew_z", "skew_change_5d",
    "credit_stress", "credit_stress_z", "credit_momentum",
    "hy_ig_spread",
]


def compute_external_features(es_daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute all external daily features (VIX + DXY).

    Loads CSVs internally. Gracefully skips if files are missing.

    Returns:
        (features_df, feature_names) tuple. features_df is aligned to es_daily index.
    """
    all_feats = pd.DataFrame(index=es_daily.index)
    feature_names = []

    # VIX features
    vix_df = load_vix()
    if vix_df is not None:
        vix_feats = compute_vix_features(es_daily, vix_df)
        all_feats = pd.concat([all_feats, vix_feats], axis=1)
        feature_names.extend(vix_feats.columns.tolist())
        print(f"    {len(vix_feats.columns)} VIX features")
    else:
        print("    VIX data not found — skipping VIX features")

    # DXY features
    dxy_df = load_dxy()
    if dxy_df is not None:
        dxy_feats = compute_dxy_features(es_daily, dxy_df)
        all_feats = pd.concat([all_feats, dxy_feats], axis=1)
        feature_names.extend(dxy_feats.columns.tolist())
        print(f"    {len(dxy_feats.columns)} DXY features")
    else:
        print("    DXY data not found — skipping DXY features")

    # Leading indicator features (yield curve, SKEW, credit stress)
    leading_feats = compute_leading_indicator_features(es_daily)
    # Only include columns that were actually computed (data may be missing)
    computed = [c for c in LEADING_FEATURE_NAMES if c in leading_feats.columns]
    if computed:
        all_feats = pd.concat([all_feats, leading_feats[computed]], axis=1)
        feature_names.extend(computed)
        print(f"    {len(computed)} leading indicator features")
    else:
        print("    Leading indicator data not found — skipping")

    all_feats = all_feats.fillna(0)
    return all_feats, feature_names
