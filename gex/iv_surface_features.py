"""IV surface and options positioning features from EOD chain data.

Extracts ~20 features from an options chain snapshot:

1. **IV surface** — ATM IV, skew, term structure, butterfly (curvature)
2. **OI positioning** — put/call ratios, moneyness distribution, large OI
3. **VRP** — volatility risk premium (backward-looking)
4. **Changes** — 1d/5d changes in key surface metrics

All features use prior-day chain for causality (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


# Canonical feature list
IV_SURFACE_FEATURE_NAMES = [
    # IV surface (level)
    "atm_iv",
    "iv_skew_25d",
    "iv_term_slope",
    "iv_term_ratio",
    "iv_butterfly",
    # OI positioning
    "put_call_oi_ratio",
    "put_call_oi_ratio_otm",
    "atm_oi_change",
    "put_oi_change",
    "call_oi_change",
    "put_oi_avg_moneyness",
    "call_oi_avg_moneyness",
    # OI structure
    "zero_dte_oi_pct",
    "near_dte_oi_pct",
    # IV changes (1d)
    "atm_iv_chg1d",
    "iv_skew_25d_chg1d",
    "iv_term_slope_chg1d",
    "iv_butterfly_chg1d",
    # IV changes (5d)
    "atm_iv_chg5d",
    "iv_skew_25d_chg5d",
    "iv_term_slope_chg5d",
    "iv_butterfly_chg5d",
]


def extract_iv_surface_features(
    chain: pd.DataFrame,
    spot: float,
    prev_chain: Optional[pd.DataFrame] = None,
    prev_spot: Optional[float] = None,
    prev5_chain: Optional[pd.DataFrame] = None,
    prev5_spot: Optional[float] = None,
    realized_vol_20d: Optional[float] = None,
) -> Dict[str, float]:
    """Extract IV surface and OI features from an options chain.

    Args:
        chain: Options chain DataFrame (from load_chain_for_date).
        spot: Current underlying spot price.
        prev_chain: Previous day's chain (for 1d changes).
        prev_spot: Previous day's spot.
        prev5_chain: Chain from 5 days ago (for 5d changes).
        prev5_spot: Spot from 5 days ago.
        realized_vol_20d: 20-day realized vol (annualized) for VRP.

    Returns:
        Dict mapping feature names to float values.
    """
    features: Dict[str, float] = {}

    if chain.empty:
        for name in IV_SURFACE_FEATURE_NAMES:
            features[name] = 0.0
        return features

    # Prepare chain
    chain = _prepare(chain, spot)

    # === IV Surface features ===
    iv_feats = _compute_iv_surface(chain, spot)
    features.update(iv_feats)

    # === OI positioning features ===
    oi_feats = _compute_oi_features(chain, spot, prev_chain, prev_spot)
    features.update(oi_feats)

    # === 1d changes ===
    if prev_chain is not None and prev_spot is not None and not prev_chain.empty:
        prev_prepared = _prepare(prev_chain, prev_spot)
        prev_iv = _compute_iv_surface(prev_prepared, prev_spot)
        for base in ["atm_iv", "iv_skew_25d", "iv_term_slope", "iv_butterfly"]:
            features[f"{base}_chg1d"] = features.get(base, 0.0) - prev_iv.get(base, 0.0)
    else:
        for base in ["atm_iv", "iv_skew_25d", "iv_term_slope", "iv_butterfly"]:
            features[f"{base}_chg1d"] = 0.0

    # === 5d changes ===
    if prev5_chain is not None and prev5_spot is not None and not prev5_chain.empty:
        prev5_prepared = _prepare(prev5_chain, prev5_spot)
        prev5_iv = _compute_iv_surface(prev5_prepared, prev5_spot)
        for base in ["atm_iv", "iv_skew_25d", "iv_term_slope", "iv_butterfly"]:
            features[f"{base}_chg5d"] = features.get(base, 0.0) - prev5_iv.get(base, 0.0)
    else:
        for base in ["atm_iv", "iv_skew_25d", "iv_term_slope", "iv_butterfly"]:
            features[f"{base}_chg5d"] = 0.0

    return features


def _prepare(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Add derived columns needed for feature extraction."""
    df = chain.copy()

    if "expiration" in df.columns and "trade_date" in df.columns:
        exp = pd.to_datetime(df["expiration"])
        trade = pd.to_datetime(df["trade_date"])
        df["dte_days"] = (exp - trade).dt.days
        df["dte_days"] = df["dte_days"].clip(lower=0)

    if "option_type" not in df.columns and "instrument_class" in df.columns:
        df["option_type"] = df["instrument_class"].astype(str).str.upper().str[0]

    # Moneyness: (strike / spot - 1), negative = OTM puts, positive = OTM calls
    df["moneyness"] = df["strike"] / spot - 1.0

    # Delta proxy via moneyness (rough, for bucketing)
    # ~0.50 ATM, ~0.25 at +-0.03 moneyness for typical IV
    return df


def _compute_iv_surface(chain: pd.DataFrame, spot: float) -> Dict[str, float]:
    """Compute IV surface level features."""
    feats: Dict[str, float] = {}

    calls = chain[chain["option_type"] == "C"]
    puts = chain[chain["option_type"] == "P"]

    # ATM IV: average of nearest call and put IVs within 0.5% of spot
    atm_mask = chain["moneyness"].abs() < 0.005
    atm = chain[atm_mask]
    feats["atm_iv"] = float(atm["iv"].median()) if len(atm) > 0 else 0.0

    # 25-delta skew: IV of ~25d put - IV of ~25d call
    # Approximate 25-delta as moneyness ~ -0.03 to -0.05 (puts) and +0.03 to +0.05 (calls)
    put_25d = puts[(puts["moneyness"] > -0.06) & (puts["moneyness"] < -0.02)]
    call_25d = calls[(calls["moneyness"] > 0.02) & (calls["moneyness"] < 0.06)]
    put_25d_iv = float(put_25d["iv"].median()) if len(put_25d) > 0 else feats["atm_iv"]
    call_25d_iv = float(call_25d["iv"].median()) if len(call_25d) > 0 else feats["atm_iv"]
    feats["iv_skew_25d"] = put_25d_iv - call_25d_iv

    # Term structure: use chains with DTE data
    if "dte_days" in chain.columns:
        near = chain[(chain["dte_days"] > 0) & (chain["dte_days"] <= 10) & atm_mask]
        far = chain[(chain["dte_days"] > 20) & (chain["dte_days"] <= 60) & atm_mask]

        near_iv = float(near["iv"].median()) if len(near) > 0 else feats["atm_iv"]
        far_iv = float(far["iv"].median()) if len(far) > 0 else feats["atm_iv"]

        # Slope: (far - near) / time spread (positive = contango)
        feats["iv_term_slope"] = far_iv - near_iv
        feats["iv_term_ratio"] = far_iv / max(near_iv, 1e-6)
    else:
        feats["iv_term_slope"] = 0.0
        feats["iv_term_ratio"] = 1.0

    # Butterfly: curvature = (put_25d_iv + call_25d_iv) / 2 - atm_iv
    feats["iv_butterfly"] = (put_25d_iv + call_25d_iv) / 2 - feats["atm_iv"]

    return feats


def _compute_oi_features(
    chain: pd.DataFrame,
    spot: float,
    prev_chain: Optional[pd.DataFrame] = None,
    prev_spot: Optional[float] = None,
) -> Dict[str, float]:
    """Compute OI positioning features."""
    feats: Dict[str, float] = {}

    calls = chain[chain["option_type"] == "C"]
    puts = chain[chain["option_type"] == "P"]

    total_call_oi = calls["open_interest"].sum()
    total_put_oi = puts["open_interest"].sum()
    total_oi = total_call_oi + total_put_oi

    # Put/call OI ratio (all strikes)
    feats["put_call_oi_ratio"] = total_put_oi / max(total_call_oi, 1) if total_call_oi > 0 else 1.0

    # OTM put/call ratio (more informative — strips out hedging noise)
    otm_puts = puts[puts["moneyness"] < -0.01]
    otm_calls = calls[calls["moneyness"] > 0.01]
    otm_put_oi = otm_puts["open_interest"].sum()
    otm_call_oi = otm_calls["open_interest"].sum()
    feats["put_call_oi_ratio_otm"] = otm_put_oi / max(otm_call_oi, 1)

    # Average moneyness of put/call OI (where is positioning concentrated?)
    if total_put_oi > 0:
        feats["put_oi_avg_moneyness"] = float(
            np.average(puts["moneyness"], weights=puts["open_interest"])
        )
    else:
        feats["put_oi_avg_moneyness"] = 0.0

    if total_call_oi > 0:
        feats["call_oi_avg_moneyness"] = float(
            np.average(calls["moneyness"], weights=calls["open_interest"])
        )
    else:
        feats["call_oi_avg_moneyness"] = 0.0

    # OI changes (vs prev chain)
    if prev_chain is not None and not prev_chain.empty and prev_spot is not None:
        prev_prepared = _prepare(prev_chain, prev_spot)
        prev_calls = prev_prepared[prev_prepared["option_type"] == "C"]
        prev_puts = prev_prepared[prev_prepared["option_type"] == "P"]

        prev_total_call_oi = prev_calls["open_interest"].sum()
        prev_total_put_oi = prev_puts["open_interest"].sum()
        prev_total_oi = prev_total_call_oi + prev_total_put_oi

        feats["call_oi_change"] = (total_call_oi - prev_total_call_oi) / max(prev_total_oi, 1)
        feats["put_oi_change"] = (total_put_oi - prev_total_put_oi) / max(prev_total_oi, 1)

        # ATM OI change
        atm = chain[chain["moneyness"].abs() < 0.01]
        prev_atm = prev_prepared[prev_prepared["moneyness"].abs() < 0.01]
        atm_oi = atm["open_interest"].sum()
        prev_atm_oi = prev_atm["open_interest"].sum()
        feats["atm_oi_change"] = (atm_oi - prev_atm_oi) / max(prev_total_oi, 1)
    else:
        feats["call_oi_change"] = 0.0
        feats["put_oi_change"] = 0.0
        feats["atm_oi_change"] = 0.0

    # DTE structure
    if "dte_days" in chain.columns and total_oi > 0:
        zero_dte = chain[chain["dte_days"] <= 1]["open_interest"].sum()
        near_dte = chain[chain["dte_days"] <= 7]["open_interest"].sum()
        feats["zero_dte_oi_pct"] = zero_dte / total_oi
        feats["near_dte_oi_pct"] = near_dte / total_oi
    else:
        feats["zero_dte_oi_pct"] = 0.0
        feats["near_dte_oi_pct"] = 0.0

    return feats


def build_daily_iv_features(
    daily: pd.DataFrame,
    chain_path: str = "gex/gamma_shares_combined.parquet",
    max_days: int = 0,
) -> pd.DataFrame:
    """Build IV surface features for each trading day using prior-day chain.

    This ensures causality: features for day D use chain from day D-1.

    Args:
        daily: Daily OHLCV DataFrame with DatetimeIndex.
        chain_path: Path to combined chain parquet.
        max_days: Limit to last N days (0 = all).

    Returns:
        DataFrame indexed by date with IV surface features.
    """
    from gex.gex_surface import load_chain_for_date

    dates = daily.index.sort_values()
    if max_days > 0:
        dates = dates[-max_days:]

    # Pre-load and pre-index chains by date for fast lookup
    print("  Loading parquet and building date index...")
    parquet_df = pd.read_parquet(chain_path)
    parquet_df["trade_date"] = pd.to_datetime(parquet_df["trade_date"])
    parquet_df["expiration"] = pd.to_datetime(parquet_df["expiration"])

    # Pre-group by trade_date for O(1) lookups
    _chain_by_date = {}
    for td, group in parquet_df.groupby(parquet_df["trade_date"].dt.date):
        _chain_by_date[td] = group

    del parquet_df  # free memory

    def _load_chain_fast(target_date):
        """Load chain for target_date from pre-indexed dict."""
        td = pd.Timestamp(target_date).date()
        for offset in range(4):  # try up to 3 business day fallbacks
            check = (pd.Timestamp(td) - pd.tseries.offsets.BDay(offset)).date()
            if check in _chain_by_date:
                chain = _chain_by_date[check].copy()
                chain = chain[chain["expiration"].dt.date > td]
                return chain
        return pd.DataFrame()

    records = []
    n = len(dates)
    skipped = 0

    for i, d in enumerate(dates):
        d_ts = pd.Timestamp(d)
        d_str = d_ts.strftime("%Y-%m-%d")

        # Prior-day chain (causal)
        prev_bday = (d_ts - pd.tseries.offsets.BDay(1)).date()
        chain = _load_chain_fast(prev_bday)
        if chain.empty:
            skipped += 1
            continue

        row = daily.loc[d]
        spot = float(row["open"])  # use open price (known at prediction time)

        # 2-day-ago chain for 1d changes
        prev2_bday = (d_ts - pd.tseries.offsets.BDay(2)).date()
        prev_chain = _load_chain_fast(prev2_bday)
        prev_spot = float(daily.loc[daily.index < d, "open"].iloc[-1]) if len(daily.loc[daily.index < d]) > 0 else spot

        # 6-day-ago chain for 5d changes
        prev5_bday = (d_ts - pd.tseries.offsets.BDay(6)).date()
        prev5_chain = _load_chain_fast(prev5_bday)
        idx_5d_ago = daily.index[daily.index <= (d_ts - pd.tseries.offsets.BDay(5))]
        prev5_spot = float(daily.loc[idx_5d_ago[-1], "open"]) if len(idx_5d_ago) > 0 else spot

        try:
            feats = extract_iv_surface_features(
                chain, spot,
                prev_chain=prev_chain if not prev_chain.empty else None,
                prev_spot=prev_spot if not prev_chain.empty else None,
                prev5_chain=prev5_chain if not prev5_chain.empty else None,
                prev5_spot=prev5_spot if not prev5_chain.empty else None,
            )
            feats["date"] = d
            records.append(feats)
        except Exception as e:
            skipped += 1
            continue

        if (i + 1) % 100 == 0:
            print(f"  IV features: {i+1}/{n} days ({skipped} skipped)")

    print(f"IV features done: {len(records)} days, {skipped} skipped")
    result = pd.DataFrame(records)
    if not result.empty:
        result = result.set_index("date")
    return result
