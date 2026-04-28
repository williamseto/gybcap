#!/usr/bin/env python
"""Historical validation of the GEX surface model against 2.5 years of ES data.

Tests:
1. Positive-gamma days → smaller realised intraday ranges?
2. Max-gamma strike → price closes nearer to it than it opened?
3. Gamma-flip crossing → subsequent volatility increase?
4. Feature correlations with realised range / returns

Usage::

    source ~/ml-venv/bin/activate
    python -m gex.validate_gex_model [--es-path PATH] [--chain-path PATH]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from gex.gex_surface import DealerPositionModel, load_chain_for_date
from gex.gex_features import extract_gex_features, GEX_FEATURE_NAMES

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Data loading ──────────────────────────────────────────────────────


def load_es_daily(es_path: str) -> pd.DataFrame:
    """Load 1-min ES data and aggregate to daily RTH bars."""
    df = pd.read_csv(es_path)
    df["dt"] = pd.to_datetime(df["Date"] + " " + df["Time"])

    # RTH only (06:30 - 13:00 Pacific / 09:30 - 16:00 ET)
    df["time_str"] = df["Time"]
    rth = df[(df["time_str"] >= "06:30:00") & (df["time_str"] < "13:00:00")].copy()

    daily = (
        rth.groupby("trading_day")
        .agg(
            open=("Open", "first"),
            high=("High", "max"),
            low=("Low", "min"),
            close=("Close", "last"),
            volume=("Volume", "sum"),
        )
        .reset_index()
    )

    daily["range_pts"] = daily["high"] - daily["low"]
    daily["atr_14"] = daily["range_pts"].rolling(14, min_periods=1).mean()

    # Map trading_day to actual dates
    # trading_day is an integer counter; reconstruct dates from the raw data
    day_dates = (
        rth.groupby("trading_day")["Date"]
        .first()
        .reset_index()
        .rename(columns={"Date": "date_str"})
    )
    daily = daily.merge(day_dates, on="trading_day")
    daily["date"] = pd.to_datetime(daily["date_str"])

    return daily


# ── Core validation ───────────────────────────────────────────────────


def build_daily_features(
    daily: pd.DataFrame,
    chain_path: str,
    max_days: int = 0,
) -> pd.DataFrame:
    """For each trading day, load the prior chain, build GEX model, extract features."""
    records = []
    dates = daily["date"].sort_values().unique()
    if max_days > 0:
        dates = dates[-max_days:]

    n = len(dates)
    skipped = 0

    for i, d in enumerate(dates):
        d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        prev_str = (pd.Timestamp(d) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")

        try:
            chain = load_chain_for_date(prev_str, chain_path)
        except Exception:
            skipped += 1
            continue

        if chain.empty:
            skipped += 1
            continue

        row = daily[daily["date"] == d]
        if row.empty:
            skipped += 1
            continue

        row = row.iloc[0]
        spot = float(row["open"])
        atr = float(row["atr_14"])

        try:
            model = DealerPositionModel(chain, spot=spot, reference_spot=spot)
            feats = extract_gex_features(model, spot, atr_14=atr)
        except Exception as e:
            skipped += 1
            continue

        feats["date"] = d_str
        feats["range_pts"] = float(row["range_pts"])
        feats["open"] = spot
        feats["close"] = float(row["close"])
        feats["high"] = float(row["high"])
        feats["low"] = float(row["low"])
        feats["atr_14"] = atr
        feats["regime_label"] = model.regime.label

        kl = model.key_levels
        feats["gamma_flip_price"] = kl.gamma_flip
        feats["max_gamma_price"] = kl.max_gamma_strike
        feats["put_wall_price"] = kl.put_wall
        feats["call_wall_price"] = kl.call_wall

        records.append(feats)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n} days ({skipped} skipped)")

    print(f"Done: {len(records)} days with features, {skipped} skipped")
    return pd.DataFrame(records)


# ── Analysis functions ────────────────────────────────────────────────


def test_regime_vs_range(df: pd.DataFrame) -> None:
    """Test 1: positive-gamma days should have smaller realised ranges."""
    print("\n" + "=" * 60)
    print("TEST 1: Gamma regime vs realised intraday range")
    print("=" * 60)

    for label in ["positive_gamma", "negative_gamma", "gamma_pinned", "low_gamma"]:
        subset = df[df["regime_label"] == label]
        if len(subset) < 5:
            continue
        mean_range = subset["range_pts"].mean()
        med_range = subset["range_pts"].median()
        print(f"  {label:18s}: n={len(subset):4d}  "
              f"mean_range={mean_range:6.1f}  median={med_range:6.1f}")

    pos = df[df["gex_regime_sign"] > 0]["range_pts"]
    neg = df[df["gex_regime_sign"] < 0]["range_pts"]

    if len(pos) > 5 and len(neg) > 5:
        t_stat, p_val = stats.ttest_ind(pos, neg, equal_var=False)
        print(f"\n  Positive vs Negative gamma: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Positive mean: {pos.mean():.1f}  Negative mean: {neg.mean():.1f}")
        if p_val < 0.05:
            direction = "SMALLER" if pos.mean() < neg.mean() else "LARGER"
            print(f"  → Significant! Positive-gamma ranges are {direction}")
        else:
            print("  → Not significant at p<0.05")


def test_max_gamma_magnetism(df: pd.DataFrame) -> None:
    """Test 2: price closes nearer to max-gamma strike than it opened."""
    print("\n" + "=" * 60)
    print("TEST 2: Max-gamma strike magnetism")
    print("=" * 60)

    valid = df[df["max_gamma_price"].notna()].copy()
    if valid.empty:
        print("  No data with max_gamma_price")
        return

    valid["dist_open"] = abs(valid["open"] - valid["max_gamma_price"])
    valid["dist_close"] = abs(valid["close"] - valid["max_gamma_price"])
    valid["closer_at_close"] = valid["dist_close"] < valid["dist_open"]

    pct_closer = valid["closer_at_close"].mean()
    print(f"  Days where close is nearer to max-gamma than open: {pct_closer:.1%} ({len(valid)} days)")

    # Binomial test against 50%
    n_closer = int(valid["closer_at_close"].sum())
    p_val = stats.binomtest(n_closer, len(valid), 0.5).pvalue
    print(f"  Binomial test vs 50%: p={p_val:.4f}")

    if pct_closer > 0.5 and p_val < 0.05:
        print("  → Significant magnetism effect!")
    else:
        print("  → No significant magnetism")


def test_gamma_flip_volatility(df: pd.DataFrame) -> None:
    """Test 3: crossing gamma flip → higher subsequent volatility."""
    print("\n" + "=" * 60)
    print("TEST 3: Gamma-flip crossing → volatility increase")
    print("=" * 60)

    valid = df[df["gamma_flip_price"].notna()].copy()
    if valid.empty:
        print("  No data with gamma_flip_price")
        return

    # Did price cross the gamma flip during the day?
    valid["crossed_flip"] = (
        (valid["low"] <= valid["gamma_flip_price"])
        & (valid["high"] >= valid["gamma_flip_price"])
    )

    crossed = valid[valid["crossed_flip"]]
    not_crossed = valid[~valid["crossed_flip"]]

    if len(crossed) < 5 or len(not_crossed) < 5:
        print("  Not enough data in both groups")
        return

    print(f"  Days crossing flip: {len(crossed)}  Not crossing: {len(not_crossed)}")
    print(f"  Crossed range:     mean={crossed['range_pts'].mean():.1f}  "
          f"median={crossed['range_pts'].median():.1f}")
    print(f"  Not-crossed range: mean={not_crossed['range_pts'].mean():.1f}  "
          f"median={not_crossed['range_pts'].median():.1f}")

    t_stat, p_val = stats.ttest_ind(
        crossed["range_pts"], not_crossed["range_pts"], equal_var=False
    )
    print(f"  t={t_stat:.3f}, p={p_val:.4f}")


def test_feature_correlations(df: pd.DataFrame) -> None:
    """Test 4: feature correlations with realised range."""
    print("\n" + "=" * 60)
    print("TEST 4: Feature correlations with realised range")
    print("=" * 60)

    target = df["range_pts"]
    for feat in GEX_FEATURE_NAMES:
        if feat not in df.columns:
            continue
        vals = df[feat]
        valid_mask = vals.notna() & target.notna()
        if valid_mask.sum() < 20:
            continue
        r, p = stats.pearsonr(vals[valid_mask], target[valid_mask])
        sig = "*" if p < 0.05 else " "
        print(f"  {feat:30s}  r={r:+.4f}  p={p:.4f} {sig}")


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Validate GEX surface model")
    parser.add_argument(
        "--es-path",
        default="raw_data/es_min_3y_clean_td_gamma.csv",
        help="Path to 1-min ES data",
    )
    parser.add_argument(
        "--chain-path",
        default="gex/gamma_shares_combined.parquet",
        help="Path to historical options chain parquet",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Limit to last N trading days (0 = all)",
    )
    parser.add_argument(
        "--output",
        default="data/gex_validation.csv",
        help="Output CSV with daily features + realised stats",
    )
    args = parser.parse_args()

    print("Loading ES data...")
    daily = load_es_daily(args.es_path)
    print(f"  {len(daily)} trading days, range {daily['date'].min()} to {daily['date'].max()}")

    print("\nBuilding daily GEX features...")
    df = build_daily_features(daily, args.chain_path, max_days=args.max_days)

    if df.empty:
        print("ERROR: No features computed. Check chain parquet path.")
        return

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")

    # Run tests
    test_regime_vs_range(df)
    test_max_gamma_magnetism(df)
    test_gamma_flip_volatility(df)
    test_feature_correlations(df)

    # Summary stats
    print("\n" + "=" * 60)
    print("REGIME DISTRIBUTION")
    print("=" * 60)
    print(df["regime_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
