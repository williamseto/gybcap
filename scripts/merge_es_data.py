"""Merge ES historical and project data into a single combined file.

Reads:
  - /mnt/d/data/archive/es_historical_public222.txt  (Feb 2009 – Oct 2022)
  - raw_data/es_min_3y_clean_td_gamma.csv             (Oct 2022 – Oct 2025)

Outputs:
  - raw_data/es_min_combined.csv  (unified schema, ~Feb 2009 – Oct 2025)

Schema: datetime,open,high,low,close,volume
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
HIST_PATH = Path("/mnt/d/data/archive/es_historical_public222.txt")
PROJ_PATH = ROOT / "raw_data" / "es_min_3y_clean_td_gamma.csv"
OUT_PATH = ROOT / "raw_data" / "es_min_combined.csv"


def load_historical(path: Path) -> pd.DataFrame:
    """Load ES historical file (YYYY/M/D 24hr, 'Last' column)."""
    df = pd.read_csv(path)
    # Columns: Date, Time, Open, High, Low, Last, Volume, ...
    df.columns = df.columns.str.strip()
    dt_str = df["Date"].str.strip() + " " + df["Time"].str.strip()
    df["datetime"] = pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S.%f")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Last": "close", "Volume": "volume",
    })
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def load_project(path: Path) -> pd.DataFrame:
    """Load ES project file (MM/DD/YYYY, separate Date+Time, 'Close')."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    dt_str = df["Date"].str.strip() + " " + df["Time"].str.strip()
    df["datetime"] = pd.to_datetime(dt_str, format="%m/%d/%Y %H:%M:%S")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    return df[["datetime", "open", "high", "low", "close", "volume"]].copy()


def merge(hist: pd.DataFrame, proj: pd.DataFrame) -> pd.DataFrame:
    """Merge with deduplication — prefer project data in overlap region."""
    overlap_start = proj["datetime"].min()
    overlap_end = hist["datetime"].max()
    print(f"Overlap window: {overlap_start} – {overlap_end}")

    n_overlap_hist = (hist["datetime"] >= overlap_start).sum()
    n_overlap_proj = (proj["datetime"] <= overlap_end).sum()
    print(f"  Historical bars in overlap: {n_overlap_hist:,}")
    print(f"  Project bars in overlap:    {n_overlap_proj:,}")

    # Keep historical data before overlap, project data from overlap onward
    hist_before = hist[hist["datetime"] < overlap_start]
    combined = pd.concat([hist_before, proj], ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # Remove exact duplicates
    n_before = len(combined)
    combined = combined.drop_duplicates(subset="datetime", keep="last")
    n_dupes = n_before - len(combined)
    if n_dupes > 0:
        print(f"  Removed {n_dupes:,} duplicate timestamps")

    return combined


def validate(df: pd.DataFrame) -> None:
    """Basic validation checks."""
    assert df["datetime"].is_monotonic_increasing, "Datetime not monotonic"
    assert df["datetime"].duplicated().sum() == 0, "Duplicate timestamps"
    assert df[["open", "high", "low", "close"]].notna().all().all(), "NaN in OHLC"
    assert (df["high"] >= df["low"]).all(), "high < low found"

    # Check for large gaps (> 3 calendar days = likely weekend/holiday, OK up to ~4 days)
    diffs = df["datetime"].diff().dropna()
    max_gap = diffs.max()
    big_gaps = diffs[diffs > pd.Timedelta(days=5)]
    if len(big_gaps) > 0:
        print(f"  WARNING: {len(big_gaps)} gaps > 5 days (max: {max_gap})")
        for idx in big_gaps.index[:5]:
            print(f"    {df.loc[idx-1, 'datetime']} → {df.loc[idx, 'datetime']}")

    print("  Validation passed")


def main():
    parser = argparse.ArgumentParser(description="Merge ES data sources")
    parser.add_argument("--hist", type=Path, default=HIST_PATH)
    parser.add_argument("--proj", type=Path, default=PROJ_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading historical data...")
    hist = load_historical(args.hist)
    print(f"  {len(hist):,} bars, {hist['datetime'].min()} – {hist['datetime'].max()}")

    print("Loading project data...")
    proj = load_project(args.proj)
    print(f"  {len(proj):,} bars, {proj['datetime'].min()} – {proj['datetime'].max()}")

    print("\nMerging...")
    combined = merge(hist, proj)
    print(f"  Combined: {len(combined):,} bars")
    print(f"  Range: {combined['datetime'].min()} – {combined['datetime'].max()}")

    print("\nValidating...")
    validate(combined)

    if args.dry_run:
        print("\nDry run — not writing output")
    else:
        combined.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")
        print(f"  File size: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
