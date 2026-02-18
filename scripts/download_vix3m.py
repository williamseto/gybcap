"""Download VIX3M (CBOE 3-month VIX) daily data via yfinance.

Usage:
    source ~/ml-venv/bin/activate
    python scripts/download_vix3m.py
"""
import sys
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "raw_data" / "VIX3M_daily.csv"


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    start = "2009-01-01"
    end = "2026-02-16"

    print("Downloading ^VIX3M...")
    data = yf.download("^VIX3M", start=start, end=end, progress=False)

    if data is None or len(data) < 100:
        print(f"ERROR: Only got {len(data) if data is not None else 0} rows for ^VIX3M")
        sys.exit(1)

    # Handle MultiIndex columns from newer yfinance (single ticker → flatten)
    if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
        data.columns = data.columns.droplevel(1)

    # Normalize column names to lowercase
    data.columns = [c.lower() for c in data.columns]
    data.index.name = "date"
    data.to_csv(OUT_PATH)
    print(f"Saved {len(data)} rows to {OUT_PATH}")
    print(f"  Date range: {data.index.min().date()} – {data.index.max().date()}")


if __name__ == "__main__":
    main()
