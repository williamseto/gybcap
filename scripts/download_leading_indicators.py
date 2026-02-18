"""Download leading indicator data for BEAR detection via yfinance.

Downloads:
  - ^TNX (10Y Treasury yield)
  - ^IRX (13-week T-bill yield)
  - ^SKEW (CBOE SKEW index)
  - HYG (High-yield corporate bond ETF)
  - LQD (Investment-grade corporate bond ETF)

Usage:
    source ~/ml-venv/bin/activate
    python scripts/download_leading_indicators.py
"""
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "raw_data"

TICKERS = {
    "^TNX": "TNX_daily.csv",
    "^IRX": "IRX_daily.csv",
    "^SKEW": "SKEW_daily.csv",
    "HYG": "HYG_daily.csv",
    "LQD": "LQD_daily.csv",
}


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    start = "2009-01-01"
    end = "2026-02-16"

    for ticker, filename in TICKERS.items():
        out_path = OUT_DIR / filename
        print(f"Downloading {ticker}...")
        data = yf.download(ticker, start=start, end=end, progress=False)

        if data is None or len(data) < 100:
            print(f"  WARNING: Only got {len(data) if data is not None else 0} rows for {ticker}")
            continue

        # Handle MultiIndex columns from newer yfinance
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.droplevel(1)

        data.columns = [c.lower() for c in data.columns]
        data.index.name = "date"
        data.to_csv(out_path)
        print(f"  Saved {len(data)} rows to {out_path}")
        print(f"  Date range: {data.index.min().date()} -- {data.index.max().date()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
