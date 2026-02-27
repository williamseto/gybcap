from datetime import date

import pandas as pd

from dashboard.data_fetcher import DailyDataFetcher


def test_read_external_csv_supports_standard_date_column(tmp_path):
    csv_path = tmp_path / "DXY_daily.csv"
    pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-23"],
            "open": [108.0, 108.5],
            "high": [108.4, 109.0],
            "low": [107.7, 108.1],
            "close": [108.2, 108.8],
        }
    ).to_csv(csv_path, index=False)

    fetcher = DailyDataFetcher()
    out = fetcher._read_external_csv(csv_path)

    assert out is not None
    assert len(out) == 2
    assert out.index.max().date() == date(2026, 2, 23)
    assert list(out.columns) == ["open", "high", "low", "close"]


def test_topup_external_fetches_from_first_missing_day(tmp_path, monkeypatch):
    csv_path = tmp_path / "DXY_daily.csv"
    pd.DataFrame(
        {
            "date": ["2026-02-20", "2026-02-21"],
            "open": [108.0, 108.2],
            "high": [108.5, 108.6],
            "low": [107.8, 108.0],
            "close": [108.3, 108.4],
        }
    ).to_csv(csv_path, index=False)

    calls: dict[str, str] = {}

    def fake_download(yf_symbol: str, start_str: str):
        calls["symbol"] = yf_symbol
        calls["start_str"] = start_str
        idx = pd.to_datetime(["2026-02-24", "2026-02-25"])
        return pd.DataFrame(
            {
                "Open": [108.7, 108.9],
                "High": [109.0, 109.2],
                "Low": [108.5, 108.7],
                "Close": [108.8, 109.1],
            },
            index=idx,
        )

    fetcher = DailyDataFetcher()
    monkeypatch.setattr(fetcher, "_download_yfinance", fake_download)

    fetcher._topup_one_external("DX-Y.NYB", csv_path, date(2026, 2, 25))

    assert calls["symbol"] == "DX-Y.NYB"
    assert calls["start_str"] == "2026-02-22"

    out = pd.read_csv(csv_path)
    assert len(out) == 4
    assert out["date"].iloc[0] == "2026-02-20"
    assert out["date"].iloc[-1] == "2026-02-25"
