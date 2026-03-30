import datetime as dt

import pandas as pd

from strategies.data.history_sync import MinuteHistorySyncConfig, sync_minute_history


class _FakeSchwabClient:
    def __init__(self, candles):
        self._candles = candles
        self.requests = []

    def fetch_price_history(self, req):
        self.requests.append(req)
        return list(self._candles)


def _candle(local_ts: str, o: float, h: float, l: float, c: float, v: int):
    ts = pd.Timestamp(local_ts, tz="America/Los_Angeles")
    return {
        "datetime": int(ts.tz_convert("UTC").timestamp() * 1000),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
    }


def test_sync_bootstrap_creates_csv_with_expected_schema(tmp_path):
    csv_path = tmp_path / "es_min.csv"
    candles = [
        _candle("2026-02-19 14:59:00", 6100.0, 6101.0, 6099.5, 6100.5, 100),
        _candle("2026-02-19 15:00:00", 6100.5, 6102.0, 6100.0, 6101.5, 120),
    ]
    client = _FakeSchwabClient(candles)

    cfg = MinuteHistorySyncConfig(
        symbol="/ES",
        csv_path=str(csv_path),
        lookback_days_if_missing=5,
        stale_after_minutes=1,
    )
    now_utc = dt.datetime(2026, 2, 20, 0, 10, tzinfo=dt.timezone.utc)

    result = sync_minute_history(client, cfg, now_utc=now_utc)

    assert result.status == "ok"
    assert result.rows_before == 0
    assert result.rows_after == 2
    assert result.rows_added == 2

    out = pd.read_csv(csv_path)
    assert list(out.columns) == [
        "Date",
        "Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "BidVolume",
        "AskVolume",
        "ovn",
        "trading_day",
        "nearby_gamma_score",
    ]

    # 15:00 PT should roll to next trading_day.
    row_1500 = out[out["Time"] == "15:00:00"].iloc[0]
    assert str(row_1500["trading_day"]) == "2026-02-20"


def test_sync_deduplicates_and_appends(tmp_path):
    csv_path = tmp_path / "es_min.csv"

    existing = pd.DataFrame(
        {
            "Date": ["02/19/2026"],
            "Time": ["15:00:00"],
            "Open": [6100.5],
            "High": [6101.0],
            "Low": [6100.0],
            "Close": [6100.75],
            "Volume": [110],
            "BidVolume": [0],
            "AskVolume": [0],
            "ovn": [1],
            "trading_day": ["2026-02-20"],
            "nearby_gamma_score": [0.0],
        }
    )
    existing.to_csv(csv_path, index=False)

    candles = [
        _candle("2026-02-19 15:00:00", 6100.5, 6102.0, 6100.0, 6101.5, 120),
        _candle("2026-02-19 15:01:00", 6101.5, 6103.0, 6101.0, 6102.5, 130),
    ]
    client = _FakeSchwabClient(candles)

    cfg = MinuteHistorySyncConfig(
        symbol="/ES",
        csv_path=str(csv_path),
        lookback_days_if_missing=5,
        stale_after_minutes=0,
    )
    now_utc = dt.datetime(2026, 2, 20, 0, 10, tzinfo=dt.timezone.utc)

    result = sync_minute_history(client, cfg, now_utc=now_utc)

    assert result.status == "ok"
    assert result.rows_before == 1
    assert result.rows_after == 2
    assert result.rows_added == 1

    out = pd.read_csv(csv_path)
    assert len(out) == 2
    assert (out["Time"] == "15:00:00").sum() == 1
    assert (out["Time"] == "15:01:00").sum() == 1


def test_sync_skips_when_history_is_fresh(tmp_path):
    csv_path = tmp_path / "es_min.csv"
    existing = pd.DataFrame(
        {
            "Date": ["02/19/2026"],
            "Time": ["15:00:00"],
            "Open": [6100.5],
            "High": [6101.0],
            "Low": [6100.0],
            "Close": [6100.75],
            "Volume": [110],
            "BidVolume": [0],
            "AskVolume": [0],
            "ovn": [1],
            "trading_day": ["2026-02-20"],
            "nearby_gamma_score": [0.0],
        }
    )
    existing.to_csv(csv_path, index=False)

    client = _FakeSchwabClient(candles=[])
    cfg = MinuteHistorySyncConfig(
        symbol="/ES",
        csv_path=str(csv_path),
        stale_after_minutes=120,
    )
    # 15:00 PT == 23:00 UTC in Feb (PST)
    now_utc = dt.datetime(2026, 2, 19, 23, 30, tzinfo=dt.timezone.utc)
    result = sync_minute_history(client, cfg, now_utc=now_utc)

    assert result.status == "skipped"
    assert result.message == "local history is fresh"
    assert len(client.requests) == 0
