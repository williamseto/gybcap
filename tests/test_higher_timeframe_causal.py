import numpy as np
import pandas as pd
import pytest

from strategies.features.higher_timeframe import HigherTimeframeProvider


def _make_minute_fixture() -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-05 06:30:00")
    day_bases = [100.0, 110.0, 90.0, 130.0]
    for i, close_base in enumerate(day_bases):
        day_ts = base + pd.Timedelta(days=i)
        day = day_ts.date().isoformat()
        for m in range(3):
            ts = day_ts + pd.Timedelta(minutes=m)
            c = close_base + m * 0.1
            rows.append(
                {
                    "dt": ts,
                    "trading_day": day,
                    "open": c - 0.2,
                    "high": c + 0.5,
                    "low": c - 0.5,
                    "close": c,
                    "volume": 1000.0,
                    "bidvolume": 500.0,
                    "askvolume": 500.0,
                    "ovn": 0,
                }
            )
    return pd.DataFrame(rows)


def test_higher_timeframe_eod_features_are_shifted_by_one_day() -> None:
    df = _make_minute_fixture()
    provider = HigherTimeframeProvider()

    out = provider.compute(df)
    daily = provider._compute_daily_features(provider._aggregate_to_daily(df))

    expected = (
        daily.set_index("trading_day")["daily_close_vs_sma20"]
        .shift(1)
        .fillna(0.0)
    )
    observed = out.groupby("trading_day")["daily_close_vs_sma20"].first()

    pd.testing.assert_series_equal(observed, expected, check_names=False, check_exact=False, rtol=1e-8, atol=1e-8)


def test_higher_timeframe_rejects_duplicate_week_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _make_minute_fixture()
    provider = HigherTimeframeProvider()
    orig = provider._compute_weekly_features

    def _dup_week_rows(weekly: pd.DataFrame) -> pd.DataFrame:
        out = orig(weekly)
        return pd.concat([out, out.iloc[[0]]], ignore_index=True)

    monkeypatch.setattr(provider, "_compute_weekly_features", _dup_week_rows)
    with pytest.raises(ValueError, match="year_week"):
        provider.compute(df)
