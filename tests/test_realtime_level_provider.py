import numpy as np
import pandas as pd

from strategies.realtime.level_provider import DayPriceLevelProvider


def test_compute_bar_stats_volume_z_is_causal() -> None:
    dt = pd.date_range("2026-03-10 06:30:00", periods=25, freq="1min")
    vol = np.concatenate(([100.0], np.full(24, 1000.0)))
    bars = pd.DataFrame(
        {
            "open": np.full(25, 100.0),
            "high": np.full(25, 101.0),
            "low": np.full(25, 99.0),
            "close": np.full(25, 100.0),
            "volume": vol,
            "bidvolume": np.full(25, 500.0),
            "askvolume": np.full(25, 500.0),
        },
        index=dt,
    )
    out = DayPriceLevelProvider.compute_bar_stats(bars)

    expected_mean = bars["volume"].rolling(window=20, center=False, min_periods=1).mean()
    expected_std = bars["volume"].rolling(window=20, center=False, min_periods=1).std().add(1e-6)
    expected = (bars["volume"] - expected_mean) / expected_std
    pd.testing.assert_series_equal(out["vol_z"], expected, check_names=False, check_exact=False, rtol=1e-9, atol=1e-9)


def test_compute_bar_stats_rth_levels_track_from_open_but_activate_after_or() -> None:
    dt = pd.date_range("2026-03-10 06:30:00", periods=36, freq="1min")
    n = len(dt)
    bars = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 102.0),
            "low": np.full(n, 98.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "bidvolume": np.full(n, 500.0),
            "askvolume": np.full(n, 500.0),
        },
        index=dt,
    )
    # Opening-range extremes.
    bars.loc[pd.Timestamp("2026-03-10 06:32:00"), "high"] = 110.0
    bars.loc[pd.Timestamp("2026-03-10 06:55:00"), "low"] = 90.0
    # At 7:00, post-OR bar should still reference opening extremes.
    bars.loc[pd.Timestamp("2026-03-10 07:00:00"), "high"] = 103.0
    bars.loc[pd.Timestamp("2026-03-10 07:00:00"), "low"] = 97.0

    out = DayPriceLevelProvider.compute_bar_stats(bars)

    row_659 = out.loc[pd.Timestamp("2026-03-10 06:59:00")]
    row_700 = out.loc[pd.Timestamp("2026-03-10 07:00:00")]
    assert row_659["rth_hi"] == 0.0
    assert row_659["rth_lo"] == 0.0
    assert row_700["rth_hi"] == 110.0
    assert row_700["rth_lo"] == 90.0


def test_compute_bar_stats_ofi_z_is_finite_with_flat_orderflow() -> None:
    dt = pd.date_range("2026-03-10 06:30:00", periods=80, freq="1min")
    n = len(dt)
    bars = pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "bidvolume": np.full(n, 500.0),
            "askvolume": np.full(n, 500.0),
        },
        index=dt,
    )
    out = DayPriceLevelProvider.compute_bar_stats(bars)
    assert np.isfinite(out["ofi_z"].to_numpy()).all()
    assert (out["ofi_z"] == 0.0).all()
