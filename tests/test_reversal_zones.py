import numpy as np
import pandas as pd

from strategies.labeling.reversal_zones import ReversalBreakoutLabeler


def _make_ambiguity_fixture() -> pd.DataFrame:
    dt = pd.date_range("2026-03-10 06:30:00", periods=21, freq="1min")
    n = len(dt)
    df = pd.DataFrame(
        {
            "dt": dt,
            "trading_day": ["2026-03-10"] * n,
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "bidvolume": np.full(n, 500.0),
            "askvolume": np.full(n, 500.0),
            "ovn": np.zeros(n, dtype=np.int8),
            "ib_lo": np.full(n, 100.0),
        }
    )
    # Bar +1: first reversal hit for side=above-level.
    df.loc[1, "high"] = 102.5
    # Bar +2: opposite-side event arrives shortly after.
    df.loc[2, "low"] = 97.5
    return df


def test_reversal_breakout_labeler_drop_ambiguous_direction() -> None:
    df = _make_ambiguity_fixture()

    base = ReversalBreakoutLabeler(
        proximity_pts=1.0,
        forward_window=5,
        reversal_threshold_pts=2.0,
        breakout_threshold_pts=2.0,
        tracked_levels=["ib_lo"],
        drop_ambiguous_direction=False,
    ).fit(df)
    strict = ReversalBreakoutLabeler(
        proximity_pts=1.0,
        forward_window=5,
        reversal_threshold_pts=2.0,
        breakout_threshold_pts=2.0,
        tracked_levels=["ib_lo"],
        drop_ambiguous_direction=True,
        ambiguity_bar_tolerance=2,
    ).fit(df)

    assert base.loc[0, "outcome"] == "reversal"
    assert strict.loc[0, "outcome"] == "inconclusive"
    assert strict.loc[0, "p_reversal"] == 0.15
