import numpy as np

from strategies.labeling.near_level import assign_level_side


def test_assign_level_side_hysteresis_and_flip_confirm() -> None:
    close = np.array([100.2, 99.9, 99.6, 99.4, 100.4, 100.6], dtype=np.float64)
    nearest_level_name = np.array(["ib_lo"] * len(close), dtype=object)
    nearest_level_price = np.array([100.0] * len(close), dtype=np.float64)

    side = assign_level_side(
        close,
        nearest_level_name,
        nearest_level_price,
        side_hysteresis_pts=0.3,
        side_flip_confirm_pts=0.5,
    )

    # Stays bullish until opposite cross exceeds flip_confirm.
    # Then stays bearish until opposite cross exceeds flip_confirm again.
    assert side.tolist() == [1, 1, 1, -1, -1, 1]


def test_assign_level_side_resets_state_by_trading_day() -> None:
    close = np.array([99.4, 100.1], dtype=np.float64)
    nearest_level_name = np.array(["ib_lo", "ib_lo"], dtype=object)
    nearest_level_price = np.array([100.0, 100.0], dtype=np.float64)
    trading_day = np.array(["2026-03-10", "2026-03-11"], dtype=object)

    side = assign_level_side(
        close,
        nearest_level_name,
        nearest_level_price,
        side_hysteresis_pts=0.3,
        side_flip_confirm_pts=0.5,
        trading_day=trading_day,
    )

    # Day-2 should not inherit Day-1 bearish state.
    assert side.tolist() == [-1, 1]
