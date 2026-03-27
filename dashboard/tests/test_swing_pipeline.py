import numpy as np
import pandas as pd

from strategies.swing.config import SwingConfig
from strategies.swing.pipeline import build_training_frame


def _make_daily(start_close: float, drift: float) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=40, freq="B")
    close = start_close + drift * np.arange(len(idx))
    return pd.DataFrame(
        {
            "open": close - 1.0,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.linspace(1000, 2000, len(idx)),
        },
        index=idx,
    )


def test_build_training_frame_defaults_to_aligned_dashboard_and_research_inputs():
    es_daily = _make_daily(6000.0, 5.0)
    nq_daily = _make_daily(21000.0, 10.0)
    zn_daily = _make_daily(110.0, 0.05)
    swing_cfg = SwingConfig()

    artifacts = build_training_frame(
        es_daily=es_daily,
        other_dailys=[("NQ", nq_daily), ("ZN", zn_daily)],
        config=swing_cfg,
        include_vp=False,
        include_external=False,
    )

    assert "breadth_proxy" in artifacts.features.columns
    assert float(artifacts.features["breadth_proxy"].iloc[-1]) == 1.0
    assert "range_pos_5d" in artifacts.feature_cols
    assert "range_pos_5d" in artifacts.feature_groups["range"]
    assert {"y_macro", "y_micro", "y_structural"} <= set(artifacts.labels.columns)


def test_build_training_frame_can_disable_macro_breadth_and_range_when_requested():
    es_daily = _make_daily(6000.0, 5.0)
    nq_daily = _make_daily(21000.0, 10.0)
    zn_daily = _make_daily(110.0, 0.05)
    swing_cfg = SwingConfig()

    artifacts = build_training_frame(
        es_daily=es_daily,
        other_dailys=[("NQ", nq_daily), ("ZN", zn_daily)],
        config=swing_cfg,
        include_vp=False,
        include_external=False,
        include_range=False,
        use_other_dailys_for_macro=False,
    )

    assert float(artifacts.features["breadth_proxy"].iloc[-1]) == 0.5
    assert "range_pos_5d" not in artifacts.feature_cols
    assert artifacts.feature_groups["range"] == []


def test_build_training_frame_generates_cl_gc_cross_features_when_provided():
    es_daily = _make_daily(6000.0, 5.0)
    nq_daily = _make_daily(21000.0, 10.0)
    zn_daily = _make_daily(110.0, 0.05)
    cl_daily = _make_daily(70.0, 0.2)
    gc_daily = _make_daily(2000.0, 1.0)
    swing_cfg = SwingConfig()

    artifacts = build_training_frame(
        es_daily=es_daily,
        other_dailys=[("NQ", nq_daily), ("ZN", zn_daily), ("CL", cl_daily), ("GC", gc_daily)],
        config=swing_cfg,
        include_vp=False,
        include_external=False,
    )

    assert "corr_cl_20d" in artifacts.features.columns
    assert "corr_gc_20d" in artifacts.features.columns
    assert "corr_cl_20d" in artifacts.feature_cols
    assert "corr_gc_20d" in artifacts.feature_cols
