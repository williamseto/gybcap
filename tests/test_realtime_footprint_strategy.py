import json

import numpy as np
import pandas as pd

from strategies.reversal.realtime_footprint_strategy import FootprintRealtimeStrategy


class _DummyModel:
    def eval(self) -> "_DummyModel":
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("not used in this smoke test")


def _make_bars(start: str, periods: int, trading_day: str) -> pd.DataFrame:
    dt = pd.date_range(start, periods=periods, freq="1min")
    n = len(dt)
    return pd.DataFrame(
        {
            "dt": dt,
            "trading_day": [trading_day] * n,
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
            "bidvolume": np.full(n, 500.0),
            "askvolume": np.full(n, 500.0),
            "ovn": np.zeros(n, dtype=np.int8),
        }
    )


def test_footprint_strategy_parent_state_smoke(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "footprint_bundle"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_cols": ["close_z20", "vwap_z"],
                "tracked_levels": ["ib_lo"],
                "same_day_bidask_only": True,
                "threshold": 0.55,
                "proximity_pts": 5.0,
            }
        )
    )

    monkeypatch.setattr(
        "strategies.reversal.realtime_footprint_strategy.load_footprint_bundle",
        lambda model_dir, device: (_DummyModel(), json.loads((tmp_path / "footprint_bundle" / "metadata.json").read_text())),
    )

    strategy = FootprintRealtimeStrategy(model_dir=str(model_dir), device="cpu")
    assert strategy._tracked_levels == ["ib_lo"]
    assert strategy._frontier_router_enabled is False
    assert strategy._frontier_virtual_gate_calibration_enabled is False

    history = _make_bars("2026-03-09 06:30:00", periods=120, trading_day="2026-03-09")
    strategy.set_historical_context(history)

    current = _make_bars("2026-03-10 06:30:00", periods=110, trading_day="2026-03-10").set_index("dt")
    out = strategy.process(current)
    assert isinstance(out, list)
