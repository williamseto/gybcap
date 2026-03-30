import json

import numpy as np
import pandas as pd
import pytest

from strategies.realtime.protocol import RealtimeSignal
from strategies.realtime.signal_handler import JsonlFileSignalHandler


def test_jsonl_file_signal_handler_writes_expected_payload(tmp_path) -> None:
    out_path = tmp_path / "signals.jsonl"
    handler = JsonlFileSignalHandler(str(out_path), truncate_on_start=True)

    sig = RealtimeSignal(
        strategy_name="reversal_predictor",
        trigger_ts=pd.Timestamp("2026-03-11T09:31:00-08:00"),
        entry_ts=pd.Timestamp("2026-03-11T09:31:00-08:00"),
        entry_price=5901.25,
        direction="bull",
        level_name="ovn_lo",
        level_value=5898.75,
        pred_proba=0.74,
        metadata={"lane": "high"},
    )
    handler.handle([sig])

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["strategy_name"] == "reversal_predictor"
    assert payload["direction"] == "bull"
    assert payload["entry_price"] == 5901.25
    assert payload["pred_proba"] == 0.74
    assert payload["metadata"]["lane"] == "high"
    assert "signal_id" in payload and payload["signal_id"]
    assert "emitted_utc" in payload and payload["emitted_utc"]


def test_jsonl_file_signal_handler_coerces_non_serializable_metadata(tmp_path) -> None:
    out_path = tmp_path / "signals.jsonl"
    handler = JsonlFileSignalHandler(str(out_path), truncate_on_start=True)

    sig = RealtimeSignal(
        strategy_name="reversal_predictor",
        trigger_ts=pd.Timestamp("2026-03-11T09:31:00-08:00"),
        entry_ts=pd.Timestamp("2026-03-11T09:31:00-08:00"),
        entry_price=np.float32(5901.25),
        direction="bull",
        level_name="ovn_lo",
        level_value=np.float64(5898.75),
        pred_proba=np.float32(0.74),
        metadata={
            "ts": pd.Timestamp("2026-03-11T09:31:00-08:00"),
            "tags": {"a", "b"},
            "arr": np.array([1.0, np.nan]),
            "np_int": np.int64(7),
        },
    )
    handler.handle([sig])

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["entry_price"] == 5901.25
    assert payload["pred_proba"] == pytest.approx(0.74, rel=1e-6)
    assert payload["metadata"]["ts"].startswith("2026-03-11T09:31:00")
    assert sorted(payload["metadata"]["tags"]) == ["a", "b"]
    assert payload["metadata"]["arr"] == [1.0, None]
    assert payload["metadata"]["np_int"] == 7
