import argparse
import json

from strategies.realtime.strategy_cli import (
    add_footprint_predictor_args,
    add_reversal_predictor_args,
    build_footprint_predictor_config,
    build_reversal_predictor_config,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_reversal_predictor_args(parser)
    add_footprint_predictor_args(parser)
    return parser


def test_reversal_cli_defaults_with_model_dir() -> None:
    parser = _make_parser()
    args = parser.parse_args(["--reversal-model-dir", "models/reversal_phase3"])
    cfg = build_reversal_predictor_config(args, default_history_csv="raw_data/schwab/es_minute_history.csv")
    assert cfg is not None
    assert cfg.kind == "reversal_predictor"
    assert cfg.params["model_dir"] == "models/reversal_phase3"
    assert cfg.params["pred_threshold"] == 0.50
    assert cfg.params["use_episode_gating"] is False
    assert cfg.params["trade_budget_per_day"] == 0
    assert cfg.params["frontier_virtual_gate_calibration_enabled"] is True
    assert cfg.params["historical_csv_path"] == "raw_data/schwab/es_minute_history.csv"


def test_reversal_json_config_and_cli_override(tmp_path) -> None:
    parser = _make_parser()
    cfg_path = tmp_path / "reversal.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_dir": "models/reversal_phase3_sameday_bidask",
                "pred_threshold": 0.62,
                "use_episode_gating": True,
                "trade_budget_per_day": 4,
                "pred_threshold_rth": 0.55,
                "pred_threshold_ovn": 0.62,
                "historical_csv_path": "raw_data/custom.csv",
            }
        )
    )
    args = parser.parse_args(
        [
            "--reversal-config",
            str(cfg_path),
            "--reversal-threshold",
            "0.66",
        ]
    )
    cfg = build_reversal_predictor_config(args, default_history_csv="raw_data/schwab/es_minute_history.csv")
    assert cfg is not None
    assert cfg.params["model_dir"] == "models/reversal_phase3_sameday_bidask"
    assert cfg.params["pred_threshold"] == 0.66
    assert cfg.params["use_episode_gating"] is True
    assert cfg.params["trade_budget_per_day"] == 4
    assert cfg.params["pred_threshold_rth"] == 0.55
    assert cfg.params["pred_threshold_ovn"] == 0.62
    assert cfg.params["historical_csv_path"] == "raw_data/custom.csv"


def test_reversal_frontier_virtual_gate_calibration_cli_toggle(tmp_path) -> None:
    parser = _make_parser()
    cfg_path = tmp_path / "reversal_calibration.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_dir": "models/reversal_phase3_sameday_bidask",
                "frontier_virtual_gate_calibration_enabled": True,
                "frontier_virtual_gate_calibration_days": 33,
            }
        )
    )

    args = parser.parse_args(
        [
            "--reversal-config",
            str(cfg_path),
            "--reversal-skip-startup-calibration",
        ]
    )
    cfg = build_reversal_predictor_config(args, default_history_csv="raw_data/schwab/es_minute_history.csv")
    assert cfg is not None
    assert cfg.params["frontier_virtual_gate_calibration_enabled"] is False
    assert cfg.params["frontier_virtual_gate_calibration_days"] == 33

    args_default_on = parser.parse_args(
        [
            "--reversal-model-dir",
            "models/reversal_phase3_sameday_bidask",
        ]
    )
    cfg_default_on = build_reversal_predictor_config(
        args_default_on, default_history_csv="raw_data/schwab/es_minute_history.csv"
    )
    assert cfg_default_on is not None
    assert cfg_default_on.params["frontier_virtual_gate_calibration_enabled"] is True


def test_footprint_json_params_wrapper(tmp_path) -> None:
    parser = _make_parser()
    cfg_path = tmp_path / "footprint.json"
    cfg_path.write_text(
        json.dumps(
            {
                "params": {
                    "model_dir": "models/footprint_bundle",
                    "pred_threshold": 0.58,
                    "device": "cpu",
                    "warmup_days": 45,
                }
            }
        )
    )
    args = parser.parse_args(["--footprint-config", str(cfg_path)])
    cfg = build_footprint_predictor_config(args, default_history_csv="raw_data/schwab/es_minute_history.csv")
    assert cfg is not None
    assert cfg.kind == "footprint_predictor"
    assert cfg.params["model_dir"] == "models/footprint_bundle"
    assert cfg.params["pred_threshold"] == 0.58
    assert cfg.params["device"] == "cpu"
    assert cfg.params["warmup_days"] == 45


def test_reversal_frontier_qlookup_args(tmp_path) -> None:
    parser = _make_parser()
    cfg_path = tmp_path / "reversal_qlookup.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_dir": "models/reversal_phase3_sameday_episode",
                "frontier_router_enabled": True,
                "frontier_quality_source": "q_lookup",
                "frontier_quality_lookup_path": "sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet",
                "frontier_group_unresolved_enabled": True,
            }
        )
    )
    args = parser.parse_args(["--reversal-config", str(cfg_path)])
    cfg = build_reversal_predictor_config(
        args,
        default_history_csv="raw_data/schwab/es_minute_history.csv",
    )
    assert cfg is not None
    assert cfg.params["frontier_router_enabled"] is True
    assert cfg.params["frontier_quality_source"] == "q_lookup"
    assert (
        cfg.params["frontier_quality_lookup_path"]
        == "sandbox/results/upstream_twohead_regime_rows_v1_nf3_fast6.parquet"
    )
    assert cfg.params["frontier_group_unresolved_enabled"] is True


def test_reversal_frontier_blend_from_json_config(tmp_path) -> None:
    parser = _make_parser()
    cfg_path = tmp_path / "reversal_blend.json"
    cfg_path.write_text(
        json.dumps(
            {
                "model_dir": "models/reversal_phase3_sameday_episode",
                "frontier_router_enabled": True,
                "frontier_quality_source": "blend_policy_q2",
                "frontier_quality_model_dir": "models/reversal_frontier_qtwohead_runtime_v2",
                "frontier_blend_live_cap": 5,
                "frontier_blend_q2_cap": 2,
                "frontier_blend_q2_min_q": 0.16,
                "frontier_blend_q2_override_prob": 0.24,
            }
        )
    )
    args = parser.parse_args(["--reversal-config", str(cfg_path)])
    cfg = build_reversal_predictor_config(
        args,
        default_history_csv="raw_data/schwab/es_minute_history.csv",
    )
    assert cfg is not None
    assert cfg.params["frontier_quality_source"] == "blend_policy_q2"
    assert cfg.params["frontier_quality_model_dir"] == "models/reversal_frontier_qtwohead_runtime_v2"
    assert cfg.params["frontier_blend_live_cap"] == 5
    assert cfg.params["frontier_blend_q2_cap"] == 2
    assert cfg.params["frontier_blend_q2_min_q"] == 0.16
    assert cfg.params["frontier_blend_q2_override_prob"] == 0.24
