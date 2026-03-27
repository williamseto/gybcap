import numpy as np
import pandas as pd

import dashboard.runner as runner_mod
from dashboard.runner import DashboardRunnerConfig, SwingDashboardRunner


def _make_inputs():
    idx = pd.date_range("2026-02-17", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "f1": np.linspace(0.1, 0.6, len(idx)),
            "y_structural": [0, 1, 2, 1, 0, 2],
        },
        index=idx,
    )
    es_daily = pd.DataFrame(
        {
            "open": np.linspace(6100, 6110, len(idx)),
            "high": np.linspace(6110, 6120, len(idx)),
            "low": np.linspace(6090, 6100, len(idx)),
            "close": np.linspace(6105, 6115, len(idx)),
            "volume": np.linspace(1000, 1500, len(idx)),
        },
        index=idx,
    )
    return df, es_daily


def test_update_prediction_caches_backfills_missing_days(tmp_path, monkeypatch):
    oos_path = tmp_path / "oos_cache.parquet"
    anomaly_path = tmp_path / "anomaly_cache.parquet"
    monkeypatch.setattr(runner_mod, "OOS_CACHE", oos_path)
    monkeypatch.setattr(runner_mod, "ANOMALY_CACHE", anomaly_path)

    df, es_daily = _make_inputs()
    idx = df.index

    pd.DataFrame(
        {
            "pred": [0, 1, 2],
            "actual": [0, 1, -1],
            "p_bear": [0.7, 0.2, 0.1],
            "p_bal": [0.2, 0.6, 0.2],
            "p_bull": [0.1, 0.2, 0.7],
        },
        index=idx[:3],
    ).to_parquet(oos_path)

    pd.DataFrame({"anomaly_score": [0.05, 0.10, 0.15]}, index=idx[:3]).to_parquet(anomaly_path)

    runner = SwingDashboardRunner(DashboardRunnerConfig(min_train_days=2, strict_backfill=True))

    proba_map = {
        3: [0.1, 0.3, 0.6],
        4: [0.6, 0.2, 0.2],
        5: [0.2, 0.5, 0.3],
    }

    def fake_fit(df_valid, feature_cols, es_hist, pred_idx, pred_date=None):
        top = [("f1", 0.9)] if pred_idx == 5 else []
        return proba_map[pred_idx], top

    def fake_score(df_valid, feature_cols, pred_idx):
        return round(0.1 * pred_idx, 3)

    monkeypatch.setattr(runner, "_fit_xgb_for_index", fake_fit)
    monkeypatch.setattr(runner, "_score_anomaly_for_index", fake_score)

    today_proba, top_features = runner._update_prediction_caches(df, ["f1"], es_daily)

    assert today_proba == proba_map[5]
    assert top_features == [("f1", 0.9)]

    oos = pd.read_parquet(oos_path).sort_index()
    assert len(oos) == 6
    assert int(oos.loc[idx[2], "actual"]) == int(df.loc[idx[2], "y_structural"])
    assert int(oos.loc[idx[3], "actual"]) == int(df.loc[idx[3], "y_structural"])
    assert int(oos.loc[idx[4], "actual"]) == int(df.loc[idx[4], "y_structural"])
    assert int(oos.loc[idx[5], "actual"]) == -1

    anomaly = pd.read_parquet(anomaly_path).sort_index()
    assert len(anomaly) == 6
    assert float(anomaly.loc[idx[5], "anomaly_score"]) == 0.5


def test_update_prediction_caches_skips_refit_when_caches_are_current(tmp_path, monkeypatch):
    oos_path = tmp_path / "oos_cache.parquet"
    anomaly_path = tmp_path / "anomaly_cache.parquet"
    monkeypatch.setattr(runner_mod, "OOS_CACHE", oos_path)
    monkeypatch.setattr(runner_mod, "ANOMALY_CACHE", anomaly_path)

    df, es_daily = _make_inputs()
    idx = df.index

    oos = pd.DataFrame(
        {
            "pred": [0, 1, 2, 1, 0, 2],
            "actual": [0, 1, 2, 1, 0, -1],
            "p_bear": [0.6, 0.2, 0.1, 0.2, 0.7, 0.1],
            "p_bal": [0.2, 0.6, 0.2, 0.5, 0.2, 0.7],
            "p_bull": [0.2, 0.2, 0.7, 0.3, 0.1, 0.2],
        },
        index=idx,
    )
    oos.to_parquet(oos_path)
    pd.DataFrame({"anomaly_score": np.linspace(0.05, 0.3, len(idx))}, index=idx).to_parquet(anomaly_path)

    runner = SwingDashboardRunner(DashboardRunnerConfig(min_train_days=2, strict_backfill=True))

    def fail_fit(*args, **kwargs):
        raise AssertionError("XGB refit should not run when cache is current")

    def fail_score(*args, **kwargs):
        raise AssertionError("Anomaly refit should not run when cache is current")

    monkeypatch.setattr(runner, "_fit_xgb_for_index", fail_fit)
    monkeypatch.setattr(runner, "_score_anomaly_for_index", fail_score)

    today_proba, top_features = runner._update_prediction_caches(df, ["f1"], es_daily)

    assert today_proba == [0.1, 0.7, 0.2]
    assert top_features == []


def test_update_prediction_caches_ignores_warmup_history_gaps(tmp_path, monkeypatch):
    oos_path = tmp_path / "oos_cache.parquet"
    anomaly_path = tmp_path / "anomaly_cache.parquet"
    monkeypatch.setattr(runner_mod, "OOS_CACHE", oos_path)
    monkeypatch.setattr(runner_mod, "ANOMALY_CACHE", anomaly_path)

    df, es_daily = _make_inputs()
    idx = df.index

    # Expected cache coverage starts at min_train_days=2, so idx[:2] are warmup-only.
    pd.DataFrame(
        {
            "pred": [2, 1, 0, 2],
            "actual": [2, 1, 0, -1],
            "p_bear": [0.1, 0.2, 0.7, 0.1],
            "p_bal": [0.2, 0.6, 0.2, 0.7],
            "p_bull": [0.7, 0.2, 0.1, 0.2],
        },
        index=idx[2:],
    ).to_parquet(oos_path)
    pd.DataFrame({"anomaly_score": np.linspace(0.1, 0.4, 4)}, index=idx[2:]).to_parquet(anomaly_path)

    runner = SwingDashboardRunner(DashboardRunnerConfig(min_train_days=2, strict_backfill=True))

    def fail_fit(*args, **kwargs):
        raise AssertionError("Warmup history should not trigger OOS refit")

    def fail_score(*args, **kwargs):
        raise AssertionError("Warmup history should not trigger anomaly refit")

    monkeypatch.setattr(runner, "_fit_xgb_for_index", fail_fit)
    monkeypatch.setattr(runner, "_score_anomaly_for_index", fail_score)

    today_proba, top_features = runner._update_prediction_caches(df, ["f1"], es_daily)

    assert today_proba == [0.1, 0.7, 0.2]
    assert top_features == []


def test_update_prediction_caches_production_uses_single_train_backfill(tmp_path, monkeypatch):
    oos_path = tmp_path / "oos_cache.parquet"
    anomaly_path = tmp_path / "anomaly_cache.parquet"
    monkeypatch.setattr(runner_mod, "OOS_CACHE", oos_path)
    monkeypatch.setattr(runner_mod, "ANOMALY_CACHE", anomaly_path)

    df, es_daily = _make_inputs()
    idx = df.index

    pd.DataFrame(
        {
            "pred": [0, 1, 2],
            "actual": [0, 1, -1],
            "p_bear": [0.7, 0.2, 0.1],
            "p_bal": [0.2, 0.6, 0.2],
            "p_bull": [0.1, 0.2, 0.7],
        },
        index=idx[:3],
    ).to_parquet(oos_path)
    pd.DataFrame({"anomaly_score": [0.05, 0.10, 0.15]}, index=idx[:3]).to_parquet(anomaly_path)

    runner = SwingDashboardRunner(DashboardRunnerConfig(min_train_days=2, strict_backfill=False))

    def fail_strict_fit(*args, **kwargs):
        raise AssertionError("Strict per-day XGB fit should not run in production mode")

    def fail_strict_score(*args, **kwargs):
        raise AssertionError("Strict per-day anomaly fit should not run in production mode")

    def fake_fast_predict(df_valid, feature_cols, es_hist, missing_dates):
        out = {}
        for d in missing_dates:
            if d == idx[-1]:
                out[d] = [0.15, 0.35, 0.50]
            else:
                out[d] = [0.6, 0.2, 0.2]
        return out, [("f1", 0.8)]

    def fake_fast_scores(df_valid, feature_cols, missing_dates):
        return {d: 0.42 for d in missing_dates}

    monkeypatch.setattr(runner, "_fit_xgb_for_index", fail_strict_fit)
    monkeypatch.setattr(runner, "_score_anomaly_for_index", fail_strict_score)
    monkeypatch.setattr(runner, "_predict_missing_oos_fast", fake_fast_predict)
    monkeypatch.setattr(runner, "_score_missing_anomaly_fast", fake_fast_scores)

    today_proba, top_features = runner._update_prediction_caches(df, ["f1"], es_daily)
    assert today_proba == [0.15, 0.35, 0.50]
    assert top_features == [("f1", 0.8)]

    oos = pd.read_parquet(oos_path).sort_index()
    anomaly = pd.read_parquet(anomaly_path).sort_index()
    assert len(oos) == 6
    assert len(anomaly) == 6
    assert float(anomaly.loc[idx[-1], "anomaly_score"]) == 0.42


def test_dashboard_runner_config_strict_backfill_defaults():
    default_cfg = DashboardRunnerConfig()
    strict_cfg = DashboardRunnerConfig(strict_backfill=True)
    assert default_cfg.strict_backfill is False
    assert strict_cfg.strict_backfill is True


def test_fetch_cross_instrument_dailys_requests_nq_zn_cl_gc(monkeypatch):
    runner = SwingDashboardRunner(DashboardRunnerConfig())
    idx = pd.DatetimeIndex(["2026-03-10"])
    sample = pd.DataFrame({"close": [1.0]}, index=idx)
    calls: list[str] = []

    def fake_fetch(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return sample

    monkeypatch.setattr(runner._fetcher, "fetch_and_update", fake_fetch)

    out = runner._fetch_cross_instrument_dailys()

    assert calls == ["NQ", "ZN", "CL", "GC"]
    assert list(out.keys()) == ["NQ", "ZN", "CL", "GC"]
