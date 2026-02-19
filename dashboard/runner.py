"""SwingDashboardRunner: wraps the full swing pipeline into a daily refresh.

Architecture
------------
Two parquet caches separate expensive one-time work from cheap daily updates:

  oos_cache.parquet     — walk-forward XGB predictions per date (built once)
  anomaly_cache.parquet — walk-forward IF anomaly scores per date (built once)

Bootstrap (first run, slow ~4-6 min):
  - Run 5-fold walk-forward XGB → oos_cache
  - Run 5-fold walk-forward Isolation Forest → anomaly_cache

Daily refresh (fast ~15-25s):
  1. Fetch new daily bar (yfinance top-up, ~5s)
  2. Compute features vectorially (~2s)
  3. Check caches — if today already present, skip retraining
  4. Fit final XGB on all-but-today → predict today → append to oos_cache
  5. Fit IF on all-but-today → score today → append to anomaly_cache
  6. Compute change/risk vectorially on full history (~0.5s)
  7. Assemble DashboardState using full ES history (not OOS-only)

Causal discipline:
  - XGB and IF are always fitted on data[:-1] before scoring today
  - History charts use cached OOS scores (also causal — each fold was OOS)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from dashboard.state import DashboardState, DayState, make_empty_state
from dashboard.data_fetcher import DailyDataFetcher

logger = logging.getLogger(__name__)

CACHE_DIR      = Path(__file__).resolve().parent / "cache"
SNAPSHOT_PATH  = CACHE_DIR / "state_snapshot.json"
OOS_CACHE      = CACHE_DIR / "oos_cache.parquet"      # date, pred, actual, p_bear, p_bal, p_bull
ANOMALY_CACHE  = CACHE_DIR / "anomaly_cache.parquet"  # date, anomaly_score

RISK_WEIGHTS = {"range": 0.15, "anomaly": 0.25, "change": 0.60}


@dataclass
class DashboardRunnerConfig:
    data_mode: str = "csv_plus_yfinance"  # or "yfinance_only"
    n_folds: int = 5
    min_train_days: int = 500
    chart_history_days: int = 252
    refresh_secret: str = "changeme"
    hmm_n_states: int = 3


class SwingDashboardRunner:

    def __init__(self, config: DashboardRunnerConfig | None = None):
        self._config = config or DashboardRunnerConfig()
        self._fetcher = DailyDataFetcher(data_mode=self._config.data_mode)
        self._last_state: DashboardState | None = None

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def load_cached_state(self) -> Optional[DashboardState]:
        if SNAPSHOT_PATH.exists():
            try:
                state = DashboardState.from_json(SNAPSHOT_PATH.read_text())
                logger.info("Loaded cached state (as_of: %s)", state.as_of_date)
                return state
            except Exception as e:
                logger.warning("Failed to load cached state: %s", e)
        return None

    def refresh(self) -> DashboardState:
        t0 = time.time()
        try:
            state = self._run_pipeline()
        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            state = make_empty_state(str(e))

        state.refresh_duration_sec = round(time.time() - t0, 2)
        self._last_state = state

        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            SNAPSHOT_PATH.write_text(state.to_json())
        except Exception as e:
            logger.warning("Failed to save state snapshot: %s", e)

        return state

    @property
    def last_state(self) -> Optional[DashboardState]:
        return self._last_state

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #

    def _run_pipeline(self) -> DashboardState:
        # ── 1. Fetch daily data ─────────────────────────────────────────
        logger.info("Fetching daily data...")
        self._fetcher.topup_external_csvs()
        es_daily = self._fetcher.fetch_and_update("ES")
        nq_daily = self._fetcher.fetch_and_update("NQ")
        zn_daily = self._fetcher.fetch_and_update("ZN")

        # ── 2. Compute features (vectorized, ~2s) ───────────────────────
        logger.info("Computing features (%d ES days)...", len(es_daily))
        features, feature_cols = self._compute_features(es_daily, nq_daily, zn_daily)
        labels_df = self._compute_labels(es_daily)
        df = features.join(labels_df).dropna(subset=["y_structural"])

        # ── 3. Bootstrap caches if missing (slow, one-time) ────────────
        self._ensure_historical_caches(df, feature_cols, es_daily)

        # ── 4. Incremental update for today (~15-25s) ───────────────────
        today_proba, top_features = self._update_today(df, feature_cols, es_daily)

        # ── 5. Load caches + compute change/risk on full history ────────
        anomaly_cache = pd.read_parquet(ANOMALY_CACHE)
        oos_cache     = pd.read_parquet(OOS_CACHE)
        change_df, risk_df = self._compute_change_risk(es_daily, df, feature_cols, anomaly_cache)

        # ── 6. Assemble state ───────────────────────────────────────────
        logger.info("Assembling state...")
        return self._build_state(
            es_daily, nq_daily, zn_daily,
            anomaly_cache, change_df, risk_df, oos_cache,
            today_proba, top_features,
        )

    # ------------------------------------------------------------------ #
    # Feature computation
    # ------------------------------------------------------------------ #

    def _compute_features(
        self,
        es_daily: pd.DataFrame,
        nq_daily: pd.DataFrame,
        zn_daily: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        from strategies.swing.features.daily_technical import (
            compute_daily_technical, FEATURE_NAMES as TECH_FEATURES,
        )
        from strategies.swing.features.volume_profile_daily import (
            compute_vp_daily_features, FEATURE_NAMES as VP_FEATURES,
        )
        from strategies.swing.features.macro_context import (
            compute_macro_context, FEATURE_NAMES as MACRO_FEATURES,
        )
        from strategies.swing.features.range_features import (
            compute_range_features, FEATURE_NAMES as RANGE_FEATURES,
        )
        from strategies.swing.features.cross_instrument import compute_cross_features
        from strategies.swing.features.external_daily import compute_external_features

        feature_cols: list[str] = []

        tech = compute_daily_technical(es_daily)
        feature_cols += list(TECH_FEATURES)

        vp = pd.DataFrame(index=es_daily.index)
        if "vp_poc_rel" in es_daily.columns:
            vp = compute_vp_daily_features(es_daily)
            feature_cols += list(VP_FEATURES)

        macro = compute_macro_context(es_daily, other_dailys=None)
        feature_cols += list(MACRO_FEATURES)

        range_feats = compute_range_features(es_daily)
        feature_cols += list(RANGE_FEATURES)

        nq_df = pd.DataFrame({"close": nq_daily["close"].reindex(es_daily.index, method="ffill")}, index=es_daily.index)
        zn_df = pd.DataFrame({"close": zn_daily["close"].reindex(es_daily.index, method="ffill")}, index=es_daily.index)
        cross = compute_cross_features(es_daily, [("NQ", nq_df), ("ZN", zn_df)], corr_windows=[10, 20, 60])
        cross_cols = [c for c in cross.columns if c not in feature_cols]
        feature_cols += cross_cols

        try:
            ext, ext_names = compute_external_features(es_daily)
            feature_cols += [c for c in ext_names if c not in feature_cols]
        except Exception as e:
            logger.warning("External features failed: %s", e)
            ext = pd.DataFrame(index=es_daily.index)

        all_feats = pd.concat([tech, vp, macro, range_feats, cross, ext], axis=1)
        all_feats = all_feats.reindex(es_daily.index).fillna(0)
        return all_feats, feature_cols

    def _compute_labels(self, es_daily: pd.DataFrame) -> pd.DataFrame:
        from strategies.swing.labeling.structural_regime import compute_labels
        return compute_labels(es_daily)

    # ------------------------------------------------------------------ #
    # Bootstrap: build caches once via full walk-forward
    # ------------------------------------------------------------------ #

    def _ensure_historical_caches(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
    ) -> None:
        """Run walk-forward XGB + IF if either cache is missing. No-op otherwise."""
        oos_missing     = not OOS_CACHE.exists()
        anomaly_missing = not ANOMALY_CACHE.exists()

        if not oos_missing and not anomaly_missing:
            return

        cfg = self._config
        # df_train excludes today (last row has y=-1)
        df_train = df.iloc[:-1]
        valid_mask = df_train["y_structural"].isin([0, 1, 2])
        df_valid = df_train[valid_mask]
        days = sorted(df_valid.index.unique())
        n_days = len(days)

        n_folds = cfg.n_folds
        if n_days < cfg.min_train_days + n_folds:
            n_folds = max(1, n_days - cfg.min_train_days)
        test_days_per_fold = max(1, (n_days - cfg.min_train_days) // n_folds)

        if oos_missing:
            logger.info("Building OOS cache (walk-forward XGB, one-time ~4-6 min)...")
            self._build_oos_cache(df_train, df_valid, days, feature_cols, es_daily,
                                  n_folds, test_days_per_fold)

        if anomaly_missing:
            logger.info("Building anomaly cache (walk-forward IF, one-time ~15s)...")
            feature_matrix = df_valid[[c for c in feature_cols if c in df_valid.columns]].fillna(0)
            self._build_anomaly_cache(feature_matrix, days, n_folds, test_days_per_fold)

    def _build_oos_cache(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        days: list,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
        n_folds: int,
        test_days_per_fold: int,
    ) -> None:
        from strategies.swing.training.regime_trainer import walk_forward_cv
        from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward

        cfg = self._config

        def hmm_fn(df_fold, train_end_idx):
            return compute_hmm_features_walkforward(es_daily, train_end_idx, n_states=cfg.hmm_n_states)

        result = walk_forward_cv(
            df=df_train,
            feature_cols=[c for c in feature_cols if c in df_train.columns],
            target_col="y_structural",
            n_folds=n_folds,
            min_train_days=cfg.min_train_days,
            params={"bear_upweight": 1.5},
            hmm_fn=hmm_fn,
            verbose=False,
        )

        # Map predictions back to dates: OOS covers days[min_train_days:]
        n_preds = len(result.oos_predictions)
        oos_days = days[cfg.min_train_days:][:n_preds]

        cache = pd.DataFrame({
            "pred":    result.oos_predictions,
            "actual":  result.oos_actuals,
            "p_bear":  result.oos_probas[:, 0],
            "p_bal":   result.oos_probas[:, 1],
            "p_bull":  result.oos_probas[:, 2],
        }, index=pd.DatetimeIndex(oos_days))
        cache.index.name = "date"
        cache.to_parquet(OOS_CACHE)
        logger.info("OOS cache saved: %d predictions (%s → %s)",
                    len(cache), oos_days[0], oos_days[-1])

    def _build_anomaly_cache(
        self,
        feature_matrix: pd.DataFrame,
        days: list,
        n_folds: int,
        test_days_per_fold: int,
    ) -> None:
        from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector

        cfg = self._config
        score_parts = []
        for fold in range(n_folds):
            train_end = cfg.min_train_days + fold * test_days_per_fold
            test_end  = train_end + test_days_per_fold if fold < n_folds - 1 else len(days)
            test_days_list = days[train_end:test_end]

            detector = RollingAnomalyDetector(n_estimators=200)
            result = detector.fit_score(feature_matrix, train_end)
            score_parts.append(result.loc[result.index.isin(test_days_list), "anomaly_score"])

        all_scores = pd.concat(score_parts).sort_index()
        cache = pd.DataFrame({"anomaly_score": all_scores})
        cache.index.name = "date"
        cache.to_parquet(ANOMALY_CACHE)
        logger.info("Anomaly cache saved: %d scores", len(cache))

    # ------------------------------------------------------------------ #
    # Incremental daily update (~15-25s)
    # ------------------------------------------------------------------ #

    def _update_today(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        """Fit final XGB + IF on all-but-today, score today, append to caches."""
        today = df.index[-1]

        # ── OOS cache: append today's prediction if not already there ───
        oos_cache = pd.read_parquet(OOS_CACHE) if OOS_CACHE.exists() else pd.DataFrame()
        if not oos_cache.empty and today in oos_cache.index:
            today_proba = [
                float(oos_cache.at[today, "p_bear"]),
                float(oos_cache.at[today, "p_bal"]),
                float(oos_cache.at[today, "p_bull"]),
            ]
            top_features: list[tuple[str, float]] = []
            logger.info("Today's prediction already cached, skipping XGB refit")
        else:
            logger.info("Fitting final XGB on all-but-today...")
            today_proba, top_features = self._fit_final_xgb(df, feature_cols, es_daily)
            # Append to OOS cache
            pred = int(np.argmax(today_proba))
            new_row = pd.DataFrame({
                "pred":   [pred],
                "actual": [-1],   # unknown until tomorrow
                "p_bear": [today_proba[0]],
                "p_bal":  [today_proba[1]],
                "p_bull": [today_proba[2]],
            }, index=pd.DatetimeIndex([today]))
            new_row.index.name = "date"
            updated = pd.concat([oos_cache, new_row])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
            updated.to_parquet(OOS_CACHE)

        # ── Anomaly cache: append today's score if not already there ────
        anomaly_cache = pd.read_parquet(ANOMALY_CACHE) if ANOMALY_CACHE.exists() else pd.DataFrame()
        if not anomaly_cache.empty and today in anomaly_cache.index:
            logger.info("Today's anomaly score already cached, skipping IF refit")
        else:
            logger.info("Fitting IF on all-but-today for today's anomaly score...")
            today_score = self._score_today_anomaly(df, feature_cols)
            new_row = pd.DataFrame({"anomaly_score": [today_score]},
                                   index=pd.DatetimeIndex([today]))
            new_row.index.name = "date"
            updated = pd.concat([anomaly_cache, new_row])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
            updated.to_parquet(ANOMALY_CACHE)

        return today_proba, top_features

    def _fit_final_xgb(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        from strategies.swing.training.regime_trainer import DEFAULT_PARAMS
        from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward
        from xgboost import XGBClassifier

        cfg = self._config
        valid_mask = df["y_structural"].isin([0, 1, 2])
        df_valid = df[valid_mask].copy()
        n = len(df_valid)

        if n < cfg.min_train_days + 1:
            return [0.33, 0.34, 0.33], []

        # HMM fitted on all-but-today
        hmm_feats = compute_hmm_features_walkforward(es_daily, n - 1, n_states=cfg.hmm_n_states)
        hmm_cols = [c for c in hmm_feats.columns if c not in df_valid.columns]
        for c in hmm_cols:
            df_valid[c] = hmm_feats[c].reindex(df_valid.index).fillna(0)

        all_feat_cols = [c for c in feature_cols if c in df_valid.columns] + hmm_cols
        X = df_valid[all_feat_cols].fillna(0).values
        y = df_valid["y_structural"].values.astype(int)

        X_train, y_train = X[:-1], y[:-1]
        X_today = X[-1:]

        class_counts = np.maximum(np.bincount(y_train, minlength=3).astype(float), 1.0)
        class_weights = len(y_train) / (3.0 * class_counts)
        class_weights[0] *= 1.5
        sample_weights = class_weights[y_train]

        params = {k: v for k, v in DEFAULT_PARAMS.items() if k != "bear_upweight"}
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        proba = model.predict_proba(X_today)[0].tolist()
        top_idx = np.argsort(model.feature_importances_)[::-1][:10]
        top_features = [(all_feat_cols[i], float(model.feature_importances_[i])) for i in top_idx]
        return proba, top_features

    def _score_today_anomaly(self, df: pd.DataFrame, feature_cols: list[str]) -> float:
        from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector

        valid_mask = df["y_structural"].isin([0, 1, 2])
        df_valid = df[valid_mask]
        feature_matrix = df_valid[[c for c in feature_cols if c in df_valid.columns]].fillna(0)
        n = len(feature_matrix)

        detector = RollingAnomalyDetector(n_estimators=200)
        result = detector.fit_score(feature_matrix, train_end_idx=n - 1)
        return float(result["anomaly_score"].iloc[-1])

    # ------------------------------------------------------------------ #
    # Change + risk (vectorized on full history, ~0.5s)
    # ------------------------------------------------------------------ #

    def _compute_change_risk(
        self,
        es_daily: pd.DataFrame,
        df: pd.DataFrame,
        feature_cols: list[str],
        anomaly_cache: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        from strategies.swing.detection.change_detector import compute_change_features
        from strategies.swing.detection.regime_risk import compute_regime_risk_score
        from strategies.swing.features.range_features import FEATURE_NAMES as RANGE_FEATURES

        # Anomaly scores aligned to full ES history (0 for warmup days)
        anomaly_scores = anomaly_cache["anomaly_score"].reindex(es_daily.index).fillna(0)
        daily_ret = es_daily["close"].pct_change().fillna(0)

        change_df = compute_change_features(daily_ret, anomaly_scores)

        anomaly_df = pd.DataFrame({"anomaly_score": anomaly_scores}, index=es_daily.index)
        range_cols = [c for c in RANGE_FEATURES if c in df.columns]
        range_feats = df[range_cols].reindex(es_daily.index).fillna(0)

        risk_df = compute_regime_risk_score(range_feats, anomaly_df, change_df, RISK_WEIGHTS)
        risk_df["anomaly_score"]      = anomaly_scores
        risk_df["return_cusum_score"] = change_df["return_cusum_score"]
        risk_df["anomaly_ewma_z"]     = change_df["anomaly_ewma_z"]

        return change_df, risk_df

    # ------------------------------------------------------------------ #
    # State assembly
    # ------------------------------------------------------------------ #

    def _build_state(
        self,
        es_daily: pd.DataFrame,
        nq_daily: pd.DataFrame,
        zn_daily: pd.DataFrame,
        anomaly_cache: pd.DataFrame,
        change_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        oos_cache: pd.DataFrame,
        today_proba: list[float],
        top_features: list[tuple[str, float]],
    ) -> DashboardState:
        cfg = self._config

        # HMM fitted on all-but-today, scored on full history
        from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward
        hmm_hist = compute_hmm_features_walkforward(
            es_daily, len(es_daily) - 1, n_states=cfg.hmm_n_states
        ).reindex(es_daily.index).fillna(0)

        # Rolling correlations on full history
        es_ret = es_daily["close"].pct_change()
        nq_ret = nq_daily["close"].pct_change().reindex(es_daily.index, method="ffill")
        zn_ret = zn_daily["close"].pct_change().reindex(es_daily.index, method="ffill")
        corr_nq = pd.DataFrame(index=es_daily.index)
        corr_zn = pd.DataFrame(index=es_daily.index)
        for w in [10, 20, 60]:
            corr_nq[f"corr_nq_{w}d"] = es_ret.rolling(w, min_periods=w // 2).corr(nq_ret)
            corr_zn[f"corr_zn_{w}d"] = es_ret.rolling(w, min_periods=w // 2).corr(zn_ret)

        # History: last N days of full ES history
        all_days = es_daily.index
        hist_days = all_days[-cfg.chart_history_days:]
        today_date = all_days[-1]

        history = []
        for d in hist_days:
            proba = self._get_proba(d, oos_cache, today_date, today_proba)
            row = self._build_day_state(
                d, es_daily, risk_df, change_df, hmm_hist, corr_nq, corr_zn, proba, []
            )
            if row is not None:
                history.append(row)

        today_state = self._build_day_state(
            today_date, es_daily, risk_df, change_df, hmm_hist, corr_nq, corr_zn,
            today_proba, top_features,
        )

        # OOS chart data (from cache)
        oos_dates = [
            d.date().isoformat() if hasattr(d, "date") else str(d)
            for d in oos_cache.index
        ]
        oos_preds   = oos_cache["pred"].tolist()
        oos_actuals = oos_cache["actual"].tolist()
        oos_probas  = oos_cache[["p_bear", "p_bal", "p_bull"]].values.tolist()

        # Model metrics (exclude today's row where actual=-1)
        valid_mask = oos_cache["actual"].isin([0, 1, 2])
        if valid_mask.sum() > 0:
            y_true = oos_cache.loc[valid_mask, "actual"].values
            y_pred = oos_cache.loc[valid_mask, "pred"].values
            acc = float(accuracy_score(y_true, y_pred))
            f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        else:
            acc = f1 = 0.0

        as_of = today_date.date().isoformat() if hasattr(today_date, "date") else str(today_date)
        return DashboardState(
            computed_at=datetime.utcnow().isoformat(),
            as_of_date=as_of,
            today=today_state,
            history=history,
            oos_predictions=oos_preds,
            oos_actuals=oos_actuals,
            oos_probas=oos_probas,
            oos_dates=oos_dates,
            model_accuracy=acc,
            model_f1=f1,
        )

    def _get_proba(
        self,
        d,
        oos_cache: pd.DataFrame,
        today_date,
        today_proba: list[float],
    ) -> list[float]:
        """Return [p_bear, p_bal, p_bull] for date d from the OOS cache."""
        if d == today_date:
            return today_proba
        if not oos_cache.empty and d in oos_cache.index:
            row = oos_cache.loc[d]
            return [float(row["p_bear"]), float(row["p_bal"]), float(row["p_bull"])]
        # Before OOS period (warmup days): no prediction available
        return [0.33, 0.34, 0.33]

    def _build_day_state(
        self,
        d,
        es_daily: pd.DataFrame,
        risk_df: pd.DataFrame,
        change_df: pd.DataFrame,
        hmm_hist: pd.DataFrame,
        corr_nq: pd.DataFrame,
        corr_zn: pd.DataFrame,
        proba: list[float],
        top_features: list[tuple[str, float]],
    ) -> Optional[DayState]:
        def _get(df, col, default=0.0):
            try:
                v = df.at[d, col]
                return float(v) if pd.notna(v) else default
            except (KeyError, TypeError):
                return default

        date_str = d.date().isoformat() if hasattr(d, "date") else str(d)
        predicted = int(np.argmax(proba))

        return DayState(
            date=date_str,
            open=_get(es_daily, "open"),
            high=_get(es_daily, "high"),
            low=_get(es_daily, "low"),
            close=_get(es_daily, "close"),
            volume=_get(es_daily, "volume"),
            risk_score=_get(risk_df, "risk_score"),
            risk_regime=int(_get(risk_df, "risk_regime")),
            range_stress=_get(risk_df, "range_stress"),
            anomaly_intensity=_get(risk_df, "anomaly_intensity"),
            change_momentum=_get(risk_df, "change_momentum"),
            anomaly_score=_get(risk_df, "anomaly_score"),
            return_cusum_score=_get(change_df, "return_cusum_score"),
            anomaly_ewma_z=_get(change_df, "anomaly_ewma_z"),
            p_bear=proba[0],
            p_balance=proba[1],
            p_bull=proba[2],
            predicted_regime=predicted,
            hmm_state=int(_get(hmm_hist, "hmm_state", 1)),
            hmm_bull_prob=_get(hmm_hist, "hmm_bull_prob", 0.33),
            hmm_bear_prob=_get(hmm_hist, "hmm_bear_prob", 0.33),
            corr_nq_10d=_get(corr_nq, "corr_nq_10d"),
            corr_nq_20d=_get(corr_nq, "corr_nq_20d"),
            corr_nq_60d=_get(corr_nq, "corr_nq_60d"),
            corr_zn_10d=_get(corr_zn, "corr_zn_10d"),
            corr_zn_20d=_get(corr_zn, "corr_zn_20d"),
            corr_zn_60d=_get(corr_zn, "corr_zn_60d"),
            top_features=top_features,
        )
