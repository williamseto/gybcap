"""SwingDashboardRunner: wraps the full swing pipeline into a daily refresh.

Architecture
------------
Two parquet caches separate expensive one-time work from cheap daily updates:

  oos_cache.parquet     — walk-forward XGB predictions per date (built once)
  anomaly_cache.parquet — walk-forward IF anomaly scores per date (built once)

Bootstrap (first run, slow ~4-6 min):
  - Run 5-fold walk-forward XGB → oos_cache
  - Run 5-fold walk-forward Isolation Forest → anomaly_cache

Daily refresh (fast if no gaps; scales with missing days):
  1. Fetch new daily bar (yfinance top-up, ~5s)
  2. Compute features vectorially (~2s)
  3. Check caches and fill any missing dates since last refresh
  4. For each missing date, fit XGB on prior data only → append OOS row
  5. For each missing date, fit IF on prior data only → append anomaly row
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
    # False: fast single-train backlog catchup (production)
    # True: strict per-missing-day causal backfill (research)
    strict_backfill: bool = False


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
        t0 = time.perf_counter()

        # ── 1. Fetch daily data ─────────────────────────────────────────
        t_fetch = time.perf_counter()
        logger.info("Fetching daily data...")
        self._fetcher.topup_external_csvs()
        es_daily = self._fetcher.fetch_and_update("ES")
        nq_daily = self._fetcher.fetch_and_update("NQ")
        zn_daily = self._fetcher.fetch_and_update("ZN")
        logger.info("Stage timing: data fetch %.2fs", time.perf_counter() - t_fetch)

        # ── 2. Compute features (vectorized, ~2s) ───────────────────────
        t_features = time.perf_counter()
        logger.info("Computing features (%d ES days)...", len(es_daily))
        features, feature_cols = self._compute_features(es_daily, nq_daily, zn_daily)
        labels_df = self._compute_labels(es_daily)
        df = features.join(labels_df).dropna(subset=["y_structural"])
        logger.info("Stage timing: features+labels %.2fs", time.perf_counter() - t_features)

        # ── 3. Bootstrap caches if missing (slow, one-time) ────────────
        t_bootstrap = time.perf_counter()
        self._ensure_historical_caches(df, feature_cols, es_daily)
        logger.info("Stage timing: cache bootstrap/check %.2fs", time.perf_counter() - t_bootstrap)

        # ── 4. Incremental cache update (fills any missing days causally) ─
        t_update = time.perf_counter()
        today_proba, top_features = self._update_prediction_caches(df, feature_cols, es_daily)
        logger.info("Stage timing: prediction cache update %.2fs", time.perf_counter() - t_update)

        # ── 5. Load caches + compute change/risk on full history ────────
        t_risk = time.perf_counter()
        anomaly_cache = pd.read_parquet(ANOMALY_CACHE)
        oos_cache     = pd.read_parquet(OOS_CACHE)
        change_df, risk_df = self._compute_change_risk(es_daily, df, feature_cols, anomaly_cache)
        logger.info("Stage timing: change+risk %.2fs", time.perf_counter() - t_risk)

        # ── 6. Assemble state ───────────────────────────────────────────
        t_state = time.perf_counter()
        logger.info("Assembling state...")
        state = self._build_state(
            es_daily, nq_daily, zn_daily,
            anomaly_cache, change_df, risk_df, oos_cache,
            today_proba, top_features,
        )
        logger.info("Stage timing: state assembly %.2fs", time.perf_counter() - t_state)
        logger.info("Stage timing: pipeline total %.2fs", time.perf_counter() - t0)
        return state

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
        # Build historical caches on all rows except the latest prediction row.
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

    def _update_prediction_caches(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        """Fill missing OOS/anomaly cache rows causally and return latest prediction."""
        valid_mask = df["y_structural"].isin([0, 1, 2])
        df_valid = df[valid_mask].copy()
        if df_valid.empty:
            return [0.33, 0.34, 0.33], []

        all_dates = list(df_valid.index)
        latest_date = all_dates[-1]
        top_features: list[tuple[str, float]] = []

        oos_cache = (
            pd.read_parquet(OOS_CACHE)
            if OOS_CACHE.exists()
            else pd.DataFrame(columns=["pred", "actual", "p_bear", "p_bal", "p_bull"])
        )
        anomaly_cache = (
            pd.read_parquet(ANOMALY_CACHE)
            if ANOMALY_CACHE.exists()
            else pd.DataFrame(columns=["anomaly_score"])
        )
        if not oos_cache.empty:
            oos_cache.index = pd.to_datetime(oos_cache.index)
            oos_cache = oos_cache.sort_index()
        if not anomaly_cache.empty:
            anomaly_cache.index = pd.to_datetime(anomaly_cache.index)
            anomaly_cache = anomaly_cache.sort_index()

        # Fill stale placeholder actuals once labels become available.
        oos_cache_before = oos_cache.copy()
        oos_cache = self._backfill_oos_actuals(oos_cache, df_valid)
        oos_backfill_changed = not oos_cache.equals(oos_cache_before)

        oos_missing = self._find_missing_dates(all_dates, oos_cache)
        if oos_missing:
            logger.info(
                "Backfilling OOS cache for %d missing day(s): %s -> %s",
                len(oos_missing),
                oos_missing[0].date() if hasattr(oos_missing[0], "date") else oos_missing[0],
                oos_missing[-1].date() if hasattr(oos_missing[-1], "date") else oos_missing[-1],
            )
            if self._config.strict_backfill:
                new_rows = []
                for d in oos_missing:
                    pred_idx = int(df_valid.index.get_loc(d))
                    proba, feat_imp = self._fit_xgb_for_index(
                        df_valid, feature_cols, es_daily, pred_idx, pred_date=d
                    )
                    pred = int(np.argmax(proba))
                    actual = -1 if d == latest_date else int(df_valid.at[d, "y_structural"])
                    new_rows.append({
                        "date": d,
                        "pred": pred,
                        "actual": actual,
                        "p_bear": float(proba[0]),
                        "p_bal": float(proba[1]),
                        "p_bull": float(proba[2]),
                    })
                    if d == latest_date:
                        top_features = feat_imp
            else:
                logger.info("Production mode: single-train OOS backfill for %d day(s)", len(oos_missing))
                proba_map, top_features = self._predict_missing_oos_fast(
                    df_valid, feature_cols, es_daily, oos_missing
                )
                new_rows = []
                for d in oos_missing:
                    proba = proba_map.get(d, [0.33, 0.34, 0.33])
                    pred = int(np.argmax(proba))
                    actual = -1 if d == latest_date else int(df_valid.at[d, "y_structural"])
                    new_rows.append({
                        "date": d,
                        "pred": pred,
                        "actual": actual,
                        "p_bear": float(proba[0]),
                        "p_bal": float(proba[1]),
                        "p_bull": float(proba[2]),
                    })

            new_df = pd.DataFrame(new_rows).set_index("date")
            updated = pd.concat([oos_cache, new_df])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
            updated.to_parquet(OOS_CACHE)
            oos_cache = updated
        elif oos_backfill_changed:
            oos_cache.to_parquet(OOS_CACHE)

        anomaly_missing = self._find_missing_dates(all_dates, anomaly_cache)
        if anomaly_missing:
            logger.info(
                "Backfilling anomaly cache for %d missing day(s): %s -> %s",
                len(anomaly_missing),
                anomaly_missing[0].date() if hasattr(anomaly_missing[0], "date") else anomaly_missing[0],
                anomaly_missing[-1].date() if hasattr(anomaly_missing[-1], "date") else anomaly_missing[-1],
            )
            if self._config.strict_backfill:
                new_rows = []
                for d in anomaly_missing:
                    pred_idx = int(df_valid.index.get_loc(d))
                    score = self._score_anomaly_for_index(df_valid, feature_cols, pred_idx)
                    new_rows.append({"date": d, "anomaly_score": score})
            else:
                logger.info("Production mode: single-fit anomaly backfill for %d day(s)", len(anomaly_missing))
                score_map = self._score_missing_anomaly_fast(df_valid, feature_cols, anomaly_missing)
                new_rows = [{"date": d, "anomaly_score": float(score_map.get(d, 0.0))} for d in anomaly_missing]

            new_df = pd.DataFrame(new_rows).set_index("date")
            updated = pd.concat([anomaly_cache, new_df])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
            updated.to_parquet(ANOMALY_CACHE)
            anomaly_cache = updated

        if not oos_cache.empty and latest_date in oos_cache.index:
            today_row = oos_cache.loc[latest_date]
            today_proba = [
                float(today_row["p_bear"]),
                float(today_row["p_bal"]),
                float(today_row["p_bull"]),
            ]
        else:
            logger.warning("Latest date %s missing from OOS cache; using neutral probabilities", latest_date)
            today_proba = [0.33, 0.34, 0.33]

        return today_proba, top_features

    def _find_missing_dates(self, all_dates: list, cache_df: pd.DataFrame) -> list:
        """Return prediction-eligible dates that are absent in cache index."""
        if not all_dates:
            return []

        # Caches intentionally do not cover warmup history before min_train_days.
        start_idx = min(max(self._config.min_train_days, 0), len(all_dates) - 1)
        candidate_dates = all_dates[start_idx:]
        if not candidate_dates:
            return []

        if cache_df.empty:
            return [candidate_dates[-1]]
        cached = set(pd.to_datetime(cache_df.index))
        return [d for d in candidate_dates if d not in cached]

    def _backfill_oos_actuals(self, oos_cache: pd.DataFrame, df_valid: pd.DataFrame) -> pd.DataFrame:
        """Update prior cache rows with actual labels once they are no longer the latest day."""
        if oos_cache.empty or "actual" not in oos_cache.columns:
            return oos_cache

        latest_date = df_valid.index[-1]
        labels = df_valid["y_structural"].astype(int)
        pending_idx = [
            d for d in oos_cache[oos_cache["actual"].eq(-1)].index.unique()
            if d < latest_date
        ]
        if not pending_idx:
            return oos_cache

        updated = oos_cache.copy()
        n = 0
        for d in pending_idx:
            if d in labels.index:
                updated.at[d, "actual"] = int(labels.at[d])
                n += 1

        if n > 0:
            logger.info("Backfilled actual labels for %d prior OOS day(s)", n)
        return updated

    def _predict_missing_oos_fast(
        self,
        df_valid: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
        missing_dates: list,
    ) -> tuple[dict, list[tuple[str, float]]]:
        """Fast backlog fill: one model fit, predict all missing dates (hindsight-biased)."""
        from strategies.swing.training.regime_trainer import DEFAULT_PARAMS
        from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward
        from xgboost import XGBClassifier

        if not missing_dates:
            return {}, []

        latest_idx = len(df_valid) - 1
        if latest_idx < self._config.min_train_days:
            neutral = {d: [0.33, 0.34, 0.33] for d in missing_dates}
            return neutral, []

        df_aug = df_valid.copy()
        try:
            hmm_train_end = int(es_daily.index.get_loc(df_valid.index[-1]))
        except Exception:
            hmm_train_end = latest_idx
        hmm_feats = compute_hmm_features_walkforward(
            es_daily, hmm_train_end, n_states=self._config.hmm_n_states
        )
        hmm_cols = [c for c in hmm_feats.columns if c not in df_aug.columns]
        for c in hmm_cols:
            df_aug[c] = hmm_feats[c].reindex(df_aug.index).fillna(0)

        all_feat_cols = [c for c in feature_cols if c in df_aug.columns] + hmm_cols
        X = df_aug[all_feat_cols].fillna(0).values
        y = df_aug["y_structural"].values.astype(int)
        X_train, y_train = X[:-1], y[:-1]
        if len(X_train) == 0:
            neutral = {d: [0.33, 0.34, 0.33] for d in missing_dates}
            return neutral, []

        missing_classes = set([0, 1, 2]) - set(np.unique(y_train))
        if missing_classes:
            for cls in missing_classes:
                X_train = np.vstack([X_train, np.zeros((1, X_train.shape[1]))])
                y_train = np.append(y_train, cls)

        class_counts = np.maximum(np.bincount(y_train, minlength=3).astype(float), 1.0)
        class_weights = len(y_train) / (3.0 * class_counts)
        class_weights[0] *= 1.5
        sample_weights = class_weights[y_train]

        params = {k: v for k, v in DEFAULT_PARAMS.items() if k != "bear_upweight"}
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        pred_idx = [int(df_valid.index.get_loc(d)) for d in missing_dates]
        probas = model.predict_proba(X[pred_idx])
        proba_map = {d: probas[i].tolist() for i, d in enumerate(missing_dates)}

        top_idx = np.argsort(model.feature_importances_)[::-1][:10]
        top_features = [(all_feat_cols[i], float(model.feature_importances_[i])) for i in top_idx]
        return proba_map, top_features

    def _score_missing_anomaly_fast(
        self,
        df_valid: pd.DataFrame,
        feature_cols: list[str],
        missing_dates: list,
    ) -> dict:
        """Fast backlog fill: one IF fit, score all missing dates (hindsight-biased)."""
        from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector

        if not missing_dates:
            return {}

        feature_matrix = df_valid[[c for c in feature_cols if c in df_valid.columns]].fillna(0)
        if len(feature_matrix) < 2:
            return {d: 0.0 for d in missing_dates}

        detector = RollingAnomalyDetector(n_estimators=200)
        result = detector.fit_score(feature_matrix, train_end_idx=len(feature_matrix) - 1)
        return {d: float(result.at[d, "anomaly_score"]) if d in result.index else 0.0 for d in missing_dates}

    def _fit_final_xgb(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        """Backward-compatible wrapper for scoring the latest valid row."""
        valid_mask = df["y_structural"].isin([0, 1, 2])
        df_valid = df[valid_mask].copy()
        if df_valid.empty:
            return [0.33, 0.34, 0.33], []
        pred_idx = len(df_valid) - 1
        return self._fit_xgb_for_index(df_valid, feature_cols, es_daily, pred_idx)

    def _fit_xgb_for_index(
        self,
        df_valid: pd.DataFrame,
        feature_cols: list[str],
        es_daily: pd.DataFrame,
        pred_idx: int,
        pred_date=None,
    ) -> tuple[list[float], list[tuple[str, float]]]:
        from strategies.swing.training.regime_trainer import DEFAULT_PARAMS
        from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward
        from xgboost import XGBClassifier

        cfg = self._config
        n = len(df_valid)
        if pred_idx < 0 or pred_idx >= n:
            raise IndexError(f"pred_idx out of range: {pred_idx} (n={n})")

        # Need at least min_train_days before the prediction row.
        if pred_idx < cfg.min_train_days:
            return [0.33, 0.34, 0.33], []

        df_slice = df_valid.iloc[: pred_idx + 1].copy()

        # HMM fitted on all rows before the prediction date.
        hmm_train_end = pred_idx
        if pred_date is not None:
            try:
                hmm_train_end = int(es_daily.index.get_loc(pred_date))
            except Exception:
                hmm_train_end = pred_idx
        hmm_feats = compute_hmm_features_walkforward(es_daily, hmm_train_end, n_states=cfg.hmm_n_states)
        hmm_cols = [c for c in hmm_feats.columns if c not in df_valid.columns]
        for c in hmm_cols:
            df_slice[c] = hmm_feats[c].reindex(df_slice.index).fillna(0)

        all_feat_cols = [c for c in feature_cols if c in df_slice.columns] + hmm_cols
        X = df_slice[all_feat_cols].fillna(0).values
        y = df_slice["y_structural"].values.astype(int)

        X_train, y_train = X[:-1], y[:-1]
        X_today = X[-1:]
        if len(X_train) == 0:
            return [0.33, 0.34, 0.33], []

        missing = set([0, 1, 2]) - set(np.unique(y_train))
        if missing:
            for cls in missing:
                X_train = np.vstack([X_train, np.zeros((1, X_train.shape[1]))])
                y_train = np.append(y_train, cls)

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
        """Backward-compatible wrapper for scoring anomaly on latest valid row."""
        valid_mask = df["y_structural"].isin([0, 1, 2])
        df_valid = df[valid_mask].copy()
        if df_valid.empty:
            return 0.0
        return self._score_anomaly_for_index(df_valid, feature_cols, len(df_valid) - 1)

    def _score_anomaly_for_index(
        self,
        df_valid: pd.DataFrame,
        feature_cols: list[str],
        pred_idx: int,
    ) -> float:
        from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector

        feature_matrix = df_valid[[c for c in feature_cols if c in df_valid.columns]].fillna(0)
        feature_slice = feature_matrix.iloc[: pred_idx + 1]
        if len(feature_slice) < 2:
            return 0.0

        detector = RollingAnomalyDetector(n_estimators=200)
        result = detector.fit_score(feature_slice, train_end_idx=len(feature_slice) - 1)
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
