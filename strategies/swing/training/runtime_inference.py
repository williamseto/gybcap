"""Shared runtime inference helpers for daily swing regime predictions."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NEUTRAL_PROBA = [0.33, 0.34, 0.33]


def fit_predict_proba_for_index(
    df_valid: pd.DataFrame,
    feature_cols: list[str],
    es_daily: pd.DataFrame,
    pred_idx: int,
    *,
    min_train_days: int,
    hmm_n_states: int,
    params: dict[str, Any] | None = None,
    pred_date=None,
) -> tuple[list[float], list[tuple[str, float]]]:
    """Fit on rows before pred_idx, then score the prediction row."""
    n = len(df_valid)
    if pred_idx < 0 or pred_idx >= n:
        raise IndexError(f"pred_idx out of range: {pred_idx} (n={n})")
    if pred_idx < min_train_days:
        return list(NEUTRAL_PROBA), []

    df_slice = df_valid.iloc[: pred_idx + 1].copy()
    hmm_train_end = pred_idx
    if pred_date is not None:
        try:
            hmm_train_end = int(es_daily.index.get_loc(pred_date))
        except Exception:
            hmm_train_end = pred_idx

    df_aug, all_feat_cols = _augment_with_hmm(
        df_slice,
        feature_cols,
        es_daily,
        hmm_train_end=hmm_train_end,
        hmm_n_states=hmm_n_states,
    )

    X = df_aug[all_feat_cols].fillna(0).values
    y = df_aug["y_structural"].values.astype(int)
    X_train, y_train = X[:-1], y[:-1]
    if len(X_train) == 0:
        return list(NEUTRAL_PROBA), []

    model = _fit_multiclass_xgb(X_train, y_train, params=params)
    proba = model.predict_proba(X[-1:])[0].tolist()
    return proba, _top_feature_importances(model, all_feat_cols)


def fit_predict_proba_for_dates_fast(
    df_valid: pd.DataFrame,
    feature_cols: list[str],
    es_daily: pd.DataFrame,
    missing_dates: list,
    *,
    min_train_days: int,
    hmm_n_states: int,
    params: dict[str, Any] | None = None,
) -> tuple[dict, list[tuple[str, float]]]:
    """Fit once on the latest training set, then score all missing dates."""
    if not missing_dates:
        return {}, []

    latest_idx = len(df_valid) - 1
    if latest_idx < min_train_days:
        return ({d: list(NEUTRAL_PROBA) for d in missing_dates}, [])

    try:
        hmm_train_end = int(es_daily.index.get_loc(df_valid.index[-1]))
    except Exception:
        hmm_train_end = latest_idx

    df_aug, all_feat_cols = _augment_with_hmm(
        df_valid.copy(),
        feature_cols,
        es_daily,
        hmm_train_end=hmm_train_end,
        hmm_n_states=hmm_n_states,
    )

    X = df_aug[all_feat_cols].fillna(0).values
    y = df_aug["y_structural"].values.astype(int)
    X_train, y_train = X[:-1], y[:-1]
    if len(X_train) == 0:
        return ({d: list(NEUTRAL_PROBA) for d in missing_dates}, [])

    model = _fit_multiclass_xgb(X_train, y_train, params=params)
    pred_idx = [int(df_valid.index.get_loc(d)) for d in missing_dates]
    probas = model.predict_proba(X[pred_idx])
    proba_map = {d: probas[i].tolist() for i, d in enumerate(missing_dates)}
    return proba_map, _top_feature_importances(model, all_feat_cols)


def _augment_with_hmm(
    df_frame: pd.DataFrame,
    feature_cols: list[str],
    es_daily: pd.DataFrame,
    *,
    hmm_train_end: int,
    hmm_n_states: int,
) -> tuple[pd.DataFrame, list[str]]:
    from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward

    hmm_feats = compute_hmm_features_walkforward(
        es_daily,
        hmm_train_end,
        n_states=hmm_n_states,
    )
    hmm_cols = [c for c in hmm_feats.columns if c not in df_frame.columns]
    for col in hmm_cols:
        df_frame[col] = hmm_feats[col].reindex(df_frame.index).fillna(0)

    all_feat_cols = [c for c in feature_cols if c in df_frame.columns] + hmm_cols
    return df_frame, all_feat_cols


def _fit_multiclass_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
):
    from xgboost import XGBClassifier

    from strategies.swing.training.regime_trainer import DEFAULT_PARAMS

    xgb_params = {**DEFAULT_PARAMS, **(params or {})}
    bear_upweight = float(xgb_params.pop("bear_upweight", 1.5))

    missing_classes = set([0, 1, 2]) - set(np.unique(y_train))
    if missing_classes:
        for cls in missing_classes:
            X_train = np.vstack([X_train, np.zeros((1, X_train.shape[1]))])
            y_train = np.append(y_train, cls)

    class_counts = np.maximum(np.bincount(y_train, minlength=3).astype(float), 1.0)
    class_weights = len(y_train) / (3.0 * class_counts)
    class_weights[0] *= bear_upweight
    sample_weights = class_weights[y_train]

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    return model


def _top_feature_importances(model, feature_cols: list[str]) -> list[tuple[str, float]]:
    top_idx = np.argsort(model.feature_importances_)[::-1][:10]
    return [(feature_cols[i], float(model.feature_importances_[i])) for i in top_idx]
