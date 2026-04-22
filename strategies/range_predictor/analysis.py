"""Core range prediction evaluation utilities.

Containment metrics and walk-forward OOS prediction generation.
Newsletter comparison functions have moved to strategies.range_predictor.newsletter.
"""

import numpy as np
import pandas as pd

from strategies.range_predictor.config import RangePredictorConfig
from strategies.range_predictor.features import (
    prepare_dataset,
    prepare_rth_dataset,
)


def compute_containment_rate(
    pred_low: pd.Series,
    pred_high: pd.Series,
    realized_low: pd.Series,
    realized_high: pd.Series,
) -> dict:
    """Compute how often predicted range contains the realized range.

    Returns dict with various containment metrics.
    """
    n = len(pred_low)

    # Full containment: predicted range fully contains realized range
    full_contain = (
        (pred_low <= realized_low) & (pred_high >= realized_high)
    ).sum() / n

    # High contained: realized high within predicted range
    high_within = (realized_high <= pred_high).sum() / n

    # Low contained: realized low within predicted range
    low_within = (realized_low >= pred_low).sum() / n

    # Partial overlap
    overlap = (
        (pred_low <= realized_high) & (pred_high >= realized_low)
    ).sum() / n

    # Width comparison
    pred_width = (pred_high - pred_low).mean()
    realized_width = (realized_high - realized_low).mean()

    return {
        'full_containment': full_contain,
        'high_contained': high_within,
        'low_contained': low_within,
        'any_overlap': overlap,
        'avg_pred_width': pred_width,
        'avg_realized_width': realized_width,
        'width_ratio': pred_width / realized_width if realized_width > 0 else float('inf'),
        'n_samples': n,
    }


def _generate_oos_predictions(
    daily: pd.DataFrame,
    timeframe: str,
    n_folds: int = 5,
    min_train_days: int = 100,
) -> pd.DataFrame:
    """Generate walk-forward OOS predictions for a given timeframe.

    Uses the same fold structure as RangeTrainer so the OOS predictions
    are directly comparable to the reported CV metrics.

    Returns:
        DataFrame with columns [pred_range_high, pred_range_low],
        indexed by date, covering only the OOS portion of the dataset.
    """
    from xgboost import XGBRegressor

    features_df, targets_df, feature_names = prepare_dataset(daily, timeframe=timeframe)
    config = RangePredictorConfig()

    n = len(features_df)
    test_per_fold = (n - min_train_days) // n_folds

    X_all = features_df[feature_names].values
    y_width = targets_df['width_pct'].values
    y_center = targets_df['center_pct'].values
    dates = features_df.index

    oos_width = {}
    oos_center = {}

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_per_fold
        test_end = train_end + test_per_fold
        if fold == n_folds - 1:
            test_end = n

        X_tr = X_all[:train_end]
        X_te = X_all[train_end:test_end]
        dates_te = dates[train_end:test_end]

        if len(X_te) == 0:
            continue

        m_width = XGBRegressor(**config.xgb_params)
        m_width.fit(X_tr, y_width[:train_end])
        m_center = XGBRegressor(**config.xgb_params)
        m_center.fit(X_tr, y_center[:train_end])

        for d, pw, pc in zip(dates_te, m_width.predict(X_te), m_center.predict(X_te)):
            oos_width[d] = pw
            oos_center[d] = pc

    prev_close = daily['close'].shift(1)
    result = pd.DataFrame({
        'pred_width_pct':  pd.Series(oos_width),
        'pred_center_pct': pd.Series(oos_center),
    })
    result['pred_range_high_pct'] = result['pred_width_pct'] / 2 + result['pred_center_pct']
    result['pred_range_low_pct'] = (result['pred_width_pct'] / 2 - result['pred_center_pct']).clip(lower=0)
    result['pred_range_high'] = prev_close.reindex(result.index) * (1 + result['pred_range_high_pct'])
    result['pred_range_low']  = prev_close.reindex(result.index) * (1 - result['pred_range_low_pct'])

    return result[['pred_range_high', 'pred_range_low']].dropna()


def _generate_rth_oos_predictions(
    full_daily: pd.DataFrame,
    rth_daily: pd.DataFrame,
    n_folds: int = 5,
    min_train_days: int = 100,
) -> pd.DataFrame:
    """Generate walk-forward OOS predictions for the RTH model.

    Uses the same fold structure as RangeTrainer.walk_forward_cv so the OOS
    predictions are directly comparable to the reported CV metrics.

    Args:
        full_daily: Full-session daily OHLCV (DatetimeIndex).
        rth_daily: RTH daily DataFrame with rth_open/high/low/close.
        n_folds: Number of walk-forward folds.
        min_train_days: Minimum training set size per fold.

    Returns:
        DataFrame with columns [pred_rth_high, pred_rth_low, pred_rth_width],
        indexed by date, covering only the OOS portion of the dataset.
    """
    from xgboost import XGBRegressor

    features_df, targets_df, feature_names = prepare_rth_dataset(
        full_daily, rth_daily
    )
    config = RangePredictorConfig()

    n = len(features_df)
    test_per_fold = (n - min_train_days) // n_folds

    X_all = features_df[feature_names].values
    y_high = targets_df['rth_range_high_pct'].values
    y_low = targets_df['rth_range_low_pct'].values
    dates = features_df.index

    oos_high = {}
    oos_low = {}

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_per_fold
        test_end = train_end + test_per_fold
        if fold == n_folds - 1:
            test_end = n

        X_tr = X_all[:train_end]
        X_te = X_all[train_end:test_end]
        dates_te = dates[train_end:test_end]

        if len(X_te) == 0:
            continue

        m_high = XGBRegressor(**config.xgb_params)
        m_high.fit(X_tr, y_high[:train_end])
        m_low = XGBRegressor(**config.xgb_params)
        m_low.fit(X_tr, y_low[:train_end])

        for d, ph, pl in zip(dates_te, m_high.predict(X_te), m_low.predict(X_te)):
            oos_high[d] = ph
            oos_low[d] = pl

    # Convert pct predictions to price levels using rth_open
    rth_open = rth_daily['rth_open']
    result = pd.DataFrame({
        'pred_rth_range_high_pct': pd.Series(oos_high),
        'pred_rth_range_low_pct': pd.Series(oos_low),
    })
    rth_open_aligned = rth_open.reindex(result.index)
    result['pred_rth_high'] = rth_open_aligned * (1 + result['pred_rth_range_high_pct'])
    result['pred_rth_low'] = rth_open_aligned * (1 - result['pred_rth_range_low_pct'])
    result['pred_rth_width'] = result['pred_rth_high'] - result['pred_rth_low']

    return result[['pred_rth_high', 'pred_rth_low', 'pred_rth_width']].dropna()
