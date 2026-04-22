"""Reverse-engineered newsletter range formula.

Fits and applies the linear formula that reproduces the newsletter's daily ES
range, derived by reverse-engineering on ~490 days of overlapping data.

The newsletter behaves as a GARCH-style recursive update:

    nl_width_t ≈ α + β₁ · nl_width_{t-1} + β₂ · prev_ret + β₃ · |prev_ret|

Typical fitted coefs on our dataset (walk-forward OOS):
    α  ≈ +7.0        (baseline drift)
    β₁ ≈ +0.91       (high AR persistence — barely changes day to day)
    β₂ ≈ -480        (directional leverage — down moves widen strongly)
    β₃ ≈ +218        (absolute-return response — any big move widens)

OOS performance vs newsletter (5-fold walk-forward, 386 days):
    MAE (width)  = 4.5 pts
    corr         = 0.96
    R²           = 0.92

This absolutely dominates the 60-feature XGBoost replica (MAE 15.8, corr 0.50)
because the underlying process is linear; trees step-function-approximate it
and add noise.

Midpoint: set to prev_close (adding predicted drift adds no OOS value — the
shift distribution has std≈16pt but no model beats the 0-shift baseline on
MAE; see sandbox/newsletter_formula.py for details).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_WIDTH_FEATURES: Tuple[str, ...] = (
    'nl_width_lag1', 'prev_ret', 'prev_abs_ret',
)


def build_formula_features(
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
) -> pd.DataFrame:
    """Build the minimal feature frame used by the newsletter formula.

    Args:
        daily: Daily OHLCV DataFrame (DatetimeIndex).
        newsletter: Newsletter DataFrame with columns date, symbol, timeframe,
                    range_low, range_high (or pre-filtered with these cols).

    Returns:
        DataFrame indexed by date with columns:
            prev_close, nl_width, nl_mid, nl_high, nl_low,
            nl_width_lag1, prev_ret, prev_abs_ret.
        Rows with missing values (first 2 days) are dropped.
    """
    nl = newsletter.copy()
    nl['date'] = pd.to_datetime(nl['date'])
    if 'symbol' in nl.columns and 'timeframe' in nl.columns:
        nl = nl[(nl['symbol'] == 'ES') & (nl['timeframe'] == 'daily')].copy()
    nl = nl.set_index('date').sort_index()

    # Normalize column names
    if 'ES_range_low' in nl.columns:
        nl = nl.rename(columns={'ES_range_low': 'range_low',
                                'ES_range_high': 'range_high'})
    if 'range_low' not in nl.columns:
        raise ValueError("Newsletter DataFrame missing range_low/range_high")

    c = daily['close']
    common = c.dropna().index.intersection(nl.index).sort_values()

    df = pd.DataFrame(index=common)
    df['prev_close'] = c.shift(1).reindex(common)
    df['nl_low'] = nl.loc[common, 'range_low']
    df['nl_high'] = nl.loc[common, 'range_high']
    df['nl_width'] = df['nl_high'] - df['nl_low']
    df['nl_mid'] = (df['nl_high'] + df['nl_low']) / 2

    df['nl_width_lag1'] = df['nl_width'].shift(1)
    df['prev_ret'] = c.pct_change().reindex(common).shift(1)
    df['prev_abs_ret'] = df['prev_ret'].abs()

    return df.dropna()


@dataclass
class FormulaFit:
    intercept: float
    coefs: Dict[str, float]   # feature_name -> coefficient
    n_samples: int

    def to_dict(self) -> Dict:
        return {
            'intercept': self.intercept,
            'coefs': self.coefs,
            'n_samples': self.n_samples,
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predict width from a DataFrame with the required feature columns."""
        result = np.full(len(features), self.intercept)
        for name, coef in self.coefs.items():
            if name not in features.columns:
                raise KeyError(f"Missing feature: {name}")
            result = result + coef * features[name].values
        return pd.Series(result, index=features.index).clip(lower=0)


def fit_formula(
    df: pd.DataFrame,
    feature_cols: Tuple[str, ...] = DEFAULT_WIDTH_FEATURES,
    target_col: str = 'nl_width',
) -> FormulaFit:
    """Fit the linear newsletter-width formula on the full available dataset.

    Args:
        df: Output of build_formula_features().
        feature_cols: Features to regress on. Default is the proven triple
                      (nl_width_lag1, prev_ret, prev_abs_ret).
        target_col: Target column name (default nl_width).

    Returns:
        FormulaFit object with intercept and coefficients.
    """
    X = df[list(feature_cols)].values
    y = df[target_col].values
    Xb = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return FormulaFit(
        intercept=float(coef[0]),
        coefs={name: float(coef[i + 1]) for i, name in enumerate(feature_cols)},
        n_samples=len(df),
    )


def predict_walk_forward(
    df: pd.DataFrame,
    feature_cols: Tuple[str, ...] = DEFAULT_WIDTH_FEATURES,
    n_folds: int = 5,
    min_train_days: int = 100,
) -> pd.DataFrame:
    """Walk-forward OOS newsletter-width predictions using the linear formula.

    Args:
        df: Output of build_formula_features().
        feature_cols: Features to use.
        n_folds: Number of walk-forward folds.
        min_train_days: Minimum training window size for fold 0.

    Returns:
        DataFrame with columns [pred_width, pred_mid, pred_high, pred_low]
        indexed by date. OOS-only rows; in-sample rows are omitted.
    """
    X = df[list(feature_cols)].values
    y = df['nl_width'].values
    prev_close = df['prev_close'].values
    n = len(df)

    preds = np.full(n, np.nan)
    test_per_fold = (n - min_train_days) // n_folds

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_per_fold
        test_end = train_end + test_per_fold if fold < n_folds - 1 else n

        Xtr_b = np.column_stack([np.ones(train_end), X[:train_end]])
        Xte_b = np.column_stack([np.ones(test_end - train_end),
                                 X[train_end:test_end]])
        coef, *_ = np.linalg.lstsq(Xtr_b, y[:train_end], rcond=None)
        preds[train_end:test_end] = Xte_b @ coef

    mask = ~np.isnan(preds)
    out = pd.DataFrame(index=df.index[mask])
    out['pred_width'] = np.maximum(preds[mask], 0)
    out['pred_mid'] = prev_close[mask]   # prev_close is best midpoint baseline
    out['pred_high'] = out['pred_mid'] + out['pred_width'] / 2
    out['pred_low'] = out['pred_mid'] - out['pred_width'] / 2
    return out


def evaluate_formula(
    df: pd.DataFrame,
    preds: pd.DataFrame,
    verbose: bool = True,
) -> Dict:
    """Compute evaluation metrics on OOS newsletter predictions.

    Compares predicted low/high against both the actual newsletter
    and realized daily high/low.

    Args:
        df: Output of build_formula_features() — provides the ground truth
            newsletter values and prev_close.
        preds: Output of predict_walk_forward() — OOS predictions.
        verbose: Print summary.

    Returns:
        Dict with containment and MAE metrics.
    """
    idx = preds.index
    if not idx.isin(df.index).all():
        raise ValueError("preds index not a subset of df index")

    pred_hi = preds['pred_high'].values
    pred_lo = preds['pred_low'].values
    pred_w = preds['pred_width'].values
    pred_m = preds['pred_mid'].values

    nl_hi = df.loc[idx, 'nl_high'].values
    nl_lo = df.loc[idx, 'nl_low'].values
    nl_w = df.loc[idx, 'nl_width'].values
    nl_m = df.loc[idx, 'nl_mid'].values

    width_mae = float(np.mean(np.abs(pred_w - nl_w)))
    width_corr = float(np.corrcoef(pred_w, nl_w)[0, 1])
    ss_res = float(np.sum((nl_w - pred_w) ** 2))
    ss_tot = float(np.sum((nl_w - np.mean(nl_w)) ** 2))
    width_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    high_mae_nl = float(np.mean(np.abs(pred_hi - nl_hi)))
    low_mae_nl = float(np.mean(np.abs(pred_lo - nl_lo)))
    mid_mae_nl = float(np.mean(np.abs(pred_m - nl_m)))

    result = {
        'n_samples': len(preds),
        'vs_newsletter': {
            'width_mae': width_mae,
            'width_corr': width_corr,
            'width_r2': width_r2,
            'high_mae': high_mae_nl,
            'low_mae': low_mae_nl,
            'mid_mae': mid_mae_nl,
        },
    }

    if verbose:
        print(f"\n  OOS newsletter-formula predictions: {len(preds)} days")
        print(f"  vs Newsletter:")
        print(f"    width MAE    = {width_mae:6.2f} pts")
        print(f"    width corr   = {width_corr:+.4f}")
        print(f"    width R²     = {width_r2:+.4f}")
        print(f"    high MAE     = {high_mae_nl:6.2f} pts")
        print(f"    low MAE      = {low_mae_nl:6.2f} pts")
        print(f"    midpoint MAE = {mid_mae_nl:6.2f} pts")

    return result


def run_formula_analysis(
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
    feature_cols: Tuple[str, ...] = DEFAULT_WIDTH_FEATURES,
    n_folds: int = 5,
    min_train_days: int = 100,
    verbose: bool = True,
) -> Dict:
    """End-to-end fit-and-evaluate pipeline for the newsletter formula.

    Args:
        daily: Daily OHLCV DataFrame.
        newsletter: Newsletter predictions DataFrame.
        feature_cols: Feature columns for the formula.
        n_folds: Walk-forward folds.
        min_train_days: Minimum training window.
        verbose: Print fit and evaluation.

    Returns:
        Dict with 'final_fit', 'oos_preds', 'metrics'.
    """
    df = build_formula_features(daily, newsletter)
    if verbose:
        print(f"\n{'='*72}")
        print("NEWSLETTER FORMULA — reverse-engineered linear replica")
        print(f"{'='*72}")
        print(f"Dataset: {len(df)} days  ({df.index[0].date()} — {df.index[-1].date()})")
        print(f"Features: {list(feature_cols)}")

    # Fit on full history (for deployment)
    final_fit = fit_formula(df, feature_cols=feature_cols)
    if verbose:
        print(f"\n  FINAL FIT (full history, for live use):")
        print(f"    intercept = {final_fit.intercept:+.3f}")
        for k, v in final_fit.coefs.items():
            print(f"    {k:<20} coef = {v:+.4f}")

    # Walk-forward OOS for honest evaluation
    preds = predict_walk_forward(
        df, feature_cols=feature_cols,
        n_folds=n_folds, min_train_days=min_train_days,
    )
    metrics = evaluate_formula(df, preds, verbose=verbose)

    if verbose:
        print(f"\n{'='*72}\n")

    return {
        'final_fit': final_fit,
        'oos_preds': preds,
        'metrics': metrics,
        'feature_dataframe': df,
    }
