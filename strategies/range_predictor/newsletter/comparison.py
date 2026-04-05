"""Newsletter comparison and reverse-engineering analysis.

Compares the independent range prediction model against the newsletter's
predictions. Outputs diagnostic metrics and summary.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from strategies.range_predictor.config import TIMEFRAME_HORIZONS, RangePredictorConfig
from strategies.range_predictor.features import (
    compute_targets,
    compute_range_features,
    _compute_atr,
)
from strategies.range_predictor.predictor import RangePredictor
from strategies.range_predictor.analysis import (
    compute_containment_rate,
    _generate_oos_predictions,
)


def _filter_es_daily_newsletter(newsletter: pd.DataFrame) -> pd.DataFrame:
    """Filter newsletter DataFrame to ES daily rows only.

    Handles both the raw multi-symbol CSV (with symbol/timeframe columns)
    and pre-filtered DataFrames with ES_range_low/ES_range_high columns.

    Returns DataFrame with columns: date, nl_range_low, nl_range_high.
    """
    nl = newsletter.copy()
    nl['date'] = pd.to_datetime(nl['date'])

    # Raw CSV format: has symbol and timeframe columns
    if 'symbol' in nl.columns and 'timeframe' in nl.columns:
        nl = nl[(nl['symbol'] == 'ES') & (nl['timeframe'] == 'daily')].copy()
        nl = nl.rename(columns={'range_low': 'nl_range_low', 'range_high': 'nl_range_high'})
        return nl[['date', 'nl_range_low', 'nl_range_high']].dropna()

    # Pre-filtered format: ES_range_low / ES_range_high columns
    if 'ES_range_low' in nl.columns and 'ES_range_high' in nl.columns:
        nl = nl.rename(columns={
            'ES_range_low': 'nl_range_low',
            'ES_range_high': 'nl_range_high',
        })
        return nl[['date', 'nl_range_low', 'nl_range_high']].dropna()

    # Fallback: range_low / range_high columns
    if 'range_low' in nl.columns and 'range_high' in nl.columns:
        nl = nl.rename(columns={'range_low': 'nl_range_low', 'range_high': 'nl_range_high'})
        return nl[['date', 'nl_range_low', 'nl_range_high']].dropna()

    raise ValueError(
        "Cannot identify range columns in newsletter DataFrame. "
        "Expected: (symbol, timeframe, range_low, range_high) or "
        "(ES_range_low, ES_range_high)."
    )


def _align_newsletter_to_daily(
    newsletter: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Align newsletter predictions with realized daily OHLCV.

    Args:
        newsletter: DataFrame with columns [date, ES_range_low, ES_range_high, ...].
        daily: Daily OHLCV DataFrame (DatetimeIndex).

    Returns:
        Merged DataFrame with both predictions and realized values.
    """
    nl = newsletter.copy()
    nl['date'] = pd.to_datetime(nl['date'])

    realized = daily[['open', 'high', 'low', 'close']].copy()
    realized.index = pd.to_datetime(realized.index)
    realized = realized.reset_index()
    realized.columns = ['date', 'realized_open', 'realized_high',
                        'realized_low', 'realized_close']

    merged = nl.merge(realized, on='date', how='inner')
    return merged


def reverse_engineer_newsletter(
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Fit models predicting newsletter ranges to understand what drives them.

    Decomposes newsletter predictions into:
    - Autoregressive component: how much is explained by yesterday's newsletter range?
    - Market component: how much is explained by ATR, vol, momentum features?

    Uses walk-forward CV to avoid look-ahead bias.

    Args:
        daily: Daily OHLCV DataFrame (DatetimeIndex).
        newsletter: Newsletter predictions DataFrame (raw or pre-filtered).
        verbose: Print summary table.

    Returns:
        Dict with R2 decomposition and feature importances.
    """
    from xgboost import XGBRegressor

    # Step 1: Filter to ES daily and align with realized data
    nl = _filter_es_daily_newsletter(newsletter)
    if len(nl) == 0:
        print("WARNING: No ES daily newsletter rows found")
        return {}

    # Get realized daily data aligned by date
    realized = daily[['open', 'high', 'low', 'close']].copy()
    realized.index = pd.to_datetime(realized.index)
    realized = realized.reset_index().rename(columns={
        realized.index.name or 'index': 'date',
        'open': 'realized_open', 'high': 'realized_high',
        'low': 'realized_low', 'close': 'realized_close',
    })

    aligned = nl.merge(realized, on='date', how='inner')
    aligned = aligned.sort_values('date').set_index('date')

    if verbose:
        print(f"\n{'='*60}")
        print("NEWSLETTER REVERSE-ENGINEERING")
        print(f"{'='*60}")
        print(f"Aligned ES daily rows: {len(aligned)}")

    # Step 2: Compute prev_close and newsletter targets
    prev_close = daily['close'].shift(1).reindex(aligned.index)
    aligned['prev_close'] = prev_close
    aligned = aligned.dropna(subset=['prev_close', 'nl_range_low', 'nl_range_high'])

    aligned['nl_width'] = aligned['nl_range_high'] - aligned['nl_range_low']
    aligned['nl_high_pct'] = (aligned['nl_range_high'] - aligned['prev_close']) / aligned['prev_close']
    aligned['nl_low_pct'] = (aligned['prev_close'] - aligned['nl_range_low']) / aligned['prev_close']
    aligned['nl_width_pct'] = aligned['nl_width'] / aligned['prev_close']

    # Step 3: Add market features (reuse compute_range_features)
    market_features = compute_range_features(daily)
    market_features = market_features.reindex(aligned.index)

    # Step 4: Add autoregressive newsletter features
    for lag in range(1, 6):
        aligned[f'nl_width_pct_lag{lag}'] = aligned['nl_width_pct'].shift(lag)
    for lag in range(1, 4):
        aligned[f'nl_high_pct_lag{lag}'] = aligned['nl_high_pct'].shift(lag)
        aligned[f'nl_low_pct_lag{lag}'] = aligned['nl_low_pct'].shift(lag)

    # Join market features
    combined = aligned.join(market_features, how='left', rsuffix='_mkt')

    # Drop rows with NaN targets or missing lags (first few rows)
    target = 'nl_width_pct'
    combined = combined.dropna(subset=[target])

    ar_feat_names = [c for c in combined.columns if c.startswith('nl_') and 'lag' in c]
    market_feat_names = [c for c in market_features.columns if c in combined.columns]

    # Fill remaining NaN with 0 for features
    combined = combined.fillna(0.0)

    n = len(combined)
    if n < 80:
        print(f"WARNING: Only {n} samples after alignment. Results may be unreliable.")

    if verbose:
        print(f"Samples after alignment: {n}")
        print(f"AR features: {len(ar_feat_names)}, Market features: {len(market_feat_names)}")

    # Walk-forward CV helper
    def wf_r2(X_data: np.ndarray, y_data: np.ndarray, n_folds: int = 4) -> float:
        n_samples = len(X_data)
        min_train = max(60, n_samples // (n_folds + 1))
        test_per_fold = (n_samples - min_train) // n_folds
        if test_per_fold < 5:
            return float('nan')

        all_preds, all_actuals = [], []
        for fold in range(n_folds):
            train_end = min_train + fold * test_per_fold
            test_end = train_end + test_per_fold
            if fold == n_folds - 1:
                test_end = n_samples

            X_tr, y_tr = X_data[:train_end], y_data[:train_end]
            X_te, y_te = X_data[train_end:test_end], y_data[train_end:test_end]
            if len(X_te) == 0:
                continue

            m = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
            )
            m.fit(X_tr, y_tr)
            all_preds.extend(m.predict(X_te).tolist())
            all_actuals.extend(y_te.tolist())

        p, a = np.array(all_preds), np.array(all_actuals)
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    y = combined[target].values

    # Model A: AR only
    X_ar = combined[ar_feat_names].values if ar_feat_names else np.zeros((n, 1))
    r2_ar = wf_r2(X_ar, y) if ar_feat_names else float('nan')

    # Model B: Market only
    X_market = combined[market_feat_names].values if market_feat_names else np.zeros((n, 1))
    r2_market = wf_r2(X_market, y) if market_feat_names else float('nan')

    # Model C: Combined
    all_feat_names = ar_feat_names + [f for f in market_feat_names if f not in ar_feat_names]
    X_all = combined[all_feat_names].values
    r2_combined = wf_r2(X_all, y)

    # Train final model on all data for feature importance
    final_model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
    )
    final_model.fit(X_all, y)
    importances = pd.Series(
        final_model.feature_importances_, index=all_feat_names
    ).sort_values(ascending=False)

    # Classify each feature as AR or market
    ar_set = set(ar_feat_names)

    results = {
        'r2_ar': r2_ar,
        'r2_market': r2_market,
        'r2_combined': r2_combined,
        'n_samples': n,
        'feature_importances': importances.to_dict(),
        'ar_feat_names': ar_feat_names,
        'market_feat_names': market_feat_names,
    }

    if verbose:
        print(f"\nNewsletter Reverse-Engineering (ES Daily, {n} samples)")
        print(f"  Target: nl_width_pct (newsletter range width / prev_close)")
        print(f"")
        fmt_r2 = lambda r: f"{r:.4f}" if not np.isnan(r) else "  N/A "
        print(f"  Autoregressive only  (newsletter lags):  R2={fmt_r2(r2_ar)}")
        print(f"  Market features only (ATR, vol, mom):    R2={fmt_r2(r2_market)}")
        print(f"  Combined (AR + market):                  R2={fmt_r2(r2_combined)}")
        print(f"")
        print(f"  Top 15 features driving newsletter width:")
        total_imp = importances.sum()
        for i, (fname, imp) in enumerate(importances.head(15).items()):
            pct = imp / total_imp * 100 if total_imp > 0 else 0
            ftype = "(AR)" if fname in ar_set else "(market)"
            print(f"    {i+1:2d}. {fname:<35} {pct:5.1f}%  {ftype}")

        # Also check high/low pct targets
        print(f"\n  Checking nl_high_pct target:")
        y_high = combined['nl_high_pct'].values
        r2_h_ar = wf_r2(X_ar, y_high) if ar_feat_names else float('nan')
        r2_h_mkt = wf_r2(X_market, y_high) if market_feat_names else float('nan')
        r2_h_all = wf_r2(X_all, y_high)
        print(f"    AR={fmt_r2(r2_h_ar)}  Market={fmt_r2(r2_h_mkt)}  Combined={fmt_r2(r2_h_all)}")

        print(f"\n  Checking nl_low_pct target:")
        y_low = combined['nl_low_pct'].values
        r2_l_ar = wf_r2(X_ar, y_low) if ar_feat_names else float('nan')
        r2_l_mkt = wf_r2(X_market, y_low) if market_feat_names else float('nan')
        r2_l_all = wf_r2(X_all, y_low)
        print(f"    AR={fmt_r2(r2_l_ar)}  Market={fmt_r2(r2_l_mkt)}  Combined={fmt_r2(r2_l_all)}")

        print(f"\n{'='*60}")

    return results


def comprehensive_comparison(
    predictor: RangePredictor,
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Head-to-head comparison of model vs newsletter vs realized daily ranges.

    Extends run_analysis() with:
    - Extended accuracy metrics (MAE, width MAE, midpoint bias)
    - Regime-segmented analysis (volatility, trend, day-of-week)
    - Look-forward trade opportunity analysis (fade high/low)
    - Error decomposition

    Args:
        predictor: Trained RangePredictor with loaded models.
        daily: Daily OHLCV DataFrame.
        newsletter: Newsletter predictions (raw or pre-filtered).
        output_path: Save results CSV to this path (optional).
        verbose: Print comparison tables.

    Returns:
        Dict with all comparison metrics.
    """
    # ── Step 1: Align 3 datasets on overlapping dates ──────────────────
    nl = _filter_es_daily_newsletter(newsletter)
    nl = nl.set_index('date').sort_index()

    model_preds = predictor.predict_series(daily, timeframe='daily')

    # Build realized DataFrame
    realized = daily[['open', 'high', 'low', 'close']].copy()
    realized.index = pd.to_datetime(realized.index)

    # Find intersection of all three date sets
    common_dates = (
        nl.index
        .intersection(model_preds.index)
        .intersection(realized.index)
    )
    common_dates = common_dates.sort_values()

    if len(common_dates) == 0:
        print("WARNING: No overlapping dates across all three datasets")
        return {}

    nl_aligned = nl.loc[common_dates]
    model_aligned = model_preds.loc[common_dates]
    realized_aligned = realized.loc[common_dates]
    prev_close = daily['close'].shift(1).reindex(common_dates)

    # ── Step 2: ATR baseline ───────────────────────────────────────────
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)
    atr_aligned = atr_14.reindex(common_dates)

    if verbose:
        print(f"\n{'='*70}")
        print("MODEL vs NEWSLETTER vs REALIZED  (ES Daily)")
        print(f"{'='*70}")
        print(f"Overlapping dates: {len(common_dates)}  "
              f"({common_dates[0].date()} — {common_dates[-1].date()})")

    # ── Helper: extended metrics ───────────────────────────────────────
    def extended_metrics(
        pred_low: pd.Series, pred_high: pd.Series,
        real_low: pd.Series, real_high: pd.Series,
        prev_cl: pd.Series,
    ) -> dict:
        """Compute core accuracy + MAE metrics."""
        contain = compute_containment_rate(pred_low, pred_high, real_low, real_high)

        pred_mid = (pred_high + pred_low) / 2
        real_mid = (real_high + real_low) / 2
        pred_width = pred_high - pred_low
        real_width = real_high - real_low

        high_mae = np.mean(np.abs(pred_high - real_high))
        low_mae = np.mean(np.abs(pred_low - real_low))
        width_mae = np.mean(np.abs(pred_width - real_width))
        midpoint_bias = np.mean(pred_mid - real_mid)
        high_bias = np.mean(pred_high - real_high)
        low_bias = np.mean(pred_low - real_low)

        # Asymmetry: are we better at predicting highs or lows?
        asymmetry = high_mae - low_mae  # positive = worse at highs

        return {
            **contain,
            'high_mae': high_mae,
            'low_mae': low_mae,
            'width_mae': width_mae,
            'midpoint_bias': midpoint_bias,
            'high_bias': high_bias,
            'low_bias': low_bias,
            'asymmetry': asymmetry,
            'width_ratio': contain['avg_pred_width'] / contain['avg_realized_width']
            if contain['avg_realized_width'] > 0 else float('inf'),
        }

    # ── Compute overall metrics for all 3 sources ──────────────────────
    nl_pred_low = nl_aligned['nl_range_low']
    nl_pred_high = nl_aligned['nl_range_high']
    model_pred_low = model_aligned['pred_range_low']
    model_pred_high = model_aligned['pred_range_high']

    # ATR 0.7x baseline
    atr_low = prev_close - 0.7 * atr_aligned
    atr_high = prev_close + 0.7 * atr_aligned

    nl_metrics = extended_metrics(
        nl_pred_low, nl_pred_high, realized_aligned['low'], realized_aligned['high'], prev_close
    )
    model_metrics = extended_metrics(
        model_pred_low, model_pred_high, realized_aligned['low'], realized_aligned['high'], prev_close
    )
    atr_metrics = extended_metrics(
        atr_low, atr_high, realized_aligned['low'], realized_aligned['high'], prev_close
    )

    results = {
        'overall': {
            'model': model_metrics,
            'newsletter': nl_metrics,
            'atr_0.7x': atr_metrics,
        },
        'n_samples': len(common_dates),
    }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"{'OVERALL METRICS':<35} {'Model':>10} {'Newsletter':>12} {'ATR-0.7x':>10}")
        print(f"{'─'*70}")
        metrics_to_show = [
            ('Full containment', 'full_containment', '{:.1%}'),
            ('High contained', 'high_contained', '{:.1%}'),
            ('Low contained', 'low_contained', '{:.1%}'),
            ('High MAE (pts)', 'high_mae', '{:.2f}'),
            ('Low MAE (pts)', 'low_mae', '{:.2f}'),
            ('Width MAE (pts)', 'width_mae', '{:.2f}'),
            ('Width ratio', 'width_ratio', '{:.3f}'),
            ('Midpoint bias (pts)', 'midpoint_bias', '{:.2f}'),
            ('High bias (pts)', 'high_bias', '{:.2f}'),
            ('Low bias (pts)', 'low_bias', '{:.2f}'),
        ]
        for label, key, fmt in metrics_to_show:
            m_val = fmt.format(model_metrics[key])
            n_val = fmt.format(nl_metrics[key])
            a_val = fmt.format(atr_metrics[key])
            print(f"  {label:<33} {m_val:>10} {n_val:>12} {a_val:>10}")

    # ── Step 3: Regime analysis ────────────────────────────────────────
    regime_results = {}

    # --- By ATR volatility regime ---
    atr_pctile_20 = np.percentile(atr_aligned.dropna(), 20)
    atr_pctile_80 = np.percentile(atr_aligned.dropna(), 80)
    vol_regime = pd.Series('normal', index=common_dates)
    vol_regime[atr_aligned < atr_pctile_20] = 'low_vol'
    vol_regime[atr_aligned > atr_pctile_80] = 'high_vol'

    if verbose:
        print(f"\n{'─'*70}")
        print(f"{'VOLATILITY REGIME':<35} {'Model':>10} {'Newsletter':>12} {'ATR-0.7x':>10}")
        print(f"  ATR pctiles: 20th={atr_pctile_20:.1f}, 80th={atr_pctile_80:.1f} pts")
        print(f"{'─'*70}")

    vol_regime_metrics = {}
    for regime_name in ['low_vol', 'normal', 'high_vol']:
        mask = vol_regime == regime_name
        if mask.sum() < 10:
            continue
        idx = common_dates[mask]
        vm = extended_metrics(
            nl_pred_low[idx], nl_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        mm = extended_metrics(
            model_pred_low[idx], model_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        am = extended_metrics(
            atr_low[idx], atr_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        vol_regime_metrics[regime_name] = {'model': mm, 'newsletter': vm, 'atr_0.7x': am, 'n': mask.sum()}
        if verbose:
            print(f"  {regime_name:<15} (n={mask.sum():3d}): "
                  f"Model {mm['full_containment']:>6.1%}  "
                  f"Newsletter {vm['full_containment']:>6.1%}  "
                  f"ATR {am['full_containment']:>6.1%}")

    regime_results['volatility'] = vol_regime_metrics

    # --- By trend regime (20d return) ---
    ret_20d = daily['close'].pct_change(20).reindex(common_dates)
    trend_regime = pd.Series('sideways', index=common_dates)
    trend_regime[ret_20d > 0.01] = 'uptrend'
    trend_regime[ret_20d < -0.01] = 'downtrend'

    if verbose:
        print(f"\n{'─'*70}")
        print(f"TREND REGIME (20d return threshold: ±1%)")
        print(f"{'─'*70}")

    trend_regime_metrics = {}
    for regime_name in ['uptrend', 'sideways', 'downtrend']:
        mask = trend_regime == regime_name
        if mask.sum() < 10:
            continue
        idx = common_dates[mask]
        vm = extended_metrics(
            nl_pred_low[idx], nl_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        mm = extended_metrics(
            model_pred_low[idx], model_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        am = extended_metrics(
            atr_low[idx], atr_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        trend_regime_metrics[regime_name] = {'model': mm, 'newsletter': vm, 'atr_0.7x': am, 'n': mask.sum()}
        if verbose:
            print(f"  {regime_name:<15} (n={mask.sum():3d}): "
                  f"Model {mm['full_containment']:>6.1%}  "
                  f"Newsletter {vm['full_containment']:>6.1%}  "
                  f"ATR {am['full_containment']:>6.1%}")

    regime_results['trend'] = trend_regime_metrics

    # --- By day of week ---
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    dow = pd.Series(common_dates.dayofweek, index=common_dates)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"DAY-OF-WEEK PATTERN")
        print(f"{'─'*70}")

    dow_metrics = {}
    for dow_val, dow_name in sorted(dow_names.items()):
        mask = dow == dow_val
        if mask.sum() < 5:
            continue
        idx = common_dates[mask]
        vm = extended_metrics(
            nl_pred_low[idx], nl_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        mm = extended_metrics(
            model_pred_low[idx], model_pred_high[idx], realized_aligned['low'][idx], realized_aligned['high'][idx], prev_close[idx]
        )
        dow_metrics[dow_name] = {'model': mm, 'newsletter': vm, 'n': mask.sum()}
        if verbose:
            print(f"  {dow_name:<10} (n={mask.sum():3d}): "
                  f"Model {mm['full_containment']:>6.1%}  "
                  f"Newsletter {vm['full_containment']:>6.1%}  "
                  f"Model_width={mm['avg_pred_width']:.1f}  "
                  f"Realized_width={mm['avg_realized_width']:.1f}")

    regime_results['dow'] = dow_metrics

    results['regime'] = regime_results

    # ── Step 4: Look-forward trade opportunity analysis ────────────────
    def trade_opportunity_metrics(
        pred_low_s: pd.Series, pred_high_s: pd.Series,
        real_low_s: pd.Series, real_high_s: pd.Series,
        real_close_s: pd.Series,
    ) -> dict:
        """Compute fade high/low opportunity metrics."""
        n = len(pred_low_s)

        # Fade high
        touched_high = real_high_s >= pred_high_s
        faded_high_win = touched_high & (real_close_s < pred_high_s)
        faded_high_loss = touched_high & (real_close_s >= pred_high_s)

        # Fade low
        touched_low = real_low_s <= pred_low_s
        faded_low_win = touched_low & (real_close_s > pred_low_s)
        faded_low_loss = touched_low & (real_close_s <= pred_low_s)

        n_touched_high = touched_high.sum()
        n_touched_low = touched_low.sum()

        fade_high_wr = faded_high_win.sum() / n_touched_high if n_touched_high > 0 else float('nan')
        fade_low_wr = faded_low_win.sum() / n_touched_low if n_touched_low > 0 else float('nan')

        return {
            'pct_touch_high': n_touched_high / n,
            'fade_high_win_rate': fade_high_wr,
            'n_touched_high': int(n_touched_high),
            'pct_touch_low': n_touched_low / n,
            'fade_low_win_rate': fade_low_wr,
            'n_touched_low': int(n_touched_low),
        }

    nl_trade = trade_opportunity_metrics(
        nl_pred_low, nl_pred_high,
        realized_aligned['low'], realized_aligned['high'], realized_aligned['close']
    )
    model_trade = trade_opportunity_metrics(
        model_pred_low, model_pred_high,
        realized_aligned['low'], realized_aligned['high'], realized_aligned['close']
    )
    atr_trade = trade_opportunity_metrics(
        atr_low, atr_high,
        realized_aligned['low'], realized_aligned['high'], realized_aligned['close']
    )

    results['trade_opportunities'] = {
        'model': model_trade,
        'newsletter': nl_trade,
        'atr_0.7x': atr_trade,
    }

    if verbose:
        print(f"\n{'─'*70}")
        print(f"TRADE OPPORTUNITY ANALYSIS  (daily OHLC approximation)")
        print(f"  'Fade high': short at pred_high when price touches it, win if close < pred_high")
        print(f"  'Fade low':  long at pred_low when price touches it, win if close > pred_low")
        print(f"{'─'*70}")
        print(f"{'Metric':<35} {'Model':>10} {'Newsletter':>12} {'ATR-0.7x':>10}")
        print(f"{'─'*70}")

        def fmt_pct(v):
            return f"{v:.1%}" if not np.isnan(v) else "  N/A  "

        trade_rows = [
            ('% days touch predicted high', 'pct_touch_high'),
            ('Fade-high win rate', 'fade_high_win_rate'),
            ('% days touch predicted low', 'pct_touch_low'),
            ('Fade-low win rate', 'fade_low_win_rate'),
        ]
        for label, key in trade_rows:
            m_val = fmt_pct(model_trade[key])
            n_val = fmt_pct(nl_trade[key])
            a_val = fmt_pct(atr_trade[key])
            print(f"  {label:<33} {m_val:>10} {n_val:>12} {a_val:>10}")

        print(f"\n  (n_touched_high: model={model_trade['n_touched_high']}, "
              f"nl={nl_trade['n_touched_high']}, atr={atr_trade['n_touched_high']})")
        print(f"  (n_touched_low:  model={model_trade['n_touched_low']}, "
              f"nl={nl_trade['n_touched_low']}, atr={atr_trade['n_touched_low']})")

    # ── Step 5: Error decomposition ────────────────────────────────────
    error_decomp = {}
    for name, pred_l, pred_h in [
        ('model', model_pred_low, model_pred_high),
        ('newsletter', nl_pred_low, nl_pred_high),
    ]:
        real_l = realized_aligned['low']
        real_h = realized_aligned['high']
        pred_w = pred_h - pred_l
        real_w = real_h - real_l

        error_decomp[name] = {
            'high_bias': float(np.mean(pred_h - real_h)),
            'low_bias': float(np.mean(pred_l - real_l)),
            'width_bias': float(np.mean(pred_w - real_w)),
            'high_std': float(np.std(pred_h - real_h)),
            'low_std': float(np.std(pred_l - real_l)),
            'high_corr': float(np.corrcoef(pred_h, real_h)[0, 1]),
            'low_corr': float(np.corrcoef(pred_l, real_l)[0, 1]),
        }

    results['error_decomp'] = error_decomp

    if verbose:
        print(f"\n{'─'*70}")
        print(f"ERROR DECOMPOSITION")
        print(f"{'─'*70}")
        print(f"  {'Metric':<30} {'Model':>12} {'Newsletter':>14}")
        decomp_rows = [
            ('High bias (pts)', 'high_bias', '{:+.2f}'),
            ('Low bias (pts)', 'low_bias', '{:+.2f}'),
            ('Width bias (pts)', 'width_bias', '{:+.2f}'),
            ('High error std (pts)', 'high_std', '{:.2f}'),
            ('Low error std (pts)', 'low_std', '{:.2f}'),
            ('High corr w/ realized', 'high_corr', '{:.4f}'),
            ('Low corr w/ realized', 'low_corr', '{:.4f}'),
        ]
        for label, key, fmt in decomp_rows:
            m_val = fmt.format(error_decomp['model'][key])
            n_val = fmt.format(error_decomp['newsletter'][key])
            print(f"  {label:<30} {m_val:>12} {n_val:>14}")

        print(f"\n{'='*70}")

    # ── Save results CSV ───────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        rows = []
        for date in common_dates:
            rows.append({
                'date': date,
                'realized_high': realized_aligned['high'][date],
                'realized_low': realized_aligned['low'][date],
                'realized_close': realized_aligned['close'][date],
                'nl_high': nl_pred_high[date],
                'nl_low': nl_pred_low[date],
                'model_high': model_pred_high[date],
                'model_low': model_pred_low[date],
                'atr_14': atr_aligned[date],
                'vol_regime': vol_regime[date],
                'trend_regime': trend_regime[date],
            })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_path, index=False)
        if verbose:
            print(f"Results saved to {output_path}")

    return results


def run_analysis(
    predictor: RangePredictor,
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Compare model predictions against newsletter.

    Args:
        predictor: Trained RangePredictor with loaded models.
        daily: Daily OHLCV DataFrame.
        newsletter: Newsletter predictions CSV (date, ES_range_low, ES_range_high, ...).
        verbose: Print results.

    Returns:
        Dict with comparison metrics.
    """
    # Get model predictions for all days
    model_preds = predictor.predict_series(daily, timeframe='daily')

    # Align newsletter with realized data
    aligned = _align_newsletter_to_daily(newsletter, daily)
    if len(aligned) == 0:
        print("WARNING: No overlapping dates between newsletter and daily data")
        return {}

    if verbose:
        print(f"\n{'='*60}")
        print("RANGE PREDICTION ANALYSIS")
        print(f"{'='*60}")
        print(f"Newsletter predictions: {len(newsletter)}")
        print(f"Daily bars: {len(daily)}")
        print(f"Overlapping days: {len(aligned)}")

    results = {}

    # --- Newsletter metrics ---
    if 'ES_range_low' in aligned.columns and 'ES_range_high' in aligned.columns:
        nl_metrics = compute_containment_rate(
            aligned['ES_range_low'],
            aligned['ES_range_high'],
            aligned['realized_low'],
            aligned['realized_high'],
        )
        results['newsletter'] = nl_metrics

        if verbose:
            print(f"\n--- Newsletter (ES) ---")
            print(f"  Full containment: {nl_metrics['full_containment']:.1%}")
            print(f"  High contained:   {nl_metrics['high_contained']:.1%}")
            print(f"  Low contained:    {nl_metrics['low_contained']:.1%}")
            print(f"  Avg width: {nl_metrics['avg_pred_width']:.1f} pts "
                  f"(realized: {nl_metrics['avg_realized_width']:.1f} pts)")

    # --- Model metrics ---
    # Align model predictions with the same date range
    model_aligned = model_preds.loc[model_preds.index.isin(aligned['date'])]

    if len(model_aligned) > 0 and 'pred_range_high' in model_aligned.columns:
        # Get realized values for matching dates
        realized_for_model = daily.loc[model_aligned.index]

        model_metrics = compute_containment_rate(
            model_aligned['pred_range_low'],
            model_aligned['pred_range_high'],
            realized_for_model['low'],
            realized_for_model['high'],
        )
        results['model'] = model_metrics

        if verbose:
            print(f"\n--- Model (ES) ---")
            print(f"  Full containment: {model_metrics['full_containment']:.1%}")
            print(f"  High contained:   {model_metrics['high_contained']:.1%}")
            print(f"  Low contained:    {model_metrics['low_contained']:.1%}")
            print(f"  Avg width: {model_metrics['avg_pred_width']:.1f} pts "
                  f"(realized: {model_metrics['avg_realized_width']:.1f} pts)")

    # --- Width accuracy (MAE on range boundaries) ---
    if 'ES_range_low' in aligned.columns:
        nl_high_mae = np.mean(np.abs(
            aligned['ES_range_high'] - aligned['realized_high']
        ))
        nl_low_mae = np.mean(np.abs(
            aligned['ES_range_low'] - aligned['realized_low']
        ))

        results['newsletter_mae'] = {
            'high_mae': nl_high_mae,
            'low_mae': nl_low_mae,
        }

        if verbose:
            print(f"\n--- Boundary MAE ---")
            print(f"  Newsletter: high={nl_high_mae:.1f} pts, low={nl_low_mae:.1f} pts")

    if len(model_aligned) > 0 and 'pred_range_high' in model_aligned.columns:
        model_high_mae = np.mean(np.abs(
            model_aligned['pred_range_high'].values - realized_for_model['high'].values
        ))
        model_low_mae = np.mean(np.abs(
            model_aligned['pred_range_low'].values - realized_for_model['low'].values
        ))

        results['model_mae'] = {
            'high_mae': model_high_mae,
            'low_mae': model_low_mae,
        }

        if verbose:
            print(f"  Model:      high={model_high_mae:.1f} pts, low={model_low_mae:.1f} pts")

    # --- ATR-based baseline ---
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)

    atr_aligned = atr_14.loc[atr_14.index.isin(aligned['date'])]
    if len(atr_aligned) > 0:
        prev_close = daily['close'].shift(1).loc[atr_aligned.index]
        atr_vals = atr_aligned.values

        # Newsletter approximation: close +/- 0.7 * ATR
        for mult in [0.5, 0.7, 1.0]:
            atr_low = prev_close - mult * atr_vals
            atr_high = prev_close + mult * atr_vals
            realized_sub = daily.loc[atr_aligned.index]

            atr_metrics = compute_containment_rate(
                atr_low, atr_high,
                realized_sub['low'], realized_sub['high'],
            )
            results[f'atr_{mult}x'] = atr_metrics

            if verbose:
                print(f"\n  ATR {mult}x baseline: "
                      f"containment={atr_metrics['full_containment']:.1%}, "
                      f"width={atr_metrics['avg_pred_width']:.1f}")

    if verbose:
        print(f"\n{'='*60}")

    return results


def compare_all_timeframes(
    predictor: RangePredictor,
    daily: pd.DataFrame,
    newsletter: pd.DataFrame,
    oos_mode: bool = True,
    n_folds: int = 5,
    min_train_days: int = 100,
    verbose: bool = True,
) -> dict:
    """Compare model vs newsletter vs realized for each timeframe separately.

    For each timeframe (daily/weekly/monthly/quarterly):
    - Realized range = actual max high / min low over the forward horizon
    - Model = walk-forward OOS predictions (oos_mode=True, default) or
              final model in-sample predictions (oos_mode=False)
    - Newsletter = newsletter's range prediction for that timeframe
    - ATR baseline = prev_close +/- 0.7 * ATR * sqrt(horizon) (horizon-scaled)

    Args:
        predictor: Trained RangePredictor with loaded models.
        daily: Daily OHLCV DataFrame.
        newsletter: Newsletter predictions (raw multi-symbol CSV).
        oos_mode: Use walk-forward OOS predictions instead of final model.
        n_folds: Walk-forward folds (only used when oos_mode=True).
        min_train_days: Minimum training set size per fold.
        verbose: Print summary tables.

    Returns:
        Dict keyed by timeframe with comparison metrics.
    """
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14)
    prev_close_series = daily['close'].shift(1)

    # Normalise newsletter date column once
    nl_raw = newsletter.copy()
    nl_raw['date'] = pd.to_datetime(nl_raw['date'])

    all_results = {}

    for timeframe, horizon in TIMEFRAME_HORIZONS.items():
        if timeframe not in predictor.models:
            continue

        # ── Newsletter for this timeframe ──────────────────────────────
        if 'symbol' in nl_raw.columns and 'timeframe' in nl_raw.columns:
            nl_tf = nl_raw[
                (nl_raw['symbol'] == 'ES') & (nl_raw['timeframe'] == timeframe)
            ].copy()
            nl_tf = nl_tf.rename(columns={
                'range_low': 'nl_range_low', 'range_high': 'nl_range_high'
            })
        else:
            nl_tf = _filter_es_daily_newsletter(nl_raw)

        if len(nl_tf) == 0:
            if verbose:
                print(f"\n  [{timeframe}] No newsletter data — skipping.")
            continue

        nl_tf = nl_tf.set_index('date')[['nl_range_low', 'nl_range_high']].sort_index()
        nl_tf = nl_tf.dropna()

        # ── Model predictions (OOS or in-sample) ──────────────────────
        if oos_mode:
            if verbose:
                print(f"  [{timeframe}] Generating OOS predictions (walk-forward)...")
            model_preds = _generate_oos_predictions(
                daily, timeframe=timeframe,
                n_folds=n_folds, min_train_days=min_train_days,
            )
        else:
            model_preds = predictor.predict_series(daily, timeframe=timeframe)

        # ── Realized forward range over horizon ────────────────────────
        targets = compute_targets(daily, horizon)
        realized_high = prev_close_series * (1 + targets['range_high_pct'])
        realized_low  = prev_close_series * (1 - targets['range_low_pct'])

        # ── Align on dates common to all three ────────────────────────
        common_dates = (
            nl_tf.index
            .intersection(model_preds.dropna(subset=['pred_range_high', 'pred_range_low']).index)
            .intersection(realized_high.dropna().index)
        )
        common_dates = common_dates.sort_values()

        if len(common_dates) < 10:
            if verbose:
                print(f"\n  [{timeframe}] Only {len(common_dates)} overlapping dates — skipping.")
            continue

        nl_hi  = nl_tf.loc[common_dates, 'nl_range_high']
        nl_lo  = nl_tf.loc[common_dates, 'nl_range_low']
        mdl_hi = model_preds.loc[common_dates, 'pred_range_high']
        mdl_lo = model_preds.loc[common_dates, 'pred_range_low']
        rlz_hi = realized_high.loc[common_dates]
        rlz_lo = realized_low.loc[common_dates]
        pc     = prev_close_series.loc[common_dates]

        # ATR baseline scaled by sqrt(horizon)
        scale  = 0.7 * np.sqrt(horizon)
        atr_al = atr_14.loc[common_dates]
        atr_hi = pc + scale * atr_al
        atr_lo = pc - scale * atr_al

        # ── Compute metrics ────────────────────────────────────────────
        def _m(pred_lo, pred_hi, real_lo, real_hi):
            cm = compute_containment_rate(pred_lo, pred_hi, real_lo, real_hi)
            pred_width = pred_hi - pred_lo
            real_width = real_hi - real_lo
            pred_mid   = (pred_hi + pred_lo) / 2
            real_mid   = (real_hi + real_lo) / 2

            # OOS R² on the width
            pred_w_arr = pred_width.values if hasattr(pred_width, 'values') else np.array(pred_width)
            real_w_arr = real_width.values if hasattr(real_width, 'values') else np.array(real_width)
            ss_res = np.sum((real_w_arr - pred_w_arr) ** 2)
            ss_tot = np.sum((real_w_arr - np.mean(real_w_arr)) ** 2)
            width_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

            return {
                **cm,
                'high_mae':   float(np.mean(np.abs(pred_hi - real_hi))),
                'low_mae':    float(np.mean(np.abs(pred_lo - real_lo))),
                'width_mae':  float(np.mean(np.abs(pred_width - real_width))),
                'high_bias':  float(np.mean(pred_hi - real_hi)),
                'low_bias':   float(np.mean(pred_lo - real_lo)),
                'mid_bias':   float(np.mean(pred_mid - real_mid)),
                'width_r2':   width_r2,
            }

        nl_m  = _m(nl_lo,  nl_hi,  rlz_lo, rlz_hi)
        mdl_m = _m(mdl_lo, mdl_hi, rlz_lo, rlz_hi)
        atr_m = _m(atr_lo, atr_hi, rlz_lo, rlz_hi)

        all_results[timeframe] = {
            'model':      mdl_m,
            'newsletter': nl_m,
            'atr_scaled': atr_m,
            'n_samples':  len(common_dates),
            'horizon':    horizon,
            'date_range': (common_dates[0].date(), common_dates[-1].date()),
            'oos_mode':   oos_mode,
        }

    if verbose:
        _print_timeframe_comparison(all_results)

    return all_results


def _print_timeframe_comparison(all_results: dict) -> None:
    """Print a compact side-by-side comparison table for all timeframes."""
    # Detect if OOS mode was used (consistent across all timeframes)
    oos_mode = next(iter(all_results.values())).get('oos_mode', False)
    mode_label = "OOS walk-forward" if oos_mode else "in-sample (final model)"

    print(f"\n{'='*80}")
    print(f"MODEL vs NEWSLETTER vs REALIZED — ALL TIMEFRAMES (ES)")
    print(f"  Model predictions: {mode_label}")
    print(f"{'='*80}")

    metrics_rows = [
        ('Full containment',    'full_containment',   '{:.1%}'),
        ('High contained',      'high_contained',     '{:.1%}'),
        ('Low contained',       'low_contained',      '{:.1%}'),
        ('Width R²',            'width_r2',           '{:.3f}'),
        ('Width ratio',         'width_ratio',        '{:.3f}'),
        ('High MAE (pts)',       'high_mae',           '{:.1f}'),
        ('Low MAE (pts)',        'low_mae',            '{:.1f}'),
        ('Width MAE (pts)',      'width_mae',          '{:.1f}'),
        ('High bias (pts)',      'high_bias',          '{:+.1f}'),
        ('Low bias (pts)',       'low_bias',           '{:+.1f}'),
        ('Mid bias (pts)',       'mid_bias',           '{:+.1f}'),
        ('Avg pred width',       'avg_pred_width',     '{:.1f}'),
        ('Avg realized width',   'avg_realized_width', '{:.1f}'),
    ]

    for timeframe, res in all_results.items():
        horizon  = res['horizon']
        n        = res['n_samples']
        dr       = res['date_range']
        mdl_m    = res['model']
        nl_m     = res['newsletter']
        atr_m    = res['atr_scaled']

        print(f"\n{'─'*80}")
        print(f"  {timeframe.upper():<12}  horizon={horizon}d   n={n}   "
              f"({dr[0]} — {dr[1]})")
        print(f"{'─'*80}")
        print(f"  {'Metric':<28} {'Model':>10} {'Newsletter':>12} {'ATR×√h':>10}")
        print(f"  {'─'*62}")

        for label, key, fmt in metrics_rows:
            try:
                m_val = fmt.format(mdl_m[key])
                n_val = fmt.format(nl_m.get(key, float('nan')))
                a_val = fmt.format(atr_m.get(key, float('nan')))
            except (KeyError, ValueError):
                m_val = n_val = a_val = '  N/A '
            # Highlight width_r2 row since it's the key calibration signal
            marker = ' *' if key == 'width_r2' else '  '
            print(f"{marker} {label:<28} {m_val:>10} {n_val:>12} {a_val:>10}")

    print(f"\n{'='*80}")
    print("  * Width R²: how much of day-to-day width variance the model captures")
    print("    A mean predictor would have width_ratio≈1.00 but width R²≈0.00")
    print("  ATR×√h baseline: prev_close ± 0.7 × ATR(14) × sqrt(horizon)")
    print(f"{'='*80}")
