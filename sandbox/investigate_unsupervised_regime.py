"""Unsupervised regime detection: range anomaly + online change detection.

Three complementary signals combined into a composite risk score:
  1. Multi-timeframe range breakouts (structural price position)
  2. Feature-space anomaly detection (Isolation Forest)
  3. Online change detection (CUSUM on returns, EWMA on anomaly scores)

Parts:
  0. Data pipeline (load, aggregate, compute features)
  1. Anomaly detection analysis (walk-forward IF, temporal alignment)
  2. Change detection analysis (CUSUM + EWMA calibration)
  3. Composite risk score (weight sweep, regime analysis)
  4. Strategy backtest (risk sizing, adaptive DD exit, combined)
  5. Supervised baseline comparison (3-class XGB walk-forward)
  6. Summary + figures

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_unsupervised_regime.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_unsupervised_regime.py --quick --folds 2
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_unsupervised_regime.py --weight-range 0.3 --weight-anomaly 0.4 --weight-change 0.3
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategies.swing.config import SwingConfig, INSTRUMENTS
from strategies.swing.data_loader import load_instruments
from strategies.swing.daily_aggregator import DailyAggregator, align_daily
from strategies.swing.features.daily_technical import (
    compute_daily_technical, FEATURE_NAMES as TECH_FEATURES,
)
from strategies.swing.features.volume_profile_daily import (
    compute_vp_daily_features, FEATURE_NAMES as VP_FEATURES,
)
from strategies.swing.features.cross_instrument import (
    compute_cross_features, get_feature_names as get_cross_names,
)
from strategies.swing.features.macro_context import (
    compute_macro_context, FEATURE_NAMES as MACRO_FEATURES,
)
from strategies.swing.features.external_daily import compute_external_features
from strategies.swing.features.range_features import (
    compute_range_features, FEATURE_NAMES as RANGE_FEATURES,
)
from strategies.swing.labeling.structural_regime import compute_labels
from strategies.swing.labeling.hmm_regime import (
    compute_hmm_features_walkforward, FEATURE_NAMES as HMM_FEATURES,
)
from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector
from strategies.swing.detection.change_detector import compute_change_features
from strategies.swing.detection.regime_risk import compute_regime_risk_score
from strategies.swing.training.regime_trainer import walk_forward_cv, TrainerResult


FIGURES_DIR = Path("sandbox/figures/unsupervised_regime")
CLASS_NAMES = {0: "BEAR", 1: "BALANCE", 2: "BULL"}


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised regime detection")
    parser.add_argument("--quick", action="store_true", help="Quick debug run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--es-only", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--weight-range", type=float, default=0.25)
    parser.add_argument("--weight-anomaly", type=float, default=0.40)
    parser.add_argument("--weight-change", type=float, default=0.35)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Part 0: Data Pipeline
# ─────────────────────────────────────────────────────────────────────

def load_data_pipeline(args) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load data, compute all features (base + range), compute labels.

    Returns (df, es_daily, feature_cols) where df has features + labels.
    """
    config = SwingConfig(
        n_folds=args.folds if not args.quick else 2,
        min_train_days=args.min_train if not args.quick else 200,
    )

    print("=" * 60)
    print("PART 0: Loading data & computing features")
    print("=" * 60)

    symbols = ["ES"]
    if not args.es_only:
        symbols += config.correlation_instruments
    minute_data = load_instruments(symbols)
    if "ES" not in minute_data:
        print("ERROR: ES data not found")
        sys.exit(1)

    aggregator = DailyAggregator()
    daily_data = {}
    for sym, minute_df in minute_data.items():
        compute_vp = (sym == "ES")
        print(f"  Aggregating {sym}...")
        daily_data[sym] = aggregator.aggregate(minute_df, compute_vp=compute_vp)
        print(f"    {len(daily_data[sym])} trading days")

    if len(daily_data) > 1:
        daily_data = align_daily(daily_data, primary="ES")

    es_daily = daily_data["ES"]
    print(f"  ES: {len(es_daily)} days, {es_daily.index.min().date()} -- {es_daily.index.max().date()}")

    # Base features (~145)
    print("  Computing features...")
    tech_feats = compute_daily_technical(es_daily)
    feature_cols = list(TECH_FEATURES)

    vp_feats = pd.DataFrame(index=es_daily.index)
    if "vp_poc_rel" in es_daily.columns:
        vp_feats = compute_vp_daily_features(es_daily)
        feature_cols += VP_FEATURES

    cross_feats = pd.DataFrame(index=es_daily.index)
    if not args.es_only and len(daily_data) > 1:
        other_dailys = [(sym, df) for sym, df in daily_data.items() if sym != "ES"]
        cross_feats = compute_cross_features(es_daily, other_dailys)
        cross_names = get_cross_names([sym for sym, _ in other_dailys])
        cross_names = [c for c in cross_names if c in cross_feats.columns]
        feature_cols += cross_names

    other_for_macro = [(sym, df) for sym, df in daily_data.items() if sym != "ES"] if len(daily_data) > 1 else None
    macro_feats = compute_macro_context(es_daily, other_for_macro)
    feature_cols += MACRO_FEATURES

    ext_feats, ext_names = compute_external_features(es_daily)
    feature_cols += ext_names

    # Range features (16 new)
    print("  Computing range features (16)...")
    range_feats = compute_range_features(es_daily)
    feature_cols += RANGE_FEATURES

    all_feats = pd.concat([tech_feats, vp_feats, cross_feats, macro_feats, ext_feats, range_feats], axis=1)
    all_feats = all_feats.reindex(es_daily.index).fillna(0)

    # Labels (structural, for evaluation only)
    labels = compute_labels(es_daily)
    df = all_feats.join(labels)

    print(f"  Total features: {len(feature_cols)} ({len(feature_cols) - len(RANGE_FEATURES)} base + {len(RANGE_FEATURES)} range)")

    struct = df["y_structural"]
    valid = struct[struct.isin([0, 1, 2])]
    dist = valid.value_counts(normalize=True).sort_index()
    for k, v in dist.items():
        print(f"    {CLASS_NAMES[k]}: {v:.1%} ({valid.value_counts()[k]} days)")

    return df, es_daily, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Shared: Strategy metrics
# ─────────────────────────────────────────────────────────────────────

def compute_strategy_metrics(strat_df: pd.DataFrame, label: str) -> dict:
    """Compute Sharpe, MaxDD, Calmar, time-in-market, etc."""
    ret = strat_df["strategy_return"]
    pos = strat_df["position"]
    n_days = len(strat_df)
    years = n_days / 252

    # Time in market
    in_market_pct = (pos > 0).mean() * 100

    # Returns
    total_return = (1 + ret).prod() - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    vol = ret.std() * np.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else 0

    # Max drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Year-by-year
    yearly = {}
    strat_copy = strat_df.copy()
    strat_copy["year"] = strat_copy.index.year
    for yr, grp in strat_copy.groupby("year"):
        yr_ret = (1 + grp["strategy_return"]).prod() - 1
        yearly[yr] = yr_ret

    return {
        "label": label,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "time_in_market_pct": in_market_pct,
        "yearly": yearly,
    }


def make_strategy_df(daily_ret: np.ndarray, position: np.ndarray, index) -> pd.DataFrame:
    """Build strategy DataFrame from position array."""
    strat_df = pd.DataFrame(index=index)
    strat_df["position"] = position
    # Causal: position at day t earns return at day t+1
    strat_df["strategy_return"] = pd.Series(position, index=index).shift(1).fillna(1.0) * daily_ret
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


# ─────────────────────────────────────────────────────────────────────
# Part 1: Anomaly Detection Analysis
# ─────────────────────────────────────────────────────────────────────

def run_anomaly_analysis(
    df: pd.DataFrame,
    es_daily: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_train_days: int,
) -> pd.DataFrame:
    """Walk-forward Isolation Forest analysis.

    Returns DataFrame with anomaly scores for the OOS period.
    """
    print(f"\n{'='*60}")
    print("PART 1: Anomaly Detection Analysis")
    print("=" * 60)

    days = sorted(df.index.unique())
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds

    feature_matrix = df[feature_cols].fillna(0)

    # Walk-forward: fit per fold, collect OOS scores
    all_scores_list = []

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_days_per_fold
        test_end = train_end + test_days_per_fold if fold < n_folds - 1 else n_days

        train_days_list = days[:train_end]
        test_days_list = days[train_end:test_end]

        detector = RollingAnomalyDetector(n_estimators=200)
        result = detector.fit_score(feature_matrix, train_end)

        # Take only OOS scores for this fold
        fold_scores = result.loc[result.index.isin(test_days_list), "anomaly_score"]
        all_scores_list.append(fold_scores)

        print(f"  Fold {fold+1}: train={len(train_days_list)}d, test={len(test_days_list)}d, "
              f"mean_score={fold_scores.mean():.3f}, max={fold_scores.max():.3f}")

    all_scores = pd.concat(all_scores_list)

    # Build full anomaly DataFrame for OOS period
    oos_days = sorted(all_scores.index.unique())
    anomaly_df = pd.DataFrame(index=oos_days)
    anomaly_df["anomaly_score"] = all_scores.reindex(oos_days)

    # Also get labels for the OOS period
    struct = df["y_structural"].reindex(oos_days)
    anomaly_df["actual_regime"] = struct

    # Temporal alignment: anomaly score during each regime
    print(f"\n  Anomaly score by actual regime:")
    for cls in [0, 1, 2]:
        mask = anomaly_df["actual_regime"] == cls
        if mask.sum() > 0:
            score = anomaly_df.loc[mask, "anomaly_score"]
            print(f"    {CLASS_NAMES[cls]}: mean={score.mean():.3f}, "
                  f"median={score.median():.3f}, p75={score.quantile(0.75):.3f}, "
                  f"p90={score.quantile(0.90):.3f}")

    # Lead time analysis: find >5% drawdown events
    close = es_daily["close"].reindex(oos_days)
    cum_max = close.cummax()
    drawdown = (close - cum_max) / cum_max

    # Find drawdown events >5%
    in_dd = drawdown < -0.05
    dd_starts = in_dd & ~in_dd.shift(1, fill_value=False)
    dd_start_dates = dd_starts[dd_starts].index

    if len(dd_start_dates) > 0:
        print(f"\n  Lead time analysis ({len(dd_start_dates)} drawdown events >5%):")
        lead_times = []
        for dd_date in dd_start_dates:
            # Look back 30 days for anomaly score > 0.5
            lookback_start = dd_date - pd.Timedelta(days=45)  # calendar days
            window = anomaly_df.loc[
                (anomaly_df.index >= lookback_start) & (anomaly_df.index < dd_date),
                "anomaly_score"
            ]
            elevated = window[window > 0.5]
            if len(elevated) > 0:
                first_elevated = elevated.index[0]
                lead_days = len(anomaly_df.loc[
                    (anomaly_df.index >= first_elevated) & (anomaly_df.index < dd_date)
                ])
                lead_times.append(lead_days)
                print(f"    DD {dd_date.date()}: score elevated {lead_days}d before")
            else:
                lead_times.append(0)
                print(f"    DD {dd_date.date()}: no elevated score in prior 30d")

        if lead_times:
            arr = np.array(lead_times)
            print(f"  Mean lead time: {arr.mean():.1f}d (median={np.median(arr):.0f}d)")
            print(f"  Detected (lead>0): {(arr > 0).sum()}/{len(arr)} = {(arr > 0).mean():.0%}")
    else:
        print("  No >5% drawdown events in OOS period")

    # False positive analysis
    high_score = anomaly_df["anomaly_score"] > 0.5
    n_high = high_score.sum()
    if n_high > 0:
        # Check if a >5% drawdown follows within 20 trading days
        future_dd = pd.Series(False, index=oos_days)
        for dd_date in dd_start_dates:
            window_start = dd_date - pd.Timedelta(days=30)
            future_dd.loc[(future_dd.index >= window_start) & (future_dd.index <= dd_date)] = True

        true_alarms = (high_score & future_dd).sum()
        false_alarms = (high_score & ~future_dd).sum()
        print(f"\n  False alarm analysis (score > 0.5):")
        print(f"    High-score days: {n_high} ({n_high/len(oos_days):.1%} of OOS)")
        print(f"    True alarms (within 20d of DD): {true_alarms}")
        print(f"    False alarms: {false_alarms} ({false_alarms/n_high:.0%})")

    return anomaly_df


# ─────────────────────────────────────────────────────────────────────
# Part 2: Change Detection Analysis
# ─────────────────────────────────────────────────────────────────────

def run_change_analysis(
    es_daily: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> pd.DataFrame:
    """CUSUM + EWMA change detection analysis.

    Returns DataFrame with change features aligned to OOS period.
    """
    print(f"\n{'='*60}")
    print("PART 2: Change Detection Analysis")
    print("=" * 60)

    oos_days = anomaly_df.index
    daily_ret = es_daily["close"].pct_change().reindex(oos_days).fillna(0)
    anomaly_scores = anomaly_df["anomaly_score"].fillna(0)

    change_df = compute_change_features(daily_ret, anomaly_scores)

    # Analyze by regime
    actual = anomaly_df["actual_regime"]
    print(f"\n  Change features by actual regime:")
    for feat in ["return_cusum_score", "anomaly_ewma", "combined_change_score"]:
        print(f"\n  {feat}:")
        for cls in [0, 1, 2]:
            mask = actual == cls
            if mask.sum() > 0:
                vals = change_df.loc[mask, feat]
                print(f"    {CLASS_NAMES[cls]}: mean={vals.mean():.3f}, "
                      f"median={vals.median():.3f}, p90={vals.quantile(0.90):.3f}")

    # CUSUM threshold calibration
    print(f"\n  CUSUM direction during BEAR periods:")
    bear_mask = actual == 0
    if bear_mask.sum() > 0:
        bear_direction = change_df.loc[bear_mask, "return_cusum_direction"]
        neg_pct = (bear_direction < 0).mean()
        print(f"    Negative direction: {neg_pct:.0%} of BEAR days")

    return change_df


# ─────────────────────────────────────────────────────────────────────
# Part 3: Composite Risk Score
# ─────────────────────────────────────────────────────────────────────

def run_composite_analysis(
    df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    change_df: pd.DataFrame,
    feature_cols: list[str],
    weights: dict,
) -> pd.DataFrame:
    """Combine range + anomaly + change into composite risk score.

    Returns DataFrame with risk_score and risk_regime for OOS period.
    """
    print(f"\n{'='*60}")
    print("PART 3: Composite Risk Score")
    print("=" * 60)

    oos_days = anomaly_df.index

    # Range features for OOS
    range_cols = [c for c in feature_cols if c.startswith("range_")]
    range_feats = df[range_cols].reindex(oos_days).fillna(0)

    # Anomaly features (need anomaly_score column)
    anomaly_feats = anomaly_df[["anomaly_score"]].copy()

    # Change features
    change_feats = change_df.copy()

    risk_df = compute_regime_risk_score(
        range_features=range_feats,
        anomaly_features=anomaly_feats,
        change_features=change_feats,
        weights=weights,
    )

    # Analyze risk regimes vs actual
    actual = anomaly_df["actual_regime"]
    print(f"\n  Risk score by actual regime (weights: range={weights['range']:.2f}, "
          f"anomaly={weights['anomaly']:.2f}, change={weights['change']:.2f}):")
    for cls in [0, 1, 2]:
        mask = actual == cls
        if mask.sum() > 0:
            score = risk_df.loc[mask, "risk_score"]
            print(f"    {CLASS_NAMES[cls]}: mean={score.mean():.3f}, "
                  f"median={score.median():.3f}, p75={score.quantile(0.75):.3f}")

    # Risk regime distribution
    print(f"\n  Risk regime distribution:")
    regime_names = {0: "low", 1: "elevated", 2: "high", 3: "extreme"}
    for r in range(4):
        mask = risk_df["risk_regime"] == r
        n = mask.sum()
        pct = n / len(risk_df) * 100
        # What fraction of each risk regime is actually BEAR?
        bear_in_regime = (actual.reindex(risk_df.index)[mask] == 0).mean() if mask.sum() > 0 else 0
        print(f"    {regime_names[r]:>10s}: {n:4d} days ({pct:5.1f}%), BEAR rate={bear_in_regime:.1%}")

    # Weight sweep (if not custom weights)
    if weights == {"range": 0.25, "anomaly": 0.40, "change": 0.35}:
        print(f"\n  Weight sweep (risk_score mean during BEAR):")
        best_sep = 0
        best_w = None
        bear_mask = actual == 0
        bull_mask = actual == 2
        for wr in [0.15, 0.25, 0.35]:
            for wa in [0.25, 0.35, 0.45, 0.55]:
                wc = 1.0 - wr - wa
                if wc < 0.1:
                    continue
                w = {"range": wr, "anomaly": wa, "change": wc}
                r = compute_regime_risk_score(range_feats, anomaly_feats, change_feats, w)
                bear_mean = r.loc[bear_mask, "risk_score"].mean() if bear_mask.sum() > 0 else 0
                bull_mean = r.loc[bull_mask, "risk_score"].mean() if bull_mask.sum() > 0 else 0
                sep = bear_mean - bull_mean
                if sep > best_sep:
                    best_sep = sep
                    best_w = w
        if best_w:
            print(f"    Best separation: range={best_w['range']:.2f}, "
                  f"anomaly={best_w['anomaly']:.2f}, change={best_w['change']:.2f} "
                  f"(gap={best_sep:.3f})")
            # Recompute with best weights
            risk_df = compute_regime_risk_score(range_feats, anomaly_feats, change_feats, best_w)

    return risk_df


# ─────────────────────────────────────────────────────────────────────
# Part 4: Strategy Backtest
# ─────────────────────────────────────────────────────────────────────

def compute_risk_scaled_strategy(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    alpha: float,
    index,
) -> pd.DataFrame:
    """Position = max(0, 1 - alpha * risk_score). Long-biased with risk scaling."""
    position = np.maximum(0, 1.0 - alpha * risk_score)
    return make_strategy_df(daily_ret, position, index)


def compute_adaptive_dd_exit(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    base_dd: float,
    beta: float,
    reentry_days: int,
    index,
) -> pd.DataFrame:
    """Drawdown exit with adaptive threshold based on risk score.

    Threshold = base_dd * (1 - beta * risk_score).
    Higher risk → tighter threshold → earlier exit.
    """
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    flat_counter = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if flat_counter > 0:
            position[i] = 0.0
            flat_counter -= 1
            if flat_counter == 0:
                peak_equity = equity
        else:
            # Adaptive threshold: tighter when risk is high
            dd_threshold = -base_dd * (1 - beta * risk_score[i])
            dd_threshold = min(dd_threshold, -0.005)  # floor at 0.5%

            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < dd_threshold:
                position[i] = 0.0
                flat_counter = reentry_days
            else:
                position[i] = 1.0

    return make_strategy_df(daily_ret, position, index)


def compute_combined_risk_strategy(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    range_pos_monthly: np.ndarray,
    alpha: float,
    base_dd: float,
    beta: float,
    reentry_days: int,
    index,
) -> pd.DataFrame:
    """Combined: risk sizing + adaptive DD exit + range-gated re-entry."""
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    flat_counter = 0
    in_market = True

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if not in_market:
            flat_counter -= 1
            if flat_counter <= 0:
                # Range-gated re-entry: only re-enter when monthly range_pos is centered
                rp = range_pos_monthly[i] if i < len(range_pos_monthly) else 0.5
                if 0.2 <= rp <= 0.8:
                    in_market = True
                    peak_equity = equity
                    position[i] = max(0, 1.0 - alpha * risk_score[i])
                else:
                    position[i] = 0.0
            else:
                position[i] = 0.0
        else:
            # Risk-scaled position
            position[i] = max(0, 1.0 - alpha * risk_score[i])

            # Adaptive DD exit check
            dd_threshold = -base_dd * (1 - beta * risk_score[i])
            dd_threshold = min(dd_threshold, -0.005)
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < dd_threshold:
                in_market = False
                position[i] = 0.0
                flat_counter = reentry_days

    return make_strategy_df(daily_ret, position, index)


def compute_drawdown_exit_strategy(
    daily_ret: np.ndarray,
    dd_threshold: float,
    reentry_days: int,
    index,
) -> pd.DataFrame:
    """Simple mechanical drawdown exit (baseline)."""
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    flat_counter = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if flat_counter > 0:
            position[i] = 0.0
            flat_counter -= 1
            if flat_counter == 0:
                peak_equity = equity
        elif (equity - peak_equity) / peak_equity < dd_threshold:
            position[i] = 0.0
            flat_counter = reentry_days
        else:
            position[i] = 1.0

    return make_strategy_df(daily_ret, position, index)


def find_drawdown_events(close: pd.Series, threshold: float = -0.05) -> list[tuple]:
    """Find drawdown events exceeding threshold. Returns [(start, trough, end, depth), ...]."""
    cum_max = close.cummax()
    dd = (close - cum_max) / cum_max

    events = []
    in_dd = False
    start = None
    trough_val = 0
    trough_date = None

    for i in range(len(dd)):
        if dd.iloc[i] < threshold and not in_dd:
            in_dd = True
            # Walk back to find the peak before this drawdown
            start = dd.index[i]
            trough_val = dd.iloc[i]
            trough_date = dd.index[i]
        elif in_dd:
            if dd.iloc[i] < trough_val:
                trough_val = dd.iloc[i]
                trough_date = dd.index[i]
            if dd.iloc[i] > threshold * 0.3:  # recovered enough
                events.append((start, trough_date, dd.index[i], trough_val))
                in_dd = False

    if in_dd:
        events.append((start, trough_date, dd.index[-1], trough_val))

    return events


def run_strategy_backtest(
    es_daily: pd.DataFrame,
    risk_df: pd.DataFrame,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run all strategy variants and collect metrics."""
    print(f"\n{'='*60}")
    print("PART 4: Strategy Backtest")
    print("=" * 60)

    oos_days = risk_df.index
    daily_ret = es_daily["close"].pct_change().reindex(oos_days).fillna(0).values
    risk_score = risk_df["risk_score"].values

    # Monthly range position for re-entry gating
    range_pos_col = "range_pos_20d"
    if range_pos_col in df.columns:
        range_pos_monthly = df[range_pos_col].reindex(oos_days).fillna(0.5).values
    else:
        range_pos_monthly = np.full(len(oos_days), 0.5)

    variants = []
    strat_dfs = {}

    # Buy & Hold
    bh_pos = np.ones(len(oos_days))
    bh_df = make_strategy_df(daily_ret, bh_pos, oos_days)
    strat_dfs["buy_hold"] = bh_df
    variants.append(compute_strategy_metrics(bh_df, "Buy&Hold"))

    # DD-only baselines
    for dd_pct in [0.02, 0.03, 0.04]:
        label = f"DD-only {dd_pct:.0%}"
        sdf = compute_drawdown_exit_strategy(daily_ret, -dd_pct, 5, oos_days)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Part 4a: Risk-scaled position sizing
    print("\n  4a: Risk-scaled position sizing")
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        label = f"RiskScale a={alpha:.1f}"
        sdf = compute_risk_scaled_strategy(daily_ret, risk_score, alpha, oos_days)
        strat_dfs[label] = sdf
        m = compute_strategy_metrics(sdf, label)
        variants.append(m)
        print(f"    {label}: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}, "
              f"InMkt={m['time_in_market_pct']:.0f}%")

    # Part 4b: Adaptive DD exit
    print("\n  4b: Adaptive DD exit")
    for base_dd in [0.02, 0.03, 0.04]:
        for beta in [0.3, 0.5, 0.7]:
            label = f"AdaptDD dd={base_dd:.0%} b={beta:.1f}"
            sdf = compute_adaptive_dd_exit(daily_ret, risk_score, base_dd, beta, 5, oos_days)
            m = compute_strategy_metrics(sdf, label)
            strat_dfs[label] = sdf
            variants.append(m)

    # Print best adaptive DD
    adapt_variants = [v for v in variants if v["label"].startswith("AdaptDD")]
    if adapt_variants:
        best_adapt = max(adapt_variants, key=lambda x: x["sharpe"])
        print(f"    Best: {best_adapt['label']} → Sharpe={best_adapt['sharpe']:.2f}, "
              f"MaxDD={best_adapt['max_dd']:.1%}")

    # Part 4c: Combined strategies
    print("\n  4c: Combined (risk scaling + adaptive DD + range re-entry)")
    combined_configs = [
        (1.0, 0.03, 0.5, 5),
        (1.5, 0.03, 0.5, 5),
        (1.0, 0.02, 0.5, 5),
        (1.5, 0.02, 0.7, 5),
    ]
    for alpha, base_dd, beta, reentry in combined_configs:
        label = f"Combined a={alpha:.1f} dd={base_dd:.0%} b={beta:.1f}"
        sdf = compute_combined_risk_strategy(
            daily_ret, risk_score, range_pos_monthly,
            alpha, base_dd, beta, reentry, oos_days,
        )
        m = compute_strategy_metrics(sdf, label)
        strat_dfs[label] = sdf
        variants.append(m)
        print(f"    {label}: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # Part 4d: Summary table
    print(f"\n  {'Label':<35s} {'TotRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
          f"{'Calmar':>7s} {'InMkt':>6s}")
    print("  " + "-" * 75)
    for v in sorted(variants, key=lambda x: -x["sharpe"]):
        print(f"  {v['label']:<35s} {v['total_return']:>7.1%} {v['sharpe']:>7.2f} "
              f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['time_in_market_pct']:>5.1f}%")

    # Drawdown capture analysis
    close = es_daily["close"].reindex(oos_days)
    dd_events = find_drawdown_events(close)
    if dd_events:
        print(f"\n  Drawdown capture analysis ({len(dd_events)} events >5%):")
        top_labels = ["Buy&Hold", "DD-only 2%", "DD-only 3%"]
        # Add best risk-scaled and best combined
        best_risk = max([v for v in variants if v["label"].startswith("RiskScale")],
                       key=lambda x: x["sharpe"])
        best_combined = max([v for v in variants if v["label"].startswith("Combined")],
                          key=lambda x: x["sharpe"])
        top_labels += [best_risk["label"], best_combined["label"]]

        for label in top_labels:
            if label not in strat_dfs:
                continue
            sdf = strat_dfs[label]
            avg_exposure = []
            for start, trough, end, depth in dd_events:
                mask = (sdf.index >= start) & (sdf.index <= end)
                if mask.sum() > 0:
                    avg_pos = sdf.loc[mask, "position"].mean()
                    avg_exposure.append(avg_pos)
            if avg_exposure:
                print(f"    {label:<35s}: avg exposure during DD = {np.mean(avg_exposure):.2f}")

    return variants, strat_dfs


# ─────────────────────────────────────────────────────────────────────
# Part 5: Supervised Baseline Comparison
# ─────────────────────────────────────────────────────────────────────

def run_supervised_baseline(
    df: pd.DataFrame,
    es_daily: pd.DataFrame,
    feature_cols: list[str],
    risk_df: pd.DataFrame,
    n_folds: int,
    min_train_days: int,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run 3-class supervised walk-forward XGB and build strategy variants."""
    print(f"\n{'='*60}")
    print("PART 5: Supervised Baseline Comparison")
    print("=" * 60)

    oos_days = risk_df.index
    daily_ret = es_daily["close"].pct_change().reindex(oos_days).fillna(0).values

    # Walk-forward 3-class XGB
    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(es_daily, train_end_idx, n_states=3)

    print("  Training 3-class structural regime model...")
    result = walk_forward_cv(
        df=df,
        feature_cols=feature_cols,
        target_col="y_structural",
        n_folds=n_folds,
        min_train_days=min_train_days,
        params={"bear_upweight": 1.5},
        hmm_fn=hmm_fn,
        verbose=True,
    )

    print(f"  Aggregate accuracy: {result.aggregate_accuracy:.3f}")
    print(f"  Directional accuracy: {result.aggregate_directional_acc:.3f}")

    # Map OOS predictions back to dates
    days = sorted(df.index.unique())
    n_days = len(days)
    preds = result.oos_predictions
    actuals = result.oos_actuals
    probas = result.oos_probas

    pred_days = days[n_days - len(preds):]

    oos_pred_df = pd.DataFrame(index=pred_days)
    oos_pred_df["pred"] = preds
    oos_pred_df["actual"] = actuals[:len(pred_days)]
    oos_pred_df["p_bear"] = probas[:len(pred_days), 0]
    oos_pred_df["p_bull"] = probas[:len(pred_days), 2]
    oos_pred_df["daily_return"] = es_daily["close"].pct_change().reindex(pred_days).fillna(0).values

    # Align to OOS period from unsupervised
    common_days = sorted(set(oos_days) & set(pred_days))
    if len(common_days) < len(oos_days) * 0.5:
        print(f"  WARNING: Only {len(common_days)} common days between supervised and unsupervised OOS")

    # Strategy variants from supervised model
    variants = []
    strat_dfs = {}

    # 5a: Long-biased + model exit
    print("\n  5a: Supervised long-biased strategies")
    for conf in [0.40, 0.50, 0.60]:
        label = f"Sup: LongBiased conf={conf:.2f}"
        n = len(common_days)
        sup_aligned = oos_pred_df.reindex(common_days)
        position = np.ones(n)
        for i in range(n):
            if sup_aligned["pred"].iloc[i] == 0 and sup_aligned["p_bear"].iloc[i] > conf:
                position[i] = 0.0

        dr = es_daily["close"].pct_change().reindex(common_days).fillna(0).values
        sdf = make_strategy_df(dr, position, common_days)
        strat_dfs[label] = sdf
        m = compute_strategy_metrics(sdf, label)
        variants.append(m)
        print(f"    {label}: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # 5b: DD exit + model re-entry
    print("\n  5b: DD exit + supervised model re-entry")
    for dd_pct in [0.02, 0.03]:
        label = f"Sup: DD {dd_pct:.0%} + model re-entry"
        n = len(common_days)
        sup_aligned = oos_pred_df.reindex(common_days)
        dr = es_daily["close"].pct_change().reindex(common_days).fillna(0).values

        position = np.ones(n)
        equity = 1.0
        peak_equity = 1.0
        in_market = True
        bull_consec = 0

        for i in range(1, n):
            equity *= (1 + position[i - 1] * dr[i])
            peak_equity = max(peak_equity, equity)

            if in_market:
                if (equity - peak_equity) / peak_equity < -dd_pct:
                    in_market = False
                    position[i] = 0.0
                    bull_consec = 0
                else:
                    position[i] = 1.0
            else:
                # Re-enter on 3 consecutive BULL predictions
                pred_val = sup_aligned["pred"].iloc[i]
                p_bull_val = sup_aligned["p_bull"].iloc[i]
                if pred_val == 2 and p_bull_val > 0.40:
                    bull_consec += 1
                else:
                    bull_consec = 0
                if bull_consec >= 3:
                    in_market = True
                    peak_equity = equity
                    position[i] = 1.0
                else:
                    position[i] = 0.0

        sdf = make_strategy_df(dr, position, common_days)
        strat_dfs[label] = sdf
        m = compute_strategy_metrics(sdf, label)
        variants.append(m)
        print(f"    {label}: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # 5c: Combined DD + model
    print("\n  5c: Combined DD + supervised asymmetric")
    label = "Sup: Combined DD+Asym"
    n = len(common_days)
    sup_aligned = oos_pred_df.reindex(common_days)
    dr = es_daily["close"].pct_change().reindex(common_days).fillna(0).values

    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    in_market = True
    bear_consec = 0
    bull_consec = 0
    flat_counter = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * dr[i])
        peak_equity = max(peak_equity, equity)

        pred_val = sup_aligned["pred"].iloc[i]
        p_bear_val = sup_aligned["p_bear"].iloc[i]
        p_bull_val = sup_aligned["p_bull"].iloc[i]

        if pred_val == 0 and p_bear_val > 0.50:
            bear_consec += 1
            bull_consec = 0
        elif pred_val == 2 and p_bull_val > 0.40:
            bull_consec += 1
            bear_consec = 0
        else:
            bear_consec = 0
            bull_consec = 0

        if in_market:
            # Model exit
            if bear_consec >= 1:
                in_market = False
                position[i] = 0.0
                flat_counter = 0
            # DD exit
            elif (equity - peak_equity) / peak_equity < -0.03:
                in_market = False
                position[i] = 0.0
                flat_counter = 5
            else:
                position[i] = 1.0
        else:
            if flat_counter > 0:
                flat_counter -= 1
                position[i] = 0.0
            elif bull_consec >= 3:
                in_market = True
                peak_equity = equity
                position[i] = 1.0
            else:
                position[i] = 0.0

    sdf = make_strategy_df(dr, position, common_days)
    strat_dfs[label] = sdf
    m = compute_strategy_metrics(sdf, label)
    variants.append(m)
    print(f"    {label}: Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # Summary
    print(f"\n  Supervised strategies summary:")
    print(f"  {'Label':<35s} {'TotRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
          f"{'Calmar':>7s} {'InMkt':>6s}")
    print("  " + "-" * 75)
    for v in sorted(variants, key=lambda x: -x["sharpe"]):
        print(f"  {v['label']:<35s} {v['total_return']:>7.1%} {v['sharpe']:>7.2f} "
              f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['time_in_market_pct']:>5.1f}%")

    return variants, strat_dfs


# ─────────────────────────────────────────────────────────────────────
# Part 6: Summary + Figures
# ─────────────────────────────────────────────────────────────────────

def plot_figures(
    es_daily: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    unsup_variants: list[dict],
    unsup_strat_dfs: dict[str, pd.DataFrame],
    sup_variants: list[dict],
    sup_strat_dfs: dict[str, pd.DataFrame],
    skip_plots: bool = False,
):
    """Generate all summary figures."""
    if skip_plots:
        print("\n  Skipping plots (--skip-plots)")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    oos_days = risk_df.index

    # Figure 1: Equity curves comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    close = es_daily["close"].reindex(oos_days)
    bh_cum = (close / close.iloc[0] - 1) * 100

    ax.plot(oos_days, bh_cum, label="Buy&Hold", color="gray", alpha=0.7)

    # Best unsupervised variants
    key_labels = ["DD-only 2%", "DD-only 3%"]
    best_risk = max([v for v in unsup_variants if v["label"].startswith("RiskScale")],
                   key=lambda x: x["sharpe"], default=None)
    best_combined = max([v for v in unsup_variants if v["label"].startswith("Combined")],
                       key=lambda x: x["sharpe"], default=None)
    if best_risk:
        key_labels.append(best_risk["label"])
    if best_combined:
        key_labels.append(best_combined["label"])

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    for i, label in enumerate(key_labels):
        if label in unsup_strat_dfs:
            sdf = unsup_strat_dfs[label]
            ax.plot(sdf.index, sdf["cum_return"] * 100, label=label,
                   color=colors[i % len(colors)], linewidth=1.5)

    # Best supervised
    if sup_variants:
        best_sup = max(sup_variants, key=lambda x: x["sharpe"])
        if best_sup["label"] in sup_strat_dfs:
            sdf = sup_strat_dfs[best_sup["label"]]
            ax.plot(sdf.index, sdf["cum_return"] * 100, label=f"[Sup] {best_sup['label']}",
                   color="C6", linewidth=1.5, linestyle="--")

    # Shade BEAR periods
    actual = anomaly_df["actual_regime"]
    bear_mask = actual == 0
    groups = (bear_mask != bear_mask.shift()).cumsum()
    for _, grp in bear_mask.groupby(groups):
        if grp.iloc[0]:
            ax.axvspan(grp.index[0], grp.index[-1], color="red", alpha=0.1)

    ax.set_title("Equity Curves: Unsupervised vs Supervised vs Baselines")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'equity_curves.png'}")

    # Figure 2: Anomaly score + price with drawdown shading
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(oos_days, close.values, color="black", linewidth=0.8)
    ax1.set_ylabel("ES Close")
    ax1.set_title("ES Price with Anomaly Score")

    # Shade BEAR periods
    for _, grp in bear_mask.groupby(groups):
        if grp.iloc[0]:
            ax1.axvspan(grp.index[0], grp.index[-1], color="red", alpha=0.1)

    ax2.fill_between(oos_days, 0, anomaly_df["anomaly_score"].values,
                     alpha=0.5, color="C1")
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("Anomaly Score")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_score_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'anomaly_score_timeseries.png'}")

    # Figure 3: Risk score heatmap (component breakdown)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    components = ["range_stress", "anomaly_intensity", "change_momentum", "risk_score"]
    titles = ["Range Stress", "Anomaly Intensity", "Change Momentum", "Composite Risk Score"]
    colors_cm = ["C0", "C1", "C2", "C3"]

    for ax, comp, title, c in zip(axes, components, titles, colors_cm):
        vals = risk_df[comp].values
        ax.fill_between(oos_days, 0, vals, alpha=0.5, color=c)
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        # Shade BEAR
        for _, grp in bear_mask.groupby(groups):
            if grp.iloc[0]:
                ax.axvspan(grp.index[0], grp.index[-1], color="red", alpha=0.1)

    axes[0].set_title("Risk Score Components vs Actual BEAR Periods (red shading)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "risk_score_components.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'risk_score_components.png'}")

    # Figure 4: Strategy comparison table
    all_variants = []
    for v in unsup_variants:
        v_copy = v.copy()
        v_copy["source"] = "unsupervised"
        all_variants.append(v_copy)
    for v in sup_variants:
        v_copy = v.copy()
        v_copy["source"] = "supervised"
        all_variants.append(v_copy)

    # Sort by Sharpe
    all_variants.sort(key=lambda x: -x["sharpe"])
    top_n = min(20, len(all_variants))

    fig, ax = plt.subplots(figsize=(12, 0.4 * top_n + 1.5))
    ax.axis("off")
    headers = ["Strategy", "Src", "TotRet", "Sharpe", "MaxDD", "Calmar", "InMkt%"]
    cell_data = []
    for v in all_variants[:top_n]:
        cell_data.append([
            v["label"][:35],
            v.get("source", "?")[:5],
            f"{v['total_return']:.1%}",
            f"{v['sharpe']:.2f}",
            f"{v['max_dd']:.1%}",
            f"{v['calmar']:.2f}",
            f"{v['time_in_market_pct']:.0f}%",
        ])

    table = ax.table(cellText=cell_data, colLabels=headers, loc="center",
                    cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    ax.set_title("Top Strategies by Sharpe (Unsupervised + Supervised)", fontsize=11, pad=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "strategy_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'strategy_comparison_table.png'}")


def print_final_summary(
    unsup_variants: list[dict],
    sup_variants: list[dict],
):
    """Print final comparison summary."""
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    # Baselines
    bh = next((v for v in unsup_variants if v["label"] == "Buy&Hold"), None)
    dd2 = next((v for v in unsup_variants if v["label"] == "DD-only 2%"), None)
    dd3 = next((v for v in unsup_variants if v["label"] == "DD-only 3%"), None)

    print("\n  Baselines:")
    for v in [bh, dd2, dd3]:
        if v:
            print(f"    {v['label']:<25s}: Sharpe={v['sharpe']:.2f}, MaxDD={v['max_dd']:.1%}")

    # Best unsupervised
    unsup_best = max(unsup_variants, key=lambda x: x["sharpe"])
    best_risk = max([v for v in unsup_variants if v["label"].startswith("RiskScale")],
                   key=lambda x: x["sharpe"], default=None)
    best_adapt = max([v for v in unsup_variants if v["label"].startswith("AdaptDD")],
                    key=lambda x: x["sharpe"], default=None)
    best_combined = max([v for v in unsup_variants if v["label"].startswith("Combined")],
                       key=lambda x: x["sharpe"], default=None)

    print("\n  Best unsupervised strategies:")
    for label, v in [("Risk-scaled", best_risk), ("Adaptive DD", best_adapt),
                     ("Combined", best_combined)]:
        if v:
            print(f"    {label}: {v['label']}")
            print(f"      Sharpe={v['sharpe']:.2f}, MaxDD={v['max_dd']:.1%}, "
                  f"Calmar={v['calmar']:.2f}, InMkt={v['time_in_market_pct']:.0f}%")

    # Best supervised
    if sup_variants:
        sup_best = max(sup_variants, key=lambda x: x["sharpe"])
        print(f"\n  Best supervised strategy:")
        print(f"    {sup_best['label']}")
        print(f"      Sharpe={sup_best['sharpe']:.2f}, MaxDD={sup_best['max_dd']:.1%}, "
              f"Calmar={sup_best['calmar']:.2f}, InMkt={sup_best['time_in_market_pct']:.0f}%")

    # Success criteria
    print(f"\n  Success criteria:")
    dd_only_sharpe = dd2["sharpe"] if dd2 else 0.79
    dd_only_maxdd = dd2["max_dd"] if dd2 else -0.10

    if best_combined:
        sharpe_pass = best_combined["sharpe"] > dd_only_sharpe
        maxdd_improve = abs(best_combined["max_dd"]) < abs(dd_only_maxdd) - 0.05
        print(f"    Sharpe > DD-only ({dd_only_sharpe:.2f}): "
              f"{best_combined['sharpe']:.2f} → {'PASS' if sharpe_pass else 'FAIL'}")
        print(f"    OR MaxDD improve >5pp ({dd_only_maxdd:.1%}): "
              f"{best_combined['max_dd']:.1%} → {'PASS' if maxdd_improve else 'FAIL'}")
        print(f"    Overall: {'PASS' if sharpe_pass or maxdd_improve else 'FAIL'}")

    # Year-by-year for top strategies
    print(f"\n  Year-by-year returns:")
    top_labels = ["Buy&Hold"]
    if dd2:
        top_labels.append(dd2["label"])
    if best_combined:
        top_labels.append(best_combined["label"])
    if sup_variants:
        sup_best = max(sup_variants, key=lambda x: x["sharpe"])
        top_labels.append(sup_best["label"])

    all_v = unsup_variants + sup_variants
    for tl in top_labels:
        v = next((x for x in all_v if x["label"] == tl), None)
        if v and "yearly" in v:
            yearly_str = "  ".join(f"{yr}:{r:+.1%}" for yr, r in sorted(v["yearly"].items()))
            print(f"    {v['label']:<35s}: {yearly_str}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200
    weights = {
        "range": args.weight_range,
        "anomaly": args.weight_anomaly,
        "change": args.weight_change,
    }

    # Part 0: Data
    df, es_daily, feature_cols = load_data_pipeline(args)

    # Part 1: Anomaly detection
    anomaly_df = run_anomaly_analysis(df, es_daily, feature_cols, n_folds, min_train)

    # Part 2: Change detection
    change_df = run_change_analysis(es_daily, anomaly_df)

    # Part 3: Composite risk score
    risk_df = run_composite_analysis(df, anomaly_df, change_df, feature_cols, weights)

    # Part 4: Strategy backtest
    unsup_variants, unsup_strat_dfs = run_strategy_backtest(es_daily, risk_df, df, feature_cols)

    # Part 5: Supervised baseline
    sup_variants, sup_strat_dfs = run_supervised_baseline(
        df, es_daily, feature_cols, risk_df, n_folds, min_train,
    )

    # Part 6: Summary + Figures
    print(f"\n{'='*60}")
    print("PART 6: Summary + Figures")
    print("=" * 60)

    plot_figures(
        es_daily, anomaly_df, risk_df,
        unsup_variants, unsup_strat_dfs,
        sup_variants, sup_strat_dfs,
        skip_plots=args.skip_plots,
    )

    print_final_summary(unsup_variants, sup_variants)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
