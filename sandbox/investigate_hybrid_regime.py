"""Hybrid regime strategy + multi-instrument generalization.

Two improvements over previous unsupervised/supervised experiments:
  1. Hybrid: Unsupervised adaptive DD exit + supervised BULL re-entry
  2. Discrete tiers: Step-function position sizing instead of continuous scaling

Tests generalization across 6 instruments: ES, NQ, YM, CL, GC, ZN.

Parts:
  0. Data pipeline (load, aggregate, transferable features only)
  1. Unsupervised risk score (walk-forward IF + CUSUM/EWMA)
  2. Supervised BULL model (walk-forward 3-class XGB)
  3. Strategy variants (B&H, DD-only, discrete tiers, adaptive DD, hybrid, hybrid+tiers)
  4. Multi-instrument comparison
  5. Summary + figures

Usage:
    source ~/ml-venv/bin/activate

    # ES only (fast, ~5 min)
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_hybrid_regime.py

    # Quick debug
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_hybrid_regime.py --quick --folds 2

    # Multi-instrument
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_hybrid_regime.py --instruments ES NQ YM CL GC ZN

    # Specific instruments
    PYTHONPATH=/home/william/gybcap python -u sandbox/investigate_hybrid_regime.py --instruments ES CL GC
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strategies.swing.config import InstrumentConfig, INSTRUMENTS, SwingConfig
from strategies.swing.data_loader import InstrumentLoader
from strategies.swing.daily_aggregator import DailyAggregator
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
from strategies.swing.labeling.structural_regime import compute_labels
from strategies.swing.labeling.hmm_regime import compute_hmm_features_walkforward
from strategies.swing.detection.anomaly_detector import RollingAnomalyDetector
from strategies.swing.detection.change_detector import compute_change_features
from strategies.swing.detection.regime_risk import compute_regime_risk_score
from strategies.swing.training.regime_trainer import walk_forward_cv


FIGURES_DIR = Path("sandbox/figures/hybrid_regime")
CLASS_NAMES = {0: "BEAR", 1: "BALANCE", 2: "BULL"}

# Instrument configs — experiment-only, extends config.py's 3 instruments
EXTRA_INSTRUMENTS = {
    "YM": InstrumentConfig("YM", Path("/mnt/d/data/ym_min_2011.csv"), "mnt_standard", tick_size=1.0),
    "CL": InstrumentConfig("CL", Path("/mnt/d/data/cl_min_2011.csv"), "mnt_standard", tick_size=0.01),
    "GC": InstrumentConfig("GC", Path("/mnt/d/data/gc_min_2011.csv"), "mnt_standard", tick_size=0.1),
}

ALL_INSTRUMENTS = {**INSTRUMENTS, **EXTRA_INSTRUMENTS}


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid regime strategy + multi-instrument")
    parser.add_argument("--instruments", nargs="+", default=["ES"],
                        help="Instruments to test (default: ES)")
    parser.add_argument("--quick", action="store_true", help="Quick debug run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Shared: Strategy metrics
# ─────────────────────────────────────────────────────────────────────

def compute_strategy_metrics(strat_df: pd.DataFrame, label: str) -> dict:
    """Compute Sharpe, MaxDD, Calmar, time-in-market, etc."""
    ret = strat_df["strategy_return"]
    pos = strat_df["position"]
    n_days = len(strat_df)
    years = n_days / 252

    in_market_pct = (pos > 0).mean() * 100

    total_return = (1 + ret).prod() - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    vol = ret.std() * np.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else 0

    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

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
    strat_df["strategy_return"] = pd.Series(position, index=index).shift(1).fillna(1.0) * daily_ret
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


# ─────────────────────────────────────────────────────────────────────
# Part 0: Data Pipeline (per instrument)
# ─────────────────────────────────────────────────────────────────────

def load_instrument_pipeline(
    symbol: str,
    n_folds: int,
    min_train_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load one instrument, compute transferable features + labels.

    Only uses OHLCV-based features (no cross-instrument, no external).
    Returns (df_with_features_and_labels, daily, feature_cols).
    """
    cfg = ALL_INSTRUMENTS[symbol]
    if not cfg.path.exists():
        raise FileNotFoundError(f"{symbol} data not found at {cfg.path}")

    loader = InstrumentLoader()
    print(f"  Loading {symbol} from {cfg.path}...")
    minute_df = loader.load(cfg)
    print(f"    {len(minute_df):,} bars, {minute_df.index.min()} – {minute_df.index.max()}")

    aggregator = DailyAggregator()
    compute_vp = "volume" in minute_df.columns
    daily = aggregator.aggregate(minute_df, compute_vp=compute_vp)
    print(f"    {len(daily)} trading days")

    # Transferable features only (instrument-agnostic)
    print(f"  Computing features for {symbol}...")
    feature_cols = []

    # 1. Daily technical (47 features) — OHLCV-based
    tech_feats = compute_daily_technical(daily)
    feature_cols += list(TECH_FEATURES)

    # 2. Volume profile daily (11 features) — aggregator VP
    vp_feats = pd.DataFrame(index=daily.index)
    if "vp_poc_rel" in daily.columns:
        vp_feats = compute_vp_daily_features(daily)
        feature_cols += VP_FEATURES

    # 3. Macro context (9 features) — price-based, no cross-instrument
    macro_feats = compute_macro_context(daily, other_dailys=None)
    feature_cols += MACRO_FEATURES

    # 4. Range features (16 features)
    range_feats = compute_range_features(daily)
    feature_cols += RANGE_FEATURES

    all_feats = pd.concat([tech_feats, vp_feats, macro_feats, range_feats], axis=1)
    all_feats = all_feats.reindex(daily.index).fillna(0)

    # Labels (structural regime)
    labels = compute_labels(daily)
    df = all_feats.join(labels)

    print(f"    {len(feature_cols)} features (transferable only)")

    struct = df["y_structural"]
    valid = struct[struct.isin([0, 1, 2])]
    if len(valid) > 0:
        dist = valid.value_counts(normalize=True).sort_index()
        for k, v in dist.items():
            print(f"      {CLASS_NAMES.get(k, '?')}: {v:.1%}")

    return df, daily, feature_cols


# ─────────────────────────────────────────────────────────────────────
# Part 1: Unsupervised Risk Score
# ─────────────────────────────────────────────────────────────────────

def compute_unsupervised_risk(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_train_days: int,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Walk-forward anomaly + change detection → composite risk score.

    Returns risk_df with columns: anomaly_score, risk_score, risk_regime, etc.
    """
    if weights is None:
        weights = {"range": 0.15, "anomaly": 0.25, "change": 0.60}

    days = sorted(df.index.unique())
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds

    feature_matrix = df[feature_cols].fillna(0)

    # Walk-forward Isolation Forest
    all_scores_list = []
    for fold in range(n_folds):
        train_end = min_train_days + fold * test_days_per_fold
        test_end = train_end + test_days_per_fold if fold < n_folds - 1 else n_days
        test_days_list = days[train_end:test_end]

        detector = RollingAnomalyDetector(n_estimators=200)
        result = detector.fit_score(feature_matrix, train_end)
        fold_scores = result.loc[result.index.isin(test_days_list), "anomaly_score"]
        all_scores_list.append(fold_scores)

    all_scores = pd.concat(all_scores_list)
    oos_days = sorted(all_scores.index.unique())

    anomaly_df = pd.DataFrame(index=oos_days)
    anomaly_df["anomaly_score"] = all_scores.reindex(oos_days)

    # Change detection
    daily_ret = daily["close"].pct_change().reindex(oos_days).fillna(0)
    change_df = compute_change_features(daily_ret, anomaly_df["anomaly_score"])
    change_df = change_df.reindex(oos_days).fillna(0)

    # Range features for composite
    range_feats = df[[c for c in RANGE_FEATURES if c in df.columns]].reindex(oos_days).fillna(0)

    # Composite risk score
    risk_df = compute_regime_risk_score(range_feats, anomaly_df, change_df, weights)
    risk_df["anomaly_score"] = anomaly_df["anomaly_score"]

    return risk_df


# ─────────────────────────────────────────────────────────────────────
# Part 2: Supervised BULL Model
# ─────────────────────────────────────────────────────────────────────

def train_supervised_model(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_train_days: int,
) -> pd.DataFrame:
    """Walk-forward 3-class XGB → OOS predictions + P(BULL).

    Returns pred_df with columns: pred, p_bear, p_bull.
    """
    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(daily, train_end_idx, n_states=3)

    result = walk_forward_cv(
        df=df,
        feature_cols=feature_cols,
        target_col="y_structural",
        n_folds=n_folds,
        min_train_days=min_train_days,
        params={"bear_upweight": 1.5},
        hmm_fn=hmm_fn,
        verbose=False,
    )

    print(f"    Supervised: Acc={result.aggregate_accuracy:.3f}, "
          f"DirAcc={result.aggregate_directional_acc:.3f}, "
          f"F1={result.aggregate_f1:.3f}")

    # Map predictions back to dates
    days = sorted(df.index.unique())
    n_days = len(days)
    preds = result.oos_predictions
    probas = result.oos_probas

    pred_days = days[n_days - len(preds):]
    pred_df = pd.DataFrame(index=pred_days)
    pred_df["pred"] = preds
    pred_df["p_bear"] = probas[:len(pred_days), 0]
    pred_df["p_bull"] = probas[:len(pred_days), 2]

    return pred_df


# ─────────────────────────────────────────────────────────────────────
# Part 3: Strategy Variants
# ─────────────────────────────────────────────────────────────────────

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


def compute_adaptive_dd_exit(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    base_dd: float,
    beta: float,
    reentry_days: int,
    index,
) -> pd.DataFrame:
    """Adaptive DD exit: threshold = base_dd * (1 - beta * risk_score)."""
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
            dd_threshold = -base_dd * (1 - beta * risk_score[i])
            dd_threshold = min(dd_threshold, -0.005)
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < dd_threshold:
                position[i] = 0.0
                flat_counter = reentry_days
            else:
                position[i] = 1.0

    return make_strategy_df(daily_ret, position, index)


def compute_discrete_tier_strategy(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    tiers: list[tuple[float, float]],
    index,
) -> pd.DataFrame:
    """Step-function position sizing based on risk score.

    Args:
        tiers: List of (threshold, position) pairs sorted ascending by threshold.
               E.g. [(0.50, 1.0), (0.75, 0.50), (inf, 0.0)]
               means: risk<0.50 → pos=1.0, 0.50≤risk<0.75 → pos=0.50, risk≥0.75 → pos=0.0
    """
    n = len(daily_ret)
    position = np.ones(n)

    for i in range(n):
        r = risk_score[i]
        for threshold, pos in tiers:
            if r < threshold:
                position[i] = pos
                break

    return make_strategy_df(daily_ret, position, index)


def compute_hybrid_strategy(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    sup_preds: np.ndarray,
    sup_p_bull: np.ndarray,
    base_dd: float,
    beta: float,
    bull_consec_required: int,
    p_bull_threshold: float,
    reentry_days_max: int,
    index,
) -> pd.DataFrame:
    """Unsupervised adaptive DD exit + supervised BULL re-entry.

    EXIT: Adaptive DD threshold = base_dd * (1 - beta * risk_score)
    RE-ENTRY: bull_consec_required consecutive BULL predictions with P(BULL) > p_bull_threshold
    FALLBACK: Re-enter after reentry_days_max if no BULL signal (prevents permanent flat)
    """
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    in_market = True
    bull_consec = 0
    flat_days = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if in_market:
            # Adaptive DD exit
            dd_threshold = -base_dd * (1 - beta * risk_score[i])
            dd_threshold = min(dd_threshold, -0.005)
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < dd_threshold:
                in_market = False
                position[i] = 0.0
                bull_consec = 0
                flat_days = 0
            else:
                position[i] = 1.0
        else:
            flat_days += 1
            # Check BULL re-entry
            if sup_preds[i] == 2 and sup_p_bull[i] > p_bull_threshold:
                bull_consec += 1
            else:
                bull_consec = 0

            if bull_consec >= bull_consec_required:
                in_market = True
                peak_equity = equity
                position[i] = 1.0
            elif flat_days >= reentry_days_max:
                # Fallback: re-enter after max flat days
                in_market = True
                peak_equity = equity
                position[i] = 1.0
            else:
                position[i] = 0.0

    return make_strategy_df(daily_ret, position, index)


def compute_hybrid_tier_strategy(
    daily_ret: np.ndarray,
    risk_score: np.ndarray,
    sup_preds: np.ndarray,
    sup_p_bull: np.ndarray,
    tiers: list[tuple[float, float]],
    base_dd: float,
    beta: float,
    bull_consec_required: int,
    p_bull_threshold: float,
    reentry_days_max: int,
    index,
) -> pd.DataFrame:
    """Discrete tiers for position sizing + adaptive DD exit + supervised re-entry."""
    n = len(daily_ret)
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    in_market = True
    bull_consec = 0
    flat_days = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if in_market:
            # Discrete tier position sizing
            r = risk_score[i]
            pos = 0.0
            for threshold, tier_pos in tiers:
                if r < threshold:
                    pos = tier_pos
                    break
            position[i] = pos

            # Adaptive DD exit check (only when effectively in market)
            dd_threshold = -base_dd * (1 - beta * risk_score[i])
            dd_threshold = min(dd_threshold, -0.005)
            current_dd = (equity - peak_equity) / peak_equity
            if current_dd < dd_threshold:
                in_market = False
                position[i] = 0.0
                bull_consec = 0
                flat_days = 0
        else:
            flat_days += 1
            if sup_preds[i] == 2 and sup_p_bull[i] > p_bull_threshold:
                bull_consec += 1
            else:
                bull_consec = 0

            if bull_consec >= bull_consec_required:
                in_market = True
                peak_equity = equity
                # Apply tier sizing on re-entry
                r = risk_score[i]
                pos = 0.0
                for threshold, tier_pos in tiers:
                    if r < threshold:
                        pos = tier_pos
                        break
                position[i] = pos
            elif flat_days >= reentry_days_max:
                in_market = True
                peak_equity = equity
                r = risk_score[i]
                pos = 0.0
                for threshold, tier_pos in tiers:
                    if r < threshold:
                        pos = tier_pos
                        break
                position[i] = pos
            else:
                position[i] = 0.0

    return make_strategy_df(daily_ret, position, index)


def run_strategies_for_instrument(
    daily: pd.DataFrame,
    risk_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    symbol: str,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run all strategy variants for one instrument."""
    oos_days = risk_df.index
    common_days = sorted(set(oos_days) & set(pred_df.index))
    print(f"    OOS days: {len(oos_days)}, supervised overlap: {len(common_days)}")

    # Use common days for fair comparison
    daily_ret = daily["close"].pct_change().reindex(common_days).fillna(0).values
    risk_score = risk_df["risk_score"].reindex(common_days).fillna(0).values
    sup_preds = pred_df["pred"].reindex(common_days).fillna(1).values.astype(int)
    sup_p_bull = pred_df["p_bull"].reindex(common_days).fillna(0.33).values

    variants = []
    strat_dfs = {}

    # --- a. Buy & Hold ---
    bh_pos = np.ones(len(common_days))
    bh_df = make_strategy_df(daily_ret, bh_pos, common_days)
    strat_dfs["Buy&Hold"] = bh_df
    variants.append(compute_strategy_metrics(bh_df, "Buy&Hold"))

    # --- b. DD-only baselines ---
    for dd_pct in [0.02, 0.03]:
        label = f"DD-only {dd_pct:.0%}"
        sdf = compute_drawdown_exit_strategy(daily_ret, -dd_pct, 5, common_days)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # --- c. Discrete tiers ---
    tier_configs = [
        ("Tiers 50/75", [(0.50, 1.0), (0.75, 0.50), (float("inf"), 0.0)]),
        ("Tiers 40/65", [(0.40, 1.0), (0.65, 0.50), (float("inf"), 0.0)]),
        ("Tiers 50/75 3-step", [(0.50, 1.0), (0.75, 0.50), (0.90, 0.25), (float("inf"), 0.0)]),
    ]
    for label, tiers in tier_configs:
        sdf = compute_discrete_tier_strategy(daily_ret, risk_score, tiers, common_days)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # --- d. Adaptive DD exit ---
    for base_dd in [0.03, 0.04]:
        for beta in [0.5, 0.7]:
            label = f"AdaptDD dd={base_dd:.0%} b={beta:.1f}"
            sdf = compute_adaptive_dd_exit(daily_ret, risk_score, base_dd, beta, 5, common_days)
            strat_dfs[label] = sdf
            variants.append(compute_strategy_metrics(sdf, label))

    # --- e. Hybrid (unsupervised exit + supervised re-entry) ---
    for base_dd in [0.02, 0.03, 0.04]:
        for beta in [0.5, 0.7]:
            label = f"Hybrid dd={base_dd:.0%} b={beta:.1f}"
            sdf = compute_hybrid_strategy(
                daily_ret, risk_score, sup_preds, sup_p_bull,
                base_dd, beta,
                bull_consec_required=3, p_bull_threshold=0.40,
                reentry_days_max=30, index=common_days,
            )
            strat_dfs[label] = sdf
            variants.append(compute_strategy_metrics(sdf, label))

    # --- f. Hybrid + tiers ---
    tier_default = [(0.50, 1.0), (0.75, 0.50), (float("inf"), 0.0)]
    for base_dd in [0.03, 0.04]:
        for beta in [0.5, 0.7]:
            label = f"HybTier dd={base_dd:.0%} b={beta:.1f}"
            sdf = compute_hybrid_tier_strategy(
                daily_ret, risk_score, sup_preds, sup_p_bull,
                tier_default, base_dd, beta,
                bull_consec_required=3, p_bull_threshold=0.40,
                reentry_days_max=30, index=common_days,
            )
            strat_dfs[label] = sdf
            variants.append(compute_strategy_metrics(sdf, label))

    return variants, strat_dfs


# ─────────────────────────────────────────────────────────────────────
# Part 4: Full instrument pipeline
# ─────────────────────────────────────────────────────────────────────

def run_single_instrument(
    symbol: str,
    n_folds: int,
    min_train_days: int,
) -> tuple[list[dict], dict[str, pd.DataFrame], pd.DataFrame]:
    """Run full pipeline for one instrument.

    Returns (variants, strat_dfs, daily).
    """
    print(f"\n{'='*60}")
    print(f"INSTRUMENT: {symbol}")
    print("=" * 60)

    # Part 0: Data
    print(f"\n  Part 0: Data pipeline")
    df, daily, feature_cols = load_instrument_pipeline(symbol, n_folds, min_train_days)

    n_days = len(sorted(df.index.unique()))
    if n_days < min_train_days + n_folds:
        print(f"  ERROR: Not enough days ({n_days}) for {n_folds} folds "
              f"with min_train={min_train_days}. Skipping {symbol}.")
        return [], {}, daily

    # Part 1: Unsupervised risk
    print(f"\n  Part 1: Unsupervised risk score")
    risk_df = compute_unsupervised_risk(
        df, daily, feature_cols, n_folds, min_train_days,
    )
    print(f"    Risk score: mean={risk_df['risk_score'].mean():.3f}, "
          f"std={risk_df['risk_score'].std():.3f}, "
          f"p90={risk_df['risk_score'].quantile(0.90):.3f}")

    # Part 2: Supervised BULL model
    print(f"\n  Part 2: Supervised BULL model")
    pred_df = train_supervised_model(
        df, daily, feature_cols, n_folds, min_train_days,
    )

    # Part 3: Strategies
    print(f"\n  Part 3: Strategy variants")
    variants, strat_dfs = run_strategies_for_instrument(
        daily, risk_df, pred_df, symbol,
    )

    # Print summary table
    if variants:
        print(f"\n  {'Label':<35s} {'TotRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
              f"{'Calmar':>7s} {'InMkt':>6s}")
        print("  " + "-" * 75)
        for v in sorted(variants, key=lambda x: -x["sharpe"]):
            print(f"  {v['label']:<35s} {v['total_return']:>7.1%} {v['sharpe']:>7.2f} "
                  f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['time_in_market_pct']:>5.1f}%")

    return variants, strat_dfs, daily


# ─────────────────────────────────────────────────────────────────────
# Part 5: Multi-instrument comparison + figures
# ─────────────────────────────────────────────────────────────────────

def print_cross_instrument_comparison(all_results: dict[str, list[dict]]):
    """Print cross-instrument strategy comparison."""
    print(f"\n{'='*60}")
    print("CROSS-INSTRUMENT COMPARISON")
    print("=" * 60)

    # Collect strategy labels across all instruments
    all_labels = set()
    for variants in all_results.values():
        for v in variants:
            all_labels.add(v["label"])

    # For each strategy, compute median Sharpe across instruments
    strategy_summary = []
    for label in sorted(all_labels):
        sharpes = []
        max_dds = []
        for sym, variants in all_results.items():
            v = next((x for x in variants if x["label"] == label), None)
            if v:
                sharpes.append(v["sharpe"])
                max_dds.append(v["max_dd"])
        if sharpes:
            strategy_summary.append({
                "label": label,
                "n_instruments": len(sharpes),
                "median_sharpe": np.median(sharpes),
                "mean_sharpe": np.mean(sharpes),
                "min_sharpe": min(sharpes),
                "max_sharpe": max(sharpes),
                "median_maxdd": np.median(max_dds),
            })

    strategy_summary.sort(key=lambda x: -x["median_sharpe"])

    print(f"\n  {'Strategy':<35s} {'#Inst':>5s} {'MedSharpe':>10s} {'MeanSharpe':>11s} "
          f"{'MinSharpe':>10s} {'MaxSharpe':>10s} {'MedMaxDD':>9s}")
    print("  " + "-" * 95)
    for s in strategy_summary[:20]:
        print(f"  {s['label']:<35s} {s['n_instruments']:>5d} {s['median_sharpe']:>10.2f} "
              f"{s['mean_sharpe']:>11.2f} {s['min_sharpe']:>10.2f} {s['max_sharpe']:>10.2f} "
              f"{s['median_maxdd']:>8.1%}")

    # Per-instrument best strategy
    print(f"\n  Per-instrument best strategy:")
    for sym, variants in sorted(all_results.items()):
        if not variants:
            continue
        best = max(variants, key=lambda x: x["sharpe"])
        bh = next((x for x in variants if x["label"] == "Buy&Hold"), None)
        bh_sharpe = bh["sharpe"] if bh else 0
        beat = best["sharpe"] > bh_sharpe
        print(f"    {sym}: {best['label']:<35s} Sharpe={best['sharpe']:.2f} "
              f"(B&H={bh_sharpe:.2f}) {'> B&H' if beat else '< B&H'}")

    # Count instruments where hybrid beats B&H
    hybrid_labels = [l for l in all_labels if l.startswith("Hybrid") or l.startswith("HybTier")]
    if hybrid_labels:
        print(f"\n  Hybrid generalization:")
        for label in sorted(hybrid_labels):
            beats_bh = 0
            total = 0
            for sym, variants in all_results.items():
                v = next((x for x in variants if x["label"] == label), None)
                bh = next((x for x in variants if x["label"] == "Buy&Hold"), None)
                if v and bh:
                    total += 1
                    if v["sharpe"] > bh["sharpe"]:
                        beats_bh += 1
            if total > 0:
                print(f"    {label:<35s}: beats B&H on {beats_bh}/{total} instruments")

    return strategy_summary


def plot_figures(
    all_results: dict[str, list[dict]],
    all_strat_dfs: dict[str, dict[str, pd.DataFrame]],
    all_dailys: dict[str, pd.DataFrame],
    skip_plots: bool = False,
):
    """Generate summary figures."""
    if skip_plots:
        print("\n  Skipping plots (--skip-plots)")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Per-instrument equity curves (best hybrid vs B&H vs DD-only)
    instruments = sorted(all_results.keys())
    n_inst = len(instruments)

    if n_inst > 0:
        fig, axes = plt.subplots(
            min(n_inst, 3), max(1, (n_inst + 2) // 3),
            figsize=(6 * max(1, (n_inst + 2) // 3), 5 * min(n_inst, 3)),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for idx, sym in enumerate(instruments):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            variants = all_results[sym]
            strat_dfs = all_strat_dfs.get(sym, {})

            # B&H
            if "Buy&Hold" in strat_dfs:
                sdf = strat_dfs["Buy&Hold"]
                ax.plot(sdf.index, sdf["cum_return"] * 100, label="B&H", color="gray", alpha=0.7)

            # DD-only 2%
            if "DD-only 2%" in strat_dfs:
                sdf = strat_dfs["DD-only 2%"]
                ax.plot(sdf.index, sdf["cum_return"] * 100, label="DD-only 2%",
                        color="C0", linewidth=1)

            # Best hybrid
            hybrid_variants = [v for v in variants
                               if v["label"].startswith("Hybrid") or v["label"].startswith("HybTier")]
            if hybrid_variants:
                best_hyb = max(hybrid_variants, key=lambda x: x["sharpe"])
                if best_hyb["label"] in strat_dfs:
                    sdf = strat_dfs[best_hyb["label"]]
                    ax.plot(sdf.index, sdf["cum_return"] * 100,
                            label=f'{best_hyb["label"]} (S={best_hyb["sharpe"]:.2f})',
                            color="C3", linewidth=1.5)

            # Best adaptive DD
            adapt_variants = [v for v in variants if v["label"].startswith("AdaptDD")]
            if adapt_variants:
                best_adapt = max(adapt_variants, key=lambda x: x["sharpe"])
                if best_adapt["label"] in strat_dfs:
                    sdf = strat_dfs[best_adapt["label"]]
                    ax.plot(sdf.index, sdf["cum_return"] * 100,
                            label=f'{best_adapt["label"]} (S={best_adapt["sharpe"]:.2f})',
                            color="C2", linewidth=1, linestyle="--")

            ax.set_title(f"{sym}", fontsize=11)
            ax.set_ylabel("Cum Return (%)")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_inst, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Equity Curves: Hybrid vs Baselines", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "equity_curves_per_instrument.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {FIGURES_DIR / 'equity_curves_per_instrument.png'}")

    # Figure 2: Cross-instrument heatmap (strategy × instrument → Sharpe)
    if n_inst > 1:
        # Collect key strategies
        key_prefixes = ["Buy&Hold", "DD-only 2%", "DD-only 3%", "AdaptDD", "Hybrid", "HybTier", "Tiers"]
        key_labels = set()
        for variants in all_results.values():
            for v in variants:
                for prefix in key_prefixes:
                    if v["label"].startswith(prefix):
                        key_labels.add(v["label"])

        key_labels = sorted(key_labels)
        if len(key_labels) > 15:
            # Keep only best by median sharpe
            label_sharpes = {}
            for label in key_labels:
                sharpes = []
                for variants in all_results.values():
                    v = next((x for x in variants if x["label"] == label), None)
                    if v:
                        sharpes.append(v["sharpe"])
                if sharpes:
                    label_sharpes[label] = np.median(sharpes)
            key_labels = sorted(label_sharpes, key=lambda x: -label_sharpes[x])[:15]

        heatmap_data = np.full((len(key_labels), n_inst), np.nan)
        for j, sym in enumerate(instruments):
            for i, label in enumerate(key_labels):
                v = next((x for x in all_results[sym] if x["label"] == label), None)
                if v:
                    heatmap_data[i, j] = v["sharpe"]

        fig, ax = plt.subplots(figsize=(max(8, n_inst * 1.5), max(6, len(key_labels) * 0.4)))
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.5)

        ax.set_xticks(range(n_inst))
        ax.set_xticklabels(instruments, fontsize=9)
        ax.set_yticks(range(len(key_labels)))
        ax.set_yticklabels([l[:30] for l in key_labels], fontsize=8)

        # Annotate cells
        for i in range(len(key_labels)):
            for j in range(n_inst):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, label="Sharpe")
        ax.set_title("Strategy x Instrument Sharpe Heatmap")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cross_instrument_heatmap.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: {FIGURES_DIR / 'cross_instrument_heatmap.png'}")


def print_final_summary(
    all_results: dict[str, list[dict]],
    strategy_summary: list[dict],
):
    """Print final success criteria check."""
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    # ES-specific checks
    es_variants = all_results.get("ES", [])
    if es_variants:
        bh = next((v for v in es_variants if v["label"] == "Buy&Hold"), None)
        dd2 = next((v for v in es_variants if v["label"] == "DD-only 2%"), None)

        hybrid_variants = [v for v in es_variants
                           if v["label"].startswith("Hybrid") or v["label"].startswith("HybTier")]
        best_hybrid = max(hybrid_variants, key=lambda x: x["sharpe"]) if hybrid_variants else None

        adapt_variants = [v for v in es_variants if v["label"].startswith("AdaptDD")]
        best_adapt = max(adapt_variants, key=lambda x: x["sharpe"]) if adapt_variants else None

        tier_variants = [v for v in es_variants if v["label"].startswith("Tiers")]
        best_tier = max(tier_variants, key=lambda x: x["sharpe"]) if tier_variants else None

        print(f"\n  ES Results:")
        for v in [bh, dd2, best_adapt, best_tier, best_hybrid]:
            if v:
                print(f"    {v['label']:<35s}: Sharpe={v['sharpe']:.2f}, "
                      f"MaxDD={v['max_dd']:.1%}, Calmar={v['calmar']:.2f}, "
                      f"InMkt={v['time_in_market_pct']:.0f}%")

        print(f"\n  Success Criteria (ES):")

        # 1. Hybrid Sharpe > 0.85 (beat AdaptDD)
        if best_hybrid:
            passed = best_hybrid["sharpe"] > 0.85
            print(f"    [{'PASS' if passed else 'FAIL'}] Hybrid Sharpe > 0.85: "
                  f"{best_hybrid['sharpe']:.2f}")

        # 2. Hybrid MaxDD better than DD-only 2%
        if best_hybrid and dd2:
            passed = abs(best_hybrid["max_dd"]) < abs(dd2["max_dd"])
            print(f"    [{'PASS' if passed else 'FAIL'}] Hybrid MaxDD < DD-only 2% "
                  f"({dd2['max_dd']:.1%}): {best_hybrid['max_dd']:.1%}")

        # 3. Discrete tiers preserve more return than continuous at similar DD
        if best_tier and best_adapt:
            better_ret = best_tier["total_return"] > best_adapt["total_return"] * 0.9
            similar_dd = abs(best_tier["max_dd"]) < abs(best_adapt["max_dd"]) * 1.2
            passed = better_ret and similar_dd
            print(f"    [{'PASS' if passed else 'FAIL'}] Discrete tiers preserve return: "
                  f"tier={best_tier['total_return']:.1%} vs adapt={best_adapt['total_return']:.1%}")

    # Multi-instrument generalization
    if len(all_results) > 1:
        # Find best hybrid label across instruments
        hybrid_labels = set()
        for variants in all_results.values():
            for v in variants:
                if v["label"].startswith("Hybrid") or v["label"].startswith("HybTier"):
                    hybrid_labels.add(v["label"])

        best_generalization = None
        best_beats = 0
        for label in hybrid_labels:
            beats = 0
            total = 0
            for sym, variants in all_results.items():
                v = next((x for x in variants if x["label"] == label), None)
                bh = next((x for x in variants if x["label"] == "Buy&Hold"), None)
                if v and bh:
                    total += 1
                    if v["sharpe"] > bh["sharpe"]:
                        beats += 1
            if beats > best_beats:
                best_beats = beats
                best_generalization = (label, beats, total)

        if best_generalization:
            label, beats, total = best_generalization
            passed = beats >= 3
            print(f"\n    [{'PASS' if passed else 'FAIL'}] Generalization ({label}): "
                  f"beats B&H on {beats}/{total} instruments (target: >=3)")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()

    n_folds = args.folds if not args.quick else 2
    min_train = args.min_train if not args.quick else 200

    instruments = args.instruments
    print(f"Instruments: {instruments}")
    print(f"Config: folds={n_folds}, min_train={min_train}, quick={args.quick}")

    # Validate instruments
    valid_instruments = []
    for sym in instruments:
        if sym not in ALL_INSTRUMENTS:
            print(f"  WARNING: Unknown instrument '{sym}', skipping")
        elif not ALL_INSTRUMENTS[sym].path.exists():
            print(f"  WARNING: {sym} data not found at {ALL_INSTRUMENTS[sym].path}, skipping")
        else:
            valid_instruments.append(sym)

    if not valid_instruments:
        print("ERROR: No valid instruments found")
        sys.exit(1)

    # Run each instrument
    all_results = {}
    all_strat_dfs = {}
    all_dailys = {}

    for sym in valid_instruments:
        try:
            variants, strat_dfs, daily = run_single_instrument(sym, n_folds, min_train)
            all_results[sym] = variants
            all_strat_dfs[sym] = strat_dfs
            all_dailys[sym] = daily
        except Exception as e:
            print(f"  ERROR running {sym}: {e}")
            import traceback
            traceback.print_exc()
            all_results[sym] = []
            all_strat_dfs[sym] = {}

    # Multi-instrument comparison
    active_results = {k: v for k, v in all_results.items() if v}
    strategy_summary = []
    if len(active_results) > 1:
        strategy_summary = print_cross_instrument_comparison(active_results)

    # Figures
    print(f"\n{'='*60}")
    print("FIGURES")
    print("=" * 60)
    plot_figures(active_results, all_strat_dfs, all_dailys, skip_plots=args.skip_plots)

    # Final summary
    print_final_summary(active_results, strategy_summary)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
