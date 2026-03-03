"""Regime classifier analysis & filtered swing trading strategy.

Runs the walk-forward regime pipeline, then performs 4-part analysis:
  Part 1: Classifier diagnostics (durations, transitions, calibration, confidence)
  Part 2: Filtered swing strategy (confidence + persistence filtering)
  Part 3: Large move capture analysis
  Part 4: Figures

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_regime.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_regime.py --skip-training
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_regime.py --es-only
    PYTHONPATH=/home/william/gybcap python -u sandbox/analyze_regime.py --skip-plots
"""
import argparse
import sys
import time
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from strategies.swing.config import SwingConfig
from strategies.swing.data_loader import load_instruments
from strategies.swing.daily_aggregator import DailyAggregator, align_daily
from strategies.swing.labeling.hmm_regime import (
    compute_hmm_features_walkforward,
)
from strategies.swing.pipeline import build_training_frame
from strategies.swing.training.regime_trainer import (
    walk_forward_cv, TrainerResult, FoldResult,
)


FIGURES_DIR = Path("sandbox/figures/regime_analysis")
CACHE_PATH = Path("strategies/data/regime_oos_cache.npz")
CLASS_NAMES = {0: "BEAR", 1: "BALANCE", 2: "BULL"}


# ─────────────────────────────────────────────────────────────────────
# Part 0: Pipeline + OOS DataFrame
# ─────────────────────────────────────────────────────────────────────

def run_regime_pipeline(args) -> tuple[pd.DataFrame, dict, list[str]]:
    """Replicate train_regime.py pipeline, return (es_daily, results_dict, feature_cols)."""
    config = SwingConfig(
        n_folds=args.folds,
        min_train_days=args.min_train,
        detect_threshold=args.detect_threshold,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
    )

    # Load data
    print("=" * 60)
    print("STEP 1: Loading instrument data")
    print("=" * 60)
    symbols = ["ES"]
    if not args.es_only:
        symbols += config.correlation_instruments
    minute_data = load_instruments(symbols)
    if "ES" not in minute_data:
        print("ERROR: ES data not found")
        sys.exit(1)

    # Aggregate to daily
    print(f"\n{'='*60}")
    print("STEP 2: Aggregating to daily bars")
    print("=" * 60)
    aggregator = DailyAggregator()
    daily_data = {}
    for sym, minute_df in minute_data.items():
        compute_vp = (sym == "ES")
        print(f"  Aggregating {sym} (VP={'yes' if compute_vp else 'no'})...")
        daily_data[sym] = aggregator.aggregate(minute_df, compute_vp=compute_vp)
        print(f"    {len(daily_data[sym])} trading days")

    if len(daily_data) > 1:
        daily_data = align_daily(daily_data, primary="ES")

    es_daily = daily_data["ES"]
    print(f"\nES daily: {len(es_daily)} days, {es_daily.index.min().date()} – {es_daily.index.max().date()}")

    # Compute features
    print(f"\n{'='*60}")
    print("STEP 3: Computing features")
    print("=" * 60)
    print("  Building shared feature pipeline...")
    other_dailys = [(sym, df) for sym, df in daily_data.items() if sym != "ES"] if len(daily_data) > 1 else None
    artifacts = build_training_frame(
        es_daily=es_daily,
        other_dailys=None if args.es_only else other_dailys,
        config=config,
        include_vp=True,
        include_external=not args.no_external,
        include_range=True,
        use_other_dailys_for_macro=True,
    )
    feature_cols = artifacts.feature_cols
    labels = artifacts.labels
    df = artifacts.df
    feature_groups = artifacts.feature_groups

    print(f"    {len(feature_groups['technical'])} technical features")
    if feature_groups["volume_profile"]:
        print(f"    {len(feature_groups['volume_profile'])} VP features")
    else:
        print("  Skipping VP features")
    if feature_groups["cross"]:
        print(f"    {len(feature_groups['cross'])} cross-instrument features")
    else:
        print("  Skipping cross-instrument features")
    print(f"    {len(feature_groups['macro'])} macro context features")
    if args.no_external:
        print("  Skipping external features")
    elif feature_groups["external"]:
        print(f"    {len(feature_groups['external'])} external features")
    else:
        print("  External features unavailable")
    print(f"\n  Total features: {len(feature_cols)}")

    # Compute labels
    print(f"\n{'='*60}")
    print("STEP 4: Computing regime labels")
    print("=" * 60)
    # Structural label diagnostics
    if "y_structural" in labels.columns:
        struct = labels["y_structural"]
        groups = (struct != struct.shift()).cumsum()
        runs = struct.groupby(groups).agg(["first", "count"])
        runs.columns = ["regime", "duration"]
        n_transitions = len(runs) - 1
        years = len(labels) / 252
        print(f"\n  Structural label: {n_transitions} transitions ({n_transitions/years:.1f}/year)")
        for cls in [0, 1, 2]:
            d = runs.loc[runs["regime"] == cls, "duration"]
            if len(d) > 0:
                name = {0: "BEAR", 1: "BALANCE", 2: "BULL"}[cls]
                print(f"    {name}: {len(d)} runs, mean={d.mean():.1f}d, "
                      f"min={d.min():.0f}d, max={d.max():.0f}d")

    # Walk-forward training
    print(f"\n{'='*60}")
    print("STEP 5: Walk-forward training")
    print("=" * 60)

    def hmm_fn(df_fold, train_end_idx):
        return compute_hmm_features_walkforward(
            es_daily, train_end_idx, n_states=config.hmm_n_states
        )

    # Train selected target
    target = args.target
    xgb_params = {"bear_upweight": args.bear_upweight}
    print(f"  BEAR upweight: {args.bear_upweight}x")

    results = {}
    result = walk_forward_cv(
        df=df, feature_cols=feature_cols, target_col=target,
        n_folds=config.n_folds, min_train_days=config.min_train_days,
        params=xgb_params, hmm_fn=hmm_fn, verbose=True,
    )
    results[target] = result

    return es_daily, results, feature_cols


def build_oos_dataframe(result: TrainerResult, daily: pd.DataFrame) -> pd.DataFrame:
    """Map OOS arrays back to dated DataFrame."""
    days = sorted(daily.index.unique())
    n_days = len(days)
    preds = result.oos_predictions
    actuals = result.oos_actuals
    probas = result.oos_probas

    oos_days = days[n_days - len(preds):]

    oos_df = pd.DataFrame(index=oos_days)
    oos_df["pred"] = preds
    oos_df["actual"] = actuals[:len(oos_days)]
    oos_df["p_bear"] = probas[:len(oos_days), 0]
    oos_df["p_balance"] = probas[:len(oos_days), 1]
    oos_df["p_bull"] = probas[:len(oos_days), 2]
    oos_df["max_proba"] = probas[:len(oos_days)].max(axis=1)
    oos_df["close"] = daily["close"].reindex(oos_days).values
    oos_df["daily_return"] = daily["close"].pct_change().reindex(oos_days).fillna(0).values
    return oos_df


def save_cache(oos_df: pd.DataFrame, result: TrainerResult, target_name: str = "y_micro"):
    """Cache OOS data to .npz for --skip-training."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dates_str = np.array([str(d.date()) for d in oos_df.index])
    # Save fold feature importances as a dict of arrays
    fold_imp_features = []
    fold_imp_values = []
    for fr in result.fold_results:
        if fr.feature_importances is not None:
            fold_imp_features.append(fr.feature_importances["feature"].values)
            fold_imp_values.append(fr.feature_importances["importance"].values)
    np.savez(
        CACHE_PATH,
        oos_probas=result.oos_probas,
        oos_predictions=result.oos_predictions,
        oos_actuals=result.oos_actuals,
        dates=dates_str,
        close=oos_df["close"].values,
        daily_return=oos_df["daily_return"].values,
        n_folds=len(result.fold_results),
        fold_imp_features=np.array(fold_imp_features, dtype=object),
        fold_imp_values=np.array(fold_imp_values, dtype=object),
        target_name=np.array(target_name),
    )
    print(f"  Cached OOS data to {CACHE_PATH} (target={target_name})")


def load_cache() -> tuple[pd.DataFrame, TrainerResult]:
    """Load cached OOS data."""
    data = np.load(CACHE_PATH, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    oos_df = pd.DataFrame(index=dates)
    probas = data["oos_probas"]
    oos_df["pred"] = data["oos_predictions"]
    oos_df["actual"] = data["oos_actuals"]
    oos_df["p_bear"] = probas[:, 0]
    oos_df["p_balance"] = probas[:, 1]
    oos_df["p_bull"] = probas[:, 2]
    oos_df["max_proba"] = probas.max(axis=1)
    oos_df["close"] = data["close"]
    oos_df["daily_return"] = data["daily_return"]

    # Reconstruct minimal TrainerResult with fold importances
    n_folds = int(data["n_folds"])
    fold_imp_features = data["fold_imp_features"]
    fold_imp_values = data["fold_imp_values"]
    fold_results = []
    for i in range(n_folds):
        imp_df = None
        if i < len(fold_imp_features):
            imp_df = pd.DataFrame({
                "feature": fold_imp_features[i],
                "importance": fold_imp_values[i],
            })
        fold_results.append(FoldResult(
            fold=i, train_days=0, test_days=0, train_samples=0, test_samples=0,
            accuracy=0, f1_macro=0, directional_accuracy=0,
            per_class_precision={}, per_class_recall={},
            feature_importances=imp_df,
        ))

    result = TrainerResult(
        target="y_micro", fold_results=fold_results,
        aggregate_accuracy=0, aggregate_f1=0, aggregate_directional_acc=0,
        fold_std=0,
        oos_predictions=data["oos_predictions"],
        oos_actuals=data["oos_actuals"],
        oos_probas=probas,
        class_names=["BEAR", "BALANCE", "BULL"],
    )
    return oos_df, result


# ─────────────────────────────────────────────────────────────────────
# Part 1: Classifier Diagnostics
# ─────────────────────────────────────────────────────────────────────

def analyze_regime_durations(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze contiguous regime run lengths for each actual class."""
    actual = oos_df["actual"].astype(int)
    groups = (actual != actual.shift()).cumsum()
    runs = actual.groupby(groups).agg(["first", "count"])
    runs.columns = ["regime", "duration"]

    print(f"\n{'='*60}")
    print("PART 1a: Regime Durations (actual)")
    print("=" * 60)
    stats_rows = []
    for cls in [0, 1, 2]:
        d = runs.loc[runs["regime"] == cls, "duration"]
        if len(d) == 0:
            continue
        row = {
            "regime": CLASS_NAMES[cls],
            "n_runs": len(d),
            "mean": d.mean(),
            "median": d.median(),
            "min": d.min(),
            "max": d.max(),
        }
        stats_rows.append(row)
        print(f"  {CLASS_NAMES[cls]:>8s}: {len(d):3d} runs, "
              f"mean={d.mean():.1f}d, median={d.median():.0f}d, "
              f"min={d.min():.0f}d, max={d.max():.0f}d")
    return runs


def analyze_transition_lag(oos_df: pd.DataFrame) -> pd.DataFrame:
    """How many days after an actual regime change does the model catch up?"""
    actual = oos_df["actual"].astype(int).values
    pred = oos_df["pred"].astype(int).values

    transitions = []
    for i in range(1, len(actual)):
        if actual[i] != actual[i - 1]:
            new_regime = actual[i]
            lag = None
            for j in range(i, len(pred)):
                if pred[j] == new_regime:
                    lag = j - i
                    break
            transitions.append({
                "day_idx": i,
                "from": CLASS_NAMES[actual[i - 1]],
                "to": CLASS_NAMES[new_regime],
                "lag": lag if lag is not None else len(pred) - i,
            })

    trans_df = pd.DataFrame(transitions)
    print(f"\n{'='*60}")
    print("PART 1b: Transition Lag")
    print("=" * 60)
    if len(trans_df) > 0:
        for (fr, to), grp in trans_df.groupby(["from", "to"]):
            lags = grp["lag"]
            print(f"  {fr:>8s} → {to:<8s}: n={len(lags):3d}, "
                  f"mean={lags.mean():.1f}d, median={lags.median():.0f}d")
    return trans_df


def analyze_transition_confusion(oos_df: pd.DataFrame) -> dict:
    """What does the model predict in the first N days after a regime transition?"""
    actual = oos_df["actual"].astype(int).values
    pred = oos_df["pred"].astype(int).values
    windows = [1, 3, 5, 10]

    print(f"\n{'='*60}")
    print("PART 1c: Transition Confusion (model predictions after transitions)")
    print("=" * 60)

    results = {}
    for i in range(1, len(actual)):
        if actual[i] != actual[i - 1]:
            key = f"{CLASS_NAMES[actual[i-1]]}→{CLASS_NAMES[actual[i]]}"
            if key not in results:
                results[key] = {w: [] for w in windows}
            for w in windows:
                end = min(i + w, len(pred))
                preds_window = pred[i:end]
                if len(preds_window) > 0:
                    results[key][w].append(preds_window)

    for trans_type, window_data in sorted(results.items()):
        print(f"\n  {trans_type}:")
        for w in windows:
            if not window_data[w]:
                continue
            all_preds = np.concatenate(window_data[w])
            dist = {CLASS_NAMES[c]: (all_preds == c).mean() for c in [0, 1, 2]}
            dist_str = ", ".join(f"{k}={v:.0%}" for k, v in dist.items())
            print(f"    First {w:2d}d: {dist_str}")

    return results


def analyze_calibration(oos_df: pd.DataFrame) -> dict:
    """Reliability diagram + ECE per class."""
    print(f"\n{'='*60}")
    print("PART 1d: Calibration")
    print("=" * 60)

    cal_data = {}
    for cls, name in CLASS_NAMES.items():
        p_col = ["p_bear", "p_balance", "p_bull"][cls]
        probs = oos_df[p_col].values
        actual_binary = (oos_df["actual"].values == cls).astype(float)

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []
        bin_counts = []

        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs < bin_edges[b + 1])
            if b == n_bins - 1:
                mask = (probs >= bin_edges[b]) & (probs <= bin_edges[b + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_freqs.append(actual_binary[mask].mean())
                bin_counts.append(mask.sum())

        # ECE
        ece = 0
        total = len(probs)
        for bc, bf, bn in zip(bin_centers, bin_freqs, bin_counts):
            ece += bn / total * abs(bf - bc)

        cal_data[name] = {
            "bin_centers": bin_centers,
            "bin_freqs": bin_freqs,
            "bin_counts": bin_counts,
            "ece": ece,
        }
        print(f"  {name}: ECE={ece:.4f}")

    return cal_data


def analyze_feature_importance_stability(result: TrainerResult) -> pd.DataFrame:
    """Spearman rank correlation of feature importances across folds."""
    print(f"\n{'='*60}")
    print("PART 1e: Feature Importance Stability")
    print("=" * 60)

    fold_imps = []
    for fr in result.fold_results:
        if fr.feature_importances is not None:
            imp = fr.feature_importances.set_index("feature")["importance"]
            fold_imps.append(imp)

    if len(fold_imps) < 2:
        print("  Not enough folds with feature importances")
        return pd.DataFrame()

    # Align features across folds
    all_features = sorted(set().union(*[set(fi.index) for fi in fold_imps]))
    imp_matrix = pd.DataFrame(index=all_features)
    for i, fi in enumerate(fold_imps):
        imp_matrix[f"fold_{i}"] = fi.reindex(all_features).fillna(0)

    # Rank correlations between fold pairs
    fold_cols = imp_matrix.columns
    corrs = []
    for i, j in combinations(range(len(fold_cols)), 2):
        rho, _ = spearmanr(imp_matrix[fold_cols[i]], imp_matrix[fold_cols[j]])
        corrs.append(rho)
        print(f"  Fold {i} vs {j}: Spearman rho={rho:.3f}")

    print(f"  Mean pairwise Spearman: {np.mean(corrs):.3f}")

    # Top 20 features by mean importance
    imp_matrix["mean_imp"] = imp_matrix[fold_cols].mean(axis=1)
    imp_matrix["std_imp"] = imp_matrix[fold_cols].std(axis=1)
    imp_matrix["cv"] = imp_matrix["std_imp"] / (imp_matrix["mean_imp"] + 1e-10)
    top20 = imp_matrix.sort_values("mean_imp", ascending=False).head(20)

    print(f"\n  Top 20 features (mean importance, CV):")
    for feat, row in top20.iterrows():
        print(f"    {feat:45s} imp={row['mean_imp']:.4f}  CV={row['cv']:.2f}")

    return imp_matrix


def analyze_confidence_distribution(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Histogram of max_proba + conditional accuracy at each threshold."""
    print(f"\n{'='*60}")
    print("PART 1f: Confidence Distribution & Conditional Accuracy")
    print("=" * 60)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    correct = (oos_df["pred"] == oos_df["actual"]).astype(int)

    rows = []
    for t in thresholds:
        mask = oos_df["max_proba"] >= t
        n = mask.sum()
        pct = n / len(oos_df) * 100
        acc = correct[mask].mean() if n > 0 else 0
        # Directional accuracy (exclude BALANCE actual)
        dir_mask = mask & (oos_df["actual"] != 1)
        dir_acc = correct[dir_mask].mean() if dir_mask.sum() > 0 else 0

        rows.append({
            "threshold": t, "n_days": n, "pct_dataset": pct,
            "accuracy": acc, "dir_accuracy": dir_acc,
        })
        print(f"  conf >= {t:.2f}: {n:4d} days ({pct:5.1f}%), "
              f"acc={acc:.3f}, dir_acc={dir_acc:.3f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Part 2: Filtered Swing Strategy
# ─────────────────────────────────────────────────────────────────────

def compute_filtered_strategy(
    oos_df: pd.DataFrame,
    confidence_threshold: float = 0.50,
    persistence_days: int = 0,
    proportional: bool = False,
) -> pd.DataFrame:
    """Compute filtered swing positions.

    Entry: long if p_bull > thresh AND N consecutive bull preds;
           short if p_bear > thresh AND N consecutive bear preds.
    Exit: confidence drops below threshold OR prediction flips.
    """
    n = len(oos_df)
    pred = oos_df["pred"].values.astype(int)
    p_bull = oos_df["p_bull"].values
    p_bear = oos_df["p_bear"].values

    # Count consecutive same predictions
    consec = np.ones(n, dtype=int)
    for i in range(1, n):
        if pred[i] == pred[i - 1]:
            consec[i] = consec[i - 1] + 1

    position = np.zeros(n)
    for i in range(n):
        if pred[i] == 2 and p_bull[i] > confidence_threshold and consec[i] >= persistence_days:
            if proportional:
                position[i] = (p_bull[i] - 0.33) / 0.67
            else:
                position[i] = 1.0
        elif pred[i] == 0 and p_bear[i] > confidence_threshold and consec[i] >= persistence_days:
            if proportional:
                position[i] = -(p_bear[i] - 0.33) / 0.67
            else:
                position[i] = -1.0

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    # Causal: position determined at close of day t, earns return on day t+1
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(0) * oos_df["daily_return"].values
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def compute_strategy_metrics(strat_df: pd.DataFrame, label: str) -> dict:
    """Compute comprehensive strategy metrics."""
    ret = strat_df["strategy_return"]
    pos = strat_df["position"]
    n_days = len(strat_df)
    years = n_days / 252

    # Trade count: a new trade starts when position changes sign (non-zero)
    pos_shifted = pos.shift(1).fillna(0)
    trade_changes = ((pos != 0) & (pos != pos_shifted)).sum()
    trades_per_year = trade_changes / years if years > 0 else 0

    # Time in market
    in_market_pct = (pos != 0).mean() * 100

    # Average hold time
    groups = (pos != pos.shift()).cumsum()
    non_zero_runs = pos[pos != 0].groupby(groups).count()
    avg_hold = non_zero_runs.mean() if len(non_zero_runs) > 0 else 0

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

    # Per-trade win rate
    trade_starts = (pos != pos_shifted) & (pos != 0)
    trade_ends = (pos != pos.shift(-1).fillna(0)) & (pos != 0)
    trade_returns = []
    start_idx = None
    for i in range(len(pos)):
        if trade_starts.iloc[i]:
            start_idx = i
        if trade_ends.iloc[i] and start_idx is not None:
            tr = ret.iloc[start_idx + 1: i + 2].sum() if start_idx + 1 < len(ret) else 0
            trade_returns.append(tr)
            start_idx = None
    win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0

    # Year-by-year
    yearly = {}
    strat_df_copy = strat_df.copy()
    strat_df_copy["year"] = strat_df_copy.index.year
    for yr, grp in strat_df_copy.groupby("year"):
        yr_ret = (1 + grp["strategy_return"]).prod() - 1
        yearly[yr] = yr_ret

    return {
        "label": label,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "trades_per_year": trades_per_year,
        "avg_hold_days": avg_hold,
        "win_rate": win_rate,
        "time_in_market_pct": in_market_pct,
        "n_trades": len(trade_returns),
        "yearly": yearly,
    }


def run_strategy_sweep(oos_df: pd.DataFrame) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run all strategy variants and collect metrics."""
    print(f"\n{'='*60}")
    print("PART 2: Filtered Swing Strategy Sweep")
    print("=" * 60)

    variants = []
    strat_dfs = {}

    # Buy-and-hold benchmark
    bh_df = pd.DataFrame(index=oos_df.index)
    bh_df["position"] = 1.0
    bh_df["strategy_return"] = oos_df["daily_return"].values
    bh_df["cum_return"] = (1 + bh_df["strategy_return"]).cumprod() - 1
    strat_dfs["buy_hold"] = bh_df
    variants.append(compute_strategy_metrics(bh_df, "Buy&Hold"))

    # Confidence-only sweeps
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        label = f"conf_{t:.2f}"
        sdf = compute_filtered_strategy(oos_df, confidence_threshold=t, persistence_days=1)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Persistence-only sweeps (conf=0.33 ≈ no filter)
    for p in [3, 5, 10]:
        label = f"persist_{p}d"
        sdf = compute_filtered_strategy(oos_df, confidence_threshold=0.33, persistence_days=p)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Combined sweeps
    for t, p in [(0.50, 3), (0.50, 5), (0.55, 3), (0.55, 5)]:
        label = f"conf_{t:.2f}_p{p}d"
        sdf = compute_filtered_strategy(oos_df, confidence_threshold=t, persistence_days=p)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Proportional variant with best confidence
    label = "conf_0.50_prop"
    sdf = compute_filtered_strategy(oos_df, confidence_threshold=0.50, proportional=True)
    strat_dfs[label] = sdf
    variants.append(compute_strategy_metrics(sdf, label))

    # Print summary table
    print(f"\n  {'Label':<22s} {'TotRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
          f"{'Calmar':>7s} {'Tr/Yr':>6s} {'Hold':>6s} {'WR':>6s} {'InMkt':>6s}")
    print("  " + "-" * 90)
    for v in variants:
        print(f"  {v['label']:<22s} {v['total_return']:>7.1%} {v['sharpe']:>7.2f} "
              f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['trades_per_year']:>6.1f} "
              f"{v['avg_hold_days']:>6.1f} {v['win_rate']:>5.0%} "
              f"{v['time_in_market_pct']:>5.1f}%")

    # Year-by-year for top variants
    print(f"\n  Year-by-year returns (top variants):")
    top_labels = ["buy_hold", "conf_0.50", "conf_0.55", "conf_0.50_p3d"]
    for v in variants:
        if v["label"].replace("&", "_").lower().replace("buy_hold", "buy_hold") in [
            tl.replace("&", "_").lower() for tl in top_labels
        ] or v["label"] in top_labels:
            yearly_str = "  ".join(f"{yr}:{r:+.1%}" for yr, r in sorted(v["yearly"].items()))
            print(f"    {v['label']:<22s}: {yearly_str}")

    return variants, strat_dfs


# ─────────────────────────────────────────────────────────────────────
# Part 2b: Long-Biased & Position Sizing Strategies
# ─────────────────────────────────────────────────────────────────────

def compute_long_biased_strategy(
    oos_df: pd.DataFrame,
    confidence_threshold: float = 0.50,
    go_short: bool = False,
) -> pd.DataFrame:
    """Default long. Only exit/short on confident BEAR.

    ES has strong upward bias — the model only needs to detect bear periods.
    """
    n = len(oos_df)
    pred = oos_df["pred"].values.astype(int)
    p_bear = oos_df["p_bear"].values

    position = np.ones(n)  # always long by default
    for i in range(n):
        if pred[i] == 0 and p_bear[i] > confidence_threshold:
            position[i] = -1.0 if go_short else 0.0

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(1.0) * oos_df["daily_return"].values
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def compute_sized_strategy(
    oos_df: pd.DataFrame,
    mode: str = "linear",
    threshold: float = 0.50,
    max_leverage: float = 2.0,
    base_size: float = 1.0,
    bear_size: float = 0.0,
    tiers: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Confidence-based position sizing.

    Modes:
      'linear': Scale position linearly with confidence.
      'tiered': Discrete leverage tiers based on confidence.
      'long_sized': Long-biased with sizing up on BULL, down on BEAR.
    """
    n = len(oos_df)
    pred = oos_df["pred"].values.astype(int)
    p_bear = oos_df["p_bear"].values
    p_bull = oos_df["p_bull"].values

    position = np.ones(n) * base_size

    if mode == "linear":
        for i in range(n):
            if pred[i] == 2 and p_bull[i] > threshold:
                # Scale up long
                size = 1.0 + (p_bull[i] - threshold) / (1.0 - threshold) * (max_leverage - 1.0)
                position[i] = size
            elif pred[i] == 0 and p_bear[i] > threshold:
                # Scale up short
                size = 1.0 + (p_bear[i] - threshold) / (1.0 - threshold) * (max_leverage - 1.0)
                position[i] = -size
            else:
                position[i] = base_size

    elif mode == "tiered":
        if tiers is None:
            tiers = [(0.70, 2.0), (0.60, 1.5), (0.50, 1.0)]
        for i in range(n):
            if pred[i] == 2:
                for min_conf, lev in sorted(tiers, reverse=True):
                    if p_bull[i] >= min_conf:
                        position[i] = lev
                        break
                else:
                    position[i] = base_size
            elif pred[i] == 0:
                for min_conf, lev in sorted(tiers, reverse=True):
                    if p_bear[i] >= min_conf:
                        position[i] = -lev
                        break
                else:
                    position[i] = base_size

    elif mode == "long_sized":
        bull_threshold = threshold
        bear_threshold = threshold
        for i in range(n):
            if pred[i] == 2 and p_bull[i] > bull_threshold:
                # Size up from base
                position[i] = base_size + (p_bull[i] - bull_threshold) / (1.0 - bull_threshold) * (max_leverage - base_size)
            elif pred[i] == 0 and p_bear[i] > bear_threshold:
                position[i] = -bear_size
            else:
                position[i] = base_size

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(base_size) * oos_df["daily_return"].values
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def compute_drawdown_exit_strategy(
    oos_df: pd.DataFrame,
    dd_threshold: float = -0.03,
    reentry_days: int = 5,
) -> pd.DataFrame:
    """Long-biased with mechanical drawdown stop.

    Always long. Exit to flat if equity drops dd_threshold from peak.
    Wait reentry_days before re-entering long.
    """
    n = len(oos_df)
    daily_ret = oos_df["daily_return"].values
    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    flat_counter = 0

    for i in range(1, n):
        # Apply previous position's return
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        if flat_counter > 0:
            # Waiting to re-enter
            position[i] = 0.0
            flat_counter -= 1
            if flat_counter == 0:
                # Re-enter and reset peak
                peak_equity = equity
        elif (equity - peak_equity) / peak_equity < dd_threshold:
            # Drawdown triggered — exit
            position[i] = 0.0
            flat_counter = reentry_days
        else:
            position[i] = 1.0

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(1.0) * daily_ret
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def compute_asymmetric_strategy(
    oos_df: pd.DataFrame,
    bear_exit_days: int = 1,
    bull_entry_days: int = 3,
    confidence_threshold: float = 0.50,
) -> pd.DataFrame:
    """Asymmetric: fast exit on BEAR signal, slow re-entry on BULL.

    Default long. Exit on bear_exit_days consecutive BEAR predictions
    above confidence threshold. Re-enter only after bull_entry_days
    consecutive BULL predictions.
    """
    n = len(oos_df)
    pred = oos_df["pred"].values.astype(int)
    p_bear = oos_df["p_bear"].values
    p_bull = oos_df["p_bull"].values
    daily_ret = oos_df["daily_return"].values

    position = np.ones(n)
    in_market = True
    bear_consec = 0
    bull_consec = 0

    for i in range(n):
        # Count consecutive predictions
        if pred[i] == 0 and p_bear[i] > confidence_threshold:
            bear_consec += 1
            bull_consec = 0
        elif pred[i] == 2 and p_bull[i] > confidence_threshold:
            bull_consec += 1
            bear_consec = 0
        else:
            bear_consec = 0
            bull_consec = 0

        if in_market:
            if bear_consec >= bear_exit_days:
                in_market = False
                position[i] = 0.0
            else:
                position[i] = 1.0
        else:
            if bull_consec >= bull_entry_days:
                in_market = True
                position[i] = 1.0
            else:
                position[i] = 0.0

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(1.0) * daily_ret
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def compute_combined_strategy(
    oos_df: pd.DataFrame,
    dd_threshold: float = -0.03,
    confidence_threshold: float = 0.50,
    bear_exit_days: int = 1,
    bull_entry_days: int = 3,
    reentry_days: int = 5,
) -> pd.DataFrame:
    """Combined drawdown stop + asymmetric regime filter.

    Long by default with two exit triggers:
    1. Model predicts BEAR with confidence > threshold → exit
    2. Trailing drawdown hits dd_threshold → exit (catches what model misses)

    Re-enter after bull_entry_days consecutive BULL or reentry_days after DD exit.
    """
    n = len(oos_df)
    pred = oos_df["pred"].values.astype(int)
    p_bear = oos_df["p_bear"].values
    p_bull = oos_df["p_bull"].values
    daily_ret = oos_df["daily_return"].values

    position = np.ones(n)
    equity = 1.0
    peak_equity = 1.0
    in_market = True
    exit_reason = None  # 'model' or 'drawdown'
    flat_counter = 0
    bear_consec = 0
    bull_consec = 0

    for i in range(1, n):
        equity *= (1 + position[i - 1] * daily_ret[i])
        peak_equity = max(peak_equity, equity)

        # Count consecutive predictions
        if pred[i] == 0 and p_bear[i] > confidence_threshold:
            bear_consec += 1
            bull_consec = 0
        elif pred[i] == 2 and p_bull[i] > confidence_threshold:
            bull_consec += 1
            bear_consec = 0
        else:
            bear_consec = 0
            bull_consec = 0

        if in_market:
            # Check model exit
            if bear_consec >= bear_exit_days:
                in_market = False
                exit_reason = "model"
                position[i] = 0.0
            # Check drawdown exit
            elif (equity - peak_equity) / peak_equity < dd_threshold:
                in_market = False
                exit_reason = "drawdown"
                flat_counter = reentry_days
                position[i] = 0.0
            else:
                position[i] = 1.0
        else:
            if exit_reason == "drawdown":
                flat_counter -= 1
                if flat_counter <= 0 and bull_consec >= 1:
                    in_market = True
                    peak_equity = equity
                    position[i] = 1.0
                else:
                    position[i] = 0.0
            else:  # model exit
                if bull_consec >= bull_entry_days:
                    in_market = True
                    peak_equity = equity
                    position[i] = 1.0
                elif (equity - peak_equity) / peak_equity < dd_threshold:
                    # Also check DD while waiting for model re-entry
                    flat_counter = reentry_days
                    exit_reason = "drawdown"
                    position[i] = 0.0
                else:
                    position[i] = 0.0

    strat_df = pd.DataFrame(index=oos_df.index)
    strat_df["position"] = position
    strat_df["strategy_return"] = pd.Series(position, index=oos_df.index).shift(1).fillna(1.0) * daily_ret
    strat_df["cum_return"] = (1 + strat_df["strategy_return"]).cumprod() - 1
    return strat_df


def run_extended_strategy_sweep(
    oos_df: pd.DataFrame,
    existing_variants: list[dict],
    existing_strat_dfs: dict[str, pd.DataFrame],
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """Run long-biased and position sizing strategy variants."""
    print(f"\n{'='*60}")
    print("PART 2b: Long-Biased & Position Sizing Strategies")
    print("=" * 60)

    variants = list(existing_variants)
    strat_dfs = dict(existing_strat_dfs)

    # Long-biased flat on BEAR
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        label = f"long_biased_flat_{t:.2f}"
        sdf = compute_long_biased_strategy(oos_df, confidence_threshold=t, go_short=False)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Long-biased short on BEAR
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        label = f"long_biased_short_{t:.2f}"
        sdf = compute_long_biased_strategy(oos_df, confidence_threshold=t, go_short=True)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Linear sizing
    for max_lev in [1.5, 2.0]:
        for t in [0.50, 0.55, 0.60]:
            label = f"linear_{max_lev:.1f}x_{t:.2f}"
            sdf = compute_sized_strategy(oos_df, mode="linear", threshold=t, max_leverage=max_lev)
            strat_dfs[label] = sdf
            variants.append(compute_strategy_metrics(sdf, label))

    # Long-sized
    for max_lev in [1.5, 2.0]:
        label = f"long_sized_{max_lev:.1f}x_0.50"
        sdf = compute_sized_strategy(oos_df, mode="long_sized", threshold=0.50,
                                      max_leverage=max_lev, base_size=1.0, bear_size=0.0)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Tiered
    label = "tiered_0.50"
    sdf = compute_sized_strategy(oos_df, mode="tiered", threshold=0.50,
                                  tiers=[(0.70, 2.0), (0.60, 1.5), (0.50, 1.0)])
    strat_dfs[label] = sdf
    variants.append(compute_strategy_metrics(sdf, label))

    # Drawdown exit strategies
    for dd_pct in [2, 3, 4, 5]:
        label = f"dd_exit_{dd_pct}pct"
        sdf = compute_drawdown_exit_strategy(oos_df, dd_threshold=-dd_pct / 100.0, reentry_days=5)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Asymmetric strategies
    for exit_d, entry_d in [(1, 3), (1, 5), (2, 3), (2, 5)]:
        label = f"asym_{exit_d}d_exit_{entry_d}d_entry"
        sdf = compute_asymmetric_strategy(oos_df, bear_exit_days=exit_d,
                                           bull_entry_days=entry_d, confidence_threshold=0.50)
        strat_dfs[label] = sdf
        variants.append(compute_strategy_metrics(sdf, label))

    # Combined strategies
    for dd_pct in [3, 4]:
        for conf in [0.45, 0.50]:
            label = f"combined_dd{dd_pct}pct_conf{conf:.2f}"
            sdf = compute_combined_strategy(
                oos_df, dd_threshold=-dd_pct / 100.0,
                confidence_threshold=conf, bear_exit_days=1,
                bull_entry_days=3, reentry_days=5,
            )
            strat_dfs[label] = sdf
            variants.append(compute_strategy_metrics(sdf, label))

    # Print extended summary
    new_variants = variants[len(existing_variants):]
    print(f"\n  {'Label':<28s} {'TotRet':>8s} {'Sharpe':>7s} {'MaxDD':>8s} "
          f"{'Calmar':>7s} {'Tr/Yr':>6s} {'Hold':>6s} {'WR':>6s} {'InMkt':>6s}")
    print("  " + "-" * 95)
    for v in new_variants:
        print(f"  {v['label']:<28s} {v['total_return']:>7.1%} {v['sharpe']:>7.2f} "
              f"{v['max_dd']:>7.1%} {v['calmar']:>7.2f} {v['trades_per_year']:>6.1f} "
              f"{v['avg_hold_days']:>6.1f} {v['win_rate']:>5.0%} "
              f"{v['time_in_market_pct']:>5.1f}%")

    return variants, strat_dfs


def analyze_expected_value_by_confidence(oos_df: pd.DataFrame) -> pd.DataFrame:
    """Bin OOS predictions by confidence and compute realized E[return] per bin.

    Validates whether higher confidence maps to higher realized returns.
    """
    print(f"\n{'='*60}")
    print("PART 2c: Expected Value by Confidence")
    print("=" * 60)

    # Use max_proba as confidence, direction from prediction
    pred = oos_df["pred"].values.astype(int)
    max_p = oos_df["max_proba"].values
    daily_ret = oos_df["daily_return"].values

    # Directional return: positive if prediction direction aligns with return
    dir_return = np.zeros(len(oos_df))
    for i in range(len(oos_df) - 1):
        if pred[i] == 2:  # BULL
            dir_return[i] = daily_ret[i + 1]
        elif pred[i] == 0:  # BEAR
            dir_return[i] = -daily_ret[i + 1]
        else:
            dir_return[i] = 0  # BALANCE: no position

    bins = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    rows = []
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        mask = (max_p >= lo) & (max_p < hi) & (pred != 1)  # exclude BALANCE pred
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            e_ret = dir_return[mask].mean()
            win_rate = (dir_return[mask] > 0).mean()
        else:
            e_ret = 0
            win_rate = 0
        row = {
            "bin": f"[{lo:.2f}, {hi:.2f})",
            "n": n_in_bin,
            "e_return": e_ret,
            "win_rate": win_rate,
            "e_return_bps": e_ret * 10000,
        }
        rows.append(row)
        print(f"  {row['bin']:<16s} n={n_in_bin:>5d}  E[r]={e_ret*10000:>+6.1f}bps  WR={win_rate:.1%}")

    ev_df = pd.DataFrame(rows)

    # Check monotonicity
    e_rets = ev_df["e_return"].values
    non_zero = e_rets[ev_df["n"].values > 10]
    if len(non_zero) >= 3:
        monotonic = all(non_zero[i] <= non_zero[i + 1] for i in range(len(non_zero) - 1))
        print(f"\n  Monotonic E[return] with confidence: {'YES' if monotonic else 'NO'}")

    return ev_df


# ─────────────────────────────────────────────────────────────────────
# Part 3: Large Move Capture Analysis
# ─────────────────────────────────────────────────────────────────────

def identify_large_moves(oos_df: pd.DataFrame, threshold: float = 0.05) -> list[dict]:
    """Find drawdowns and rallies > threshold using rolling windows."""
    close = oos_df["close"].values
    dates = oos_df.index

    moves = []
    for window in [20, 40, 60]:
        for i in range(window, len(close)):
            segment = close[i - window: i + 1]
            peak = segment.max()
            trough = segment.min()

            # Drawdown: peak then trough
            peak_idx_local = segment.argmax()
            trough_idx_local = segment.argmin()

            if peak_idx_local < trough_idx_local:
                drawdown = (trough - peak) / peak
                if drawdown < -threshold:
                    start = i - window + peak_idx_local
                    end = i - window + trough_idx_local
                    moves.append({
                        "type": "drawdown",
                        "start_date": dates[start],
                        "end_date": dates[end],
                        "start_idx": start,
                        "end_idx": end,
                        "magnitude": drawdown,
                        "start_price": close[start],
                        "end_price": close[end],
                        "window": window,
                    })

            # Rally: trough then peak
            if trough_idx_local < peak_idx_local:
                rally = (peak - trough) / trough
                if rally > threshold:
                    start = i - window + trough_idx_local
                    end = i - window + peak_idx_local
                    moves.append({
                        "type": "rally",
                        "start_date": dates[start],
                        "end_date": dates[end],
                        "start_idx": start,
                        "end_idx": end,
                        "magnitude": rally,
                        "start_price": close[start],
                        "end_price": close[end],
                        "window": window,
                    })

    # Deduplicate overlapping moves of same type
    moves_df = pd.DataFrame(moves)
    if len(moves_df) == 0:
        return []

    deduped = []
    for move_type in ["drawdown", "rally"]:
        subset = moves_df[moves_df["type"] == move_type].sort_values(
            "magnitude", ascending=(move_type == "drawdown"), key=abs
        )
        used_ranges = []
        for _, row in subset.iterrows():
            overlap = False
            for (us, ue) in used_ranges:
                # Check overlap > 50%
                overlap_start = max(row["start_idx"], us)
                overlap_end = min(row["end_idx"], ue)
                if overlap_end > overlap_start:
                    overlap_len = overlap_end - overlap_start
                    move_len = row["end_idx"] - row["start_idx"]
                    if move_len > 0 and overlap_len / move_len > 0.5:
                        overlap = True
                        break
            if not overlap:
                deduped.append(row.to_dict())
                used_ranges.append((row["start_idx"], row["end_idx"]))

    return deduped


def compute_capture_analysis(
    oos_df: pd.DataFrame,
    large_moves: list[dict],
    strat_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Analyze how well each strategy captures large moves."""
    print(f"\n{'='*60}")
    print("PART 3: Large Move Capture Analysis")
    print("=" * 60)

    n_drawdowns = sum(1 for m in large_moves if m["type"] == "drawdown")
    n_rallies = sum(1 for m in large_moves if m["type"] == "rally")
    print(f"  Found {len(large_moves)} large moves: {n_drawdowns} drawdowns, {n_rallies} rallies")

    if not large_moves:
        print("  No large moves found — skipping capture analysis")
        return pd.DataFrame()

    print(f"\n  Large moves identified:")
    for m in sorted(large_moves, key=lambda x: x["start_date"]):
        print(f"    {m['type']:>9s}: {m['start_date'].strftime('%Y-%m-%d')} → "
              f"{m['end_date'].strftime('%Y-%m-%d')} ({m['magnitude']:+.1%}, "
              f"{m['end_idx']-m['start_idx']}d)")

    # Analyze each strategy
    results = []
    select_strats = ["buy_hold", "conf_0.50", "conf_0.55", "conf_0.60",
                     "conf_0.50_p3d", "dd_exit_3pct", "asym_1d_exit_3d_entry",
                     "combined_dd3pct_conf0.50"]
    select_strats = [s for s in select_strats if s in strat_dfs]

    for strat_name in select_strats:
        sdf = strat_dfs[strat_name]
        pos = sdf["position"].values

        capture_at_start = 0
        full_capture = 0
        pain_days = 0
        total_move_days = 0
        missed = []

        for m in large_moves:
            si, ei = m["start_idx"], m["end_idx"]
            if ei <= si:
                continue

            # Correct position for this move type
            if m["type"] == "drawdown":
                correct_sign = -1  # should be short
                wrong_sign = 1
            else:
                correct_sign = 1  # should be long
                wrong_sign = -1

            move_positions = pos[si:ei + 1]
            move_days = len(move_positions)
            total_move_days += move_days

            # Correct at start?
            start_pos = pos[si] if si < len(pos) else 0
            if start_pos * correct_sign > 0 or start_pos == 0:
                capture_at_start += 1
            else:
                missed.append(m)

            # Full capture: correctly positioned for >50% of days
            correct_days = np.sum(move_positions * correct_sign > 0) + np.sum(move_positions == 0)
            if correct_days / move_days > 0.5:
                full_capture += 1

            # Pain days: positioned in wrong direction
            pain_days += np.sum(move_positions * wrong_sign > 0)

        n_moves = len(large_moves)
        result = {
            "strategy": strat_name,
            "capture_rate": capture_at_start / n_moves if n_moves else 0,
            "full_capture_rate": full_capture / n_moves if n_moves else 0,
            "pain_rate": pain_days / total_move_days if total_move_days else 0,
            "n_missed": len(missed),
        }
        results.append(result)

        print(f"\n  {strat_name}:")
        print(f"    Capture at start: {result['capture_rate']:.0%} ({capture_at_start}/{n_moves})")
        print(f"    Full capture (>50%): {result['full_capture_rate']:.0%}")
        print(f"    Pain rate: {result['pain_rate']:.1%}")
        if missed:
            for m in missed:
                print(f"    MISSED: {m['type']} {m['start_date'].strftime('%Y-%m-%d')} "
                      f"({m['magnitude']:+.1%})")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────
# Part 4: Figures
# ─────────────────────────────────────────────────────────────────────

def plot_equity_curves(oos_df, strat_dfs, variants):
    """ES price with regime shading + equity curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top: ES price with actual regime shading
    dates = oos_df.index
    close = oos_df["close"].values
    actual = oos_df["actual"].values

    ax1.plot(dates, close, color="black", linewidth=0.8)
    colors = {0: "#ffcccc", 1: "#f0f0f0", 2: "#ccffcc"}
    groups = (pd.Series(actual) != pd.Series(actual).shift()).cumsum()
    for gid in groups.unique():
        mask = groups == gid
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        regime = actual[idx[0]]
        ax1.axvspan(dates[idx[0]], dates[idx[-1]], alpha=0.3, color=colors[regime])

    ax1.set_ylabel("ES Price")
    ax1.set_title("ES Price with Actual Regime Shading (Red=BEAR, Gray=BALANCE, Green=BULL)")
    ax1.grid(True, alpha=0.3)

    # Bottom: equity curves
    select = ["buy_hold", "conf_0.50", "conf_0.55", "conf_0.60", "conf_0.50_p3d"]
    select = [s for s in select if s in strat_dfs]
    for label in select:
        sdf = strat_dfs[label]
        cum = (1 + sdf["strategy_return"]).cumprod()
        ax2.plot(sdf.index, cum, label=label, linewidth=1.2)

    ax2.set_ylabel("Cumulative Return (growth of $1)")
    ax2.set_title("Strategy Equity Curves")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_diagnostics_panel(oos_df, regime_runs, cal_data, conf_df):
    """2x2 diagnostics grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Regime duration histograms
    ax = axes[0, 0]
    colors = {0: "red", 1: "gray", 2: "green"}
    for cls in [0, 1, 2]:
        d = regime_runs.loc[regime_runs["regime"] == cls, "duration"]
        if len(d) > 0:
            ax.hist(d, bins=20, alpha=0.5, color=colors[cls],
                    label=CLASS_NAMES[cls], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Regime Duration Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Calibration reliability diagram
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for name, data in cal_data.items():
        color = {"BEAR": "red", "BALANCE": "gray", "BULL": "green"}[name]
        ax.plot(data["bin_centers"], data["bin_freqs"], "o-", color=color,
                label=f"{name} (ECE={data['ece']:.3f})", markersize=4)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("(b) Calibration Reliability Diagram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Confidence distribution with accuracy overlay
    ax = axes[1, 0]
    ax.hist(oos_df["max_proba"], bins=40, alpha=0.6, color="steelblue",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Max Probability (Confidence)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Confidence Distribution")
    ax.grid(True, alpha=0.3)

    # Overlay accuracy
    if conf_df is not None and len(conf_df) > 0:
        ax2 = ax.twinx()
        ax2.plot(conf_df["threshold"], conf_df["accuracy"], "ro-",
                 markersize=5, label="Accuracy")
        ax2.plot(conf_df["threshold"], conf_df["dir_accuracy"], "bs-",
                 markersize=5, label="Dir. Accuracy")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="upper right", fontsize=8)

    # (d) Transition lag histogram
    ax = axes[1, 1]
    actual = oos_df["actual"].astype(int).values
    pred = oos_df["pred"].astype(int).values
    lags = []
    for i in range(1, len(actual)):
        if actual[i] != actual[i - 1]:
            for j in range(i, len(pred)):
                if pred[j] == actual[i]:
                    lags.append(j - i)
                    break
    if lags:
        ax.hist(lags, bins=30, alpha=0.6, color="orange",
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Count")
    ax.set_title("(d) Transition Detection Lag")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "diagnostics_panel.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_large_move_capture(oos_df, large_moves, strat_dfs):
    """ES price with large moves marked and strategy position shading."""
    fig, ax = plt.subplots(figsize=(16, 7))

    dates = oos_df.index
    close = oos_df["close"].values
    ax.plot(dates, close, color="black", linewidth=0.8)

    # Strategy position shading (use conf_0.50 as reference)
    strat_key = "conf_0.50" if "conf_0.50" in strat_dfs else list(strat_dfs.keys())[0]
    pos = strat_dfs[strat_key]["position"].values
    for i in range(len(dates) - 1):
        if pos[i] > 0:
            ax.axvspan(dates[i], dates[i + 1], alpha=0.1, color="green", linewidth=0)
        elif pos[i] < 0:
            ax.axvspan(dates[i], dates[i + 1], alpha=0.1, color="red", linewidth=0)

    # Mark large moves
    for m in large_moves:
        color = "red" if m["type"] == "drawdown" else "green"
        ax.axvspan(m["start_date"], m["end_date"], alpha=0.25, color=color, linewidth=0)
        mid_date = m["start_date"] + (m["end_date"] - m["start_date"]) / 2
        mid_price = (m["start_price"] + m["end_price"]) / 2
        ax.annotate(f"{m['magnitude']:+.1%}", xy=(mid_date, mid_price),
                    fontsize=7, ha="center", color=color,
                    fontweight="bold", alpha=0.8)

    ax.set_ylabel("ES Price")
    ax.set_title(f"Large Moves (>5%) with {strat_key} Position Shading")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "large_move_capture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_metrics_summary(variants):
    """Table image of strategy metrics."""
    fig, ax = plt.subplots(figsize=(16, max(4, len(variants) * 0.4 + 1.5)))
    ax.axis("off")

    col_labels = ["Strategy", "Tot.Ret", "Ann.Ret", "Sharpe", "MaxDD",
                  "Calmar", "Tr/Yr", "Hold(d)", "WR", "InMkt%"]
    cell_data = []
    for v in variants:
        cell_data.append([
            v["label"],
            f"{v['total_return']:.1%}",
            f"{v['ann_return']:.1%}",
            f"{v['sharpe']:.2f}",
            f"{v['max_dd']:.1%}",
            f"{v['calmar']:.2f}",
            f"{v['trades_per_year']:.1f}",
            f"{v['avg_hold_days']:.1f}",
            f"{v['win_rate']:.0%}",
            f"{v['time_in_market_pct']:.1f}%",
        ])

    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("Strategy Metrics Summary", fontsize=14, pad=20)
    plt.tight_layout()
    path = FIGURES_DIR / "metrics_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_importance_stability(imp_matrix):
    """Heatmap of top 20 features x folds."""
    if imp_matrix is None or len(imp_matrix) == 0:
        return

    fold_cols = [c for c in imp_matrix.columns if c.startswith("fold_")]
    if not fold_cols:
        return

    top20 = imp_matrix.sort_values("mean_imp", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    data = top20[fold_cols].values
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(fold_cols)))
    ax.set_xticklabels(fold_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=8)
    ax.set_title("Feature Importance Stability (Top 20)")
    fig.colorbar(im, ax=ax, label="Importance")

    plt.tight_layout()
    path = FIGURES_DIR / "feature_importance_stability.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_position_sizing_curves(oos_df, strat_dfs):
    """Equity curves comparing position sizing strategies."""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Select representative strategies to plot
    to_plot = [
        ("buy_hold", "Buy & Hold (1x)", "black"),
        ("long_biased_flat_0.50", "Long-biased flat", "steelblue"),
        ("long_sized_1.5x_0.50", "Long-sized 1.5x", "green"),
        ("dd_exit_3pct", "DD exit 3%", "orange"),
        ("asym_1d_exit_3d_entry", "Asym 1d/3d", "red"),
        ("combined_dd3pct_conf0.50", "Combined DD3%+conf0.50", "purple"),
        ("combined_dd4pct_conf0.50", "Combined DD4%+conf0.50", "brown"),
    ]

    for key, label, color in to_plot:
        if key in strat_dfs:
            sdf = strat_dfs[key]
            cum = (1 + sdf["strategy_return"]).cumprod()
            ax.plot(sdf.index, cum, label=label, linewidth=1.2, color=color)

    ax.set_ylabel("Growth of $1")
    ax.set_title("Position Sizing Strategy Comparison")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    path = FIGURES_DIR / "position_sizing_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_expected_value(ev_df):
    """Bar chart of E[return] by confidence bin."""
    if ev_df is None or len(ev_df) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = ev_df["bin"].values
    x = range(len(bins))

    # E[return] bars
    colors = ["green" if v > 0 else "red" for v in ev_df["e_return_bps"].values]
    ax1.bar(x, ev_df["e_return_bps"].values, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("E[return] (bps)")
    ax1.set_title("Expected Return by Confidence Bin")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # Win rate bars
    ax2.bar(x, ev_df["win_rate"].values * 100, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Win Rate by Confidence Bin")
    ax2.axhline(50, color="black", linewidth=0.5, linestyle="--")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "expected_value_by_confidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Regime classifier analysis")
    parser.add_argument("--skip-training", action="store_true",
                        help="Use cached OOS data instead of retraining")
    parser.add_argument("--es-only", action="store_true",
                        help="Skip cross-instrument features")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--detect-threshold", type=float, default=0.05,
                        help="Zigzag detection threshold")
    parser.add_argument("--bull-threshold", type=float, default=0.10,
                        help="Min rally for BULL")
    parser.add_argument("--bear-threshold", type=float, default=0.07,
                        help="Min drawdown for BEAR")
    parser.add_argument("--target", choices=["y_micro", "y_macro", "y_structural"],
                        default="y_structural", help="Target label to train on")
    parser.add_argument("--no-external", action="store_true",
                        help="Skip VIX/DXY features")
    parser.add_argument("--bear-upweight", type=float, default=1.5,
                        help="Extra multiplier for BEAR class weight (default 1.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── Part 0: Get OOS data ──
    if args.skip_training:
        if not CACHE_PATH.exists():
            print(f"ERROR: Cache not found at {CACHE_PATH}. Run without --skip-training first.")
            sys.exit(1)
        print("Loading cached OOS data...")
        oos_df, result = load_cache()
        print(f"  {len(oos_df)} OOS days loaded from cache")
    else:
        es_daily, results, feature_cols = run_regime_pipeline(args)
        target = args.target
        result = results[target]
        oos_df = build_oos_dataframe(result, es_daily)
        save_cache(oos_df, result, target_name=target)
        print(f"\n  OOS DataFrame: {len(oos_df)} days, "
              f"{oos_df.index.min().date()} – {oos_df.index.max().date()}")
        print(f"  Target: {target}")

    # Quick summary
    correct = (oos_df["pred"] == oos_df["actual"])
    print(f"\n  OOS accuracy: {correct.mean():.3f}")
    print(f"  Prediction distribution: "
          + ", ".join(f"{CLASS_NAMES[c]}={(oos_df['pred']==c).mean():.1%}"
                      for c in [0, 1, 2]))
    print(f"  Actual distribution:     "
          + ", ".join(f"{CLASS_NAMES[c]}={(oos_df['actual']==c).mean():.1%}"
                      for c in [0, 1, 2]))

    # ── Part 1: Classifier Diagnostics ──
    regime_runs = analyze_regime_durations(oos_df)
    trans_lag = analyze_transition_lag(oos_df)
    trans_conf = analyze_transition_confusion(oos_df)
    cal_data = analyze_calibration(oos_df)
    imp_matrix = analyze_feature_importance_stability(result)
    conf_df = analyze_confidence_distribution(oos_df)

    # ── Part 2: Filtered Swing Strategy ──
    variants, strat_dfs = run_strategy_sweep(oos_df)

    # ── Part 2b: Long-biased & position sizing ──
    variants, strat_dfs = run_extended_strategy_sweep(oos_df, variants, strat_dfs)

    # ── Part 2c: Expected value analysis ──
    ev_df = analyze_expected_value_by_confidence(oos_df)

    # ── Part 3: Large Move Capture ──
    large_moves = identify_large_moves(oos_df, threshold=0.05)
    capture_df = compute_capture_analysis(oos_df, large_moves, strat_dfs)

    # ── Part 4: Figures ──
    if not args.skip_plots:
        print(f"\n{'='*60}")
        print("PART 4: Generating Figures")
        print("=" * 60)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        plot_equity_curves(oos_df, strat_dfs, variants)
        plot_diagnostics_panel(oos_df, regime_runs, cal_data, conf_df)
        plot_large_move_capture(oos_df, large_moves, strat_dfs)
        plot_metrics_summary(variants)
        plot_feature_importance_stability(imp_matrix)
        plot_position_sizing_curves(oos_df, strat_dfs)
        plot_expected_value(ev_df)

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — {elapsed:.1f}s elapsed")
    print("=" * 60)

    # Key findings — separate by category
    bh = [v for v in variants if v["label"] == "Buy&Hold"]
    base_variants = [v for v in variants if not v["label"].startswith("long_biased")
                     and not v["label"].startswith("long_sized")
                     and not v["label"].startswith("linear_")
                     and not v["label"].startswith("tiered")
                     and v["label"] != "Buy&Hold"]
    lb_variants = [v for v in variants if v["label"].startswith("long_biased")]
    sized_variants = [v for v in variants if v["label"].startswith("long_sized")
                      or v["label"].startswith("linear_")
                      or v["label"].startswith("tiered")]

    if bh:
        print(f"\n  Buy&Hold baseline: Sharpe={bh[0]['sharpe']:.2f}, Return={bh[0]['total_return']:.1%}")

    if base_variants:
        best_base = max(base_variants, key=lambda v: v["sharpe"])
        print(f"  Best base strategy: {best_base['label']} (Sharpe={best_base['sharpe']:.2f})")

    if lb_variants:
        best_lb = max(lb_variants, key=lambda v: v["sharpe"])
        print(f"  Best long-biased: {best_lb['label']} (Sharpe={best_lb['sharpe']:.2f}, "
              f"Tr/Yr={best_lb['trades_per_year']:.1f})")

    if sized_variants:
        best_sized = max(sized_variants, key=lambda v: v["sharpe"])
        print(f"  Best sized strategy: {best_sized['label']} (Sharpe={best_sized['sharpe']:.2f}, "
              f"Return={best_sized['total_return']:.1%})")

    best = max(variants, key=lambda v: v["sharpe"])
    print(f"\n  OVERALL best Sharpe: {best['label']} (Sharpe={best['sharpe']:.2f}, "
          f"Tr/Yr={best['trades_per_year']:.1f}, WR={best['win_rate']:.0%})")

    # Confidence→accuracy check
    if len(conf_df) >= 2:
        accs = conf_df["accuracy"].values
        monotonic = all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1))
        print(f"  Confidence→accuracy monotonic: {'YES' if monotonic else 'NO'}")


if __name__ == "__main__":
    main()
