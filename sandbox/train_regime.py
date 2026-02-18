"""Main experiment script for daily regime classification.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_regime.py
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_regime.py --es-only
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_regime.py --quick
"""
import argparse
import sys
import time

import pandas as pd
import numpy as np

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
from strategies.swing.labeling.structural_regime import compute_labels
from strategies.swing.labeling.hmm_regime import (
    compute_hmm_features_walkforward, FEATURE_NAMES as HMM_FEATURES,
)
from strategies.swing.training.regime_trainer import walk_forward_cv, compute_pnl_proxy


def parse_args():
    parser = argparse.ArgumentParser(description="Train daily regime classifier")
    parser.add_argument("--es-only", action="store_true", help="Skip cross-instrument features")
    parser.add_argument("--quick", action="store_true", help="Quick debug run (2 folds, fewer days)")
    parser.add_argument("--no-hmm", action="store_true", help="Skip HMM features")
    parser.add_argument("--no-vp", action="store_true", help="Skip VP features (faster loading)")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--min-train", type=int, default=500, help="Min training days")
    parser.add_argument("--micro-threshold", type=float, default=0.003, help="Micro label threshold")
    parser.add_argument("--detect-threshold", type=float, default=0.05, help="Zigzag detection threshold")
    parser.add_argument("--bull-threshold", type=float, default=0.10, help="Min rally for BULL")
    parser.add_argument("--bear-threshold", type=float, default=0.07, help="Min drawdown for BEAR")
    parser.add_argument("--target", choices=["y_micro", "y_macro", "y_structural"], default="y_structural",
                        help="Target label to train on")
    parser.add_argument("--no-external", action="store_true", help="Skip VIX/DXY features")
    parser.add_argument("--bear-upweight", type=float, default=1.5,
                        help="Extra multiplier for BEAR class weight (default 1.5)")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    config = SwingConfig(
        n_folds=args.folds if not args.quick else 2,
        min_train_days=args.min_train if not args.quick else 200,
        micro_threshold_pct=args.micro_threshold,
        detect_threshold=args.detect_threshold,
        bull_threshold=args.bull_threshold,
        bear_threshold=args.bear_threshold,
    )

    # ── Step 1: Load data ──
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

    # ── Step 2: Aggregate to daily ──
    print(f"\n{'='*60}")
    print("STEP 2: Aggregating to daily bars")
    print("=" * 60)

    aggregator = DailyAggregator()
    daily_data = {}
    for sym, minute_df in minute_data.items():
        compute_vp = (sym == "ES" and not args.no_vp)
        print(f"  Aggregating {sym} (VP={'yes' if compute_vp else 'no'})...")
        daily_data[sym] = aggregator.aggregate(minute_df, compute_vp=compute_vp)
        print(f"    {len(daily_data[sym])} trading days")

    # Align instruments
    if len(daily_data) > 1:
        daily_data = align_daily(daily_data, primary="ES")

    es_daily = daily_data["ES"]
    print(f"\nES daily: {len(es_daily)} days, {es_daily.index.min().date()} – {es_daily.index.max().date()}")

    # ── Step 3: Compute features ──
    print(f"\n{'='*60}")
    print("STEP 3: Computing features")
    print("=" * 60)

    # Technical features
    print("  Computing daily technical features...")
    tech_feats = compute_daily_technical(es_daily)
    feature_cols = list(TECH_FEATURES)
    print(f"    {len(TECH_FEATURES)} technical features")

    # VP features
    if not args.no_vp and "vp_poc_rel" in es_daily.columns:
        print("  Computing VP daily features...")
        vp_feats = compute_vp_daily_features(es_daily)
        feature_cols += VP_FEATURES
        print(f"    {len(VP_FEATURES)} VP features")
    else:
        vp_feats = pd.DataFrame(index=es_daily.index)
        print("  Skipping VP features")

    # Cross-instrument features
    if not args.es_only and len(daily_data) > 1:
        print("  Computing cross-instrument features...")
        other_dailys = [(sym, df) for sym, df in daily_data.items() if sym != "ES"]
        cross_feats = compute_cross_features(es_daily, other_dailys, config.corr_windows)
        cross_names = get_cross_names([sym for sym, _ in other_dailys])
        # Only add columns that actually exist
        cross_names = [c for c in cross_names if c in cross_feats.columns]
        feature_cols += cross_names
        print(f"    {len(cross_names)} cross-instrument features")
    else:
        cross_feats = pd.DataFrame(index=es_daily.index)
        print("  Skipping cross-instrument features")

    # Macro context
    print("  Computing macro context features...")
    other_for_macro = [(sym, df) for sym, df in daily_data.items() if sym != "ES"] if len(daily_data) > 1 else None
    macro_feats = compute_macro_context(es_daily, other_for_macro)
    feature_cols += MACRO_FEATURES
    print(f"    {len(MACRO_FEATURES)} macro context features")

    # External features (VIX, DXY)
    if not args.no_external:
        print("  Computing external daily features (VIX, DXY)...")
        ext_feats, ext_names = compute_external_features(es_daily)
        feature_cols += ext_names
    else:
        ext_feats = pd.DataFrame(index=es_daily.index)
        print("  Skipping external features")

    # Combine all features
    all_feats = pd.concat([tech_feats, vp_feats, cross_feats, macro_feats, ext_feats], axis=1)
    all_feats = all_feats.reindex(es_daily.index).fillna(0)

    print(f"\n  Total features: {len(feature_cols)}")

    # ── Step 4: Compute labels ──
    print(f"\n{'='*60}")
    print("STEP 4: Computing regime labels")
    print("=" * 60)

    labels = compute_labels(
        es_daily,
        swing_lookback=config.swing_lookback,
        micro_threshold_pct=config.micro_threshold_pct,
        detect_threshold=config.detect_threshold,
        bull_threshold=config.bull_threshold,
        bear_threshold=config.bear_threshold,
    )

    # Print distributions
    for col in ["y_macro", "y_micro", "y_structural"]:
        valid = labels[col][labels[col].isin([0, 1, 2])]
        dist = valid.value_counts(normalize=True).sort_index()
        names = {0: "DOWN/BEAR", 1: "BALANCE", 2: "UP/BULL"}
        print(f"\n  {col} distribution:")
        for k, v in dist.items():
            print(f"    {names.get(k, '?')}: {v:.1%} ({int(valid.value_counts()[k])} days)")

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

    # Merge features + labels
    df = all_feats.join(labels)

    # ── Step 5: Walk-forward training ──
    print(f"\n{'='*60}")
    print("STEP 5: Walk-forward training")
    print("=" * 60)

    # HMM callback
    hmm_fn = None
    hmm_feature_cols = []
    if not args.no_hmm:
        def hmm_fn(df_fold, train_end_idx):
            return compute_hmm_features_walkforward(
                es_daily, train_end_idx, n_states=config.hmm_n_states
            )
        hmm_feature_cols = HMM_FEATURES
        print(f"  HMM features enabled ({len(HMM_FEATURES)} features, refitted per fold)")

    feature_cols_with_hmm = feature_cols + hmm_feature_cols

    # Train for selected target + y_micro (for PnL proxy)
    targets_to_train = [args.target]
    if "y_micro" not in targets_to_train:
        targets_to_train.append("y_micro")

    xgb_params = {"bear_upweight": args.bear_upweight}
    print(f"  BEAR upweight: {args.bear_upweight}x")

    results = {}
    for target in targets_to_train:
        result = walk_forward_cv(
            df=df,
            feature_cols=feature_cols,
            target_col=target,
            n_folds=config.n_folds,
            min_train_days=config.min_train_days,
            params=xgb_params,
            hmm_fn=hmm_fn,
            verbose=True,
        )
        results[target] = result

    # ── Step 6: PnL proxy (micro regime) ──
    print(f"\n{'='*60}")
    print("STEP 6: PnL proxy (micro regime)")
    print("=" * 60)

    micro_result = results["y_micro"]
    pnl_df = compute_pnl_proxy(micro_result, es_daily)

    total_return = pnl_df["strategy_return"].sum()
    sharpe = pnl_df["strategy_return"].mean() / pnl_df["strategy_return"].std() * np.sqrt(252) if pnl_df["strategy_return"].std() > 0 else 0
    n_long = (pnl_df["position"] == 1).sum()
    n_short = (pnl_df["position"] == -1).sum()
    n_flat = (pnl_df["position"] == 0).sum()

    print(f"  Total return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Annualized Sharpe: {sharpe:.2f}")
    print(f"  Positions: {n_long} long, {n_short} short, {n_flat} flat")
    print(f"  Final cum PnL: {pnl_df['cum_pnl'].iloc[-1]:.4f}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — {elapsed:.1f}s elapsed")
    print("=" * 60)
    print(f"  Features: {len(feature_cols)} base + {len(hmm_feature_cols)} HMM = {len(feature_cols_with_hmm)} total")
    print(f"  Days: {len(es_daily)}")
    for target, result in results.items():
        print(f"  {target}: Dir.Acc={result.aggregate_directional_acc:.3f}, F1={result.aggregate_f1:.3f}")


if __name__ == "__main__":
    main()
