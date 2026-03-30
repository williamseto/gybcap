#!/usr/bin/env python
"""Reproducible entrypoint for Phase 4 footprint fusion training.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_footprint_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import research_utils as ru
import train_level_models_orderflow as ofexp

from strategies.data.sec_features import SEC_DATA_DIR
from strategies.features.orderflow_heatmaps import extract_batch_footprints
from strategies.reversal.footprint_bundle import save_footprint_bundle
from strategies.reversal.orderflow_trainer import FootprintTrainer

NON_FEATURE_COLS = {
    "dt",
    "date",
    "time",
    "trading_day",
    "outcome",
    "nearest_level_name",
    "nearest_level_price",
    "side",
    "trade_direction",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_prepared_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    if args.prepared_prefix:
        return (
            f"{args.prepared_prefix}_samples.parquet",
            f"{args.prepared_prefix}_ohlcv.parquet",
            f"{args.prepared_prefix}_meta.json",
        )

    return args.samples_parquet, args.ohlcv_parquet, args.meta_json


def _infer_scalar_cols(samples_df: pd.DataFrame) -> list[str]:
    cols = []
    for col in samples_df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(samples_df[col]):
            cols.append(col)
    return cols


def _suggest_min_train_days(valid_days: int, n_folds: int) -> int:
    if valid_days <= 2:
        return 1

    max_allowed = max(1, valid_days - 1)
    if valid_days > n_folds:
        max_allowed = max(1, valid_days - n_folds)

    candidate = int(round(valid_days * 0.70))
    candidate = max(5, candidate)
    return min(candidate, max_allowed)


def _hhmm_to_minute(hhmm: str) -> int:
    hhmm = hhmm.strip()
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM value: {hhmm}")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid HH:MM value: {hhmm}")
    return hour * 60 + minute


def _sample_stratified_by_day(samples_df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Keep recent samples while maintaining day coverage."""
    if max_samples <= 0 or len(samples_df) <= max_samples:
        return samples_df.sort_values("dt").copy()

    sdf = samples_df.sort_values("dt").copy()
    grouped = [(day, grp) for day, grp in sdf.groupby("trading_day", sort=True)]
    n_days = len(grouped)
    if n_days == 0:
        return sdf.iloc[-max_samples:].copy()

    quotas: dict[Any, int] = {}
    if n_days >= max_samples:
        # One sample per most-recent day when days exceed sample budget.
        recent_days = [day for day, _ in grouped][-max_samples:]
        recent_set = set(recent_days)
        for day, _ in grouped:
            quotas[day] = 1 if day in recent_set else 0
    else:
        base = max_samples // n_days
        rem = max_samples % n_days
        for day, _ in grouped:
            quotas[day] = base
        for day, _ in reversed(grouped):
            if rem <= 0:
                break
            quotas[day] += 1
            rem -= 1

    picked = []
    for day, grp in grouped:
        q = int(quotas.get(day, 0))
        if q <= 0:
            continue
        picked.append(grp.tail(min(q, len(grp))))

    if not picked:
        return sdf.iloc[-max_samples:].copy()

    out = pd.concat(picked, axis=0).sort_values("dt")
    if len(out) > max_samples:
        out = out.iloc[-max_samples:].copy()
    return out


def _prepare_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    print("\nPreparing 1-min labels and scalar features...")
    stage_t0 = time.time()

    ohlcv = ofexp.load_data(args.data)
    if args.max_trading_days > 0:
        days = sorted(ohlcv["trading_day"].astype(str).unique())
        keep_days = set(days[-args.max_trading_days :])
        ohlcv = ohlcv[ohlcv["trading_day"].astype(str).isin(keep_days)].copy()
        ohlcv = ohlcv.reset_index(drop=True)
        print(
            f"Restricted to recent {args.max_trading_days} trading days: "
            f"{len(ohlcv):,} bars"
        )

    ohlcv = ofexp.compute_levels(ohlcv)
    ohlcv, feature_cols = ofexp.compute_all_features(ohlcv)
    ohlcv = ofexp.label_reversals_breakouts(ohlcv)
    ohlcv, level_encoding_cols = ofexp.compute_level_encoding_features(ohlcv)
    ohlcv, of_cols = ofexp.merge_orderflow_features(
        ohlcv, add_missing_indicators=args.add_missing_indicators
    )

    baseline_cols = list(dict.fromkeys(feature_cols + level_encoding_cols))
    if args.exclude_orderflow_scalars:
        scalar_cols = baseline_cols
    else:
        scalar_cols = list(dict.fromkeys(baseline_cols + of_cols))

    samples_df = ohlcv.loc[ohlcv["outcome"].isin([0, 1])].copy()

    if args.rth_only_samples:
        start_minute = _hhmm_to_minute(args.rth_start)
        end_minute = _hhmm_to_minute(args.rth_end)
        dt_vals = pd.to_datetime(samples_df["dt"])
        minute_of_day = dt_vals.dt.hour * 60 + dt_vals.dt.minute
        rth_mask = (minute_of_day >= start_minute) & (minute_of_day < end_minute)
        before = len(samples_df)
        samples_df = samples_df.loc[rth_mask].copy()
        print(
            f"\nFiltered to RTH samples ({args.rth_start}-{args.rth_end}): "
            f"{len(samples_df):,}/{before:,}"
        )

    if args.max_samples > 0 and len(samples_df) > args.max_samples:
        if args.sample_selection == "stratified_by_day":
            samples_df = _sample_stratified_by_day(samples_df, args.max_samples)
            print(
                f"\nDownsampled to {len(samples_df):,} samples "
                f"(stratified by trading day)"
            )
        else:
            samples_df = samples_df.sort_values("dt").iloc[-args.max_samples :].copy()
            print(f"\nDownsampled to most recent {len(samples_df):,} samples")

    stats = {
        "prepare_sec": float(time.time() - stage_t0),
        "prepare_rss_mb": float(ru.rss_mb()),
        "n_ohlcv_bars": int(len(ohlcv)),
        "n_samples": int(len(samples_df)),
        "n_sample_days": int(samples_df["trading_day"].astype(str).nunique()),
        "n_scalar_features": int(len(scalar_cols)),
        "n_positive": int((samples_df["outcome"] == 1).sum()),
        "n_negative": int((samples_df["outcome"] == 0).sum()),
    }
    return ohlcv, samples_df, scalar_cols, stats


def _save_prepared_artifacts(
    ohlcv: pd.DataFrame,
    samples_df: pd.DataFrame,
    scalar_cols: list[str],
    samples_path: str,
    ohlcv_path: str,
    meta_path: str,
    source_config: dict[str, Any],
) -> None:
    if not samples_path or not ohlcv_path:
        return

    os.makedirs(os.path.dirname(samples_path), exist_ok=True)
    os.makedirs(os.path.dirname(ohlcv_path), exist_ok=True)

    keep_cols = [
        "dt",
        "trading_day",
        "close",
        "high",
        "low",
        "trade_direction",
    ]
    keep_cols = [c for c in keep_cols if c in ohlcv.columns]
    ohlcv_out = ohlcv[keep_cols].copy()

    print(f"\nSaving prepared samples -> {samples_path}")
    samples_df.to_parquet(samples_path)
    print(f"Saving prepared ohlcv -> {ohlcv_path}")
    ohlcv_out.to_parquet(ohlcv_path)

    if meta_path:
        meta_payload = {
            "created_at": ru.now_iso(),
            "git_commit": ru.get_git_commit(),
            "n_samples": int(len(samples_df)),
            "n_ohlcv": int(len(ohlcv_out)),
            "scalar_cols": scalar_cols,
            "source_config": source_config,
        }
        ru.save_json(meta_path, meta_payload)
        print(f"Saving prepared metadata -> {meta_path}")


def _load_prepared_artifacts(
    samples_path: str,
    ohlcv_path: str,
    meta_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    if not samples_path or not ohlcv_path:
        raise ValueError(
            "Prepared artifacts not configured. Set --prepared-prefix or both "
            "--samples-parquet and --ohlcv-parquet when using --skip-feature-compute/--train-only."
        )

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Prepared samples parquet not found: {samples_path}")
    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"Prepared ohlcv parquet not found: {ohlcv_path}")

    print(f"\nLoading prepared samples <- {samples_path}")
    samples_df = pd.read_parquet(samples_path)
    print(f"Loading prepared ohlcv <- {ohlcv_path}")
    ohlcv = pd.read_parquet(ohlcv_path)

    scalar_cols: list[str] = []
    meta: dict[str, Any] = {}
    if meta_path and os.path.exists(meta_path):
        meta = ru.load_json(meta_path, {})
        scalar_cols = list(meta.get("scalar_cols", []))

    if not scalar_cols:
        scalar_cols = _infer_scalar_cols(samples_df)
        print(
            "WARNING: scalar feature list not found in metadata; "
            "using inferred numeric columns."
        )

    return ohlcv, samples_df, scalar_cols, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train footprint fusion model on intraday reversal samples"
    )
    parser.add_argument("--data", default=ofexp.DATA_PATH)
    parser.add_argument("--sec-data-dir", default=SEC_DATA_DIR)
    parser.add_argument(
        "--cache-path",
        default="strategies/data/sec_cache/footprints_phase4_{n}.pkl",
    )
    parser.add_argument(
        "--output-json",
        default="sandbox/results/footprint_training_result.json",
    )

    # Prepared artifact paths (fast iteration mode)
    parser.add_argument("--prepared-prefix", default="")
    parser.add_argument("--samples-parquet", default="")
    parser.add_argument("--ohlcv-parquet", default="")
    parser.add_argument("--meta-json", default="")

    parser.add_argument("--skip-feature-compute", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-train-days", type=int, default=100)
    parser.add_argument("--auto-min-train-days", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--model-arch",
        default="cnn_fusion",
        choices=["cnn_fusion", "temporal_tcn", "multires_event", "transformer_cross"],
        help="Footprint model architecture.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-trading-days",
        type=int,
        default=0,
        help="If >0, keep only the most recent N trading days before feature compute.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60000,
        help="Use most recent N labeled samples to bound memory.",
    )
    parser.add_argument(
        "--sample-selection",
        default="last_n",
        choices=["last_n", "stratified_by_day"],
        help="How to downsample when max-samples is hit.",
    )
    parser.add_argument(
        "--rth-only-samples",
        action="store_true",
        help="Keep only regular trading hours samples before downsampling.",
    )
    parser.add_argument(
        "--rth-start",
        default="06:30",
        help="RTH window start (HH:MM, Pacific).",
    )
    parser.add_argument(
        "--rth-end",
        default="13:15",
        help="RTH window end (HH:MM, Pacific).",
    )
    parser.add_argument("--add-missing-indicators", action="store_true")
    parser.add_argument("--exclude-orderflow-scalars", action="store_true")
    parser.add_argument(
        "--predictions-parquet",
        default="",
        help="Optional parquet path to export OOS probabilities + sample indices.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Train and save a deployable final model bundle to --model-dir.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/reversal_footprint_cnn",
        help="Output directory for deployable footprint model bundle.",
    )
    parser.add_argument(
        "--final-pred-threshold",
        type=float,
        default=-1.0,
        help="Prediction threshold stored in bundle metadata (default: --threshold).",
    )
    parser.add_argument(
        "--final-val-day-frac",
        type=float,
        default=0.2,
        help="Temporal holdout fraction (by trading days) for final-model early stopping.",
    )
    parser.add_argument(
        "--live-proximity-pts",
        type=float,
        default=5.0,
        help="Near-level proximity threshold stored for realtime inference.",
    )
    parser.add_argument(
        "--live-stop-pts",
        type=float,
        default=4.0,
        help="Execution stop (points) stored in metadata for live routing.",
    )
    parser.add_argument(
        "--live-target-pts",
        type=float,
        default=6.0,
        help="Execution target (points) stored in metadata for live routing.",
    )
    parser.add_argument(
        "--live-max-bars",
        type=int,
        default=45,
        help="Execution max-bars horizon stored in metadata for live routing.",
    )
    args = parser.parse_args()

    if args.train_only:
        args.skip_feature_compute = True

    started_at = ru.now_iso()
    t0 = time.time()
    set_seed(args.seed)

    print("=" * 80)
    print("PHASE 4 FOOTPRINT TRAINING")
    print("=" * 80)

    samples_path, ohlcv_path, meta_path = _resolve_prepared_paths(args)
    stage_metrics: dict[str, Any] = {}

    if args.skip_feature_compute:
        ohlcv, samples_df, scalar_cols, meta = _load_prepared_artifacts(
            samples_path=samples_path,
            ohlcv_path=ohlcv_path,
            meta_path=meta_path,
        )
        stage_metrics["prepared_load_sec"] = float(time.time() - t0)
        stage_metrics["prepared_meta"] = meta
    else:
        ohlcv, samples_df, scalar_cols, prep_stats = _prepare_features(args)
        stage_metrics.update(prep_stats)
        _save_prepared_artifacts(
            ohlcv=ohlcv,
            samples_df=samples_df,
            scalar_cols=scalar_cols,
            samples_path=samples_path,
            ohlcv_path=ohlcv_path,
            meta_path=meta_path,
            source_config=vars(args),
        )

    print(
        f"\nTraining samples: {len(samples_df):,} "
        f"(rev={(samples_df['outcome'] == 1).sum():,}, "
        f"bo={(samples_df['outcome'] == 0).sum():,})"
    )
    print(f"Scalar feature count: {len(scalar_cols)}")
    print(f"Current RSS: {ru.rss_mb():.1f} MB")

    sample_bars = samples_df[["dt", "nearest_level_price", "trading_day"]].copy()
    cache_path = args.cache_path.format(n=len(samples_df))

    if args.train_only and not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"--train-only was set but footprint cache does not exist: {cache_path}"
        )

    print("\nExtracting footprint tensors...")
    fp_t0 = time.time()
    footprints = extract_batch_footprints(
        sec_data_dir=args.sec_data_dir,
        sample_bars=sample_bars,
        cache_path=cache_path,
    )
    stage_metrics["footprint_extract_sec"] = float(time.time() - fp_t0)
    stage_metrics["footprint_rss_mb"] = float(ru.rss_mb())

    n_valid = int(footprints["valid_mask"].sum())
    print(
        f"Valid footprint tensors: {n_valid:,}/{len(samples_df):,} "
        f"({100 * n_valid / max(len(samples_df), 1):.1f}%)"
    )

    valid_days = int(
        samples_df.iloc[np.where(footprints["valid_mask"])[0]]["trading_day"]
        .astype(str)
        .nunique()
    )
    min_train_days = args.min_train_days
    if args.auto_min_train_days:
        min_train_days = _suggest_min_train_days(valid_days, args.n_folds)
        print(
            f"Auto min_train_days enabled: using {min_train_days} "
            f"(valid_days={valid_days}, folds={args.n_folds})"
        )

    if valid_days > 0 and min_train_days >= valid_days:
        adjusted = max(1, valid_days - max(1, args.n_folds))
        print(
            f"Adjusting min_train_days {min_train_days} -> {adjusted} "
            f"to allow walk-forward splits"
        )
        min_train_days = adjusted

    payload = {
        "config": vars(args),
        "resolved_paths": {
            "samples_parquet": samples_path,
            "ohlcv_parquet": ohlcv_path,
            "meta_json": meta_path,
            "footprint_cache": cache_path,
        },
        "started_at": started_at,
        "git_commit": ru.get_git_commit(),
        "n_samples": int(len(samples_df)),
        "n_valid_footprints": n_valid,
        "valid_trading_days": valid_days,
        "min_train_days_used": int(min_train_days),
        "stage_metrics": stage_metrics,
    }

    if args.cache_only:
        payload.update(
            {
                "status": "cache_only",
                "elapsed_sec": float(time.time() - t0),
                "ended_at": ru.now_iso(),
                "rss_mb_end": float(ru.rss_mb()),
            }
        )
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print("\nCache-only mode complete.")
        print(f"Result JSON: {args.output_json}")
        return

    train_t0 = time.time()
    trainer = FootprintTrainer(
        n_folds=args.n_folds,
        min_train_days=min_train_days,
        threshold=args.threshold,
        random_seed=args.seed,
        model_arch=args.model_arch,
    )
    result = trainer.train(
        samples_df=samples_df,
        feature_cols=scalar_cols,
        footprints=footprints,
        ohlcv=ohlcv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )
    stage_metrics["train_sec"] = float(time.time() - train_t0)

    payload.update(
        {
            "status": "trained",
            "overall_roc_auc": float(result.overall_roc_auc),
            "total_trades": int(result.total_trades),
            "overall_win_rate": float(result.overall_win_rate),
            "overall_mean_pnl": float(result.overall_mean_pnl),
            "total_pnl": float(result.total_pnl),
            "fold_results": [asdict(fr) for fr in result.fold_results],
            "elapsed_sec": float(time.time() - t0),
            "ended_at": ru.now_iso(),
            "rss_mb_end": float(ru.rss_mb()),
        }
    )

    if args.save_model:
        final_threshold = (
            float(args.final_pred_threshold)
            if args.final_pred_threshold >= 0
            else float(args.threshold)
        )
        final_trainer = FootprintTrainer(
            n_folds=args.n_folds,
            min_train_days=min_train_days,
            threshold=final_threshold,
            random_seed=args.seed,
            model_arch=args.model_arch,
        )
        final_model, final_summary = final_trainer.train_final_model(
            samples_df=samples_df,
            feature_cols=scalar_cols,
            footprints=footprints,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_day_fraction=float(args.final_val_day_frac),
            verbose=True,
        )

        metadata = {
            "created_at": ru.now_iso(),
            "git_commit": ru.get_git_commit(),
            "model_arch": args.model_arch,
            "feature_cols": scalar_cols,
            "n_features": len(scalar_cols),
            "n_price_bins": int(footprints["current"].shape[2]),
            "current_time_steps": int(footprints["current"].shape[3]),
            "context_time_steps": int(footprints["context"].shape[3]),
            "context_minutes": int(footprints["context"].shape[3] // 60),
            "threshold": final_threshold,
            "proximity_pts": float(args.live_proximity_pts),
            "stop_pts": float(args.live_stop_pts),
            "target_pts": float(args.live_target_pts),
            "max_bars": int(args.live_max_bars),
            "source_training_json": args.output_json,
            "final_training": final_summary,
        }
        model_path, metadata_path = save_footprint_bundle(
            final_model,
            model_dir=args.model_dir,
            metadata=metadata,
        )
        payload["deployable_model"] = {
            "model_dir": args.model_dir,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "threshold": final_threshold,
            "proximity_pts": float(args.live_proximity_pts),
            "stop_pts": float(args.live_stop_pts),
            "target_pts": float(args.live_target_pts),
            "max_bars": int(args.live_max_bars),
            "final_training": final_summary,
        }
        print(f"\nSaved deployable model bundle to {args.model_dir}")

    if args.predictions_parquet:
        pred_idx = result.all_test_indices.astype(np.int64)
        pred_df = pd.DataFrame(
            {
                "ohlcv_index": pred_idx,
                "y_true": result.all_y_true.astype(np.int32),
                "y_prob": result.all_y_prob.astype(np.float32),
            }
        )
        if len(pred_idx) > 0 and "dt" in ohlcv.columns:
            pred_df["dt"] = pd.to_datetime(ohlcv.iloc[pred_idx]["dt"]).values
        if len(pred_idx) > 0 and "trading_day" in ohlcv.columns:
            pred_df["trading_day"] = (
                ohlcv.iloc[pred_idx]["trading_day"].astype(str).values
            )
        pred_df = pred_df.sort_values("ohlcv_index").reset_index(drop=True)
        pred_dir = os.path.dirname(args.predictions_parquet)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
        pred_df.to_parquet(args.predictions_parquet)
        payload["prediction_export_path"] = args.predictions_parquet
        payload["n_predictions"] = int(len(pred_df))
    else:
        payload["prediction_export_path"] = ""
        payload["n_predictions"] = int(len(result.all_test_indices))

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("FOOTPRINT TRAINING SUMMARY")
    print("=" * 80)
    print(f"AUC: {payload['overall_roc_auc']:.4f}")
    print(f"Trades: {payload['total_trades']:,}")
    print(f"WR: {payload['overall_win_rate']:.2%}")
    print(f"E[PnL]: {payload['overall_mean_pnl']:+.3f}")
    print(f"Total PnL: {payload['total_pnl']:+.1f}")
    print(f"Result JSON: {args.output_json}")
    print(f"Elapsed: {payload['elapsed_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
