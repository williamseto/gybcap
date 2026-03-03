"""Walk-forward 3-class XGBoost trainer for regime classification.

Supports two prediction targets:
  - y_micro: Next-day micro regime (UP/DOWN/BALANCE) — primary
  - y_macro: Current macro regime (BULL/BEAR/BALANCE) — secondary

Walk-forward expanding window with HMM features refitted per fold.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)


@dataclass
class FoldResult:
    fold: int
    train_days: int
    test_days: int
    train_samples: int
    test_samples: int
    accuracy: float
    f1_macro: float
    directional_accuracy: float  # UP/DOWN only
    per_class_precision: dict
    per_class_recall: dict
    feature_importances: Optional[pd.DataFrame] = None


@dataclass
class TrainerResult:
    target: str
    fold_results: list[FoldResult]
    aggregate_accuracy: float
    aggregate_f1: float
    aggregate_directional_acc: float
    fold_std: float
    oos_predictions: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    oos_actuals: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    oos_probas: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    class_names: list[str] = field(default_factory=list)


DEFAULT_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "n_estimators": 500,
    "tree_method": "hist",
    "max_depth": 5,
    "min_child_weight": 10,
    "gamma": 0.5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "random_state": 42,
    "bear_upweight": 1.5,  # extra multiplier for BEAR class weight
}


def walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_folds: int = 5,
    min_train_days: int = 500,
    params: dict | None = None,
    hmm_fn=None,
    verbose: bool = True,
) -> TrainerResult:
    """Walk-forward cross-validation for 3-class regime prediction.

    Args:
        df: DataFrame with features, target, and DatetimeIndex
        feature_cols: List of feature column names
        target_col: 'y_micro' or 'y_macro'
        n_folds: Number of walk-forward folds
        min_train_days: Minimum training window in days
        params: XGBoost parameters (overrides defaults)
        hmm_fn: Optional callable(df, train_end_idx) → DataFrame with HMM features.
                 Called per fold so HMM is refitted on training data only.
        verbose: Print progress

    Returns:
        TrainerResult with per-fold and aggregate metrics
    """
    xgb_params = {**DEFAULT_PARAMS, **(params or {})}
    bear_upweight = xgb_params.pop("bear_upweight", 1.5)

    # Filter valid labels
    valid_mask = df[target_col].isin([0, 1, 2])
    df_valid = df[valid_mask].copy()

    days = sorted(df_valid.index.unique())
    n_days = len(days)

    if n_days < min_train_days + n_folds:
        raise ValueError(
            f"Not enough days ({n_days}) for {n_folds} folds with "
            f"min_train_days={min_train_days}"
        )

    test_days_per_fold = (n_days - min_train_days) // n_folds

    if verbose:
        class_names = {0: "DOWN/BEAR", 1: "BALANCE", 2: "UP/BULL"}
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD CV — target: {target_col}")
        print(f"{'='*60}")
        print(f"Total days: {n_days}, Min train: {min_train_days}, Folds: {n_folds}")
        print(f"Test days per fold: ~{test_days_per_fold}")
        dist = df_valid[target_col].value_counts(normalize=True).sort_index()
        for k, v in dist.items():
            print(f"  Class {int(k)} ({class_names.get(int(k), '?')}): {v:.1%}")

    fold_results = []
    all_preds = []
    all_actuals = []
    all_probas = []

    for fold in range(n_folds):
        train_end = min_train_days + fold * test_days_per_fold
        test_end = train_end + test_days_per_fold if fold < n_folds - 1 else n_days

        train_days_list = days[:train_end]
        test_days_list = days[train_end:test_end]

        train_mask = df_valid.index.isin(train_days_list)
        test_mask = df_valid.index.isin(test_days_list)

        df_fold = df_valid.copy()

        # Refit HMM per fold if provided
        hmm_cols = []
        if hmm_fn is not None:
            hmm_feats = hmm_fn(df_fold, train_end)
            hmm_cols = [c for c in hmm_feats.columns if c not in df_fold.columns]
            for c in hmm_cols:
                df_fold[c] = hmm_feats[c].reindex(df_fold.index).fillna(0)

        fold_features = feature_cols + hmm_cols

        X_train = df_fold.loc[train_mask, fold_features].fillna(0).values
        y_train = df_fold.loc[train_mask, target_col].values.astype(int)
        X_test = df_fold.loc[test_mask, fold_features].fillna(0).values
        y_test = df_fold.loc[test_mask, target_col].values.astype(int)

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        # Ensure all 3 classes present in training set (XGBoost requirement)
        missing = set([0, 1, 2]) - set(np.unique(y_train))
        if missing:
            # Add synthetic zero-feature rows for missing classes
            for cls in missing:
                X_train = np.vstack([X_train, np.zeros((1, X_train.shape[1]))])
                y_train = np.append(y_train, cls)

        # Compute inverse-frequency class weights with BEAR upweight
        class_counts = np.bincount(y_train, minlength=3).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = len(y_train) / (3.0 * class_counts)
        class_weights[0] *= bear_upweight  # BEAR = class 0
        sample_weights = class_weights[y_train]

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        y_proba = model.predict_proba(X_test)
        y_pred = y_proba.argmax(axis=1)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        # Directional accuracy: only UP(2) and DOWN(0), ignore BALANCE(1)
        dir_mask = (y_test == 0) | (y_test == 2)
        if dir_mask.sum() > 0:
            dir_acc = accuracy_score(y_test[dir_mask], y_pred[dir_mask])
        else:
            dir_acc = 0.5

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        per_class_prec = {int(k): v["precision"] for k, v in report.items() if k.isdigit()}
        per_class_rec = {int(k): v["recall"] for k, v in report.items() if k.isdigit()}

        # Feature importance
        importance = pd.DataFrame({
            "feature": fold_features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        fr = FoldResult(
            fold=fold,
            train_days=len(train_days_list),
            test_days=len(test_days_list),
            train_samples=len(X_train),
            test_samples=len(X_test),
            accuracy=acc,
            f1_macro=f1,
            directional_accuracy=dir_acc,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            feature_importances=importance,
        )
        fold_results.append(fr)

        all_preds.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        all_probas.extend(y_proba.tolist())

        if verbose:
            print(f"\nFold {fold+1}/{n_folds}: train={len(train_days_list)}d, test={len(test_days_list)}d")
            print(f"  Accuracy: {acc:.3f}, F1(macro): {f1:.3f}, Dir.Acc: {dir_acc:.3f}")
            print(f"  Precision: {per_class_prec}")
            recall_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(per_class_rec.items()))
            print(f"  Recall:    {{{recall_str}}}")

    # Aggregate
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_probas = np.array(all_probas)

    agg_acc = accuracy_score(all_actuals, all_preds)
    agg_f1 = f1_score(all_actuals, all_preds, average="macro", zero_division=0)

    dir_mask = (all_actuals == 0) | (all_actuals == 2)
    agg_dir_acc = accuracy_score(all_actuals[dir_mask], all_preds[dir_mask]) if dir_mask.sum() > 0 else 0.5

    fold_accs = [fr.accuracy for fr in fold_results]
    fold_std = np.std(fold_accs) if len(fold_accs) > 1 else 0.0

    result = TrainerResult(
        target=target_col,
        fold_results=fold_results,
        aggregate_accuracy=agg_acc,
        aggregate_f1=agg_f1,
        aggregate_directional_acc=agg_dir_acc,
        fold_std=fold_std,
        oos_predictions=all_preds,
        oos_actuals=all_actuals,
        oos_probas=all_probas,
        class_names=["DOWN/BEAR", "BALANCE", "UP/BULL"],
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS — {target_col}")
        print(f"{'='*60}")
        print(f"Accuracy: {agg_acc:.3f}")
        print(f"F1 (macro): {agg_f1:.3f}")
        print(f"Directional accuracy (UP/DOWN only): {agg_dir_acc:.3f}")
        print(f"Fold accuracy std: {fold_std:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_actuals, all_preds)
        print(f"\nConfusion matrix:")
        print(f"  {'':>10s} {'Pred DOWN':>10s} {'Pred BAL':>10s} {'Pred UP':>10s}")
        labels = ["Act DOWN", "Act BAL", "Act UP"]
        for i, label in enumerate(labels):
            row = cm[i] if i < len(cm) else [0, 0, 0]
            print(f"  {label:>10s} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")

        # Top features (from last fold)
        if fold_results and fold_results[-1].feature_importances is not None:
            print(f"\nTop 15 features (last fold):")
            for _, row in fold_results[-1].feature_importances.head(15).iterrows():
                print(f"  {row['feature']:40s} {row['importance']:.4f}")

        # Success criteria check
        print(f"\n{'='*60}")
        print("SUCCESS CRITERIA CHECK")
        print(f"{'='*60}")
        checks = [
            ("Dir. accuracy > 55%", agg_dir_acc > 0.55),
            ("Macro F1 > 0.40", agg_f1 > 0.40),
            ("Fold std < 0.05", fold_std < 0.05),
        ]
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")

    return result


def compute_pnl_proxy(
    result: TrainerResult,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative PnL proxy: long on UP, short on DOWN, flat on BALANCE.

    Args:
        result: TrainerResult from walk_forward_cv
        daily: Daily DataFrame matching the OOS predictions

    Returns:
        DataFrame with columns: position, daily_return, strategy_return, cum_pnl
    """
    # Get OOS days
    days = sorted(daily.index.unique())
    n_days = len(days)

    # Reconstruct which days are OOS
    preds = result.oos_predictions
    actuals = result.oos_actuals

    # We need to map predictions back to days
    # For simplicity, use the tail of the daily data
    oos_days = days[n_days - len(preds):]

    if len(oos_days) != len(preds):
        print(f"  Warning: OOS day count mismatch ({len(oos_days)} vs {len(preds)})")
        min_len = min(len(oos_days), len(preds))
        oos_days = oos_days[:min_len]
        preds = preds[:min_len]

    pnl_df = pd.DataFrame(index=oos_days)
    pnl_df["prediction"] = preds
    pnl_df["actual"] = actuals[:len(oos_days)]

    # Position: +1 for UP, -1 for DOWN, 0 for BALANCE
    pnl_df["position"] = 0
    pnl_df.loc[pnl_df["prediction"] == 2, "position"] = 1
    pnl_df.loc[pnl_df["prediction"] == 0, "position"] = -1

    # Daily return (from daily DataFrame)
    daily_ret = daily["close"].pct_change().reindex(oos_days).fillna(0)
    pnl_df["daily_return"] = daily_ret.values

    # Strategy return = position * next day's return (since we predict next day)
    pnl_df["strategy_return"] = pnl_df["position"].shift(1).fillna(0) * pnl_df["daily_return"]
    pnl_df["cum_pnl"] = pnl_df["strategy_return"].cumsum()

    return pnl_df
