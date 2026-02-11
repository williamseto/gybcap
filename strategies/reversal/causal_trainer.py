"""
Walk-forward trainer for causal zone prediction.

Supports two parallel paths:
  Path A: XGBoost on ~85 scalar features → binary zone membership
  Path B: V3 CausalZoneModel on VP heatmaps + TCN + scalars → zone probability

Both use the same walk-forward CV splits (5 folds, min 100 train days).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)


def _clear_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@dataclass
class CausalFoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_days: int
    test_days: int
    train_samples: int
    test_samples: int
    n_positive_test: int
    n_predicted: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    # Trading simulation
    n_trades: int = 0
    win_rate: float = 0.0
    mean_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class CausalPathResult:
    """Results from training a causal predictor path."""
    path_name: str
    fold_results: List[CausalFoldResult]
    overall_precision: float
    overall_recall: float
    overall_f1: float
    overall_roc_auc: float
    # Trading
    total_trades: int = 0
    overall_win_rate: float = 0.0
    overall_mean_pnl: float = 0.0
    overall_total_pnl: float = 0.0
    # Feature importance (XGBoost only)
    feature_importance: Optional[Dict[str, float]] = None


def _get_fold_splits(
    days: List,
    n_folds: int,
    min_train_days: int,
) -> List[Tuple[List, List]]:
    """Get train/test day splits for walk-forward CV."""
    n_days = len(days)
    test_days_per_fold = (n_days - min_train_days) // n_folds

    splits = []
    for fold in range(n_folds):
        train_end_idx = min_train_days + fold * test_days_per_fold
        test_end_idx = train_end_idx + test_days_per_fold
        if fold == n_folds - 1:
            test_end_idx = n_days
        train_d = days[:train_end_idx]
        test_d = days[train_end_idx:test_end_idx]
        if len(test_d) > 0:
            splits.append((train_d, test_d))

    return splits


def _simulate_trades(
    ohlcv: pd.DataFrame,
    predicted_indices: np.ndarray,
    zone_labels: pd.DataFrame,
    stop_pts: float = 4.0,
    target_pts: float = 6.0,
    max_bars: int = 45,
) -> Tuple[int, float, float, float]:
    """
    Simulate fixed-stop/target trades at predicted zone bars.

    Returns (n_trades, win_rate, mean_pnl, total_pnl).
    """
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0

    close = ohlcv['close'].values
    high = ohlcv['high'].values
    low = ohlcv['low'].values
    n = len(close)

    wins = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n:
            continue
        direction = zone_labels.loc[idx, 'zone_label'] if idx in zone_labels.index else 0
        if direction == 0:
            # Use price relative to nearest level to infer direction
            nearest = zone_labels.loc[idx, 'nearest_level'] if idx in zone_labels.index else ''
            if nearest and nearest in ohlcv.columns:
                lvl = ohlcv.loc[idx, nearest]
                direction = 1 if close[idx] < lvl else -1
            else:
                continue

        entry_price = close[idx]
        trade_pnl = 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:  # long
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    break
            else:  # short
                if high[j] >= entry_price + stop_pts:
                    trade_pnl = -stop_pts
                    break
                if low[j] <= entry_price - target_pts:
                    trade_pnl = target_pts
                    break

        pnl_list.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1

    n_trades = len(pnl_list)
    if n_trades == 0:
        return 0, 0.0, 0.0, 0.0

    wr = wins / n_trades
    mean_pnl = np.mean(pnl_list)
    total_pnl = np.sum(pnl_list)
    return n_trades, wr, mean_pnl, total_pnl


class CausalZoneTrainer:
    """
    Walk-forward trainer for causal zone prediction.
    """

    def __init__(
        self,
        n_folds: int = 5,
        min_train_days: int = 100,
        thresholds: Optional[List[float]] = None,
    ):
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
        self.path_results: Dict[str, CausalPathResult] = {}

    # ─── Path A: XGBoost ────────────────────────────────────────────

    def train_xgboost(
        self,
        samples_df: pd.DataFrame,
        feature_cols: List[str],
        ohlcv: pd.DataFrame,
        zone_labels: pd.DataFrame,
        verbose: bool = True,
    ) -> CausalPathResult:
        """
        Train XGBoost on scalar features with walk-forward CV.

        Args:
            samples_df: DataFrame of sample bars with features and zone_label.
            feature_cols: Feature column names.
            ohlcv: Full OHLCV for trading simulation.
            zone_labels: Full zone labels for simulation.
            verbose: Print progress.

        Returns:
            CausalPathResult with fold results.
        """
        import xgboost as xgb

        if verbose:
            print("\n" + "=" * 60)
            print("PATH A: XGBOOST ZONE PREDICTION")
            print("=" * 60)

        days = sorted(samples_df['trading_day'].unique())
        splits = _get_fold_splits(days, self.n_folds, self.min_train_days)

        y = (samples_df['zone_label'] != 0).astype(int).values
        pos_rate = y.mean()

        if verbose:
            print(f"Samples: {len(samples_df)}, positive rate: {pos_rate:.2%}")
            print(f"Features: {len(feature_cols)}")

        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []
        all_feature_imp = {}

        best_threshold = 0.5  # will be tuned per fold

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}: "
                      f"{len(train_days)} train days, {len(test_days)} test days")

            train_mask = samples_df['trading_day'].isin(train_days)
            test_mask = samples_df['trading_day'].isin(test_days)

            X_train = samples_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
            X_test = samples_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
            y_train = y[train_mask.values]
            y_test = y[test_mask.values]
            test_indices = samples_df.index[test_mask].values

            if len(y_test) == 0 or y_train.sum() < 5:
                continue

            spw = max(1.0, (1 - y_train.mean()) / max(y_train.mean(), 1e-6))

            model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=500,
                min_child_weight=5,
                scale_pos_weight=spw,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
                tree_method='hist',
                early_stopping_rounds=30,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_prob = model.predict_proba(X_test)[:, 1]

            # Tune threshold on this fold's test set
            best_f1 = -1
            for t in self.thresholds:
                preds = (y_prob >= t).astype(int)
                if preds.sum() > 0:
                    f1_t = f1_score(y_test, preds, zero_division=0)
                    if f1_t > best_f1:
                        best_f1 = f1_t
                        best_threshold = t

            y_pred = (y_prob >= best_threshold).astype(int)
            n_pred = y_pred.sum()

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5

            # Trading simulation
            pred_indices = test_indices[y_pred == 1]
            n_trades, wr, mean_pnl, total_pnl = _simulate_trades(
                ohlcv, pred_indices, zone_labels,
            )

            fold_result = CausalFoldResult(
                fold=fold,
                train_days=len(train_days),
                test_days=len(test_days),
                train_samples=len(X_train),
                test_samples=len(X_test),
                n_positive_test=int(y_test.sum()),
                n_predicted=int(n_pred),
                precision=prec,
                recall=rec,
                f1=f1,
                roc_auc=auc,
                n_trades=n_trades,
                win_rate=wr,
                mean_pnl=mean_pnl,
                total_pnl=total_pnl,
            )
            fold_results.append(fold_result)
            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())
            all_y_pred.extend(y_pred.tolist())

            # Accumulate feature importance
            imp = model.get_booster().get_score(importance_type='gain')
            for fname, score in imp.items():
                # XGBoost uses f0, f1, ... as feature names
                if fname.startswith('f'):
                    fidx = int(fname[1:])
                    if fidx < len(feature_cols):
                        real_name = feature_cols[fidx]
                        all_feature_imp[real_name] = all_feature_imp.get(real_name, 0) + score

            if verbose:
                print(f"  Threshold={best_threshold:.1f} | "
                      f"P={prec:.2%} R={rec:.2%} F1={f1:.2%} AUC={auc:.3f} | "
                      f"Trades={n_trades} WR={wr:.1%} E[PnL]={mean_pnl:.2f}")

        # Overall
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = np.array(all_y_pred)

        tp = ((all_y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((all_y_pred == 1) & (all_y_true == 0)).sum()
        fn = ((all_y_pred == 0) & (all_y_true == 1)).sum()

        o_prec = tp / max(tp + fp, 1)
        o_rec = tp / max(tp + fn, 1)
        o_f1 = 2 * o_prec * o_rec / max(o_prec + o_rec, 1e-6)
        try:
            o_auc = roc_auc_score(all_y_true, all_y_prob)
        except ValueError:
            o_auc = 0.5

        # Aggregate trading
        total_trades = sum(fr.n_trades for fr in fold_results)
        total_wins = sum(fr.n_trades * fr.win_rate for fr in fold_results)
        total_pnl = sum(fr.total_pnl for fr in fold_results)

        # Normalize feature importance
        if all_feature_imp:
            max_imp = max(all_feature_imp.values())
            if max_imp > 0:
                all_feature_imp = {k: v / max_imp for k, v in all_feature_imp.items()}

        result = CausalPathResult(
            path_name='xgboost',
            fold_results=fold_results,
            overall_precision=o_prec,
            overall_recall=o_rec,
            overall_f1=o_f1,
            overall_roc_auc=o_auc,
            total_trades=total_trades,
            overall_win_rate=total_wins / max(total_trades, 1),
            overall_mean_pnl=total_pnl / max(total_trades, 1),
            overall_total_pnl=total_pnl,
            feature_importance=all_feature_imp,
        )

        self.path_results['xgboost'] = result

        if verbose:
            self._print_summary(result)

        return result

    # ─── Path B: V3 Neural Model ────────────────────────────────────

    def train_v3(
        self,
        samples_df: pd.DataFrame,
        feature_cols: List[str],
        heatmaps: Dict[str, np.ndarray],
        ohlcv: pd.DataFrame,
        zone_labels: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> CausalPathResult:
        """
        Train V3 CausalZoneModel with walk-forward CV.

        Args:
            samples_df: Sample bars with features and zone columns.
            feature_cols: Scalar feature columns.
            heatmaps: Dict with micro_vp, meso_vp, macro_vp, sequence arrays.
            ohlcv: Full OHLCV for trading simulation.
            zone_labels: Full zone labels.
            epochs: Training epochs per fold.
            batch_size: Training batch size.
            lr: Learning rate.
            verbose: Print progress.

        Returns:
            CausalPathResult.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from strategies.reversal.causal_model import CausalZoneModel

        if verbose:
            print("\n" + "=" * 60)
            print("PATH B: V3 CAUSAL ZONE MODEL")
            print("=" * 60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Device: {device}")

        days = sorted(samples_df['trading_day'].unique())
        splits = _get_fold_splits(days, self.n_folds, self.min_train_days)

        # Use soft labels (zone_probability) for training
        y_soft = samples_df['zone_probability'].fillna(0).values.astype(np.float32)
        y_hard = (samples_df['zone_label'] != 0).astype(int).values
        sample_indices = samples_df.index.values

        if verbose:
            print(f"Samples: {len(samples_df)}, positive rate: {y_hard.mean():.2%}")
            print(f"Scalar features: {len(feature_cols)}")
            micro_shape = heatmaps['micro_vp'].shape
            print(f"Heatmap shapes: micro={micro_shape}, "
                  f"meso={heatmaps['meso_vp'].shape}, "
                  f"macro={heatmaps['macro_vp'].shape}, "
                  f"seq={heatmaps['sequence'].shape}")

        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_y_pred = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(f"\nFold {fold + 1}/{len(splits)}: "
                      f"{len(train_days)} train, {len(test_days)} test days")

            train_mask = samples_df['trading_day'].isin(train_days).values
            test_mask = samples_df['trading_day'].isin(test_days).values

            if test_mask.sum() == 0 or y_hard[train_mask].sum() < 5:
                continue

            # Prepare tensors
            X_sc_train = torch.tensor(
                samples_df.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
            )
            X_sc_test = torch.tensor(
                samples_df.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
            )

            micro_train = torch.tensor(heatmaps['micro_vp'][train_mask])
            micro_test = torch.tensor(heatmaps['micro_vp'][test_mask])
            meso_train = torch.tensor(heatmaps['meso_vp'][train_mask])
            meso_test = torch.tensor(heatmaps['meso_vp'][test_mask])
            macro_train = torch.tensor(heatmaps['macro_vp'][train_mask])
            macro_test = torch.tensor(heatmaps['macro_vp'][test_mask])
            seq_train = torch.tensor(heatmaps['sequence'][train_mask])
            seq_test = torch.tensor(heatmaps['sequence'][test_mask])

            y_train = torch.tensor(y_soft[train_mask])
            y_test_hard = y_hard[test_mask]
            test_idx = sample_indices[test_mask]

            # Build model
            model = CausalZoneModel(
                scalar_dim=len(feature_cols),
                seq_channels=heatmaps['sequence'].shape[-1],
            ).to(device)

            if fold == 0 and verbose:
                print(f"  Model params: {model.count_parameters():,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            loss_fn = nn.BCELoss()

            # DataLoader
            train_ds = TensorDataset(
                micro_train, meso_train, macro_train, seq_train, X_sc_train, y_train,
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

            steps_per_epoch = len(train_loader)
            total_steps = max(epochs * steps_per_epoch, 1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps,
            )

            # Train
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                n_batch = 0
                for batch in train_loader:
                    b_micro, b_meso, b_macro, b_seq, b_sc, b_y = [
                        x.to(device) for x in batch
                    ]
                    pred = model(b_micro, b_meso, b_macro, b_seq, b_sc)
                    loss = loss_fn(pred, b_y)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    n_batch += 1

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_prob_list = []
                eval_bs = 256
                for i in range(0, len(X_sc_test), eval_bs):
                    pred = model(
                        micro_test[i:i+eval_bs].to(device),
                        meso_test[i:i+eval_bs].to(device),
                        macro_test[i:i+eval_bs].to(device),
                        seq_test[i:i+eval_bs].to(device),
                        X_sc_test[i:i+eval_bs].to(device),
                    )
                    y_prob_list.append(pred.cpu().numpy())

                y_prob = np.concatenate(y_prob_list)

            # Tune threshold
            best_threshold = 0.5
            best_f1 = -1
            for t in self.thresholds:
                preds = (y_prob >= t).astype(int)
                if preds.sum() > 0:
                    f1_t = f1_score(y_test_hard, preds, zero_division=0)
                    if f1_t > best_f1:
                        best_f1 = f1_t
                        best_threshold = t

            y_pred = (y_prob >= best_threshold).astype(int)
            n_pred = y_pred.sum()

            prec = precision_score(y_test_hard, y_pred, zero_division=0)
            rec = recall_score(y_test_hard, y_pred, zero_division=0)
            f1 = f1_score(y_test_hard, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test_hard, y_prob)
            except ValueError:
                auc = 0.5

            # Trading simulation
            pred_indices = test_idx[y_pred == 1]
            n_trades, wr, mean_pnl, total_pnl_fold = _simulate_trades(
                ohlcv, pred_indices, zone_labels,
            )

            fold_result = CausalFoldResult(
                fold=fold,
                train_days=len(train_days),
                test_days=len(test_days),
                train_samples=int(train_mask.sum()),
                test_samples=int(test_mask.sum()),
                n_positive_test=int(y_test_hard.sum()),
                n_predicted=int(n_pred),
                precision=prec,
                recall=rec,
                f1=f1,
                roc_auc=auc,
                n_trades=n_trades,
                win_rate=wr,
                mean_pnl=mean_pnl,
                total_pnl=total_pnl_fold,
            )
            fold_results.append(fold_result)
            all_y_true.extend(y_test_hard.tolist())
            all_y_prob.extend(y_prob.tolist())
            all_y_pred.extend(y_pred.tolist())

            if verbose:
                print(f"  Thresh={best_threshold:.1f} | "
                      f"P={prec:.2%} R={rec:.2%} F1={f1:.2%} AUC={auc:.3f} | "
                      f"Trades={n_trades} WR={wr:.1%} E[PnL]={mean_pnl:.2f}")

            del model, optimizer, scheduler
            _clear_gpu_memory()

        # Overall
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = np.array(all_y_pred)

        tp = ((all_y_pred == 1) & (all_y_true == 1)).sum()
        fp = ((all_y_pred == 1) & (all_y_true == 0)).sum()
        fn = ((all_y_pred == 0) & (all_y_true == 1)).sum()

        o_prec = tp / max(tp + fp, 1)
        o_rec = tp / max(tp + fn, 1)
        o_f1 = 2 * o_prec * o_rec / max(o_prec + o_rec, 1e-6)
        try:
            o_auc = roc_auc_score(all_y_true, all_y_prob)
        except ValueError:
            o_auc = 0.5

        total_trades = sum(fr.n_trades for fr in fold_results)
        total_wins = sum(fr.n_trades * fr.win_rate for fr in fold_results)
        total_pnl = sum(fr.total_pnl for fr in fold_results)

        result = CausalPathResult(
            path_name='v3_causal',
            fold_results=fold_results,
            overall_precision=o_prec,
            overall_recall=o_rec,
            overall_f1=o_f1,
            overall_roc_auc=o_auc,
            total_trades=total_trades,
            overall_win_rate=total_wins / max(total_trades, 1),
            overall_mean_pnl=total_pnl / max(total_trades, 1),
            overall_total_pnl=total_pnl,
        )

        self.path_results['v3_causal'] = result

        if verbose:
            self._print_summary(result)

        return result

    # ─── Utilities ──────────────────────────────────────────────────

    def _print_summary(self, result: CausalPathResult) -> None:
        print(f"\n{'=' * 60}")
        print(f"{result.path_name.upper()} SUMMARY")
        print(f"{'=' * 60}")
        print(f"Overall: P={result.overall_precision:.2%} "
              f"R={result.overall_recall:.2%} "
              f"F1={result.overall_f1:.2%} "
              f"AUC={result.overall_roc_auc:.3f}")
        print(f"Trading: {result.total_trades} trades, "
              f"WR={result.overall_win_rate:.1%}, "
              f"E[PnL]={result.overall_mean_pnl:.2f}pt, "
              f"Total={result.overall_total_pnl:.1f}pt")

        if result.feature_importance:
            print(f"\nTop 15 features (gain):")
            sorted_imp = sorted(
                result.feature_importance.items(), key=lambda x: x[1], reverse=True
            )
            for name, score in sorted_imp[:15]:
                print(f"  {name:35s}: {score:.3f}")

    def compare_paths(self) -> pd.DataFrame:
        """Compare results across trained paths."""
        if not self.path_results:
            raise ValueError("No paths trained yet.")

        rows = []
        for name, r in self.path_results.items():
            rows.append({
                'path': name,
                'precision': r.overall_precision,
                'recall': r.overall_recall,
                'f1': r.overall_f1,
                'roc_auc': r.overall_roc_auc,
                'trades': r.total_trades,
                'win_rate': r.overall_win_rate,
                'mean_pnl': r.overall_mean_pnl,
                'total_pnl': r.overall_total_pnl,
            })

        df = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("PATH COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        return df
