"""
Phase 4: Walk-Forward Trainer for FootprintFusionModel.

Trains the deep footprint model with walk-forward CV, honest early
stopping, and trading simulation. Follows the pattern of causal_trainer.py.

Usage:
    source ~/ml-venv/bin/activate
    PYTHONPATH=/home/william/gybcap python -u sandbox/train_footprint_model.py
"""

import gc
import os
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FootprintFoldResult:
    fold: int
    train_days: int
    test_days: int
    train_samples: int
    test_samples: int
    n_positive_test: int
    roc_auc: float
    n_trades: int
    win_rate: float
    mean_pnl: float
    total_pnl: float
    best_epoch: int


@dataclass
class FootprintResult:
    path_name: str
    fold_results: List[FootprintFoldResult]
    overall_roc_auc: float
    total_trades: int
    overall_win_rate: float
    overall_mean_pnl: float
    total_pnl: float
    all_y_true: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    all_y_prob: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    all_test_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


def _clear_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _get_fold_splits(
    days: list, n_folds: int, min_train_days: int
) -> List[Tuple[list, list]]:
    """Walk-forward expanding-window splits."""
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
    stop_pts: float = 4.0,
    target_pts: float = 6.0,
    max_bars: int = 45,
) -> Tuple[int, float, float, float]:
    """Simulate fixed-stop/target trades."""
    if len(predicted_indices) == 0:
        return 0, 0.0, 0.0, 0.0

    close = ohlcv["close"].values.astype(np.float64)
    high = ohlcv["high"].values.astype(np.float64)
    low = ohlcv["low"].values.astype(np.float64)
    trade_dir = ohlcv["trade_direction"].values
    n = len(close)

    wins = 0
    pnl_list = []

    for idx in predicted_indices:
        if idx >= n:
            continue
        direction = trade_dir[idx]
        if direction == 0:
            continue

        entry_price = close[idx]
        trade_pnl = 0.0

        for j in range(idx + 1, min(idx + max_bars + 1, n)):
            if direction == 1:
                if low[j] <= entry_price - stop_pts:
                    trade_pnl = -stop_pts
                    break
                if high[j] >= entry_price + target_pts:
                    trade_pnl = target_pts
                    break
            else:
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

    return n_trades, wins / n_trades, float(np.mean(pnl_list)), float(np.sum(pnl_list))


class FootprintTrainer:
    """Walk-forward trainer for FootprintFusionModel."""

    def __init__(
        self,
        n_folds: int = 5,
        min_train_days: int = 100,
        threshold: float = 0.50,
        random_seed: int = 42,
        model_arch: str = "cnn_fusion",
    ):
        self.n_folds = n_folds
        self.min_train_days = min_train_days
        self.threshold = threshold
        self.random_seed = random_seed
        self.model_arch = model_arch

    def _build_model(
        self,
        scalar_dim: int,
        n_price_bins: int,
        current_time_steps: int,
        context_time_steps: int,
    ):
        from strategies.reversal.footprint_bundle import build_footprint_model

        return build_footprint_model(
            model_arch=self.model_arch,
            scalar_dim=scalar_dim,
            n_price_bins=n_price_bins,
            current_time_steps=current_time_steps,
            context_time_steps=context_time_steps,
        )

    def train(
        self,
        samples_df: pd.DataFrame,
        feature_cols: List[str],
        footprints: dict,
        ohlcv: pd.DataFrame,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> FootprintResult:
        """Train FootprintFusionModel with walk-forward CV.

        Args:
            samples_df: DataFrame with outcome, trading_day, and feature columns.
            feature_cols: List of scalar feature column names.
            footprints: Dict with 'current' (N,4,20,60), 'context' (N,4,20,300),
                       'valid_mask' (N,) bool arrays.
            ohlcv: Full OHLCV DataFrame for trade simulation.
            epochs: Training epochs per fold.
            batch_size: Batch size.
            lr: Learning rate.
            verbose: Print progress.

        Returns:
            FootprintResult with per-fold and aggregate metrics.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from torch.optim.lr_scheduler import OneCycleLR
        from sklearn.metrics import roc_auc_score

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_random_seed(self.random_seed)
        if verbose:
            print(f"\n{'=' * 70}")
            print(
                f"FOOTPRINT MODEL TRAINING (device={device}, arch={self.model_arch})"
            )
            print(f"{'=' * 70}")

        # Filter to valid footprint samples
        valid_mask = footprints["valid_mask"]
        valid_indices = np.where(valid_mask)[0]

        if verbose:
            print(
                f"  Valid footprint samples: {len(valid_indices):,} / "
                f"{len(samples_df):,} ({100 * len(valid_indices) / len(samples_df):.1f}%)"
            )

        # Prepare arrays
        current_fp = footprints["current"][valid_indices]  # (N', 4, 20, 60)
        context_fp = footprints["context"][valid_indices]  # (N', 4, 20, 300)
        scalars = (
            samples_df.iloc[valid_indices][feature_cols]
            .fillna(0)
            .values.astype(np.float32)
        )
        y = (
            samples_df.iloc[valid_indices]["outcome"] == 1
        ).astype(int).values
        trading_days = samples_df.iloc[valid_indices]["trading_day"].values
        original_indices = samples_df.index[valid_indices]

        days = sorted(set(trading_days))
        splits = _get_fold_splits(days, self.n_folds, self.min_train_days)

        if verbose:
            print(f"  Features: {len(feature_cols)} scalars + footprint tensors")
            print(f"  Folds: {len(splits)}, epochs: {epochs}")

        fold_results = []
        all_y_true = []
        all_y_prob = []
        all_test_indices = []

        for fold, (train_days, test_days) in enumerate(splits):
            if verbose:
                print(
                    f"\n  Fold {fold + 1}/{len(splits)}: "
                    f"{len(train_days)} train days, {len(test_days)} test days"
                )

            train_mask = np.isin(trading_days, train_days)
            test_mask = np.isin(trading_days, test_days)

            X_sc_train = scalars[train_mask]
            X_sc_test = scalars[test_mask]
            fp_cur_train = current_fp[train_mask]
            fp_cur_test = current_fp[test_mask]
            fp_ctx_train = context_fp[train_mask]
            fp_ctx_test = context_fp[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]
            test_ohlcv_idx = original_indices[test_mask]

            if len(y_test) == 0 or y_train.sum() < 5:
                if verbose:
                    print("    Skipping fold (insufficient data)")
                continue

            # Honest early stopping: 80/20 temporal split within training
            n_train_days = len(train_days)
            val_split = int(n_train_days * 0.8)
            val_day_set = set(train_days[val_split:])
            train_day_vals = trading_days[train_mask]
            val_within = np.array([d in val_day_set for d in train_day_vals])
            tr_within = ~val_within

            model = self._build_model(
                scalar_dim=len(feature_cols),
                n_price_bins=current_fp.shape[2],
                current_time_steps=current_fp.shape[3],
                context_time_steps=context_fp.shape[3],
            )
            model.to(device)

            if fold == 0 and verbose:
                print(f"    Model params: {model.count_parameters():,}")

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
            loss_fn = nn.BCELoss()

            # Training DataLoader
            train_dataset = TensorDataset(
                torch.tensor(fp_cur_train[tr_within]),
                torch.tensor(fp_ctx_train[tr_within]),
                torch.tensor(X_sc_train[tr_within]),
                torch.tensor(y_train[tr_within].astype(np.float32)),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                generator=torch.Generator().manual_seed(self.random_seed + fold),
            )

            # Validation data
            val_cur = torch.tensor(fp_cur_train[val_within]).to(device)
            val_ctx = torch.tensor(fp_ctx_train[val_within]).to(device)
            val_sc = torch.tensor(X_sc_train[val_within]).to(device)
            val_y = y_train[val_within]

            steps_per_epoch = len(train_loader)
            if steps_per_epoch == 0:
                if verbose:
                    print("    No training batches, skipping fold")
                del model, optimizer
                _clear_gpu_memory()
                continue

            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=epochs * steps_per_epoch,
            )

            # Training loop
            best_val_loss = float("inf")
            best_epoch = 0
            patience = 10
            no_improve = 0
            best_state = None

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch in train_loader:
                    cur, ctx, sc, yb = [x.to(device) for x in batch]
                    pred = model(cur, ctx, sc)
                    loss = loss_fn(pred, yb)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                # Validation loss
                if len(val_y) > 0:
                    model.eval()
                    with torch.no_grad():
                        val_pred = model(val_cur, val_ctx, val_sc)
                        val_loss = loss_fn(
                            val_pred,
                            torch.tensor(val_y.astype(np.float32)).to(device),
                        ).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_state = {
                            k: v.cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= patience:
                        if verbose:
                            print(
                                f"    Early stop at epoch {epoch + 1} "
                                f"(best={best_epoch + 1})"
                            )
                        break

            # Restore best model
            if best_state is not None:
                model.load_state_dict(best_state)

            if verbose:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(
                    f"    Epochs: {epoch + 1}, best_epoch: {best_epoch + 1}, "
                    f"train_loss: {avg_loss:.4f}, val_loss: {best_val_loss:.4f}"
                )

            # Evaluate on test set (batched)
            model.eval()
            test_probs = []
            eval_batch = 256

            with torch.no_grad():
                for i in range(0, len(y_test), eval_batch):
                    end = min(i + eval_batch, len(y_test))
                    cur_b = torch.tensor(fp_cur_test[i:end]).to(device)
                    ctx_b = torch.tensor(fp_ctx_test[i:end]).to(device)
                    sc_b = torch.tensor(X_sc_test[i:end]).to(device)
                    pred = model(cur_b, ctx_b, sc_b)
                    test_probs.extend(pred.cpu().numpy().tolist())

            y_prob = np.array(test_probs)

            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5

            # Trade simulation at threshold
            pred_mask = y_prob >= self.threshold
            pred_indices = test_ohlcv_idx[pred_mask]
            nt, wr, mp, tp = _simulate_trades(ohlcv, pred_indices)

            if verbose:
                print(
                    f"    AUC={auc:.3f}, trades={nt:,}, "
                    f"WR={wr:.1%}, E[PnL]={mp:+.2f}"
                )

            fold_results.append(
                FootprintFoldResult(
                    fold=fold,
                    train_days=len(train_days),
                    test_days=len(test_days),
                    train_samples=int(tr_within.sum()),
                    test_samples=len(y_test),
                    n_positive_test=int(y_test.sum()),
                    roc_auc=auc,
                    n_trades=nt,
                    win_rate=wr,
                    mean_pnl=mp,
                    total_pnl=tp,
                    best_epoch=best_epoch + 1,
                )
            )

            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())
            all_test_indices.extend(test_ohlcv_idx.tolist())

            # Cleanup
            del model, optimizer, scheduler, best_state
            del val_cur, val_ctx, val_sc
            _clear_gpu_memory()

        # Aggregate
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_test_indices = np.array(all_test_indices, dtype=np.int64)

        try:
            overall_auc = roc_auc_score(all_y_true, all_y_prob)
        except ValueError:
            overall_auc = 0.5

        total_trades = sum(fr.n_trades for fr in fold_results)
        total_wins = sum(
            int(fr.win_rate * fr.n_trades) for fr in fold_results
        )
        total_pnl = sum(fr.total_pnl for fr in fold_results)

        if verbose:
            print(f"\n  Overall AUC: {overall_auc:.4f}")
            print(f"  Total trades: {total_trades:,}")
            if total_trades > 0:
                print(f"  Overall WR: {total_wins / total_trades:.1%}")
                print(f"  Overall E[PnL]: {total_pnl / total_trades:+.2f}")
                print(f"  Total PnL: {total_pnl:+.1f}")

        return FootprintResult(
            path_name=self.model_arch,
            fold_results=fold_results,
            overall_roc_auc=overall_auc,
            total_trades=total_trades,
            overall_win_rate=total_wins / max(total_trades, 1),
            overall_mean_pnl=total_pnl / max(total_trades, 1),
            total_pnl=total_pnl,
            all_y_true=all_y_true,
            all_y_prob=all_y_prob,
            all_test_indices=all_test_indices,
        )

    def train_final_model(
        self,
        samples_df: pd.DataFrame,
        feature_cols: List[str],
        footprints: dict,
        epochs: int = 25,
        batch_size: int = 256,
        lr: float = 1e-3,
        val_day_fraction: float = 0.2,
        verbose: bool = True,
    ):
        """Train one deployable model on all valid samples with temporal holdout."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim.lr_scheduler import OneCycleLR
        from sklearn.metrics import roc_auc_score

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_random_seed(self.random_seed)

        valid_mask = footprints["valid_mask"]
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid footprint samples available for final training")

        current_fp = footprints["current"][valid_indices]
        context_fp = footprints["context"][valid_indices]
        scalars = (
            samples_df.iloc[valid_indices][feature_cols]
            .fillna(0)
            .values.astype(np.float32)
        )
        y = (samples_df.iloc[valid_indices]["outcome"] == 1).astype(int).values
        trading_days = samples_df.iloc[valid_indices]["trading_day"].astype(str).values

        days = sorted(set(trading_days))
        if len(days) < 5:
            raise ValueError(
                f"Need at least 5 trading days for final holdout, got {len(days)}"
            )

        val_day_count = max(1, int(round(len(days) * val_day_fraction)))
        val_day_count = min(val_day_count, len(days) - 1)
        val_days = set(days[-val_day_count:])
        train_mask = ~np.isin(trading_days, list(val_days))
        val_mask_arr = ~train_mask

        y_train = y[train_mask]
        y_val = y[val_mask_arr]
        if y_train.sum() < 10 or len(y_val) < 32:
            raise ValueError(
                "Final split is too small; adjust data window or val_day_fraction"
            )

        model = self._build_model(
            scalar_dim=len(feature_cols),
            n_price_bins=current_fp.shape[2],
            current_time_steps=current_fp.shape[3],
            context_time_steps=context_fp.shape[3],
        )
        model.to(device)

        if verbose:
            print(f"\nTraining final deployable model ({self.model_arch}, device={device})")
            print(
                f"  samples={len(y):,}, train={int(train_mask.sum()):,}, "
                f"val={int(val_mask_arr.sum()):,}, features={len(feature_cols)}"
            )
            print(f"  model params={model.count_parameters():,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.BCELoss()

        train_dataset = TensorDataset(
            torch.tensor(current_fp[train_mask]),
            torch.tensor(context_fp[train_mask]),
            torch.tensor(scalars[train_mask]),
            torch.tensor(y_train.astype(np.float32)),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.random_seed),
        )

        val_cur = torch.tensor(current_fp[val_mask_arr]).to(device)
        val_ctx = torch.tensor(context_fp[val_mask_arr]).to(device)
        val_sc = torch.tensor(scalars[val_mask_arr]).to(device)
        val_target = torch.tensor(y_val.astype(np.float32)).to(device)

        steps_per_epoch = len(train_loader)
        if steps_per_epoch <= 0:
            raise ValueError("No training batches for final model")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=epochs * steps_per_epoch,
        )

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience = max(5, min(12, epochs // 2))
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                cur_b, ctx_b, sc_b, y_b = [x.to(device) for x in batch]
                pred = model(cur_b, ctx_b, sc_b)
                loss = loss_fn(pred, y_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            model.eval()
            with torch.no_grad():
                val_pred = model(val_cur, val_ctx, val_sc)
                val_loss = float(loss_fn(val_pred, val_target).item())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                avg_loss = epoch_loss / max(n_batches, 1)
                print(
                    f"  epoch {epoch + 1:02d}/{epochs}: "
                    f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if no_improve >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch + 1} (best={best_epoch + 1})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            val_prob = model(val_cur, val_ctx, val_sc).detach().cpu().numpy()

        try:
            val_auc = float(roc_auc_score(y_val, val_prob))
        except ValueError:
            val_auc = 0.5

        pred_mask = val_prob >= self.threshold
        n_val_trades = int(pred_mask.sum())

        summary: Dict[str, Any] = {
            "model_arch": self.model_arch,
            "device": str(device),
            "n_samples_total": int(len(y)),
            "n_train_samples": int(train_mask.sum()),
            "n_val_samples": int(val_mask_arr.sum()),
            "n_days_total": int(len(days)),
            "n_val_days": int(len(val_days)),
            "best_epoch": int(best_epoch + 1),
            "best_val_loss": float(best_val_loss),
            "val_auc": float(val_auc),
            "val_pred_threshold": float(self.threshold),
            "n_val_trades_at_threshold": n_val_trades,
        }

        return model, summary
