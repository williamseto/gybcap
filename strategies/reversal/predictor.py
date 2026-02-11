"""
Dual-path reversal predictors: XGBoost and TCN.

This module provides predictor classes for both approaches:
1. XGBoostReversalPredictor: Hand-crafted features + XGBoost
2. TCNReversalPredictor: Sequence learning on raw data
3. HybridReversalPredictor: TCN embeddings + hand-crafted features

Each predictor follows the same interface for easy comparison.
"""

from typing import Optional, Dict, List, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from xgboost import XGBClassifier

from strategies.reversal.normalization import NormalizationPipeline
from strategies.reversal.tcn_model import (
    TCNClassifier,
    TCNMultiTask,
    HybridModel,
    create_tcn_model
)


def get_device(requested_device: Optional[str] = None) -> Tuple[str, bool, int]:
    """
    Get the best available device with fallback to CPU.

    Args:
        requested_device: Explicitly requested device ('cuda', 'cpu', or None for auto)

    Returns:
        Tuple of (device_str, pin_memory, num_workers)
    """
    if requested_device == 'cpu':
        return 'cpu', False, 0

    if requested_device == 'cuda' or requested_device is None:
        if torch.cuda.is_available():
            try:
                # Test CUDA is actually working
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
                return 'cuda', True, 4
            except Exception as e:
                print(f"CUDA available but failed to initialize: {e}")
                print("Falling back to CPU")
                return 'cpu', False, 0
        else:
            if requested_device == 'cuda':
                print("CUDA requested but not available, falling back to CPU")
            return 'cpu', False, 0

    return 'cpu', False, 0


@dataclass
class PredictionResult:
    """Container for prediction results."""
    reversal_prob: np.ndarray        # Probability of any reversal
    bull_prob: np.ndarray            # Probability of bullish reversal
    bear_prob: np.ndarray            # Probability of bearish reversal
    magnitude: Optional[np.ndarray]  # Predicted move magnitude (if available)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        data = {
            'reversal_prob': self.reversal_prob,
            'bull_prob': self.bull_prob,
            'bear_prob': self.bear_prob,
        }
        if self.magnitude is not None:
            data['magnitude'] = self.magnitude
        return pd.DataFrame(data)


class BaseReversalPredictor(ABC):
    """Abstract base class for reversal predictors."""

    @abstractmethod
    def train(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train the predictor."""
        pass

    @abstractmethod
    def predict(self, bars: pd.DataFrame) -> PredictionResult:
        """Make predictions on new data."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from file."""
        pass


class XGBoostReversalPredictor(BaseReversalPredictor):
    """
    Feature-based prediction using XGBoost.

    Features used:
    - Higher TF context (daily RSI, gap, prior day type)
    - Intraday context (VWAP deviation, IB extension, session range)
    - Price level features (existing: ovn_lo_z, vwap_z, etc.)
    - Volume features (existing: vol_at_level, delta, etc.)
    - Reversion quality features (existing: wick ratio, etc.)
    """

    DEFAULT_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 3,  # [none, bull, bear]
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'n_estimators': 500,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'random_state': 42,
    }

    def __init__(
        self,
        feature_cols: List[str],
        params: Optional[Dict[str, Any]] = None,
        threshold: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            feature_cols: List of feature column names to use
            params: XGBoost parameters (overrides defaults)
            threshold: Probability threshold for prediction
        """
        self.feature_cols = feature_cols
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.threshold = threshold

        self.model: Optional[XGBClassifier] = None
        self.normalizer = NormalizationPipeline()
        self.training_results: Dict[str, Any] = {}

    def train(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            ohlcv: OHLCV DataFrame (used if features_df is None)
            labels: DataFrame with reversal_label column
            features_df: Pre-computed features DataFrame

        Returns:
            Training results dictionary
        """
        # Merge labels with features
        if features_df is not None:
            df = features_df.copy()
        else:
            df = ohlcv.copy()

        df['reversal_label'] = labels['reversal_label'].values

        # Map labels to multiclass: -1 (bear) -> 2, 0 (none) -> 0, 1 (bull) -> 1
        y = df['reversal_label'].values.copy()
        y[y == -1] = 2

        # Prepare features
        X_df = df[self.feature_cols].fillna(0)

        # Fit normalizer and transform
        self.normalizer.fit(X_df)
        X_norm = self.normalizer.normalize_for_xgboost(X_df)
        X = X_norm.values

        # Train model
        self.model = XGBClassifier(**self.params)
        self.model.fit(X, y)

        # Evaluate on training data
        y_proba = self.model.predict_proba(X)
        y_pred = y_proba.argmax(axis=1)

        accuracy = (y_pred == y).mean()
        reversal_mask = y != 0
        reversal_accuracy = (y_pred[reversal_mask] == y[reversal_mask]).mean()

        self.training_results = {
            'n_samples': len(y),
            'n_reversals': reversal_mask.sum(),
            'accuracy': accuracy,
            'reversal_accuracy': reversal_accuracy,
            'class_distribution': {
                'none': (y == 0).sum(),
                'bull': (y == 1).sum(),
                'bear': (y == 2).sum()
            }
        }

        return self.training_results

    def predict(self, bars: pd.DataFrame) -> PredictionResult:
        """
        Make predictions.

        Args:
            bars: DataFrame with feature columns

        Returns:
            PredictionResult with probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_df = bars[self.feature_cols].fillna(0)
        X_norm = self.normalizer.normalize_for_xgboost(X_df)
        X = X_norm.values

        proba = self.model.predict_proba(X)

        # proba columns: [none, bull, bear]
        return PredictionResult(
            reversal_prob=1.0 - proba[:, 0],
            bull_prob=proba[:, 1],
            bear_prob=proba[:, 2],
            magnitude=None
        )

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained.")
        self.model.save_model(path)
        # Also save normalizer stats
        self.normalizer.save_stats(path.replace('.json', '_norm.json'))

    def load(self, path: str) -> None:
        """Load model from file."""
        self.model = XGBClassifier()
        self.model.load_model(path)
        # Load normalizer stats if available
        try:
            self.normalizer.load_stats(path.replace('.json', '_norm.json'))
        except FileNotFoundError:
            pass


class ReversalSequenceDataset(Dataset):
    """Dataset for TCN reversal prediction."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        magnitudes: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.

        Args:
            sequences: (N, seq_len, channels) array
            labels: (N,) array of multiclass labels (0=none, 1=bull, 2=bear)
            magnitudes: Optional (N,) array of move magnitudes
        """
        self.sequences = torch.from_numpy(sequences).float()
        self.labels = torch.from_numpy(labels).long()
        self.magnitudes = (
            torch.from_numpy(magnitudes).float()
            if magnitudes is not None else None
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.magnitudes is not None:
            mag = self.magnitudes[idx]
            return seq, label, mag
        else:
            return seq, label


class TCNReversalPredictor(BaseReversalPredictor):
    """
    Sequence-based prediction using TCN.

    Input: Last N bars of OHLCV + delta + VAP bins (normalized)
    Output: Reversal probability + magnitude
    """

    def __init__(
        self,
        lookback_bars: int = 60,
        hidden_channels: int = 64,
        num_levels: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 30,
        device: Optional[str] = None
    ):
        """
        Initialize predictor.

        Args:
            lookback_bars: Number of bars in input sequence
            hidden_channels: TCN hidden dimension
            num_levels: Number of TCN levels
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            num_epochs: Number of training epochs
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.lookback_bars = lookback_bars
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Device selection with fallback
        self.device, self.pin_memory, self.num_workers = get_device(device)

        self.model: Optional[nn.Module] = None
        self.normalizer = NormalizationPipeline()
        self.training_results: Dict[str, Any] = {}

        # Will be set after seeing data
        self.n_input_channels: Optional[int] = None

    def _prepare_sequences(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequence data for training.

        Returns:
            sequences: (N, lookback, channels) array
            multiclass_labels: (N,) array
            magnitudes: (N,) array
        """
        # Import microstructure provider for sequence generation
        from strategies.features.microstructure import MicrostructureSequenceProvider

        # Compute sequences
        micro_provider = MicrostructureSequenceProvider(
            lookback_bars=self.lookback_bars,
            include_bidask='bidvolume' in ohlcv.columns
        )
        micro_provider._compute_impl(ohlcv)

        # Get all combined sequences
        all_sequences = micro_provider.get_all_combined_sequences()
        self.n_input_channels = micro_provider.n_input_channels

        # Get labels
        multiclass_labels = labels['reversal_label'].values.copy()
        multiclass_labels[multiclass_labels == -1] = 2  # Map bear to 2

        magnitudes = labels['reversal_magnitude'].values

        # Filter out sequences that are all zeros (not enough lookback)
        valid_mask = np.abs(all_sequences).sum(axis=(1, 2)) > 0
        sequences = all_sequences[valid_mask]
        multiclass_labels = multiclass_labels[valid_mask]
        magnitudes = magnitudes[valid_mask]

        # Normalize sequences
        sequences = self.normalizer.normalize_for_tcn(sequences)

        return sequences, multiclass_labels, magnitudes

    def train(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None  # Not used for TCN
    ) -> Dict[str, Any]:
        """
        Train TCN model.

        Args:
            ohlcv: OHLCV DataFrame
            labels: DataFrame with reversal labels

        Returns:
            Training results
        """
        # Prepare sequences
        sequences, y_labels, magnitudes = self._prepare_sequences(ohlcv, labels)

        print(f"Prepared {len(sequences)} sequences with {self.n_input_channels} channels")

        # Create dataset
        dataset = ReversalSequenceDataset(sequences, y_labels, magnitudes)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

        # Create model
        self.model = TCNMultiTask(
            input_channels=self.n_input_channels,
            hidden_channels=self.hidden_channels,
            num_levels=self.num_levels,
            dropout=self.dropout
        ).to(self.device)

        # Loss functions
        cls_loss_fn = nn.CrossEntropyLoss()
        reg_loss_fn = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # Training loop
        self.model.train()
        losses = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_reg_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                seq, label, mag = batch
                seq = seq.to(self.device)
                label = label.to(self.device)
                mag = mag.to(self.device)

                optimizer.zero_grad()

                # Forward
                _, direction_probs, pred_mag = self.model(seq)

                # Losses
                cls_loss = cls_loss_fn(direction_probs, label)

                # Only compute regression loss for actual reversals
                reversal_mask = label != 0
                if reversal_mask.sum() > 0:
                    reg_loss = reg_loss_fn(
                        pred_mag[reversal_mask],
                        mag[reversal_mask]
                    )
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)

                # Combined loss
                loss = cls_loss + 0.1 * reg_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_reg_loss += reg_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_cls = epoch_cls_loss / max(n_batches, 1)
            avg_reg = epoch_reg_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}: "
                      f"loss={avg_loss:.4f}, cls={avg_cls:.4f}, reg={avg_reg:.4f}")

        # Evaluate on full data
        self.model.eval()
        with torch.no_grad():
            all_seq = torch.from_numpy(sequences).float().to(self.device)
            _, probs, _ = self.model(all_seq)
            preds = probs.argmax(dim=-1).cpu().numpy()

        accuracy = (preds == y_labels).mean()
        reversal_mask = y_labels != 0
        reversal_accuracy = (preds[reversal_mask] == y_labels[reversal_mask]).mean()

        self.training_results = {
            'n_samples': len(sequences),
            'n_reversals': reversal_mask.sum(),
            'accuracy': accuracy,
            'reversal_accuracy': reversal_accuracy,
            'final_loss': losses[-1],
            'class_distribution': {
                'none': (y_labels == 0).sum(),
                'bull': (y_labels == 1).sum(),
                'bear': (y_labels == 2).sum()
            }
        }

        return self.training_results

    def predict(self, bars: pd.DataFrame) -> PredictionResult:
        """Make predictions with batched inference to avoid OOM."""
        if self.model is None:
            raise ValueError("Model not trained.")

        # Prepare sequences
        from strategies.features.microstructure import MicrostructureSequenceProvider

        micro_provider = MicrostructureSequenceProvider(
            lookback_bars=self.lookback_bars,
            include_bidask='bidvolume' in bars.columns
        )
        micro_provider._compute_impl(bars)

        sequences = micro_provider.get_all_combined_sequences()
        sequences = self.normalizer.normalize_for_tcn(sequences)

        # Batched prediction to avoid OOM
        self.model.eval()
        batch_size = 1024  # Smaller batch for inference
        n_samples = len(sequences)

        all_rev_prob = []
        all_direction_probs = []
        all_magnitude = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_seq = sequences[start_idx:end_idx]
                seq_tensor = torch.from_numpy(batch_seq).float().to(self.device)

                rev_prob, direction_probs, magnitude = self.model(seq_tensor)

                all_rev_prob.append(rev_prob.cpu().numpy())
                all_direction_probs.append(direction_probs.cpu().numpy())
                all_magnitude.append(magnitude.cpu().numpy())

        # Concatenate all batches
        import numpy as np
        all_rev_prob = np.concatenate(all_rev_prob)
        all_direction_probs = np.concatenate(all_direction_probs)
        all_magnitude = np.concatenate(all_magnitude)

        return PredictionResult(
            reversal_prob=all_rev_prob,
            bull_prob=all_direction_probs[:, 1],
            bear_prob=all_direction_probs[:, 2],
            magnitude=all_magnitude
        )

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        torch.save({
            'model_state': self.model.state_dict(),
            'n_input_channels': self.n_input_channels,
            'hidden_channels': self.hidden_channels,
            'num_levels': self.num_levels,
            'lookback_bars': self.lookback_bars,
        }, path)

    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.n_input_channels = checkpoint['n_input_channels']
        self.hidden_channels = checkpoint['hidden_channels']
        self.num_levels = checkpoint['num_levels']
        self.lookback_bars = checkpoint['lookback_bars']

        self.model = TCNMultiTask(
            input_channels=self.n_input_channels,
            hidden_channels=self.hidden_channels,
            num_levels=self.num_levels,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])


class HybridReversalPredictor(BaseReversalPredictor):
    """
    Hybrid predictor combining TCN embeddings with hand-crafted features.

    This model:
    1. Encodes sequences with TCN
    2. Concatenates with higher-timeframe features
    3. Uses MLP for final prediction

    Best of both: learns patterns from raw data + leverages domain knowledge.
    """

    def __init__(
        self,
        feature_cols: List[str],
        lookback_bars: int = 60,
        tcn_hidden: int = 64,
        tcn_levels: int = 4,
        fusion_hidden: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 30,
        device: Optional[str] = None
    ):
        """Initialize hybrid predictor."""
        self.feature_cols = feature_cols
        self.lookback_bars = lookback_bars
        self.tcn_hidden = tcn_hidden
        self.tcn_levels = tcn_levels
        self.fusion_hidden = fusion_hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Device selection with fallback
        self.device, self.pin_memory, self.num_workers = get_device(device)

        self.model: Optional[HybridModel] = None
        self.normalizer = NormalizationPipeline()
        self.training_results: Dict[str, Any] = {}
        self.n_input_channels: Optional[int] = None

    def train(
        self,
        ohlcv: pd.DataFrame,
        labels: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Train hybrid model."""
        if features_df is None:
            raise ValueError("Hybrid model requires features_df")

        # Prepare sequences
        from strategies.features.microstructure import MicrostructureSequenceProvider

        micro_provider = MicrostructureSequenceProvider(
            lookback_bars=self.lookback_bars,
            include_bidask='bidvolume' in ohlcv.columns
        )
        micro_provider._compute_impl(ohlcv)
        sequences = micro_provider.get_all_combined_sequences()
        self.n_input_channels = micro_provider.n_input_channels

        # Prepare features
        X_df = features_df[self.feature_cols].fillna(0)
        self.normalizer.fit(X_df)
        X_norm = self.normalizer.transform(X_df)
        htf_features = X_norm.values

        # Get labels
        y_labels = labels['reversal_label'].values.copy()
        y_labels[y_labels == -1] = 2
        magnitudes = labels['reversal_magnitude'].values

        # Filter valid sequences
        valid_mask = np.abs(sequences).sum(axis=(1, 2)) > 0
        sequences = sequences[valid_mask]
        htf_features = htf_features[valid_mask]
        y_labels = y_labels[valid_mask]
        magnitudes = magnitudes[valid_mask]

        # Normalize sequences
        sequences = self.normalizer.normalize_for_tcn(sequences)

        print(f"Prepared {len(sequences)} samples with "
              f"{self.n_input_channels} seq channels and {len(self.feature_cols)} HTF features")

        # Create model
        self.model = HybridModel(
            seq_input_channels=self.n_input_channels,
            htf_feature_dim=len(self.feature_cols),
            tcn_hidden=self.tcn_hidden,
            tcn_levels=self.tcn_levels,
            fusion_hidden=self.fusion_hidden,
            dropout=self.dropout
        ).to(self.device)

        # Training
        cls_loss_fn = nn.CrossEntropyLoss()
        reg_loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Convert to tensors
        seq_tensor = torch.from_numpy(sequences).float()
        htf_tensor = torch.from_numpy(htf_features).float()
        y_tensor = torch.from_numpy(y_labels).long()
        mag_tensor = torch.from_numpy(magnitudes).float()

        dataset = torch.utils.data.TensorDataset(
            seq_tensor, htf_tensor, y_tensor, mag_tensor
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for seq, htf, label, mag in dataloader:
                seq = seq.to(self.device)
                htf = htf.to(self.device)
                label = label.to(self.device)
                mag = mag.to(self.device)

                optimizer.zero_grad()

                _, direction_probs, pred_mag = self.model(seq, htf)

                cls_loss = cls_loss_fn(direction_probs, label)

                reversal_mask = label != 0
                if reversal_mask.sum() > 0:
                    reg_loss = reg_loss_fn(pred_mag[reversal_mask], mag[reversal_mask])
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)

                loss = cls_loss + 0.1 * reg_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}: loss={epoch_loss / n_batches:.4f}")

        # Evaluate
        self.model.eval()
        with torch.no_grad():
            _, probs, _ = self.model(
                seq_tensor.to(self.device),
                htf_tensor.to(self.device)
            )
            preds = probs.argmax(dim=-1).cpu().numpy()

        accuracy = (preds == y_labels).mean()
        reversal_mask = y_labels != 0
        reversal_accuracy = (preds[reversal_mask] == y_labels[reversal_mask]).mean()

        self.training_results = {
            'n_samples': len(sequences),
            'n_reversals': reversal_mask.sum(),
            'accuracy': accuracy,
            'reversal_accuracy': reversal_accuracy,
        }

        return self.training_results

    def predict(self, bars: pd.DataFrame) -> PredictionResult:
        """Make predictions with batched inference to avoid OOM."""
        if self.model is None:
            raise ValueError("Model not trained.")

        # Prepare sequences
        from strategies.features.microstructure import MicrostructureSequenceProvider

        micro_provider = MicrostructureSequenceProvider(
            lookback_bars=self.lookback_bars,
            include_bidask='bidvolume' in bars.columns
        )
        micro_provider._compute_impl(bars)
        sequences = micro_provider.get_all_combined_sequences()
        sequences = self.normalizer.normalize_for_tcn(sequences)

        # Prepare features
        htf_features = bars[self.feature_cols].fillna(0)
        htf_norm = self.normalizer.transform(htf_features)
        htf_values = htf_norm.values

        # Batched prediction to avoid OOM
        self.model.eval()
        batch_size = 1024
        n_samples = len(sequences)

        all_rev_prob = []
        all_direction_probs = []
        all_magnitude = []

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_seq = sequences[start_idx:end_idx]
                batch_htf = htf_values[start_idx:end_idx]

                seq_tensor = torch.from_numpy(batch_seq).float().to(self.device)
                htf_tensor = torch.from_numpy(batch_htf).float().to(self.device)

                rev_prob, direction_probs, magnitude = self.model(seq_tensor, htf_tensor)

                all_rev_prob.append(rev_prob.cpu().numpy())
                all_direction_probs.append(direction_probs.cpu().numpy())
                all_magnitude.append(magnitude.cpu().numpy())

        # Concatenate all batches
        import numpy as np
        all_rev_prob = np.concatenate(all_rev_prob)
        all_direction_probs = np.concatenate(all_direction_probs)
        all_magnitude = np.concatenate(all_magnitude)

        return PredictionResult(
            reversal_prob=all_rev_prob,
            bull_prob=all_direction_probs[:, 1],
            bear_prob=all_direction_probs[:, 2],
            magnitude=all_magnitude
        )

    def save(self, path: str) -> None:
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained.")
        torch.save({
            'model_state': self.model.state_dict(),
            'n_input_channels': self.n_input_channels,
            'feature_cols': self.feature_cols,
        }, path)
        self.normalizer.save_stats(path.replace('.pt', '_norm.json'))

    def load(self, path: str) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.n_input_channels = checkpoint['n_input_channels']
        self.feature_cols = checkpoint['feature_cols']

        self.model = HybridModel(
            seq_input_channels=self.n_input_channels,
            htf_feature_dim=len(self.feature_cols),
            tcn_hidden=self.tcn_hidden,
            tcn_levels=self.tcn_levels,
            fusion_hidden=self.fusion_hidden,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        self.normalizer.load_stats(path.replace('.pt', '_norm.json'))
