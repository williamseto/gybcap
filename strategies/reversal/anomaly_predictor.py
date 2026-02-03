"""
Anomaly detection-based reversal predictor.

Key insight: Instead of training a classifier on 499:1 imbalanced data,
train on reversal data ONLY and detect similar patterns.

The model learns "what reversals look like" rather than trying to
distinguish rare positives from massive negatives.
"""

from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from strategies.reversal.autoencoder import (
    FeatureOnlyAutoencoder,
    HybridReversalAutoencoder,
    create_autoencoder
)


@dataclass
class AnomalyPredictionResult:
    """Container for anomaly prediction results."""
    anomaly_score: np.ndarray      # Higher = more like a reversal
    is_reversal: np.ndarray        # Binary prediction based on threshold
    threshold: float               # Threshold used for prediction
    direction: Optional[np.ndarray] = None  # 1=bull, -1=bear, 0=none


class AnomalyReversalPredictor:
    """
    Anomaly detection-based reversal predictor.

    Train on reversal data ONLY, detect similar patterns.

    Supports two modes:
    - Feature-only mode: Just hand-crafted features (fast baseline)
    - Hybrid mode: Sequences + features (recommended for best recall)

    Key differences from classification approach:
    1. Train on positive class (reversals) only
    2. Use reconstruction error as anomaly score
    3. Tune threshold to achieve target recall
    4. Optionally weight by Dalton day type
    """

    # Dalton day type weights for threshold adjustment
    DALTON_WEIGHTS = {
        'trend': 1.2,      # Harder threshold (fewer signals) on trend days
        'normal': 0.9,     # Easier threshold (more signals) on balance days
        'double': 1.0,     # Standard threshold
        'p_shape': 1.0,
        'b_shape': 1.0,
    }

    def __init__(
        self,
        feature_cols: List[str],
        use_sequences: bool = False,
        seq_lookback: int = 60,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        tcn_hidden: int = 32,
        dropout: float = 0.2,
        dalton_integration: str = 'weighted',  # 'none', 'weighted', 'gated'
        device: Optional[str] = None
    ):
        """
        Initialize anomaly predictor.

        Args:
            feature_cols: List of feature column names to use
            use_sequences: Whether to use hybrid mode with sequences
            seq_lookback: Number of bars in sequence lookback
            latent_dim: Dimension of autoencoder latent space
            hidden_dim: Hidden dimension for feature encoder
            tcn_hidden: Hidden dimension for TCN (if using sequences)
            dropout: Dropout rate
            dalton_integration: How to integrate Dalton day types
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.feature_cols = feature_cols
        self.use_sequences = use_sequences
        self.seq_lookback = seq_lookback
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.tcn_hidden = tcn_hidden
        self.dropout = dropout
        self.dalton_integration = dalton_integration

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.autoencoder: Optional[nn.Module] = None
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self.training_stats: Dict[str, Any] = {}

        # For sequence mode
        self.microstructure_provider = None
        self.n_seq_channels: Optional[int] = None

    def train(
        self,
        ohlcv_df: pd.DataFrame,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        target_recall: float = 0.7,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train autoencoder on REVERSAL samples only.

        Args:
            ohlcv_df: OHLCV DataFrame
            features_df: Pre-computed features DataFrame
            labels_df: Labels DataFrame with reversal_label column
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Training batch size
            target_recall: Target recall for threshold tuning
            verbose: Print progress

        Returns:
            Training statistics dictionary
        """
        # Extract reversal samples only
        reversal_mask = labels_df['reversal_label'] != 0
        reversal_indices = reversal_mask[reversal_mask].index.tolist()
        n_reversals = len(reversal_indices)

        if verbose:
            print(f"Training on {n_reversals} reversal samples only")
            print(f"  Bull: {(labels_df['reversal_label'] == 1).sum()}")
            print(f"  Bear: {(labels_df['reversal_label'] == -1).sum()}")

        # Prepare features for reversal samples
        reversal_features = features_df.loc[reversal_indices, self.feature_cols].fillna(0)
        X_feat = self.scaler.fit_transform(reversal_features.values)
        X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32)

        # Prepare sequences if using hybrid mode
        if self.use_sequences:
            from strategies.features.microstructure import MicrostructureSequenceProvider

            self.microstructure_provider = MicrostructureSequenceProvider(
                lookback_bars=self.seq_lookback,
                include_bidask='bidvolume' in ohlcv_df.columns
            )
            self.microstructure_provider._compute_impl(ohlcv_df)

            # Get all sequences, then filter to reversal indices
            all_sequences = self.microstructure_provider.get_all_combined_sequences()
            self.n_seq_channels = self.microstructure_provider.n_input_channels

            # Map reversal indices to integer positions
            reversal_positions = [ohlcv_df.index.get_loc(idx) for idx in reversal_indices]
            X_seq = all_sequences[reversal_positions]

            # Normalize sequences per-sample
            X_seq = self._normalize_sequences(X_seq)
            X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)

            if verbose:
                print(f"  Sequence shape: {X_seq.shape}")
                print(f"  Feature shape: {X_feat.shape}")

            # Create hybrid autoencoder
            self.autoencoder = HybridReversalAutoencoder(
                seq_channels=self.n_seq_channels,
                seq_length=self.seq_lookback,
                feature_dim=len(self.feature_cols),
                latent_dim=self.latent_dim,
                tcn_hidden=self.tcn_hidden,
                dropout=self.dropout
            ).to(self.device)

            dataset = TensorDataset(X_seq_tensor, X_feat_tensor)
        else:
            # Create feature-only autoencoder
            self.autoencoder = FeatureOnlyAutoencoder(
                input_dim=len(self.feature_cols),
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout
            ).to(self.device)

            dataset = TensorDataset(X_feat_tensor,)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=min(batch_size, n_reversals),
            shuffle=True,
            drop_last=False
        )

        # Training loop
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        losses = []

        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                optimizer.zero_grad()

                if self.use_sequences:
                    seq, feat = batch
                    seq = seq.to(self.device)
                    feat = feat.to(self.device)
                    error = self.autoencoder.reconstruction_error(seq, feat)
                else:
                    feat = batch[0].to(self.device)
                    error = self.autoencoder.reconstruction_error(feat)

                loss = error.mean()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.6f}")

        # Compute threshold from training data (reversal samples)
        self.autoencoder.eval()
        with torch.no_grad():
            if self.use_sequences:
                X_seq_tensor = X_seq_tensor.to(self.device)
                X_feat_tensor = X_feat_tensor.to(self.device)
                train_scores = self.autoencoder.anomaly_score(X_seq_tensor, X_feat_tensor)
            else:
                X_feat_tensor = X_feat_tensor.to(self.device)
                train_scores = self.autoencoder.anomaly_score(X_feat_tensor)
            train_scores = train_scores.cpu().numpy()

        # Set threshold to achieve target recall on training data
        # Since we train on reversals only, threshold = percentile of scores
        # 70% recall means keeping 70% of reversals above threshold
        # So threshold = 30th percentile of reversal scores
        percentile = (1.0 - target_recall) * 100
        self.threshold = np.percentile(train_scores, percentile)

        if verbose:
            print(f"\nThreshold set at {percentile:.0f}th percentile: {self.threshold:.4f}")
            print(f"  Score range on reversals: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
            print(f"  Score mean: {train_scores.mean():.4f}")

        self.training_stats = {
            'n_reversals': n_reversals,
            'n_bull': (labels_df['reversal_label'] == 1).sum(),
            'n_bear': (labels_df['reversal_label'] == -1).sum(),
            'final_loss': losses[-1],
            'threshold': self.threshold,
            'target_recall': target_recall,
            'train_score_mean': train_scores.mean(),
            'train_score_std': train_scores.std(),
            'train_score_min': train_scores.min(),
            'train_score_max': train_scores.max(),
        }

        return self.training_stats

    def predict(
        self,
        ohlcv_df: pd.DataFrame,
        features_df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> AnomalyPredictionResult:
        """
        Predict reversal probability for all bars.

        Args:
            ohlcv_df: OHLCV DataFrame
            features_df: Pre-computed features DataFrame
            threshold: Override threshold (default: use trained threshold)

        Returns:
            AnomalyPredictionResult with scores and predictions
        """
        if self.autoencoder is None:
            raise ValueError("Model not trained. Call train() first.")

        if threshold is None:
            threshold = self.threshold

        self.autoencoder.eval()

        # Prepare features
        X_feat = features_df[self.feature_cols].fillna(0).values
        X_feat = self.scaler.transform(X_feat)
        X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if self.use_sequences:
                # Get sequences
                if self.microstructure_provider is None:
                    from strategies.features.microstructure import MicrostructureSequenceProvider
                    self.microstructure_provider = MicrostructureSequenceProvider(
                        lookback_bars=self.seq_lookback,
                        include_bidask='bidvolume' in ohlcv_df.columns
                    )
                    self.microstructure_provider._compute_impl(ohlcv_df)

                all_sequences = self.microstructure_provider.get_all_combined_sequences()
                X_seq = self._normalize_sequences(all_sequences)
                X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

                scores = self.autoencoder.anomaly_score(X_seq_tensor, X_feat_tensor)
            else:
                scores = self.autoencoder.anomaly_score(X_feat_tensor)

            scores = scores.cpu().numpy()

        # Apply Dalton weighting if enabled
        if self.dalton_integration == 'weighted':
            scores = self._apply_dalton_weights(scores, features_df)

        # Apply threshold
        is_reversal = scores >= threshold

        return AnomalyPredictionResult(
            anomaly_score=scores,
            is_reversal=is_reversal,
            threshold=threshold
        )

    def _normalize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Normalize sequences per-sample for stable training.

        Args:
            sequences: Array of shape (N, seq_len, channels)

        Returns:
            Normalized sequences
        """
        result = sequences.copy()
        N, seq_len, channels = result.shape

        for i in range(N):
            for c in range(channels):
                seq = result[i, :, c]
                mean = np.mean(seq)
                std = np.std(seq)
                if std > 1e-6:
                    result[i, :, c] = (seq - mean) / std
                else:
                    result[i, :, c] = 0

        return result

    def _apply_dalton_weights(
        self,
        scores: np.ndarray,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Weight scores based on Dalton day type.

        Trend days: harder threshold (need higher score)
        Balance days: easier threshold (lower score OK)

        Args:
            scores: Raw anomaly scores
            features_df: Features DataFrame with Dalton features

        Returns:
            Weighted scores
        """
        weights = np.ones(len(scores))

        # Check if Dalton features exist
        if 'dalton_trend_prob' not in features_df.columns:
            return scores

        # Trend days: harder threshold (reduce effective score)
        trend_prob = features_df['dalton_trend_prob'].fillna(0).values
        trend_mask = trend_prob > 0.6
        weights[trend_mask] *= self.DALTON_WEIGHTS['trend']

        # Balance days: easier threshold (boost effective score)
        if 'dalton_balance_prob' in features_df.columns:
            balance_prob = features_df['dalton_balance_prob'].fillna(0).values
            balance_mask = balance_prob > 0.6
            weights[balance_mask] *= self.DALTON_WEIGHTS['normal']

        # Apply weights: divide scores by weights
        # Higher weight = need higher raw score to pass threshold
        return scores / weights

    def tune_threshold_for_recall(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        target_recall: float = 0.7
    ) -> Tuple[float, float, float]:
        """
        Find threshold that achieves target recall.

        Args:
            scores: Anomaly scores
            y_true: True labels (binary: 1=reversal, 0=not)
            target_recall: Target recall to achieve

        Returns:
            threshold: Score threshold
            precision: Precision at this threshold
            f1: F1 score at this threshold
        """
        # Sort scores descending
        sorted_indices = np.argsort(-scores)

        n_positives = y_true.sum()
        if n_positives == 0:
            return scores.min(), 0.0, 0.0

        # Walk through thresholds until target recall achieved
        tp = 0
        threshold = scores.max()
        precision = 0.0

        for i, idx in enumerate(sorted_indices):
            if y_true[idx]:
                tp += 1
            recall = tp / n_positives
            if recall >= target_recall:
                threshold = scores[idx]
                precision = tp / (i + 1) if (i + 1) > 0 else 0.0
                break

        # Calculate F1
        if precision + target_recall > 0:
            f1 = 2 * (precision * target_recall) / (precision + target_recall)
        else:
            f1 = 0.0

        return threshold, precision, f1

    def evaluate(
        self,
        ohlcv_df: pd.DataFrame,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate predictor on labeled data.

        Args:
            ohlcv_df: OHLCV DataFrame
            features_df: Features DataFrame
            labels_df: Labels DataFrame with reversal_label
            threshold: Override threshold

        Returns:
            Dictionary of evaluation metrics
        """
        result = self.predict(ohlcv_df, features_df, threshold)

        y_true = (labels_df['reversal_label'] != 0).astype(int).values
        y_pred = result.is_reversal.astype(int)

        # Basic metrics
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        accuracy = (tp + tn) / len(y_true)

        # Daily false positive rate
        n_days = features_df['trading_day'].nunique() if 'trading_day' in features_df.columns else 1
        daily_fp = fp / n_days

        return {
            'threshold': result.threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'n_predictions': int(y_pred.sum()),
            'daily_fp_rate': daily_fp,
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        if self.autoencoder is None:
            raise ValueError("Model not trained.")

        import json

        # Save model state
        torch.save({
            'model_state': self.autoencoder.state_dict(),
            'use_sequences': self.use_sequences,
            'feature_cols': self.feature_cols,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'tcn_hidden': self.tcn_hidden,
            'seq_lookback': self.seq_lookback,
            'n_seq_channels': self.n_seq_channels,
            'threshold': self.threshold,
            'training_stats': self.training_stats,
        }, path)

        # Save scaler separately
        scaler_path = path.replace('.pt', '_scaler.json')
        scaler_stats = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_stats, f)

    def load(self, path: str) -> None:
        """Load model from file."""
        import json

        checkpoint = torch.load(path, map_location=self.device)

        self.use_sequences = checkpoint['use_sequences']
        self.feature_cols = checkpoint['feature_cols']
        self.latent_dim = checkpoint['latent_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.tcn_hidden = checkpoint['tcn_hidden']
        self.seq_lookback = checkpoint['seq_lookback']
        self.n_seq_channels = checkpoint.get('n_seq_channels')
        self.threshold = checkpoint['threshold']
        self.training_stats = checkpoint['training_stats']

        # Recreate model
        if self.use_sequences:
            self.autoencoder = HybridReversalAutoencoder(
                seq_channels=self.n_seq_channels,
                seq_length=self.seq_lookback,
                feature_dim=len(self.feature_cols),
                latent_dim=self.latent_dim,
                tcn_hidden=self.tcn_hidden,
            ).to(self.device)
        else:
            self.autoencoder = FeatureOnlyAutoencoder(
                input_dim=len(self.feature_cols),
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

        self.autoencoder.load_state_dict(checkpoint['model_state'])

        # Load scaler
        scaler_path = path.replace('.pt', '_scaler.json')
        try:
            with open(scaler_path, 'r') as f:
                scaler_stats = json.load(f)
            self.scaler.mean_ = np.array(scaler_stats['mean'])
            self.scaler.scale_ = np.array(scaler_stats['scale'])
        except FileNotFoundError:
            pass
