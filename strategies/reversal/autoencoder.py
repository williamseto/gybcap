"""
Autoencoder models for anomaly-based reversal detection.

Key insight: Train on reversal samples ONLY to learn what reversals look like.
At inference, LOW reconstruction error = looks like a known reversal pattern.

Two architectures:
1. FeatureOnlyAutoencoder: Simple MLP on hand-crafted features (baseline)
2. HybridReversalAutoencoder: TCN for sequences + MLP for features (recommended)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from strategies.reversal.tcn_model import TCNEncoder


class FeatureOnlyAutoencoder(nn.Module):
    """
    Simple feature-based autoencoder as baseline.

    Architecture:
        Input (feature_dim) -> Encoder -> Latent (latent_dim) -> Decoder -> Output (feature_dim)

    Uses symmetric encoder/decoder with dropout for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        """
        Initialize feature-only autoencoder.

        Args:
            input_dim: Number of input features
            latent_dim: Dimension of latent space
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input -> hidden -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder: latent -> hidden -> input (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features to latent space.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Latent tensor of shape (batch, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstructed features.

        Args:
            z: Latent tensor of shape (batch, latent_dim)

        Returns:
            Reconstructed features of shape (batch, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Reconstructed tensor of shape (batch, input_dim)
        """
        z = self.encode(x)
        return self.decode(z)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error (MSE).

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Error tensor of shape (batch,)
        """
        x_recon = self.forward(x)
        return torch.mean((x - x_recon) ** 2, dim=1)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score (higher = more like a reversal).

        Since we train on reversals only, LOW reconstruction error means
        the input looks like a reversal. We invert to get a score where
        HIGHER = more likely reversal.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Score tensor of shape (batch,)
        """
        error = self.reconstruction_error(x)
        # Transform: low error -> high score
        # Using 1/(1+error) for bounded [0, 1] output
        return 1.0 / (1.0 + error)


class HybridReversalAutoencoder(nn.Module):
    """
    Hybrid autoencoder combining sequence learning + hand-crafted features.

    Trained ONLY on reversal patterns.
    LOW reconstruction error = looks like a reversal.

    Architecture:
        [60-bar Sequence] -> TCNEncoder -> seq_embedding
        [Hand-crafted Features] -> FeatureEncoder -> feat_embedding
        [Concat(seq_emb, feat_emb)] -> FusionLayer -> latent
        latent -> SeqDecoder -> reconstructed sequence
        latent -> FeatDecoder -> reconstructed features

    Loss = alpha * seq_recon_error + (1 - alpha) * feature_recon_error
    """

    def __init__(
        self,
        seq_channels: int = 16,
        seq_length: int = 60,
        feature_dim: int = 18,
        latent_dim: int = 32,
        tcn_hidden: int = 32,
        tcn_levels: int = 3,
        dropout: float = 0.2,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid autoencoder.

        Args:
            seq_channels: Number of channels in sequence input (OHLCV + delta + VAP)
            seq_length: Length of input sequence (lookback bars)
            feature_dim: Number of hand-crafted features
            latent_dim: Dimension of fused latent space
            tcn_hidden: TCN encoder output dimension
            tcn_levels: Number of TCN levels
            dropout: Dropout rate
            alpha: Weight for sequence loss vs feature loss
        """
        super().__init__()

        self.seq_channels = seq_channels
        self.seq_length = seq_length
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.tcn_hidden = tcn_hidden
        self.alpha = alpha

        # Sequence encoder: reuse existing TCNEncoder
        self.seq_encoder = TCNEncoder(
            input_channels=seq_channels,
            hidden_channels=tcn_hidden,
            num_levels=tcn_levels,
            dropout=dropout,
            pool_type='max'
        )

        # Feature encoder: simple MLP
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden, tcn_hidden)
        )

        # Fusion layer: combine sequence + feature embeddings
        self.fusion = nn.Sequential(
            nn.Linear(tcn_hidden * 2, latent_dim),
            nn.ReLU()
        )

        # Sequence decoder: project latent back to sequence shape
        # Using ConvTranspose1d to upsample back to original resolution
        self._seq_decoder_projection_dim = tcn_hidden * (seq_length // 4)
        self.seq_decoder = nn.Sequential(
            nn.Linear(latent_dim, self._seq_decoder_projection_dim),
            nn.ReLU(),
        )
        self.seq_deconv = nn.Sequential(
            nn.Unflatten(1, (tcn_hidden, seq_length // 4)),
            nn.ConvTranspose1d(tcn_hidden, tcn_hidden // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(tcn_hidden // 2, seq_channels, kernel_size=4, stride=2, padding=1),
        )

        # Feature decoder: simple MLP
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_dim, tcn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_hidden, feature_dim)
        )

    def encode(
        self,
        seq: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode both inputs to fused latent space.

        Args:
            seq: Sequence tensor of shape (batch, seq_length, seq_channels) or
                 (batch, seq_channels, seq_length)
            features: Feature tensor of shape (batch, feature_dim)

        Returns:
            Latent tensor of shape (batch, latent_dim)
        """
        # TCN encoder expects (batch, channels, seq_len) but handles both
        seq_emb = self.seq_encoder(seq)  # (batch, tcn_hidden)
        feat_emb = self.feature_encoder(features)  # (batch, tcn_hidden)

        # Concatenate and fuse
        combined = torch.cat([seq_emb, feat_emb], dim=1)  # (batch, tcn_hidden * 2)
        return self.fusion(combined)  # (batch, latent_dim)

    def decode(
        self,
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to reconstructed sequence and features.

        Args:
            latent: Latent tensor of shape (batch, latent_dim)

        Returns:
            seq_recon: Reconstructed sequence of shape (batch, seq_channels, seq_length)
            feat_recon: Reconstructed features of shape (batch, feature_dim)
        """
        # Decode sequence
        seq_proj = self.seq_decoder(latent)  # (batch, tcn_hidden * seq_length // 4)
        seq_recon = self.seq_deconv(seq_proj)  # (batch, seq_channels, seq_length)

        # Decode features
        feat_recon = self.feature_decoder(latent)  # (batch, feature_dim)

        return seq_recon, feat_recon

    def forward(
        self,
        seq: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.

        Args:
            seq: Sequence tensor
            features: Feature tensor

        Returns:
            seq_recon: Reconstructed sequence
            feat_recon: Reconstructed features
        """
        latent = self.encode(seq, features)
        return self.decode(latent)

    def reconstruction_error(
        self,
        seq: torch.Tensor,
        features: torch.Tensor,
        alpha: Optional[float] = None
    ) -> torch.Tensor:
        """
        Combined reconstruction error.

        Args:
            seq: Sequence tensor of shape (batch, seq_length, seq_channels)
            features: Feature tensor of shape (batch, feature_dim)
            alpha: Weight for sequence error vs feature error (default: self.alpha)

        Returns:
            Error tensor of shape (batch,)
        """
        if alpha is None:
            alpha = self.alpha

        seq_recon, feat_recon = self.forward(seq, features)

        # Ensure seq is in (batch, channels, seq_len) format for comparison
        if seq.size(-1) == self.seq_channels:
            seq = seq.permute(0, 2, 1)

        # Sequence reconstruction error (MSE per sample)
        seq_error = torch.mean((seq - seq_recon) ** 2, dim=(1, 2))

        # Feature reconstruction error (MSE per sample)
        feat_error = torch.mean((features - feat_recon) ** 2, dim=1)

        # Combined weighted error
        return alpha * seq_error + (1 - alpha) * feat_error

    def anomaly_score(
        self,
        seq: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomaly score (higher = more like a reversal).

        Args:
            seq: Sequence tensor
            features: Feature tensor

        Returns:
            Score tensor of shape (batch,)
        """
        error = self.reconstruction_error(seq, features)
        # Transform: low error -> high score
        return 1.0 / (1.0 + error)

    def get_latent(
        self,
        seq: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get latent representation for analysis.

        Args:
            seq: Sequence tensor
            features: Feature tensor

        Returns:
            Latent tensor of shape (batch, latent_dim)
        """
        return self.encode(seq, features)


class VariationalFeatureAutoencoder(nn.Module):
    """
    Variational autoencoder (VAE) variant for features.

    VAE provides:
    - Probabilistic latent space (better for anomaly detection)
    - KL divergence regularization
    - Reconstruction probability for scoring

    Not used by default, but available for experimentation.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: input -> hidden -> (mu, logvar)
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to distribution parameters."""
        h = self.encoder_hidden(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            x_recon: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VAE loss: reconstruction + KL divergence.

        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            beta: Weight for KL term (beta-VAE)

        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """
        Compute anomaly score using reconstruction probability.

        Uses Monte Carlo sampling for more robust score.
        """
        mu, logvar = self.encode(x)

        scores = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            error = torch.mean((x - x_recon) ** 2, dim=1)
            scores.append(error)

        # Average error across samples
        avg_error = torch.mean(torch.stack(scores), dim=0)

        return 1.0 / (1.0 + avg_error)


def create_autoencoder(
    model_type: str,
    feature_dim: int,
    seq_channels: int = 16,
    seq_length: int = 60,
    latent_dim: int = 16,
    **kwargs
) -> nn.Module:
    """
    Factory function to create autoencoder models.

    Args:
        model_type: 'feature_only', 'hybrid', or 'vae'
        feature_dim: Number of hand-crafted features
        seq_channels: Channels in sequence (for hybrid)
        seq_length: Sequence length (for hybrid)
        latent_dim: Latent space dimension
        **kwargs: Additional model parameters

    Returns:
        Instantiated autoencoder model
    """
    if model_type == 'feature_only':
        return FeatureOnlyAutoencoder(
            input_dim=feature_dim,
            latent_dim=latent_dim,
            **kwargs
        )
    elif model_type == 'hybrid':
        return HybridReversalAutoencoder(
            seq_channels=seq_channels,
            seq_length=seq_length,
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            **kwargs
        )
    elif model_type == 'vae':
        return VariationalFeatureAutoencoder(
            input_dim=feature_dim,
            latent_dim=latent_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
