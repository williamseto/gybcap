"""
V3 Causal Zone Prediction Model: five-input PyTorch fusion network.

Predicts P(pre-reversal zone) at every bar near a price level,
using only backward-looking (causal) data.

Architecture:
    Branch 1a: MicroVPCNN  — Conv2D on (1,20,30) → 24d  (30-min volume profile)
    Branch 1b: MesoVPCNN   — Conv2D on (1,20,10) → 24d  (5-hour volume profile)
    Branch 1c: MacroVPCNN  — Conv2D on (1,20,15) → 24d  (session volume profile)
    Branch 2:  TCNEncoder   — TCN on (60,6) → 64d        (1-hour 1-min sequence)
    Branch 3:  ScalarMLP    — BN → MLP on scalars → 32d  (~85 features)

    Fusion: Concat(24+24+24+64+32=168) → MLP → Sigmoid → P(zone)

Total parameters: ~60-70K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from strategies.reversal.tcn_model import TCNEncoder


class VPConvBlock(nn.Module):
    """Small Conv2D block for volume profile heatmaps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class VPBranchCNN(nn.Module):
    """
    CNN branch for a single volume profile heatmap.

    Conv → BN → ReLU → Conv → BN → ReLU → AdaptiveAvgPool → embed_dim
    """

    def __init__(self, input_h: int, input_w: int, embed_dim: int = 24):
        super().__init__()
        self.conv1 = VPConvBlock(1, 16, kernel_size=3)
        self.conv2 = VPConvBlock(16, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(32 * 2 * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W) volume profile heatmap

        Returns:
            (batch, embed_dim) embedding
        """
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)


class CausalZoneModel(nn.Module):
    """
    V3 causal zone prediction model.

    Predicts P(pre-reversal zone) using only backward-looking data:
    - 3 multi-scale volume profile heatmaps (micro/meso/macro)
    - 1-min sequence (60 bars = 1 hour)
    - ~85 scalar features

    Trained with soft labels: zone_probability ∈ [0,1] decaying with
    distance from reversal bar.

    Inputs:
        micro_vp: (batch, 1, 20, 30) — 30-min volume profile
        meso_vp:  (batch, 1, 20, 10) — 5-hour volume profile
        macro_vp: (batch, 1, 20, 15) — session volume profile
        sequence: (batch, 60, 6)     — 1-min price/volume sequence
        scalars:  (batch, D_scalar)  — rejection features + HTF context

    Output:
        zone_prob: (batch,) — P(bar is in pre-reversal zone)
    """

    def __init__(
        self,
        scalar_dim: int,
        vp_embed_dim: int = 24,
        tcn_hidden: int = 64,
        tcn_levels: int = 5,
        context_embed_dim: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.2,
        seq_channels: int = 6,
    ):
        super().__init__()

        # Branch 1: Volume Profile Pyramid (3 sub-branches)
        self.micro_cnn = VPBranchCNN(20, 30, vp_embed_dim)
        self.meso_cnn = VPBranchCNN(20, 10, vp_embed_dim)
        self.macro_cnn = VPBranchCNN(20, 15, vp_embed_dim)

        # Branch 2: Sequence TCN with attention pooling
        self.tcn_encoder = TCNEncoder(
            input_channels=seq_channels,
            hidden_channels=tcn_hidden,
            num_levels=tcn_levels,
            kernel_size=3,
            dropout=dropout,
            pool_type='attention',
        )

        # Branch 3: Scalar BatchNorm + MLP
        self.scalar_bn = nn.BatchNorm1d(scalar_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, context_embed_dim),
            nn.ReLU(),
        )

        # Fusion
        total_embed = 3 * vp_embed_dim + tcn_hidden + context_embed_dim  # 168
        self.fusion = nn.Sequential(
            nn.Linear(total_embed, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        micro_vp: torch.Tensor,
        meso_vp: torch.Tensor,
        macro_vp: torch.Tensor,
        sequence: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            zone_prob: (batch,) probability bar is in pre-reversal zone
        """
        # VP pyramid embeddings
        z_micro = self.micro_cnn(micro_vp)    # (batch, 24)
        z_meso = self.meso_cnn(meso_vp)       # (batch, 24)
        z_macro = self.macro_cnn(macro_vp)     # (batch, 24)

        # Sequence embedding
        z_seq = self.tcn_encoder(sequence)     # (batch, 64)

        # Scalar embedding
        scalars_normed = self.scalar_bn(scalars)
        z_ctx = self.scalar_mlp(scalars_normed)  # (batch, 32)

        # Concatenate and fuse
        z = torch.cat([z_micro, z_meso, z_macro, z_seq, z_ctx], dim=-1)  # (batch, 168)
        logit = self.fusion(z).squeeze(-1)     # (batch,)

        return torch.sigmoid(logit)

    def get_embeddings(
        self,
        micro_vp: torch.Tensor,
        meso_vp: torch.Tensor,
        macro_vp: torch.Tensor,
        sequence: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Get concatenated embeddings before fusion (for analysis)."""
        z_micro = self.micro_cnn(micro_vp)
        z_meso = self.meso_cnn(meso_vp)
        z_macro = self.macro_cnn(macro_vp)
        z_seq = self.tcn_encoder(sequence)
        scalars_normed = self.scalar_bn(scalars)
        z_ctx = self.scalar_mlp(scalars_normed)
        return torch.cat([z_micro, z_meso, z_macro, z_seq, z_ctx], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced binary classification.

    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(1e-7, 1 - 1e-7)
        bce = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()
