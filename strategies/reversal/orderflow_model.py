"""
Phase 4: FootprintCNN + ScalarMLP Fusion Model.

Architecture:
  Branch 1: FootprintCNN on (4, 20, 60) current-bar footprint → 32d
  Branch 2: ApproachCNN  on (4, 20, 300) 5-min context footprint → 32d
  Branch 3: ScalarMLP    on ~150 scalar features → 32d
  Fusion: Concat(96d) → MLP(96→64→1) → Sigmoid

Key difference from failed V3 VPBranchCNN:
  - 4 channels (total_vol, bid_vol, ask_vol, delta) vs 1 channel
  - Per-second resolution vs per-minute
  - Actual bid/ask split vs estimated
"""

import torch
import torch.nn as nn


class FootprintConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU block for footprint tensors."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FootprintCNN(nn.Module):
    """CNN branch for processing 2D footprint tensors.

    Input: (batch, 4, n_price_bins, n_time_steps)
    Output: (batch, embed_dim)
    """

    def __init__(
        self,
        n_price_bins: int = 20,
        n_time_steps: int = 60,
        embed_dim: int = 32,
        in_channels: int = 4,
    ):
        super().__init__()
        self.conv1 = FootprintConvBlock(in_channels, 16, kernel_size=3)
        self.conv2 = FootprintConvBlock(16, 32, kernel_size=3)
        self.conv3 = FootprintConvBlock(32, 64, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 4 * 4, embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc(x))
        return x


class ScalarMLP(nn.Module):
    """MLP branch for processing scalar features.

    Input: (batch, scalar_dim)
    Output: (batch, embed_dim)
    """

    def __init__(self, scalar_dim: int, embed_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(scalar_dim)
        self.net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        return self.net(x)


class FootprintFusionModel(nn.Module):
    """Multi-branch fusion model for reversal prediction from footprint data.

    Combines:
      - Current bar footprint (4, 20, 60) via FootprintCNN → 32d
      - Approach context (4, 20, 300) via FootprintCNN → 32d
      - Scalar features (D_scalar,) via ScalarMLP → 32d
    Fuses via MLP → sigmoid probability.
    """

    def __init__(
        self,
        scalar_dim: int,
        n_price_bins: int = 20,
        current_time_steps: int = 60,
        context_time_steps: int = 300,
        embed_dim: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.current_cnn = FootprintCNN(
            n_price_bins=n_price_bins,
            n_time_steps=current_time_steps,
            embed_dim=embed_dim,
            in_channels=4,
        )
        self.context_cnn = FootprintCNN(
            n_price_bins=n_price_bins,
            n_time_steps=context_time_steps,
            embed_dim=embed_dim,
            in_channels=4,
        )
        self.scalar_mlp = ScalarMLP(
            scalar_dim=scalar_dim, embed_dim=embed_dim, dropout=dropout
        )

        fusion_dim = embed_dim * 3  # 32 + 32 + 32 = 96
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(
        self,
        current_fp: torch.Tensor,
        context_fp: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            current_fp: (batch, 4, 20, 60) current bar footprint
            context_fp: (batch, 4, 20, 300) approach context
            scalars:    (batch, D_scalar) scalar features

        Returns:
            (batch,) probability ∈ [0, 1]
        """
        e_current = self.current_cnn(current_fp)
        e_context = self.context_cnn(context_fp)
        e_scalar = self.scalar_mlp(scalars)

        fused = torch.cat([e_current, e_context, e_scalar], dim=1)
        logit = self.fusion(fused).squeeze(-1)
        return torch.sigmoid(logit)

    def get_embeddings(
        self,
        current_fp: torch.Tensor,
        context_fp: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        """Get pre-fusion embeddings for analysis."""
        e_current = self.current_cnn(current_fp)
        e_context = self.context_cnn(context_fp)
        e_scalar = self.scalar_mlp(scalars)
        return torch.cat([e_current, e_context, e_scalar], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
