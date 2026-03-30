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
import torch.nn.functional as F


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


class TemporalResidualBlock(nn.Module):
    """Dilated 1D residual block over time dimension."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TemporalFootprintBranch(nn.Module):
    """Price-aware spatial stem + multi-scale temporal encoder."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        embed_dim: int = 32,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.spatial = nn.Sequential(
            FootprintConvBlock(in_channels, 16, kernel_size=3),
            FootprintConvBlock(16, hidden_channels, kernel_size=3),
        )

        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )

        self.attn = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.proj = nn.Linear(hidden_channels, embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, price_bins, time_steps)
        x = self.spatial(x)
        x = x.mean(dim=2)  # price-pool -> (batch, hidden, time_steps)
        x = self.temporal(x)
        w = torch.softmax(self.attn(x), dim=-1)  # (batch, 1, time_steps)
        pooled = (x * w).sum(dim=-1)  # (batch, hidden)
        return self.relu(self.proj(pooled))


class TemporalSeqEncoder(nn.Module):
    """Generic dilated temporal encoder with attention pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        embed_dim: int = 32,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=hidden_channels,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
                for d in dilations
            ]
        )
        self.attn = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.proj = nn.Linear(hidden_channels, embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time_steps)
        h = self.stem(x)
        h = self.blocks(h)
        w = torch.softmax(self.attn(h), dim=-1)
        pooled = (h * w).sum(dim=-1)
        return self.relu(self.proj(pooled))


class FootprintTimeEncoder(nn.Module):
    """Encodes 2D footprint tensor into temporal embedding."""

    def __init__(
        self,
        in_channels: int = 4,
        spatial_channels: int = 32,
        embed_dim: int = 32,
        dilations: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial = nn.Sequential(
            FootprintConvBlock(in_channels, 16, kernel_size=3),
            FootprintConvBlock(16, spatial_channels, kernel_size=3),
        )
        self.temporal = TemporalSeqEncoder(
            in_channels=spatial_channels,
            hidden_channels=spatial_channels,
            embed_dim=embed_dim,
            dilations=dilations,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, price_bins, time_steps)
        s = self.spatial(x).mean(dim=2)  # (batch, spatial_channels, time_steps)
        return self.temporal(s)


class EventClockEncoder(nn.Module):
    """Event-time representation using volume/imbalance clocks."""

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 32,
        hidden_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal = TemporalSeqEncoder(
            in_channels=in_channels * 3,  # wall-clock + vol-clock + imbalance-clock
            hidden_channels=hidden_channels,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8],
            dropout=dropout,
        )

    @staticmethod
    def _build_clock(weight: torch.Tensor) -> torch.Tensor:
        # weight: (batch, time_steps)
        w = torch.relu(weight) + 1e-6
        c = torch.cumsum(w, dim=-1)
        return c / c[:, -1:].clamp_min(1e-6)

    @staticmethod
    def _resample_by_clock(seq: torch.Tensor, clock: torch.Tensor) -> torch.Tensor:
        # seq: (batch, channels, time_steps), clock: (batch, time_steps) in [0,1]
        b, c, t = seq.shape
        if t <= 1:
            return seq

        targets = torch.linspace(
            0.0, 1.0, t, device=seq.device, dtype=seq.dtype
        ).unsqueeze(0).expand(b, -1)
        idx_hi = torch.searchsorted(clock, targets, right=True)
        idx_hi = idx_hi.clamp(1, t - 1)
        idx_lo = idx_hi - 1

        clock_lo = torch.gather(clock, 1, idx_lo)
        clock_hi = torch.gather(clock, 1, idx_hi)
        alpha = (targets - clock_lo) / (clock_hi - clock_lo).clamp_min(1e-6)
        alpha = alpha.unsqueeze(1)

        gather_lo = seq.gather(2, idx_lo.unsqueeze(1).expand(-1, c, -1))
        gather_hi = seq.gather(2, idx_hi.unsqueeze(1).expand(-1, c, -1))
        return gather_lo + (gather_hi - gather_lo) * alpha

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        # x2d: (batch, channels, price_bins, time_steps)
        seq = x2d.mean(dim=2)  # (batch, channels, time_steps)

        vol_clock = self._build_clock(seq[:, 0, :].abs())
        delta_idx = 3 if seq.shape[1] > 3 else seq.shape[1] - 1
        imb_clock = self._build_clock(seq[:, delta_idx, :].abs())

        vol_resampled = self._resample_by_clock(seq, vol_clock)
        imb_resampled = self._resample_by_clock(seq, imb_clock)

        feat = torch.cat([seq, vol_resampled, imb_resampled], dim=1)
        return self.temporal(feat)


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


class TemporalFootprintFusionModel(nn.Module):
    """Temporal multi-scale footprint model for variable-speed reversals."""

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
        # Longer context gets deeper dilation stack.
        self.current_branch = TemporalFootprintBranch(
            in_channels=4,
            hidden_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8, 16],
            dropout=dropout * 0.5,
        )
        self.context_branch = TemporalFootprintBranch(
            in_channels=4,
            hidden_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8, 16, 32],
            dropout=dropout * 0.5,
        )
        self.scalar_mlp = ScalarMLP(
            scalar_dim=scalar_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        fusion_dim = embed_dim * 3
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
        e_current = self.current_branch(current_fp)
        e_context = self.context_branch(context_fp)
        e_scalar = self.scalar_mlp(scalars)

        fused = torch.cat([e_current, e_context, e_scalar], dim=1)
        logit = self.fusion(fused).squeeze(-1)
        return torch.sigmoid(logit)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiResEventFootprintFusionModel(nn.Module):
    """Multi-resolution context + cross-scale attention + event-time encoding."""

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
        self.current_encoder = FootprintTimeEncoder(
            in_channels=4,
            spatial_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8],
            dropout=dropout * 0.5,
        )
        self.ctx30_encoder = FootprintTimeEncoder(
            in_channels=4,
            spatial_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4],
            dropout=dropout * 0.5,
        )
        self.ctx90_encoder = FootprintTimeEncoder(
            in_channels=4,
            spatial_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8],
            dropout=dropout * 0.5,
        )
        self.ctx300_encoder = FootprintTimeEncoder(
            in_channels=4,
            spatial_channels=32,
            embed_dim=embed_dim,
            dilations=[1, 2, 4, 8, 16],
            dropout=dropout * 0.5,
        )
        self.scale_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.event_encoder = EventClockEncoder(
            in_channels=4,
            embed_dim=embed_dim,
            hidden_channels=32,
            dropout=dropout * 0.5,
        )
        self.scalar_mlp = ScalarMLP(
            scalar_dim=scalar_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        fusion_dim = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def _context_scales(self, context_fp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # context_fp: (batch, 4, price_bins, 300)
        price_bins = context_fp.shape[2]
        ctx30 = F.adaptive_avg_pool2d(context_fp, (price_bins, 30))
        ctx90 = F.adaptive_avg_pool2d(context_fp, (price_bins, 90))
        return ctx30, ctx90

    def forward(
        self,
        current_fp: torch.Tensor,
        context_fp: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        e_current = self.current_encoder(current_fp)
        ctx30, ctx90 = self._context_scales(context_fp)
        e30 = self.ctx30_encoder(ctx30)
        e90 = self.ctx90_encoder(ctx90)
        e300 = self.ctx300_encoder(context_fp)

        scale_tokens = torch.stack([e30, e90, e300], dim=1)  # (batch, 3, embed_dim)
        # Use current-branch embedding as query to select scale relevance.
        scale_ctx, _ = self.scale_attn(
            query=e_current.unsqueeze(1),
            key=scale_tokens,
            value=scale_tokens,
            need_weights=False,
        )
        e_scale = scale_ctx.squeeze(1)

        e_event = self.event_encoder(context_fp)
        e_scalar = self.scalar_mlp(scalars)

        fused = torch.cat([e_current, e_scale, e_event, e_scalar], dim=1)
        logit = self.fusion(fused).squeeze(-1)
        return torch.sigmoid(logit)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FootprintTokenEncoder(nn.Module):
    """Convert 2D footprints into time-token embeddings."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 32,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial = nn.Sequential(
            FootprintConvBlock(in_channels, 16, kernel_size=3),
            FootprintConvBlock(16, hidden_channels, kernel_size=3),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, price_bins, time_steps)
        h = self.spatial(x).mean(dim=2)  # (batch, hidden_channels, time_steps)
        tok = h.transpose(1, 2)  # (batch, time_steps, hidden_channels)
        return self.proj(tok)


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional encoding."""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        return x + self.pos[:, :t, :]


class TransformerCrossAttentionFootprintModel(nn.Module):
    """Transformer variant: current-query cross-attends to context tokens."""

    def __init__(
        self,
        scalar_dim: int,
        n_price_bins: int = 20,
        current_time_steps: int = 60,
        context_time_steps: int = 300,
        embed_dim: int = 64,
        num_heads: int = 4,
        fusion_hidden: int = 96,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.context_token_steps = 75
        self.current_tok = FootprintTokenEncoder(
            in_channels=4,
            hidden_channels=32,
            embed_dim=embed_dim,
            dropout=dropout * 0.5,
        )
        self.context_tok = FootprintTokenEncoder(
            in_channels=4,
            hidden_channels=32,
            embed_dim=embed_dim,
            dropout=dropout * 0.5,
        )
        self.pos_cur = LearnedPositionalEncoding(
            max_len=max(current_time_steps, 64),
            embed_dim=embed_dim,
        )
        self.pos_ctx = LearnedPositionalEncoding(
            max_len=max(self.context_token_steps, 128),
            embed_dim=embed_dim,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout * 0.5,
            activation="gelu",
            batch_first=True,
        )
        self.ctx_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cur_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(embed_dim)
        self.cross_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.cross_ff_ln = nn.LayerNorm(embed_dim)

        self.cur_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )
        self.cross_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )

        self.scalar_mlp = ScalarMLP(
            scalar_dim=scalar_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        fusion_dim = embed_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def _compress_context(self, context_fp: torch.Tensor) -> torch.Tensor:
        # context_fp: (batch, channels, price_bins, time_steps)
        b, c, p, t = context_fp.shape
        x = context_fp.reshape(b, c * p, t)
        x = F.adaptive_avg_pool1d(x, self.context_token_steps)
        x = x.reshape(b, c, p, self.context_token_steps)
        return x

    @staticmethod
    def _attn_pool(tokens: torch.Tensor, scorer: nn.Module) -> torch.Tensor:
        # tokens: (batch, time, dim)
        w = torch.softmax(scorer(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        return (tokens * w).sum(dim=1)

    def forward(
        self,
        current_fp: torch.Tensor,
        context_fp: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        cur_tok = self.current_tok(current_fp)          # (B, 60, D)
        ctx_small = self._compress_context(context_fp)  # (B, C, P, 75)
        ctx_tok = self.context_tok(ctx_small)           # (B, 75, D)

        cur_tok = self.pos_cur(cur_tok)
        ctx_tok = self.pos_ctx(ctx_tok)

        cur_tok = self.cur_encoder(cur_tok)
        ctx_tok = self.ctx_encoder(ctx_tok)

        cross, _ = self.cross_attn(
            query=cur_tok,
            key=ctx_tok,
            value=ctx_tok,
            need_weights=False,
        )
        cross = self.cross_ln(cur_tok + cross)
        cross_ff = self.cross_ff(cross)
        cross = self.cross_ff_ln(cross + cross_ff)

        e_cur = self._attn_pool(cur_tok, self.cur_pool)
        e_cross = self._attn_pool(cross, self.cross_pool)
        e_scalar = self.scalar_mlp(scalars)

        fused = torch.cat([e_cur, e_cross, e_scalar], dim=1)
        logit = self.fusion(fused).squeeze(-1)
        return torch.sigmoid(logit)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
