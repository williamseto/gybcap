"""
TCN (Temporal Convolutional Network) architecture for reversal prediction.

Ported from sandbox/pred_util.py with modifications for multi-channel input
and different output heads (classification + regression).
"""

from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNBlock(nn.Module):
    """
    Single TCN block with dilated causal convolution.

    Features:
    - Dilated convolution for expanding receptive field
    - Batch normalization for stable training
    - Residual connection
    - Causal padding (no future leakage)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        # Causal padding: (kernel_size - 1) * dilation
        self.pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, seq_len)

        Returns:
            Output tensor of same shape
        """
        # Causal convolution: remove future padding
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # Trim to original length

        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionPooling1d(nn.Module):
    """
    Learned attention pooling over the temporal dimension.

    Instead of AdaptiveMaxPool1d (which discards temporal structure),
    this computes per-timestep importance scores via a small MLP,
    applies softmax, and returns a weighted sum.

    Exposes attention weights for interpretability.
    """

    def __init__(self, channels: int, attn_hidden: int = 32):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(channels, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self._last_weights = None  # for interpretability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, channels) — attention-weighted sum over seq_len
        """
        # x: (B, C, T) → (B, T, C) for the MLP
        x_t = x.permute(0, 2, 1)          # (B, T, C)
        scores = self.attn_mlp(x_t)       # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        self._last_weights = weights.detach()
        out = (x_t * weights).sum(dim=1)  # (B, C)
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return (batch, seq_len, 1) attention weights from last forward pass."""
        return self._last_weights


class TCNEncoder(nn.Module):
    """
    TCN encoder that produces a fixed-size embedding from variable-length sequences.

    Architecture:
    - Stack of TCN blocks with increasing dilation (1, 2, 4, 8, ...)
    - Global pooling to produce fixed-size output
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        pool_type: str = 'max'
    ):
        """
        Initialize TCN encoder.

        Args:
            input_channels: Number of input channels (features per timestep)
            hidden_channels: Number of channels in hidden layers
            num_levels: Number of TCN blocks (receptive field = 2^levels - 1)
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            pool_type: 'max', 'avg', or 'attention' for global pooling
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.pool_type = pool_type

        # Build TCN blocks
        layers = []
        channels = [input_channels] + [hidden_channels] * num_levels

        for i in range(num_levels):
            layers.append(
                TCNBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # Global pooling
        if pool_type == 'attention':
            self.pool = AttentionPooling1d(hidden_channels)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to fixed-size embedding.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)
               or (batch, input_channels, seq_len)

        Returns:
            Embedding tensor of shape (batch, hidden_channels)
        """
        # Ensure channels-first format
        if x.size(-1) == self.input_channels:
            x = x.permute(0, 2, 1)  # (batch, channels, seq_len)

        # Apply TCN
        h = self.tcn(x)  # (batch, hidden_channels, seq_len)

        # Global pooling
        if self.pool_type == 'attention':
            z = self.pool(h)  # AttentionPooling1d returns (batch, channels)
        else:
            z = self.pool(h).squeeze(-1)  # (batch, hidden_channels)

        return z

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights if using attention pooling."""
        if self.pool_type == 'attention' and hasattr(self.pool, 'get_attention_weights'):
            return self.pool.get_attention_weights()
        return None

    @property
    def output_dim(self) -> int:
        """Dimension of output embedding."""
        return self.hidden_channels


class TCNClassifier(nn.Module):
    """
    TCN model for reversal classification.

    Outputs:
    - reversal_prob: Probability of any reversal
    - direction_logits: [bull_prob, bear_prob] (2-class)
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 32
    ):
        super().__init__()

        self.encoder = TCNEncoder(
            input_channels,
            hidden_channels,
            num_levels,
            kernel_size,
            dropout
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 3)  # [no_reversal, bull, bear]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)

        Returns:
            reversal_prob: (batch,) probability of any reversal
            direction_probs: (batch, 3) [none, bull, bear] probabilities
        """
        z = self.encoder(x)
        logits = self.head(z)

        # Softmax over 3 classes
        direction_probs = F.softmax(logits, dim=-1)

        # Probability of any reversal = 1 - P(none)
        reversal_prob = 1.0 - direction_probs[:, 0]

        return reversal_prob, direction_probs


class TCNMultiTask(nn.Module):
    """
    TCN model for multi-task prediction:
    - Classification: Is this a reversal? Which direction?
    - Regression: How large will the move be?

    This combines best of both approaches.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        head_hidden: int = 32
    ):
        super().__init__()

        self.encoder = TCNEncoder(
            input_channels,
            hidden_channels,
            num_levels,
            kernel_size,
            dropout
        )

        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(hidden_channels, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.cls_head = nn.Linear(head_hidden, 3)  # [none, bull, bear]

        # Regression head (magnitude prediction)
        self.reg_head = nn.Linear(head_hidden, 1)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)

        Returns:
            reversal_prob: (batch,) probability of any reversal
            direction_probs: (batch, 3) [none, bull, bear] probabilities
            magnitude: (batch,) predicted move magnitude
        """
        z = self.encoder(x)
        h = self.shared(z)

        # Classification
        cls_logits = self.cls_head(h)
        direction_probs = F.softmax(cls_logits, dim=-1)
        reversal_prob = 1.0 - direction_probs[:, 0]

        # Regression
        magnitude = self.reg_head(h).squeeze(-1)

        return reversal_prob, direction_probs, magnitude


class HybridModel(nn.Module):
    """
    Hybrid model combining TCN sequence encoding with hand-crafted features.

    Architecture:
    [Price Sequence] → [TCN Encoder] → [Latent]
                                            ↓
    [HTF Features] ────────────────────→ [Concat] → [MLP] → [Prediction]

    This allows the model to learn patterns from raw data while also
    leveraging domain knowledge encoded in higher-timeframe features.
    """

    def __init__(
        self,
        seq_input_channels: int,
        htf_feature_dim: int,
        tcn_hidden: int = 64,
        tcn_levels: int = 4,
        fusion_hidden: int = 64,
        dropout: float = 0.2
    ):
        """
        Initialize hybrid model.

        Args:
            seq_input_channels: Channels in sequence input (OHLCV + delta + VAP)
            htf_feature_dim: Number of higher-timeframe features
            tcn_hidden: Hidden dimension for TCN
            tcn_levels: Number of TCN levels
            fusion_hidden: Hidden dimension for fusion MLP
            dropout: Dropout rate
        """
        super().__init__()

        self.seq_encoder = TCNEncoder(
            seq_input_channels,
            tcn_hidden,
            tcn_levels,
            dropout=dropout
        )

        # HTF feature encoder (simple MLP)
        self.htf_encoder = nn.Sequential(
            nn.Linear(htf_feature_dim, fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        combined_dim = tcn_hidden + fusion_hidden // 2
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output heads
        self.cls_head = nn.Linear(fusion_hidden, 3)  # [none, bull, bear]
        self.reg_head = nn.Linear(fusion_hidden, 1)  # magnitude

    def forward(
        self,
        seq: torch.Tensor,
        htf_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            seq: Sequence tensor (batch, seq_len, seq_channels)
            htf_features: HTF feature tensor (batch, htf_dim)

        Returns:
            reversal_prob, direction_probs, magnitude
        """
        # Encode sequence
        z_seq = self.seq_encoder(seq)

        # Encode HTF features
        z_htf = self.htf_encoder(htf_features)

        # Fuse
        z_combined = torch.cat([z_seq, z_htf], dim=-1)
        h = self.fusion(z_combined)

        # Outputs
        cls_logits = self.cls_head(h)
        direction_probs = F.softmax(cls_logits, dim=-1)
        reversal_prob = 1.0 - direction_probs[:, 0]
        magnitude = self.reg_head(h).squeeze(-1)

        return reversal_prob, direction_probs, magnitude


def create_tcn_model(
    model_type: str,
    input_channels: int,
    htf_feature_dim: int = 0,
    **kwargs
) -> nn.Module:
    """
    Factory function to create TCN models.

    Args:
        model_type: 'classifier', 'multitask', or 'hybrid'
        input_channels: Number of input channels
        htf_feature_dim: Number of HTF features (for hybrid model)
        **kwargs: Additional model parameters

    Returns:
        Instantiated model
    """
    if model_type == 'classifier':
        return TCNClassifier(input_channels, **kwargs)
    elif model_type == 'multitask':
        return TCNMultiTask(input_channels, **kwargs)
    elif model_type == 'hybrid':
        if htf_feature_dim <= 0:
            raise ValueError("hybrid model requires htf_feature_dim > 0")
        return HybridModel(input_channels, htf_feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
