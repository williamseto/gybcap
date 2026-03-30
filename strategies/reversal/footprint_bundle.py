"""Utilities for saving/loading deployable footprint model bundles."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple


def build_footprint_model(
    model_arch: str,
    scalar_dim: int,
    n_price_bins: int = 20,
    current_time_steps: int = 60,
    context_time_steps: int = 300,
):
    """Construct a footprint model by architecture name."""
    from strategies.reversal.orderflow_model import (
        FootprintFusionModel,
        MultiResEventFootprintFusionModel,
        TemporalFootprintFusionModel,
        TransformerCrossAttentionFootprintModel,
    )

    if model_arch == "temporal_tcn":
        return TemporalFootprintFusionModel(
            scalar_dim=scalar_dim,
            n_price_bins=n_price_bins,
            current_time_steps=current_time_steps,
            context_time_steps=context_time_steps,
        )
    if model_arch == "multires_event":
        return MultiResEventFootprintFusionModel(
            scalar_dim=scalar_dim,
            n_price_bins=n_price_bins,
            current_time_steps=current_time_steps,
            context_time_steps=context_time_steps,
        )
    if model_arch == "transformer_cross":
        return TransformerCrossAttentionFootprintModel(
            scalar_dim=scalar_dim,
            n_price_bins=n_price_bins,
            current_time_steps=current_time_steps,
            context_time_steps=context_time_steps,
        )
    return FootprintFusionModel(
        scalar_dim=scalar_dim,
        n_price_bins=n_price_bins,
        current_time_steps=current_time_steps,
        context_time_steps=context_time_steps,
    )


def save_footprint_bundle(
    model,
    model_dir: str,
    metadata: Dict[str, Any],
    model_filename: str = "model.pt",
    metadata_filename: str = "metadata.json",
) -> Tuple[str, str]:
    """Save model state and metadata into a deployable directory."""
    import torch

    os.makedirs(model_dir, exist_ok=True)

    payload = {
        "model_arch": metadata.get("model_arch", "cnn_fusion"),
        "scalar_dim": int(metadata.get("n_features", len(metadata.get("feature_cols", [])))),
        "n_price_bins": int(metadata.get("n_price_bins", 20)),
        "current_time_steps": int(metadata.get("current_time_steps", 60)),
        "context_time_steps": int(metadata.get("context_time_steps", 300)),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    model_path = os.path.join(model_dir, model_filename)
    torch.save(payload, model_path)

    metadata_path = os.path.join(model_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path, metadata_path


def load_footprint_bundle(
    model_dir: str,
    device: str = "cpu",
    model_filename: str = "model.pt",
    metadata_filename: str = "metadata.json",
):
    """Load a deployable footprint model bundle."""
    import torch

    model_path = os.path.join(model_dir, model_filename)
    metadata_path = os.path.join(model_dir, metadata_filename)

    with open(metadata_path) as f:
        metadata = json.load(f)

    checkpoint = torch.load(model_path, map_location=device)
    model_arch = checkpoint.get("model_arch", metadata.get("model_arch", "cnn_fusion"))
    scalar_dim = int(
        checkpoint.get(
            "scalar_dim",
            metadata.get("n_features", len(metadata.get("feature_cols", []))),
        )
    )
    n_price_bins = int(checkpoint.get("n_price_bins", metadata.get("n_price_bins", 20)))
    current_steps = int(
        checkpoint.get("current_time_steps", metadata.get("current_time_steps", 60))
    )
    context_steps = int(
        checkpoint.get("context_time_steps", metadata.get("context_time_steps", 300))
    )

    model = build_footprint_model(
        model_arch=model_arch,
        scalar_dim=scalar_dim,
        n_price_bins=n_price_bins,
        current_time_steps=current_steps,
        context_time_steps=context_steps,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, metadata
