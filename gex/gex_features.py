"""Extract ML-ready features from a DealerPositionModel.

Produces ~13 features in 4 categories:

1. **Gamma regime** — mean-revert vs trend signal
2. **Key level distances** — magnetic / support / resistance effects
3. **Hedging flow** — quantifies dampening / amplifying force
4. **Structure** — concentration, time sensitivity, slope
"""

import numpy as np
from typing import Dict

from gex.gex_surface import DealerPositionModel


# Canonical feature list — keep in sync with extract_gex_features()
GEX_FEATURE_NAMES = [
    # Regime
    "gex_regime_sign",
    "gex_net_normalized",
    "gamma_flip_distance",
    # Key level distances
    "dist_to_max_gamma",
    "dist_to_put_wall",
    "dist_to_call_wall",
    # Hedging flow
    "hedge_flow_1pct_up",
    "hedge_flow_1pct_down",
    "hedge_flow_asymmetry",
    # Structure
    "gex_concentration",
    "near_expiry_gamma_pct",
    "gex_slope_at_spot",
    "nearby_gamma_score",
]


def extract_gex_features(
    model: DealerPositionModel,
    spot: float,
    atr_14: float = 1.0,
    gex_rolling_std: float = 1.0,
) -> Dict[str, float]:
    """Extract GEX features from a built :class:`DealerPositionModel`.

    Args:
        model: A fully-constructed DealerPositionModel.
        spot: Current spot price for feature extraction.
        atr_14: 14-period ATR used to normalise distance features.
        gex_rolling_std: Rolling 60-day std of net GEX at spot (for z-scoring).

    Returns:
        Dict mapping each name in :data:`GEX_FEATURE_NAMES` to a float value.
    """
    features: Dict[str, float] = {}
    regime = model.regime
    key = model.key_levels

    atr = max(atr_14, 0.01)
    gex_std = max(gex_rolling_std, 1e-10)

    # ── 1. Gamma regime ──────────────────────────────────────────

    features["gex_regime_sign"] = float(regime.sign)
    features["gex_net_normalized"] = regime.net_gex / gex_std

    if key.gamma_flip is not None:
        features["gamma_flip_distance"] = (spot - key.gamma_flip) / atr
    else:
        features["gamma_flip_distance"] = 0.0

    # ── 2. Key level distances ───────────────────────────────────

    features["dist_to_max_gamma"] = (
        (spot - key.max_gamma_strike) / atr if key.max_gamma_strike is not None else 0.0
    )
    features["dist_to_put_wall"] = (
        (spot - key.put_wall) / atr if key.put_wall is not None else 0.0
    )
    features["dist_to_call_wall"] = (
        (spot - key.call_wall) / atr if key.call_wall is not None else 0.0
    )

    # ── 3. Hedging flow ──────────────────────────────────────────

    move = spot * 0.01
    features["hedge_flow_1pct_up"] = model.hedge_flow(spot, spot + move)
    features["hedge_flow_1pct_down"] = model.hedge_flow(spot, spot - move)
    features["hedge_flow_asymmetry"] = (
        abs(features["hedge_flow_1pct_up"]) - abs(features["hedge_flow_1pct_down"])
    )

    # ── 4. Structure ─────────────────────────────────────────────

    total_abs_gex = float(np.sum(np.abs(model.gex_grid)))
    local_gex = abs(model.gex_at(spot))
    features["gex_concentration"] = local_gex / max(total_abs_gex, 1e-10)

    features["near_expiry_gamma_pct"] = model.near_expiry_gamma_fraction(max_dte_days=3)
    features["gex_slope_at_spot"] = model.gex_slope_at(spot)

    # Backward-compatible nearby_gamma_score (local / neighbourhood ratio)
    gex_here = abs(model.gex_at(spot))
    gex_hood = sum(abs(model.gex_at(spot + off)) for off in range(-15, 16, 5))
    features["nearby_gamma_score"] = gex_here / max(gex_hood, 1e-10)

    return features
