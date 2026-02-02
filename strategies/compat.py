"""Backward compatibility utilities for legacy models."""

from typing import Optional, List, Any
import pandas as pd
import numpy as np
from xgboost import XGBClassifier


def load_legacy_model(
    model_path: str,
    feature_cols: Optional[List[str]] = None
) -> XGBClassifier:
    """
    Load a legacy model saved with test_bo_retest.py.

    Args:
        model_path: Path to saved XGBoost model (.json)
        feature_cols: Feature column names (for compatibility check)

    Returns:
        Loaded XGBClassifier model
    """
    model = XGBClassifier()
    model.load_model(model_path)
    return model


def legacy_predict(
    model: XGBClassifier,
    features: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Make predictions using legacy feature format.

    Args:
        model: Loaded XGBClassifier
        features: DataFrame with feature values
        feature_cols: List of feature column names in order
        threshold: Prediction threshold

    Returns:
        Binary predictions array
    """
    X = features[feature_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)


def migrate_model(
    old_model_path: str,
    new_model_path: str,
    feature_cols: List[str]
) -> None:
    """
    Migrate a legacy model to new format.

    Currently this just copies the model since the format is compatible.
    This function exists for future migrations if needed.

    Args:
        old_model_path: Path to legacy model
        new_model_path: Path for new model
        feature_cols: Feature column names
    """
    model = load_legacy_model(old_model_path, feature_cols)
    model.save_model(new_model_path)
    print(f"Model migrated from {old_model_path} to {new_model_path}")


class LegacyStrategyAdapter:
    """
    Adapter to use legacy test_bo_retest.py strategies with new framework.

    Allows gradual migration from old code to new code.
    """

    def __init__(self, legacy_strategy: Any):
        """
        Initialize adapter.

        Args:
            legacy_strategy: Instance of legacy BreakoutRetestStrategy or ReversionStrategy
        """
        self.legacy = legacy_strategy

    def find_retest_and_build_trades(
        self,
        stop_buffer_pct: float = 0.0025,
        rr: float = 1.5,
        fixed_size: float = 1.0
    ):
        """Call legacy method with same interface."""
        return self.legacy.find_retest_and_build_trades(
            stop_buffer_pct=stop_buffer_pct,
            rr=rr,
            fixed_size=fixed_size
        )


# Feature column mappings for different versions
LEGACY_FEATURE_COLS_V1 = [
    'close_z20', 'ovn_lo_z', 'ovn_hi_z', 'ib_lo_z', 'ib_hi_z',
    'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z', 'nearby_gamma_score', 'bear'
]

LEGACY_FEATURE_COLS_V0 = [
    'close_z20', 'ovn_lo_z', 'ovn_hi_z', 'ib_lo_z', 'ib_hi_z',
    'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z', 'bear'
]


def detect_feature_version(model_path: str) -> str:
    """
    Detect which feature version a model was trained with.

    Args:
        model_path: Path to model file

    Returns:
        Version string ('v0', 'v1', etc.)
    """
    model = load_legacy_model(model_path)

    n_features = model.n_features_in_

    if n_features == len(LEGACY_FEATURE_COLS_V1):
        return 'v1'
    elif n_features == len(LEGACY_FEATURE_COLS_V0):
        return 'v0'
    else:
        return 'unknown'


def get_feature_cols_for_version(version: str) -> List[str]:
    """Get feature columns for a specific version."""
    if version == 'v1':
        return LEGACY_FEATURE_COLS_V1
    elif version == 'v0':
        return LEGACY_FEATURE_COLS_V0
    else:
        raise ValueError(f"Unknown version: {version}")
