"""Shared swing pipeline composition for production and research flows.

This module centralizes feature + label assembly so dashboard runtime and
research scripts use the same canonical construction path.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd

from strategies.swing.config import SwingConfig
from strategies.swing.features.cross_instrument import compute_cross_features
from strategies.swing.features.daily_technical import (
    FEATURE_NAMES as TECH_FEATURES,
    compute_daily_technical,
)
from strategies.swing.features.external_daily import compute_external_features
from strategies.swing.features.macro_context import (
    FEATURE_NAMES as MACRO_FEATURES,
    compute_macro_context,
)
from strategies.swing.features.range_features import (
    FEATURE_NAMES as RANGE_FEATURES,
    compute_range_features,
)
from strategies.swing.features.volume_profile_daily import (
    FEATURE_NAMES as VP_FEATURES,
    compute_vp_daily_features,
)
from strategies.swing.labeling.structural_regime import compute_labels

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    features: pd.DataFrame
    labels: pd.DataFrame
    df: pd.DataFrame
    feature_cols: list[str]
    feature_groups: dict[str, list[str]]


def build_training_frame(
    es_daily: pd.DataFrame,
    other_dailys: list[tuple[str, pd.DataFrame]] | None = None,
    config: SwingConfig | None = None,
    include_vp: bool = True,
    include_external: bool = True,
    include_range: bool = True,
    use_other_dailys_for_macro: bool = True,
) -> PipelineArtifacts:
    """Build the canonical feature + label frame for swing regime models."""
    swing_cfg = config or SwingConfig()
    features, feature_cols, feature_groups = compute_feature_frame(
        es_daily=es_daily,
        other_dailys=other_dailys,
        include_vp=include_vp,
        include_external=include_external,
        include_range=include_range,
        corr_windows=swing_cfg.corr_windows,
        use_other_dailys_for_macro=use_other_dailys_for_macro,
    )
    labels = compute_label_frame(es_daily, swing_cfg)
    df = features.join(labels)
    return PipelineArtifacts(
        features=features,
        labels=labels,
        df=df,
        feature_cols=feature_cols,
        feature_groups=feature_groups,
    )


def compute_feature_frame(
    es_daily: pd.DataFrame,
    other_dailys: list[tuple[str, pd.DataFrame]] | None = None,
    include_vp: bool = True,
    include_external: bool = True,
    include_range: bool = True,
    corr_windows: list[int] | None = None,
    use_other_dailys_for_macro: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Compute the shared feature matrix and its feature column contract."""
    aligned_other = _align_other_dailys(es_daily.index, other_dailys)

    feature_cols: list[str] = []
    feature_groups: dict[str, list[str]] = {}
    parts: list[pd.DataFrame] = []

    tech = compute_daily_technical(es_daily)
    parts.append(tech)
    _extend_unique(feature_cols, feature_groups, "technical", TECH_FEATURES)

    vp = pd.DataFrame(index=es_daily.index)
    if include_vp and "vp_poc_rel" in es_daily.columns:
        vp = compute_vp_daily_features(es_daily)
        _extend_unique(feature_cols, feature_groups, "volume_profile", VP_FEATURES)
    else:
        feature_groups["volume_profile"] = []
    parts.append(vp)

    macro_inputs = aligned_other if use_other_dailys_for_macro else None
    macro = compute_macro_context(es_daily, other_dailys=macro_inputs)
    parts.append(macro)
    _extend_unique(feature_cols, feature_groups, "macro", MACRO_FEATURES)

    range_feats = pd.DataFrame(index=es_daily.index)
    if include_range:
        range_feats = compute_range_features(es_daily)
        _extend_unique(feature_cols, feature_groups, "range", RANGE_FEATURES)
    else:
        feature_groups["range"] = []
    parts.append(range_feats)

    cross = pd.DataFrame(index=es_daily.index)
    if aligned_other:
        cross = compute_cross_features(es_daily, aligned_other, corr_windows=corr_windows)
        cross_cols = [c for c in cross.columns if c not in feature_cols]
        _extend_unique(feature_cols, feature_groups, "cross", cross_cols)
    else:
        feature_groups["cross"] = []
    parts.append(cross)

    ext = pd.DataFrame(index=es_daily.index)
    if include_external:
        try:
            ext, ext_names = compute_external_features(es_daily)
            ext_cols = [c for c in ext_names if c in ext.columns]
            _extend_unique(feature_cols, feature_groups, "external", ext_cols)
        except Exception as exc:
            logger.warning("External features failed: %s", exc)
            feature_groups["external"] = []
    else:
        feature_groups["external"] = []
    parts.append(ext)

    features = pd.concat(parts, axis=1)
    features = features.reindex(es_daily.index).fillna(0)
    return features, feature_cols, feature_groups


def compute_label_frame(
    es_daily: pd.DataFrame,
    config: SwingConfig | None = None,
) -> pd.DataFrame:
    """Compute labels using the shared swing configuration contract."""
    swing_cfg = config or SwingConfig()
    return compute_labels(
        es_daily,
        swing_lookback=swing_cfg.swing_lookback,
        micro_threshold_pct=swing_cfg.micro_threshold_pct,
        detect_threshold=swing_cfg.detect_threshold,
        bull_threshold=swing_cfg.bull_threshold,
        bear_threshold=swing_cfg.bear_threshold,
    )


def _align_other_dailys(
    es_index: pd.DatetimeIndex,
    other_dailys: list[tuple[str, pd.DataFrame]] | None,
) -> list[tuple[str, pd.DataFrame]]:
    """Normalize auxiliary inputs to aligned close-only DataFrames."""
    aligned: list[tuple[str, pd.DataFrame]] = []
    if not other_dailys:
        return aligned

    for symbol, other_df in other_dailys:
        if other_df is None or "close" not in other_df.columns:
            continue
        close_only = pd.DataFrame(
            {"close": other_df["close"].reindex(es_index, method="ffill")},
            index=es_index,
        )
        aligned.append((symbol, close_only))
    return aligned


def _extend_unique(
    feature_cols: list[str],
    feature_groups: dict[str, list[str]],
    group_name: str,
    columns,
) -> None:
    unique_cols = [str(col) for col in columns if str(col) not in feature_cols]
    feature_cols.extend(unique_cols)
    feature_groups[group_name] = unique_cols
