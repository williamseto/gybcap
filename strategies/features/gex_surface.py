"""GEX surface feature provider — full dealer positioning features.

Replaces the simplistic ``nearby_gamma_score`` from :mod:`gamma` with ~13
features derived from a proper :class:`DealerPositionModel`.

Registration name: ``'gex_surface'``

Usage::

    from strategies.features.registry import FeatureRegistry
    provider = FeatureRegistry.create('gex_surface')
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry

# Ensure gex package is importable
_gex_root = str(Path(__file__).resolve().parent.parent.parent)
if _gex_root not in sys.path:
    sys.path.insert(0, _gex_root)

from gex.gex_surface import DealerPositionModel, load_chain_for_date
from gex.gex_features import extract_gex_features, GEX_FEATURE_NAMES


@FeatureRegistry.register("gex_surface")
class GEXSurfaceFeatureProvider(BaseFeatureProvider):
    """Compute dealer-positioning features for each bar.

    For each trading day the provider:
    1. Loads the previous day's options chain from the historical parquet.
    2. Builds a :class:`DealerPositionModel`.
    3. Extracts ~13 features at each bar's close price.

    The ``atr_14`` and ``gex_rolling_std`` normalisers are computed from the
    OHLCV data itself so that no external context is required.
    """

    def __init__(
        self,
        parquet_path: str = "gex/gamma_shares_combined.parquet",
        spx_es_offset: float = 0.0,
    ):
        """
        Args:
            parquet_path: Path to the combined gamma-shares parquet.
            spx_es_offset: Fixed SPX - ES price offset (if needed).
                           Set to 0 when chain is already in ES terms.
        """
        super().__init__()
        self._parquet_path = parquet_path
        self._spx_es_offset = spx_es_offset
        self._chain_cache: Dict[str, pd.DataFrame] = {}

    @property
    def name(self) -> str:
        return "gex_surface"

    @property
    def feature_names(self) -> List[str]:
        return list(GEX_FEATURE_NAMES)

    def _load_chain(self, trade_date_str: str) -> pd.DataFrame:
        if trade_date_str not in self._chain_cache:
            self._chain_cache[trade_date_str] = load_chain_for_date(
                trade_date_str, self._parquet_path
            )
        return self._chain_cache[trade_date_str]

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        result = ohlcv.copy()

        # Initialise feature columns to zero
        for col in GEX_FEATURE_NAMES:
            result[col] = 0.0

        # Pre-compute ATR-14 for the whole dataset (used for normalisation)
        atr_series = self._compute_atr(ohlcv, period=14)

        # Rolling GEX std accumulator (updated per day)
        gex_history: list[float] = []

        for trading_day, group in ohlcv.groupby("trading_day"):
            try:
                day_str = str(trading_day)[:10]

                # Load previous day's chain
                prev_day = (
                    pd.to_datetime(day_str) - pd.tseries.offsets.BDay(1)
                ).strftime("%Y-%m-%d")
                chain = self._load_chain(prev_day)
                if chain.empty:
                    continue

                # Use the day's opening price as reference spot (ES)
                spot_open = float(group["close"].iloc[0])
                ref_spot = spot_open + self._spx_es_offset

                model = DealerPositionModel(
                    chain,
                    spot=ref_spot,
                    reference_spot=ref_spot,
                )

                # Normalisation values
                atr_14 = float(atr_series.loc[group.index].median())
                gex_std = float(np.std(gex_history[-60:])) if len(gex_history) >= 10 else 1.0

                for idx, row in group.iterrows():
                    spot = float(row["close"]) + self._spx_es_offset
                    feats = extract_gex_features(
                        model, spot, atr_14=atr_14, gex_rolling_std=gex_std
                    )
                    for k, v in feats.items():
                        result.at[idx, k] = v

                # Track daily net GEX for rolling std
                day_close = float(group["close"].iloc[-1]) + self._spx_es_offset
                gex_history.append(model.gex_at(day_close))

            except Exception as e:
                print(f"Warning: GEX surface failed for {trading_day}: {e}")
                continue

        return result

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
        """Simple ATR on the full OHLCV (not grouped by day)."""
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
