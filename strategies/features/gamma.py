"""
Gamma feature provider.

Wraps the IVMapper from gex/gex_utils.py to compute nearby gamma score.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
import os

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


@FeatureRegistry.register('gamma')
class GammaFeatureProvider(BaseFeatureProvider):
    """
    Computes gamma-based features from options data.

    Uses IVMapper to calculate nearby gamma score based on
    open interest and implied volatility surfaces.
    """

    def __init__(self, gamma_shares_path: Optional[str] = None):
        """
        Initialize provider.

        Args:
            gamma_shares_path: Path to gamma shares parquet file.
                             Defaults to gex/gamma_shares_combined.parquet
        """
        super().__init__()
        self.gamma_shares_path = gamma_shares_path
        self._mapper = None

    @property
    def name(self) -> str:
        return "gamma"

    @property
    def feature_names(self) -> List[str]:
        return ['nearby_gamma_score']

    def _get_mapper(self):
        """Lazy load IVMapper."""
        if self._mapper is None:
            # Add gex to path if needed
            gex_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gex')
            if gex_path not in sys.path:
                sys.path.insert(0, gex_path)

            try:
                from gex_utils import IVMapper
                self._mapper = IVMapper()
            except ImportError as e:
                raise ImportError(
                    "Could not import IVMapper. Make sure gex/gex_utils.py exists. "
                    f"Error: {e}"
                )

        return self._mapper

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compute gamma features."""
        mapper = self._get_mapper()

        result = ohlcv.copy()
        result['nearby_gamma_score'] = 0.0

        # Process by trading day
        for trading_day, group in ohlcv.groupby('trading_day'):
            try:
                # Prepare DataFrame in expected format
                es_df = group.copy()

                # Ensure proper index
                if 'dt' in es_df.columns:
                    es_df = es_df.set_index('dt')

                # Add time column if needed
                if 'time' not in es_df.columns:
                    es_df['time'] = es_df.index.strftime('%H:%M:%S')

                gamma_scores = mapper.compute_gamma_stats(es_df)

                if isinstance(gamma_scores, np.ndarray) and len(gamma_scores) > 0:
                    result.loc[group.index, 'nearby_gamma_score'] = gamma_scores
            except Exception as e:
                # Log but don't fail on gamma computation errors
                print(f"Warning: Gamma computation failed for {trading_day}: {e}")
                continue

        return result

    def compute_for_day(
        self,
        es_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Compute gamma score for a single day.

        Args:
            es_df: DataFrame with dt index, must have 'close' and 'time' columns

        Returns:
            Array of nearby_gamma_score values
        """
        mapper = self._get_mapper()
        return mapper.compute_gamma_stats(es_df)
