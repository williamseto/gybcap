"""
Dalton day type feature provider.

Wraps the OnlinePredictor from vp/dalton_day_classifier.py
to provide real-time day type classification features.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
import os

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


@FeatureRegistry.register('dalton')
class DaltonFeatureProvider(BaseFeatureProvider):
    """
    Provides Dalton day type classification features.

    Uses OnlinePredictor to classify the current trading day as
    trend, balance, or other day types based on intraday price action.

    Features:
    - dalton_trend_prob: Probability of trend day
    - dalton_balance_prob: Probability of balance day
    - dalton_predicted_type: Predicted day type (encoded)
    """

    # Day type encoding
    DAY_TYPES = {
        'Trend': 0,
        'Trend Day': 0,
        'Normal Day': 1,
        'Normal': 1,
        'Normal Variation': 2,
        'Normal Variation Day': 2,
        'Neutral Day': 3,
        'Neutral': 3,
        'Other': 4,
        'Unknown': 4,
    }

    DEFAULT_MODEL_PATH = 'vp/dalton_artifacts/dalton_classifier.joblib'

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize provider.

        Args:
            model_path: Path to saved DaltonClassifier model.
                       Defaults to vp/dalton_artifacts/dalton_classifier.joblib
        """
        super().__init__()
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self._predictor = None
        self._current_day = None

    @property
    def name(self) -> str:
        return "dalton"

    @property
    def feature_names(self) -> List[str]:
        return ['dalton_trend_prob', 'dalton_balance_prob', 'dalton_predicted_type']

    def _get_predictor(self):
        """Lazy load OnlinePredictor."""
        if self._predictor is None:
            # Add vp to path if needed
            vp_path = os.path.join(os.path.dirname(__file__), '..', '..', 'vp')
            if vp_path not in sys.path:
                sys.path.insert(0, vp_path)

            try:
                from dalton_day_classifier import OnlinePredictor
                self._predictor = OnlinePredictor(self.model_path)
            except ImportError as e:
                raise ImportError(
                    "Could not import OnlinePredictor. Make sure "
                    "vp/dalton_day_classifier.py exists. "
                    f"Error: {e}"
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Dalton model not found at {self.model_path}. "
                    "Train a model first or provide correct path. "
                    f"Error: {e}"
                )

        return self._predictor

    def _encode_day_type(self, day_type: str) -> int:
        """Encode day type string to integer."""
        return self.DAY_TYPES.get(day_type, 4)  # Default to Unknown

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Compute Dalton features for all bars."""
        predictor = self._get_predictor()

        result = ohlcv.copy()
        result['dalton_trend_prob'] = 0.0
        result['dalton_balance_prob'] = 0.0
        result['dalton_predicted_type'] = 4  # Unknown

        # Process by trading day
        for trading_day in ohlcv['trading_day'].unique():
            # Reset predictor for new day
            predictor.reset_day()
            self._current_day = trading_day

            day_mask = ohlcv['trading_day'] == trading_day
            day_bars = ohlcv[day_mask]

            for idx, row in day_bars.iterrows():
                # Build OHLCV bar dict
                bar_dict = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                }

                # Add bid/ask volumes if available
                if 'bidvolume' in row and 'askvolume' in row:
                    bar_dict['bid_volume'] = row['bidvolume']
                    bar_dict['ask_volume'] = row['askvolume']

                try:
                    predicted_type, probs = predictor.update(bar_dict)

                    # Extract probabilities
                    # Assume first prob is trend, second is balance/normal
                    trend_prob = probs[0] if len(probs) > 0 else 0.0
                    balance_prob = probs[1] if len(probs) > 1 else 0.0

                    result.loc[idx, 'dalton_trend_prob'] = trend_prob
                    result.loc[idx, 'dalton_balance_prob'] = balance_prob
                    result.loc[idx, 'dalton_predicted_type'] = self._encode_day_type(predicted_type)
                except Exception as e:
                    # Log but don't fail
                    print(f"Warning: Dalton prediction failed at {idx}: {e}")
                    continue

        return result

    def update_single_bar(self, bar_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Update with a single bar and return features.

        Useful for real-time prediction in live trading.

        Args:
            bar_dict: Dictionary with open, high, low, close, volume keys

        Returns:
            Dictionary with dalton_trend_prob, dalton_balance_prob, dalton_predicted_type
        """
        predictor = self._get_predictor()
        predicted_type, probs = predictor.update(bar_dict)

        trend_prob = probs[0] if len(probs) > 0 else 0.0
        balance_prob = probs[1] if len(probs) > 1 else 0.0

        return {
            'dalton_trend_prob': trend_prob,
            'dalton_balance_prob': balance_prob,
            'dalton_predicted_type': self._encode_day_type(predicted_type),
        }

    def reset_day(self):
        """Reset the predictor for a new trading day."""
        if self._predictor is not None:
            self._predictor.reset_day()
