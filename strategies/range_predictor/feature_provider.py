"""FeatureRegistry integration for range predictor.

Provides predicted range features for use by other strategies (breakout, reversion).
Features include predicted range width and position within predicted range.
"""

import os
from typing import Dict, List, Optional, Any

import pandas as pd

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry
from strategies.range_predictor.config import RangePredictorConfig
from strategies.range_predictor.predictor import RangePredictor
from strategies.range_predictor.features import aggregate_to_daily, aggregate_to_rth_daily


@FeatureRegistry.register('range_predictor')
class RangePredictorFeatureProvider(BaseFeatureProvider):
    """Provides predicted range features for other strategies.

    Loads a trained range predictor and computes features like:
    - pred_range_width_daily: predicted daily range width in points
    - pred_range_pos_daily: where price is within predicted range (0=low, 1=high)
    - pred_range_width_weekly: predicted weekly range width
    - pred_range_pos_weekly: position in weekly range
    """

    DAILY_FEATURES = [
        'pred_range_width_daily',
        'pred_range_pos_daily',
        'pred_range_high_daily',
        'pred_range_low_daily',
    ]

    WEEKLY_FEATURES = [
        'pred_range_width_weekly',
        'pred_range_pos_weekly',
    ]

    RTH_FEATURES = [
        'pred_rth_high',
        'pred_rth_low',
        'pred_rth_width',
    ]

    def __init__(
        self,
        model_dir: str = 'models/range_predictor',
        timeframes: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.timeframes = timeframes or ['daily', 'weekly']
        self._predictor: Optional[RangePredictor] = None
        self._predictions_cache: Optional[Dict] = None

    @property
    def name(self) -> str:
        return "range_predictor"

    @property
    def feature_names(self) -> List[str]:
        features = list(self.DAILY_FEATURES)
        if 'weekly' in self.timeframes:
            features.extend(self.WEEKLY_FEATURES)
        # Include RTH features if RTH models are loaded
        if self._predictor is not None and 'rth' in self._predictor.models:
            features.extend(self.RTH_FEATURES)
        return features

    def _ensure_predictor(self) -> None:
        """Lazy-load predictor models."""
        if self._predictor is not None:
            return

        if not os.path.exists(os.path.join(self.model_dir, 'metadata.json')):
            raise FileNotFoundError(
                f"No trained range predictor found in {self.model_dir}. "
                "Run: python -m strategies.range_predictor train"
            )

        config = RangePredictorConfig(model_dir=self.model_dir)
        self._predictor = RangePredictor(config)
        self._predictor.load_models()

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Compute range predictor features from 1-min OHLCV data.

        Args:
            ohlcv: 1-min DataFrame with trading_day, dt, OHLCV columns.
            context: Optional context (unused).

        Returns:
            DataFrame with predicted range features, same index as ohlcv.
        """
        self._ensure_predictor()
        result = ohlcv.copy()

        # Aggregate to daily for predictions
        daily = aggregate_to_daily(ohlcv)

        # Predict for each timeframe
        for tf in self.timeframes:
            if tf not in self._predictor.timeframes:
                continue

            preds = self._predictor.predict_series(daily, timeframe=tf)

            # Map daily predictions to trading_day
            day_map = ohlcv.groupby('trading_day')['dt'].first().dt.date
            pred_by_day = {}
            for td_int, date in day_map.items():
                date_ts = pd.Timestamp(date)
                if date_ts in preds.index:
                    pred_by_day[td_int] = preds.loc[date_ts]

            # Broadcast to intraday bars
            if tf == 'daily':
                highs = []
                lows = []
                for idx, row in ohlcv.iterrows():
                    td = row['trading_day']
                    pred_row = pred_by_day.get(td)
                    if pred_row is not None and 'pred_range_high' in pred_row.index:
                        highs.append(pred_row['pred_range_high'])
                        lows.append(pred_row['pred_range_low'])
                    else:
                        highs.append(float('nan'))
                        lows.append(float('nan'))

                result['pred_range_high_daily'] = highs
                result['pred_range_low_daily'] = lows
                result['pred_range_width_daily'] = (
                    result['pred_range_high_daily'] - result['pred_range_low_daily']
                )

                width = result['pred_range_width_daily'].replace(0, float('nan'))
                result['pred_range_pos_daily'] = (
                    (result['close'] - result['pred_range_low_daily']) / width
                )

            elif tf == 'weekly':
                # For weekly, use the Monday prediction for the whole week
                widths = []
                for idx, row in ohlcv.iterrows():
                    td = row['trading_day']
                    pred_row = pred_by_day.get(td)
                    if pred_row is not None and 'pred_range_high' in pred_row.index:
                        w = pred_row['pred_range_high'] - pred_row['pred_range_low']
                        pos = (
                            (row['close'] - pred_row['pred_range_low']) / w
                            if w > 0 else 0.5
                        )
                        widths.append((w, pos))
                    else:
                        widths.append((float('nan'), float('nan')))

                result['pred_range_width_weekly'] = [w[0] for w in widths]
                result['pred_range_pos_weekly'] = [w[1] for w in widths]

        # RTH predictions (if RTH models are loaded)
        if 'rth' in self._predictor.models and 'ovn' in ohlcv.columns:
            rth_daily = aggregate_to_rth_daily(ohlcv)
            rth_preds = self._predictor.predict_rth_series(daily, rth_daily)

            # Broadcast RTH predictions to RTH bars via trading_day mapping
            rth_highs = []
            rth_lows = []
            rth_widths = []
            for _, row in ohlcv.iterrows():
                td = row['trading_day']
                pred_row = pred_by_day.get(td)
                date = day_map.get(td)
                date_ts = pd.Timestamp(date) if date is not None else None

                if date_ts is not None and date_ts in rth_preds.index:
                    rp = rth_preds.loc[date_ts]
                    rth_highs.append(rp.get('pred_rth_high', float('nan')))
                    rth_lows.append(rp.get('pred_rth_low', float('nan')))
                    rth_widths.append(rp.get('pred_rth_width', float('nan')))
                else:
                    rth_highs.append(float('nan'))
                    rth_lows.append(float('nan'))
                    rth_widths.append(float('nan'))

            result['pred_rth_high'] = rth_highs
            result['pred_rth_low'] = rth_lows
            result['pred_rth_width'] = rth_widths

        # Fill NaN
        all_features = list(self.DAILY_FEATURES)
        if 'weekly' in self.timeframes:
            all_features.extend(self.WEEKLY_FEATURES)
        if 'rth' in self._predictor.models:
            all_features.extend(self.RTH_FEATURES)
        for col in all_features:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)

        return result
