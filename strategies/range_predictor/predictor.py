"""Range prediction inference.

Loads trained models and predicts daily/weekly/monthly/quarterly ES ranges.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from strategies.range_predictor.config import RangePredictorConfig, TIMEFRAME_HORIZONS
from strategies.range_predictor.features import (
    compute_range_features,
    compute_rth_gap_features,
    get_feature_names,
)


class RangePredictor:
    """Load trained models and predict ES ranges."""

    def __init__(self, config: Optional[RangePredictorConfig] = None):
        self.config = config or RangePredictorConfig()
        self.models: Dict[str, Dict[str, XGBRegressor]] = {}
        self.feature_names: List[str] = []
        self.rth_feature_names: List[str] = []
        self.timeframes: List[str] = []

    def load_models(self, model_dir: Optional[str] = None) -> None:
        """Load models from disk.

        Args:
            model_dir: Directory containing saved models. Defaults to config.model_dir.
        """
        model_dir = model_dir or self.config.model_dir

        meta_path = os.path.join(model_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata.json found in {model_dir}")

        with open(meta_path) as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.timeframes = metadata['timeframes']
        self.rth_feature_names = metadata.get('rth_feature_names', [])

        for tf in self.timeframes:
            self.models[tf] = {}
            # RTH models use rth_range_high/low_pct target names
            if tf == 'rth':
                targets = ['rth_range_high_pct', 'rth_range_low_pct']
            else:
                targets = ['range_high_pct', 'range_low_pct']
            for target in targets:
                key = f"{tf}/{target}"
                if key not in metadata:
                    continue
                fpath = os.path.join(model_dir, metadata[key]['file'])
                model = XGBRegressor()
                model.load_model(fpath)
                self.models[tf][target] = model

        print(f"Loaded models for timeframes: {self.timeframes}")

    def predict(self, daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Predict ranges for all timeframes using the latest bar.

        Args:
            daily: Daily OHLCV DataFrame (DatetimeIndex) with enough history
                   for feature computation (~250 bars minimum).

        Returns:
            Dict mapping timeframe to predicted range:
            {
                'daily': {
                    'range_low': 6746.0,
                    'range_high': 6857.0,
                    'range_width': 111.0,
                    'range_high_pct': 0.008,
                    'range_low_pct': 0.006,
                },
                ...
            }
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")

        features = compute_range_features(daily)

        # Filter to available feature names
        available = [f for f in self.feature_names if f in features.columns]
        X = features[available].iloc[[-1]].fillna(0.0).values

        last_close = daily['close'].iloc[-1]
        results = {}

        for tf in self.timeframes:
            if tf not in self.models:
                continue

            tf_result = {}

            if 'range_high_pct' in self.models[tf]:
                high_pct = self.models[tf]['range_high_pct'].predict(X)[0]
                tf_result['range_high_pct'] = float(high_pct)
                tf_result['range_high'] = round(
                    last_close * (1 + high_pct), 2
                )

            if 'range_low_pct' in self.models[tf]:
                low_pct = self.models[tf]['range_low_pct'].predict(X)[0]
                tf_result['range_low_pct'] = float(low_pct)
                tf_result['range_low'] = round(
                    last_close * (1 - low_pct), 2
                )

            if 'range_high' in tf_result and 'range_low' in tf_result:
                tf_result['range_width'] = round(
                    tf_result['range_high'] - tf_result['range_low'], 2
                )

            results[tf] = tf_result

        return results

    def predict_series(
        self,
        daily: pd.DataFrame,
        timeframe: str = 'daily',
    ) -> pd.DataFrame:
        """Predict ranges for every day in the dataset (for backtesting).

        Args:
            daily: Daily OHLCV DataFrame.
            timeframe: Timeframe to predict.

        Returns:
            DataFrame with predicted range columns, aligned to daily index.
        """
        if timeframe not in self.models:
            raise ValueError(f"No model for timeframe '{timeframe}'")

        features = compute_range_features(daily)
        available = [f for f in self.feature_names if f in features.columns]
        X = features[available].fillna(0.0).values

        preds = pd.DataFrame(index=daily.index)

        for target in ['range_high_pct', 'range_low_pct']:
            if target in self.models[timeframe]:
                preds[f'pred_{target}'] = self.models[timeframe][target].predict(X)

        # Convert to price levels
        prev_close = daily['close'].shift(1)
        if 'pred_range_high_pct' in preds.columns:
            preds['pred_range_high'] = prev_close * (1 + preds['pred_range_high_pct'])
        if 'pred_range_low_pct' in preds.columns:
            preds['pred_range_low'] = prev_close * (1 - preds['pred_range_low_pct'])

        return preds

    def predict_rth(
        self,
        daily: pd.DataFrame,
        rth_open: float,
        rth_daily: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """Predict RTH range for today given the RTH open price.

        Args:
            daily: Full-session daily OHLCV (enough history for features).
            rth_open: Today's RTH open price.
            rth_daily: Optional RTH daily history (for gap features). If None,
                       computes gap from rth_open vs last daily close.

        Returns:
            Dict with pred_rth_high, pred_rth_low, pred_rth_width (price levels).
        """
        if 'rth' not in self.models:
            raise ValueError("No RTH models loaded. Train with --rth flag first.")

        features = compute_range_features(daily)

        # Compute gap features
        last_close = daily['close'].iloc[-1]
        if rth_daily is not None and len(rth_daily) > 0:
            prev_rth_close = rth_daily['rth_close'].iloc[-1]
        else:
            prev_rth_close = last_close

        ovn_gap_pct = (rth_open - prev_rth_close) / prev_rth_close
        features.loc[features.index[-1], 'ovn_gap_pct'] = ovn_gap_pct
        features.loc[features.index[-1], 'ovn_gap_abs_pct'] = abs(ovn_gap_pct)

        feat_names = self.rth_feature_names or self.feature_names
        available = [f for f in feat_names if f in features.columns]
        X = features[available].iloc[[-1]].fillna(0.0).values

        result = {}
        rth_models = self.models['rth']

        if 'rth_range_high_pct' in rth_models:
            high_pct = float(rth_models['rth_range_high_pct'].predict(X)[0])
            result['pred_rth_high_pct'] = high_pct
            result['pred_rth_high'] = round(rth_open * (1 + high_pct), 2)

        if 'rth_range_low_pct' in rth_models:
            low_pct = float(rth_models['rth_range_low_pct'].predict(X)[0])
            result['pred_rth_low_pct'] = low_pct
            result['pred_rth_low'] = round(rth_open * (1 - low_pct), 2)

        if 'pred_rth_high' in result and 'pred_rth_low' in result:
            result['pred_rth_width'] = round(
                result['pred_rth_high'] - result['pred_rth_low'], 2
            )

        return result

    def predict_rth_series(
        self,
        full_daily: pd.DataFrame,
        rth_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict RTH range for every day in the dataset (for backtesting).

        Uses actual rth_open from rth_daily for conversion.

        Args:
            full_daily: Full-session daily OHLCV DataFrame.
            rth_daily: RTH daily DataFrame with rth_open/high/low/close.

        Returns:
            DataFrame with pred_rth_high/low/width columns, indexed by date.
        """
        if 'rth' not in self.models:
            raise ValueError("No RTH models loaded.")

        features = compute_range_features(full_daily)

        # Add gap features
        gap_features = compute_rth_gap_features(full_daily, rth_daily)
        features = features.join(gap_features, how='left')

        feat_names = self.rth_feature_names or self.feature_names
        available = [f for f in feat_names if f in features.columns]

        # Align to common dates between features and rth_daily
        common_idx = features.index.intersection(rth_daily.index)
        X = features.loc[common_idx, available].fillna(0.0).values

        preds = pd.DataFrame(index=common_idx)

        rth_models = self.models['rth']
        for target in ['rth_range_high_pct', 'rth_range_low_pct']:
            if target in rth_models:
                preds[f'pred_{target}'] = rth_models[target].predict(X)

        # Convert pct to price levels using rth_open
        rth_open = rth_daily['rth_open'].reindex(common_idx)
        if 'pred_rth_range_high_pct' in preds.columns:
            preds['pred_rth_high'] = rth_open * (1 + preds['pred_rth_range_high_pct'])
        if 'pred_rth_range_low_pct' in preds.columns:
            preds['pred_rth_low'] = rth_open * (1 - preds['pred_rth_range_low_pct'])
        if 'pred_rth_high' in preds.columns and 'pred_rth_low' in preds.columns:
            preds['pred_rth_width'] = preds['pred_rth_high'] - preds['pred_rth_low']

        return preds

    def save_predictions_csv(
        self,
        predictions: Dict[str, Dict[str, float]],
        output_path: str,
        date: Optional[str] = None,
    ) -> None:
        """Save predictions to CSV for use by other strategies.

        Args:
            predictions: Output from predict().
            output_path: CSV path.
            date: Date string (defaults to today).
        """
        if date is None:
            date = pd.Timestamp.now().strftime('%Y-%m-%d')

        rows = []
        for tf, vals in predictions.items():
            rows.append({
                'date': date,
                'timeframe': tf,
                'range_low': vals.get('range_low'),
                'range_high': vals.get('range_high'),
                'range_width': vals.get('range_width'),
            })

        df = pd.DataFrame(rows)

        # Append or create
        if os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            # Remove existing entries for this date
            existing = existing[existing['date'] != date]
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
