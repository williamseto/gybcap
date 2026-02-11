"""
RealtimeStrategy implementation for Phase 3 reversal prediction.

Predicts P(reversal) at near-level bars using the Phase 3 XGBoost model.
Replicates the exact feature computation pipeline from train_level_models.py
to ensure consistency between offline training and realtime inference.

Requires historical context (50-60 days of 1-min bars) for feature warm-up.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from strategies.labeling.reversal_zones import TRACKED_LEVELS
from strategies.realtime.protocol import RealtimeSignal

logger = logging.getLogger(__name__)


class ReversalPredictorStrategy:
    """Predicts P(reversal) at near-level bars using Phase 3 model."""

    def __init__(
        self,
        model_dir: str,
        pred_threshold: float = 0.50,
        proximity_pts: float = 5.0,
    ):
        self._model_dir = model_dir
        self._pred_threshold = pred_threshold
        self._proximity_pts = proximity_pts

        # Load model and metadata
        self._model = xgb.XGBClassifier()
        self._model.load_model(os.path.join(model_dir, 'model.json'))

        with open(os.path.join(model_dir, 'metadata.json')) as f:
            self._metadata = json.load(f)

        self._feature_cols: List[str] = self._metadata['feature_cols']

        # Historical context for feature warm-up
        self._historical_ohlcv: Optional[pd.DataFrame] = None

        # Dedup: track emitted signals by (level_name, direction, bar_timestamp)
        self._emitted: Set[Tuple[str, str, str]] = set()

        # Cache last processed bar count to avoid reprocessing
        self._last_processed_count: int = 0

        logger.info(
            "ReversalPredictorStrategy loaded: %d features, threshold=%.2f",
            len(self._feature_cols), self._pred_threshold,
        )

    @property
    def name(self) -> str:
        return "reversal_predictor"

    # ------------------------------------------------------------------
    # Historical context
    # ------------------------------------------------------------------

    def set_historical_context(self, ohlcv_history: pd.DataFrame) -> None:
        """Provide multi-day history for feature warm-up.

        Called once during initialization with 50-60 days of 1-min bars.
        Must have columns: open, high, low, close, volume, trading_day.
        """
        self._historical_ohlcv = ohlcv_history.copy()
        n_days = ohlcv_history['trading_day'].nunique() if 'trading_day' in ohlcv_history.columns else 0
        logger.info("Historical context set: %d bars, %d days", len(ohlcv_history), n_days)

    # ------------------------------------------------------------------
    # Main process
    # ------------------------------------------------------------------

    def process(self, bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        """Process current day's 1-min bars. Returns new signals.

        Args:
            bars_1m: 1-minute OHLCV bars with datetime index (LA timezone).
                     Must have columns: open, high, low, close, volume.
                     May have: buys/sells (or bidvolume/askvolume), price.
        """
        if bars_1m.empty or len(bars_1m) <= self._last_processed_count:
            return []

        # Combine historical + current day into a single DataFrame
        combined = self._build_combined_df(bars_1m)
        if combined is None or len(combined) < 100:
            return []

        # Compute all features on the combined data
        featured = self._compute_features(combined)
        if featured is None:
            return []

        # Extract only current-day bars for prediction
        current_day_start = len(combined) - len(bars_1m)
        current_day_df = featured.iloc[current_day_start:].copy()

        # Find nearest level and label for each bar
        current_day_df = self._find_nearest_levels(current_day_df)

        # Compute level-encoding features
        current_day_df = self._compute_level_encoding(current_day_df, featured)

        # Predict on new near-level bars
        new_signals = self._predict_and_emit(current_day_df, bars_1m)

        self._last_processed_count = len(bars_1m)
        return new_signals

    def reset_day(self) -> None:
        """Reset per-day state for a new trading day."""
        self._emitted.clear()
        self._last_processed_count = 0

    # ------------------------------------------------------------------
    # Feature computation (replicates train_level_models.py pipeline)
    # ------------------------------------------------------------------

    def _build_combined_df(self, bars_1m: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Combine historical context with current day bars."""
        # Convert bars_1m from datetime-indexed to flat DataFrame
        current = bars_1m.copy()
        if current.index.name == 'dt' or isinstance(current.index, pd.DatetimeIndex):
            current = current.reset_index()
            if 'dt' not in current.columns:
                current.rename(columns={current.columns[0]: 'dt'}, inplace=True)

        # Ensure dt column exists
        if 'dt' not in current.columns and isinstance(bars_1m.index, pd.DatetimeIndex):
            current['dt'] = bars_1m.index

        # Ensure required columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in current.columns:
                logger.warning("Missing column %s in bars_1m", col)
                return None

        # Add trading_day if missing
        if 'trading_day' not in current.columns and 'dt' in current.columns:
            current['trading_day'] = self._compute_trading_day(current['dt'])

        if self._historical_ohlcv is not None and len(self._historical_ohlcv) > 0:
            historical = self._historical_ohlcv.copy()
            # Ensure consistent columns
            common_cols = list(set(historical.columns) & set(current.columns))
            combined = pd.concat([historical[common_cols], current[common_cols]], ignore_index=True)
        else:
            combined = current

        return combined

    @staticmethod
    def _compute_trading_day(dt_series: pd.Series) -> pd.Series:
        """Compute trading_day from datetime series (6 PM ET boundary)."""
        # Convert to naive if tz-aware
        if hasattr(dt_series.dt, 'tz') and dt_series.dt.tz is not None:
            dt_naive = dt_series.dt.tz_localize(None)
        else:
            dt_naive = dt_series

        # Bars after 3 PM LA (6 PM ET) belong to next day
        trading_day = dt_naive.dt.date.astype(str)
        after_boundary = dt_naive.dt.hour >= 15
        next_day = (dt_naive + pd.Timedelta(days=1)).dt.date.astype(str)
        trading_day = trading_day.where(~after_boundary, next_day)
        return trading_day

    def _compute_features(self, ohlcv: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Replicate the exact feature pipeline from train_level_models.py."""
        try:
            # 1. Compute price levels
            ohlcv = self._compute_levels(ohlcv)

            # 2. Compute all feature providers
            ohlcv = self._compute_all_providers(ohlcv)

            return ohlcv
        except Exception as e:
            logger.error("Feature computation failed: %s", e, exc_info=True)
            return None

    @staticmethod
    def _compute_levels(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute price levels (mirrors train_level_models.compute_levels)."""
        from strategies.features.price_levels import PriceLevelProvider

        plp = PriceLevelProvider(include_gamma='gamma_score' in ohlcv.columns)
        feat_df = plp._compute_impl(ohlcv)

        level_cols = ['vwap', 'ovn_lo', 'ovn_hi', 'rth_lo', 'rth_hi', 'ib_lo', 'ib_hi']
        if 'dt' in feat_df.columns:
            feat_df = feat_df.set_index('dt')
        ohlcv_dt = ohlcv.set_index('dt') if 'dt' in ohlcv.columns else ohlcv
        for col in level_cols:
            if col in feat_df.columns:
                ohlcv[col] = feat_df[col].reindex(ohlcv_dt.index).values

        levels = plp.prev_day_levels(ohlcv)
        ohlcv['prev_high'] = ohlcv['trading_day'].map(levels['prev_high'])
        ohlcv['prev_low'] = ohlcv['trading_day'].map(levels['prev_low'])
        return ohlcv

    @staticmethod
    def _compute_all_providers(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Run all feature providers (mirrors train_level_models.compute_all_features)."""
        from strategies.features.higher_timeframe import HigherTimeframeProvider
        from strategies.features.volume_microstructure import VolumeMicrostructureProvider
        from strategies.features.reversion_quality import ReversionQualityProvider
        from strategies.features.temporal_interactions import TemporalInteractionProvider

        htf = HigherTimeframeProvider()
        htf_df = htf._compute_impl(ohlcv)
        for col in htf.feature_names:
            if col in htf_df.columns:
                ohlcv[col] = htf_df[col].values

        has_bidask = 'bidvolume' in ohlcv.columns
        vmp = VolumeMicrostructureProvider(include_bidask=has_bidask)
        vol_df = vmp._compute_impl(ohlcv)
        for col in vmp.feature_names:
            if col in vol_df.columns:
                ohlcv[col] = vol_df[col].values

        rqp = ReversionQualityProvider()
        qual_df = rqp._compute_impl(ohlcv)
        for col in rqp.feature_names:
            if col in qual_df.columns:
                ohlcv[col] = qual_df[col].values

        tip = TemporalInteractionProvider()
        temp_df = tip._compute_impl(ohlcv)
        for col in tip.feature_names:
            if col in temp_df.columns:
                ohlcv[col] = temp_df[col].values

        return ohlcv

    def _find_nearest_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find nearest tracked level for each bar (mirrors label_reversals_breakouts logic)."""
        close_arr = df['close'].values.astype(np.float64)
        n = len(df)

        nearest_level_name = np.empty(n, dtype=object)
        nearest_level_price = np.full(n, np.nan)
        side_arr = np.zeros(n, dtype=np.int8)

        level_arrs = {}
        for lvl_name in TRACKED_LEVELS:
            if lvl_name in df.columns:
                level_arrs[lvl_name] = df[lvl_name].values.astype(np.float64)

        for lvl_name, lvl_vals in level_arrs.items():
            for i in range(n):
                d = abs(close_arr[i] - lvl_vals[i])
                if np.isnan(d):
                    continue
                if d <= self._proximity_pts:
                    prev_dist = abs(close_arr[i] - nearest_level_price[i]) if not np.isnan(nearest_level_price[i]) else np.inf
                    if d < prev_dist:
                        nearest_level_name[i] = lvl_name
                        nearest_level_price[i] = lvl_vals[i]

        df['nearest_level_name'] = nearest_level_name
        df['nearest_level_price'] = nearest_level_price
        df['side'] = side_arr
        # Set side: 1 = above level (support), -1 = below level (resistance)
        near_mask = ~np.isnan(nearest_level_price)
        above = near_mask & (close_arr >= nearest_level_price)
        below = near_mask & (close_arr < nearest_level_price)
        df.loc[df.index[above], 'side'] = 1
        df.loc[df.index[below], 'side'] = -1

        return df

    def _compute_level_encoding(self, current_df: pd.DataFrame,
                                 full_df: pd.DataFrame) -> pd.DataFrame:
        """Compute level-encoding features (mirrors compute_level_encoding_features).

        Uses full_df for trailing reversal rates (needs historical outcomes).
        current_df gets the features added.
        """
        nearest = current_df['nearest_level_name'].values

        # 1. One-hot encoding
        for lvl in TRACKED_LEVELS:
            current_df[f'is_{lvl}'] = (nearest == lvl).astype(np.float32)

        # 2. level_is_support
        current_df['level_is_support'] = (current_df['side'] == 1).astype(np.float32)

        # 3. Trailing reversal rates — set to 0.5 (neutral) since we don't have
        # outcomes during realtime. In playback mode, these come from historical data.
        current_df['level_trailing_rev_rate_20d'] = 0.5
        current_df['level_trailing_rev_rate_50d'] = 0.5

        # If full_df has outcome data (playback mode), compute actual rates
        if 'outcome' in full_df.columns:
            self._compute_trailing_rates(current_df, full_df)

        # 4. BB interaction
        if 'daily_bb_pct_b' in current_df.columns:
            current_df['level_side_bb_interaction'] = (
                current_df['level_is_support'] * current_df['daily_bb_pct_b']
            )
        else:
            current_df['level_side_bb_interaction'] = 0.0

        # 5. Approach direction
        close_vals = current_df['close'].values.astype(np.float64)
        lvl_price_vals = current_df['nearest_level_price'].values.astype(np.float64)
        n = len(current_df)
        approach_dir = np.zeros(n, dtype=np.float32)
        for i in range(5, n):
            if np.isnan(lvl_price_vals[i]):
                continue
            price_change = close_vals[i] - close_vals[i - 5]
            if close_vals[i] >= lvl_price_vals[i]:
                approach_dir[i] = -1.0 if price_change < 0 else 1.0
            else:
                approach_dir[i] = 1.0 if price_change > 0 else -1.0
        current_df['approach_dir_vs_level'] = approach_dir

        return current_df

    def _compute_trailing_rates(self, current_df: pd.DataFrame,
                                 full_df: pd.DataFrame) -> None:
        """Compute trailing reversal rates from historical outcomes."""
        if 'outcome' not in full_df.columns or 'trading_day' not in full_df.columns:
            return

        near_mask = full_df['outcome'].isin([0, 1])
        if not near_mask.any():
            return

        near_df = full_df.loc[near_mask, ['trading_day', 'nearest_level_name', 'outcome']].copy()
        days = sorted(full_df['trading_day'].unique())

        daily_stats = near_df.groupby(['trading_day', 'nearest_level_name']).agg(
            n_rev=('outcome', lambda x: (x == 1).sum()),
            n_total=('outcome', 'count'),
        ).reset_index()

        for lvl in TRACKED_LEVELS:
            lvl_stats = daily_stats[daily_stats['nearest_level_name'] == lvl].copy()
            lvl_stats = lvl_stats.set_index('trading_day').reindex(days).fillna(0)

            lvl_stats['rev_20d'] = lvl_stats['n_rev'].rolling(20, min_periods=1).sum()
            lvl_stats['total_20d'] = lvl_stats['n_total'].rolling(20, min_periods=1).sum()
            lvl_stats['rev_rate_20d'] = lvl_stats['rev_20d'] / lvl_stats['total_20d'].clip(lower=1)

            lvl_stats['rev_50d'] = lvl_stats['n_rev'].rolling(50, min_periods=1).sum()
            lvl_stats['total_50d'] = lvl_stats['n_total'].rolling(50, min_periods=1).sum()
            lvl_stats['rev_rate_50d'] = lvl_stats['rev_50d'] / lvl_stats['total_50d'].clip(lower=1)

            # Shift by 1 for causality
            lvl_stats['rev_rate_20d'] = lvl_stats['rev_rate_20d'].shift(1)
            lvl_stats['rev_rate_50d'] = lvl_stats['rev_rate_50d'].shift(1)

            rate_20d_map = lvl_stats['rev_rate_20d'].to_dict()
            rate_50d_map = lvl_stats['rev_rate_50d'].to_dict()

            lvl_mask = current_df['nearest_level_name'] == lvl
            if lvl_mask.any() and 'trading_day' in current_df.columns:
                mapped_20d = current_df.loc[lvl_mask, 'trading_day'].map(rate_20d_map)
                mapped_50d = current_df.loc[lvl_mask, 'trading_day'].map(rate_50d_map)
                current_df.loc[lvl_mask, 'level_trailing_rev_rate_20d'] = mapped_20d.fillna(0.5)
                current_df.loc[lvl_mask, 'level_trailing_rev_rate_50d'] = mapped_50d.fillna(0.5)

    # ------------------------------------------------------------------
    # Prediction and signal emission
    # ------------------------------------------------------------------

    def _predict_and_emit(self, df: pd.DataFrame,
                          bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        """Run XGBoost on near-level bars and emit signals above threshold."""
        near_mask = ~df['nearest_level_price'].isna()
        if not near_mask.any():
            return []

        near_df = df.loc[near_mask].copy()
        signals = []

        # Gather feature matrix — only include features the model expects
        available_cols = [c for c in self._feature_cols if c in near_df.columns]
        missing_cols = [c for c in self._feature_cols if c not in near_df.columns]
        if missing_cols:
            logger.debug("Missing %d feature cols (filling with 0): %s",
                         len(missing_cols), missing_cols[:5])

        # Build feature matrix with correct column ordering
        X = pd.DataFrame(index=near_df.index)
        for col in self._feature_cols:
            if col in near_df.columns:
                X[col] = near_df[col].values
            else:
                X[col] = 0.0

        X = X.fillna(0).values.astype(np.float32)

        # Predict
        y_prob = self._model.predict_proba(X)[:, 1]

        # Emit signals for high-probability predictions
        for i, (idx, row) in enumerate(near_df.iterrows()):
            prob = float(y_prob[i])
            if prob < self._pred_threshold:
                continue

            level_name = row['nearest_level_name']
            side = int(row['side'])
            direction = 'bull' if side == 1 else 'bear'

            # Determine bar timestamp for dedup
            if 'dt' in row.index and pd.notna(row['dt']):
                bar_ts = str(row['dt'])
            else:
                bar_ts = str(idx)

            dedup_key = (level_name, direction, bar_ts)
            if dedup_key in self._emitted:
                continue

            self._emitted.add(dedup_key)

            # Build entry timestamp from bars_1m index
            if isinstance(bars_1m.index, pd.DatetimeIndex) and idx < len(bars_1m):
                entry_ts = bars_1m.index[idx] if isinstance(idx, int) else pd.Timestamp(bar_ts)
            else:
                entry_ts = pd.Timestamp(bar_ts) if bar_ts != str(idx) else pd.Timestamp.now()

            signals.append(RealtimeSignal(
                strategy_name=self.name,
                trigger_ts=entry_ts,
                entry_ts=entry_ts,
                entry_price=float(row['close']),
                direction=direction,
                level_name=str(level_name),
                level_value=float(row['nearest_level_price']),
                pred_proba=prob,
                metadata={
                    'proximity': float(abs(row['close'] - row['nearest_level_price'])),
                },
            ))

        return signals
