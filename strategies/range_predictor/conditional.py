"""Intraday conditional range prediction.

Predicts full-day realized high/low, updated every 30 minutes during RTH
using observed intraday action. As the session progresses, predictions
increasingly reflect what's already happened.

Usage (inference):

    predictor = ConditionalRangePredictor.load('models/range_predictor/conditional')
    result = predictor.predict(intraday_state)
    # result = {'pred_high': 5623.5, 'pred_low': 5571.0, 'pred_width': 52.5, ...}

Usage (training):

    trainer = ConditionalRangeTrainer()
    trainer.train(df_1min, daily, verbose=True)
    trainer.save('models/range_predictor/conditional')

Walk-forward OOS evaluation:

    results = trainer.walk_forward_evaluate(df_1min, daily)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from strategies.range_predictor.features import (
    aggregate_to_daily,
    _compute_atr,
)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class ConditionalRangeConfig:
    """Configuration for intraday conditional range prediction."""

    checkpoint_interval: int = 30     # minutes between updates
    n_folds: int = 5                  # walk-forward folds
    min_train_days: int = 100         # minimum training window
    min_rth_bars: int = 60            # skip days with < 60 RTH bars

    xgb_params: dict = field(default_factory=lambda: {
        'objective': 'reg:squarederror',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    })


FEATURE_COLS = [
    # Pre-market context
    'prev_ret', 'prev_range_pct', 'atr_14_pct', 'atr_60_pct', 'dow', 'gamma',
    # Intraday observed
    'elapsed_frac', 'session_high_pct', 'session_low_pct', 'session_range_pct',
    'gap_pct', 'intraday_ret', 'volume_frac', 'range_vs_atr',
    'bid_ask_ratio',
    'recent_high_pct', 'recent_low_pct', 'recent_range_pct',
    'new_high_last30', 'new_low_last30',
    'close_position',
]


# ── Intraday state (input to predict()) ───────────────────────────────

@dataclass
class IntradayState:
    """Snapshot of current intraday state for prediction.

    All prices are absolute (not pct). The predictor handles conversion.
    """
    prev_close: float               # previous session close
    prev_high: float                 # previous session high
    prev_low: float                  # previous session low
    prev_prev_close: float          # close two days ago (for prev_ret)
    atr_14: float                   # 14-day ATR (shifted, from prior day)
    atr_60: float                   # 60-day ATR (shifted, from prior day)
    dow: int                        # day of week (0=Mon)
    gamma: float                    # gamma score (0 if unavailable)

    # Current session observations
    elapsed_bars: int               # RTH bars elapsed so far
    total_rth_bars: int             # total expected RTH bars (typically 450)
    session_high: float             # highest price in RTH so far
    session_low: float              # lowest price in RTH so far
    session_open: float             # RTH open price
    current_close: float            # most recent bar close
    cumulative_volume: float        # cumulative RTH volume
    avg_daily_volume: float         # average daily RTH volume
    bid_volume: float = 0.0         # cumulative bid volume
    ask_volume: float = 0.0         # cumulative ask volume

    # Recent bar data (last 30 bars)
    recent_high: float = 0.0        # high of last 30 bars
    recent_low: float = 0.0         # low of last 30 bars
    new_session_high_recent: bool = False   # session high set in last 30 bars
    new_session_low_recent: bool = False    # session low set in last 30 bars

    def to_features(self) -> dict:
        """Convert to feature dict for model input."""
        pc = self.prev_close
        if pc <= 0:
            raise ValueError("prev_close must be positive")

        elapsed_frac = self.elapsed_bars / self.total_rth_bars if self.total_rth_bars > 0 else 0
        prev_ret = (pc - self.prev_prev_close) / self.prev_prev_close if self.prev_prev_close > 0 else 0
        prev_range = self.prev_high - self.prev_low
        session_range = self.session_high - self.session_low

        return {
            'prev_ret': prev_ret,
            'prev_range_pct': prev_range / pc,
            'atr_14_pct': self.atr_14 / pc,
            'atr_60_pct': self.atr_60 / pc,
            'dow': self.dow,
            'gamma': self.gamma,
            'elapsed_frac': elapsed_frac,
            'session_high_pct': (self.session_high - pc) / pc,
            'session_low_pct': (pc - self.session_low) / pc,
            'session_range_pct': session_range / pc,
            'gap_pct': (self.session_open - pc) / pc,
            'intraday_ret': (self.current_close - pc) / pc,
            'volume_frac': self.cumulative_volume / self.avg_daily_volume if self.avg_daily_volume > 0 else 0,
            'range_vs_atr': session_range / self.atr_14 if self.atr_14 > 0 else 0,
            'bid_ask_ratio': self.bid_volume / (self.bid_volume + self.ask_volume) if (self.bid_volume + self.ask_volume) > 0 else 0.5,
            'recent_high_pct': (self.recent_high - self.current_close) / pc,
            'recent_low_pct': (self.current_close - self.recent_low) / pc,
            'recent_range_pct': (self.recent_high - self.recent_low) / pc,
            'new_high_last30': int(self.new_session_high_recent),
            'new_low_last30': int(self.new_session_low_recent),
            'close_position': (self.current_close - self.session_low) / session_range if session_range > 0 else 0.5,
        }


# ── Dataset construction ─────────────────────────────────────────────

def build_checkpoint_dataset(
    df_1min: pd.DataFrame,
    daily: pd.DataFrame,
    config: Optional[ConditionalRangeConfig] = None,
) -> pd.DataFrame:
    """Build training dataset: one row per (day, checkpoint).

    Args:
        df_1min: 1-min OHLCV with trading_day, dt, ovn columns.
        daily: Daily OHLCV (DatetimeIndex), from aggregate_to_daily.
        config: Configuration (checkpoint interval, min bars, etc).

    Returns:
        DataFrame with feature columns, target columns, and metadata
        (trading_day, date, checkpoint_bar).
    """
    if config is None:
        config = ConditionalRangeConfig()

    rth = df_1min[df_1min['ovn'] == 0].copy()
    td_vals = rth['trading_day'].values
    unique_tds = sorted(rth['trading_day'].unique())

    # Pre-compute daily context (all shifted by 1 to avoid lookahead)
    prev_close = daily['close'].shift(1)
    prev_ret = daily['close'].pct_change().shift(1)
    prev_range_pct = ((daily['high'] - daily['low']) / daily['close']).shift(1)
    atr_14 = _compute_atr(daily['high'], daily['low'], daily['close'], 14).shift(1)
    atr_60 = _compute_atr(daily['high'], daily['low'], daily['close'], 60).shift(1)

    day_map = df_1min.groupby('trading_day')['dt'].first().dt.date
    td_to_date = day_map.to_dict()

    gamma = daily.get('nearby_gamma_score', pd.Series(0.0, index=daily.index)).shift(1)

    # Average RTH volume per day (for normalizing volume_frac)
    avg_daily_vol = rth.groupby(td_vals)['volume'].sum().mean()

    rows = []

    for td in unique_tds:
        date = td_to_date.get(td)
        if date is None:
            continue
        date_ts = pd.Timestamp(date)
        if date_ts not in daily.index:
            continue

        pc = prev_close.get(date_ts)
        if pd.isna(pc) or pc <= 0:
            continue

        day_mask = td_vals == td
        day_rth = rth[day_mask]
        n_bars = len(day_rth)
        if n_bars < config.min_rth_bars:
            continue

        highs = day_rth['high'].values
        lows = day_rth['low'].values
        closes = day_rth['close'].values
        volumes = day_rth['volume'].values

        has_bidask = 'bidvolume' in day_rth.columns and 'askvolume' in day_rth.columns
        bid_vols = day_rth['bidvolume'].values if has_bidask else None
        ask_vols = day_rth['askvolume'].values if has_bidask else None

        # Full-day targets
        day_high = highs.max()
        day_low = lows.min()
        full_day_high_pct = (day_high - pc) / pc
        full_day_low_pct = (pc - day_low) / pc

        # Pre-market features
        a14 = atr_14.get(date_ts, 0.0)
        a60 = atr_60.get(date_ts, 0.0)
        pre = {
            'prev_ret': prev_ret.get(date_ts, 0.0),
            'prev_range_pct': prev_range_pct.get(date_ts, 0.0),
            'atr_14_pct': a14 / pc if pc > 0 else 0.0,
            'atr_60_pct': a60 / pc if pc > 0 else 0.0,
            'dow': date_ts.dayofweek,
            'gamma': gamma.get(date_ts, 0.0),
        }

        # Generate checkpoints
        checkpoints = list(range(config.checkpoint_interval, n_bars + 1,
                                 config.checkpoint_interval))
        if checkpoints and checkpoints[-1] != n_bars:
            checkpoints.append(n_bars)

        for cp_bar in checkpoints:
            obs_highs = highs[:cp_bar]
            obs_lows = lows[:cp_bar]
            obs_closes = closes[:cp_bar]
            obs_vols = volumes[:cp_bar]

            session_high = obs_highs.max()
            session_low = obs_lows.min()
            current_close = obs_closes[-1]
            session_range = session_high - session_low
            elapsed_frac = cp_bar / n_bars

            # Recent 30-bar window
            lookback = min(30, cp_bar)
            recent_highs = obs_highs[-lookback:]
            recent_lows = obs_lows[-lookback:]

            if cp_bar > lookback:
                prev_session_high = highs[:cp_bar - lookback].max()
                prev_session_low = lows[:cp_bar - lookback].min()
                new_hi = int(session_high > prev_session_high)
                new_lo = int(session_low < prev_session_low)
            else:
                new_hi = 1
                new_lo = 1

            intra = {
                'elapsed_frac': elapsed_frac,
                'session_high_pct': (session_high - pc) / pc,
                'session_low_pct': (pc - session_low) / pc,
                'session_range_pct': session_range / pc,
                'gap_pct': (closes[0] - pc) / pc,
                'intraday_ret': (current_close - pc) / pc,
                'volume_frac': obs_vols.sum() / avg_daily_vol if avg_daily_vol > 0 else 0,
                'range_vs_atr': session_range / a14 if a14 > 0 else 0,
                'recent_high_pct': (recent_highs.max() - current_close) / pc,
                'recent_low_pct': (current_close - recent_lows.min()) / pc,
                'recent_range_pct': (recent_highs.max() - recent_lows.min()) / pc,
                'new_high_last30': new_hi,
                'new_low_last30': new_lo,
                'close_position': (current_close - session_low) / session_range if session_range > 0 else 0.5,
            }

            if bid_vols is not None:
                obs_bid = bid_vols[:cp_bar].sum()
                obs_ask = ask_vols[:cp_bar].sum()
                intra['bid_ask_ratio'] = obs_bid / (obs_bid + obs_ask) if (obs_bid + obs_ask) > 0 else 0.5
            else:
                intra['bid_ask_ratio'] = 0.5

            meta = {
                'trading_day': td,
                'date': date_ts,
                'checkpoint_bar': cp_bar,
                'prev_close': pc,
                'full_day_high_pct': full_day_high_pct,
                'full_day_low_pct': full_day_low_pct,
                'obs_high': session_high,
                'obs_low': session_low,
                'real_high': day_high,
                'real_low': day_low,
            }

            rows.append({**pre, **intra, **meta})

    return pd.DataFrame(rows)


# ── Training ──────────────────────────────────────────────────────────

class ConditionalRangeTrainer:
    """Walk-forward trainer for intraday conditional range models."""

    def __init__(self, config: Optional[ConditionalRangeConfig] = None):
        self.config = config or ConditionalRangeConfig()
        self.model_high: Optional[XGBRegressor] = None
        self.model_low: Optional[XGBRegressor] = None
        self.feature_names: List[str] = list(FEATURE_COLS)
        self.checkpoint_df: Optional[pd.DataFrame] = None

    def train(
        self,
        df_1min: pd.DataFrame,
        daily: pd.DataFrame,
        verbose: bool = True,
    ) -> None:
        """Train final models on all available data.

        Args:
            df_1min: 1-min OHLCV with trading_day, dt, ovn columns.
            daily: Daily OHLCV DataFrame.
            verbose: Print progress.
        """
        if verbose:
            print("Building checkpoint dataset...")
        self.checkpoint_df = build_checkpoint_dataset(df_1min, daily, self.config)
        self.checkpoint_df[self.feature_names] = self.checkpoint_df[self.feature_names].fillna(0.0)

        n_days = self.checkpoint_df['trading_day'].nunique()
        if verbose:
            print(f"  {len(self.checkpoint_df):,} rows ({n_days} days)")

        X = self.checkpoint_df[self.feature_names].values
        y_high = self.checkpoint_df['full_day_high_pct'].values
        y_low = self.checkpoint_df['full_day_low_pct'].values

        if verbose:
            print("Training high model...")
        self.model_high = XGBRegressor(**self.config.xgb_params)
        self.model_high.fit(X, y_high)

        if verbose:
            print("Training low model...")
        self.model_low = XGBRegressor(**self.config.xgb_params)
        self.model_low.fit(X, y_low)

        if verbose:
            print("Done.")

    def walk_forward_evaluate(
        self,
        df_1min: pd.DataFrame,
        daily: pd.DataFrame,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Walk-forward OOS evaluation, splitting by trading day.

        Returns the checkpoint DataFrame with OOS prediction columns added.
        Only OOS rows are returned (in-sample rows dropped).
        """
        if verbose:
            print("Building checkpoint dataset...")
        cp_df = build_checkpoint_dataset(df_1min, daily, self.config)
        cp_df[self.feature_names] = cp_df[self.feature_names].fillna(0.0)

        n_days_total = cp_df['trading_day'].nunique()
        if verbose:
            print(f"  {len(cp_df):,} rows ({n_days_total} days)")

        unique_days = sorted(cp_df['trading_day'].unique())
        n_days = len(unique_days)
        test_per_fold = (n_days - self.config.min_train_days) // self.config.n_folds

        pred_high = np.full(len(cp_df), np.nan)
        pred_low = np.full(len(cp_df), np.nan)

        for fold in range(self.config.n_folds):
            tr_end = self.config.min_train_days + fold * test_per_fold
            te_end = tr_end + test_per_fold if fold < self.config.n_folds - 1 else n_days

            train_days = set(unique_days[:tr_end])
            test_days = set(unique_days[tr_end:te_end])

            train_mask = cp_df['trading_day'].isin(train_days).values
            test_mask = cp_df['trading_day'].isin(test_days).values

            X_tr = cp_df.loc[train_mask, self.feature_names].values
            X_te = cp_df.loc[test_mask, self.feature_names].values

            if verbose:
                print(f"  Fold {fold}: train={train_mask.sum()} rows "
                      f"({len(train_days)} days), test={test_mask.sum()} rows "
                      f"({len(test_days)} days)")

            m_h = XGBRegressor(**self.config.xgb_params)
            m_h.fit(X_tr, cp_df.loc[train_mask, 'full_day_high_pct'].values)
            pred_high[test_mask] = m_h.predict(X_te)

            m_l = XGBRegressor(**self.config.xgb_params)
            m_l.fit(X_tr, cp_df.loc[train_mask, 'full_day_low_pct'].values)
            pred_low[test_mask] = m_l.predict(X_te)

        cp_df['pred_high_pct'] = pred_high
        cp_df['pred_low_pct'] = pred_low

        # Convert to price levels
        pc = cp_df['prev_close'].values
        cp_df['pred_high'] = pc * (1 + pred_high)
        cp_df['pred_low'] = pc * (1 - pred_low)

        # Clamp: predictions can't be inside the already-observed range
        cp_df['pred_high_clamped'] = np.maximum(cp_df['pred_high'], cp_df['obs_high'])
        cp_df['pred_low_clamped'] = np.minimum(cp_df['pred_low'], cp_df['obs_low'])

        # Drop in-sample rows
        oos = cp_df.dropna(subset=['pred_high_pct', 'pred_low_pct']).copy()
        if verbose:
            print(f"  OOS: {len(oos):,} rows ({oos['trading_day'].nunique()} days)")
        return oos

    def save(self, model_dir: str) -> None:
        """Save trained models and metadata to disk.

        Args:
            model_dir: Directory to save to.
        """
        if self.model_high is None or self.model_low is None:
            raise ValueError("No trained models. Call train() first.")

        os.makedirs(model_dir, exist_ok=True)
        self.model_high.save_model(os.path.join(model_dir, 'conditional_high.json'))
        self.model_low.save_model(os.path.join(model_dir, 'conditional_low.json'))

        metadata = {
            'feature_names': self.feature_names,
            'checkpoint_interval': self.config.checkpoint_interval,
            'config': {
                'n_folds': self.config.n_folds,
                'min_train_days': self.config.min_train_days,
                'min_rth_bars': self.config.min_rth_bars,
                'xgb_params': self.config.xgb_params,
            },
        }
        with open(os.path.join(model_dir, 'conditional_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved conditional models to {model_dir}")


# ── Inference ─────────────────────────────────────────────────────────

class ConditionalRangePredictor:
    """Load trained conditional models and predict intraday ranges.

    Usage:
        predictor = ConditionalRangePredictor.load('models/range_predictor/conditional')
        result = predictor.predict(intraday_state)
    """

    def __init__(self):
        self.model_high: Optional[XGBRegressor] = None
        self.model_low: Optional[XGBRegressor] = None
        self.feature_names: List[str] = list(FEATURE_COLS)

    @classmethod
    def load(cls, model_dir: str) -> 'ConditionalRangePredictor':
        """Load models from disk.

        Args:
            model_dir: Directory containing saved models.

        Returns:
            Loaded ConditionalRangePredictor.
        """
        pred = cls()

        meta_path = os.path.join(model_dir, 'conditional_metadata.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No conditional_metadata.json in {model_dir}")

        with open(meta_path) as f:
            metadata = json.load(f)

        pred.feature_names = metadata['feature_names']

        pred.model_high = XGBRegressor()
        pred.model_high.load_model(os.path.join(model_dir, 'conditional_high.json'))

        pred.model_low = XGBRegressor()
        pred.model_low.load_model(os.path.join(model_dir, 'conditional_low.json'))

        return pred

    def predict(self, state: IntradayState) -> Dict[str, float]:
        """Predict full-day range given current intraday state.

        The prediction is clamped to the already-observed session range:
        pred_high >= session_high, pred_low <= session_low.

        Args:
            state: Current intraday snapshot.

        Returns:
            Dict with pred_high, pred_low, pred_width, and pct values.
        """
        if self.model_high is None or self.model_low is None:
            raise ValueError("No models loaded. Call load() first.")

        features = state.to_features()
        X = np.array([[features[f] for f in self.feature_names]])

        high_pct = float(self.model_high.predict(X)[0])
        low_pct = float(self.model_low.predict(X)[0])

        pc = state.prev_close
        pred_high = pc * (1 + high_pct)
        pred_low = pc * (1 - low_pct)

        # Clamp to observed range
        pred_high = max(pred_high, state.session_high)
        pred_low = min(pred_low, state.session_low)

        return {
            'pred_high': round(pred_high, 2),
            'pred_low': round(pred_low, 2),
            'pred_width': round(pred_high - pred_low, 2),
            'pred_high_pct': high_pct,
            'pred_low_pct': low_pct,
            'elapsed_frac': state.elapsed_bars / state.total_rth_bars if state.total_rth_bars > 0 else 0,
        }

    def predict_from_bars(
        self,
        rth_bars: pd.DataFrame,
        prev_close: float,
        prev_high: float,
        prev_low: float,
        prev_prev_close: float,
        atr_14: float,
        atr_60: float,
        dow: int,
        gamma: float = 0.0,
        total_rth_bars: int = 450,
        avg_daily_volume: float = 1.0,
    ) -> Dict[str, float]:
        """Convenience method: predict from a DataFrame of RTH bars so far.

        Args:
            rth_bars: DataFrame of 1-min RTH bars observed so far (with
                      high, low, close, volume, and optionally bidvolume/askvolume).
            prev_close: Previous session close.
            prev_high: Previous session high.
            prev_low: Previous session low.
            prev_prev_close: Close from two sessions ago.
            atr_14: 14-day ATR (lagged).
            atr_60: 60-day ATR (lagged).
            dow: Day of week (0=Mon).
            gamma: Gamma score (0 if unavailable).
            total_rth_bars: Expected total RTH bars.
            avg_daily_volume: Average daily RTH volume for normalization.

        Returns:
            Dict with pred_high, pred_low, etc.
        """
        n = len(rth_bars)
        lookback = min(30, n)
        recent = rth_bars.iloc[-lookback:]

        session_high = rth_bars['high'].max()
        session_low = rth_bars['low'].min()

        prev_session_high = rth_bars['high'].iloc[:-lookback].max() if n > lookback else -np.inf
        prev_session_low = rth_bars['low'].iloc[:-lookback].min() if n > lookback else np.inf

        bid_vol = rth_bars['bidvolume'].sum() if 'bidvolume' in rth_bars.columns else 0
        ask_vol = rth_bars['askvolume'].sum() if 'askvolume' in rth_bars.columns else 0

        state = IntradayState(
            prev_close=prev_close,
            prev_high=prev_high,
            prev_low=prev_low,
            prev_prev_close=prev_prev_close,
            atr_14=atr_14,
            atr_60=atr_60,
            dow=dow,
            gamma=gamma,
            elapsed_bars=n,
            total_rth_bars=total_rth_bars,
            session_high=session_high,
            session_low=session_low,
            session_open=rth_bars['close'].iloc[0],
            current_close=rth_bars['close'].iloc[-1],
            cumulative_volume=rth_bars['volume'].sum(),
            avg_daily_volume=avg_daily_volume,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            recent_high=recent['high'].max(),
            recent_low=recent['low'].min(),
            new_session_high_recent=session_high > prev_session_high if n > lookback else True,
            new_session_low_recent=session_low < prev_session_low if n > lookback else True,
        )

        return self.predict(state)
