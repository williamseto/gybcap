"""Feature engineering for ES range prediction.

Computes features from daily OHLCV bars for predicting high/low range boundaries.
All features are strictly causal (no look-ahead).

Reuses proven feature functions from:
- strategies/swing/features/range_features.py (multi-timeframe range)
- strategies/swing/features/daily_technical.py (RSI, BB, ATR, drawdowns)

Adds range-prediction-specific features:
- Seasonality (cyclical day-of-week, month, OpEx, quarter-end)
- VIX/implied vol proxy (via ATR ratios)
- Autoregressive range targets (lagged realized ranges)
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from strategies.range_predictor.config import TIMEFRAME_HORIZONS


def aggregate_to_daily(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars to daily OHLCV.

    Args:
        df_1min: 1-min DataFrame with columns [open, high, low, close, volume, trading_day]

    Returns:
        Daily DataFrame indexed by trading_day date with OHLCV columns.
    """
    # Map integer trading_day to actual dates
    day_map = df_1min.groupby('trading_day')['dt'].first().dt.date
    daily = df_1min.groupby('trading_day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    daily.index = daily.index.map(day_map)
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.sort_index()

    # Carry forward gamma if available
    if 'nearby_gamma_score' in df_1min.columns:
        gamma_daily = df_1min.groupby('trading_day')['nearby_gamma_score'].mean()
        gamma_daily.index = gamma_daily.index.map(day_map)
        gamma_daily.index = pd.DatetimeIndex(gamma_daily.index)
        daily['nearby_gamma_score'] = gamma_daily

    # Combine duplicate dates (split sessions around holidays/DST)
    if daily.index.duplicated().any():
        daily = daily.groupby(daily.index).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            **({'nearby_gamma_score': 'mean'} if 'nearby_gamma_score' in daily.columns else {}),
        })

    return daily


def compute_targets(
    daily: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Compute range prediction targets.

    For a given horizon (e.g. 1 day, 5 days), compute:
    - range_high_pct: (max high over horizon - prev close) / prev close
    - range_low_pct:  (prev close - min low over horizon) / prev close

    These are the upside/downside moves from close, always positive.

    Args:
        daily: Daily OHLCV DataFrame (DatetimeIndex).
        horizon: Number of trading days to look ahead.

    Returns:
        DataFrame with target columns, aligned to daily index.
    """
    prev_close = daily['close'].shift(1)

    # Forward-looking realized highs/lows over the horizon
    # Use .shift(-i) to look ahead, then take max/min
    future_highs = pd.concat(
        [daily['high'].shift(-i) for i in range(horizon)], axis=1
    ).max(axis=1)
    future_lows = pd.concat(
        [daily['low'].shift(-i) for i in range(horizon)], axis=1
    ).min(axis=1)

    targets = pd.DataFrame(index=daily.index)
    targets['range_high_pct'] = (future_highs - prev_close) / prev_close
    targets['range_low_pct'] = (prev_close - future_lows) / prev_close

    return targets


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean().fillna(tr)


def compute_range_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for range prediction from daily bars.

    Returns DataFrame with ~60 features, same index as input.
    """
    c = daily['close']
    h = daily['high']
    l = daily['low']
    o = daily['open']
    v = daily['volume']
    feat = pd.DataFrame(index=daily.index)

    # ── Volatility ──────────────────────────────────────────────────────
    atr_5 = _compute_atr(h, l, c, 5)
    atr_14 = _compute_atr(h, l, c, 14)
    atr_20 = _compute_atr(h, l, c, 20)

    feat['atr_5'] = atr_5
    feat['atr_14'] = atr_14
    feat['atr_ratio_5_20'] = atr_5 / atr_20.replace(0, np.nan)

    log_ret = np.log(c / c.shift(1))
    feat['rv_10d'] = log_ret.rolling(10).std() * np.sqrt(252)
    feat['rv_21d'] = log_ret.rolling(21).std() * np.sqrt(252)
    feat['vol_of_vol'] = feat['rv_10d'].rolling(20).std()

    # ATR as pct of close (vol proxy comparable across price levels)
    feat['atr_pct_14'] = atr_14 / c

    # ── Autoregressive range features ───────────────────────────────────
    day_range_pct = (h - l) / c
    feat['range_pct_lag1'] = day_range_pct.shift(1)
    feat['range_pct_lag2'] = day_range_pct.shift(2)
    feat['range_pct_lag3'] = day_range_pct.shift(3)
    feat['range_pct_lag5'] = day_range_pct.shift(5)
    feat['range_pct_ma5'] = day_range_pct.rolling(5).mean().shift(1)
    feat['range_pct_ma20'] = day_range_pct.rolling(20).mean().shift(1)
    feat['range_pct_ma60'] = day_range_pct.rolling(60).mean().shift(1)

    # Realized high/low pct (how far from prev close)
    high_move_pct = (h - c.shift(1)) / c.shift(1)
    low_move_pct = (c.shift(1) - l) / c.shift(1)
    feat['high_move_pct_lag1'] = high_move_pct.shift(1)
    feat['low_move_pct_lag1'] = low_move_pct.shift(1)
    feat['high_move_pct_ma5'] = high_move_pct.rolling(5).mean().shift(1)
    feat['low_move_pct_ma5'] = low_move_pct.rolling(5).mean().shift(1)

    # ── Returns / momentum ──────────────────────────────────────────────
    feat['return_1d'] = c.pct_change()
    feat['return_5d'] = c.pct_change(5)
    feat['return_20d'] = c.pct_change(20)

    feat['rsi_5'] = _compute_rsi(c, 5)
    feat['rsi_14'] = _compute_rsi(c, 14)

    # ── Trend ───────────────────────────────────────────────────────────
    for w in [20, 50, 100, 200]:
        sma = c.rolling(w, min_periods=1).mean()
        std = c.rolling(w, min_periods=1).std().replace(0, np.nan)
        feat[f'close_vs_sma{w}_z'] = (c - sma) / std
        feat[f'sma{w}_slope_5d'] = sma.pct_change(5)

    sma20 = c.rolling(20, min_periods=1).mean()
    sma50 = c.rolling(50, min_periods=1).mean()
    sma100 = c.rolling(100, min_periods=1).mean()
    sma200 = c.rolling(200, min_periods=1).mean()

    # MA alignment: +1 if 20>50>200, -1 if reversed
    alignment = (
        (sma20 > sma50).astype(float) + (sma50 > sma200).astype(float)
    ) / 2.0
    feat['ma_alignment'] = alignment * 2 - 1

    # Trend strength: absolute z-score of close vs SMA200
    std200 = c.rolling(200, min_periods=1).std().replace(0, np.nan)
    feat['trend_strength'] = ((c - sma200) / std200).abs()

    # SMA100 vs SMA200 spread (intermediate trend regime)
    feat['sma100_vs_sma200'] = (sma100 - sma200) / sma200.replace(0, np.nan)

    # ── Weekly MAs (resample daily close to weekly, map back) ──────────
    weekly_close = c.resample('W-FRI').last().dropna()
    wma20 = weekly_close.rolling(20, min_periods=1).mean()
    wma50 = weekly_close.rolling(50, min_periods=1).mean()
    # Forward-fill weekly values to daily index
    wma20_daily = wma20.reindex(c.index, method='ffill')
    wma50_daily = wma50.reindex(c.index, method='ffill')
    feat['close_vs_wma20_z'] = (c - wma20_daily) / wma20_daily.replace(0, np.nan)
    feat['close_vs_wma50_z'] = (c - wma50_daily) / wma50_daily.replace(0, np.nan)
    feat['wma20_slope_4w'] = wma20.pct_change(4).reindex(c.index, method='ffill')
    feat['wma50_slope_4w'] = wma50.pct_change(4).reindex(c.index, method='ffill')

    # ── Drawdown ────────────────────────────────────────────────────────
    rolling_high_20 = c.rolling(20, min_periods=1).max()
    rolling_high_60 = c.rolling(60, min_periods=1).max()
    feat['drawdown_20d'] = (c - rolling_high_20) / rolling_high_20
    feat['drawdown_60d'] = (c - rolling_high_60) / rolling_high_60

    # ── Bollinger Bands ─────────────────────────────────────────────────
    std20 = c.rolling(20, min_periods=1).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_bandwidth = bb_upper - bb_lower

    feat['bb_pct_b'] = (c - bb_lower) / bb_bandwidth.replace(0, np.nan)
    feat['bb_width'] = bb_bandwidth / sma20.replace(0, np.nan)
    bb_width_mean = feat['bb_width'].rolling(60, min_periods=10).mean()
    bb_width_std = feat['bb_width'].rolling(60, min_periods=10).std().replace(0, np.nan)
    feat['bb_width_z'] = (feat['bb_width'] - bb_width_mean) / bb_width_std

    # ── Volume ──────────────────────────────────────────────────────────
    vol_mean = v.rolling(20, min_periods=1).mean()
    vol_std = v.rolling(20, min_periods=1).std().replace(0, np.nan)
    feat['volume_z'] = (v - vol_mean) / vol_std
    feat['volume_ma_ratio'] = v / vol_mean.replace(0, np.nan)

    # ── Gap ──────────────────────────────────────────────────────────────
    feat['gap_pct'] = (o - c.shift(1)) / c.shift(1)

    # ── Range position (multi-timeframe) ────────────────────────────────
    for tf_name, tf_days in [('5d', 5), ('20d', 20), ('63d', 63)]:
        roll_high = c.rolling(tf_days, min_periods=tf_days).max()
        roll_low = c.rolling(tf_days, min_periods=tf_days).min()
        roll_range = roll_high - roll_low

        feat[f'range_pos_{tf_name}'] = (
            (c - roll_low) / roll_range.replace(0, np.nan)
        )
        feat[f'range_width_{tf_name}'] = roll_range / c

        # Compression
        baseline_window = min(tf_days * 3, len(c))
        avg_width = (roll_range / c).rolling(
            baseline_window, min_periods=tf_days
        ).mean()
        feat[f'range_compression_{tf_name}'] = (
            (roll_range / c) / avg_width.replace(0, np.nan)
        )

    # ── Seasonality (cyclical encoding) ─────────────────────────────────
    # Day-of-week
    dow = daily.index.dayofweek
    feat['dow_sin'] = np.sin(2 * np.pi * dow / 5)
    feat['dow_cos'] = np.cos(2 * np.pi * dow / 5)

    # Day-of-month
    dom = daily.index.day
    feat['dom_sin'] = np.sin(2 * np.pi * dom / 31)
    feat['dom_cos'] = np.cos(2 * np.pi * dom / 31)

    # Month
    month = daily.index.month
    feat['month_sin'] = np.sin(2 * np.pi * month / 12)
    feat['month_cos'] = np.cos(2 * np.pi * month / 12)

    # OpEx week (3rd Friday falls on days 15-21)
    feat['is_opex_week'] = ((dom >= 15) & (dom <= 21)).astype(float)

    # Quarter-end
    feat['is_quarter_end'] = (
        (month.isin([3, 6, 9, 12])) & (dom >= 25)
    ).astype(float)

    # ── GEX (if available) ──────────────────────────────────────────────
    if 'nearby_gamma_score' in daily.columns:
        feat['nearby_gamma_score'] = daily['nearby_gamma_score']

    # Causality guard:
    # Targets for date t represent the forward range from t onward (anchored to
    # prev close). Features must therefore only use information known by the
    # start of date t. Shift all engineered daily features by 1 day so feature
    # row t is computed from data through t-1.
    return feat.shift(1).fillna(0.0)


# Feature name list for reference
FEATURE_NAMES = [
    # Volatility
    'atr_5', 'atr_14', 'atr_ratio_5_20',
    'rv_10d', 'rv_21d', 'vol_of_vol', 'atr_pct_14',
    # Autoregressive
    'range_pct_lag1', 'range_pct_lag2', 'range_pct_lag3', 'range_pct_lag5',
    'range_pct_ma5', 'range_pct_ma20', 'range_pct_ma60',
    'high_move_pct_lag1', 'low_move_pct_lag1',
    'high_move_pct_ma5', 'low_move_pct_ma5',
    # Returns / momentum
    'return_1d', 'return_5d', 'return_20d',
    'rsi_5', 'rsi_14',
    # Trend (daily)
    'close_vs_sma20_z', 'close_vs_sma50_z',
    'close_vs_sma100_z', 'close_vs_sma200_z',
    'sma20_slope_5d', 'sma50_slope_5d',
    'sma100_slope_5d', 'sma200_slope_5d',
    'ma_alignment', 'trend_strength', 'sma100_vs_sma200',
    # Trend (weekly)
    'close_vs_wma20_z', 'close_vs_wma50_z',
    'wma20_slope_4w', 'wma50_slope_4w',
    # Drawdown
    'drawdown_20d', 'drawdown_60d',
    # Bollinger
    'bb_pct_b', 'bb_width', 'bb_width_z',
    # Volume
    'volume_z', 'volume_ma_ratio',
    # Gap
    'gap_pct',
    # Multi-timeframe range
    'range_pos_5d', 'range_width_5d', 'range_compression_5d',
    'range_pos_20d', 'range_width_20d', 'range_compression_20d',
    'range_pos_63d', 'range_width_63d', 'range_compression_63d',
    # Seasonality
    'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
    'month_sin', 'month_cos',
    'is_opex_week', 'is_quarter_end',
    # GEX (optional)
    'nearby_gamma_score',
]


RTH_EXTRA_FEATURES = ['ovn_gap_pct', 'ovn_gap_abs_pct']


def get_feature_names(include_gamma: bool = True) -> List[str]:
    """Return the list of feature names used by the range predictor."""
    names = [n for n in FEATURE_NAMES if n != 'nearby_gamma_score']
    if include_gamma:
        names.append('nearby_gamma_score')
    return names


def get_rth_feature_names(include_gamma: bool = True) -> List[str]:
    """Return the list of feature names for the RTH range predictor.

    Same as get_feature_names() plus overnight gap features.
    """
    return get_feature_names(include_gamma) + list(RTH_EXTRA_FEATURES)


def aggregate_to_rth_daily(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars to RTH-only daily OHLCV.

    Filters to ovn==0 (RTH session) bars, then groups by trading_day.

    Args:
        df_1min: 1-min DataFrame with columns [open, high, low, close, volume, trading_day, ovn]

    Returns:
        Daily DataFrame indexed by trading_day date with rth_open, rth_high,
        rth_low, rth_close, rth_volume columns.
    """
    rth = df_1min[df_1min['ovn'] == 0].copy()

    day_map = rth.groupby('trading_day')['dt'].first().dt.date
    daily = rth.groupby('trading_day').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    daily.index = daily.index.map(day_map)
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.sort_index()

    # Rename to rth_ prefix
    daily = daily.rename(columns={
        'open': 'rth_open',
        'high': 'rth_high',
        'low': 'rth_low',
        'close': 'rth_close',
        'volume': 'rth_volume',
    })

    # Combine duplicate dates (split sessions around holidays/DST)
    if daily.index.duplicated().any():
        daily = daily.groupby(daily.index).agg({
            'rth_open': 'first',
            'rth_high': 'max',
            'rth_low': 'min',
            'rth_close': 'last',
            'rth_volume': 'sum',
        })

    return daily


def compute_rth_targets(rth_daily: pd.DataFrame) -> pd.DataFrame:
    """Compute RTH range prediction targets relative to RTH open.

    Args:
        rth_daily: RTH daily DataFrame with rth_open, rth_high, rth_low columns.

    Returns:
        DataFrame with rth_range_high_pct and rth_range_low_pct.
    """
    targets = pd.DataFrame(index=rth_daily.index)
    targets['rth_range_high_pct'] = (
        (rth_daily['rth_high'] - rth_daily['rth_open']) / rth_daily['rth_open']
    )
    targets['rth_range_low_pct'] = (
        (rth_daily['rth_open'] - rth_daily['rth_low']) / rth_daily['rth_open']
    )
    return targets


def compute_rth_gap_features(
    full_daily: pd.DataFrame,
    rth_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute overnight gap features from full-session and RTH daily data.

    Args:
        full_daily: Full-session daily OHLCV (DatetimeIndex).
        rth_daily: RTH daily with rth_open, rth_close columns (DatetimeIndex).

    Returns:
        DataFrame with ovn_gap_pct and ovn_gap_abs_pct, aligned to rth_daily index.
    """
    # Gap = (RTH open today - previous RTH close) / previous RTH close
    prev_rth_close = rth_daily['rth_close'].shift(1)
    gap = pd.DataFrame(index=rth_daily.index)
    gap['ovn_gap_pct'] = (
        (rth_daily['rth_open'] - prev_rth_close) / prev_rth_close
    )
    gap['ovn_gap_abs_pct'] = gap['ovn_gap_pct'].abs()
    return gap


def prepare_rth_dataset(
    full_daily: pd.DataFrame,
    rth_daily: pd.DataFrame,
    include_gamma: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare features + targets for RTH model training.

    Uses full-session daily features (available before RTH open) plus overnight
    gap features. Targets are RTH range relative to RTH open.

    Args:
        full_daily: Full-session daily OHLCV DataFrame.
        rth_daily: RTH daily DataFrame with rth_open/high/low/close.
        include_gamma: Whether to include gamma score feature.

    Returns:
        (features_df, targets_df, feature_names) — aligned, NaN rows dropped.
    """
    # Base features from full-session daily
    features = compute_range_features(full_daily)

    # Gap features
    gap_features = compute_rth_gap_features(full_daily, rth_daily)
    features = features.join(gap_features, how='left')

    # RTH targets
    targets = compute_rth_targets(rth_daily)

    # Feature names
    feature_names = get_rth_feature_names(
        include_gamma='nearby_gamma_score' in features.columns
    )
    feature_names = [f for f in feature_names if f in features.columns]

    # Align on common dates and drop NaN target rows
    common_idx = features.index.intersection(targets.index)
    combined = features.loc[common_idx, feature_names].join(targets.loc[common_idx])
    combined = combined.dropna(subset=['rth_range_high_pct', 'rth_range_low_pct'])
    combined = combined.fillna(0.0)

    return (
        combined[feature_names],
        combined[['rth_range_high_pct', 'rth_range_low_pct']],
        feature_names,
    )


def prepare_dataset(
    daily: pd.DataFrame,
    timeframe: str = 'daily',
    include_gamma: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Prepare features + targets for training.

    Args:
        daily: Daily OHLCV DataFrame.
        timeframe: One of 'daily', 'weekly', 'monthly', 'quarterly'.
        include_gamma: Whether to include gamma score feature.

    Returns:
        (features_df, targets_df, feature_names) — aligned, NaN rows dropped.
    """
    horizon = TIMEFRAME_HORIZONS[timeframe]
    features = compute_range_features(daily)
    targets = compute_targets(daily, horizon)

    feature_names = get_feature_names(
        include_gamma='nearby_gamma_score' in features.columns
    )
    # Filter to actually available features
    feature_names = [f for f in feature_names if f in features.columns]

    # Combine and drop rows where targets are NaN (end of series)
    combined = features[feature_names].join(targets)
    combined = combined.dropna(subset=['range_high_pct', 'range_low_pct'])
    combined = combined.fillna(0.0)

    return (
        combined[feature_names],
        combined[['range_high_pct', 'range_low_pct']],
        feature_names,
    )
