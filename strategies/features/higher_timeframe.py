"""
Higher timeframe feature provider.

Computes daily and weekly context features from 1-min data that help
contextualize intraday reversals with macro market conditions.

Features include:
- Daily trend & momentum (RSI, MA deviations, ATR)
- Bollinger Band proximity (daily and intraday)
- Prior day profile (POC, VA, range)
- Gap analysis (magnitude, fill status)
- Weekly context (RSI, MA)
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from strategies.features.base import BaseFeatureProvider
from strategies.features.registry import FeatureRegistry


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI indicator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(span=period, min_periods=period).mean()
    avg_loss = loss.ewm(span=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """Compute Bollinger Bands and derived metrics."""
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    # %B indicator: (price - lower) / (upper - lower)
    bandwidth = upper - lower
    pct_b = (series - lower) / bandwidth.replace(0, np.nan)

    # Band width as % of SMA
    bb_width = bandwidth / sma.replace(0, np.nan)

    return {
        'sma': sma,
        'upper': upper,
        'lower': lower,
        'pct_b': pct_b.fillna(0.5),
        'width': bb_width.fillna(0)
    }


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, min_periods=period).mean().fillna(tr)


@FeatureRegistry.register('higher_timeframe')
class HigherTimeframeProvider(BaseFeatureProvider):
    """
    Computes higher timeframe context features from 1-min data.

    These features provide macro context for intraday trading decisions:
    - Is the market trending or ranging on daily/weekly?
    - Is price at Bollinger Band extremes?
    - What was prior day's profile type?
    - What's the gap situation today?
    """

    # Daily trend & momentum features
    DAILY_TREND_FEATURES = [
        'daily_rsi_14',           # RSI on daily closes
        'daily_close_vs_sma20',   # Close relative to 20-day MA (z-score)
        'daily_close_vs_sma50',   # Close relative to 50-day MA (z-score)
        'daily_atr_14',           # Average True Range
        'daily_trend_5d',         # 5-day return
        'daily_trend_20d',        # 20-day return
    ]

    # Daily Bollinger Band features
    DAILY_BB_FEATURES = [
        'daily_bb_pct_b',         # %B indicator (0=lower, 1=upper)
        'daily_bb_upper_dist',    # Distance to upper band (ATR units)
        'daily_bb_lower_dist',    # Distance to lower band (ATR units)
        'daily_bb_width',         # Band width (volatility)
    ]

    # Prior day profile features
    PRIOR_DAY_FEATURES = [
        'prior_day_range',        # Prior day's high-low range
        'prior_day_range_pct',    # Range as % of price
        'prior_poc_rel',          # POC position in prior day range (0-1)
        'prior_va_width_rel',     # Value Area width as % of range
    ]

    # Gap analysis features
    GAP_FEATURES = [
        'gap_pct',                # (open - prior close) / prior close
        'gap_vs_prior_range',     # Gap size relative to prior day range
        'open_vs_prior_va',       # Open position: -1=below VA, 0=in VA, 1=above VA
        'gap_filled',             # Has gap been filled? (binary, updated intraday)
    ]

    # Weekly features
    WEEKLY_FEATURES = [
        'weekly_rsi_14',          # Weekly RSI
        'weekly_close_vs_sma10',  # Close vs 10-week MA
        'weekly_bb_pct_b',        # Weekly %B
        'weekly_bb_width',        # Weekly band width
    ]

    # Intraday Bollinger Bands (rolling)
    INTRADAY_BB_FEATURES = [
        'intraday_bb_pct_b',      # %B on intraday data
        'intraday_bb_upper_dist', # Distance to upper band (z-score)
        'intraday_bb_lower_dist', # Distance to lower band (z-score)
        'intraday_bb_width_z',    # Band width vs recent average
    ]

    # Range exhaustion features
    RANGE_EXHAUSTION_FEATURES = [
        'range_pct_of_atr',          # Today's running range / ATR_14
        'range_pct_of_prior_range',  # Today's range / prior day range
        'distance_from_open_atr',    # |close - open| / ATR_14
        'session_range_percentile',  # Today's range as percentile of last 20 sessions
        'remaining_atr_pct',         # (ATR - today's range) / ATR
        'prior_va_width_actual',     # Prior day VA width (from OHLC approximation)
    ]

    # Directional context features
    DIRECTIONAL_CONTEXT_FEATURES = [
        'bb_side_alignment',         # +1 at lower BB + support, -1 at upper BB + resistance
        'trend_exhaustion_score',    # RSI × BB %B interaction
        'weekly_trend_alignment',    # +1 if weekly trend supports reversal
        'approach_velocity_10bar',   # Normalized approach speed
        'approach_deceleration',     # Velocity change (slowing = reversal hint?)
        'bars_since_session_extreme', # Bars since today's high/low set
    ]

    def __init__(
        self,
        daily_sma_short: int = 20,
        daily_sma_long: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        intraday_bb_period: int = 20,  # 20 bars for intraday BB
    ):
        """
        Initialize provider.

        Args:
            daily_sma_short: Short SMA period for daily data
            daily_sma_long: Long SMA period for daily data
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
            bb_period: Bollinger Band period
            bb_std: Number of standard deviations for BB
            intraday_bb_period: Period for intraday Bollinger Bands
        """
        super().__init__()
        self.daily_sma_short = daily_sma_short
        self.daily_sma_long = daily_sma_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.intraday_bb_period = intraday_bb_period

        # Cache for daily aggregates
        self._daily_cache: Optional[pd.DataFrame] = None
        self._weekly_cache: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "higher_timeframe"

    @property
    def feature_names(self) -> List[str]:
        return (
            self.DAILY_TREND_FEATURES +
            self.DAILY_BB_FEATURES +
            self.PRIOR_DAY_FEATURES +
            self.GAP_FEATURES +
            self.WEEKLY_FEATURES +
            self.INTRADAY_BB_FEATURES +
            self.RANGE_EXHAUSTION_FEATURES +
            self.DIRECTIONAL_CONTEXT_FEATURES
        )

    def _aggregate_to_daily(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Aggregate 1-min data to daily OHLCV."""
        daily = ohlcv.groupby('trading_day').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        daily = daily.sort_values('trading_day').reset_index(drop=True)
        return daily

    def _aggregate_to_weekly(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Aggregate daily data to weekly OHLCV."""
        # Convert trading_day to datetime for week grouping
        daily = daily.copy()
        daily['week'] = pd.to_datetime(daily['trading_day']).dt.isocalendar().week
        daily['year'] = pd.to_datetime(daily['trading_day']).dt.isocalendar().year
        daily['year_week'] = daily['year'].astype(str) + '_' + daily['week'].astype(str).str.zfill(2)

        weekly = daily.groupby('year_week').agg({
            'trading_day': 'last',  # Use last trading day of week
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        weekly = weekly.sort_values('trading_day').reset_index(drop=True)
        return weekly

    def _compute_daily_features(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Compute all daily-level features."""
        df = daily.copy()

        # RSI
        df['daily_rsi_14'] = _compute_rsi(df['close'], self.rsi_period)

        # MA deviations (z-score)
        sma20 = df['close'].rolling(window=self.daily_sma_short, min_periods=1).mean()
        sma50 = df['close'].rolling(window=self.daily_sma_long, min_periods=1).mean()
        std20 = df['close'].rolling(window=self.daily_sma_short, min_periods=1).std()
        std50 = df['close'].rolling(window=self.daily_sma_long, min_periods=1).std()

        df['daily_close_vs_sma20'] = (df['close'] - sma20) / std20.replace(0, np.nan)
        df['daily_close_vs_sma50'] = (df['close'] - sma50) / std50.replace(0, np.nan)
        df['daily_close_vs_sma20'] = df['daily_close_vs_sma20'].fillna(0)
        df['daily_close_vs_sma50'] = df['daily_close_vs_sma50'].fillna(0)

        # ATR
        df['daily_atr_14'] = _compute_atr(df['high'], df['low'], df['close'], self.atr_period)

        # Trend (returns)
        df['daily_trend_5d'] = df['close'].pct_change(5).fillna(0)
        df['daily_trend_20d'] = df['close'].pct_change(20).fillna(0)

        # Bollinger Bands
        bb = _compute_bollinger_bands(df['close'], self.bb_period, self.bb_std)
        df['daily_bb_pct_b'] = bb['pct_b']
        df['daily_bb_width'] = bb['width']

        # Distance to bands in ATR units
        atr = df['daily_atr_14'].replace(0, np.nan)
        df['daily_bb_upper_dist'] = (bb['upper'] - df['close']) / atr
        df['daily_bb_lower_dist'] = (df['close'] - bb['lower']) / atr
        df['daily_bb_upper_dist'] = df['daily_bb_upper_dist'].fillna(0)
        df['daily_bb_lower_dist'] = df['daily_bb_lower_dist'].fillna(0)

        # Prior day features
        df['prior_day_range'] = (df['high'] - df['low']).shift(1).fillna(0)
        df['prior_day_range_pct'] = (df['prior_day_range'] / df['close'].shift(1)).fillna(0)

        # POC position (simplified: use close position in range)
        day_range = df['high'] - df['low']
        poc_position = (df['close'] - df['low']) / day_range.replace(0, np.nan)
        df['prior_poc_rel'] = poc_position.shift(1).fillna(0.5)

        # VA width: approximate from IQR of close prices within the day
        # (better than hardcoded 0.7 — uses actual price distribution)
        df['prior_va_width_rel'] = 0.7  # Default, overwritten below
        for i in range(1, len(df)):
            prior_range = df.iloc[i - 1]['high'] - df.iloc[i - 1]['low']
            if prior_range > 0:
                # Approximate VA as middle 70% of range using POC-centered estimate
                poc_rel = poc_position.iloc[i - 1] if not pd.isna(poc_position.iloc[i - 1]) else 0.5
                # Wider VA when POC is centered, narrower at extremes
                va_est = 0.5 + 0.3 * (1 - abs(poc_rel - 0.5) * 2)
                df.iloc[i, df.columns.get_loc('prior_va_width_rel')] = va_est

        # Range exhaustion features (daily-level, mapped later)
        df['range_pct_of_prior_range'] = (
            (df['high'] - df['low']) / df['prior_day_range'].replace(0, np.nan)
        ).fillna(1.0)

        # Session range percentile (last 20 sessions)
        day_ranges = df['high'] - df['low']
        rolling_rank = day_ranges.rolling(window=20, min_periods=1).apply(
            lambda x: (x.iloc[-1] >= x).sum() / len(x), raw=False
        )
        df['session_range_percentile'] = rolling_rank.fillna(0.5)

        # Prior VA width actual (from VP-like approximation)
        df['prior_va_width_actual'] = df['prior_va_width_rel'] * df['prior_day_range']

        # Gap features
        prior_close = df['close'].shift(1)
        df['gap_pct'] = ((df['open'] - prior_close) / prior_close).fillna(0)
        prior_range = df['prior_day_range'].replace(0, np.nan)
        df['gap_vs_prior_range'] = (df['open'] - prior_close).abs() / prior_range
        df['gap_vs_prior_range'] = df['gap_vs_prior_range'].fillna(0)

        # Open vs prior VA (simplified: compare to prior close +/- 35% of range)
        prior_va_high = prior_close + 0.35 * df['prior_day_range']
        prior_va_low = prior_close - 0.35 * df['prior_day_range']
        df['open_vs_prior_va'] = np.where(
            df['open'] > prior_va_high, 1,
            np.where(df['open'] < prior_va_low, -1, 0)
        )

        return df

    def _compute_weekly_features(self, weekly: pd.DataFrame) -> pd.DataFrame:
        """Compute all weekly-level features."""
        df = weekly.copy()

        # Weekly RSI
        df['weekly_rsi_14'] = _compute_rsi(df['close'], self.rsi_period)

        # Weekly MA deviation
        sma10 = df['close'].rolling(window=10, min_periods=1).mean()
        std10 = df['close'].rolling(window=10, min_periods=1).std()
        df['weekly_close_vs_sma10'] = (df['close'] - sma10) / std10.replace(0, np.nan)
        df['weekly_close_vs_sma10'] = df['weekly_close_vs_sma10'].fillna(0)

        # Weekly Bollinger Bands
        bb = _compute_bollinger_bands(df['close'], self.bb_period, self.bb_std)
        df['weekly_bb_pct_b'] = bb['pct_b']
        df['weekly_bb_width'] = bb['width']

        return df

    def _compute_intraday_bb_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute intraday Bollinger Band features."""
        result = ohlcv.copy()

        # Compute BB on close prices with rolling window
        bb = _compute_bollinger_bands(
            result['close'],
            period=self.intraday_bb_period,
            num_std=self.bb_std
        )

        result['intraday_bb_pct_b'] = bb['pct_b']

        # Distance to bands (z-score relative to band width)
        bandwidth = bb['upper'] - bb['lower']
        result['intraday_bb_upper_dist'] = (bb['upper'] - result['close']) / bandwidth.replace(0, np.nan)
        result['intraday_bb_lower_dist'] = (result['close'] - bb['lower']) / bandwidth.replace(0, np.nan)
        result['intraday_bb_upper_dist'] = result['intraday_bb_upper_dist'].fillna(0)
        result['intraday_bb_lower_dist'] = result['intraday_bb_lower_dist'].fillna(0)

        # Band width z-score (squeeze detection)
        avg_width = bb['width'].rolling(window=60, min_periods=1).mean()
        std_width = bb['width'].rolling(window=60, min_periods=1).std()
        result['intraday_bb_width_z'] = (bb['width'] - avg_width) / std_width.replace(0, np.nan)
        result['intraday_bb_width_z'] = result['intraday_bb_width_z'].fillna(0)

        return result

    def _compute_gap_filled(self, ohlcv: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        """Compute running gap_filled status for each intraday bar."""
        gap_filled = pd.Series(index=ohlcv.index, dtype=float)

        # Create lookup for prior close by trading day
        prior_close_map = daily.set_index('trading_day')['close'].shift(1).to_dict()

        for day, day_df in ohlcv.groupby('trading_day'):
            prior_close = prior_close_map.get(day)
            if prior_close is None or pd.isna(prior_close):
                gap_filled.loc[day_df.index] = 0
                continue

            day_open = day_df['open'].iloc[0]
            gap = day_open - prior_close

            if abs(gap) < 0.5:  # No meaningful gap
                gap_filled.loc[day_df.index] = 0
                continue

            # Check if gap is filled at each bar
            if gap > 0:  # Gap up
                # Gap filled when low touches prior close
                filled = day_df['low'].cummin() <= prior_close
            else:  # Gap down
                # Gap filled when high touches prior close
                filled = day_df['high'].cummax() >= prior_close

            gap_filled.loc[day_df.index] = filled.astype(float)

        return gap_filled

    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute all higher timeframe features.

        Aggregates 1-min data to daily/weekly, computes indicators,
        then broadcasts back to intraday bars.
        """
        result = ohlcv.copy()

        # Note: Don't pre-initialize feature columns - let merge add them
        # (Pre-initializing causes merge to use suffixes instead of overwriting)

        # Aggregate to daily
        daily = self._aggregate_to_daily(ohlcv)
        daily = self._compute_daily_features(daily)
        self._daily_cache = daily

        # Aggregate to weekly
        weekly = self._aggregate_to_weekly(daily)
        weekly = self._compute_weekly_features(weekly)
        self._weekly_cache = weekly

        # Map daily features to intraday bars (vectorized merge)
        daily_feature_cols = (
            self.DAILY_TREND_FEATURES +
            self.DAILY_BB_FEATURES +
            self.PRIOR_DAY_FEATURES +
            ['gap_pct', 'gap_vs_prior_range', 'open_vs_prior_va',
             'range_pct_of_prior_range', 'session_range_percentile',
             'prior_va_width_actual']
        )

        daily_cols_available = [c for c in daily_feature_cols if c in daily.columns]
        if daily_cols_available:
            daily_subset = daily[['trading_day'] + daily_cols_available].copy()
            # Use merge for fast vectorized mapping
            result = result.merge(
                daily_subset,
                on='trading_day',
                how='left'
            )
            # Fill NaN with 0
            for col in daily_cols_available:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)

        # Compute gap_filled (intraday update)
        result['gap_filled'] = self._compute_gap_filled(ohlcv, daily)

        # Map weekly features to intraday bars (vectorized merge)
        # Create week lookup from daily
        daily['year_week'] = (
            pd.to_datetime(daily['trading_day']).dt.isocalendar().year.astype(str) +
            '_' +
            pd.to_datetime(daily['trading_day']).dt.isocalendar().week.astype(str).str.zfill(2)
        )

        weekly_feature_cols = self.WEEKLY_FEATURES
        weekly_cols_available = [c for c in weekly_feature_cols if c in weekly.columns]

        if weekly_cols_available:
            # Create day-to-week mapping
            day_week_map = daily[['trading_day', 'year_week']].copy()
            # Get weekly features
            weekly_subset = weekly[['year_week'] + weekly_cols_available].copy()
            # Merge day to week, then to result
            day_week_features = day_week_map.merge(weekly_subset, on='year_week', how='left')
            result = result.merge(
                day_week_features[['trading_day'] + weekly_cols_available],
                on='trading_day',
                how='left'
            )
            # Fill NaN with 0
            for col in weekly_cols_available:
                if col in result.columns:
                    result[col] = result[col].fillna(0.0)

        # Compute intraday Bollinger Band features
        intraday_bb = self._compute_intraday_bb_features(ohlcv)
        for feat in self.INTRADAY_BB_FEATURES:
            if feat in intraday_bb.columns:
                result[feat] = intraday_bb[feat].values

        # --- Range exhaustion features (intraday) ---
        result = self._compute_range_exhaustion_features(result, daily)

        # --- Directional context features (intraday) ---
        result = self._compute_directional_context_features(result, daily)

        return result

    def _compute_range_exhaustion_features(
        self, result: pd.DataFrame, daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute intraday range exhaustion features."""
        # Build ATR lookup
        atr_map = daily.set_index('trading_day')['daily_atr_14'].to_dict()
        prior_range_map = daily.set_index('trading_day')['prior_day_range'].to_dict()

        # Map daily-level features that were computed in _compute_daily_features
        daily_range_feats = ['range_pct_of_prior_range', 'session_range_percentile',
                             'prior_va_width_actual']
        for feat in daily_range_feats:
            if feat in daily.columns:
                feat_map = daily.set_index('trading_day')[feat].to_dict()
                result[feat] = result['trading_day'].map(feat_map).fillna(0.0)

        # Intraday running range / ATR (updates every bar)
        range_pct_of_atr = np.zeros(len(result), dtype=np.float64)
        distance_from_open_atr = np.zeros(len(result), dtype=np.float64)
        remaining_atr_pct = np.ones(len(result), dtype=np.float64)

        for day, day_df in result.groupby('trading_day'):
            day_mask = result['trading_day'] == day
            atr = atr_map.get(day, 0)
            if atr <= 0:
                continue

            close_vals = result.loc[day_mask, 'close'].values
            high_vals = result.loc[day_mask, 'high'].values
            low_vals = result.loc[day_mask, 'low'].values

            # Running range (cumulative high - cumulative low)
            cum_high = np.maximum.accumulate(high_vals)
            cum_low = np.minimum.accumulate(low_vals)
            running_range = cum_high - cum_low

            range_pct_of_atr[day_mask.values] = running_range / atr
            distance_from_open_atr[day_mask.values] = np.abs(close_vals - close_vals[0]) / atr
            remaining_atr_pct[day_mask.values] = np.maximum(atr - running_range, 0) / atr

        result['range_pct_of_atr'] = range_pct_of_atr
        result['distance_from_open_atr'] = distance_from_open_atr
        result['remaining_atr_pct'] = remaining_atr_pct

        return result

    def _compute_directional_context_features(
        self, result: pd.DataFrame, daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute directional context features."""
        close_arr = result['close'].values.astype(np.float64)
        n = len(result)

        # BB side alignment: +1 if at lower BB + support level, -1 if at upper BB + resistance
        bb_pct_b = result.get('daily_bb_pct_b', pd.Series(np.full(n, 0.5))).values
        # At lower BB extreme (<0.1) → price likely at support → favor reversal up (+1)
        # At upper BB extreme (>0.9) → price likely at resistance → favor reversal down (-1)
        result['bb_side_alignment'] = np.where(
            bb_pct_b < 0.1, 1.0,
            np.where(bb_pct_b > 0.9, -1.0, 0.0)
        )

        # Trend exhaustion score: RSI × BB interaction
        rsi = result.get('daily_rsi_14', pd.Series(np.full(n, 50.0))).values
        # High RSI + high %B = overbought exhaustion (positive score)
        # Low RSI + low %B = oversold exhaustion (positive score)
        rsi_norm = (rsi - 50) / 50  # -1 to 1
        bb_norm = bb_pct_b * 2 - 1  # -1 to 1
        result['trend_exhaustion_score'] = np.clip(rsi_norm * bb_norm, -1, 1)

        # Weekly trend alignment
        weekly_bb = result.get('weekly_bb_pct_b', pd.Series(np.full(n, 0.5))).values
        # Weekly at extreme + daily at same extreme = exhaustion (favor reversal)
        result['weekly_trend_alignment'] = np.where(
            (weekly_bb < 0.2) & (bb_pct_b < 0.2), 1.0,  # Oversold on both → bullish reversal
            np.where(
                (weekly_bb > 0.8) & (bb_pct_b > 0.8), -1.0,  # Overbought on both → bearish reversal
                0.0
            )
        )

        # Approach velocity (10-bar normalized by ATR)
        atr_arr = result.get('daily_atr_14', pd.Series(np.ones(n))).values
        atr_arr = np.maximum(atr_arr, 0.01)
        close_series = pd.Series(close_arr)
        velocity_10 = close_series.diff(10).fillna(0).values / 10 / atr_arr
        result['approach_velocity_10bar'] = np.clip(velocity_10, -3, 3)

        # Approach deceleration (velocity change)
        velocity_5 = close_series.diff(5).fillna(0).values / 5 / atr_arr
        velocity_prev_5 = close_series.shift(5).diff(5).fillna(0).values / 5 / atr_arr
        result['approach_deceleration'] = np.clip(velocity_5 - velocity_prev_5, -3, 3)

        # Bars since session extreme
        bars_since_extreme = np.zeros(n, dtype=np.float64)
        for day, day_df in result.groupby('trading_day'):
            day_mask = result['trading_day'] == day
            high_vals = result.loc[day_mask, 'high'].values
            low_vals = result.loc[day_mask, 'low'].values
            T = len(high_vals)

            cum_high = np.maximum.accumulate(high_vals)
            cum_low = np.minimum.accumulate(low_vals)

            # Track when new extremes were set
            last_extreme_bar = np.zeros(T, dtype=np.float64)
            last_new = 0
            for t in range(T):
                if t == 0 or high_vals[t] >= cum_high[t - 1] or low_vals[t] <= cum_low[t - 1]:
                    last_new = t
                last_extreme_bar[t] = t - last_new

            bars_since_extreme[day_mask.values] = last_extreme_bar

        result['bars_since_session_extreme'] = bars_since_extreme

        return result

    def get_daily_summary(self, trading_day: str) -> Optional[Dict[str, float]]:
        """Get daily feature summary for a specific day."""
        if self._daily_cache is None:
            return None

        row = self._daily_cache[self._daily_cache['trading_day'] == trading_day]
        if row.empty:
            return None

        return row.iloc[0].to_dict()
