"""Pytest fixtures for strategy tests."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate 5 trading days of 1-min data
    dates = []
    trading_days = []
    current_date = datetime(2024, 1, 15, 6, 30)  # Start at RTH open

    for day in range(5):
        trading_day = current_date.date()

        # Overnight session (18:00-06:30)
        ovn_start = current_date.replace(hour=18) - timedelta(days=1)
        for minute in range(750):  # ~12.5 hours
            ts = ovn_start + timedelta(minutes=minute)
            dates.append(ts)
            trading_days.append(trading_day)

        # RTH session (06:30-13:00)
        rth_start = current_date.replace(hour=6, minute=30)
        for minute in range(390):  # 6.5 hours
            ts = rth_start + timedelta(minutes=minute)
            dates.append(ts)
            trading_days.append(trading_day)

        current_date += timedelta(days=1)

    n = len(dates)

    # Generate price series
    returns = 0.0001 * np.random.randn(n)
    price = 5000 * np.exp(np.cumsum(returns))

    # Add some structure
    close = price
    high = close + np.abs(2 * np.random.randn(n))
    low = close - np.abs(2 * np.random.randn(n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    volume = 1000 + 500 * np.random.rand(n)
    bidvolume = volume * (0.4 + 0.2 * np.random.rand(n))
    askvolume = volume - bidvolume

    # Create overnight mask
    ovn = np.array([1 if ts.hour < 6 or ts.hour >= 18 else 0 for ts in dates])

    df = pd.DataFrame({
        'dt': dates,
        'trading_day': trading_days,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'bidvolume': bidvolume,
        'askvolume': askvolume,
        'ovn': ovn,
    })

    return df


@pytest.fixture
def sample_bars(sample_ohlcv_data):
    """Create sample 15-min bars from OHLCV data."""
    df = sample_ohlcv_data

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'ovn': 'first',
    }

    bars = df.set_index('dt').groupby('trading_day').resample('15T').agg(agg_dict)
    bars = bars.dropna().reset_index()

    # Add price levels
    np.random.seed(42)
    base_price = 5000

    for day in bars['trading_day'].unique():
        day_mask = bars['trading_day'] == day
        day_high = bars.loc[day_mask, 'high'].max()
        day_low = bars.loc[day_mask, 'low'].min()

        bars.loc[day_mask, 'prev_high'] = day_high + np.random.rand() * 5
        bars.loc[day_mask, 'prev_low'] = day_low - np.random.rand() * 5
        bars.loc[day_mask, 'prev_mid'] = (
            bars.loc[day_mask, 'prev_high'] + bars.loc[day_mask, 'prev_low']
        ) / 2
        bars.loc[day_mask, 'vwap'] = bars.loc[day_mask, 'close'].mean()
        bars.loc[day_mask, 'ovn_hi'] = day_high + np.random.rand() * 3
        bars.loc[day_mask, 'ovn_lo'] = day_low - np.random.rand() * 3
        bars.loc[day_mask, 'rth_lo'] = day_low
        bars.loc[day_mask, 'rth_hi'] = day_high

    return bars.set_index('dt')


@pytest.fixture
def sample_trade_features():
    """Create sample trade features DataFrame for testing."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'close_z20': np.random.randn(n),
        'ovn_lo_z': np.random.randn(n),
        'ovn_hi_z': np.random.randn(n),
        'ib_lo_z': np.random.randn(n),
        'ib_hi_z': np.random.randn(n),
        'vwap_z': np.random.randn(n),
        'rsi': 30 + 40 * np.random.rand(n),
        'vol_z': np.random.randn(n),
        'adx': 10 + 30 * np.random.rand(n),
        'ofi_z': np.random.randn(n),
        'nearby_gamma_score': np.random.rand(n),
        'bear': np.random.randint(0, 2, n),
        'y_succ': np.random.randint(0, 2, n),
    })

    return df


@pytest.fixture
def legacy_trades():
    """Create sample legacy trades for regression testing."""
    from strategies.core.base import Trade

    trades = []
    np.random.seed(42)

    for i in range(20):
        entry_price = 5000 + np.random.randn() * 10
        direction = 'bull' if np.random.rand() > 0.5 else 'bear'
        risk = 10
        rr = 2.0

        if direction == 'bull':
            stop = entry_price - risk
            take = entry_price + rr * risk
            exit_price = take if np.random.rand() > 0.4 else stop
            pnl = exit_price - entry_price
        else:
            stop = entry_price + risk
            take = entry_price - rr * risk
            exit_price = take if np.random.rand() > 0.4 else stop
            pnl = entry_price - exit_price

        trade = Trade(
            entry_ts=pd.Timestamp(f'2024-01-15 10:{i:02d}:00'),
            entry_price=entry_price,
            direction=direction,
            size=1.0,
            stop=stop,
            take=take,
            exit_ts=pd.Timestamp(f'2024-01-15 11:{i:02d}:00'),
            exit_price=exit_price,
            pnl=pnl,
        )
        trades.append(trade)

    return trades
