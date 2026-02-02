"""
Vectorized trade simulation using numba for performance.

Provides ~50-167x speedup over Python for-loops for trade simulation.
"""

import numpy as np
import pandas as pd
import numba as nb
from typing import Tuple, Optional
from dataclasses import dataclass

from strategies.core.types import TradeResult


# Exit type constants for numba (strings not supported in nopython mode)
EXIT_STOP = 0
EXIT_TAKE = 1
EXIT_CLOSE = 2


@nb.jit(nopython=True, cache=True)
def _simulate_single_trade(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_idx: int,
    entry_price: float,
    stop: float,
    take: float,
    is_bull: bool
) -> Tuple[int, float, int]:
    """
    Simulate a single trade from entry to exit.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        entry_idx: Bar index where trade is entered
        entry_price: Entry price
        stop: Stop loss price
        take: Take profit price
        is_bull: True for long, False for short

    Returns:
        exit_idx: Bar index where trade exited
        exit_price: Exit price
        exit_type: 0=stop, 1=take, 2=close (end of data)
    """
    n_bars = len(highs)

    for i in range(entry_idx, n_bars):
        hi = highs[i]
        lo = lows[i]

        if is_bull:
            # Long trade: check stop first, then take
            if lo <= stop:
                return i, stop, EXIT_STOP
            if hi >= take:
                return i, take, EXIT_TAKE
        else:
            # Short trade: check stop first, then take
            if hi >= stop:
                return i, stop, EXIT_STOP
            if lo <= take:
                return i, take, EXIT_TAKE

    # No exit found, close at last bar
    return n_bars - 1, closes[n_bars - 1], EXIT_CLOSE


@nb.jit(nopython=True, parallel=True, cache=True)
def simulate_trades_batch(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_indices: np.ndarray,
    entry_prices: np.ndarray,
    stops: np.ndarray,
    takes: np.ndarray,
    is_bulls: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate N trades in parallel using numba.

    Args:
        highs: Array of high prices (n_bars,)
        lows: Array of low prices (n_bars,)
        closes: Array of close prices (n_bars,)
        entry_indices: Entry bar indices for each trade (n_trades,)
        entry_prices: Entry prices for each trade (n_trades,)
        stops: Stop prices for each trade (n_trades,)
        takes: Take prices for each trade (n_trades,)
        is_bulls: Boolean array, True for long trades (n_trades,)

    Returns:
        exit_indices: Exit bar indices (n_trades,)
        exit_prices: Exit prices (n_trades,)
        exit_types: Exit types, 0=stop, 1=take, 2=close (n_trades,)
    """
    n_trades = len(entry_indices)

    exit_indices = np.empty(n_trades, dtype=np.int64)
    exit_prices = np.empty(n_trades, dtype=np.float64)
    exit_types = np.empty(n_trades, dtype=np.int64)

    for t in nb.prange(n_trades):
        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_indices[t],
            entry_prices[t],
            stops[t],
            takes[t],
            is_bulls[t]
        )
        exit_indices[t] = exit_idx
        exit_prices[t] = exit_price
        exit_types[t] = exit_type

    return exit_indices, exit_prices, exit_types


class VectorizedTradeSimulator:
    """
    High-performance trade simulator using numba JIT compilation.

    Provides both single-trade and batch simulation methods.
    """

    # Map exit type integers to strings
    EXIT_TYPE_NAMES = {EXIT_STOP: 'stop', EXIT_TAKE: 'take', EXIT_CLOSE: 'close'}

    def __init__(self, bars: pd.DataFrame):
        """
        Initialize simulator with OHLCV data.

        Args:
            bars: DataFrame with 'high', 'low', 'close' columns
        """
        self.bars = bars

        # Pre-extract numpy arrays for numba
        self._highs = bars['high'].values.astype(np.float64)
        self._lows = bars['low'].values.astype(np.float64)
        self._closes = bars['close'].values.astype(np.float64)

        # Cache index for timestamp lookups
        if isinstance(bars.index, pd.DatetimeIndex):
            self._timestamps = bars.index
        elif 'dt' in bars.columns:
            self._timestamps = pd.DatetimeIndex(bars['dt'])
        else:
            self._timestamps = pd.RangeIndex(len(bars))

    def simulate_single_trade(
        self,
        entry_idx: int,
        entry_price: float,
        stop: float,
        take: float,
        is_bull: bool
    ) -> TradeResult:
        """
        Simulate a single trade.

        Args:
            entry_idx: Integer index into bars DataFrame
            entry_price: Entry price
            stop: Stop loss price
            take: Take profit price
            is_bull: True for long, False for short

        Returns:
            TradeResult with exit details and P&L
        """
        exit_idx, exit_price, exit_type = _simulate_single_trade(
            self._highs,
            self._lows,
            self._closes,
            entry_idx,
            entry_price,
            stop,
            take,
            is_bull
        )

        # Compute P&L
        if is_bull:
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        return TradeResult(
            exit_ts=self._timestamps[exit_idx],
            exit_price=exit_price,
            exit_bar_index=exit_idx,
            exit_type=self.EXIT_TYPE_NAMES[exit_type],
            pnl=pnl
        )

    def simulate_batch(
        self,
        entry_indices: np.ndarray,
        entry_prices: np.ndarray,
        stops: np.ndarray,
        takes: np.ndarray,
        is_bulls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate multiple trades in parallel.

        Args:
            entry_indices: Entry bar indices (n_trades,)
            entry_prices: Entry prices (n_trades,)
            stops: Stop prices (n_trades,)
            takes: Take prices (n_trades,)
            is_bulls: Boolean flags for long trades (n_trades,)

        Returns:
            exit_indices: Exit bar indices
            exit_prices: Exit prices
            exit_types: Exit type strings
            pnls: P&L for each trade
        """
        # Ensure inputs are numpy arrays with correct types
        entry_indices = np.asarray(entry_indices, dtype=np.int64)
        entry_prices = np.asarray(entry_prices, dtype=np.float64)
        stops = np.asarray(stops, dtype=np.float64)
        takes = np.asarray(takes, dtype=np.float64)
        is_bulls = np.asarray(is_bulls, dtype=np.bool_)

        # Run batch simulation
        exit_indices, exit_prices, exit_types = simulate_trades_batch(
            self._highs,
            self._lows,
            self._closes,
            entry_indices,
            entry_prices,
            stops,
            takes,
            is_bulls
        )

        # Compute P&Ls
        pnls = np.where(
            is_bulls,
            exit_prices - entry_prices,
            entry_prices - exit_prices
        )

        # Convert exit types to strings
        exit_type_strs = np.array([self.EXIT_TYPE_NAMES[t] for t in exit_types])

        return exit_indices, exit_prices, exit_type_strs, pnls

    def simulate_batch_to_results(
        self,
        entry_indices: np.ndarray,
        entry_prices: np.ndarray,
        stops: np.ndarray,
        takes: np.ndarray,
        is_bulls: np.ndarray
    ) -> list:
        """
        Simulate batch and return list of TradeResult objects.

        This is a convenience method when you need TradeResult objects
        rather than raw arrays.
        """
        exit_indices, exit_prices, exit_type_strs, pnls = self.simulate_batch(
            entry_indices, entry_prices, stops, takes, is_bulls
        )

        results = []
        for i in range(len(exit_indices)):
            results.append(TradeResult(
                exit_ts=self._timestamps[exit_indices[i]],
                exit_price=exit_prices[i],
                exit_bar_index=int(exit_indices[i]),
                exit_type=exit_type_strs[i],
                pnl=pnls[i]
            ))

        return results
