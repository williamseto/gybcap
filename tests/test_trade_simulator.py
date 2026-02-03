"""Tests for vectorized trade simulator."""

import pytest
import numpy as np
import pandas as pd
from strategies.core.trade_simulator import (
    VectorizedTradeSimulator,
    _simulate_single_trade,
    simulate_trades_batch,
    EXIT_STOP,
    EXIT_TAKE,
    EXIT_CLOSE,
)


class TestSimulateSingleTrade:
    """Tests for single trade simulation."""

    def test_bull_trade_hits_take(self):
        """Test bull trade that hits take profit."""
        # Entry is at bar 0, so we check from bar 0 onwards
        # Bar 0: high=100, low=95 - entry bar
        # Bar 1: high=105, low=100 - doesn't hit take (110)
        # Bar 2: high=110, low=105 - hits take at 110
        highs = np.array([100, 105, 110, 115, 120], dtype=np.float64)
        lows = np.array([95, 100, 105, 110, 115], dtype=np.float64)
        closes = np.array([98, 103, 108, 113, 118], dtype=np.float64)

        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_idx=0,
            entry_price=100.0,
            stop=90.0,  # Set stop below all lows so it doesn't trigger
            take=110.0,
            is_bull=True
        )

        assert exit_idx == 2  # Take hit on bar 2
        assert exit_price == 110.0
        assert exit_type == EXIT_TAKE

    def test_bull_trade_hits_stop(self):
        """Test bull trade that hits stop loss."""
        highs = np.array([100, 98, 95, 90, 85], dtype=np.float64)
        lows = np.array([95, 93, 88, 85, 80], dtype=np.float64)
        closes = np.array([98, 95, 90, 87, 82], dtype=np.float64)

        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_idx=0,
            entry_price=100.0,
            stop=90.0,
            take=115.0,
            is_bull=True
        )

        assert exit_idx == 2  # Stop hit on bar 2
        assert exit_price == 90.0
        assert exit_type == EXIT_STOP

    def test_bear_trade_hits_take(self):
        """Test bear trade that hits take profit."""
        highs = np.array([100, 98, 95, 90, 85], dtype=np.float64)
        lows = np.array([95, 93, 88, 85, 80], dtype=np.float64)
        closes = np.array([98, 95, 90, 87, 82], dtype=np.float64)

        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_idx=0,
            entry_price=100.0,
            stop=105.0,
            take=90.0,
            is_bull=False
        )

        assert exit_idx == 2  # Take hit on bar 2
        assert exit_price == 90.0
        assert exit_type == EXIT_TAKE

    def test_bear_trade_hits_stop(self):
        """Test bear trade that hits stop loss."""
        highs = np.array([100, 103, 106, 110, 115], dtype=np.float64)
        lows = np.array([95, 100, 103, 107, 112], dtype=np.float64)
        closes = np.array([98, 102, 105, 109, 114], dtype=np.float64)

        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_idx=0,
            entry_price=100.0,
            stop=105.0,
            take=85.0,
            is_bull=False
        )

        assert exit_idx == 2  # Stop hit on bar 2
        assert exit_price == 105.0
        assert exit_type == EXIT_STOP

    def test_trade_closes_at_end(self):
        """Test trade that reaches end without hitting stop or take."""
        highs = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        lows = np.array([99, 100, 101, 102, 103], dtype=np.float64)
        closes = np.array([100, 101, 102, 103, 104], dtype=np.float64)

        exit_idx, exit_price, exit_type = _simulate_single_trade(
            highs, lows, closes,
            entry_idx=0,
            entry_price=100.0,
            stop=90.0,  # Far stop
            take=120.0,  # Far take
            is_bull=True
        )

        assert exit_idx == 4  # Last bar
        assert exit_price == 104.0  # Close price
        assert exit_type == EXIT_CLOSE


class TestBatchSimulation:
    """Tests for batch trade simulation."""

    def test_batch_multiple_trades(self):
        """Test batch simulation of multiple trades."""
        np.random.seed(42)
        n_bars = 100

        # Generate random price data
        returns = 0.001 * np.random.randn(n_bars)
        price = 100 * np.exp(np.cumsum(returns))

        highs = price + np.abs(0.5 * np.random.randn(n_bars))
        lows = price - np.abs(0.5 * np.random.randn(n_bars))
        closes = price

        # Define trades
        entry_indices = np.array([0, 10, 20, 30], dtype=np.int64)
        entry_prices = closes[entry_indices]
        stops = entry_prices - 2.0
        takes = entry_prices + 4.0
        is_bulls = np.array([True, True, True, True])

        exit_indices, exit_prices, exit_types = simulate_trades_batch(
            highs, lows, closes,
            entry_indices, entry_prices,
            stops, takes, is_bulls
        )

        assert len(exit_indices) == 4
        assert len(exit_prices) == 4
        assert len(exit_types) == 4

        # All exit indices should be >= entry indices
        assert all(exit_indices >= entry_indices)

        # Exit types should be valid
        assert all(t in [EXIT_STOP, EXIT_TAKE, EXIT_CLOSE] for t in exit_types)


class TestVectorizedTradeSimulator:
    """Tests for VectorizedTradeSimulator class."""

    @pytest.fixture
    def simulator(self, sample_bars):
        """Create simulator with sample data."""
        bars = sample_bars.reset_index()
        return VectorizedTradeSimulator(bars)

    def test_simulate_single_trade(self, simulator):
        """Test single trade simulation through class."""
        result = simulator.simulate_single_trade(
            entry_idx=5,
            entry_price=5000.0,
            stop=4990.0,
            take=5020.0,
            is_bull=True
        )

        assert result.exit_bar_index >= 5
        assert result.exit_type in ['stop', 'take', 'close']
        assert isinstance(result.pnl, float)

    def test_simulate_batch(self, simulator):
        """Test batch simulation through class."""
        entry_indices = np.array([5, 10, 15, 20])
        entry_prices = np.array([5000.0, 5001.0, 5002.0, 5003.0])
        stops = entry_prices - 10.0
        takes = entry_prices + 20.0
        is_bulls = np.array([True, False, True, False])

        exit_indices, exit_prices, exit_types, pnls = simulator.simulate_batch(
            entry_indices, entry_prices, stops, takes, is_bulls
        )

        assert len(exit_indices) == 4
        assert len(pnls) == 4
        assert all(t in ['stop', 'take', 'close'] for t in exit_types)

    def test_pnl_calculation(self, simulator):
        """Test P&L is calculated correctly."""
        # Bull trade that wins
        result = simulator.simulate_single_trade(
            entry_idx=0,
            entry_price=100.0,
            stop=95.0,
            take=110.0,
            is_bull=True
        )

        if result.exit_type == 'take':
            assert result.pnl == 10.0
        elif result.exit_type == 'stop':
            assert result.pnl == -5.0

    def test_batch_results_conversion(self, simulator):
        """Test conversion to TradeResult objects."""
        entry_indices = np.array([5, 10])
        entry_prices = np.array([5000.0, 5001.0])
        stops = entry_prices - 10.0
        takes = entry_prices + 20.0
        is_bulls = np.array([True, True])

        results = simulator.simulate_batch_to_results(
            entry_indices, entry_prices, stops, takes, is_bulls
        )

        assert len(results) == 2
        assert all(hasattr(r, 'exit_ts') for r in results)
        assert all(hasattr(r, 'pnl') for r in results)
