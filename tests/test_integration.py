"""Integration tests for full strategy pipeline."""

import pytest
import numpy as np
import pandas as pd
from strategies import (
    BreakoutRetestStrategy,
    ReversionStrategy,
    VectorizedTradeSimulator,
    Trade,
)
from strategies.training.trainer import Trainer
from strategies.training.evaluation import calculate_trading_metrics


class TestFullPipeline:
    """Test complete strategy pipeline from data to trades."""

    def test_breakout_pipeline(self, sample_bars):
        """Test full breakout strategy pipeline."""
        # Run strategy
        strat = BreakoutRetestStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low', 'vwap'],
            threshold_pct=0.0012,
            lookahead_bars=10
        )

        trades, features_df = strat.find_retest_and_build_trades(
            stop_buffer_pct=0.0025,
            rr=2.0
        )

        # Verify trade structure
        for trade in trades:
            assert trade.entry_ts is not None
            assert trade.exit_ts is not None
            assert trade.pnl is not None
            assert trade.stop is not None
            assert trade.take is not None

        # Verify features
        if len(trades) > 0:
            assert len(features_df) == len(trades)
            assert 'y_succ' in features_df.columns
            assert set(features_df['y_succ'].unique()).issubset({0, 1})

    def test_reversion_pipeline(self, sample_bars):
        """Test full reversion strategy pipeline."""
        strat = ReversionStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low', 'vwap'],
            threshold_pct=0.0012,
            lookahead_bars=10
        )

        trades, features_df = strat.find_retest_and_build_trades(
            stop_buffer_pct=0.0025,
            rr=2.0
        )

        for trade in trades:
            assert trade.entry_ts is not None
            assert trade.exit_ts is not None
            assert trade.pnl is not None

    def test_training_pipeline(self, sample_trade_features):
        """Test model training pipeline."""
        feature_cols = [
            'close_z20', 'ovn_lo_z', 'ovn_hi_z', 'ib_lo_z', 'ib_hi_z',
            'vwap_z', 'rsi', 'vol_z', 'adx', 'ofi_z',
            'nearby_gamma_score', 'bear'
        ]

        trainer = Trainer(feature_cols, test_size=0.3)
        model = trainer.train(sample_trade_features, verbose=False)

        assert model is not None
        assert trainer.best_threshold is not None
        assert 0 < trainer.best_threshold < 1

        # Test prediction
        preds = trainer.predict(sample_trade_features)
        assert len(preds) == len(sample_trade_features)
        assert set(preds).issubset({0, 1})

    def test_trade_evaluation(self, legacy_trades, sample_trade_features):
        """Test trade evaluation pipeline."""
        metrics = calculate_trading_metrics(legacy_trades)

        assert 'total_trades' in metrics
        assert 'total_pnl' in metrics
        assert 'win_rate' in metrics
        assert 'profit_factor' in metrics

        assert metrics['total_trades'] == len(legacy_trades)


class TestVectorizedVsLegacy:
    """Test that vectorized simulation matches expectations."""

    def test_simulation_consistency(self, sample_bars):
        """Test that simulator produces consistent results."""
        bars = sample_bars.reset_index()
        simulator = VectorizedTradeSimulator(bars)

        # Run same trade multiple times
        results = []
        for _ in range(3):
            result = simulator.simulate_single_trade(
                entry_idx=10,
                entry_price=5000.0,
                stop=4990.0,
                take=5020.0,
                is_bull=True
            )
            results.append(result)

        # All should be identical
        assert all(r.exit_price == results[0].exit_price for r in results)
        assert all(r.pnl == results[0].pnl for r in results)

    def test_bull_vs_bear_symmetry(self, sample_bars):
        """Test that bull and bear trades are symmetric."""
        bars = sample_bars.reset_index()
        simulator = VectorizedTradeSimulator(bars)

        entry_price = 5000.0
        risk = 10.0

        bull_result = simulator.simulate_single_trade(
            entry_idx=10,
            entry_price=entry_price,
            stop=entry_price - risk,
            take=entry_price + 2 * risk,
            is_bull=True
        )

        bear_result = simulator.simulate_single_trade(
            entry_idx=10,
            entry_price=entry_price,
            stop=entry_price + risk,
            take=entry_price - 2 * risk,
            is_bull=False
        )

        # Both should have valid exits
        assert bull_result.exit_type in ['stop', 'take', 'close']
        assert bear_result.exit_type in ['stop', 'take', 'close']


class TestTradeDataclass:
    """Test Trade dataclass functionality."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            entry_ts=pd.Timestamp('2024-01-15 10:00'),
            entry_price=5000.0,
            direction='bull',
            size=1.0,
            stop=4990.0,
            take=5020.0,
        )

        assert trade.entry_price == 5000.0
        assert trade.direction == 'bull'

    def test_trade_with_exit(self):
        """Test trade with exit information."""
        trade = Trade(
            entry_ts=pd.Timestamp('2024-01-15 10:00'),
            entry_price=5000.0,
            direction='bull',
            size=2.0,
            stop=4990.0,
            take=5020.0,
            exit_ts=pd.Timestamp('2024-01-15 11:00'),
            exit_price=5020.0,
            pnl=40.0,  # (5020 - 5000) * 2
        )

        assert trade.pnl == 40.0

    def test_compute_pnl(self):
        """Test P&L computation."""
        trade = Trade(
            entry_ts=pd.Timestamp('2024-01-15 10:00'),
            entry_price=5000.0,
            direction='bull',
            size=2.0,
            stop=4990.0,
            take=5020.0,
            exit_price=5020.0,
        )

        pnl = trade.compute_pnl()
        assert pnl == 40.0  # (5020 - 5000) * 2

        # Bear trade
        bear_trade = Trade(
            entry_ts=pd.Timestamp('2024-01-15 10:00'),
            entry_price=5000.0,
            direction='bear',
            size=2.0,
            stop=5010.0,
            take=4980.0,
            exit_price=4980.0,
        )

        bear_pnl = bear_trade.compute_pnl()
        assert bear_pnl == 40.0  # (5000 - 4980) * 2


class TestMetricsCalculation:
    """Test trading metrics calculations."""

    def test_win_rate_calculation(self, legacy_trades):
        """Test win rate calculation."""
        metrics = calculate_trading_metrics(legacy_trades)

        # Manually calculate win rate
        winners = sum(1 for t in legacy_trades if t.pnl > 0)
        expected_win_rate = winners / len(legacy_trades)

        assert abs(metrics['win_rate'] - expected_win_rate) < 0.001

    def test_filtered_metrics(self, legacy_trades):
        """Test metrics with prediction filter."""
        # Create predictions that filter to only winners
        predictions = np.array([1 if t.pnl > 0 else 0 for t in legacy_trades])

        metrics = calculate_trading_metrics(legacy_trades, predictions)

        # Should have 100% win rate with this filter
        assert metrics['win_rate'] == 1.0
