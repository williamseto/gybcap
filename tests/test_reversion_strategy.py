"""Tests for reversion strategy."""

import pytest
import numpy as np
import pandas as pd
from strategies.reversion.strategy import ReversionStrategy
from strategies.reversion.detector import ReversionDetector
from strategies.reversion.config import ReversionConfig
from strategies.core.types import Direction


class TestReversionDetector:
    """Tests for reversion event detection."""

    @pytest.fixture
    def simple_bars(self):
        """Create simple bars with known reversion patterns."""
        dates = pd.date_range('2024-01-15 09:30', periods=10, freq='15T')

        # Bull reversion pattern: open above 100, low touches 100, close above 100
        # Bear reversion pattern: open below 100, high touches 100, close below 100

        bars = pd.DataFrame({
            'open': [102, 101, 98, 99, 102, 103, 97, 96, 102, 101],
            'high': [103, 102, 100, 100, 103, 104, 98, 100, 103, 102],
            'low': [100, 99, 96, 97, 100, 101, 95, 95, 100, 99],
            'close': [101, 100, 97, 98, 101, 102, 96, 97, 101, 100],
            'volume': [1000] * 10,
            'ovn': [0] * 10,
            'prev_high': [102] * 10,
            'prev_low': [100] * 10,
        }, index=dates)

        return bars

    def test_detect_bull_reversion(self, simple_bars):
        """Test detection of bull reversion (failed breakdown)."""
        detector = ReversionDetector(simple_bars, ['prev_low'])
        events = detector.detect()

        bull_events = [e for e in events if e.direction == Direction.BULL]

        # Should detect bars where:
        # - open > prev_low (100)
        # - low <= prev_low
        # - close > prev_low
        for event in bull_events:
            assert event.level_name == 'prev_low'
            assert event.direction == Direction.BULL

    def test_detect_bear_reversion(self, simple_bars):
        """Test detection of bear reversion (failed breakout)."""
        # Create specific pattern for bear reversion
        dates = pd.date_range('2024-01-15 09:30', periods=5, freq='15T')

        bars = pd.DataFrame({
            'open': [98, 99, 98, 99, 98],  # Below 100
            'high': [100, 101, 100, 101, 100],  # Touches 100
            'low': [97, 98, 97, 98, 97],
            'close': [99, 99, 98, 98, 97],  # Closes below 100
            'volume': [1000] * 5,
            'ovn': [0] * 5,
            'prev_high': [100] * 5,
        }, index=dates)

        detector = ReversionDetector(bars, ['prev_high'])
        events = detector.detect()

        bear_events = [e for e in events if e.direction == Direction.BEAR]
        assert len(bear_events) > 0

    def test_rth_filter(self, simple_bars):
        """Test that overnight bars are filtered."""
        bars = simple_bars.copy()
        bars.loc[bars.index[0:3], 'ovn'] = 1

        detector = ReversionDetector(bars, ['prev_low'], rth_only=True)
        events = detector.detect()

        ovn_indices = bars[bars['ovn'] == 1].index
        for event in events:
            assert event.trigger_ts not in ovn_indices


class TestReversionStrategy:
    """Tests for complete reversion strategy."""

    def test_strategy_initialization(self, sample_bars):
        """Test strategy initialization."""
        strat = ReversionStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low'],
            threshold_pct=0.001,
            lookahead_bars=10
        )

        assert strat.threshold_pct == 0.001
        assert strat.lookahead_bars == 10

    def test_with_config(self, sample_bars):
        """Test strategy with config object."""
        config = ReversionConfig(
            threshold_pct=0.002,
            lookahead_bars=20,
            timeframe='15min',
        )

        strat = ReversionStrategy(sample_bars, config=config)

        assert strat.threshold_pct == 0.002
        assert strat.lookahead_bars == 20

    def test_find_retest_and_build_trades(self, sample_bars):
        """Test full trade building pipeline."""
        strat = ReversionStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low', 'vwap'],
            threshold_pct=0.0012,
            lookahead_bars=10
        )

        trades, features_df = strat.find_retest_and_build_trades(
            stop_buffer_pct=0.0025,
            rr=2.0,
            fixed_size=1.0
        )

        for trade in trades:
            assert hasattr(trade, 'entry_price')
            assert hasattr(trade, 'exit_price')
            assert hasattr(trade, 'pnl')

        if len(trades) > 0:
            assert 'y_succ' in features_df.columns
            assert 'bear' in features_df.columns

    def test_legacy_interface(self, sample_bars):
        """Test legacy get_trigger_events interface."""
        strat = ReversionStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low'],
        )

        events = strat.get_trigger_events()

        for event in events:
            assert isinstance(event, tuple)
            assert len(event) == 4


class TestReversionConfig:
    """Tests for reversion configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReversionConfig.default()

        assert config.timeframe == '15min'  # Reversion default is 15min
        assert config.threshold_pct == 0.0012
        assert config.reward_risk_ratio == 2.0

    def test_different_from_breakout(self):
        """Test that reversion defaults differ from breakout."""
        from strategies.breakout.config import BreakoutConfig

        rev_config = ReversionConfig.default()
        bo_config = BreakoutConfig.default()

        # Reversion uses longer timeframe by default
        assert rev_config.timeframe == '15min'
        assert bo_config.timeframe == '5min'
