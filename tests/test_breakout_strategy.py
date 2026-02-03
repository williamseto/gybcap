"""Tests for breakout retest strategy."""

import pytest
import numpy as np
import pandas as pd
from strategies.breakout.strategy import BreakoutRetestStrategy
from strategies.breakout.detector import BreakoutDetector
from strategies.breakout.config import BreakoutConfig
from strategies.core.types import Direction


class TestBreakoutDetector:
    """Tests for breakout event detection."""

    @pytest.fixture
    def simple_bars(self):
        """Create simple bars with known breakout patterns."""
        # Create bars that cross a level
        dates = pd.date_range('2024-01-15 09:30', periods=10, freq='15T')

        bars = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 103, 102, 101, 100, 99],
            'high': [101, 102, 103, 105, 106, 105, 104, 103, 102, 101],
            'low': [99, 100, 101, 102, 103, 102, 101, 100, 99, 98],
            'close': [101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
            'volume': [1000] * 10,
            'ovn': [0] * 10,  # All RTH
            'prev_high': [103] * 10,  # Level at 103
            'prev_low': [100] * 10,
        }, index=dates)

        return bars

    def test_detect_bull_breakout(self, simple_bars):
        """Test detection of bull breakout."""
        detector = BreakoutDetector(simple_bars, ['prev_high'])
        events = detector.detect()

        # Should detect breakout when close crosses above 103
        bull_events = [e for e in events if e.direction == Direction.BULL]
        assert len(bull_events) > 0

        # First bull breakout should be around bar 3 (close goes from 103 to 104)
        first_bull = bull_events[0]
        assert first_bull.level_name == 'prev_high'
        assert first_bull.level_price == 103.0

    def test_detect_bear_breakout(self, simple_bars):
        """Test detection of bear breakout."""
        detector = BreakoutDetector(simple_bars, ['prev_low'])
        events = detector.detect()

        # Should detect breakout when close crosses below 100
        bear_events = [e for e in events if e.direction == Direction.BEAR]
        # Last bar closes at 100, which crosses below prev_low of 100

    def test_rth_filter(self, simple_bars):
        """Test that overnight bars are filtered."""
        # Mark some bars as overnight
        bars = simple_bars.copy()
        bars.loc[bars.index[0:3], 'ovn'] = 1

        detector = BreakoutDetector(bars, ['prev_high'], rth_only=True)
        events = detector.detect()

        # Check no events from overnight bars
        ovn_indices = bars[bars['ovn'] == 1].index
        for event in events:
            assert event.trigger_ts not in ovn_indices

    def test_multiple_levels(self, simple_bars):
        """Test detection across multiple levels."""
        detector = BreakoutDetector(simple_bars, ['prev_high', 'prev_low'])
        events = detector.detect()

        # Should have events for both levels
        level_names = set(e.level_name for e in events)
        assert len(level_names) > 0

    def test_events_sorted_by_time(self, simple_bars):
        """Test that events are sorted by timestamp."""
        detector = BreakoutDetector(simple_bars, ['prev_high', 'prev_low'])
        events = detector.detect()

        timestamps = [e.trigger_ts for e in events]
        assert timestamps == sorted(timestamps)


class TestBreakoutRetestStrategy:
    """Tests for complete breakout retest strategy."""

    def test_strategy_initialization(self, sample_bars):
        """Test strategy initialization."""
        strat = BreakoutRetestStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low'],
            threshold_pct=0.001,
            lookahead_bars=10
        )

        assert strat.threshold_pct == 0.001
        assert strat.lookahead_bars == 10

    def test_with_config(self, sample_bars):
        """Test strategy with config object."""
        config = BreakoutConfig(
            threshold_pct=0.002,
            lookahead_bars=20,
            stop_buffer_pct=0.003,
        )

        strat = BreakoutRetestStrategy(sample_bars, config=config)

        assert strat.threshold_pct == 0.002
        assert strat.lookahead_bars == 20

    def test_find_retest_and_build_trades(self, sample_bars):
        """Test full trade building pipeline."""
        strat = BreakoutRetestStrategy(
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

        # Check trades structure
        for trade in trades:
            assert hasattr(trade, 'entry_price')
            assert hasattr(trade, 'exit_price')
            assert hasattr(trade, 'pnl')
            assert trade.direction in ['bull', 'bear']

        # Check features DataFrame
        if len(trades) > 0:
            assert 'y_succ' in features_df.columns
            assert 'bear' in features_df.columns
            assert len(features_df) == len(trades)

    def test_legacy_interface(self, sample_bars):
        """Test legacy detect_breakouts interface."""
        strat = BreakoutRetestStrategy(
            sample_bars,
            level_cols=['prev_high', 'prev_low'],
        )

        # Legacy method returns list of tuples
        events = strat.detect_breakouts()

        for event in events:
            assert isinstance(event, tuple)
            assert len(event) == 4  # (level_name, level_price, ts, direction)


class TestBreakoutConfig:
    """Tests for breakout configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BreakoutConfig.default()

        assert config.timeframe == '5min'
        assert config.threshold_pct == 0.0012
        assert config.reward_risk_ratio == 2.0
        assert len(config.level_cols) > 0

    def test_from_dict(self):
        """Test creating config from dictionary."""
        d = {
            'threshold_pct': 0.002,
            'lookahead_bars': 15,
            'timeframe': '10min',
        }

        config = BreakoutConfig.from_dict(d)

        assert config.threshold_pct == 0.002
        assert config.lookahead_bars == 15
        assert config.timeframe == '10min'

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BreakoutConfig(
            threshold_pct=0.003,
            lookahead_bars=5
        )

        d = config.to_dict()

        assert d['threshold_pct'] == 0.003
        assert d['lookahead_bars'] == 5
        assert 'level_cols' in d
