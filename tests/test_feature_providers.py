"""Tests for feature providers."""

import pytest
import numpy as np
import pandas as pd
from strategies.features.base import BaseFeatureProvider, FeatureProvider
from strategies.features.registry import FeatureRegistry, FeaturePipeline
from strategies.features.price_levels import PriceLevelProvider


class TestFeatureRegistry:
    """Tests for feature provider registry."""

    def test_register_and_get(self):
        """Test registering and retrieving a provider."""
        # Clear registry first
        FeatureRegistry.clear()

        @FeatureRegistry.register('test_provider')
        class TestProvider:
            @property
            def name(self):
                return 'test'

            @property
            def feature_names(self):
                return ['feat1']

            def compute(self, ohlcv, context=None):
                return ohlcv

        provider_cls = FeatureRegistry.get('test_provider')
        assert provider_cls == TestProvider

        # Clean up
        FeatureRegistry.clear()

    def test_create_instance(self):
        """Test creating a provider instance."""
        FeatureRegistry.clear()

        @FeatureRegistry.register('simple')
        class SimpleProvider:
            def __init__(self, value=10):
                self.value = value

            @property
            def name(self):
                return 'simple'

            @property
            def feature_names(self):
                return []

            def compute(self, ohlcv, context=None):
                return ohlcv

        provider = FeatureRegistry.create('simple', value=42)
        assert provider.value == 42

        FeatureRegistry.clear()

    def test_list_providers(self):
        """Test listing registered providers."""
        FeatureRegistry.clear()

        @FeatureRegistry.register('a')
        class ProviderA:
            pass

        @FeatureRegistry.register('b')
        class ProviderB:
            pass

        providers = FeatureRegistry.list_providers()
        assert 'a' in providers
        assert 'b' in providers

        FeatureRegistry.clear()

    def test_get_unknown_raises(self):
        """Test that getting unknown provider raises."""
        FeatureRegistry.clear()

        with pytest.raises(KeyError):
            FeatureRegistry.get('nonexistent')


class TestFeaturePipeline:
    """Tests for feature pipeline."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers for testing."""

        class Provider1:
            @property
            def name(self):
                return 'provider1'

            @property
            def feature_names(self):
                return ['feat1', 'feat2']

            def compute(self, ohlcv, context=None):
                result = ohlcv.copy()
                result['feat1'] = 1.0
                result['feat2'] = 2.0
                return result

        class Provider2:
            @property
            def name(self):
                return 'provider2'

            @property
            def feature_names(self):
                return ['feat3']

            def compute(self, ohlcv, context=None):
                result = ohlcv.copy()
                result['feat3'] = 3.0
                return result

        return Provider1(), Provider2()

    def test_add_provider(self, mock_providers):
        """Test adding providers to pipeline."""
        p1, p2 = mock_providers

        pipeline = FeaturePipeline()
        pipeline.add_provider(p1)
        pipeline.add_provider(p2)

        assert len(pipeline.providers) == 2

    def test_compute_combines_features(self, mock_providers, sample_ohlcv_data):
        """Test that compute combines features from all providers."""
        p1, p2 = mock_providers

        pipeline = FeaturePipeline([p1, p2])
        result = pipeline.compute(sample_ohlcv_data)

        assert 'feat1' in result.columns
        assert 'feat2' in result.columns
        assert 'feat3' in result.columns

    def test_get_feature_names(self, mock_providers):
        """Test getting all feature names from pipeline."""
        p1, p2 = mock_providers

        pipeline = FeaturePipeline([p1, p2])
        names = pipeline.get_feature_names()

        assert 'feat1' in names
        assert 'feat2' in names
        assert 'feat3' in names

    def test_empty_pipeline(self, sample_ohlcv_data):
        """Test empty pipeline returns copy of input."""
        pipeline = FeaturePipeline()
        result = pipeline.compute(sample_ohlcv_data)

        pd.testing.assert_frame_equal(result, sample_ohlcv_data)


class TestPriceLevelProvider:
    """Tests for price level feature provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = PriceLevelProvider()
        assert provider.name == 'price_levels'
        assert len(provider.feature_names) > 0

    def test_prev_day_levels(self, sample_ohlcv_data):
        """Test previous day level computation."""
        provider = PriceLevelProvider()
        levels = provider.prev_day_levels(sample_ohlcv_data)

        assert 'prev_high' in levels.columns
        assert 'prev_low' in levels.columns
        assert 'prev_mid' in levels.columns

        # First day should have NaN
        assert pd.isna(levels.iloc[0]['prev_high'])

        # Other days should have values
        if len(levels) > 1:
            assert not pd.isna(levels.iloc[1]['prev_high'])

    def test_compute_features(self, sample_ohlcv_data):
        """Test feature computation."""
        provider = PriceLevelProvider(include_gamma=False)
        result = provider._compute_impl(sample_ohlcv_data)

        # Check some expected columns
        assert 'vwap' in result.columns
        assert 'rsi' in result.columns
        assert 'vwap_z' in result.columns

    def test_level_cols(self):
        """Test that level columns are defined."""
        provider = PriceLevelProvider()
        levels = provider.level_cols

        assert 'vwap' in levels
        assert 'ovn_lo' in levels
        assert 'ovn_hi' in levels

    def test_rth_levels_track_from_open_but_activate_after_opening_range(self):
        """RTH hi/lo should use 6:30+ extremes but only be active from 7:00 onward."""
        provider = PriceLevelProvider(include_gamma=False)

        dt = pd.date_range("2026-03-10 06:30:00", periods=36, freq="1min")
        n = len(dt)
        df = pd.DataFrame(
            {
                "dt": dt,
                "trading_day": ["2026-03-10"] * n,
                "open": np.full(n, 100.0),
                "high": np.full(n, 102.0),
                "low": np.full(n, 98.0),
                "close": np.full(n, 100.0),
                "volume": np.full(n, 1000.0),
                "bidvolume": np.full(n, 500.0),
                "askvolume": np.full(n, 500.0),
                "ovn": np.zeros(n, dtype=np.int8),
            }
        )
        # Opening-range extreme values.
        df.loc[df["dt"] == pd.Timestamp("2026-03-10 06:32:00"), "high"] = 110.0
        df.loc[df["dt"] == pd.Timestamp("2026-03-10 06:55:00"), "low"] = 90.0
        # Post-OR values should not replace opening-range extremes at first activation bar.
        df.loc[df["dt"] == pd.Timestamp("2026-03-10 07:00:00"), "high"] = 103.0
        df.loc[df["dt"] == pd.Timestamp("2026-03-10 07:00:00"), "low"] = 97.0

        out = provider._compute_impl(df)

        row_659 = out.loc[out["dt"] == pd.Timestamp("2026-03-10 06:59:00")].iloc[0]
        row_700 = out.loc[out["dt"] == pd.Timestamp("2026-03-10 07:00:00")].iloc[0]

        # Not active before OR completion.
        assert row_659["rth_hi"] == 0.0
        assert row_659["rth_lo"] == 0.0
        # Active at 7:00 with cumulative extremes from 6:30 onward.
        assert row_700["rth_hi"] == 110.0
        assert row_700["rth_lo"] == 90.0

    def test_ofi_z_is_finite_with_flat_orderflow(self):
        provider = PriceLevelProvider(include_gamma=False)
        dt = pd.date_range("2026-03-10 06:30:00", periods=80, freq="1min")
        n = len(dt)
        df = pd.DataFrame(
            {
                "dt": dt,
                "trading_day": ["2026-03-10"] * n,
                "open": np.full(n, 100.0),
                "high": np.full(n, 101.0),
                "low": np.full(n, 99.0),
                "close": np.full(n, 100.0),
                "volume": np.full(n, 1000.0),
                "bidvolume": np.full(n, 500.0),
                "askvolume": np.full(n, 500.0),
                "ovn": np.zeros(n, dtype=np.int8),
            }
        )
        out = provider._compute_impl(df)
        assert np.isfinite(out["ofi_z"].to_numpy()).all()
        assert (out["ofi_z"] == 0.0).all()


class TestFeatureProviderProtocol:
    """Tests for feature provider protocol."""

    def test_protocol_check(self):
        """Test that providers can be checked against protocol."""

        class ValidProvider:
            @property
            def name(self):
                return 'valid'

            @property
            def feature_names(self):
                return ['feat']

            def compute(self, ohlcv, context=None):
                return ohlcv

        provider = ValidProvider()
        assert isinstance(provider, FeatureProvider)

    def test_invalid_provider(self):
        """Test that invalid providers fail protocol check."""

        class InvalidProvider:
            pass

        provider = InvalidProvider()
        assert not isinstance(provider, FeatureProvider)
