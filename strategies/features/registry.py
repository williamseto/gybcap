"""Feature provider registry and pipeline."""

from typing import Dict, List, Optional, Type, Any
import pandas as pd

from strategies.features.base import FeatureProvider


class FeatureRegistry:
    """
    Registry for feature providers.

    Allows dynamic registration and lookup of feature providers by name.
    """

    _providers: Dict[str, Type[FeatureProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a feature provider class.

        Usage:
            @FeatureRegistry.register('dalton')
            class DaltonFeatureProvider:
                ...
        """
        def decorator(provider_cls: Type[FeatureProvider]):
            cls._providers[name] = provider_cls
            return provider_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[FeatureProvider]:
        """Get a provider class by name."""
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise KeyError(f"Unknown provider: {name}. Available: {available}")
        return cls._providers[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> FeatureProvider:
        """Create an instance of a provider by name."""
        provider_cls = cls.get(name)
        return provider_cls(**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls):
        """Clear all registered providers (mainly for testing)."""
        cls._providers.clear()


class FeaturePipeline:
    """
    Pipeline for computing features from multiple providers.

    Combines features from multiple providers into a single DataFrame.
    """

    def __init__(self, providers: Optional[List[FeatureProvider]] = None):
        """
        Initialize pipeline.

        Args:
            providers: List of feature provider instances
        """
        self.providers: List[FeatureProvider] = providers or []

    def add_provider(self, provider: FeatureProvider) -> "FeaturePipeline":
        """Add a provider to the pipeline."""
        self.providers.append(provider)
        return self

    def add_by_name(self, name: str, **kwargs) -> "FeaturePipeline":
        """Add a provider by name from the registry."""
        provider = FeatureRegistry.create(name, **kwargs)
        return self.add_provider(provider)

    @classmethod
    def from_names(cls, names: List[str], **kwargs) -> "FeaturePipeline":
        """Create pipeline from list of provider names."""
        pipeline = cls()
        for name in names:
            # Extract provider-specific kwargs if available
            provider_kwargs = kwargs.get(name, {})
            pipeline.add_by_name(name, **provider_kwargs)
        return pipeline

    def compute(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute all features from all providers.

        Args:
            ohlcv: Input OHLCV DataFrame
            context: Optional context passed to each provider

        Returns:
            DataFrame with all computed features merged
        """
        if not self.providers:
            return ohlcv.copy()

        result = ohlcv.copy()

        for provider in self.providers:
            features = provider.compute(ohlcv, context)

            # Merge features into result
            for col in provider.feature_names:
                if col in features.columns:
                    result[col] = features[col].values

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names from all providers."""
        names = []
        for provider in self.providers:
            names.extend(provider.feature_names)
        return names

    def __repr__(self) -> str:
        provider_names = [p.name for p in self.providers]
        return f"FeaturePipeline(providers={provider_names})"
