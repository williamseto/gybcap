"""Base classes and protocols for feature providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class FeatureProvider(Protocol):
    """
    Protocol for feature providers.

    Feature providers compute derived features from OHLCV data
    that can be used for ML training or trade decisions.
    """

    @property
    def name(self) -> str:
        """Unique name for this provider."""
        ...

    @property
    def feature_names(self) -> List[str]:
        """List of feature column names produced by this provider."""
        ...

    def compute(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute features from OHLCV data.

        Args:
            ohlcv: DataFrame with open, high, low, close, volume columns
                   and additional columns like trading_day, dt, etc.
            context: Optional context for stateful providers

        Returns:
            DataFrame with computed feature columns, indexed to match ohlcv
        """
        ...


class BaseFeatureProvider(ABC):
    """Abstract base class implementing common feature provider functionality."""

    def __init__(self):
        self._cached_features: Optional[pd.DataFrame] = None
        self._cache_key: Optional[str] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this provider."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature column names produced by this provider."""
        pass

    @abstractmethod
    def _compute_impl(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Internal implementation of feature computation."""
        pass

    def compute(
        self,
        ohlcv: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compute features with optional caching.

        Override _compute_impl for custom computation logic.
        """
        return self._compute_impl(ohlcv, context)

    def validate_features(self, features: pd.DataFrame) -> None:
        """Validate computed features have expected columns."""
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(
                f"Provider {self.name} missing expected features: {missing}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
