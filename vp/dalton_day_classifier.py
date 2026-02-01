#!/usr/bin/env python3
"""
Dalton Day Type Classifier

A standalone pipeline for classifying ES futures trading days according to
James Dalton's "Markets in Profile" taxonomy. Trains offline on historical
data and provides an online per-minute prediction interface.

Day Types (7-class mode):
- TrendUp: Strong directional up move (narrow VA, POC near high)
- TrendDown: Strong directional down move (narrow VA, POC near low)
- Normal: Balanced single distribution (VA width 40-60% of range, POC centered)
- NormalVariation: Wider balanced day (VA width > 60%, single distribution)
- DoubleDistribution: Two value areas (2+ peaks with significant separation)
- PShape: Value at top (POC > 65% of range, responsive buying)
- BShape: Value at bottom (POC < 35% of range, responsive selling)

Simple 3-class mode:
- Trend: TrendUp, TrendDown
- Balance: Normal, NormalVariation, PShape, BShape
- Double: DoubleDistribution
"""

import argparse
import os
import math
import warnings
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib
from scipy import stats
from scipy.signal import find_peaks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not available, will use sklearn HistGradientBoostingClassifier")
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("torch not available, autoencoder profile encoding disabled")


# -----------------------------------------------------------------------------
# Day Type Enumeration
# -----------------------------------------------------------------------------
class DaltonDayType(Enum):
    """Enumeration of Dalton day types."""
    TREND_UP = "TrendUp"
    TREND_DOWN = "TrendDown"
    NORMAL = "Normal"
    NORMAL_VARIATION = "NormalVariation"
    DOUBLE_DISTRIBUTION = "DoubleDistribution"
    P_SHAPE = "PShape"
    B_SHAPE = "BShape"

    @classmethod
    def to_simple(cls, day_type: 'DaltonDayType') -> str:
        """Map 7-class to 3-class labels."""
        mapping = {
            cls.TREND_UP: "Trend",
            cls.TREND_DOWN: "Trend",
            cls.NORMAL: "Balance",
            cls.NORMAL_VARIATION: "Balance",
            cls.DOUBLE_DISTRIBUTION: "Double",
            cls.P_SHAPE: "Balance",
            cls.B_SHAPE: "Balance",
        }
        return mapping[day_type]

    @classmethod
    def to_binary(cls, day_type: 'DaltonDayType') -> str:
        """Map 7-class to 2-class labels (Balance vs Trend)."""
        mapping = {
            cls.TREND_UP: "Trend",
            cls.TREND_DOWN: "Trend",
            cls.NORMAL: "Balance",
            cls.NORMAL_VARIATION: "Balance",
            cls.DOUBLE_DISTRIBUTION: "Balance",
            cls.P_SHAPE: "Balance",
            cls.B_SHAPE: "Balance",
        }
        return mapping[day_type]

    @classmethod
    def all_labels(cls, simple: bool = False, binary: bool = False) -> List[str]:
        """Return all label names."""
        if binary:
            return ["Balance", "Trend"]
        if simple:
            return ["Trend", "Balance", "Double"]
        return [dt.value for dt in cls]


# -----------------------------------------------------------------------------
# Volume Profile Builder
# -----------------------------------------------------------------------------
class VolumeProfileBuilder:
    """Build and analyze volume-by-price profiles."""

    def __init__(self, bin_size: float = 0.5, kernel: Tuple[float, ...] = (0.2, 0.6, 0.2)):
        self.bin_size = bin_size
        self.kernel = kernel
        self.bin_centers: Optional[np.ndarray] = None
        self.per_minute_vbp: Optional[np.ndarray] = None

    def build_minute_vbp_matrix(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build per-minute VBP matrix.

        Args:
            prices: Array of prices (length T)
            volumes: Array of volumes (length T)

        Returns:
            bin_centers: Array of price bin centers
            per_minute_vbp: Matrix of shape (T, n_bins) with volume per price bin per minute
        """
        minp, maxp = prices.min() - 5, prices.max() + 5
        bins = np.arange(math.floor(minp), math.ceil(maxp) + self.bin_size, self.bin_size)
        bin_centers = bins[:-1] + self.bin_size / 2
        n_bins = len(bin_centers)
        T = len(prices)
        per_minute_vbp = np.zeros((T, n_bins), dtype=float)

        # Nearest-bin index per minute
        idxs = np.searchsorted(bins, prices, side='right') - 1
        idxs = np.clip(idxs, 0, n_bins - 1)

        # Distribute volume into center +/-1 using kernel
        offsets = [-1, 0, 1]
        for offset, weight in zip(offsets, self.kernel):
            idxs_off = np.clip(idxs + offset, 0, n_bins - 1)
            per_minute_vbp[np.arange(T), idxs_off] += volumes * weight

        self.bin_centers = bin_centers
        self.per_minute_vbp = per_minute_vbp
        return bin_centers, per_minute_vbp

    @staticmethod
    def compute_va70(vbp: np.ndarray, bin_centers: np.ndarray) -> Tuple[float, float]:
        """
        Compute 70% Value Area from VBP.

        Returns:
            (va_low, va_high) price levels
        """
        total = vbp.sum()
        if total <= 0:
            return float(bin_centers[0]), float(bin_centers[-1])

        poc_idx = int(np.argmax(vbp))
        cum = vbp[poc_idx]
        low, high = poc_idx, poc_idx
        target = 0.7 * total

        while cum < target:
            left = vbp[low - 1] if low - 1 >= 0 else -1
            right = vbp[high + 1] if high + 1 < len(vbp) else -1
            if left >= right:
                low -= 1
                cum += vbp[low]
            else:
                high += 1
                cum += vbp[high]
            if low == 0 and high == len(vbp) - 1:
                break

        return float(bin_centers[low]), float(bin_centers[high])

    @staticmethod
    def compute_poc(vbp: np.ndarray, bin_centers: np.ndarray) -> float:
        """Compute Point of Control (price with highest volume)."""
        if vbp.sum() <= 0:
            return float(bin_centers[len(bin_centers) // 2])
        return float(bin_centers[np.argmax(vbp)])

    @staticmethod
    def compute_entropy(vbp: np.ndarray) -> float:
        """Compute Shannon entropy of volume distribution."""
        s = vbp.sum()
        if s <= 0:
            return 0.0
        p = vbp / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def compute_skew(vbp: np.ndarray, bin_centers: np.ndarray) -> float:
        """Compute skewness of volume profile."""
        total = vbp.sum()
        if total <= 0:
            return 0.0
        # Weighted mean
        mean = np.sum(bin_centers * vbp) / total
        # Weighted std
        variance = np.sum(vbp * (bin_centers - mean) ** 2) / total
        std = np.sqrt(variance) if variance > 0 else 1.0
        # Weighted skew
        skew = np.sum(vbp * ((bin_centers - mean) / std) ** 3) / total if std > 0 else 0.0
        return float(skew)

    @staticmethod
    def find_peaks_in_profile(
        vbp: np.ndarray,
        min_prominence_ratio: float = 0.25,
        min_height_ratio: float = 0.15
    ) -> Tuple[int, float]:
        """
        Find significant peaks in volume profile.

        Args:
            vbp: Volume-by-price array
            min_prominence_ratio: Minimum prominence as fraction of max (default 0.25)
            min_height_ratio: Minimum height as fraction of max (default 0.15)

        Returns:
            n_peaks: Number of significant peaks
            peak_separation: Normalized distance between top 2 peaks (0-1)
        """
        if vbp.sum() <= 0:
            return 0, 0.0

        # Smooth profile more aggressively to reduce noise
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smoothed = np.convolve(vbp, kernel, mode='same')

        # Find peaks with stricter requirements
        max_val = smoothed.max()
        prominence_threshold = min_prominence_ratio * max_val
        height_threshold = min_height_ratio * max_val

        peaks, properties = find_peaks(
            smoothed,
            prominence=prominence_threshold,
            height=height_threshold,
            distance=max(3, len(vbp) // 20)  # Minimum distance between peaks
        )

        n_peaks = len(peaks)

        # Calculate separation between top 2 peaks
        peak_separation = 0.0
        if n_peaks >= 2:
            # Sort by prominence
            prominences = properties['prominences']
            sorted_indices = np.argsort(prominences)[::-1]
            top_peaks = peaks[sorted_indices[:2]]
            peak_separation = abs(top_peaks[0] - top_peaks[1]) / len(vbp)

        return n_peaks, peak_separation

    def normalize_profile_to_fixed_bins(
        self,
        vbp: np.ndarray,
        bin_centers: np.ndarray,
        n_output_bins: int = 20
    ) -> np.ndarray:
        """
        Resample VBP to fixed number of bins normalized to price range.

        Args:
            vbp: Volume-by-price array
            bin_centers: Price bin centers
            n_output_bins: Number of output bins

        Returns:
            Normalized histogram of shape (n_output_bins,)
        """
        if vbp.sum() <= 0:
            return np.zeros(n_output_bins)

        # Find active range (non-zero volume)
        nonzero_mask = vbp > 0
        if not nonzero_mask.any():
            return np.zeros(n_output_bins)

        active_indices = np.where(nonzero_mask)[0]
        low_idx, high_idx = active_indices[0], active_indices[-1]

        # Extract active portion
        active_vbp = vbp[low_idx:high_idx + 1]

        # Resample to fixed bins using linear interpolation
        if len(active_vbp) == 1:
            result = np.zeros(n_output_bins)
            result[n_output_bins // 2] = 1.0
            return result

        # Create interpolation
        x_old = np.linspace(0, 1, len(active_vbp))
        x_new = np.linspace(0, 1, n_output_bins)
        result = np.interp(x_new, x_old, active_vbp)

        # Normalize to sum to 1
        total = result.sum()
        if total > 0:
            result = result / total

        return result


# -----------------------------------------------------------------------------
# Profile Encoder (Histogram and Autoencoder)
# -----------------------------------------------------------------------------
class ProfileEncoder:
    """
    Encode volume profiles using fixed-size histogram or autoencoder.

    Supports two encoding methods:
    1. histogram: Resample to fixed bins, direct features
    2. autoencoder: Learn compressed latent representation
    """

    def __init__(
        self,
        method: str = 'histogram',
        n_bins: int = 20,
        latent_dim: int = 8
    ):
        """
        Initialize profile encoder.

        Args:
            method: 'histogram', 'autoencoder', or 'none'
            n_bins: Number of histogram bins (for histogram method)
            latent_dim: Latent dimension (for autoencoder method)
        """
        self.method = method
        self.n_bins = n_bins
        self.latent_dim = latent_dim
        self.vp_builder = VolumeProfileBuilder()

        # Autoencoder components (initialized during training)
        self.autoencoder = None
        self.ae_scaler = None
        self._is_trained = False

    def get_feature_names(self) -> List[str]:
        """Get feature names for the encoded profile."""
        if self.method == 'none':
            return []
        elif self.method == 'histogram':
            return [f'profile_bin_{i}' for i in range(self.n_bins)]
        elif self.method == 'autoencoder':
            return [f'profile_latent_{i}' for i in range(self.latent_dim)]
        else:
            return []

    def encode_profile(
        self,
        vbp: np.ndarray,
        bin_centers: np.ndarray
    ) -> np.ndarray:
        """
        Encode a single volume profile.

        Args:
            vbp: Volume-by-price array
            bin_centers: Price bin centers

        Returns:
            Encoded features array
        """
        if self.method == 'none':
            return np.array([])

        # Normalize to fixed bins
        normalized = self.vp_builder.normalize_profile_to_fixed_bins(
            vbp, bin_centers, self.n_bins
        )

        if self.method == 'histogram':
            return normalized

        elif self.method == 'autoencoder':
            if not self._is_trained:
                # Return zeros if not trained yet
                return np.zeros(self.latent_dim)

            # Scale and encode
            normalized_scaled = self.ae_scaler.transform(normalized.reshape(1, -1))
            with torch.no_grad():
                x = torch.FloatTensor(normalized_scaled)
                latent = self.autoencoder.encode(x)
            return latent.numpy().flatten()

        return np.array([])

    def fit_autoencoder(
        self,
        profiles: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = True
    ):
        """
        Train the autoencoder on a collection of normalized profiles.

        Args:
            profiles: Array of shape (n_samples, n_bins) with normalized profiles
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Print training progress
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for autoencoder encoding")

        if self.method != 'autoencoder':
            return

        # Scale profiles
        self.ae_scaler = StandardScaler()
        profiles_scaled = self.ae_scaler.fit_transform(profiles)

        # Build autoencoder
        self.autoencoder = ProfileAutoencoder(
            input_dim=self.n_bins,
            latent_dim=self.latent_dim
        )

        # Training
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        dataset = torch.FloatTensor(profiles_scaled)
        n_samples = len(dataset)

        self.autoencoder.train()
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch = dataset[batch_idx]

                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Autoencoder epoch {epoch + 1}/{epochs}, loss: {avg_loss:.6f}")

        self.autoencoder.eval()
        self._is_trained = True

        if verbose:
            # Compute reconstruction error
            with torch.no_grad():
                reconstructed = self.autoencoder(dataset)
                final_loss = criterion(reconstructed, dataset).item()
            print(f"  Autoencoder training complete. Final reconstruction loss: {final_loss:.6f}")

    def save(self, path: str):
        """Save encoder state."""
        state = {
            'method': self.method,
            'n_bins': self.n_bins,
            'latent_dim': self.latent_dim,
            'ae_scaler': self.ae_scaler,
            '_is_trained': self._is_trained
        }
        if self._is_trained and self.autoencoder is not None:
            state['autoencoder_state'] = self.autoencoder.state_dict()
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> 'ProfileEncoder':
        """Load encoder from saved state."""
        state = joblib.load(path)
        encoder = cls(
            method=state['method'],
            n_bins=state['n_bins'],
            latent_dim=state['latent_dim']
        )
        encoder.ae_scaler = state['ae_scaler']
        encoder._is_trained = state['_is_trained']

        if encoder._is_trained and 'autoencoder_state' in state:
            encoder.autoencoder = ProfileAutoencoder(
                input_dim=state['n_bins'],
                latent_dim=state['latent_dim']
            )
            encoder.autoencoder.load_state_dict(state['autoencoder_state'])
            encoder.autoencoder.eval()

        return encoder


# Autoencoder model (only defined if torch available)
if HAS_TORCH:
    class ProfileAutoencoder(nn.Module):
        """Simple autoencoder for volume profile compression."""

        def __init__(self, input_dim: int = 20, latent_dim: int = 8):
            super().__init__()
            self.input_dim = input_dim
            self.latent_dim = latent_dim

            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def encode(self, x):
            return self.encoder(x)

        def decode(self, z):
            return self.decoder(z)

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z)
else:
    # Placeholder when torch not available
    class ProfileAutoencoder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch required for ProfileAutoencoder")


# -----------------------------------------------------------------------------
# Day Labeler
# -----------------------------------------------------------------------------
class DayLabeler:
    """Label days using heuristic rules or GMM clustering."""

    def __init__(self, simple_labels: bool = False, binary_labels: bool = False,
                 early_trend_mode: bool = False, early_minutes: int = 90):
        self.simple_labels = simple_labels
        self.binary_labels = binary_labels
        self.early_trend_mode = early_trend_mode
        self.early_minutes = early_minutes
        self.vp_builder = VolumeProfileBuilder()

    def heuristic_label(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        open_price: Optional[float] = None,
        close_price: Optional[float] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> str:
        """
        Apply Dalton-style heuristic rules to label a day.

        Returns:
            Label string (7-class, 3-class, or 2-class depending on mode)
        """
        # Build VBP and compute metrics
        bin_centers, per_minute_vbp = self.vp_builder.build_minute_vbp_matrix(prices, volumes)
        total_vbp = per_minute_vbp.sum(axis=0)

        poc_price = self.vp_builder.compute_poc(total_vbp, bin_centers)
        va_low, va_high = self.vp_builder.compute_va70(total_vbp, bin_centers)
        va_width = va_high - va_low

        day_range = max(1.0, prices.max() - prices.min())
        poc_rel = (poc_price - prices.min()) / day_range
        va_width_rel = va_width / day_range

        n_peaks, peak_separation = self.vp_builder.find_peaks_in_profile(total_vbp)

        # Trend direction
        if open_price is None:
            open_price = prices[0]
        if close_price is None:
            close_price = prices[-1]
        close_vs_open = (close_price - open_price) / max(1.0, abs(open_price)) * 100

        # For early trend mode, label based on first N minutes only
        if self.early_trend_mode:
            return self._early_trend_label(prices, volumes, highs, lows)

        # For binary mode, use more aggressive trend detection
        if self.binary_labels:
            return self._binary_trend_label(
                prices, volumes, highs, lows, poc_rel, va_width_rel,
                close_vs_open, day_range, open_price, close_price
            )

        # Classification logic (7-class)
        day_type = self._apply_heuristic_rules(
            poc_rel, va_width_rel, n_peaks, peak_separation, close_vs_open
        )

        if self.simple_labels:
            return DaltonDayType.to_simple(day_type)
        return day_type.value

    def _early_trend_label(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: Optional[np.ndarray],
        lows: Optional[np.ndarray]
    ) -> str:
        """
        Label based on early (first N minutes) trend characteristics.

        This creates labels that are more predictable from early features,
        since we're predicting whether the early period shows trend behavior,
        not what the final day type will be.

        A day is labeled as "Trend" if the first N minutes show:
        - Strong directional movement (OTF)
        - Price staying consistently away from opening range
        - Narrow value area developing
        """
        T = len(prices)
        early_end = min(self.early_minutes, T)

        # Use only first N minutes for labeling
        early_prices = prices[:early_end]
        early_volumes = volumes[:early_end]
        early_highs = highs[:early_end] if highs is not None else early_prices
        early_lows = lows[:early_end] if lows is not None else early_prices

        # Compute early characteristics
        early_range = max(0.25, early_highs.max() - early_lows.min())
        open_price = early_prices[0]

        # One-time-framing in early period
        running_high = np.maximum.accumulate(early_highs)
        running_low = np.minimum.accumulate(early_lows)
        new_highs = np.sum(running_high[1:] > running_high[:-1])
        new_lows = np.sum(running_low[1:] < running_low[:-1])
        total_ext = new_highs + new_lows
        otf_ratio = abs(new_highs - new_lows) / max(1, total_ext)
        otf_direction = (new_highs - new_lows) / max(1, total_ext)

        # Build early VBP
        bc, vbp_matrix = self.vp_builder.build_minute_vbp_matrix(early_prices, early_volumes)
        early_vbp = vbp_matrix.sum(axis=0)
        poc_price = self.vp_builder.compute_poc(early_vbp, bc)
        va_low, va_high = self.vp_builder.compute_va70(early_vbp, bc)
        va_width_rel = (va_high - va_low) / early_range

        # POC position
        poc_rel = (poc_price - early_lows.min()) / early_range

        # Price movement from open
        close_early = early_prices[-1]
        move_from_open = (close_early - open_price) / early_range

        # Opening range (first 15 minutes) breakout
        or15_high = early_highs[:min(15, early_end)].max()
        or15_low = early_lows[:min(15, early_end)].min()
        or15_breakout = 0.0
        if close_early > or15_high:
            or15_breakout = (close_early - or15_high) / (or15_high - or15_low + 0.01)
        elif close_early < or15_low:
            or15_breakout = (close_early - or15_low) / (or15_high - or15_low + 0.01)

        # Early trend scoring
        trend_score = 0.0

        # Strong one-time-framing (directional conviction)
        if otf_ratio > 0.65:
            trend_score += 2.5
        elif otf_ratio > 0.50:
            trend_score += 1.5

        # Narrow early VA (price found value quickly)
        if va_width_rel < 0.40:
            trend_score += 2.0
        elif va_width_rel < 0.55:
            trend_score += 1.0

        # POC at extreme
        if poc_rel > 0.70 or poc_rel < 0.30:
            trend_score += 1.5
        elif poc_rel > 0.60 or poc_rel < 0.40:
            trend_score += 0.75

        # Strong OR breakout
        if abs(or15_breakout) > 1.0:
            trend_score += 2.0
        elif abs(or15_breakout) > 0.5:
            trend_score += 1.0

        # Significant move from open
        if abs(move_from_open) > 0.6:
            trend_score += 1.5
        elif abs(move_from_open) > 0.4:
            trend_score += 0.75

        # Threshold: 5.0+ for early trend
        if trend_score >= 5.0:
            return "Trend"
        return "Balance"

    def _binary_trend_label(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: Optional[np.ndarray],
        lows: Optional[np.ndarray],
        poc_rel: float,
        va_width_rel: float,
        close_vs_open: float,
        day_range: float,
        open_price: float,
        close_price: float
    ) -> str:
        """
        More aggressive trend detection for binary classification.

        A trend day is characterized by:
        - One-time framing (most bars make new highs OR new lows)
        - Price moving away from open and staying away
        - POC at extreme of range
        - Close near extreme
        """
        T = len(prices)

        # Compute one-time-framing score (OTF)
        # Count bars that make new highs vs new lows
        if highs is not None and lows is not None:
            running_high = np.maximum.accumulate(highs)
            running_low = np.minimum.accumulate(lows)
        else:
            running_high = np.maximum.accumulate(prices)
            running_low = np.minimum.accumulate(prices)

        new_highs = np.sum(running_high[1:] > running_high[:-1])
        new_lows = np.sum(running_low[1:] < running_low[:-1])

        # One-time-frame ratio: how directional was the day?
        total_extensions = new_highs + new_lows
        if total_extensions > 0:
            otf_ratio = abs(new_highs - new_lows) / total_extensions
        else:
            otf_ratio = 0.0

        # Close position relative to range
        close_rel = (close_price - prices.min()) / day_range if day_range > 0 else 0.5

        # Directional conviction: how far did price move from open?
        move_from_open = abs(close_price - open_price) / day_range if day_range > 0 else 0.0

        # Volume-weighted price position: is volume concentrated at one end?
        weighted_price = np.average(prices, weights=volumes)
        weighted_price_rel = (weighted_price - prices.min()) / day_range if day_range > 0 else 0.5

        # Trend scoring - multiple signals must align
        trend_score = 0.0

        # 1. Narrow value area (strong trend signal)
        if va_width_rel < 0.45:
            trend_score += 2.0
        elif va_width_rel < 0.55:
            trend_score += 1.0

        # 2. POC at extreme (price finding value at edge)
        if poc_rel > 0.70 or poc_rel < 0.30:
            trend_score += 2.0
        elif poc_rel > 0.60 or poc_rel < 0.40:
            trend_score += 1.0

        # 3. Close at extreme (trend didn't reverse)
        if close_rel > 0.75 or close_rel < 0.25:
            trend_score += 1.5
        elif close_rel > 0.65 or close_rel < 0.35:
            trend_score += 0.5

        # 4. Strong one-time-framing
        if otf_ratio > 0.70:
            trend_score += 2.0
        elif otf_ratio > 0.50:
            trend_score += 1.0

        # 5. Significant move from open
        if move_from_open > 0.70:
            trend_score += 1.5
        elif move_from_open > 0.50:
            trend_score += 0.75

        # 6. Close vs open consistency with POC
        # Up trend: positive close_vs_open AND high POC
        # Down trend: negative close_vs_open AND low POC
        if (close_vs_open > 0.3 and poc_rel > 0.60) or (close_vs_open < -0.3 and poc_rel < 0.40):
            trend_score += 1.5

        # Threshold for trend classification
        # Score >= 5.0 indicates strong trend characteristics
        if trend_score >= 5.0:
            return "Trend"
        return "Balance"

    def _apply_heuristic_rules(
        self,
        poc_rel: float,
        va_width_rel: float,
        n_peaks: int,
        peak_separation: float,
        close_vs_open: float
    ) -> DaltonDayType:
        """Apply the heuristic classification rules."""

        # Double Distribution: 2+ significant peaks with large separation
        # More strict: require peak_separation > 0.40 (was 0.25)
        if n_peaks >= 2 and peak_separation > 0.40:
            return DaltonDayType.DOUBLE_DISTRIBUTION

        # Trend days: narrow VA and strong directional move
        # Relaxed thresholds to catch more trend days
        trend_threshold = 0.45  # was 0.35
        if va_width_rel < trend_threshold:
            # TrendUp: POC in upper portion, positive close
            if poc_rel > 0.70 and close_vs_open > 0.2:
                return DaltonDayType.TREND_UP
            # TrendDown: POC in lower portion, negative close
            if poc_rel < 0.30 and close_vs_open < -0.2:
                return DaltonDayType.TREND_DOWN

        # P-Shape: Value concentrated at top (responsive buying)
        if poc_rel > 0.65 and va_width_rel < 0.55:
            return DaltonDayType.P_SHAPE

        # B-Shape: Value concentrated at bottom (responsive selling)
        if poc_rel < 0.35 and va_width_rel < 0.55:
            return DaltonDayType.B_SHAPE

        # Normal Variation: wider balanced day
        if va_width_rel >= 0.60:
            return DaltonDayType.NORMAL_VARIATION

        # Normal: balanced single distribution with POC centered
        # This is the default for most days
        return DaltonDayType.NORMAL

    def cluster_label(
        self,
        day_features_df: pd.DataFrame,
        n_clusters: int = 7,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Use GMM clustering to label days.

        Args:
            day_features_df: DataFrame with EOD features per day
            n_clusters: Number of clusters
            random_state: Random seed

        Returns:
            cluster_labels: Array of cluster indices
            cluster_probs: Array of cluster probabilities
            cluster_to_dalton: Mapping from cluster index to Dalton type name
        """
        feature_cols = ['poc_rel', 'va_width_rel', 'profile_entropy',
                        'profile_skew', 'n_peaks', 'close_vs_open']

        # Ensure all features exist
        available_cols = [c for c in feature_cols if c in day_features_df.columns]
        X = day_features_df[available_cols].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=random_state,
            n_init=5
        )
        cluster_labels = gmm.fit_predict(X_scaled)
        cluster_probs = gmm.predict_proba(X_scaled)

        # Interpret cluster centers to map to Dalton types
        cluster_to_dalton = self._interpret_cluster_centers(
            gmm.means_, available_cols, scaler
        )

        return cluster_labels, cluster_probs, cluster_to_dalton

    def _interpret_cluster_centers(
        self,
        centers_scaled: np.ndarray,
        feature_names: List[str],
        scaler: StandardScaler
    ) -> Dict[int, str]:
        """Map GMM cluster centers to Dalton day types."""
        # Inverse transform to original scale
        centers = scaler.inverse_transform(centers_scaled)

        mapping = {}
        used_types = set()

        for i, center in enumerate(centers):
            center_dict = dict(zip(feature_names, center))

            poc_rel = center_dict.get('poc_rel', 0.5)
            va_width_rel = center_dict.get('va_width_rel', 0.5)
            n_peaks = center_dict.get('n_peaks', 1)
            close_vs_open = center_dict.get('close_vs_open', 0)

            # Apply heuristic rules to cluster center
            day_type = self._apply_heuristic_rules(
                poc_rel, va_width_rel,
                int(round(n_peaks)), 0.3 if n_peaks >= 2 else 0.0,
                close_vs_open
            )

            # Avoid duplicate mappings
            type_name = day_type.value
            suffix = 1
            while type_name in used_types:
                type_name = f"{day_type.value}_{suffix}"
                suffix += 1

            mapping[i] = type_name
            used_types.add(type_name)

        return mapping


# -----------------------------------------------------------------------------
# Feature Extractor
# -----------------------------------------------------------------------------
class FeatureExtractor:
    """Extract features for day classification (EOD and per-minute)."""

    def __init__(
        self,
        bin_size: float = 0.5,
        profile_encoder: Optional[ProfileEncoder] = None
    ):
        self.bin_size = bin_size
        self.vp_builder = VolumeProfileBuilder(bin_size=bin_size)
        self.expected_volume_curve: Optional[np.ndarray] = None
        self.profile_encoder = profile_encoder

    def set_expected_volume_curve(self, curve: np.ndarray):
        """Set the expected cumulative volume curve for vol_rate calculation."""
        self.expected_volume_curve = curve

    def set_profile_encoder(self, encoder: ProfileEncoder):
        """Set the profile encoder for shape features."""
        self.profile_encoder = encoder

    @staticmethod
    def compute_daily_stats(
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute end-of-day stats for use as prior day context.
        """
        vp_builder = VolumeProfileBuilder()
        bin_centers, per_minute_vbp = vp_builder.build_minute_vbp_matrix(prices, volumes)
        total_vbp = per_minute_vbp.sum(axis=0)

        poc_price = vp_builder.compute_poc(total_vbp, bin_centers)
        va_low, va_high = vp_builder.compute_va70(total_vbp, bin_centers)

        if highs is not None and lows is not None:
            day_high = highs.max()
            day_low = lows.min()
        else:
            day_high = prices.max()
            day_low = prices.min()

        return {
            'close': float(prices[-1]),
            'open': float(prices[0]),
            'high': float(day_high),
            'low': float(day_low),
            'poc': float(poc_price),
            'va_high': float(va_high),
            'va_low': float(va_low),
            'range': float(day_high - day_low),
        }

    def extract_eod_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        bid_volumes: Optional[np.ndarray] = None,
        ask_volumes: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract end-of-day features for labeling/clustering.

        Returns:
            Dictionary of EOD features
        """
        bin_centers, per_minute_vbp = self.vp_builder.build_minute_vbp_matrix(prices, volumes)
        total_vbp = per_minute_vbp.sum(axis=0)

        poc_price = self.vp_builder.compute_poc(total_vbp, bin_centers)
        va_low, va_high = self.vp_builder.compute_va70(total_vbp, bin_centers)
        va_width = va_high - va_low

        day_range = max(1.0, prices.max() - prices.min())

        n_peaks, peak_separation = self.vp_builder.find_peaks_in_profile(total_vbp)

        features = {
            'poc_rel': (poc_price - prices.min()) / day_range,
            'va_width_rel': va_width / day_range,
            'profile_entropy': self.vp_builder.compute_entropy(total_vbp),
            'profile_skew': self.vp_builder.compute_skew(total_vbp, bin_centers),
            'n_peaks': n_peaks,
            'peak_separation': peak_separation,
            'close_vs_open': (prices[-1] - prices[0]) / max(1.0, abs(prices[0])) * 100,
            'day_range': day_range,
            'total_volume': volumes.sum(),
        }

        # Order flow features if available
        if bid_volumes is not None and ask_volumes is not None:
            total_vol = volumes.sum()
            delta = (ask_volumes - bid_volumes).sum()
            features['cum_delta'] = delta
            features['cum_delta_rate'] = delta / max(1.0, total_vol)

        return features

    def extract_per_minute_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        bid_volumes: Optional[np.ndarray] = None,
        ask_volumes: Optional[np.ndarray] = None,
        ib_minutes: int = 60,
        prior_day_stats: Optional[Dict[str, float]] = None,
        overnight_stats: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Extract per-minute features (online-safe, no future leakage).

        Args:
            prices: Close prices
            volumes: Volumes
            highs: High prices (optional)
            lows: Low prices (optional)
            bid_volumes: Bid volumes (optional)
            ask_volumes: Ask volumes (optional)
            ib_minutes: Number of minutes for Initial Balance (default 60)

        Returns:
            DataFrame with per-minute features
        """
        T = len(prices)
        bin_centers, per_minute_vbp = self.vp_builder.build_minute_vbp_matrix(prices, volumes)

        # Cumulative VBP
        cum_vbp = np.cumsum(per_minute_vbp, axis=0)

        # Cumulative volume and VWAP
        cum_vol = np.cumsum(volumes)
        cum_pv = np.cumsum(prices * volumes)
        vwap = cum_pv / np.maximum(1, cum_vol)

        # Day range (running)
        if highs is not None and lows is not None:
            running_high = np.maximum.accumulate(highs)
            running_low = np.minimum.accumulate(lows)
        else:
            running_high = np.maximum.accumulate(prices)
            running_low = np.minimum.accumulate(prices)
        running_range = np.maximum(1.0, running_high - running_low)

        # Initial Balance (first ib_minutes)
        if T > ib_minutes:
            if highs is not None and lows is not None:
                ib_high = highs[:ib_minutes].max()
                ib_low = lows[:ib_minutes].min()
            else:
                ib_high = prices[:ib_minutes].max()
                ib_low = prices[:ib_minutes].min()
        else:
            ib_high = running_high[-1]
            ib_low = running_low[-1]

        # Opening ranges (5, 15, 30 minutes)
        def get_opening_range(minutes):
            if T >= minutes:
                if highs is not None and lows is not None:
                    return highs[:minutes].max(), lows[:minutes].min()
                else:
                    return prices[:minutes].max(), prices[:minutes].min()
            return None, None

        or5_high, or5_low = get_opening_range(5)
        or15_high, or15_low = get_opening_range(15)
        or30_high, or30_low = get_opening_range(30)

        # Early trends (price change in first N minutes)
        open_price = prices[0]

        # Total day volume (for vol_share)
        total_day_vol = volumes.sum()

        # Prior day / overnight context features (constant for all minutes)
        open_price = prices[0]

        # Gap and prior day features
        gap_pct = 0.0
        open_vs_prior_poc = 0.0
        open_vs_prior_va_high = 0.0
        open_vs_prior_va_low = 0.0
        prior_day_range_pct = 0.0
        open_in_prior_va = 0.0
        gap_vs_prior_range = 0.0

        if prior_day_stats is not None:
            prior_close = prior_day_stats.get('close', open_price)
            prior_poc = prior_day_stats.get('poc', open_price)
            prior_va_high = prior_day_stats.get('va_high', open_price)
            prior_va_low = prior_day_stats.get('va_low', open_price)
            prior_range = prior_day_stats.get('range', 1.0)

            # Gap from prior close
            gap_pct = (open_price - prior_close) / max(1.0, abs(prior_close)) * 100

            # Open position relative to prior day levels
            if prior_range > 0:
                open_vs_prior_poc = (open_price - prior_poc) / prior_range
                open_vs_prior_va_high = (open_price - prior_va_high) / prior_range
                open_vs_prior_va_low = (open_price - prior_va_low) / prior_range
                gap_vs_prior_range = abs(open_price - prior_close) / prior_range

            # Prior day range as % of price
            prior_day_range_pct = prior_range / max(1.0, prior_close) * 100

            # Is open within prior VA?
            open_in_prior_va = 1.0 if prior_va_low <= open_price <= prior_va_high else 0.0

        # Overnight features
        overnight_range_pct = 0.0
        overnight_direction = 0.0

        if overnight_stats is not None:
            ovn_range = overnight_stats.get('range', 0.0)
            ovn_open = overnight_stats.get('open', open_price)
            ovn_close = overnight_stats.get('close', open_price)

            overnight_range_pct = ovn_range / max(1.0, abs(ovn_open)) * 100
            if ovn_open != 0:
                overnight_direction = (ovn_close - ovn_open) / max(1.0, abs(ovn_open)) * 100

        rows = []
        for t in range(T):
            vbp_t = cum_vbp[t]

            # POC and VA from cumulative profile
            poc_idx = int(np.argmax(vbp_t))
            poc_price = float(bin_centers[poc_idx])
            va_low, va_high = self.vp_builder.compute_va70(vbp_t, bin_centers)
            va_width = va_high - va_low

            # Relative metrics
            range_t = float(running_range[t])
            poc_rel = (poc_price - running_low[t]) / range_t if range_t > 0 else 0.5
            va_width_rel = va_width / range_t if range_t > 0 else 0.5

            # Entropy
            entropy_t = self.vp_builder.compute_entropy(vbp_t)

            # Peak count and separation
            n_peaks, peak_sep = self.vp_builder.find_peaks_in_profile(vbp_t)

            # Returns
            ret_1m = (prices[t] - prices[t - 1]) / prices[t - 1] if t >= 1 else 0.0
            ret_5m = (prices[t] - prices[t - 5]) / prices[t - 5] if t >= 5 else 0.0

            # IB extension (after IB is formed)
            ib_high_dist = 0.0
            ib_low_dist = 0.0
            range_ext_up = 0.0
            range_ext_down = 0.0
            ib_range = ib_high - ib_low if t >= ib_minutes else 1.0

            if t >= ib_minutes:
                ib_high_dist = (prices[t] - ib_high) / ib_range if ib_range > 0 else 0.0
                ib_low_dist = (ib_low - prices[t]) / ib_range if ib_range > 0 else 0.0
                range_ext_up = max(0, running_high[t] - ib_high) / ib_range if ib_range > 0 else 0.0
                range_ext_down = max(0, ib_low - running_low[t]) / ib_range if ib_range > 0 else 0.0

            # Volume rate vs expected
            vol_rate = 1.0
            if self.expected_volume_curve is not None and t < len(self.expected_volume_curve):
                expected = self.expected_volume_curve[t]
                vol_rate = cum_vol[t] / max(1.0, expected)

            # Order flow
            cum_delta_rate = 0.0
            if bid_volumes is not None and ask_volumes is not None:
                cum_ask = np.sum(ask_volumes[:t + 1])
                cum_bid = np.sum(bid_volumes[:t + 1])
                cum_delta_rate = (cum_ask - cum_bid) / max(1.0, cum_vol[t])

            # Opening range features (available after those minutes pass)
            current_range = float(running_range[t])

            # OR5 features
            or5_width_rel = 0.0
            price_vs_or5_mid = 0.0
            if t >= 5 and or5_high is not None:
                or5_width = or5_high - or5_low
                or5_width_rel = or5_width / current_range if current_range > 0 else 0.0
                or5_mid = (or5_high + or5_low) / 2
                price_vs_or5_mid = (prices[t] - or5_mid) / max(1.0, or5_width) if or5_width > 0 else 0.0

            # OR15 features
            or15_width_rel = 0.0
            price_vs_or15_mid = 0.0
            if t >= 15 and or15_high is not None:
                or15_width = or15_high - or15_low
                or15_width_rel = or15_width / current_range if current_range > 0 else 0.0
                or15_mid = (or15_high + or15_low) / 2
                price_vs_or15_mid = (prices[t] - or15_mid) / max(1.0, or15_width) if or15_width > 0 else 0.0

            # OR30 features
            or30_width_rel = 0.0
            price_vs_or30_mid = 0.0
            if t >= 30 and or30_high is not None:
                or30_width = or30_high - or30_low
                or30_width_rel = or30_width / current_range if current_range > 0 else 0.0
                or30_mid = (or30_high + or30_low) / 2
                price_vs_or30_mid = (prices[t] - or30_mid) / max(1.0, or30_width) if or30_width > 0 else 0.0

            # Early trend features (directional movement in first N minutes)
            early_trend_5 = (prices[min(t, 4)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 4 else 0.0
            early_trend_15 = (prices[min(t, 14)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 14 else 0.0
            early_trend_30 = (prices[min(t, 29)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 29 else 0.0

            # Range expansion rate (how fast is range growing)
            range_expansion_rate = 0.0
            if t >= 5:
                prev_range = float(running_range[t - 5])
                if prev_range > 0:
                    range_expansion_rate = (current_range - prev_range) / prev_range

            # === EARLY TREND DETECTION FEATURES ===

            # One-time-framing (OTF) score: directional conviction
            # Count new highs vs new lows up to this point
            otf_score = 0.0
            otf_direction = 0.0
            if t >= 5:
                new_highs_t = np.sum(running_high[1:t+1] > running_high[:t])
                new_lows_t = np.sum(running_low[1:t+1] < running_low[:t])
                total_ext = new_highs_t + new_lows_t
                if total_ext > 0:
                    otf_score = abs(new_highs_t - new_lows_t) / total_ext
                    otf_direction = (new_highs_t - new_lows_t) / total_ext

            # Price position in current range (early trend signal)
            price_in_range = (prices[t] - running_low[t]) / current_range if current_range > 0 else 0.5

            # Distance from open (trend strength)
            dist_from_open = (prices[t] - open_price) / current_range if current_range > 0 else 0.0

            # Momentum: rate of price change (velocity)
            momentum_5 = 0.0
            momentum_10 = 0.0
            if t >= 5:
                momentum_5 = (prices[t] - prices[t-5]) / current_range if current_range > 0 else 0.0
            if t >= 10:
                momentum_10 = (prices[t] - prices[t-10]) / current_range if current_range > 0 else 0.0

            # Acceleration: change in momentum
            acceleration = 0.0
            if t >= 10:
                mom_now = prices[t] - prices[t-5]
                mom_prev = prices[t-5] - prices[t-10]
                acceleration = (mom_now - mom_prev) / current_range if current_range > 0 else 0.0

            # Breakout strength from opening ranges
            or5_breakout = 0.0
            or15_breakout = 0.0
            or30_breakout = 0.0
            if t >= 5 and or5_high is not None:
                or5_range = or5_high - or5_low
                if or5_range > 0:
                    if prices[t] > or5_high:
                        or5_breakout = (prices[t] - or5_high) / or5_range
                    elif prices[t] < or5_low:
                        or5_breakout = (prices[t] - or5_low) / or5_range
            if t >= 15 and or15_high is not None:
                or15_range = or15_high - or15_low
                if or15_range > 0:
                    if prices[t] > or15_high:
                        or15_breakout = (prices[t] - or15_high) / or15_range
                    elif prices[t] < or15_low:
                        or15_breakout = (prices[t] - or15_low) / or15_range
            if t >= 30 and or30_high is not None:
                or30_range = or30_high - or30_low
                if or30_range > 0:
                    if prices[t] > or30_high:
                        or30_breakout = (prices[t] - or30_high) / or30_range
                    elif prices[t] < or30_low:
                        or30_breakout = (prices[t] - or30_low) / or30_range

            # Volume surge: current volume vs average so far
            avg_vol_so_far = cum_vol[t] / (t + 1) if t >= 0 else 1.0
            recent_vol = volumes[t] if t < len(volumes) else avg_vol_so_far
            vol_surge = recent_vol / max(1.0, avg_vol_so_far)

            # Price vs VWAP deviation (strong trend = price stays away from VWAP)
            vwap_deviation = abs(prices[t] - vwap[t]) / current_range if current_range > 0 else 0.0

            # Close relative to range (0 = at low, 1 = at high)
            close_position = price_in_range

            # === EARLY TREND SCORE ===
            # Compute a continuous 0-1 score of how "trend-like" the day looks
            # This can be used as a feature for EOD prediction
            early_trend_score = 0.0

            # Component 1: OTF score (0-2.5 points)
            if otf_score > 0.65:
                early_trend_score += 2.5
            elif otf_score > 0.50:
                early_trend_score += 1.5
            elif otf_score > 0.35:
                early_trend_score += 0.5

            # Component 2: VA width (0-2 points) - narrow = trend
            if va_width_rel < 0.40:
                early_trend_score += 2.0
            elif va_width_rel < 0.55:
                early_trend_score += 1.0
            elif va_width_rel < 0.70:
                early_trend_score += 0.5

            # Component 3: POC position (0-1.5 points) - at extreme = trend
            if poc_rel > 0.70 or poc_rel < 0.30:
                early_trend_score += 1.5
            elif poc_rel > 0.60 or poc_rel < 0.40:
                early_trend_score += 0.75

            # Component 4: OR breakout (0-2 points)
            max_breakout = max(abs(or5_breakout), abs(or15_breakout), abs(or30_breakout))
            if max_breakout > 1.0:
                early_trend_score += 2.0
            elif max_breakout > 0.5:
                early_trend_score += 1.0
            elif max_breakout > 0.25:
                early_trend_score += 0.5

            # Component 5: Distance from open (0-1.5 points)
            if abs(dist_from_open) > 0.6:
                early_trend_score += 1.5
            elif abs(dist_from_open) > 0.4:
                early_trend_score += 0.75
            elif abs(dist_from_open) > 0.2:
                early_trend_score += 0.25

            # Normalize to 0-1 range (max possible = 9.5)
            early_trend_score = min(1.0, early_trend_score / 9.5)

            row = {
                'minute': t,
                'minute_norm': t / max(1, T - 1),
                'price': float(prices[t]),
                'vwap': float(vwap[t]),
                'price_vs_vwap': (prices[t] - vwap[t]) / max(1.0, vwap[t]) * 100,
                'running_poc_rel': poc_rel,
                'running_va_width_rel': va_width_rel,
                'cum_entropy': entropy_t,
                'cum_peak_count': n_peaks,
                'cum_peak_separation': peak_sep,
                'ret_1m': ret_1m,
                'ret_5m': ret_5m,
                'ib_high_dist': ib_high_dist,
                'ib_low_dist': ib_low_dist,
                'range_ext_up': range_ext_up,
                'range_ext_down': range_ext_down,
                'vol_rate': vol_rate,
                'cum_vol': float(cum_vol[t]),
                'vol_share': float(cum_vol[t]) / max(1.0, total_day_vol),
                'cum_delta_rate': cum_delta_rate,
                # Early trend detection features
                'otf_score': otf_score,
                'otf_direction': otf_direction,
                'price_in_range': price_in_range,
                'dist_from_open': dist_from_open,
                'momentum_5': momentum_5,
                'momentum_10': momentum_10,
                'acceleration': acceleration,
                'or5_breakout': or5_breakout,
                'or15_breakout': or15_breakout,
                'or30_breakout': or30_breakout,
                'vol_surge': vol_surge,
                'vwap_deviation': vwap_deviation,
                'range_expansion_rate': range_expansion_rate,
                'early_trend_score': early_trend_score,
                # Prior day / overnight context (constant for all minutes)
                'gap_pct': gap_pct,
                'open_vs_prior_poc': open_vs_prior_poc,
                'open_vs_prior_va_high': open_vs_prior_va_high,
                'open_vs_prior_va_low': open_vs_prior_va_low,
                'prior_day_range_pct': prior_day_range_pct,
                'overnight_range_pct': overnight_range_pct,
                'overnight_direction': overnight_direction,
                'open_in_prior_va': open_in_prior_va,
                'gap_vs_prior_range': gap_vs_prior_range,
            }

            # Add profile encoding features if encoder is set
            if self.profile_encoder is not None and self.profile_encoder.method != 'none':
                profile_features = self.profile_encoder.encode_profile(vbp_t, bin_centers)
                feature_names = self.profile_encoder.get_feature_names()
                for name, val in zip(feature_names, profile_features):
                    row[name] = float(val)

            rows.append(row)

        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Dalton Classifier (XGBoost wrapper)
# -----------------------------------------------------------------------------
class DaltonClassifier:
    """XGBoost-based classifier for Dalton day types."""

    DEFAULT_FEATURE_COLS = [
        'minute_norm', 'price_vs_vwap', 'running_poc_rel', 'running_va_width_rel',
        'cum_entropy', 'cum_peak_count', 'cum_peak_separation', 'ret_1m', 'ret_5m',
        'ib_high_dist', 'ib_low_dist', 'range_ext_up', 'range_ext_down',
        'vol_rate', 'vol_share', 'cum_delta_rate',
        # Early trend detection features
        'otf_score', 'otf_direction', 'price_in_range', 'dist_from_open',
        'momentum_5', 'momentum_10', 'acceleration',
        'or5_breakout', 'or15_breakout', 'or30_breakout',
        'early_trend_score',  # Composite early trend indicator
        'vol_surge', 'vwap_deviation', 'range_expansion_rate'
    ]

    # Optimized features for binary (Balance vs Trend) classification
    # Focus on robust early signals that generalize well
    BINARY_FEATURE_COLS = [
        'minute_norm',
        # Composite early trend indicator (most important)
        'early_trend_score',
        # Core early trend signals
        'otf_score',           # Directional conviction
        'otf_direction',       # Direction of conviction
        'dist_from_open',      # How far from open
        'price_in_range',      # Where is price in current range
        # Opening range breakouts (key early trend signals)
        'or5_breakout',        # Early breakout
        'or15_breakout',       # Medium-term breakout
        # Volume profile shape (fundamental to Dalton)
        'running_poc_rel',     # POC position
        'running_va_width_rel', # Value area width
        # Price vs VWAP (institutional reference)
        'price_vs_vwap',
        'vwap_deviation',
        # Order flow
        'cum_delta_rate',
        # IB extensions (after 60 mins)
        'range_ext_up',
        'range_ext_down',
    ]

    def __init__(
        self,
        n_classes: int = 7,
        feature_cols: Optional[List[str]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        profile_encoder: Optional[ProfileEncoder] = None,
        binary_mode: bool = False
    ):
        self.n_classes = n_classes
        self.profile_encoder = profile_encoder
        self.binary_mode = binary_mode

        # Build feature list - use binary features for 2-class mode
        if feature_cols is not None:
            base_features = feature_cols
        elif binary_mode:
            base_features = self.BINARY_FEATURE_COLS.copy()
        else:
            base_features = self.DEFAULT_FEATURE_COLS.copy()

        if profile_encoder is not None:
            profile_feature_names = profile_encoder.get_feature_names()
            base_features = base_features + profile_feature_names
        self.feature_cols = base_features

        self.scaler = StandardScaler()
        self.model = None
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}

        # Use binary classification params for 2-class mode
        # Shallow trees with high regularization for better early-minute generalization
        if binary_mode:
            self.xgb_params = xgb_params or {
                'objective': 'binary:logistic',
                'max_depth': 3,  # Shallow trees generalize better
                'learning_rate': 0.02,
                'n_estimators': 1000,
                'min_child_weight': 20,  # More samples needed per leaf
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'reg_alpha': 1.0,  # Higher L1 regularization
                'reg_lambda': 5.0,  # Higher L2 regularization
                'gamma': 1.0,  # Min loss reduction for split
                'scale_pos_weight': 1.0,  # Will be set based on class imbalance
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        else:
            self.xgb_params = xgb_params or {
                'objective': 'multi:softprob',
                'num_class': n_classes,
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'min_child_weight': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels to integers."""
        unique_labels = sorted(labels.unique())
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        return np.array([self.label_encoder[l] for l in labels])

    def _decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        """Decode integer labels to strings."""
        return np.array([self.label_decoder[i] for i in encoded])

    def compute_sample_weights(
        self,
        y_train: np.ndarray,
        minute_norm: np.ndarray,
        early_weight: float = 1.0
    ) -> np.ndarray:
        """
        Compute sample weights combining class imbalance and early-minute weighting.

        Args:
            y_train: Encoded training labels
            minute_norm: Normalized minute values [0,1]
            early_weight: Factor for early-minute weighting (>= 1.0)

        Returns:
            Sample weights array
        """
        # Class imbalance weights (inverse frequency)
        unique, counts = np.unique(y_train, return_counts=True)
        class_weights = {u: len(y_train) / (len(unique) * c) for u, c in zip(unique, counts)}
        class_weight_arr = np.array([class_weights[y] for y in y_train])

        # Early-minute weights (quadratic decay)
        early_weight_arr = 1.0 + (early_weight - 1.0) * (1.0 - minute_norm) ** 2

        # Combine
        return class_weight_arr * early_weight_arr

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_weight: float = 1.0
    ):
        """
        Train the classifier.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_weight: Early-minute weighting factor
        """
        # Filter to available features
        available_features = [c for c in self.feature_cols if c in X_train.columns]
        self.feature_cols = available_features

        X_train_feat = X_train[self.feature_cols].copy()

        # Encode labels
        y_train_encoded = self._encode_labels(y_train)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_feat)

        # Compute sample weights
        minute_norm = X_train['minute_norm'].values if 'minute_norm' in X_train.columns else np.zeros(len(X_train))
        sample_weights = self.compute_sample_weights(y_train_encoded, minute_norm, early_weight)

        if HAS_XGB:
            # Update params based on actual classes
            if self.binary_mode:
                # Binary classification - remove num_class if present
                params = {k: v for k, v in self.xgb_params.items() if k != 'num_class'}
                # Set scale_pos_weight for class imbalance
                n_pos = np.sum(y_train_encoded == 1)
                n_neg = np.sum(y_train_encoded == 0)
                params['scale_pos_weight'] = n_neg / max(1, n_pos)
            else:
                params = self.xgb_params.copy()
                params['num_class'] = len(self.label_encoder)
            self.model = xgb.XGBClassifier(**params)

            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_feat = X_val[self.feature_cols]
                X_val_scaled = self.scaler.transform(X_val_feat)
                y_val_encoded = np.array([self.label_encoder.get(l, 0) for l in y_val])
                eval_set = [(X_val_scaled, y_val_encoded)]

            self.model.fit(
                X_train_scaled, y_train_encoded,
                sample_weight=sample_weights,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=500,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train_encoded, sample_weight=sample_weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict day types."""
        X_feat = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_feat)
        y_pred_encoded = self.model.predict(X_scaled)
        return self._decode_labels(y_pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict day type probabilities."""
        X_feat = X[self.feature_cols]
        X_scaled = self.scaler.transform(X_feat)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if HAS_XGB and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.zeros(len(self.feature_cols))

        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model and metadata."""
        artifact = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'label_encoder': self.label_encoder,
            'label_decoder': self.label_decoder,
            'n_classes': self.n_classes,
            'xgb_params': self.xgb_params,
            'has_profile_encoder': self.profile_encoder is not None
        }
        joblib.dump(artifact, path)

        # Save profile encoder separately if exists
        if self.profile_encoder is not None:
            encoder_path = path.replace('.joblib', '_profile_encoder.joblib')
            self.profile_encoder.save(encoder_path)

    @classmethod
    def load(cls, path: str) -> 'DaltonClassifier':
        """Load a saved classifier."""
        artifact = joblib.load(path)

        # Load profile encoder if it exists
        profile_encoder = None
        if artifact.get('has_profile_encoder', False):
            encoder_path = path.replace('.joblib', '_profile_encoder.joblib')
            if os.path.exists(encoder_path):
                profile_encoder = ProfileEncoder.load(encoder_path)

        # Filter out profile features from feature_cols for initialization
        # (they will be added back by the encoder)
        base_feature_cols = [c for c in artifact['feature_cols']
                            if not c.startswith('profile_bin_') and not c.startswith('profile_latent_')]

        instance = cls(
            n_classes=artifact['n_classes'],
            feature_cols=base_feature_cols,
            xgb_params=artifact['xgb_params'],
            profile_encoder=profile_encoder
        )
        instance.model = artifact['model']
        instance.scaler = artifact['scaler']
        instance.label_encoder = artifact['label_encoder']
        instance.label_decoder = artifact['label_decoder']
        # Override feature_cols with the saved full list
        instance.feature_cols = artifact['feature_cols']
        return instance


# -----------------------------------------------------------------------------
# Online Predictor
# -----------------------------------------------------------------------------
class OnlinePredictor:
    """Real-time incremental predictor for day type classification."""

    def __init__(self, model_path: str):
        """
        Initialize online predictor.

        Args:
            model_path: Path to saved DaltonClassifier
        """
        self.classifier = DaltonClassifier.load(model_path)

        # Initialize feature extractor with profile encoder from classifier
        self.feature_extractor = FeatureExtractor(
            profile_encoder=self.classifier.profile_encoder
        )

        # Load expected volume curve if available
        artifacts_dir = os.path.dirname(model_path)
        vol_curve_path = os.path.join(artifacts_dir, 'expected_volume_curve.npy')
        if os.path.exists(vol_curve_path):
            self.expected_volume_curve = np.load(vol_curve_path)
            self.feature_extractor.set_expected_volume_curve(self.expected_volume_curve)
        else:
            self.expected_volume_curve = None

        self.reset_day()

    def reset_day(self):
        """Reset state at start of each trading day."""
        self.prices = []
        self.highs = []
        self.lows = []
        self.volumes = []
        self.bid_volumes = []
        self.ask_volumes = []
        self.minute_count = 0

        # Initial balance tracking
        self.ib_high = None
        self.ib_low = None
        self.ib_minutes = 60

        # Cumulative state
        self.cum_vol = 0.0
        self.cum_pv = 0.0

    def update(
        self,
        ohlcv_bar: Dict[str, float]
    ) -> Tuple[str, np.ndarray]:
        """
        Update with new bar and return prediction.

        Args:
            ohlcv_bar: Dictionary with keys: open, high, low, close, volume
                       Optional: bid_volume, ask_volume

        Returns:
            predicted_type: Predicted day type string
            probabilities: Array of class probabilities
        """
        # Update state
        close = ohlcv_bar['close']
        high = ohlcv_bar.get('high', close)
        low = ohlcv_bar.get('low', close)
        volume = ohlcv_bar['volume']
        bid_vol = ohlcv_bar.get('bid_volume', volume / 2)
        ask_vol = ohlcv_bar.get('ask_volume', volume / 2)

        self.prices.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.bid_volumes.append(bid_vol)
        self.ask_volumes.append(ask_vol)

        self.cum_vol += volume
        self.cum_pv += close * volume

        # Update IB
        if self.minute_count < self.ib_minutes:
            if self.ib_high is None:
                self.ib_high = high
                self.ib_low = low
            else:
                self.ib_high = max(self.ib_high, high)
                self.ib_low = min(self.ib_low, low)

        self.minute_count += 1

        # Extract features
        features = self.get_current_features()

        # Predict
        X = pd.DataFrame([features])
        predicted_type = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]

        return predicted_type, probabilities

    def get_current_features(self) -> Dict[str, float]:
        """Get current feature values from accumulated state."""
        t = self.minute_count - 1
        T = max(390, self.minute_count)  # Assume 390 minute day

        prices = np.array(self.prices)
        volumes = np.array(self.volumes)
        highs = np.array(self.highs)
        lows = np.array(self.lows)
        bid_vols = np.array(self.bid_volumes)
        ask_vols = np.array(self.ask_volumes)

        # Running range
        running_high = highs.max()
        running_low = lows.min()
        running_range = max(1.0, running_high - running_low)

        # VWAP
        vwap = self.cum_pv / max(1.0, self.cum_vol)

        # Build cumulative VBP
        vp_builder = VolumeProfileBuilder()
        bin_centers, per_minute_vbp = vp_builder.build_minute_vbp_matrix(prices, volumes)
        cum_vbp = per_minute_vbp.sum(axis=0)

        # POC and VA
        poc_price = vp_builder.compute_poc(cum_vbp, bin_centers)
        va_low, va_high = vp_builder.compute_va70(cum_vbp, bin_centers)
        va_width = va_high - va_low

        poc_rel = (poc_price - running_low) / running_range
        va_width_rel = va_width / running_range

        # Entropy and peaks
        entropy = vp_builder.compute_entropy(cum_vbp)
        n_peaks, peak_sep = vp_builder.find_peaks_in_profile(cum_vbp)

        # Returns
        ret_1m = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0.0
        ret_5m = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0.0

        # IB metrics
        ib_range = (self.ib_high - self.ib_low) if self.ib_high and self.ib_low else 1.0
        ib_high_dist = 0.0
        ib_low_dist = 0.0
        range_ext_up = 0.0
        range_ext_down = 0.0

        if t >= self.ib_minutes and ib_range > 0:
            ib_high_dist = (prices[-1] - self.ib_high) / ib_range
            ib_low_dist = (self.ib_low - prices[-1]) / ib_range
            range_ext_up = max(0, running_high - self.ib_high) / ib_range
            range_ext_down = max(0, self.ib_low - running_low) / ib_range

        # Volume rate
        vol_rate = 1.0
        if self.expected_volume_curve is not None and t < len(self.expected_volume_curve):
            vol_rate = self.cum_vol / max(1.0, self.expected_volume_curve[t])

        # Delta rate
        cum_delta = ask_vols.sum() - bid_vols.sum()
        cum_delta_rate = cum_delta / max(1.0, self.cum_vol)

        # Opening range features
        open_price = prices[0]

        # OR5
        or5_width_rel = 0.0
        price_vs_or5_mid = 0.0
        if t >= 5:
            or5_high = max(highs[:5])
            or5_low = min(lows[:5])
            or5_width = or5_high - or5_low
            or5_width_rel = or5_width / running_range if running_range > 0 else 0.0
            or5_mid = (or5_high + or5_low) / 2
            price_vs_or5_mid = (prices[-1] - or5_mid) / max(1.0, or5_width) if or5_width > 0 else 0.0

        # OR15
        or15_width_rel = 0.0
        price_vs_or15_mid = 0.0
        if t >= 15:
            or15_high = max(highs[:15])
            or15_low = min(lows[:15])
            or15_width = or15_high - or15_low
            or15_width_rel = or15_width / running_range if running_range > 0 else 0.0
            or15_mid = (or15_high + or15_low) / 2
            price_vs_or15_mid = (prices[-1] - or15_mid) / max(1.0, or15_width) if or15_width > 0 else 0.0

        # OR30
        or30_width_rel = 0.0
        price_vs_or30_mid = 0.0
        if t >= 30:
            or30_high = max(highs[:30])
            or30_low = min(lows[:30])
            or30_width = or30_high - or30_low
            or30_width_rel = or30_width / running_range if running_range > 0 else 0.0
            or30_mid = (or30_high + or30_low) / 2
            price_vs_or30_mid = (prices[-1] - or30_mid) / max(1.0, or30_width) if or30_width > 0 else 0.0

        # Early trends
        early_trend_5 = (prices[min(t, 4)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 4 else 0.0
        early_trend_15 = (prices[min(t, 14)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 14 else 0.0
        early_trend_30 = (prices[min(t, 29)] - open_price) / max(1.0, abs(open_price)) * 100 if t >= 29 else 0.0

        # Range expansion rate
        range_expansion_rate = 0.0
        if t >= 5:
            prev_high = max(highs[:t-4])
            prev_low = min(lows[:t-4])
            prev_range = prev_high - prev_low
            if prev_range > 0:
                range_expansion_rate = (running_range - prev_range) / prev_range

        features = {
            'minute_norm': t / (T - 1),
            'price_vs_vwap': (prices[-1] - vwap) / max(1.0, vwap) * 100,
            'running_poc_rel': poc_rel,
            'running_va_width_rel': va_width_rel,
            'cum_entropy': entropy,
            'cum_peak_count': n_peaks,
            'cum_peak_separation': peak_sep,
            'ret_1m': ret_1m,
            'ret_5m': ret_5m,
            'ib_high_dist': ib_high_dist,
            'ib_low_dist': ib_low_dist,
            'range_ext_up': range_ext_up,
            'range_ext_down': range_ext_down,
            'vol_rate': vol_rate,
            'vol_share': self.cum_vol / max(1.0, volumes.sum() * T / self.minute_count),
            'cum_delta_rate': cum_delta_rate,
            # Early-day features
            'or5_width_rel': or5_width_rel,
            'or15_width_rel': or15_width_rel,
            'or30_width_rel': or30_width_rel,
            'price_vs_or5_mid': price_vs_or5_mid,
            'price_vs_or15_mid': price_vs_or15_mid,
            'price_vs_or30_mid': price_vs_or30_mid,
            'early_trend_5': early_trend_5,
            'early_trend_15': early_trend_15,
            'early_trend_30': early_trend_30,
            'range_expansion_rate': range_expansion_rate,
        }

        # Add profile encoding features if encoder is available
        profile_encoder = self.classifier.profile_encoder
        if profile_encoder is not None and profile_encoder.method != 'none':
            profile_features = profile_encoder.encode_profile(cum_vbp, bin_centers)
            feature_names = profile_encoder.get_feature_names()
            for name, val in zip(feature_names, profile_features):
                features[name] = float(val)

        return features


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
class Evaluator:
    """Evaluation utilities for the classifier."""

    @staticmethod
    def evaluate_per_minute_accuracy(
        df_test: pd.DataFrame,
        output_dir: str
    ) -> pd.DataFrame:
        """
        Compute and save per-minute accuracy curve.

        Args:
            df_test: Test DataFrame with 'minute', 'label', 'pred' columns
            output_dir: Directory to save outputs

        Returns:
            DataFrame with minute-level accuracy
        """
        acc_by_minute = df_test.groupby('minute').apply(
            lambda g: accuracy_score(g['label'], g['pred'])
        )
        acc_df = acc_by_minute.reset_index()
        acc_df.columns = ['minute', 'accuracy']

        acc_df.to_csv(os.path.join(output_dir, 'acc_by_minute.csv'), index=False)

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(acc_df['minute'], acc_df['accuracy'], linewidth=1.5)
        plt.xlabel('Minute of Day')
        plt.ylabel('Accuracy')
        plt.title('Per-Minute Classification Accuracy (Test Set)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='70% threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'acc_by_minute.png'), dpi=150)
        plt.close()

        return acc_df

    @staticmethod
    def evaluate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        output_dir: str
    ):
        """Generate and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (End of Day)')
        plt.colorbar()

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha='right')
        plt.yticks(tick_marks, labels)

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black')

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    @staticmethod
    def evaluate_feature_importance(
        classifier: DaltonClassifier,
        output_dir: str,
        top_n: int = 20
    ):
        """Plot feature importance."""
        importance_df = classifier.get_feature_importance()

        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
        plt.close()

        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

    @staticmethod
    def find_earliest_accuracy_threshold(
        acc_df: pd.DataFrame,
        thresholds: List[float] = [0.6, 0.7, 0.75, 0.8]
    ) -> Dict[float, Optional[int]]:
        """Find earliest minute where accuracy reaches each threshold."""
        results = {}
        for thresh in thresholds:
            above = acc_df[acc_df['accuracy'] >= thresh]
            if not above.empty:
                results[thresh] = int(above['minute'].min())
            else:
                results[thresh] = None
        return results

    @staticmethod
    def compare_labels(
        heuristic_labels: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_to_dalton: Dict[int, str],
        output_dir: str
    ):
        """Compare heuristic and cluster labels."""
        # Map cluster indices to names
        cluster_names = np.array([cluster_to_dalton[c] for c in cluster_labels])

        # Agreement rate
        agreement = (heuristic_labels == cluster_names).mean()

        # Confusion matrix
        unique_labels = sorted(set(heuristic_labels) | set(cluster_names))
        cm = confusion_matrix(heuristic_labels, cluster_names, labels=unique_labels)

        # Save comparison
        comparison_df = pd.DataFrame({
            'heuristic': heuristic_labels,
            'cluster': cluster_names,
            'agree': heuristic_labels == cluster_names
        })
        comparison_df.to_csv(os.path.join(output_dir, 'label_comparison.csv'), index=False)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Heuristic vs Cluster Labels (Agreement: {agreement:.1%})')
        plt.colorbar()

        tick_marks = np.arange(len(unique_labels))
        plt.xticks(tick_marks, unique_labels, rotation=45, ha='right')
        plt.yticks(tick_marks, unique_labels)

        plt.ylabel('Heuristic Label')
        plt.xlabel('Cluster Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_agreement.png'), dpi=150)
        plt.close()

        return agreement


# -----------------------------------------------------------------------------
# Pipeline Functions
# -----------------------------------------------------------------------------
def load_and_preprocess_data(
    input_path: str,
    rth_only: bool = True
) -> pd.DataFrame:
    """
    Load and preprocess minute bar data.

    Args:
        input_path: Path to CSV file
        rth_only: If True, filter to Regular Trading Hours only (ovn=0)

    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(input_path)

    # Filter to RTH if available
    if rth_only and 'ovn' in df.columns:
        df = df[df['ovn'] == 0].copy()

    # Ensure required columns
    required = ['Close', 'Volume']
    col_mapping = {
        'close': 'Close', 'CLOSE': 'Close',
        'volume': 'Volume', 'VOLUME': 'Volume',
    }
    for old, new in col_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Create trading_day if not present
    if 'trading_day' not in df.columns:
        if 'Date' in df.columns:
            df['trading_day'] = df['Date']
        elif 'date' in df.columns:
            df['trading_day'] = df['date']
        else:
            # Create from index
            df['trading_day'] = (df.index // 390).astype(int)

    return df


def build_dataset(
    df: pd.DataFrame,
    simple_labels: bool = False,
    binary_labels: bool = False,
    early_trend_mode: bool = False,
    early_minutes: int = 90,
    compare_labels_flag: bool = False,
    output_dir: Optional[str] = None,
    profile_encoding: str = 'none',
    profile_bins: int = 20,
    latent_dim: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[ProfileEncoder]]:
    """
    Build labeled dataset from minute bar data.

    Args:
        df: DataFrame with minute bar data
        simple_labels: Use 3-class labels
        binary_labels: Use 2-class labels (Balance/Trend)
        early_trend_mode: Label based on first N minutes trend characteristics
        early_minutes: Minutes for early trend labeling
        compare_labels_flag: Compare heuristic vs cluster labels
        output_dir: Output directory
        profile_encoding: 'none', 'histogram', or 'autoencoder'
        profile_bins: Number of histogram bins
        latent_dim: Autoencoder latent dimension

    Returns:
        samples_df: Per-minute samples with features and labels
        days_df: Per-day summary with labels
        profile_encoder: Trained profile encoder (or None)
    """
    labeler = DayLabeler(
        simple_labels=simple_labels,
        binary_labels=binary_labels,
        early_trend_mode=early_trend_mode,
        early_minutes=early_minutes
    )

    # Create profile encoder if needed
    profile_encoder = None
    if profile_encoding != 'none':
        profile_encoder = ProfileEncoder(
            method=profile_encoding,
            n_bins=profile_bins,
            latent_dim=latent_dim
        )

    feature_extractor = FeatureExtractor(profile_encoder=profile_encoder)

    # Group by trading day
    grouped = df.groupby('trading_day')

    # First pass: collect EOD profiles for autoencoder training
    eod_profiles = []
    day_data_cache = []

    print("  Pass 1: Extracting day data and EOD profiles...")
    for day_id, day_df in grouped:
        day_df = day_df.sort_values(by=['Time'] if 'Time' in day_df.columns else day_df.index.name or day_df.columns[0])

        prices = day_df['Close'].values.astype(float)
        volumes = day_df['Volume'].values.astype(float)

        # Optional columns
        highs = day_df['High'].values if 'High' in day_df.columns else None
        lows = day_df['Low'].values if 'Low' in day_df.columns else None
        bid_vols = day_df['BidVolume'].values if 'BidVolume' in day_df.columns else None
        ask_vols = day_df['AskVolume'].values if 'AskVolume' in day_df.columns else None

        # Skip days with insufficient data
        if len(prices) < 60:
            continue

        # Get day label
        open_price = prices[0]
        close_price = prices[-1]
        label = labeler.heuristic_label(prices, volumes, open_price, close_price, highs, lows)

        # Compute daily stats for prior day context
        daily_stats = FeatureExtractor.compute_daily_stats(prices, volumes, highs, lows)

        # Cache day data
        day_data_cache.append({
            'day_id': day_id,
            'prices': prices,
            'volumes': volumes,
            'highs': highs,
            'lows': lows,
            'bid_vols': bid_vols,
            'ask_vols': ask_vols,
            'label': label,
            'daily_stats': daily_stats
        })

        # Collect EOD profile for autoencoder training
        if profile_encoding == 'autoencoder':
            vp_builder = VolumeProfileBuilder()
            bin_centers, per_minute_vbp = vp_builder.build_minute_vbp_matrix(prices, volumes)
            total_vbp = per_minute_vbp.sum(axis=0)
            normalized = vp_builder.normalize_profile_to_fixed_bins(total_vbp, bin_centers, profile_bins)
            eod_profiles.append(normalized)

    # Train autoencoder if needed
    if profile_encoding == 'autoencoder' and len(eod_profiles) > 0:
        print(f"  Training autoencoder on {len(eod_profiles)} EOD profiles...")
        profiles_array = np.array(eod_profiles)
        profile_encoder.fit_autoencoder(profiles_array, epochs=100, verbose=True)
        # Update feature extractor with trained encoder
        feature_extractor.set_profile_encoder(profile_encoder)

    # Second pass: extract all features with prior day context
    samples_list = []
    days_list = []
    prior_day_stats = None  # Will be set after first day

    # Also try to get overnight data if available in original df
    has_overnight = 'ovn' in df.columns

    print("  Pass 2: Extracting per-minute features...")
    for i, day_data in enumerate(day_data_cache):
        prices = day_data['prices']
        volumes = day_data['volumes']
        highs = day_data['highs']
        lows = day_data['lows']
        bid_vols = day_data['bid_vols']
        ask_vols = day_data['ask_vols']
        label = day_data['label']
        day_id = day_data['day_id']
        current_day_stats = day_data['daily_stats']

        # Get overnight stats if available
        overnight_stats = None
        if has_overnight:
            ovn_df = df[(df['trading_day'] == day_id) & (df['ovn'] == 1)]
            if len(ovn_df) > 10:  # Need some overnight data
                ovn_prices = ovn_df['Close'].values.astype(float)
                ovn_volumes = ovn_df['Volume'].values.astype(float)
                ovn_highs = ovn_df['High'].values if 'High' in ovn_df.columns else None
                ovn_lows = ovn_df['Low'].values if 'Low' in ovn_df.columns else None
                overnight_stats = FeatureExtractor.compute_daily_stats(
                    ovn_prices, ovn_volumes, ovn_highs, ovn_lows
                )

        # Extract EOD features
        eod_features = feature_extractor.extract_eod_features(
            prices, volumes, bid_vols, ask_vols
        )
        eod_features['trading_day'] = day_id
        eod_features['label'] = label
        eod_features['n_minutes'] = len(prices)
        days_list.append(eod_features)

        # Extract per-minute features with prior day context
        minute_features = feature_extractor.extract_per_minute_features(
            prices, volumes, highs, lows, bid_vols, ask_vols,
            prior_day_stats=prior_day_stats,
            overnight_stats=overnight_stats
        )
        minute_features['trading_day'] = day_id
        minute_features['label'] = label
        samples_list.append(minute_features)

        # Update prior day stats for next iteration
        prior_day_stats = current_day_stats

    samples_df = pd.concat(samples_list, ignore_index=True)
    days_df = pd.DataFrame(days_list)

    # Compare heuristic vs cluster labels if requested
    if compare_labels_flag and output_dir:
        os.makedirs(output_dir, exist_ok=True)

        cluster_labels, cluster_probs, cluster_to_dalton = labeler.cluster_label(
            days_df, n_clusters=7 if not simple_labels else 3
        )

        agreement = Evaluator.compare_labels(
            days_df['label'].values,
            cluster_labels,
            cluster_to_dalton,
            output_dir
        )
        print(f"Label agreement (heuristic vs cluster): {agreement:.1%}")

    return samples_df, days_df, profile_encoder


def train_pipeline(
    samples_df: pd.DataFrame,
    days_df: pd.DataFrame,
    output_dir: str,
    simple_labels: bool = False,
    binary_labels: bool = False,
    early_weight: float = 1.0,
    seed: int = 42,
    profile_encoder: Optional[ProfileEncoder] = None
) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        samples_df: Per-minute samples
        days_df: Per-day summaries
        output_dir: Output directory
        simple_labels: Use 3-class labels
        binary_labels: Use 2-class labels (Balance/Trend)
        early_weight: Early-minute weighting factor
        seed: Random seed
        profile_encoder: Trained profile encoder (optional)

    Returns:
        Dictionary with model, metrics, and artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get label info
    labels = DaltonDayType.all_labels(simple=simple_labels, binary=binary_labels)
    n_classes = len(set(samples_df['label'].unique()))

    print(f"Training with {n_classes} classes: {sorted(samples_df['label'].unique())}")
    print(f"Total samples: {len(samples_df)}, unique days: {samples_df['trading_day'].nunique()}")

    # Report profile encoding mode
    if profile_encoder is not None:
        print(f"Profile encoding: {profile_encoder.method} (features: {len(profile_encoder.get_feature_names())})")
    else:
        print("Profile encoding: none")

    # Label distribution
    label_dist = samples_df.groupby('label').size() / len(samples_df)
    print("\nLabel distribution:")
    for label, pct in label_dist.items():
        print(f"  {label}: {pct:.1%}")

    # Split by day (60% train, 20% val, 20% test)
    groups = samples_df['trading_day']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(gss.split(samples_df, groups=groups))

    train_val_df = samples_df.iloc[train_val_idx]
    test_df = samples_df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['trading_day']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"\nSplit: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Days: train={train_df['trading_day'].nunique()}, val={val_df['trading_day'].nunique()}, test={test_df['trading_day'].nunique()}")

    # Train classifier
    classifier = DaltonClassifier(
        n_classes=n_classes,
        profile_encoder=profile_encoder,
        binary_mode=binary_labels
    )

    X_train = train_df.drop(columns=['label', 'trading_day'], errors='ignore')
    y_train = train_df['label']
    X_val = val_df.drop(columns=['label', 'trading_day'], errors='ignore')
    y_val = val_df['label']
    X_test = test_df.drop(columns=['label', 'trading_day'], errors='ignore')
    y_test = test_df['label']

    print("\nTraining classifier...")
    classifier.fit(X_train, y_train, X_val, y_val, early_weight=early_weight)

    # Predictions
    test_df = test_df.copy()
    test_df['pred'] = classifier.predict(X_test)
    test_df['pred_proba'] = classifier.predict_proba(X_test).max(axis=1)

    # Per-minute accuracy
    acc_df = Evaluator.evaluate_per_minute_accuracy(test_df, output_dir)

    # End-of-day evaluation
    eod_df = test_df.groupby('trading_day').last().reset_index()
    eod_acc = accuracy_score(eod_df['label'], eod_df['pred'])

    print(f"\nEnd-of-day accuracy: {eod_acc:.1%}")
    print("\nClassification report (EOD):")
    print(classification_report(eod_df['label'], eod_df['pred']))

    # Confusion matrix
    unique_labels = sorted(set(eod_df['label']) | set(eod_df['pred']))
    Evaluator.evaluate_confusion_matrix(
        eod_df['label'].values, eod_df['pred'].values, unique_labels, output_dir
    )

    # Feature importance
    Evaluator.evaluate_feature_importance(classifier, output_dir)

    # Earliest minute to threshold
    thresholds = Evaluator.find_earliest_accuracy_threshold(acc_df)
    print("\nEarliest minute to accuracy threshold:")
    for thresh, minute in thresholds.items():
        if minute is not None:
            print(f"  {thresh:.0%}: minute {minute}")
        else:
            print(f"  {thresh:.0%}: not reached")

    # Save model
    model_path = os.path.join(output_dir, 'dalton_classifier.joblib')
    classifier.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save expected volume curve (average cumulative volume by minute)
    vol_curve = train_df.groupby('minute')['cum_vol'].mean().values
    np.save(os.path.join(output_dir, 'expected_volume_curve.npy'), vol_curve)

    # Save feature list
    with open(os.path.join(output_dir, 'feature_cols.txt'), 'w') as f:
        f.write('\n'.join(classifier.feature_cols))

    return {
        'classifier': classifier,
        'acc_df': acc_df,
        'test_df': test_df,
        'eod_acc': eod_acc,
        'thresholds': thresholds
    }


def train_joint_pipeline(
    df: pd.DataFrame,
    output_dir: str,
    early_minutes: int = 60,
    early_weight: float = 3.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Joint training pipeline that optimizes early-trend and EOD models together.

    Approach:
    1. Build datasets with both early-trend and EOD labels
    2. Train early-trend model first
    3. Generate early-trend predictions for all samples
    4. Add early-trend predictions as features for EOD model
    5. Train EOD model with enhanced features
    6. Evaluate both models

    Args:
        df: Raw minute bar data
        output_dir: Output directory
        early_minutes: Minutes for early-trend labeling
        early_weight: Weight for early minutes
        seed: Random seed

    Returns:
        Dictionary with both models and evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("JOINT TRAINING PIPELINE")
    print("=" * 60)

    # Create labelers for both tasks
    early_labeler = DayLabeler(early_trend_mode=True, early_minutes=early_minutes)
    eod_labeler = DayLabeler(binary_labels=True)

    feature_extractor = FeatureExtractor()

    # Group by trading day
    grouped = df.groupby('trading_day')

    # Build dataset with both labels
    print("\n[Step 1] Building dataset with dual labels...")
    day_data_cache = []

    for day_id, day_df in grouped:
        day_df = day_df.sort_values(by=['Time'] if 'Time' in day_df.columns else day_df.columns[0])

        prices = day_df['Close'].values.astype(float)
        volumes = day_df['Volume'].values.astype(float)
        highs = day_df['High'].values if 'High' in day_df.columns else None
        lows = day_df['Low'].values if 'Low' in day_df.columns else None
        bid_vols = day_df['BidVolume'].values if 'BidVolume' in day_df.columns else None
        ask_vols = day_df['AskVolume'].values if 'AskVolume' in day_df.columns else None

        if len(prices) < early_minutes:
            continue

        # Get both labels
        early_label = early_labeler.heuristic_label(prices, volumes, highs=highs, lows=lows)
        eod_label = eod_labeler.heuristic_label(prices, volumes, highs=highs, lows=lows)

        day_data_cache.append({
            'day_id': day_id,
            'prices': prices,
            'volumes': volumes,
            'highs': highs,
            'lows': lows,
            'bid_vols': bid_vols,
            'ask_vols': ask_vols,
            'early_label': early_label,
            'eod_label': eod_label,
        })

    print(f"  Processed {len(day_data_cache)} days")

    # Extract per-minute features
    print("\n[Step 2] Extracting per-minute features...")
    samples_list = []

    for day_data in day_data_cache:
        feat_df = feature_extractor.extract_per_minute_features(
            day_data['prices'], day_data['volumes'],
            highs=day_data['highs'], lows=day_data['lows'],
            bid_volumes=day_data['bid_vols'], ask_volumes=day_data['ask_vols']
        )
        feat_df['trading_day'] = day_data['day_id']
        feat_df['early_label'] = day_data['early_label']
        feat_df['eod_label'] = day_data['eod_label']
        samples_list.append(feat_df)

    samples_df = pd.concat(samples_list, ignore_index=True)
    print(f"  Total samples: {len(samples_df)}")

    # Label distributions
    early_dist = samples_df.groupby('trading_day')['early_label'].first().value_counts(normalize=True)
    eod_dist = samples_df.groupby('trading_day')['eod_label'].first().value_counts(normalize=True)
    print(f"\n  Early-trend label distribution: {dict(early_dist.round(3))}")
    print(f"  EOD label distribution: {dict(eod_dist.round(3))}")

    # Split by day
    groups = samples_df['trading_day']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(gss.split(samples_df, groups=groups))

    train_val_df = samples_df.iloc[train_val_idx]
    test_df = samples_df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['trading_day']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"\n  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # =========================================================================
    # STAGE 1: Train Early-Trend Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Stage 1] Training Early-Trend Model")
    print("=" * 60)

    early_classifier = DaltonClassifier(n_classes=2, binary_mode=True)

    X_train_early = train_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    y_train_early = train_df['early_label']
    X_val_early = val_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    y_val_early = val_df['early_label']

    early_classifier.fit(X_train_early, y_train_early, X_val_early, y_val_early, early_weight=early_weight)

    # Evaluate early model
    X_test = test_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    test_df['early_pred'] = early_classifier.predict(X_test)
    early_proba = early_classifier.predict_proba(X_test)
    test_df['early_trend_prob'] = early_proba[:, 1] if early_proba.shape[1] > 1 else early_proba[:, 0]

    # Per-minute accuracy for early model
    early_acc_by_min = test_df.groupby('minute').apply(
        lambda g: accuracy_score(g['early_label'], g['early_pred'])
    )
    early_acc_df = early_acc_by_min.reset_index()
    early_acc_df.columns = ['minute', 'accuracy']

    early_thresholds = {}
    for thresh in [0.60, 0.70, 0.75, 0.80]:
        found = early_acc_df[early_acc_df['accuracy'] >= thresh]
        early_thresholds[thresh] = int(found['minute'].min()) if len(found) > 0 else None

    print("\nEarly-Trend Model Results:")
    print(f"  EOD accuracy: {accuracy_score(test_df.groupby('trading_day')['early_label'].first(), test_df.groupby('trading_day')['early_pred'].first()):.1%}")
    print("  Earliest minute to threshold:")
    for thresh, minute in early_thresholds.items():
        print(f"    {thresh:.0%}: minute {minute if minute else 'not reached'}")

    # =========================================================================
    # STAGE 2: Train EOD Model with Early-Trend Predictions as Features
    # =========================================================================
    print("\n" + "=" * 60)
    print("[Stage 2] Training EOD Model with Early-Trend Features")
    print("=" * 60)

    # Generate early-trend predictions for training data
    train_df = train_df.copy()
    val_df = val_df.copy()

    X_train_for_early = train_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    X_val_for_early = val_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')

    train_early_proba = early_classifier.predict_proba(X_train_for_early)
    val_early_proba = early_classifier.predict_proba(X_val_for_early)

    train_df['early_model_trend_prob'] = train_early_proba[:, 1] if train_early_proba.shape[1] > 1 else train_early_proba[:, 0]
    val_df['early_model_trend_prob'] = val_early_proba[:, 1] if val_early_proba.shape[1] > 1 else val_early_proba[:, 0]
    test_df['early_model_trend_prob'] = test_df['early_trend_prob']

    # Train EOD model with the new feature
    eod_classifier = DaltonClassifier(n_classes=2, binary_mode=True)

    # Add early_model_trend_prob to feature list
    eod_feature_cols = eod_classifier.BINARY_FEATURE_COLS.copy()
    eod_feature_cols.append('early_model_trend_prob')
    eod_classifier.feature_cols = eod_feature_cols

    X_train_eod = train_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    y_train_eod = train_df['eod_label']
    X_val_eod = val_df.drop(columns=['early_label', 'eod_label', 'trading_day'], errors='ignore')
    y_val_eod = val_df['eod_label']

    eod_classifier.fit(X_train_eod, y_train_eod, X_val_eod, y_val_eod, early_weight=early_weight)

    # Evaluate EOD model
    X_test_eod = test_df.drop(columns=['early_label', 'eod_label', 'trading_day', 'early_pred', 'early_trend_prob'], errors='ignore')
    test_df['eod_pred'] = eod_classifier.predict(X_test_eod)

    # Per-minute accuracy for EOD model
    eod_acc_by_min = test_df.groupby('minute').apply(
        lambda g: accuracy_score(g['eod_label'], g['eod_pred'])
    )
    eod_acc_df = eod_acc_by_min.reset_index()
    eod_acc_df.columns = ['minute', 'accuracy']

    eod_thresholds = {}
    for thresh in [0.60, 0.70, 0.75, 0.80]:
        found = eod_acc_df[eod_acc_df['accuracy'] >= thresh]
        eod_thresholds[thresh] = int(found['minute'].min()) if len(found) > 0 else None

    eod_test_summary = test_df.groupby('trading_day').last().reset_index()
    eod_acc = accuracy_score(eod_test_summary['eod_label'], eod_test_summary['eod_pred'])

    print("\nEOD Model Results (with early-trend features):")
    print(f"  EOD accuracy: {eod_acc:.1%}")
    print("  Earliest minute to threshold:")
    for thresh, minute in eod_thresholds.items():
        print(f"    {thresh:.0%}: minute {minute if minute else 'not reached'}")

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Saving Models and Results")
    print("=" * 60)

    # Save early model
    early_model_path = os.path.join(output_dir, 'early_trend_classifier.joblib')
    early_classifier.save(early_model_path)

    # Save EOD model
    eod_model_path = os.path.join(output_dir, 'eod_classifier.joblib')
    eod_classifier.save(eod_model_path)

    # Save accuracy curves
    early_acc_df.to_csv(os.path.join(output_dir, 'early_acc_by_minute.csv'), index=False)
    eod_acc_df.to_csv(os.path.join(output_dir, 'eod_acc_by_minute.csv'), index=False)

    # Plot combined accuracy curves
    plt.figure(figsize=(12, 5))
    plt.plot(early_acc_df['minute'], early_acc_df['accuracy'], label='Early-Trend Model', linewidth=2)
    plt.plot(eod_acc_df['minute'], eod_acc_df['accuracy'], label='EOD Model (joint)', linewidth=2)
    plt.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Minute of Day')
    plt.ylabel('Accuracy')
    plt.title('Joint Model Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_accuracy_curves.png'), dpi=150)
    plt.close()

    # Feature importance for EOD model
    eod_importance = eod_classifier.get_feature_importance()
    eod_importance.to_csv(os.path.join(output_dir, 'eod_feature_importance.csv'), index=False)

    print(f"\nSaved models to: {output_dir}")
    print(f"  - {early_model_path}")
    print(f"  - {eod_model_path}")

    # Print feature importance
    print("\nEOD Model Feature Importance (top 10):")
    for _, row in eod_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return {
        'early_classifier': early_classifier,
        'eod_classifier': eod_classifier,
        'early_acc_df': early_acc_df,
        'eod_acc_df': eod_acc_df,
        'early_thresholds': early_thresholds,
        'eod_thresholds': eod_thresholds,
        'test_df': test_df,
    }


def simulate_days(n_days: int = 200, seed: int = 123) -> pd.DataFrame:
    """Generate simulated minute bar data for testing."""
    rng = np.random.RandomState(seed)
    rows = []

    for d in range(n_days):
        # Choose day type
        r = rng.rand()
        if r < 0.15:
            day_type = 'TrendUp'
        elif r < 0.30:
            day_type = 'TrendDown'
        elif r < 0.45:
            day_type = 'Normal'
        elif r < 0.60:
            day_type = 'NormalVariation'
        elif r < 0.75:
            day_type = 'DoubleDistribution'
        elif r < 0.85:
            day_type = 'PShape'
        else:
            day_type = 'BShape'

        base = 4500 + rng.randn() * 50
        T = 390

        if day_type == 'TrendUp':
            drift = 0.03
            prices = base + np.cumsum(drift + rng.normal(scale=0.2, size=T))
        elif day_type == 'TrendDown':
            drift = -0.03
            prices = base + np.cumsum(drift + rng.normal(scale=0.2, size=T))
        elif day_type == 'Normal':
            prices = np.zeros(T)
            x = base
            for t in range(T):
                x += 0.3 * (base - x) + rng.normal(scale=0.3)
                prices[t] = x
        elif day_type == 'NormalVariation':
            prices = np.zeros(T)
            x = base
            for t in range(T):
                x += 0.1 * (base - x) + rng.normal(scale=0.6)
                prices[t] = x
        elif day_type == 'DoubleDistribution':
            split = T // 2 + rng.randint(-30, 30)
            low_center = base - 15
            high_center = base + 15
            prices = np.concatenate([
                rng.normal(low_center, 2, split),
                rng.normal(high_center, 2, T - split)
            ])
        elif day_type == 'PShape':
            # Value concentrated at top
            prices = np.zeros(T)
            x = base
            for t in range(T):
                target = base + 10 if t > T // 3 else base
                x += 0.2 * (target - x) + rng.normal(scale=0.3)
                prices[t] = x
        else:  # BShape
            # Value concentrated at bottom
            prices = np.zeros(T)
            x = base
            for t in range(T):
                target = base - 10 if t > T // 3 else base
                x += 0.2 * (target - x) + rng.normal(scale=0.3)
                prices[t] = x

        # Volume profile (U-shaped with some noise)
        base_vol = 500
        vol_mult = 1 + 0.5 * ((np.linspace(0, 1, T) - 0.5) ** 2)
        volumes = rng.poisson(base_vol * vol_mult) + 50

        for t in range(T):
            rows.append({
                'trading_day': d,
                'Time': f'{9 + t // 60:02d}:{t % 60:02d}:00',
                'Open': float(prices[t] - rng.rand()),
                'High': float(prices[t] + rng.rand() * 2),
                'Low': float(prices[t] - rng.rand() * 2),
                'Close': float(prices[t]),
                'Volume': int(volumes[t]),
                'BidVolume': int(volumes[t] * (0.4 + 0.2 * rng.rand())),
                'AskVolume': int(volumes[t] * (0.4 + 0.2 * rng.rand())),
                'ovn': 0
            })

    return pd.DataFrame(rows)


def run_analysis(output_dir: str):
    """Run analysis and hyperparameter experiments."""
    print("Analysis mode: Running experiments...")
    print("(This would include hyperparameter tuning, cross-validation analysis, etc.)")
    # Placeholder for future analysis code


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Dalton Day Type Classifier - Train and evaluate day type classification'
    )
    parser.add_argument(
        '--input', type=str,
        default='../raw_data/es_min_3y_clean_td_gamma.csv',
        help='Path to input CSV with minute bar data'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='vp/dalton_artifacts',
        help='Directory to save model and evaluation outputs'
    )
    parser.add_argument(
        '--simple-labels', action='store_true',
        help='Use 3-class labels (Trend/Balance/Double) instead of 7-class'
    )
    parser.add_argument(
        '--binary-labels', action='store_true',
        help='Use 2-class labels (Balance/Trend) for early detection focus'
    )
    parser.add_argument(
        '--early-trend', action='store_true',
        help='Label based on early (first 90 min) trend characteristics for better early prediction'
    )
    parser.add_argument(
        '--early-minutes', type=int, default=90,
        help='Minutes to use for early-trend labeling (default: 90)'
    )
    parser.add_argument(
        '--joint', action='store_true',
        help='Joint training: train early-trend model first, use its predictions for EOD model'
    )
    parser.add_argument(
        '--compare-labels', action='store_true',
        help='Compare heuristic vs GMM cluster labels'
    )
    parser.add_argument(
        '--early-weight', type=float, default=1.0,
        help='Weight factor for early minutes (>=1.0, higher favors early minutes)'
    )
    parser.add_argument(
        '--simulate', action='store_true',
        help='Use simulated data instead of loading from file'
    )
    parser.add_argument(
        '--days', type=int, default=200,
        help='Number of days to simulate (with --simulate)'
    )
    parser.add_argument(
        '--analyze', action='store_true',
        help='Run analysis/tuning experiments'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--profile-encoding', type=str, default='none',
        choices=['none', 'histogram', 'autoencoder'],
        help='Profile encoding method: none (default), histogram, or autoencoder'
    )
    parser.add_argument(
        '--profile-bins', type=int, default=20,
        help='Number of histogram bins for profile encoding (default: 20)'
    )
    parser.add_argument(
        '--latent-dim', type=int, default=8,
        help='Latent dimension for autoencoder encoding (default: 8)'
    )

    args = parser.parse_args()

    if args.analyze:
        run_analysis(args.output_dir)
        return

    # Load data
    if args.simulate:
        print(f"Simulating {args.days} days of minute bar data...")
        df = simulate_days(n_days=args.days, seed=args.seed)
    else:
        print(f"Loading data from: {args.input}")
        df = load_and_preprocess_data(args.input)

    print(f"Loaded {len(df)} bars")

    # Joint training mode
    if args.joint:
        results = train_joint_pipeline(
            df,
            args.output_dir,
            early_minutes=args.early_minutes,
            early_weight=args.early_weight,
            seed=args.seed
        )
        print(f"\nAll artifacts saved to: {args.output_dir}")
        return

    # Validate autoencoder requirements
    if args.profile_encoding == 'autoencoder' and not HAS_TORCH:
        print("ERROR: PyTorch required for autoencoder encoding. Install with: pip install torch")
        print("Falling back to histogram encoding.")
        args.profile_encoding = 'histogram'

    # Build dataset
    print(f"\nBuilding dataset (extracting features, labeling days)...")
    print(f"  Profile encoding: {args.profile_encoding}")
    if args.profile_encoding != 'none':
        print(f"  Profile bins: {args.profile_bins}")
        if args.profile_encoding == 'autoencoder':
            print(f"  Latent dim: {args.latent_dim}")

    samples_df, days_df, profile_encoder = build_dataset(
        df,
        simple_labels=args.simple_labels,
        binary_labels=args.binary_labels,
        early_trend_mode=args.early_trend,
        early_minutes=args.early_minutes,
        compare_labels_flag=args.compare_labels,
        output_dir=args.output_dir,
        profile_encoding=args.profile_encoding,
        profile_bins=args.profile_bins,
        latent_dim=args.latent_dim
    )

    # Train and evaluate
    # Use binary mode for early_trend as well (it's 2-class)
    use_binary = args.binary_labels or args.early_trend
    results = train_pipeline(
        samples_df,
        days_df,
        args.output_dir,
        simple_labels=args.simple_labels,
        binary_labels=use_binary,
        early_weight=args.early_weight,
        seed=args.seed,
        profile_encoder=profile_encoder
    )

    print(f"\nAll artifacts saved to: {args.output_dir}")


# Public API for imports
def load_and_predict(model_path: str) -> OnlinePredictor:
    """Load a trained model and return an OnlinePredictor instance."""
    return OnlinePredictor(model_path)


if __name__ == '__main__':
    main()
