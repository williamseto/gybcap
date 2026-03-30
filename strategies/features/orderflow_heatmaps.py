"""
Order flow footprint tensor extraction for deep model (Phase 4).

Builds 2D footprint tensors from raw per-second data for each near-level
1-min bar. These tensors capture the full bid/ask volume profile by price
and time, which is fundamentally richer than the scalar aggregations in
Phase 1.

Representation:
  Current bar:  (4, n_price_bins, 60)  — 4 channels × price bins × 60 seconds
  Context:      (4, n_price_bins, 300) — 5-min approach footprint

Channels: [total_volume, bid_volume, ask_volume, delta (bid-ask)]
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd


# Default config
N_PRICE_BINS = 20       # ±5pt around level in 0.5pt bins
PRICE_RANGE_PTS = 5.0   # ±5 points around level
TICK_SIZE = 0.25         # ES tick size
BIN_SIZE = 0.5           # Price bin size
CONTEXT_MINUTES = 5      # Multi-minute context window
N_CHANNELS = 4           # total_vol, bid_vol, ask_vol, delta


def build_footprint_tensor(
    sec_df: pd.DataFrame,
    center_price: float,
    n_price_bins: int = N_PRICE_BINS,
    price_range: float = PRICE_RANGE_PTS,
) -> np.ndarray:
    """Build a footprint tensor from per-second data for one time window.

    Args:
        sec_df: Per-second data for the window (must have dt, close, volume,
                bidvolume, askvolume columns).
        center_price: Price to center the bins around (typically the level price).
        n_price_bins: Number of price bins.
        price_range: ± range in points around center.

    Returns:
        Tensor of shape (4, n_price_bins, n_seconds) where:
          channel 0 = total volume
          channel 1 = bid volume
          channel 2 = ask volume
          channel 3 = delta (bid - ask)
    """
    n_seconds = len(sec_df)
    if n_seconds == 0:
        return np.zeros((N_CHANNELS, n_price_bins, 1), dtype=np.float32)

    tensor = np.zeros((N_CHANNELS, n_price_bins, n_seconds), dtype=np.float32)

    # Price bin edges
    bin_low = center_price - price_range
    bin_high = center_price + price_range
    bin_edges = np.linspace(bin_low, bin_high, n_price_bins + 1)

    close_px = sec_df["close"].values
    vol = sec_df["volume"].values.astype(np.float32)
    bid = sec_df["bidvolume"].values.astype(np.float32)
    ask = sec_df["askvolume"].values.astype(np.float32)
    delta = bid - ask

    # Digitize prices into bins
    bin_indices = np.digitize(close_px, bin_edges) - 1  # 0-indexed
    bin_indices = np.clip(bin_indices, 0, n_price_bins - 1)

    for t in range(n_seconds):
        b = bin_indices[t]
        tensor[0, b, t] = vol[t]
        tensor[1, b, t] = bid[t]
        tensor[2, b, t] = ask[t]
        tensor[3, b, t] = delta[t]

    return tensor


def extract_footprint_for_bar(
    sec_df: pd.DataFrame,
    bar_minute: pd.Timestamp,
    level_price: float,
    context_minutes: int = CONTEXT_MINUTES,
    n_price_bins: int = N_PRICE_BINS,
    price_range: float = PRICE_RANGE_PTS,
) -> dict:
    """Extract current-bar and context footprint tensors for one 1-min bar.

    Args:
        sec_df: Full per-second data for the day (must be sorted by dt).
        bar_minute: The minute timestamp of the bar.
        level_price: Nearest level price (center of price bins).
        context_minutes: Number of preceding minutes for context.
        n_price_bins: Number of price bins.
        price_range: ± range in points around level.

    Returns:
        dict with:
          'current': (4, n_price_bins, 60) — current bar footprint
          'context': (4, n_price_bins, context_minutes*60) — approach footprint
    """
    # Current bar: seconds within [bar_minute, bar_minute + 1min)
    bar_end = bar_minute + pd.Timedelta(minutes=1)
    current_mask = (sec_df["dt"] >= bar_minute) & (sec_df["dt"] < bar_end)
    current_sec = sec_df[current_mask]

    current_tensor = build_footprint_tensor(
        current_sec, level_price, n_price_bins, price_range
    )
    # Pad/truncate to exactly 60 seconds
    current_tensor = _pad_or_truncate(current_tensor, 60)

    # Context: preceding N minutes
    context_start = bar_minute - pd.Timedelta(minutes=context_minutes)
    context_mask = (sec_df["dt"] >= context_start) & (sec_df["dt"] < bar_minute)
    context_sec = sec_df[context_mask]

    context_tensor = build_footprint_tensor(
        context_sec, level_price, n_price_bins, price_range
    )
    target_len = context_minutes * 60
    context_tensor = _pad_or_truncate(context_tensor, target_len)

    return {"current": current_tensor, "context": context_tensor}


def _pad_or_truncate(tensor: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (left with zeros) or truncate a tensor to exact time dimension."""
    c, h, w = tensor.shape
    if w == target_len:
        return tensor
    elif w > target_len:
        return tensor[:, :, -target_len:]  # Keep most recent
    else:
        padded = np.zeros((c, h, target_len), dtype=np.float32)
        padded[:, :, -w:] = tensor  # Right-align (pad left with zeros)
        return padded


def extract_batch_footprints(
    sec_data_dir: str,
    sample_bars: pd.DataFrame,
    cache_path: Optional[str] = None,
    n_price_bins: int = N_PRICE_BINS,
    context_minutes: int = CONTEXT_MINUTES,
) -> dict:
    """Extract footprint tensors for a batch of near-level sample bars.

    Args:
        sec_data_dir: Directory containing quarterly CSV files.
        sample_bars: DataFrame with columns [dt, nearest_level_price, trading_day].
        cache_path: Optional path to save/load cached results.
        n_price_bins: Number of price bins.
        context_minutes: Preceding minutes for context.

    Returns:
        dict with:
          'current': (N, 4, n_price_bins, 60) — current bar footprints
          'context': (N, 4, n_price_bins, context_minutes*60) — approach footprints
          'valid_mask': (N,) bool — which samples had seconds data available
    """
    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached footprints from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from strategies.data.sec_features import load_quarter, list_quarter_files

    n_samples = len(sample_bars)
    current_all = np.zeros(
        (n_samples, N_CHANNELS, n_price_bins, 60), dtype=np.float32
    )
    context_all = np.zeros(
        (n_samples, N_CHANNELS, n_price_bins, context_minutes * 60),
        dtype=np.float32,
    )
    valid_mask = np.zeros(n_samples, dtype=bool)

    # Group samples by date for efficient loading
    sample_bars = sample_bars.copy()
    sample_bars["sample_pos"] = np.arange(n_samples, dtype=np.int64)
    sample_bars["date_key"] = pd.to_datetime(sample_bars["dt"]).dt.date

    # Build quarter lookup: date → quarter file
    quarter_files = list_quarter_files(sec_data_dir=sec_data_dir)
    date_to_quarter = {}
    for qf in quarter_files:
        # Parse quarter from filename: es_sec_2023q1.csv → (2023, 1)
        parts = qf.stem.replace("es_sec_", "")
        year = int(parts[:4])
        q = int(parts[5])
        # Quarter date ranges
        q_starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        q_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        import datetime

        start = datetime.date(year, *q_starts[q])
        end = datetime.date(year, *q_ends[q])
        d = start
        while d <= end:
            date_to_quarter[d] = qf
            d += datetime.timedelta(days=1)

    # Process by quarter to minimize file loading
    dates = sorted(sample_bars["date_key"].unique())
    loaded_quarters = {}  # path → sec_df

    n_processed = 0
    for date in dates:
        date_mask = sample_bars["date_key"] == date
        date_samples = sample_bars[date_mask]

        if date not in date_to_quarter:
            continue

        qf = date_to_quarter[date]

        # Load quarter if not already loaded
        if str(qf) not in loaded_quarters:
            # Clear previous to save memory
            loaded_quarters.clear()
            print(f"    Loading {qf.name} for footprint extraction...")
            sec_df = load_quarter(qf)
            loaded_quarters[str(qf)] = sec_df

        sec_df = loaded_quarters[str(qf)]

        # Filter to this date (plus some context from previous)
        context_start_time = pd.Timestamp(date) + pd.Timedelta(hours=6)  # 6 AM PT
        day_end_time = pd.Timestamp(date) + pd.Timedelta(hours=14)  # 2 PM PT
        day_sec = sec_df[
            (sec_df["dt"] >= context_start_time) & (sec_df["dt"] < day_end_time)
        ]

        if len(day_sec) == 0:
            continue

        for sample_row in date_samples.itertuples(index=False):
            pos = int(sample_row.sample_pos)

            bar_minute = pd.Timestamp(sample_row.dt).floor("min")
            level_price = sample_row.nearest_level_price
            if pd.isna(level_price):
                continue

            result = extract_footprint_for_bar(
                day_sec,
                bar_minute,
                level_price,
                context_minutes=context_minutes,
                n_price_bins=n_price_bins,
            )

            # Check if we got meaningful data
            if result["current"].sum() > 0:
                current_all[pos] = result["current"]
                context_all[pos] = result["context"]
                valid_mask[pos] = True

            n_processed += 1
            if n_processed % 1000 == 0:
                print(
                    f"    Processed {n_processed:,}/{n_samples:,} samples "
                    f"({100 * n_processed / n_samples:.1f}%)"
                )

    print(
        f"  Footprint extraction complete: "
        f"{valid_mask.sum():,}/{n_samples:,} valid "
        f"({100 * valid_mask.mean():.1f}%)"
    )

    result = {
        "current": current_all,
        "context": context_all,
        "valid_mask": valid_mask,
    }

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f, protocol=4)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Cached footprints to {cache_path} ({size_mb:.1f} MB)")

    return result
