"""Aggregate minute bars into daily OHLCV with optional VP metrics."""
import pandas as pd
import numpy as np
from scipy.stats import entropy as scipy_entropy


class DailyAggregator:
    """Converts minute-bar DataFrames to daily bars with VP shape metrics."""

    def __init__(self, vp_bin_size: float = 0.5):
        self.vp_bin_size = vp_bin_size

    def aggregate(
        self, minute_df: pd.DataFrame, compute_vp: bool = True
    ) -> pd.DataFrame:
        """Aggregate minute bars to daily OHLCV + optional VP metrics.

        Args:
            minute_df: DatetimeIndex, columns [open, high, low, close, volume, trading_day]
            compute_vp: Whether to compute VP shape features per day

        Returns:
            DataFrame indexed by trading_day (date), columns: open/high/low/close/volume + VP
        """
        grp = minute_df.groupby("trading_day")

        daily = pd.DataFrame({
            "open": grp["open"].first(),
            "high": grp["high"].max(),
            "low": grp["low"].min(),
            "close": grp["close"].last(),
            "volume": grp["volume"].sum(),
        })
        daily.index = pd.to_datetime(daily.index)
        daily = daily.sort_index()

        if compute_vp:
            vp_metrics = grp.apply(self._compute_vp_metrics)
            vp_df = pd.DataFrame(vp_metrics.tolist(), index=vp_metrics.index)
            vp_df.index = pd.to_datetime(vp_df.index)
            daily = daily.join(vp_df)

        return daily

    def _compute_vp_metrics(self, bars: pd.DataFrame) -> dict:
        """Compute VP shape metrics for a single day's minute bars."""
        prices = bars["close"].values
        volumes = bars["volume"].values.astype(float)

        if len(prices) < 5 or volumes.sum() == 0:
            return {
                "vp_poc_rel": 0.5, "vp_va_width_rel": 1.0,
                "vp_skew": 0.0, "vp_entropy": 0.0, "vp_n_peaks": 1,
            }

        day_high = bars["high"].max()
        day_low = bars["low"].min()
        day_range = day_high - day_low
        if day_range < 0.01:
            day_range = 1.0

        # Build VBP histogram
        bins = np.arange(
            np.floor(day_low) - 1,
            np.ceil(day_high) + 1 + self.vp_bin_size,
            self.vp_bin_size,
        )
        bin_centers = bins[:-1] + self.vp_bin_size / 2
        vbp = np.zeros(len(bin_centers))

        idxs = np.searchsorted(bins, prices, side="right") - 1
        idxs = np.clip(idxs, 0, len(bin_centers) - 1)
        np.add.at(vbp, idxs, volumes)

        total_vol = vbp.sum()
        if total_vol <= 0:
            return {
                "vp_poc_rel": 0.5, "vp_va_width_rel": 1.0,
                "vp_skew": 0.0, "vp_entropy": 0.0, "vp_n_peaks": 1,
            }

        # POC
        poc_idx = int(np.argmax(vbp))
        poc_price = bin_centers[poc_idx]
        poc_rel = (poc_price - day_low) / day_range

        # VA70
        va_lo, va_hi = self._compute_va70(vbp, bin_centers)
        va_width_rel = (va_hi - va_lo) / day_range

        # Skewness (volume-weighted)
        vbp_norm = vbp / total_vol
        mean_price = np.sum(vbp_norm * bin_centers)
        std_price = np.sqrt(np.sum(vbp_norm * (bin_centers - mean_price) ** 2))
        if std_price > 0:
            vp_skew = np.sum(vbp_norm * ((bin_centers - mean_price) / std_price) ** 3)
        else:
            vp_skew = 0.0

        # Entropy
        vbp_pos = vbp_norm[vbp_norm > 0]
        vp_entropy = float(scipy_entropy(vbp_pos)) if len(vbp_pos) > 1 else 0.0

        # Peak count
        n_peaks = self._count_peaks(vbp)

        return {
            "vp_poc_rel": float(np.clip(poc_rel, 0, 1)),
            "vp_va_width_rel": float(np.clip(va_width_rel, 0, 1)),
            "vp_skew": float(np.clip(vp_skew, -3, 3)),
            "vp_entropy": float(vp_entropy),
            "vp_n_peaks": int(n_peaks),
        }

    @staticmethod
    def _compute_va70(vbp: np.ndarray, bin_centers: np.ndarray) -> tuple:
        """Compute 70% value area bounds."""
        total = vbp.sum()
        if total <= 0:
            return float(bin_centers[0]), float(bin_centers[-1])

        poc_idx = int(np.argmax(vbp))
        cum = vbp[poc_idx]
        lo, hi = poc_idx, poc_idx
        target = 0.7 * total

        while cum < target:
            left_vol = vbp[lo - 1] if lo > 0 else -1
            right_vol = vbp[hi + 1] if hi < len(vbp) - 1 else -1
            if left_vol >= right_vol:
                lo -= 1
                cum += vbp[lo]
            else:
                hi += 1
                cum += vbp[hi]
            if lo == 0 and hi == len(vbp) - 1:
                break

        return float(bin_centers[lo]), float(bin_centers[hi])

    @staticmethod
    def _count_peaks(vbp: np.ndarray, min_prominence: float = 0.05) -> int:
        """Count local maxima in VBP."""
        if len(vbp) < 3:
            return 0
        total = vbp.sum()
        if total <= 0:
            return 0
        vbp_norm = vbp / total
        left = vbp_norm[:-2]
        center = vbp_norm[1:-1]
        right = vbp_norm[2:]
        is_peak = (center > left) & (center > right) & (center > min_prominence)
        return int(is_peak.sum())


def align_daily(
    daily_dfs: dict[str, pd.DataFrame],
    primary: str = "ES",
) -> dict[str, pd.DataFrame]:
    """Align multiple daily DataFrames to shared trading days.

    Uses inner join on dates, forward-fills up to 2 days for minor gaps.
    """
    if primary not in daily_dfs:
        raise ValueError(f"Primary instrument '{primary}' not in data")

    common_dates = daily_dfs[primary].index
    for sym, df in daily_dfs.items():
        if sym != primary:
            common_dates = common_dates.intersection(df.index)

    result = {}
    for sym, df in daily_dfs.items():
        # Reindex to primary dates, ffill up to 2 days for holidays
        aligned = df.reindex(daily_dfs[primary].index).ffill(limit=2)
        # Then restrict to common dates
        aligned = aligned.loc[common_dates]
        result[sym] = aligned

    print(f"Aligned {len(daily_dfs)} instruments to {len(common_dates)} common trading days")
    print(f"  Range: {common_dates.min().date()} – {common_dates.max().date()}")
    return result
