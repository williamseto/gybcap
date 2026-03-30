"""Realtime strategy for deployable footprint fusion models."""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.features.orderflow_heatmaps import extract_footprint_for_bar
from strategies.realtime.orderflow_columns import normalize_orderflow_columns
from strategies.realtime.protocol import RealtimeSignal
from strategies.labeling.reversal_zones import TRACKED_LEVELS
from strategies.reversal.footprint_bundle import load_footprint_bundle
from strategies.reversal.realtime_strategy import ReversalPredictorStrategy

logger = logging.getLogger(__name__)


class FootprintRealtimeStrategy(ReversalPredictorStrategy):
    """Predict reversals using a saved footprint model bundle."""

    def __init__(
        self,
        model_dir: str,
        pred_threshold: Optional[float] = None,
        proximity_pts: Optional[float] = None,
        sec_window_sec: int = 900,
        device: str = "auto",
    ):
        self._model_dir = model_dir
        self._metadata = self._load_metadata(model_dir)
        self._pred_threshold = (
            float(pred_threshold)
            if pred_threshold is not None
            else float(self._metadata.get("threshold", 0.60))
        )
        self._proximity_pts = (
            float(proximity_pts)
            if proximity_pts is not None
            else float(self._metadata.get("proximity_pts", 5.0))
        )
        self._feature_cols: List[str] = list(self._metadata.get("feature_cols", []))
        md_tracked = self._metadata.get("tracked_levels", list(TRACKED_LEVELS))
        if not isinstance(md_tracked, list) or len(md_tracked) == 0:
            md_tracked = list(TRACKED_LEVELS)
        self._tracked_levels: List[str] = [str(x) for x in md_tracked]
        self._same_day_bidask_only = bool(self._metadata.get("same_day_bidask_only", False))
        self._side_hysteresis_pts = 0.0
        self._side_flip_confirm_pts = None
        self._mfe_regime_drive_move_pts = 8.0
        self._mfe_regime_drive_frac = 0.55
        self._score_feature_cols_required = set(self._feature_cols)
        self._policy_feature_cols: List[str] = []
        self._frontier_q_feature_cols: List[str] = []
        self._frontier_router_enabled = False
        self._frontier_virtual_gate_calibration_enabled = False

        self._device = self._resolve_device(device)
        self._model, self._metadata = load_footprint_bundle(
            model_dir=model_dir,
            device=self._device,
        )

        self._historical_ohlcv: Optional[pd.DataFrame] = None
        self._emitted: set[Tuple[str, str, str]] = set()
        self._last_processed_count = 0

        self._context_minutes = int(self._metadata.get("context_minutes", 5))
        self._n_price_bins = int(self._metadata.get("n_price_bins", 20))
        self._sec_window_sec = max(
            int(sec_window_sec),
            int(self._metadata.get("context_time_steps", 300) + 180),
        )
        self._sec_ticks: Deque[Dict] = deque(maxlen=max(2000, self._sec_window_sec * 4))

        logger.info(
            "FootprintRealtimeStrategy loaded: arch=%s, features=%d, threshold=%.2f, device=%s",
            self._metadata.get("model_arch", "cnn_fusion"),
            len(self._feature_cols),
            self._pred_threshold,
            self._device,
        )

    @staticmethod
    def _load_metadata(model_dir: str) -> Dict:
        import json
        import os

        path = os.path.join(model_dir, "metadata.json")
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def name(self) -> str:
        return "reversal_footprint"

    def reset_day(self) -> None:
        """Reset per-day dedup and bar counters."""
        self._emitted.clear()
        self._last_processed_count = 0

    def on_new_ticks(self, sec_ticks: pd.DataFrame) -> None:
        """Receive raw second-level ticks from the engine."""
        if sec_ticks is None or sec_ticks.empty:
            return

        sec_df = self._normalize_sec_ticks(sec_ticks)
        if sec_df.empty:
            return

        for row in sec_df.itertuples(index=False):
            self._sec_ticks.append(
                {
                    "dt": row.dt,
                    "close": float(row.close),
                    "volume": float(row.volume),
                    "bidvolume": float(row.bidvolume),
                    "askvolume": float(row.askvolume),
                }
            )

    def _normalize_sec_ticks(self, sec_ticks: pd.DataFrame) -> pd.DataFrame:
        df = normalize_orderflow_columns(
            sec_ticks.copy(),
            copy=False,
        )
        if "dt" in df.columns:
            dt = pd.to_datetime(df["dt"], errors="coerce")
            if hasattr(dt.dt, "tz") and dt.dt.tz is not None:
                dt = dt.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
        elif "timestamp" in df.columns:
            dt = (
                pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
                .dt.tz_convert("America/Los_Angeles")
                .dt.tz_localize(None)
            )
        else:
            return pd.DataFrame()

        close = None
        for c in ("close", "price", "open"):
            if c in df.columns:
                close = pd.to_numeric(df[c], errors="coerce")
                break
        if close is None:
            return pd.DataFrame()

        volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
        bid = pd.to_numeric(df.get("bidvolume", 0.0), errors="coerce").fillna(0.0)
        ask = pd.to_numeric(df.get("askvolume", 0.0), errors="coerce").fillna(0.0)

        out = pd.DataFrame(
            {
                "dt": dt,
                "close": close,
                "volume": volume,
                "bidvolume": bid,
                "askvolume": ask,
            }
        )
        out = out.dropna(subset=["dt", "close"]).sort_values("dt")
        return out

    def _sec_df(self) -> pd.DataFrame:
        if not self._sec_ticks:
            return pd.DataFrame(
                columns=["dt", "close", "volume", "bidvolume", "askvolume"]
            )
        sec_df = pd.DataFrame(list(self._sec_ticks))
        if sec_df.empty:
            return sec_df
        cutoff = sec_df["dt"].max() - pd.Timedelta(seconds=self._sec_window_sec)
        sec_df = sec_df[sec_df["dt"] >= cutoff].copy()
        return sec_df

    def _predict_and_emit(self, df: pd.DataFrame, bars_1m: pd.DataFrame) -> List[RealtimeSignal]:
        import torch

        near_mask = ~df["nearest_level_price"].isna()
        if not near_mask.any():
            return []

        sec_df = self._sec_df()
        if sec_df.empty:
            return []

        near_df = df.loc[near_mask].copy()
        signals: List[RealtimeSignal] = []

        X = pd.DataFrame(index=near_df.index)
        for col in self._feature_cols:
            if col in near_df.columns:
                X[col] = near_df[col].values
            else:
                X[col] = 0.0
        X = X.fillna(0.0).astype(np.float32)

        self._model.eval()
        with torch.no_grad():
            for i, (idx, row) in enumerate(near_df.iterrows()):
                level_name = str(row["nearest_level_name"])
                side = int(row["side"])
                direction = "bull" if side == 1 else "bear"

                if "dt" in row.index and pd.notna(row["dt"]):
                    bar_ts = pd.Timestamp(row["dt"])
                else:
                    if isinstance(bars_1m.index, pd.DatetimeIndex) and isinstance(idx, int):
                        bar_ts = pd.Timestamp(bars_1m.index[idx])
                    else:
                        continue

                if bar_ts.tzinfo is not None:
                    bar_ts_naive = bar_ts.tz_convert("America/Los_Angeles").tz_localize(None)
                else:
                    bar_ts_naive = bar_ts

                dedup_key = (level_name, direction, str(bar_ts_naive))
                if dedup_key in self._emitted:
                    continue

                level_price = float(row["nearest_level_price"])
                fp = extract_footprint_for_bar(
                    sec_df=sec_df,
                    bar_minute=bar_ts_naive.floor("min"),
                    level_price=level_price,
                    context_minutes=self._context_minutes,
                    n_price_bins=self._n_price_bins,
                )
                if fp["current"].sum() <= 0:
                    continue

                cur = torch.from_numpy(fp["current"]).unsqueeze(0).to(self._device)
                ctx = torch.from_numpy(fp["context"]).unsqueeze(0).to(self._device)
                sc = torch.from_numpy(X.iloc[i].values).unsqueeze(0).to(self._device)

                prob = float(self._model(cur, ctx, sc).item())
                if prob < self._pred_threshold:
                    continue

                self._emitted.add(dedup_key)
                signals.append(
                    RealtimeSignal(
                        strategy_name=self.name,
                        trigger_ts=bar_ts,
                        entry_ts=bar_ts,
                        entry_price=float(row["close"]),
                        direction=direction,
                        level_name=level_name,
                        level_value=level_price,
                        pred_proba=prob,
                        metadata={
                            "proximity": float(abs(row["close"] - level_price)),
                            "model_arch": self._metadata.get("model_arch", "cnn_fusion"),
                            "stop_pts": float(self._metadata.get("stop_pts", 4.0)),
                            "target_pts": float(self._metadata.get("target_pts", 6.0)),
                            "max_bars": int(self._metadata.get("max_bars", 45)),
                        },
                    )
                )

        return signals
