"""Historical market-data synchronization utilities.

Main use case:
- Keep local minute-bar CSV current from Schwab API for ES and other symbols.
- Reuse same client for future option-chain/GEX snapshot ingestion.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategies.data.schwab import PriceHistoryRequest, SchwabClient

LA_TZ = "America/Los_Angeles"
LA_ZONE = ZoneInfo(LA_TZ)
UTC = dt.timezone.utc
DEFAULT_SCHWAB_MINUTE_CSV = "raw_data/schwab/es_minute_history.csv"


@dataclass
class MinuteHistorySyncConfig:
    """Config for syncing minute bars into a local CSV store."""

    symbol: str = "/ES"
    csv_path: str = DEFAULT_SCHWAB_MINUTE_CSV
    frequency_type: str = "minute"
    frequency: int = 1
    include_extended_hours: bool = True
    lookback_days_if_missing: int = 365
    stale_after_minutes: int = 30


@dataclass
class MinuteHistorySyncResult:
    """Result summary for a minute-history sync."""

    symbol: str
    csv_path: str
    rows_before: int
    rows_after: int
    rows_added: int
    candles_fetched: int
    latest_dt: Optional[str]
    status: str
    message: str


def _ensure_dt_index_like(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        ts = ts.dt.tz_convert(LA_TZ).dt.tz_localize(None)
    return ts


def _compute_ovn_flag(dt_series: pd.Series) -> pd.Series:
    hour = dt_series.dt.hour
    minute = dt_series.dt.minute
    ovn = (
        (hour < 6)
        | (hour > 13)
        | ((hour == 6) & (minute < 30))
    )
    return ovn.astype(int)


def _compute_trading_day(dt_series: pd.Series) -> pd.Series:
    trading_day = dt_series.dt.date.astype(str)
    rollover_mask = dt_series.dt.hour >= 15
    next_day = (dt_series + pd.Timedelta(days=1)).dt.date.astype(str)
    return trading_day.where(~rollover_mask, next_day)


def _canonicalize_existing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "BidVolume": "bidvolume",
        "AskVolume": "askvolume",
        "Date": "date",
        "Time": "time",
    }
    out = out.rename(columns=rename)
    out.columns = [str(c).strip() for c in out.columns]

    if "dt" not in out.columns:
        if "date" in out.columns and "time" in out.columns:
            out["dt"] = _ensure_dt_index_like(out["date"].astype(str) + " " + out["time"].astype(str))
        elif "datetime" in out.columns:
            out["dt"] = _ensure_dt_index_like(out["datetime"])

    if "dt" not in out.columns:
        raise ValueError("CSV must include Date+Time, dt, or datetime columns.")

    out["dt"] = _ensure_dt_index_like(out["dt"])

    for col in ("open", "high", "low", "close", "volume", "bidvolume", "askvolume"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "nearby_gamma_score" not in out.columns:
        out["nearby_gamma_score"] = 0.0
    out["nearby_gamma_score"] = pd.to_numeric(out["nearby_gamma_score"], errors="coerce").fillna(0.0)

    out = out.dropna(subset=["dt"])
    out = out.sort_values("dt")
    out = out.drop_duplicates(subset=["dt"], keep="last")
    return out.reset_index(drop=True)


def _candles_to_df(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])

    raw = pd.DataFrame(candles)
    if "datetime" not in raw.columns:
        return pd.DataFrame(columns=["dt", "open", "high", "low", "close", "volume"])

    # Schwab uses epoch milliseconds for candle timestamps.
    dt_local = (
        pd.to_datetime(raw["datetime"], unit="ms", utc=True, errors="coerce")
        .dt.tz_convert(LA_TZ)
        .dt.tz_localize(None)
    )

    out = pd.DataFrame(
        {
            "dt": dt_local,
            "open": pd.to_numeric(raw.get("open", np.nan), errors="coerce"),
            "high": pd.to_numeric(raw.get("high", np.nan), errors="coerce"),
            "low": pd.to_numeric(raw.get("low", np.nan), errors="coerce"),
            "close": pd.to_numeric(raw.get("close", np.nan), errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume", 0.0), errors="coerce").fillna(0.0),
        }
    )

    out = out.dropna(subset=["dt", "open", "high", "low", "close"])
    out = out.sort_values("dt")
    out = out.drop_duplicates(subset=["dt"], keep="last")
    out["bidvolume"] = 0.0
    out["askvolume"] = 0.0
    out["nearby_gamma_score"] = 0.0
    return out.reset_index(drop=True)


def _to_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dt"] = _ensure_dt_index_like(out["dt"])
    out = out.dropna(subset=["dt"])
    out = out.sort_values("dt")
    out = out.drop_duplicates(subset=["dt"], keep="last")

    for col in ("open", "high", "low", "close", "volume", "bidvolume", "askvolume", "nearby_gamma_score"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ovn"] = _compute_ovn_flag(out["dt"])
    out["trading_day"] = _compute_trading_day(out["dt"])

    out["Date"] = out["dt"].dt.strftime("%m/%d/%Y")
    out["Time"] = out["dt"].dt.strftime("%H:%M:%S")

    # Match existing CSV conventions used across training and playback code paths.
    ordered = pd.DataFrame(
        {
            "Date": out["Date"],
            "Time": out["Time"],
            "Open": out["open"].astype(float),
            "High": out["high"].astype(float),
            "Low": out["low"].astype(float),
            "Close": out["close"].astype(float),
            "Volume": out["volume"].fillna(0.0).astype(float),
            "BidVolume": out["bidvolume"].fillna(0.0).astype(float),
            "AskVolume": out["askvolume"].fillna(0.0).astype(float),
            "ovn": out["ovn"].astype(int),
            "trading_day": out["trading_day"],
            "nearby_gamma_score": out["nearby_gamma_score"].fillna(0.0).astype(float),
        }
    )

    return ordered.reset_index(drop=True)


def _load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["dt", "open", "high", "low", "close", "volume", "bidvolume", "askvolume", "nearby_gamma_score"]
        )

    df = pd.read_csv(path)
    return _canonicalize_existing(df)


def sync_minute_history(
    client: SchwabClient,
    cfg: MinuteHistorySyncConfig,
    now_utc: Optional[dt.datetime] = None,
) -> MinuteHistorySyncResult:
    """Sync local minute history CSV from Schwab price history endpoint."""
    csv_path = Path(cfg.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_csv(csv_path)
    rows_before = len(existing)

    now_utc = now_utc or dt.datetime.now(tz=UTC)
    end_dt_utc = pd.Timestamp(now_utc).tz_convert("UTC").floor("min")

    latest_dt_local = existing["dt"].max() if not existing.empty else None

    if latest_dt_local is not None:
        latest_dt_local = pd.Timestamp(latest_dt_local)
        latest_dt_aware = latest_dt_local.tz_localize(LA_TZ)
        now_local = end_dt_utc.to_pydatetime().astimezone(LA_ZONE)
        age = pd.Timestamp(now_local) - latest_dt_aware
        if age <= pd.Timedelta(minutes=cfg.stale_after_minutes):
            return MinuteHistorySyncResult(
                symbol=cfg.symbol,
                csv_path=str(csv_path),
                rows_before=rows_before,
                rows_after=rows_before,
                rows_added=0,
                candles_fetched=0,
                latest_dt=str(latest_dt_local),
                status="skipped",
                message="local history is fresh",
            )
        start_dt_utc = latest_dt_aware.tz_convert("UTC") + pd.Timedelta(minutes=cfg.frequency)
    else:
        start_dt_utc = end_dt_utc - pd.Timedelta(days=cfg.lookback_days_if_missing)

    if start_dt_utc >= end_dt_utc:
        return MinuteHistorySyncResult(
            symbol=cfg.symbol,
            csv_path=str(csv_path),
            rows_before=rows_before,
            rows_after=rows_before,
            rows_added=0,
            candles_fetched=0,
            latest_dt=str(latest_dt_local) if latest_dt_local is not None else None,
            status="skipped",
            message="no fetch window",
        )

    req = PriceHistoryRequest(
        symbol=cfg.symbol,
        start_ms=int(start_dt_utc.timestamp() * 1000),
        end_ms=int(end_dt_utc.timestamp() * 1000),
        frequency_type=cfg.frequency_type,
        frequency=cfg.frequency,
        need_extended_hours_data=cfg.include_extended_hours,
        need_previous_close=False,
    )
    candles = client.fetch_price_history(req)

    if not candles:
        latest = str(latest_dt_local) if latest_dt_local is not None else None
        return MinuteHistorySyncResult(
            symbol=cfg.symbol,
            csv_path=str(csv_path),
            rows_before=rows_before,
            rows_after=rows_before,
            rows_added=0,
            candles_fetched=0,
            latest_dt=latest,
            status="ok",
            message="no new candles returned",
        )

    fetched = _candles_to_df(candles)

    if existing.empty:
        merged = fetched.copy()
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)
    merged = merged.sort_values("dt")
    merged = merged.drop_duplicates(subset=["dt"], keep="last")
    merged = merged.reset_index(drop=True)

    output = _to_output_schema(merged)

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    output.to_csv(tmp, index=False)
    tmp.replace(csv_path)

    rows_after = len(output)
    rows_added = max(rows_after - rows_before, 0)
    latest = str(merged["dt"].max()) if not merged.empty else None

    return MinuteHistorySyncResult(
        symbol=cfg.symbol,
        csv_path=str(csv_path),
        rows_before=rows_before,
        rows_after=rows_after,
        rows_added=rows_added,
        candles_fetched=len(fetched),
        latest_dt=latest,
        status="ok",
        message="history synchronized",
    )
