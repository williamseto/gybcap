"""Collect intraday SPX option chain snapshots via Schwab API.

Each call to :meth:`IntradayOptionsCollector.snapshot` fetches the full SPX
chain, flattens the Schwab ``callExpDateMap`` / ``putExpDateMap`` JSON into a
tidy DataFrame, and returns it along with the underlying quote.

Designed to be called every 15 minutes during RTH by the collection runner.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from strategies.data.schwab import (
    SchwabAuthConfig,
    SchwabClient,
    OptionChainRequest,
)


@dataclass
class SnapshotResult:
    """One point-in-time snapshot of the full options chain."""

    timestamp: dt.datetime
    underlying_price: float
    chain: pd.DataFrame  # flat table of all contracts


class IntradayOptionsCollector:
    """Fetches SPX option chain snapshots from Schwab."""

    def __init__(
        self,
        symbol: str = "$SPX",
        strike_count: int = 80,
        auth: Optional[SchwabAuthConfig] = None,
    ):
        self.symbol = symbol
        self.strike_count = strike_count
        self._auth = auth or SchwabAuthConfig.from_env()
        self._client = SchwabClient(self._auth)

    def snapshot(self) -> SnapshotResult:
        """Fetch one full chain snapshot right now.

        Returns:
            SnapshotResult with timestamp, underlying price, and flat chain df.
        """
        req = OptionChainRequest(
            symbol=self.symbol,
            contract_type="ALL",
            strike_count=self.strike_count,
            include_quotes=True,
            strategy="SINGLE",
        )
        raw = self._client.fetch_option_chain(req)
        ts = dt.datetime.now(dt.timezone.utc)

        underlying = raw.get("underlying", {})
        spot = underlying.get("mark") or underlying.get("last", 0.0)

        chain = self._flatten_chain(raw, ts)
        return SnapshotResult(timestamp=ts, underlying_price=float(spot), chain=chain)

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _flatten_chain(payload: dict, ts: dt.datetime) -> pd.DataFrame:
        """Flatten Schwab callExpDateMap/putExpDateMap into a single DataFrame.

        Extracts per-contract: strike, expiration, option_type, bid, ask, mark,
        iv, delta, gamma, theta, vega, open_interest, volume, dte.
        """
        rows: list[dict] = []

        for map_key, opt_type in [
            ("callExpDateMap", "C"),
            ("putExpDateMap", "P"),
        ]:
            exp_map = payload.get(map_key, {})
            for exp_key, strikes in exp_map.items():
                # exp_key looks like "2024-01-19:5" (date:dte)
                exp_date_str = exp_key.split(":")[0]
                for strike_str, contracts in strikes.items():
                    for c in contracts:
                        rows.append(
                            {
                                "timestamp": ts,
                                "option_type": opt_type,
                                "strike": float(strike_str),
                                "expiration": exp_date_str,
                                "bid": c.get("bid", np.nan),
                                "ask": c.get("ask", np.nan),
                                "mark": c.get("mark", np.nan),
                                "iv": c.get("volatility", np.nan),
                                "delta": c.get("delta", np.nan),
                                "gamma": c.get("gamma", np.nan),
                                "theta": c.get("theta", np.nan),
                                "vega": c.get("vega", np.nan),
                                "open_interest": c.get("openInterest", 0),
                                "volume": c.get("totalVolume", 0),
                                "dte": c.get("daysToExpiration", np.nan),
                            }
                        )

        df = pd.DataFrame(rows)

        if not df.empty:
            # Schwab reports IV as percentage; convert to decimal
            if df["iv"].median() > 1.0:
                df["iv"] = df["iv"] / 100.0

            df["expiration"] = pd.to_datetime(df["expiration"])

        return df
