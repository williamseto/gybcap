"""Schwab market-data client wrapper built on top of `schwab-py`.

This module keeps a small project-facing interface while delegating
OAuth/token lifecycle management to `schwab-py`.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from schwab import auth as schwab_auth
except Exception:  # pragma: no cover - optional dependency at runtime
    schwab_auth = None  # type: ignore[assignment]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_token_cache_path() -> str:
    """Resolve token cache path from env, supporting legacy variable naming."""
    cache = os.getenv("SCHWAB_TOKEN_CACHE", "").strip()
    if cache:
        return cache
    legacy = os.getenv("SCHWAB_TOKEN_PATH", "").strip()
    if legacy:
        return legacy
    return "~/.cache/gybcap/schwab_token.json"


class SchwabAPIError(RuntimeError):
    """Raised when Schwab API requests fail."""


@dataclass
class SchwabAuthConfig:
    """OAuth/client config for Schwab API."""

    app_key: str
    app_secret: str
    callback_url: str = "https://127.0.0.1:8182"
    token_cache_path: str = "~/.cache/gybcap/schwab_token.json"
    timeout_sec: float = 20.0
    interactive_login: bool = False
    manual_login: bool = False
    enforce_enums: bool = False
    max_token_age_sec: Optional[float] = None

    @classmethod
    def from_env(cls, allow_missing: bool = False) -> Optional["SchwabAuthConfig"]:
        """Load config from environment.

        Required:
        - SCHWAB_APP_KEY
        - SCHWAB_APP_SECRET

        Optional:
        - SCHWAB_CALLBACK_URL
        - SCHWAB_TOKEN_CACHE (preferred)
        - SCHWAB_TOKEN_PATH (legacy alias)
        - SCHWAB_TIMEOUT_SEC
        - SCHWAB_INTERACTIVE_LOGIN
        - SCHWAB_MANUAL_LOGIN
        - SCHWAB_ENFORCE_ENUMS
        - SCHWAB_MAX_TOKEN_AGE_SEC
        """
        app_key = os.getenv("SCHWAB_APP_KEY", "").strip()
        app_secret = os.getenv("SCHWAB_APP_SECRET", "").strip()

        if not app_key or not app_secret:
            if allow_missing:
                return None
            raise ValueError(
                "Missing Schwab credentials. Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET."
            )

        timeout_env = os.getenv("SCHWAB_TIMEOUT_SEC", "20").strip() or "20"
        timeout_sec = float(timeout_env)
        max_age_raw = os.getenv("SCHWAB_MAX_TOKEN_AGE_SEC", "").strip()
        max_age = float(max_age_raw) if max_age_raw else None

        return cls(
            app_key=app_key,
            app_secret=app_secret,
            callback_url=os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182").strip(),
            token_cache_path=_env_token_cache_path(),
            timeout_sec=timeout_sec,
            interactive_login=_env_bool("SCHWAB_INTERACTIVE_LOGIN", False),
            manual_login=_env_bool("SCHWAB_MANUAL_LOGIN", False),
            enforce_enums=_env_bool("SCHWAB_ENFORCE_ENUMS", False),
            max_token_age_sec=max_age,
        )


@dataclass
class PriceHistoryRequest:
    """Price history query request.

    Times are UNIX epoch milliseconds in Schwab API parameters.
    """

    symbol: str
    start_ms: int
    end_ms: int
    frequency_type: str = "minute"
    frequency: int = 1
    need_extended_hours_data: bool = True
    need_previous_close: bool = False


@dataclass
class OptionChainRequest:
    """Option chain query request."""

    symbol: str
    contract_type: str = "ALL"
    strike_count: int = 50
    include_quotes: bool = True
    strategy: str = "SINGLE"


class SchwabClient:
    """Project wrapper around `schwab-py` client."""

    def __init__(self, auth: SchwabAuthConfig):
        self._auth = auth
        self._client = self._build_client()

    @property
    def auth(self) -> SchwabAuthConfig:
        return self._auth

    @property
    def token_cache_path(self) -> Path:
        return Path(self._auth.token_cache_path).expanduser()

    def _build_client(self):
        if schwab_auth is None:
            raise SchwabAPIError(
                "schwab-py is not installed. Install `schwab-py` in your environment."
            )

        path = self.token_cache_path
        path.parent.mkdir(parents=True, exist_ok=True)

        kwargs: Dict[str, Any] = {
            "api_key": self._auth.app_key,
            "app_secret": self._auth.app_secret,
            "enforce_enums": self._auth.enforce_enums,
        }

        if path.exists():
            client = schwab_auth.client_from_token_file(
                token_path=str(path),
                **kwargs,
            )
        else:
            if not self._auth.interactive_login:
                raise SchwabAPIError(
                    f"Token file not found at {path}. "
                    "Run `scripts/sync_schwab_history.py init-token` once, "
                    "or set SCHWAB_INTERACTIVE_LOGIN=1 for interactive token creation."
                )

            if self._auth.manual_login:
                client = schwab_auth.client_from_manual_flow(
                    callback_url=self._auth.callback_url,
                    token_path=str(path),
                    **kwargs,
                )
            else:
                easy_kwargs = dict(
                    callback_url=self._auth.callback_url,
                    token_path=str(path),
                    interactive=True,
                    **kwargs,
                )
                if self._auth.max_token_age_sec is not None:
                    easy_kwargs["max_token_age"] = self._auth.max_token_age_sec
                client = schwab_auth.easy_client(**easy_kwargs)

        try:
            client.set_timeout(self._auth.timeout_sec)
        except Exception:
            # Timeout setting is best-effort.
            pass
        return client

    # ------------------------------------------------------------------
    # Market-data endpoints
    # ------------------------------------------------------------------

    def fetch_price_history(self, req: PriceHistoryRequest) -> List[Dict[str, Any]]:
        start_dt = dt.datetime.fromtimestamp(req.start_ms / 1000.0, tz=dt.timezone.utc)
        end_dt = dt.datetime.fromtimestamp(req.end_ms / 1000.0, tz=dt.timezone.utc)

        try:
            resp = self._client.get_price_history(
                req.symbol,
                start_datetime=start_dt,
                end_datetime=end_dt,
                frequency_type=req.frequency_type,
                frequency=int(req.frequency),
                need_extended_hours_data=bool(req.need_extended_hours_data),
                need_previous_close=bool(req.need_previous_close),
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            raise SchwabAPIError(f"Schwab price history request failed: {exc}") from exc

        candles = payload.get("candles", [])
        if candles is None:
            return []
        if not isinstance(candles, list):
            raise SchwabAPIError("Unexpected candles payload type from pricehistory endpoint.")
        return candles

    def fetch_option_chain(self, req: OptionChainRequest) -> Dict[str, Any]:
        try:
            resp = self._client.get_option_chain(
                req.symbol,
                contract_type=req.contract_type,
                strike_count=int(req.strike_count),
                include_underlying_quote=bool(req.include_quotes),
                strategy=req.strategy,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise SchwabAPIError("Unexpected option chain payload type from Schwab.")
            return payload
        except Exception as exc:
            raise SchwabAPIError(f"Schwab option chain request failed: {exc}") from exc
