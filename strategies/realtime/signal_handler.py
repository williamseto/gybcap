"""Signal dispatch: Discord, logging, and bridge handlers."""

import hashlib
import json
import os
import logging
import math
from datetime import datetime, timezone
from typing import List, Protocol

import requests
from dotenv import load_dotenv

from strategies.realtime.protocol import RealtimeSignal

logger = logging.getLogger(__name__)


class SignalHandler(Protocol):
    """Protocol for consuming emitted signals."""

    def handle(self, signals: List[RealtimeSignal]) -> None: ...


class DiscordSignalHandler:
    """Send signals to a Discord webhook."""

    MAX_MESSAGE_LEN = 2000

    def __init__(self, webhook_url: str | None = None):
        if webhook_url is None:
            load_dotenv()
            webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        self._webhook_url = webhook_url

    def handle(self, signals: List[RealtimeSignal]) -> None:
        if not signals or not self._webhook_url:
            return

        content = "\n".join(str(s) for s in signals)
        if len(content) > self.MAX_MESSAGE_LEN:
            content = content[: self.MAX_MESSAGE_LEN]

        payload = {"content": content}
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(self._webhook_url, json=payload, headers=headers)
            if not resp.ok:
                logger.warning("Discord webhook failed: %s %s", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("Discord webhook error: %s", e)


class LoggingSignalHandler:
    """Log signals via the standard logging module."""

    def handle(self, signals: List[RealtimeSignal]) -> None:
        for s in signals:
            logger.info("Signal: %s", s)


class JsonlFileSignalHandler:
    """Append emitted signals to JSONL for external consumers (e.g. NinjaTrader)."""

    def __init__(self, output_path: str, truncate_on_start: bool = False):
        if not output_path:
            raise ValueError("output_path is required")
        self._output_path = output_path
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if truncate_on_start:
            with open(self._output_path, "w", encoding="utf-8"):
                pass
            logger.info("Initialized empty signal JSONL at %s", self._output_path)

    @staticmethod
    def _signal_id(sig: RealtimeSignal) -> str:
        base = (
            f"{sig.strategy_name}|{sig.entry_ts.isoformat()}|{sig.direction}|"
            f"{sig.level_name}|{sig.level_value:.6f}"
        )
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    @classmethod
    def _json_safe(cls, value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(v) for v in value]
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except Exception:
                pass
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return cls._json_safe(item())
            except Exception:
                pass
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return cls._json_safe(tolist())
            except Exception:
                pass
        return str(value)

    def _signal_payload(self, sig: RealtimeSignal) -> dict:
        return {
            "signal_id": self._signal_id(sig),
            "strategy_name": sig.strategy_name,
            "trigger_ts": sig.trigger_ts.isoformat(),
            "entry_ts": sig.entry_ts.isoformat(),
            "entry_price": self._safe_float(sig.entry_price),
            "direction": str(sig.direction),
            "level_name": str(sig.level_name),
            "level_value": self._safe_float(sig.level_value),
            "pred_proba": self._safe_float(sig.pred_proba),
            "metadata": self._json_safe(sig.metadata or {}),
            "emitted_utc": datetime.now(timezone.utc).isoformat(),
        }

    def handle(self, signals: List[RealtimeSignal]) -> None:
        if not signals:
            return
        try:
            with open(self._output_path, "a", encoding="utf-8") as f:
                for sig in signals:
                    try:
                        payload = self._signal_payload(sig)
                        f.write(
                            json.dumps(
                                payload,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                        )
                        f.write("\n")
                    except Exception:
                        logger.warning(
                            "Failed serializing signal id=%s strategy=%s",
                            self._signal_id(sig),
                            sig.strategy_name,
                            exc_info=True,
                        )
                f.flush()
        except Exception as e:
            logger.warning("Failed writing signal JSONL path=%s error=%s", self._output_path, e)


class CompositeSignalHandler:
    """Dispatch signals to multiple handlers."""

    def __init__(self, handlers: List[SignalHandler] | None = None):
        self._handlers: List[SignalHandler] = list(handlers or [])

    def add(self, handler: SignalHandler) -> None:
        self._handlers.append(handler)

    def handle(self, signals: List[RealtimeSignal]) -> None:
        for h in self._handlers:
            h.handle(signals)
