"""Signal dispatch: Discord, logging, and composite handlers."""

import os
import logging
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


class CompositeSignalHandler:
    """Dispatch signals to multiple handlers."""

    def __init__(self, handlers: List[SignalHandler] | None = None):
        self._handlers: List[SignalHandler] = list(handlers or [])

    def add(self, handler: SignalHandler) -> None:
        self._handlers.append(handler)

    def handle(self, signals: List[RealtimeSignal]) -> None:
        for h in self._handlers:
            h.handle(signals)
