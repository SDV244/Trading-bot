"""Telegram notification adapter."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from loguru import logger

from packages.core.config import get_settings

if TYPE_CHECKING:
    from telegram import Bot


class TelegramNotifier:
    """Simple async Telegram notifier with safe no-op mode."""

    def __init__(self) -> None:
        settings = get_settings()
        self.bot_token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self._enabled = bool(self.bot_token and self.chat_id)
        self._bot: Bot | None = None
        if self._enabled:
            self._bot = self._build_bot(self.bot_token)

    @staticmethod
    def _build_bot(token: str) -> Bot:
        from telegram import Bot

        return Bot(token=token)

    @property
    def enabled(self) -> bool:
        """Whether notifications are enabled by config."""
        return self._enabled

    async def send_message(self, text: str) -> bool:
        """Send a message and return whether delivery was attempted successfully."""
        if not self._enabled or self._bot is None:
            return False
        try:
            await self._bot.send_message(chat_id=self.chat_id, text=text)
            return True
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    async def send_critical_alert(self, title: str, body: str) -> bool:
        """Send critical alert message."""
        return await self.send_message(f"[CRITICAL] {title}\n{body}")

    async def send_info(self, title: str, body: str) -> bool:
        """Send informational message."""
        return await self.send_message(f"[INFO] {title}\n{body}")


_notifier: TelegramNotifier | None = None
_notifier_lock = threading.Lock()


def get_telegram_notifier() -> TelegramNotifier:
    """Get or create singleton notifier."""
    global _notifier
    if _notifier is None:
        with _notifier_lock:
            if _notifier is None:
                _notifier = TelegramNotifier()
    return _notifier
