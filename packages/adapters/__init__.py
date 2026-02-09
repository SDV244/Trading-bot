"""External adapters package."""

from packages.adapters.binance_live import (
    BinanceLiveAdapter,
    BinanceLiveAdapterError,
    close_binance_live_adapter,
    get_binance_live_adapter,
)
from packages.adapters.binance_spot import (
    BinanceSpotAdapter,
    CandleData,
    close_binance_adapter,
    get_binance_adapter,
)
from packages.adapters.telegram_bot import TelegramNotifier, get_telegram_notifier

__all__ = [
    "BinanceLiveAdapter",
    "BinanceLiveAdapterError",
    "BinanceSpotAdapter",
    "CandleData",
    "TelegramNotifier",
    "close_binance_adapter",
    "close_binance_live_adapter",
    "get_binance_adapter",
    "get_binance_live_adapter",
    "get_telegram_notifier",
]
