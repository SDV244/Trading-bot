"""External adapters package.

This module uses lazy imports to avoid loading heavy optional dependencies
at process startup (e.g. telegram client, live adapters).
"""

from importlib import import_module
from typing import Any

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

_SYMBOL_MODULE_MAP = {
    "BinanceLiveAdapter": "packages.adapters.binance_live",
    "BinanceLiveAdapterError": "packages.adapters.binance_live",
    "close_binance_live_adapter": "packages.adapters.binance_live",
    "get_binance_live_adapter": "packages.adapters.binance_live",
    "BinanceSpotAdapter": "packages.adapters.binance_spot",
    "CandleData": "packages.adapters.binance_spot",
    "close_binance_adapter": "packages.adapters.binance_spot",
    "get_binance_adapter": "packages.adapters.binance_spot",
    "TelegramNotifier": "packages.adapters.telegram_bot",
    "get_telegram_notifier": "packages.adapters.telegram_bot",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_MODULE_MAP.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
