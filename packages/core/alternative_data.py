"""Alternative market data fetchers for AI context enrichment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from packages.core.config import get_settings


@dataclass(slots=True, frozen=True)
class MarketContext:
    """Aggregated market context values used by advisor and UI diagnostics."""

    symbol: str
    last_price: float
    change_24h: float
    quote_volume_24h: float
    fear_greed_index: int
    funding_rate: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "change_24h": self.change_24h,
            "volume_24h": self.quote_volume_24h,
            "fear_greed": self.fear_greed_index,
            "funding_rate": self.funding_rate,
        }


class AlternativeDataAggregator:
    """Fetch optional external context with safe fallbacks."""

    async def build_market_context(self, symbol: str) -> MarketContext:
        ticker_task = asyncio.create_task(self._fetch_binance_ticker(symbol))
        fear_greed_task = asyncio.create_task(self._fetch_fear_greed())
        funding_task = asyncio.create_task(self._fetch_funding_rate(symbol))

        ticker, fear_greed, funding_rate = await asyncio.gather(
            ticker_task,
            fear_greed_task,
            funding_task,
            return_exceptions=True,
        )

        last_price = 0.0
        change_24h = 0.0
        volume_24h = 0.0
        if isinstance(ticker, dict):
            last_price = float(ticker.get("last_price", 0.0))
            change_24h = float(ticker.get("change_24h", 0.0))
            volume_24h = float(ticker.get("volume_24h", 0.0))

        return MarketContext(
            symbol=symbol,
            last_price=last_price,
            change_24h=change_24h,
            quote_volume_24h=volume_24h,
            fear_greed_index=int(fear_greed) if isinstance(fear_greed, int | float) else 50,
            funding_rate=float(funding_rate) if isinstance(funding_rate, int | float) else 0.0,
        )

    async def _fetch_binance_ticker(self, symbol: str) -> dict[str, float]:
        settings = get_settings()
        endpoint = f"{settings.binance.public_market_data_url}/api/v3/ticker/24hr"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(endpoint, params={"symbol": symbol})
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"last_price": 0.0, "change_24h": 0.0, "volume_24h": 0.0}
            return {
                "last_price": float(data.get("lastPrice", 0.0)),
                "change_24h": float(data.get("priceChangePercent", 0.0)) / 100.0,
                "volume_24h": float(data.get("quoteVolume", 0.0)),
            }

    async def _fetch_fear_greed(self) -> int:
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get("https://api.alternative.me/fng/", params={"limit": 1})
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return 50
            payload = data.get("data")
            if not isinstance(payload, list) or not payload:
                return 50
            item = payload[0]
            if not isinstance(item, dict):
                return 50
            value = item.get("value")
            return int(value) if value is not None else 50

    async def _fetch_funding_rate(self, symbol: str) -> float:
        # Funding is futures data, used only as sentiment/risk context.
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": symbol},
            )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return 0.0
            return float(data.get("lastFundingRate", 0.0))


_aggregator = AlternativeDataAggregator()


def get_alternative_data_aggregator() -> AlternativeDataAggregator:
    """Get shared alternative-data aggregator."""
    return _aggregator

