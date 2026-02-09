"""
Market Data Endpoints

Endpoints for real-time market data and candles.
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from apps.api.security.auth import AuthUser, require_min_role
from packages.core.config import AuthRole

router = APIRouter()


class PriceResponse(BaseModel):
    """Current price response."""

    symbol: str
    price: str
    timestamp: datetime


class CandleResponse(BaseModel):
    """Candle data response."""

    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: str
    high: str
    low: str
    close: str
    volume: str
    trades_count: int


class DataStatusResponse(BaseModel):
    """Data status response."""

    symbol: str
    timeframes: dict[str, int]  # timeframe -> candle count
    last_update: datetime | None


@router.get("/price", response_model=PriceResponse)
async def get_current_price(
    symbol: str = Query("BTCUSDT", description="Trading pair"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> PriceResponse:
    """Get current price from Binance."""
    from packages.adapters.binance_spot import get_binance_adapter

    try:
        adapter = get_binance_adapter()
        price = await adapter.get_ticker_price(symbol)

        return PriceResponse(
            symbol=symbol,
            price=str(price),
            timestamp=datetime.now(UTC),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch price: {e}")


@router.get("/candles", response_model=list[CandleResponse])
async def get_candles(
    symbol: str = Query("BTCUSDT", description="Trading pair"),
    timeframe: str = Query("1h", description="Candle interval"),
    limit: int = Query(100, ge=1, le=500, description="Number of candles"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> list[CandleResponse]:
    """Get cached candles from database."""
    from packages.core.database.repositories import CandleRepository
    from packages.core.database.session import get_session

    async with get_session() as session:
        repo = CandleRepository(session)
        candles = await repo.get_latest_candles(symbol, timeframe, limit)

        return [
            CandleResponse(
                symbol=c.symbol,
                timeframe=c.timeframe,
                open_time=c.open_time,
                close_time=c.close_time,
                open=str(c.open),
                high=str(c.high),
                low=str(c.low),
                close=str(c.close),
                volume=str(c.volume),
                trades_count=c.trades_count,
            )
            for c in candles
        ]


@router.get("/data/status", response_model=DataStatusResponse)
async def get_data_status(
    symbol: str = Query("BTCUSDT", description="Trading pair"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> DataStatusResponse:
    """Get data storage status."""
    from packages.core.config import get_settings
    from packages.core.database.repositories import CandleRepository
    from packages.core.database.session import get_session

    settings = get_settings()
    timeframes = {}

    async with get_session() as session:
        repo = CandleRepository(session)
        for tf in settings.trading.timeframe_list:
            count = await repo.count_candles(symbol, tf)
            timeframes[tf] = count

        # Get last update time
        candles = await repo.get_latest_candles(symbol, "1h", 1)
        last_update = candles[0].close_time if candles else None

    return DataStatusResponse(
        symbol=symbol,
        timeframes=timeframes,
        last_update=last_update,
    )


@router.post("/data/fetch")
async def trigger_data_fetch(
    days: int = Query(7, ge=1, le=90, description="Days of history to fetch"),
    _: AuthUser = Depends(require_min_role(AuthRole.OPERATOR)),
) -> dict[str, Any]:
    """Trigger historical data fetch."""
    from packages.core.data_fetcher import get_data_fetcher

    fetcher = get_data_fetcher()
    results = await fetcher.fetch_all_timeframes(days)

    return {
        "status": "success",
        "fetched": results,
        "message": f"Fetched {sum(results.values())} total candles",
    }
