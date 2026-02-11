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
from packages.core.market_regime import MarketRegimeDetector

router = APIRouter()

_market_regime_detector = MarketRegimeDetector()


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


class TimeframeRequirementResponse(BaseModel):
    """Required/available candle count for one timeframe."""

    required: int
    available: int
    ready: bool


class DataRequirementsResponse(BaseModel):
    """Data requirements for the active strategy."""

    symbol: str
    active_strategy: str
    require_data_ready: bool
    data_ready: bool
    reasons: list[str]
    timeframes: dict[str, TimeframeRequirementResponse]


class MarketContextResponse(BaseModel):
    """Alternative market context payload."""

    symbol: str
    last_price: float
    change_24h: float
    volume_24h: float
    fear_greed: int
    funding_rate: float


class OrderBookResponse(BaseModel):
    """Order book intelligence payload."""

    best_bid: float
    best_ask: float
    spread_bps: float
    bid_depth_10: float
    ask_depth_10: float
    imbalance: float
    liquidity_score: float
    market_impact_1btc_bps: float


class RegimeResponse(BaseModel):
    """Current market regime diagnostics."""

    regime: str
    confidence: float
    trend_strength: float
    volatility_percentile: float
    mean_reversion_factor: float
    breakout_probability: float
    persistence_score: float
    transition_probabilities: dict[str, float]
    indicators: dict[str, float]


class MarketIntelligenceResponse(BaseModel):
    """Aggregated market intelligence response for UI/AI visibility."""

    symbol: str
    timeframe: str
    candles_used: int
    generated_at: datetime
    context: MarketContextResponse
    order_book: OrderBookResponse | None
    regime: RegimeResponse | None


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


@router.get("/data/requirements", response_model=DataRequirementsResponse)
async def get_data_requirements(
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> DataRequirementsResponse:
    """Get candle requirements and readiness for active strategy."""
    from packages.core.database.session import get_session, init_database
    from packages.core.trading_cycle import get_trading_cycle_service

    await init_database()
    async with get_session() as session:
        readiness = await get_trading_cycle_service().get_data_readiness(session)

    return DataRequirementsResponse(
        symbol=readiness.symbol,
        active_strategy=readiness.active_strategy,
        require_data_ready=readiness.require_data_ready,
        data_ready=readiness.data_ready,
        reasons=readiness.reasons,
        timeframes={
            tf: TimeframeRequirementResponse(
                required=status.required,
                available=status.available,
                ready=status.ready,
            )
            for tf, status in readiness.timeframes.items()
        },
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


@router.get("/intelligence", response_model=MarketIntelligenceResponse)
async def get_market_intelligence(
    symbol: str = Query("BTCUSDT", description="Trading pair"),
    timeframe: str = Query("1h", description="Timeframe used for regime analysis"),
    lookback: int = Query(240, ge=60, le=1000, description="Candles used for regime analysis"),
    _: AuthUser = Depends(require_min_role(AuthRole.VIEWER)),
) -> MarketIntelligenceResponse:
    """Get enriched market intelligence (context + order book + regime)."""
    from packages.adapters.binance_spot import get_binance_adapter
    from packages.core.alternative_data import get_alternative_data_aggregator
    from packages.core.database.repositories import CandleRepository
    from packages.core.database.session import get_session
    from packages.core.order_book import OrderBookAnalyzer

    context = await get_alternative_data_aggregator().build_market_context(symbol)
    order_book: OrderBookResponse | None = None
    regime_payload: RegimeResponse | None = None
    candle_count = 0

    try:
        depth_payload = await get_binance_adapter().get_order_book(symbol=symbol, limit=20)
        snapshot = OrderBookAnalyzer.from_binance_depth(symbol=symbol, payload=depth_payload)
        if snapshot is not None:
            order_book = OrderBookResponse(**snapshot.to_dict())
    except Exception:
        order_book = None

    async with get_session() as session:
        repo = CandleRepository(session)
        candles = await repo.get_latest_candles(symbol, timeframe, lookback)
        candle_count = len(candles)
        if candle_count >= 60:
            analysis = _market_regime_detector.detect_regime(
                closes=[c.close for c in candles],
                volumes=[c.volume for c in candles],
                highs=[c.high for c in candles],
                lows=[c.low for c in candles],
            )
            transition = _market_regime_detector.get_regime_transition_probability(analysis.regime)
            regime_payload = RegimeResponse(
                regime=analysis.regime.value,
                confidence=analysis.confidence,
                trend_strength=analysis.trend_strength,
                volatility_percentile=analysis.volatility_percentile,
                mean_reversion_factor=analysis.mean_reversion_factor,
                breakout_probability=analysis.breakout_probability,
                persistence_score=_market_regime_detector.get_regime_persistence_score(analysis.regime),
                transition_probabilities={key.value: value for key, value in transition.items()},
                indicators=analysis.indicators,
            )

    return MarketIntelligenceResponse(
        symbol=symbol,
        timeframe=timeframe,
        candles_used=candle_count,
        generated_at=datetime.now(UTC),
        context=MarketContextResponse(
            symbol=context.symbol,
            last_price=context.last_price,
            change_24h=context.change_24h,
            volume_24h=context.quote_volume_24h,
            fear_greed=context.fear_greed_index,
            funding_rate=context.funding_rate,
        ),
        order_book=order_book,
        regime=regime_payload,
    )
