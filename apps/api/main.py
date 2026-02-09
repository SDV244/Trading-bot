"""
FastAPI Application Entry Point

Main application with lifespan management, routers, and middleware.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from apps.api.middleware.request_id import RequestIDMiddleware
from apps.api.middleware.security_headers import SecurityHeadersMiddleware
from apps.api.routers import ai, auth, health, market, system, trading
from packages.adapters.binance_live import close_binance_live_adapter
from packages.adapters.binance_spot import close_binance_adapter
from packages.core.config import get_settings
from packages.core.database.migrations import run_migrations
from packages.core.database.session import close_database, init_database
from packages.core.logging_setup import configure_logging
from packages.core.scheduler import close_trading_scheduler


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    logger.info(f"Starting Trading Bot API on {settings.api.host}:{settings.api.port}")
    logger.info(f"Trading mode: {'LIVE' if settings.trading.live_mode else 'PAPER'}")
    logger.info(f"Trading pair: {settings.trading.pair}")
    if settings.trading.live_mode and (not settings.binance.api_key or not settings.binance.api_secret):
        raise RuntimeError("TRADING_LIVE_MODE=true requires BINANCE_API_KEY and BINANCE_API_SECRET")

    # Initialize database
    await init_database()
    if settings.database.auto_migrate:
        await asyncio.to_thread(run_migrations)
    logger.info("Database initialized")

    yield

    # Shutdown
    await close_trading_scheduler()
    await close_binance_live_adapter()
    await close_binance_adapter()
    await close_database()
    logger.info("Database connections closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title="Trading Bot API",
        description="AI-powered Binance BTCUSDT trading bot",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware (local only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allow_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(market.router, prefix="/api/market", tags=["Market"])
    app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
    app.include_router(ai.router, prefix="/api/ai", tags=["AI"])

    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "apps.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
    )
