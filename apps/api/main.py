"""
FastAPI Application Entry Point

Main application with lifespan management, routers, and middleware.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import health, system, trading
from packages.core.config import get_settings
from packages.core.database.session import close_database, init_database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    print(f"Starting Trading Bot API on {settings.api.host}:{settings.api.port}")
    print(f"Trading mode: {'LIVE' if settings.trading.live_mode else 'PAPER'}")
    print(f"Trading pair: {settings.trading.pair}")

    # Initialize database
    await init_database()
    print("Database initialized")

    yield

    # Shutdown
    await close_database()
    print("Database connections closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    get_settings()

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
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])

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
