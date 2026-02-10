"""
Health Check Endpoints

Basic health and readiness checks for the API.
"""

from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from packages.core.config import get_settings
from packages.core.database.session import get_session
from packages.core.observability import render_prometheus_metrics

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool
    database: bool
    binance: bool
    reasons: list[str]


async def _check_database_ready() -> tuple[bool, str | None]:
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:
        return False, str(exc)


async def _check_binance_ready() -> tuple[bool, str | None]:
    try:
        settings = get_settings()
        url = f"{settings.binance.public_market_data_url}/api/v3/ping"
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0)) as client:
            response = await client.get(url)
        if response.status_code != 200:
            return False, f"ping_status_{response.status_code}"
        return True, None
    except Exception as exc:
        return False, str(exc)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version="0.1.0",
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(response: Response) -> ReadinessResponse:
    """Readiness check for runtime dependencies."""
    db_ok, db_reason = await _check_database_ready()
    binance_ok, binance_reason = await _check_binance_ready()
    reasons: list[str] = []
    if db_reason:
        reasons.append(f"database: {db_reason}")
    if binance_reason:
        reasons.append(f"binance: {binance_reason}")
    ready = db_ok and binance_ok
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=ready,
        database=db_ok,
        binance=binance_ok,
        reasons=reasons,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    settings = get_settings()
    if not settings.observability.metrics_enabled:
        return Response(status_code=404)
    payload, content_type = render_prometheus_metrics()
    return Response(content=payload, media_type=content_type)
