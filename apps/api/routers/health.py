"""
Health Check Endpoints

Basic health and readiness checks for the API.
"""

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(UTC),
        version="0.1.0",
    )


@router.get("/ready")
async def readiness_check() -> dict[str, bool]:
    """Readiness check for dependencies."""
    # TODO: Add actual checks for database, Binance connection, etc.
    return {
        "ready": True,
        "database": True,
        "binance": True,
    }
