"""Authentication endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from apps.api.security.auth import (
    AuthUser,
    authenticate_credentials,
    create_access_token,
    get_current_user,
)
from packages.core.config import get_settings

router = APIRouter()


class LoginRequest(BaseModel):
    """Login request body."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    """Successful login response."""

    access_token: str
    token_type: str
    expires_at: datetime
    role: str
    username: str


class AuthStatusResponse(BaseModel):
    """Auth status response."""

    auth_enabled: bool
    username: str
    role: str


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response) -> LoginResponse:
    """Authenticate with configured local credentials."""
    settings = get_settings()
    if not settings.auth.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authentication is disabled",
        )
    user = authenticate_credentials(request.username, request.password, settings)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token, exp_epoch = create_access_token(user, settings)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_at=datetime.fromtimestamp(exp_epoch, tz=UTC),
        role=user.role.value,
        username=user.username,
    )


@router.get("/me", response_model=AuthStatusResponse)
async def me(user: AuthUser = Depends(get_current_user)) -> AuthStatusResponse:
    """Get current auth principal and role."""
    return AuthStatusResponse(
        auth_enabled=get_settings().auth.enabled,
        username=user.username,
        role=user.role.value,
    )

