"""Token authentication and RBAC helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from packages.core.config import AuthRole, Settings, get_settings

_bearer_scheme = HTTPBearer(auto_error=False)
_ROLE_ORDER: dict[AuthRole, int] = {
    AuthRole.VIEWER: 0,
    AuthRole.OPERATOR: 1,
    AuthRole.ADMIN: 2,
}


@dataclass(slots=True, frozen=True)
class AuthUser:
    """Authenticated principal."""

    username: str
    role: AuthRole


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _users_from_settings(settings: Settings) -> dict[str, tuple[str, AuthRole]]:
    users: dict[str, tuple[str, AuthRole]] = {}
    if settings.auth.admin_username and settings.auth.admin_password:
        users[settings.auth.admin_username] = (settings.auth.admin_password, AuthRole.ADMIN)
    if settings.auth.operator_username and settings.auth.operator_password:
        users[settings.auth.operator_username] = (
            settings.auth.operator_password,
            AuthRole.OPERATOR,
        )
    if settings.auth.viewer_username and settings.auth.viewer_password:
        users[settings.auth.viewer_username] = (settings.auth.viewer_password, AuthRole.VIEWER)
    return users


def _validate_runtime_auth_configuration(settings: Settings) -> None:
    if not settings.auth.enabled:
        return
    if not settings.auth.secret_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AUTH_SECRET_KEY is required when AUTH_ENABLED=true",
        )


def authenticate_credentials(username: str, password: str, settings: Settings) -> AuthUser | None:
    """Validate username/password against configured credentials."""
    users = _users_from_settings(settings)
    user_data = users.get(username)
    if user_data is None:
        return None
    configured_password, role = user_data
    if not secrets.compare_digest(password, configured_password):
        return None
    return AuthUser(username=username, role=role)


def create_access_token(user: AuthUser, settings: Settings) -> tuple[str, int]:
    """Create signed access token and expiration epoch."""
    _validate_runtime_auth_configuration(settings)
    now = int(time.time())
    exp = now + settings.auth.token_ttl_minutes * 60
    payload = {
        "sub": user.username,
        "role": user.role.value,
        "iat": now,
        "exp": exp,
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_encoded = _b64url_encode(payload_json)
    signature = hmac.new(
        settings.auth.secret_key.encode("utf-8"),
        payload_encoded.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    token = f"{payload_encoded}.{_b64url_encode(signature)}"
    return token, exp


def _decode_token_payload(token: str, settings: Settings) -> dict[str, Any]:
    _validate_runtime_auth_configuration(settings)
    try:
        payload_encoded, signature_encoded = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format") from exc

    expected_signature = hmac.new(
        settings.auth.secret_key.encode("utf-8"),
        payload_encoded.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    provided_signature = _b64url_decode(signature_encoded)
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

    payload = cast(dict[str, Any], json.loads(_b64url_decode(payload_encoded).decode("utf-8")))
    exp = int(payload.get("exp", 0))
    if exp < int(time.time()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    return payload


def _user_from_token_payload(payload: dict[str, Any]) -> AuthUser:
    username = str(payload.get("sub", "")).strip()
    role_raw = str(payload.get("role", "")).strip().lower()
    if not username or role_raw not in {r.value for r in AuthRole}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    return AuthUser(username=username, role=AuthRole(role_raw))


def _auth_header_value(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _get_development_fallback_user() -> AuthUser:
    return AuthUser(username="dev_local", role=AuthRole.ADMIN)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> AuthUser:
    """Resolve the current user from token when auth is enabled."""
    settings = get_settings()
    if not settings.auth.enabled:
        user = _get_development_fallback_user()
        request.state.auth_user = user
        return user

    token = _auth_header_value(credentials)
    payload = _decode_token_payload(token, settings)
    user = _user_from_token_payload(payload)
    request.state.auth_user = user
    return user


def require_min_role(min_role: AuthRole) -> Callable[..., Any]:
    """Create dependency enforcing minimum role level."""

    async def _dependency(user: AuthUser = Depends(get_current_user)) -> AuthUser:
        if not get_settings().auth.enabled:
            return user
        user_level = _ROLE_ORDER[user.role]
        min_level = _ROLE_ORDER[min_role]
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' cannot access this resource",
            )
        return user

    return _dependency
