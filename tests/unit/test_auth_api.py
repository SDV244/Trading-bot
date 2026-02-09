"""Tests for API authentication and RBAC."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from packages.core.config import reload_settings


@pytest.fixture
def auth_enabled(monkeypatch: pytest.MonkeyPatch):
    """Enable auth for a test and restore defaults after."""
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("AUTH_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("AUTH_ADMIN_PASSWORD", "admin-pass")
    monkeypatch.setenv("AUTH_OPERATOR_USERNAME", "operator")
    monkeypatch.setenv("AUTH_OPERATOR_PASSWORD", "operator-pass")
    monkeypatch.setenv("AUTH_VIEWER_USERNAME", "viewer")
    monkeypatch.setenv("AUTH_VIEWER_PASSWORD", "viewer-pass")
    reload_settings()
    yield
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("AUTH_SECRET_KEY", "")
    monkeypatch.setenv("AUTH_ADMIN_PASSWORD", "")
    monkeypatch.setenv("AUTH_OPERATOR_PASSWORD", "")
    monkeypatch.setenv("AUTH_VIEWER_PASSWORD", "")
    reload_settings()


def _login(client: TestClient, username: str, password: str) -> str:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.mark.usefixtures("auth_enabled")
def test_protected_endpoint_requires_token_when_enabled():
    """Protected endpoints reject unauthenticated requests."""
    client = TestClient(app)
    response = client.get("/api/system/state")
    assert response.status_code == 401


@pytest.mark.usefixtures("auth_enabled")
def test_login_and_me_flow():
    """Login returns a token that can access /me."""
    client = TestClient(app)
    token = _login(client, "admin", "admin-pass")
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["auth_enabled"] is True
    assert data["username"] == "admin"
    assert data["role"] == "admin"


@pytest.mark.usefixtures("auth_enabled")
def test_viewer_cannot_perform_operator_action():
    """Viewer role cannot call operator-level endpoints."""
    client = TestClient(app)
    token = _login(client, "viewer", "viewer-pass")
    response = client.post(
        "/api/system/state",
        json={"action": "pause", "reason": "not allowed", "changed_by": "viewer"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


@pytest.mark.usefixtures("auth_enabled")
def test_operator_cannot_perform_admin_action():
    """Operator role cannot call admin-only live order endpoint."""
    client = TestClient(app)
    token = _login(client, "operator", "operator-pass")
    response = client.post(
        "/api/trading/live/order",
        json={
            "side": "BUY",
            "quantity": "0.01",
            "ui_confirmed": True,
            "reauthenticated": True,
            "safety_acknowledged": True,
            "reason": "test",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
