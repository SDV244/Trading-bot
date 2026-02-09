"""
Trading Bot Configuration Settings

All configuration is loaded from environment variables or .env file.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Trading mode enumeration."""

    PAPER = "paper"
    LIVE = "live"


class SystemState(str, Enum):
    """System state enumeration."""

    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


class AuthRole(str, Enum):
    """Authorization roles."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class BinanceSettings(BaseSettings):
    """Binance API configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_", extra="ignore")

    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")
    testnet: bool = Field(default=True, description="Use Binance testnet")

    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on testnet setting."""
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @property
    def ws_url(self) -> str:
        """Get the appropriate WebSocket URL based on testnet setting."""
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", extra="ignore")

    bot_token: str = Field(default="", description="Telegram bot token from BotFather")
    chat_id: str = Field(default="", description="Chat ID for notifications")


class TradingSettings(BaseSettings):
    """Trading parameters configuration."""

    model_config = SettingsConfigDict(env_prefix="TRADING_", extra="ignore")

    pair: str = Field(default="BTCUSDT", description="Trading pair")
    timeframes: str = Field(default="1h,4h", description="Comma-separated timeframes")
    live_mode: bool = Field(default=False, description="Enable live trading")

    @property
    def timeframe_list(self) -> list[str]:
        """Parse timeframes string into list."""
        return [tf.strip() for tf in self.timeframes.split(",")]


class RiskSettings(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore")

    per_trade: float = Field(default=0.005, description="Risk per trade (0.5%)")
    max_daily_loss: float = Field(default=0.02, description="Max daily loss (2%)")
    max_exposure: float = Field(default=0.25, description="Max exposure (25%)")
    fee_bps: int = Field(default=10, description="Trading fee in basis points")
    slippage_bps: int = Field(default=5, description="Expected slippage in basis points")


class ApprovalSettings(BaseSettings):
    """Approval workflow configuration."""

    model_config = SettingsConfigDict(env_prefix="APPROVAL_", extra="ignore")

    timeout_hours: int = Field(default=2, description="Approval timeout in hours")


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")
    allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated CORS allowlist",
    )

    @property
    def allow_origin_list(self) -> list[str]:
        """Parse CORS allowlist."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


class AppRuntimeSettings(BaseSettings):
    """Application runtime profile settings."""

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    env: str = Field(default="dev", description="Runtime environment profile")
    secrets_dir: str = Field(default="", description="Directory containing secret files")
    log_json: bool = Field(default=False, description="Emit JSON logs")
    log_file: str = Field(default="./logs/app.log", description="Log file path")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    url: str = Field(
        default="sqlite+aiosqlite:///./data/trading.db",
        alias="DATABASE_URL",
        description="Database connection URL",
    )
    auto_migrate: bool = Field(default=True, description="Run Alembic migrations at startup")


class LogSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(extra="ignore")

    level: str = Field(default="INFO", alias="LOG_LEVEL", description="Log level")


class LiveExecutionSettings(BaseSettings):
    """Live execution resilience settings."""

    model_config = SettingsConfigDict(env_prefix="LIVE_", extra="ignore")

    max_retries: int = Field(default=3, description="Live order retry attempts")
    min_interval_ms: int = Field(default=250, description="Minimum spacing between live API requests")
    recv_window_ms: int = Field(default=5000, description="Binance recvWindow for signed requests")


class AuthSettings(BaseSettings):
    """Authentication and RBAC configuration."""

    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")

    enabled: bool = Field(default=False, description="Enable API auth and role checks")
    secret_key: str = Field(default="", description="HMAC signing secret for access tokens")
    token_ttl_minutes: int = Field(default=480, description="Access token lifetime")
    admin_username: str = Field(default="admin", description="Admin username")
    admin_password: str = Field(default="", description="Admin password")
    operator_username: str = Field(default="operator", description="Operator username")
    operator_password: str = Field(default="", description="Operator password")
    viewer_username: str = Field(default="viewer", description="Viewer username")
    viewer_password: str = Field(default="", description="Viewer password")


class Settings(BaseSettings):
    """Main settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.dev"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    app: AppRuntimeSettings = Field(default_factory=AppRuntimeSettings)
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    log: LogSettings = Field(default_factory=LogSettings)
    live: LiveExecutionSettings = Field(default_factory=LiveExecutionSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


# Global settings instance
_settings: Settings | None = None
SettingsSection = TypeVar("SettingsSection", bound=BaseSettings)


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = _load_settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = _load_settings()
    return _settings


def _build_settings_section(
    settings_cls: type[SettingsSection],
    settings_kwargs: dict[str, object],
) -> SettingsSection:
    return settings_cls(**cast(dict[str, Any], settings_kwargs))


def _load_settings() -> Settings:
    """Load settings with environment profile and optional secrets directory."""
    app_env = os.getenv("APP_ENV", "dev")
    env_files = [".env", f".env.{app_env}"]
    existing_env_files = [path for path in env_files if Path(path).exists()]
    secrets_dir = os.getenv("APP_SECRETS_DIR") or os.getenv("SECRETS_DIR") or ""
    settings_kwargs: dict[str, object] = {}
    if existing_env_files:
        settings_kwargs["_env_file"] = tuple(existing_env_files)
    if secrets_dir:
        settings_kwargs["_secrets_dir"] = secrets_dir

    return Settings(
        app=_build_settings_section(AppRuntimeSettings, settings_kwargs),
        binance=_build_settings_section(BinanceSettings, settings_kwargs),
        telegram=_build_settings_section(TelegramSettings, settings_kwargs),
        trading=_build_settings_section(TradingSettings, settings_kwargs),
        risk=_build_settings_section(RiskSettings, settings_kwargs),
        approval=_build_settings_section(ApprovalSettings, settings_kwargs),
        api=_build_settings_section(APISettings, settings_kwargs),
        database=_build_settings_section(DatabaseSettings, settings_kwargs),
        log=_build_settings_section(LogSettings, settings_kwargs),
        live=_build_settings_section(LiveExecutionSettings, settings_kwargs),
        auth=_build_settings_section(AuthSettings, settings_kwargs),
    )
