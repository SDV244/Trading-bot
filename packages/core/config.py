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

    model_config = SettingsConfigDict(env_prefix="BINANCE_", extra="ignore", env_ignore_empty=True)

    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")
    testnet: bool = Field(default=True, description="Use Binance testnet")
    market_data_base_url: str = Field(
        default="https://data-api.binance.vision",
        description="Public market data base URL (candles/tickers)",
    )
    market_max_retries: int = Field(default=3, ge=1, le=10, description="Market-data request retry attempts")
    market_min_interval_ms: int = Field(
        default=50,
        ge=0,
        le=10_000,
        description="Minimum spacing between public market-data API requests",
    )

    @property
    def trading_base_url(self) -> str:
        """Get signed/private endpoint base URL based on testnet setting."""
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @property
    def base_url(self) -> str:
        """Backward-compatible alias for trading base URL."""
        return self.trading_base_url

    @property
    def public_market_data_url(self) -> str:
        """Get public market-data endpoint base URL."""
        return self.market_data_base_url.rstrip("/")

    @property
    def ws_url(self) -> str:
        """Get the appropriate WebSocket URL based on testnet setting."""
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", extra="ignore", env_ignore_empty=True)

    bot_token: str = Field(default="", description="Telegram bot token from BotFather")
    chat_id: str = Field(default="", description="Chat ID for notifications")
    heartbeat_enabled: bool = Field(default=True, description="Send periodic heartbeat notifications")
    heartbeat_hours: int = Field(default=4, ge=1, le=24, description="Heartbeat interval in hours")


class TradingSettings(BaseSettings):
    """Trading parameters configuration."""

    model_config = SettingsConfigDict(env_prefix="TRADING_", extra="ignore", env_ignore_empty=True)

    pair: str = Field(default="BTCUSDT", description="Trading pair")
    timeframes: str = Field(default="1h,4h", description="Comma-separated timeframes")
    live_mode: bool = Field(default=False, description="Enable live trading")
    active_strategy: str = Field(default="trend_ema", description="Strategy registry key")
    require_data_ready: bool = Field(
        default=True,
        description="Block scheduler start if required candle history is missing",
    )
    spot_position_mode: str = Field(
        default="long_flat",
        description="Spot position behavior: long_flat (single position) or incremental (allow add/reduce)",
    )
    grid_lookback_1h: int = Field(default=120, ge=30, le=1000, description="Adaptive grid lookback window")
    grid_atr_period_1h: int = Field(default=14, ge=5, le=200, description="ATR period for grid spacing")
    grid_levels: int = Field(default=6, ge=3, le=20, description="Number of grid levels")
    grid_spacing_mode: str = Field(
        default="geometric",
        pattern="^(geometric|arithmetic)$",
        description="Grid spacing type: geometric (percentage) or arithmetic (absolute step)",
    )
    grid_min_spacing_bps: int = Field(
        default=25,
        ge=5,
        le=500,
        description="Minimum grid spacing in basis points",
    )
    grid_max_spacing_bps: int = Field(
        default=220,
        ge=10,
        le=1500,
        description="Maximum grid spacing in basis points",
    )
    grid_trend_tilt: float = Field(
        default=1.25,
        ge=0.0,
        le=5.0,
        description="Trend tilt factor for adaptive grid center",
    )
    grid_volatility_blend: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Multiplier applied to ATR-based spacing",
    )
    grid_take_profit_buffer: float = Field(
        default=0.02,
        ge=0.0,
        le=0.5,
        description="Optional profit buffer beyond upper grid band before forced take-profit signal",
    )
    grid_stop_loss_buffer: float = Field(
        default=0.05,
        ge=0.0,
        le=0.8,
        description="Optional stop-loss buffer beyond lower grid band before forced stop signal",
    )
    grid_cooldown_seconds: int = Field(
        default=0,
        ge=0,
        le=86_400,
        description="Minimum cooldown between executed paper fills",
    )
    grid_auto_inventory_bootstrap: bool = Field(
        default=True,
        description="When smart grid starts flat and emits SELL, allow a bootstrap BUY to seed inventory",
    )
    grid_bootstrap_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of risk-sized BUY quantity used when bootstrapping smart-grid inventory",
    )
    grid_enforce_fee_floor: bool = Field(
        default=False,
        description="Block trading readiness when minimum grid edge is below configured net-profit floor",
    )
    grid_min_net_profit_bps: int = Field(
        default=30,
        ge=0,
        le=500,
        description="Minimum expected net basis points per grid after fees/slippage",
    )
    grid_out_of_bounds_alert_cooldown_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Minimum minutes between repeated out-of-bounds alerts",
    )
    advisor_interval_cycles: int = Field(
        default=30,
        ge=1,
        le=1000,
        description="How often scheduler invokes AI advisor",
    )
    paper_starting_equity: float = Field(
        default=10000.0,
        gt=0,
        description="Starting equity baseline for paper-mode risk sizing and snapshots",
    )

    @property
    def timeframe_list(self) -> list[str]:
        """Parse timeframes string into list."""
        return [tf.strip() for tf in self.timeframes.split(",")]


class RiskSettings(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_prefix="RISK_", extra="ignore", env_ignore_empty=True)

    per_trade: float = Field(default=0.005, description="Risk per trade (0.5%)")
    max_daily_loss: float = Field(default=0.02, description="Max daily loss (2%)")
    max_exposure: float = Field(default=0.25, description="Max exposure (25%)")
    fee_bps: int = Field(default=10, description="Trading fee in basis points")
    slippage_bps: int = Field(default=5, description="Expected slippage in basis points")


class ApprovalSettings(BaseSettings):
    """Approval workflow configuration."""

    model_config = SettingsConfigDict(env_prefix="APPROVAL_", extra="ignore", env_ignore_empty=True)

    timeout_hours: int = Field(default=2, description="Approval timeout in hours")


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore", env_ignore_empty=True)

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

    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore", env_ignore_empty=True)

    env: str = Field(default="dev", description="Runtime environment profile")
    secrets_dir: str = Field(default="", description="Directory containing secret files")
    log_json: bool = Field(default=False, description="Emit JSON logs")
    log_file: str = Field(default="./logs/app.log", description="Log file path")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore", env_ignore_empty=True)

    url: str = Field(
        default="sqlite+aiosqlite:///./data/trading.db",
        alias="DATABASE_URL",
        description="Database connection URL",
    )
    auto_migrate: bool = Field(default=True, description="Run Alembic migrations at startup")


class LogSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    level: str = Field(default="INFO", alias="LOG_LEVEL", description="Log level")


class LiveExecutionSettings(BaseSettings):
    """Live execution resilience settings."""

    model_config = SettingsConfigDict(env_prefix="LIVE_", extra="ignore", env_ignore_empty=True)

    max_retries: int = Field(default=3, description="Live order retry attempts")
    min_interval_ms: int = Field(default=250, description="Minimum spacing between live API requests")
    recv_window_ms: int = Field(default=5000, description="Binance recvWindow for signed requests")


class AuthSettings(BaseSettings):
    """Authentication and RBAC configuration."""

    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore", env_ignore_empty=True)

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
        env_ignore_empty=True,
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
    settings_kwargs: dict[str, object] = {}
    if existing_env_files:
        settings_kwargs["_env_file"] = tuple(existing_env_files)

    # Resolve APP_SECRETS_DIR from both process env and env files.
    app_settings = _build_settings_section(AppRuntimeSettings, settings_kwargs)
    secrets_dir = (
        os.getenv("APP_SECRETS_DIR")
        or os.getenv("SECRETS_DIR")
        or app_settings.secrets_dir
        or ""
    )
    if secrets_dir:
        settings_kwargs["_secrets_dir"] = secrets_dir
        app_settings = _build_settings_section(AppRuntimeSettings, settings_kwargs)

    return Settings(
        app=app_settings,
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
