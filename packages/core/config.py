"""
Trading Bot Configuration Settings

All configuration is loaded from environment variables or .env file.
"""

from enum import Enum

from pydantic import Field, computed_field
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


class BinanceSettings(BaseSettings):
    """Binance API configuration."""

    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: str = Field(default="", description="Binance API key")
    api_secret: str = Field(default="", description="Binance API secret")
    testnet: bool = Field(default=True, description="Use Binance testnet")

    @computed_field
    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on testnet setting."""
        if self.testnet:
            return "https://testnet.binance.vision"
        return "https://api.binance.com"

    @computed_field
    @property
    def ws_url(self) -> str:
        """Get the appropriate WebSocket URL based on testnet setting."""
        if self.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"


class TelegramSettings(BaseSettings):
    """Telegram bot configuration."""

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_")

    bot_token: str = Field(default="", description="Telegram bot token from BotFather")
    chat_id: str = Field(default="", description="Chat ID for notifications")


class TradingSettings(BaseSettings):
    """Trading parameters configuration."""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    pair: str = Field(default="BTCUSDT", description="Trading pair")
    timeframes: str = Field(default="1h,4h", description="Comma-separated timeframes")
    live_mode: bool = Field(default=False, description="Enable live trading")

    @computed_field
    @property
    def timeframe_list(self) -> list[str]:
        """Parse timeframes string into list."""
        return [tf.strip() for tf in self.timeframes.split(",")]


class RiskSettings(BaseSettings):
    """Risk management configuration."""

    model_config = SettingsConfigDict(env_prefix="RISK_")

    per_trade: float = Field(default=0.005, description="Risk per trade (0.5%)")
    max_daily_loss: float = Field(default=0.02, description="Max daily loss (2%)")
    max_exposure: float = Field(default=0.25, description="Max exposure (25%)")
    fee_bps: int = Field(default=10, description="Trading fee in basis points")
    slippage_bps: int = Field(default=5, description="Expected slippage in basis points")


class ApprovalSettings(BaseSettings):
    """Approval workflow configuration."""

    model_config = SettingsConfigDict(env_prefix="APPROVAL_")

    timeout_hours: int = Field(default=2, description="Approval timeout in hours")


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="sqlite+aiosqlite:///./data/trading.db",
        alias="DATABASE_URL",
        description="Database connection URL",
    )


class LogSettings(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL", description="Log level")


class Settings(BaseSettings):
    """Main settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Nested settings
    binance: BinanceSettings = Field(default_factory=BinanceSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    approval: ApprovalSettings = Field(default_factory=ApprovalSettings)
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    log: LogSettings = Field(default_factory=LogSettings)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
