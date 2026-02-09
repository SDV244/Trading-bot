"""
Tests for configuration settings.
"""




class TestSettings:
    """Test cases for Settings."""

    def test_default_settings(self, monkeypatch):
        """Default settings are applied correctly."""
        # Clear any existing env vars
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)

        # Reset cached settings
        import packages.core.config as config_module

        config_module._settings = None

        settings = config_module.get_settings()

        assert settings.trading.pair == "BTCUSDT"
        assert settings.trading.live_mode is False
        assert settings.risk.per_trade == 0.005
        assert settings.risk.max_daily_loss == 0.02
        assert settings.approval.timeout_hours == 2

    def test_timeframes_parsing(self, monkeypatch):
        """Timeframes string is parsed correctly."""
        monkeypatch.setenv("TRADING_TIMEFRAMES", "1h,4h,1d")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()

        assert settings.trading.timeframe_list == ["1h", "4h", "1d"]

    def test_binance_testnet_url(self, monkeypatch):
        """Testnet URLs are returned when testnet is enabled."""
        monkeypatch.setenv("BINANCE_TESTNET", "true")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()

        assert "testnet" in settings.binance.base_url
        assert "testnet" in settings.binance.ws_url

    def test_binance_production_url(self, monkeypatch):
        """Production URLs are returned when testnet is disabled."""
        monkeypatch.setenv("BINANCE_TESTNET", "false")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()

        assert "testnet" not in settings.binance.base_url
        assert "api.binance.com" in settings.binance.base_url

    def test_reload_settings(self, monkeypatch):
        """Settings can be reloaded."""
        import packages.core.config as config_module

        monkeypatch.setenv("TRADING_PAIR", "ETHUSDT")
        config_module._settings = None
        settings1 = config_module.get_settings()
        assert settings1.trading.pair == "ETHUSDT"

        monkeypatch.setenv("TRADING_PAIR", "BTCUSDT")
        settings2 = config_module.reload_settings()
        assert settings2.trading.pair == "BTCUSDT"
