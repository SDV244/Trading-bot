"""
Tests for configuration settings.
"""




class TestSettings:
    """Test cases for Settings."""

    def test_default_settings(self, monkeypatch):
        """Default settings are applied correctly."""
        # Clear any existing env vars
        monkeypatch.delenv("BINANCE_API_KEY", raising=False)
        monkeypatch.setenv("TRADING_LIVE_MODE", "false")
        monkeypatch.setenv("BINANCE_TESTNET", "false")
        monkeypatch.setenv("BINANCE_MARKET_DATA_BASE_URL", "https://data-api.binance.vision")
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
        monkeypatch.setenv("TRADING_ADVISOR_INTERVAL_CYCLES", "30")
        monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "long_flat")
        monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "10000")
        monkeypatch.setenv("TRADING_GRID_LEVELS", "6")
        monkeypatch.setenv("TRADING_GRID_MIN_SPACING_BPS", "25")
        monkeypatch.setenv("TRADING_GRID_MAX_SPACING_BPS", "220")
        monkeypatch.setenv("TRADING_GRID_TREND_TILT", "1.25")
        monkeypatch.setenv("TRADING_GRID_VOLATILITY_BLEND", "0.7")
        monkeypatch.setenv("TRADING_GRID_TAKE_PROFIT_BUFFER", "0.02")
        monkeypatch.setenv("TRADING_GRID_STOP_LOSS_BUFFER", "0.05")
        monkeypatch.setenv("TRADING_GRID_BOOTSTRAP_FRACTION", "1.0")
        monkeypatch.setenv("TRADING_GRID_ENFORCE_FEE_FLOOR", "false")
        monkeypatch.setenv("TRADING_GRID_COOLDOWN_SECONDS", "0")
        monkeypatch.setenv("TRADING_GRID_MIN_NET_PROFIT_BPS", "30")
        monkeypatch.setenv("RISK_PER_TRADE", "0.005")
        monkeypatch.setenv("RISK_MAX_DAILY_LOSS", "0.02")
        monkeypatch.setenv("RISK_MAX_EXPOSURE", "0.25")
        monkeypatch.setenv("LLM_ENABLED", "false")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4.1-mini")
        monkeypatch.setenv("MULTIAGENT_ENABLED", "false")

        # Reset cached settings
        import packages.core.config as config_module

        config_module._settings = None

        settings = config_module.get_settings()

        assert settings.trading.pair == "BTCUSDT"
        assert settings.trading.live_mode is False
        assert settings.trading.active_strategy == "trend_ema"
        assert settings.trading.require_data_ready is True
        assert settings.trading.spot_position_mode == "long_flat"
        assert settings.trading.grid_lookback_1h == 120
        assert settings.trading.grid_atr_period_1h == 14
        assert settings.trading.grid_levels == 6
        assert settings.trading.grid_spacing_mode == "geometric"
        assert settings.trading.grid_min_spacing_bps == 25
        assert settings.trading.grid_max_spacing_bps == 220
        assert settings.trading.grid_trend_tilt == 1.25
        assert settings.trading.grid_volatility_blend == 0.7
        assert settings.trading.grid_take_profit_buffer == 0.02
        assert settings.trading.grid_stop_loss_buffer == 0.05
        assert settings.trading.grid_cooldown_seconds == 0
        assert settings.trading.grid_auto_inventory_bootstrap is True
        assert settings.trading.grid_bootstrap_fraction == 1.0
        assert settings.trading.grid_enforce_fee_floor is False
        assert settings.trading.grid_min_net_profit_bps == 30
        assert settings.trading.grid_out_of_bounds_alert_cooldown_minutes == 60
        assert settings.trading.grid_recenter_mode == "aggressive"
        assert settings.trading.regime_adaptation_enabled is True
        assert settings.trading.inventory_profit_levels_list == (
            (0.015, 0.25),
            (0.025, 0.5),
            (0.04, 1.0),
        )
        assert settings.trading.stop_loss_enabled is True
        assert settings.trading.stop_loss_global_equity_pct == 0.15
        assert settings.trading.stop_loss_max_drawdown_pct == 0.20
        assert settings.trading.stop_loss_auto_close_positions is True
        assert settings.trading.advisor_interval_cycles == 30
        assert settings.trading.paper_starting_equity == 10000.0
        assert settings.risk.per_trade == 0.005
        assert settings.risk.min_per_trade == 0.001
        assert settings.risk.max_per_trade == 0.02
        assert settings.risk.dynamic_sizing_enabled is True
        assert settings.risk.max_daily_loss == 0.02
        assert settings.approval.timeout_hours == 2
        assert settings.approval.auto_approve_enabled is False
        assert settings.approval.emergency_ai_enabled is True
        assert settings.approval.emergency_max_proposals == 3
        assert settings.llm.enabled is False
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4.1-mini"
        assert settings.llm.prefer_llm is True
        assert settings.llm.fallback_to_rules is True
        assert settings.multiagent.enabled is False
        assert settings.multiagent.max_proposals == 5

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
        assert "testnet" in settings.binance.trading_base_url
        assert "testnet" in settings.binance.ws_url

    def test_binance_production_url(self, monkeypatch):
        """Production URLs are returned when testnet is disabled."""
        monkeypatch.setenv("BINANCE_TESTNET", "false")
        monkeypatch.setenv("BINANCE_MARKET_DATA_BASE_URL", "https://data-api.binance.vision")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()

        assert "testnet" not in settings.binance.base_url
        assert "api.binance.com" in settings.binance.base_url
        assert settings.binance.public_market_data_url == "https://data-api.binance.vision"

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

    def test_paper_starting_equity_override(self, monkeypatch):
        """Paper starting equity can be overridden from env."""
        monkeypatch.setenv("TRADING_PAPER_STARTING_EQUITY", "25000")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.trading.paper_starting_equity == 25000.0

    def test_spot_position_mode_override(self, monkeypatch):
        """Spot position mode can be overridden from env."""
        monkeypatch.setenv("TRADING_SPOT_POSITION_MODE", "incremental")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.trading.spot_position_mode == "incremental"

    def test_grid_settings_override(self, monkeypatch):
        """Grid strategy settings can be overridden from env."""
        monkeypatch.setenv("TRADING_GRID_LOOKBACK_1H", "160")
        monkeypatch.setenv("TRADING_GRID_ATR_PERIOD_1H", "20")
        monkeypatch.setenv("TRADING_GRID_LEVELS", "8")
        monkeypatch.setenv("TRADING_GRID_SPACING_MODE", "arithmetic")
        monkeypatch.setenv("TRADING_GRID_MIN_SPACING_BPS", "30")
        monkeypatch.setenv("TRADING_GRID_MAX_SPACING_BPS", "260")
        monkeypatch.setenv("TRADING_GRID_TREND_TILT", "1.5")
        monkeypatch.setenv("TRADING_GRID_VOLATILITY_BLEND", "0.9")
        monkeypatch.setenv("TRADING_GRID_TAKE_PROFIT_BUFFER", "0.03")
        monkeypatch.setenv("TRADING_GRID_STOP_LOSS_BUFFER", "0.07")
        monkeypatch.setenv("TRADING_GRID_COOLDOWN_SECONDS", "120")
        monkeypatch.setenv("TRADING_GRID_AUTO_INVENTORY_BOOTSTRAP", "false")
        monkeypatch.setenv("TRADING_GRID_BOOTSTRAP_FRACTION", "0.6")
        monkeypatch.setenv("TRADING_GRID_ENFORCE_FEE_FLOOR", "true")
        monkeypatch.setenv("TRADING_GRID_MIN_NET_PROFIT_BPS", "40")
        monkeypatch.setenv("TRADING_GRID_OUT_OF_BOUNDS_ALERT_COOLDOWN_MINUTES", "30")
        monkeypatch.setenv("TRADING_GRID_RECENTER_MODE", "conservative")
        monkeypatch.setenv("TRADING_REGIME_ADAPTATION_ENABLED", "false")
        monkeypatch.setenv("TRADING_INVENTORY_PROFIT_LEVELS", "0.02:0.5,0.05:1")
        monkeypatch.setenv("TRADING_INVENTORY_TRAILING_STOP_PCT", "0.03")
        monkeypatch.setenv("TRADING_INVENTORY_TIME_STOP_HOURS", "72")
        monkeypatch.setenv("TRADING_INVENTORY_MIN_PROFIT_FOR_TIME_STOP", "0.01")
        monkeypatch.setenv("TRADING_STOP_LOSS_ENABLED", "true")
        monkeypatch.setenv("TRADING_STOP_LOSS_GLOBAL_EQUITY_PCT", "0.12")
        monkeypatch.setenv("TRADING_STOP_LOSS_MAX_DRAWDOWN_PCT", "0.18")
        monkeypatch.setenv("TRADING_STOP_LOSS_AUTO_CLOSE_POSITIONS", "false")
        monkeypatch.setenv("TRADING_ADVISOR_INTERVAL_CYCLES", "15")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.trading.grid_lookback_1h == 160
        assert settings.trading.grid_atr_period_1h == 20
        assert settings.trading.grid_levels == 8
        assert settings.trading.grid_spacing_mode == "arithmetic"
        assert settings.trading.grid_min_spacing_bps == 30
        assert settings.trading.grid_max_spacing_bps == 260
        assert settings.trading.grid_trend_tilt == 1.5
        assert settings.trading.grid_volatility_blend == 0.9
        assert settings.trading.grid_take_profit_buffer == 0.03
        assert settings.trading.grid_stop_loss_buffer == 0.07
        assert settings.trading.grid_cooldown_seconds == 120
        assert settings.trading.grid_auto_inventory_bootstrap is False
        assert settings.trading.grid_bootstrap_fraction == 0.6
        assert settings.trading.grid_enforce_fee_floor is True
        assert settings.trading.grid_min_net_profit_bps == 40
        assert settings.trading.grid_out_of_bounds_alert_cooldown_minutes == 30
        assert settings.trading.grid_recenter_mode == "conservative"
        assert settings.trading.regime_adaptation_enabled is False
        assert settings.trading.inventory_profit_levels_list == ((0.02, 0.5), (0.05, 1.0))
        assert settings.trading.inventory_trailing_stop_pct == 0.03
        assert settings.trading.inventory_time_stop_hours == 72
        assert settings.trading.inventory_min_profit_for_time_stop == 0.01
        assert settings.trading.stop_loss_enabled is True
        assert settings.trading.stop_loss_global_equity_pct == 0.12
        assert settings.trading.stop_loss_max_drawdown_pct == 0.18
        assert settings.trading.stop_loss_auto_close_positions is False
        assert settings.trading.advisor_interval_cycles == 15

    def test_apply_runtime_config_patch_updates_known_keys(self, monkeypatch):
        """Runtime config patch mutates in-memory settings sections safely."""
        monkeypatch.setenv("TRADING_ACTIVE_STRATEGY", "trend_ema")
        monkeypatch.setenv("RISK_PER_TRADE", "0.005")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.trading.active_strategy == "trend_ema"
        assert settings.risk.per_trade == 0.005

        applied = config_module.apply_runtime_config_patch(
            {
                "trading": {"active_strategy": "trend_ema_fast"},
                "risk": {"per_trade": 0.0035},
                "unknown_section": {"ignored": True},
            }
        )
        assert applied["trading"] == ["active_strategy"]
        assert applied["risk"] == ["per_trade"]
        assert config_module.get_settings().trading.active_strategy == "trend_ema_fast"
        assert config_module.get_settings().risk.per_trade == 0.0035

    def test_approval_auto_approve_override(self, monkeypatch):
        """Approval auto-approve can be overridden from env."""
        monkeypatch.setenv("APPROVAL_AUTO_APPROVE_ENABLED", "true")
        monkeypatch.setenv("APPROVAL_EMERGENCY_AI_ENABLED", "false")
        monkeypatch.setenv("APPROVAL_EMERGENCY_MAX_PROPOSALS", "5")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.approval.auto_approve_enabled is True
        assert settings.approval.emergency_ai_enabled is False
        assert settings.approval.emergency_max_proposals == 5
        config_module.reload_settings()

    def test_llm_settings_override(self, monkeypatch):
        """LLM provider settings can be overridden from env."""
        monkeypatch.setenv("LLM_ENABLED", "true")
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/api/chat")
        monkeypatch.setenv("LLM_MAX_PROPOSALS", "5")
        monkeypatch.setenv("LLM_MIN_CONFIDENCE", "0.65")
        monkeypatch.setenv("LLM_PREFER_LLM", "true")
        monkeypatch.setenv("LLM_FALLBACK_TO_RULES", "false")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.llm.enabled is True
        assert settings.llm.provider == "ollama"
        assert settings.llm.model == "llama3.1:8b"
        assert settings.llm.base_url == "http://127.0.0.1:11434/api/chat"
        assert settings.llm.max_proposals == 5
        assert settings.llm.min_confidence == 0.65
        assert settings.llm.prefer_llm is True
        assert settings.llm.fallback_to_rules is False

    def test_multiagent_settings_override(self, monkeypatch):
        """Multi-agent settings can be overridden from env."""
        monkeypatch.setenv("MULTIAGENT_ENABLED", "true")
        monkeypatch.setenv("MULTIAGENT_MAX_PROPOSALS", "7")
        monkeypatch.setenv("MULTIAGENT_MIN_CONFIDENCE", "0.6")
        monkeypatch.setenv("MULTIAGENT_META_AGENT_ENABLED", "false")
        monkeypatch.setenv("MULTIAGENT_SENTIMENT_AGENT_ENABLED", "false")

        import packages.core.config as config_module

        config_module._settings = None
        settings = config_module.get_settings()
        assert settings.multiagent.enabled is True
        assert settings.multiagent.max_proposals == 7
        assert settings.multiagent.min_confidence == 0.6
        assert settings.multiagent.meta_agent_enabled is False
        assert settings.multiagent.sentiment_agent_enabled is False
