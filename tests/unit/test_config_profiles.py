"""Tests for config profiles and secrets loading."""


def test_loads_secrets_dir(monkeypatch, tmp_path):
    """Settings load values from secrets dir files."""
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir(parents=True)
    (secrets_dir / "BINANCE_API_KEY").write_text("secret_key_value", encoding="utf-8")
    (secrets_dir / "TRADING_PAIR").write_text("BTCUSDT", encoding="utf-8")

    monkeypatch.setenv("APP_SECRETS_DIR", str(secrets_dir))

    import packages.core.config as config_module

    config_module._settings = None
    settings = config_module.get_settings()
    assert settings.binance.api_key == "secret_key_value"


def test_loads_profile_env_file(monkeypatch, tmp_path):
    """Settings resolve profile-specific env file."""
    env_file = tmp_path / ".env.testprofile"
    env_file.write_text("TRADING_PAIR=ETHUSDT\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_ENV", "testprofile")

    import packages.core.config as config_module

    config_module._settings = None
    settings = config_module.get_settings()
    assert settings.trading.pair == "ETHUSDT"
