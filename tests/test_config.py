import pytest
from pydantic import ValidationError

from libertai_telegram_agent.config import Settings

REQUIRED_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-token-123",
    "LIBERTAI_API_KEY": "test-api-key-456",
    "BOT_ENCRYPTION_KEY": "test-encryption-key-789",
}


class TestConfigDefaults:
    """Config loads from env vars with correct defaults."""

    def test_loads_required_vars(self, monkeypatch):
        for key, value in REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)

        settings = Settings()

        assert settings.telegram_bot_token == "test-token-123"
        assert settings.libertai_api_key == "test-api-key-456"
        assert settings.bot_encryption_key == "test-encryption-key-789"

    def test_default_values(self, monkeypatch):
        for key, value in REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)

        settings = Settings()

        assert settings.libertai_api_base_url == "https://api.libertai.io/v1"
        assert settings.default_model == "qwen3-code-next"
        assert settings.free_tier_daily_messages == 50
        assert settings.free_tier_daily_images == 5
        assert settings.max_conversation_messages == 20

    def test_override_defaults_via_env(self, monkeypatch):
        for key, value in REQUIRED_ENV.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setenv("LIBERTAI_API_BASE_URL", "https://custom.api.io/v2")
        monkeypatch.setenv("DEFAULT_MODEL", "llama-3-70b")
        monkeypatch.setenv("FREE_TIER_DAILY_MESSAGES", "100")
        monkeypatch.setenv("FREE_TIER_DAILY_IMAGES", "10")
        monkeypatch.setenv("MAX_CONVERSATION_MESSAGES", "50")

        settings = Settings()

        assert settings.libertai_api_base_url == "https://custom.api.io/v2"
        assert settings.default_model == "llama-3-70b"
        assert settings.free_tier_daily_messages == 100
        assert settings.free_tier_daily_images == 10
        assert settings.max_conversation_messages == 50


class TestConfigValidation:
    """Config raises ValidationError when required vars are missing."""

    def test_missing_telegram_bot_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("LIBERTAI_API_KEY", "test-api-key")
        monkeypatch.setenv("BOT_ENCRYPTION_KEY", "test-key")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "telegram_bot_token" in field_names

    def test_missing_libertai_api_key(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.delenv("LIBERTAI_API_KEY", raising=False)
        monkeypatch.setenv("BOT_ENCRYPTION_KEY", "test-key")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "libertai_api_key" in field_names

    def test_missing_bot_encryption_key(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("LIBERTAI_API_KEY", "test-api-key")
        monkeypatch.delenv("BOT_ENCRYPTION_KEY", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "bot_encryption_key" in field_names

    def test_missing_all_required_vars(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("LIBERTAI_API_KEY", raising=False)
        monkeypatch.delenv("BOT_ENCRYPTION_KEY", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "telegram_bot_token" in field_names
        assert "libertai_api_key" in field_names
        assert "bot_encryption_key" in field_names
