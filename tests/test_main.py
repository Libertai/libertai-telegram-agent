from unittest.mock import patch

from libertai_telegram_agent.main import create_application


def test_create_application_returns_app():
    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "LIBERTAI_API_KEY": "test-key",
        "BOT_ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS1mb3ItdGVzdGluZzE=",
    }):
        app = create_application()
        assert app is not None


def test_application_has_handlers():
    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "LIBERTAI_API_KEY": "test-key",
        "BOT_ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS1mb3ItdGVzdGluZzE=",
    }):
        app = create_application()
        # Should have command handlers + callback + message handlers
        assert len(app.handlers[0]) >= 11


def test_application_has_bot_data():
    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "LIBERTAI_API_KEY": "test-key",
        "BOT_ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS1mb3ItdGVzdGluZzE=",
    }):
        app = create_application()
        assert "inference" in app.bot_data
        assert "rate_limiter" in app.bot_data
        assert "encryption_key" in app.bot_data
