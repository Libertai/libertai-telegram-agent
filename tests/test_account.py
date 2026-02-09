"""Tests for account handlers: /login, /logout, /account."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.handlers.account import (
    account_command,
    login_command,
    logout_command,
    validate_api_key,
)
from libertai_telegram_agent.services.encryption import decrypt_api_key, encrypt_api_key

ENCRYPTION_KEY = Fernet.generate_key().decode()


# ── Fixtures & helpers ────────────────────────────────────────────────


@pytest.fixture
async def db(tmp_path):
    """Create and initialize a temporary database."""
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter():
    return MagicMock()


def make_context(db, rate_limiter, encryption_key):
    context = MagicMock()
    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "encryption_key": encryption_key,
        "inference": MagicMock(),
    }
    context.args = []
    context.bot = MagicMock()
    context.bot.send_message = AsyncMock()
    return context


def make_update(user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    return update


# ── validate_api_key ──────────────────────────────────────────────────


class TestValidateApiKey:
    async def test_returns_balance_on_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"balance": 42.5}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("libertai_telegram_agent.handlers.account.httpx.AsyncClient", return_value=mock_client):
            result = await validate_api_key("https://api.test.io/v1", "test-key")

        assert result == 42.5
        mock_client.get.assert_awaited_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://api.test.io/v1/credits/balance"

    async def test_returns_none_on_non_200(self):
        mock_response = MagicMock()
        mock_response.status_code = 401

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("libertai_telegram_agent.handlers.account.httpx.AsyncClient", return_value=mock_client):
            result = await validate_api_key("https://api.test.io/v1", "bad-key")

        assert result is None

    async def test_returns_none_on_exception(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("connection error"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("libertai_telegram_agent.handlers.account.httpx.AsyncClient", return_value=mock_client):
            result = await validate_api_key("https://api.test.io/v1", "test-key")

        assert result is None


# ── login_command ─────────────────────────────────────────────────────


class TestLoginCommand:
    async def test_no_args_shows_usage(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = []

        await login_command(update, context)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "usage" in text.lower() or "/login" in text.lower()

    async def test_deletes_message_with_key(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["test-api-key-123"]

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=10.5,
        ):
            await login_command(update, context)

        update.message.delete.assert_awaited_once()

    async def test_stores_encrypted_key(self, db, rate_limiter):
        update = make_update(user_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["my-secret-api-key"]

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=10.5,
        ):
            await login_command(update, context)

        user = await db.get_user(12345)
        assert user is not None
        assert user["api_key"] is not None
        # The stored key should be encrypted, not plaintext
        assert user["api_key"] != "my-secret-api-key"
        # Decrypting should return the original key
        decrypted = decrypt_api_key(user["api_key"], ENCRYPTION_KEY)
        assert decrypted == "my-secret-api-key"

    async def test_sends_confirmation_via_bot(self, db, rate_limiter):
        update = make_update(user_id=12345, chat_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["my-secret-api-key"]

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=10.5,
        ):
            await login_command(update, context)

        context.bot.send_message.assert_awaited_once()
        call_args = context.bot.send_message.call_args
        assert call_args[1]["chat_id"] == 12345 or call_args[0][0] == 12345

    async def test_invalid_key_sends_error(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["bad-key"]

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await login_command(update, context)

        # Should still delete the message (to protect the key)
        update.message.delete.assert_awaited_once()
        # Should send error via bot.send_message
        context.bot.send_message.assert_awaited_once()
        call_args = context.bot.send_message.call_args
        if len(call_args[0]) > 1:
            text = call_args[0][1]
        else:
            text = call_args[1].get("text", "")
        assert "invalid" in text.lower() or "failed" in text.lower() or "error" in text.lower()

    async def test_group_login_blocked(self, db, rate_limiter):
        update = make_update(user_id=12345, chat_id=-1001, chat_type="group")
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["my-api-key"]

        await login_command(update, context)

        call_text = update.message.reply_text.call_args[0][0]
        assert "DM" in call_text or "private" in call_text.lower()

    async def test_delete_message_failure_does_not_crash(self, db, rate_limiter):
        """If deleting the message fails (e.g., no permission), login should continue."""
        update = make_update()
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)
        context.args = ["my-api-key"]
        update.message.delete = AsyncMock(side_effect=Exception("Forbidden"))

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=10.5,
        ):
            await login_command(update, context)

        # Should still complete successfully
        context.bot.send_message.assert_awaited_once()


# ── logout_command ────────────────────────────────────────────────────


class TestLogoutCommand:
    async def test_clears_api_key(self, db, rate_limiter):
        # Set up a user with an API key
        await db.ensure_user(12345)
        encrypted = encrypt_api_key("some-key", ENCRYPTION_KEY)
        await db.set_user_api_key(12345, encrypted)

        update = make_update(user_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)

        await logout_command(update, context)

        user = await db.get_user(12345)
        assert user is not None
        assert user["api_key"] is None

    async def test_confirms_logout(self, db, rate_limiter):
        await db.ensure_user(12345)

        update = make_update(user_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)

        await logout_command(update, context)

        update.message.reply_text.assert_awaited_once()

    async def test_group_logout_blocked(self, db, rate_limiter):
        update = make_update(user_id=12345, chat_id=-1001, chat_type="group")
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)

        await logout_command(update, context)

        call_text = update.message.reply_text.call_args[0][0]
        assert "DM" in call_text or "private" in call_text.lower()


# ── account_command ───────────────────────────────────────────────────


class TestAccountCommand:
    async def test_shows_connected_with_balance(self, db, rate_limiter):
        await db.ensure_user(12345)
        encrypted = encrypt_api_key("my-key", ENCRYPTION_KEY)
        await db.set_user_api_key(12345, encrypted)

        update = make_update(user_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)

        with patch(
            "libertai_telegram_agent.handlers.account.validate_api_key",
            new_callable=AsyncMock,
            return_value=25.0,
        ):
            await account_command(update, context)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "connected" in text.lower() or "logged in" in text.lower()
        assert "25" in text

    async def test_shows_free_tier_when_no_key(self, db, rate_limiter):
        await db.ensure_user(12345)

        update = make_update(user_id=12345)
        context = make_context(db, rate_limiter, ENCRYPTION_KEY)

        await account_command(update, context)

        update.message.reply_text.assert_awaited_once()
        text = update.message.reply_text.call_args[0][0]
        assert "free" in text.lower()
