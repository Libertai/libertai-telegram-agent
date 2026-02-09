"""Tests for command handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from libertai_telegram_agent.handlers.commands import (
    start_command,
    new_command,
    help_command,
    model_command,
    model_callback,
    usage_command,
)
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter(db):
    return RateLimiter(db, daily_messages=50, daily_images=5)


def make_context(db, rate_limiter):
    context = MagicMock()
    context.bot_data = {"db": db, "rate_limiter": rate_limiter}
    context.bot = MagicMock()
    context.bot.username = "test_bot"
    return context


def make_update(user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.reply_text = AsyncMock()
    return update


def make_callback_update(user_id=12345, data="model:gemma-3-27b"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.callback_query.from_user.id = user_id
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    return update


# ── /start ─────────────────────────────────────────────────────────────


class TestStartCommand:
    async def test_sends_welcome_with_libertai(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await start_command(update, context)
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "LibertAI" in call_text

    async def test_creates_user(self, db, rate_limiter):
        update = make_update(user_id=99999)
        context = make_context(db, rate_limiter)
        await start_command(update, context)
        user = await db.get_user(99999)
        assert user is not None

    async def test_creates_conversation(self, db, rate_limiter):
        update = make_update(user_id=99999, chat_id=99999)
        context = make_context(db, rate_limiter)
        await start_command(update, context)
        conv = await db.fetch_one(
            "SELECT * FROM conversations WHERE chat_id = ? AND active = 1",
            (99999,),
        )
        assert conv is not None


# ── /new ───────────────────────────────────────────────────────────────


class TestNewCommand:
    async def test_confirms_new_conversation(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.get_or_create_conversation(12345, "private")
        await new_command(update, context)
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "new" in call_text.lower() or "conversation" in call_text.lower()

    async def test_deactivates_old_conversation(self, db, rate_limiter):
        update = make_update(chat_id=77777)
        context = make_context(db, rate_limiter)
        old_conv = await db.get_or_create_conversation(77777, "private")
        await new_command(update, context)
        old = await db.fetch_one(
            "SELECT * FROM conversations WHERE id = ?", (old_conv["id"],)
        )
        assert old["active"] == 0


# ── /help ──────────────────────────────────────────────────────────────


class TestHelpCommand:
    async def test_lists_image_command(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await help_command(update, context)
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "/image" in call_text

    async def test_lists_login_command(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await help_command(update, context)
        call_text = update.message.reply_text.call_args[0][0]
        assert "/login" in call_text

    async def test_lists_all_key_commands(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await help_command(update, context)
        call_text = update.message.reply_text.call_args[0][0]
        for cmd in ["/new", "/model", "/usage", "/help"]:
            assert cmd in call_text


# ── /model ─────────────────────────────────────────────────────────────


class TestModelCommand:
    async def test_shows_inline_keyboard(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await model_command(update, context)
        update.message.reply_text.assert_called_once()
        kwargs = update.message.reply_text.call_args[1]
        assert "reply_markup" in kwargs

    async def test_marks_current_model(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await model_command(update, context)
        kwargs = update.message.reply_text.call_args[1]
        markup = kwargs["reply_markup"]
        # Find the button for the default model (gemma-3-27b) which should be marked current
        buttons = [btn for row in markup.inline_keyboard for btn in row]
        current_buttons = [b for b in buttons if "current" in b.text.lower()]
        assert len(current_buttons) == 1
        assert "Gemma" in current_buttons[0].text

    async def test_pro_label_for_unconnected_user(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await model_command(update, context)
        kwargs = update.message.reply_text.call_args[1]
        markup = kwargs["reply_markup"]
        buttons = [btn for row in markup.inline_keyboard for btn in row]
        pro_buttons = [b for b in buttons if "Pro" in b.text]
        assert len(pro_buttons) >= 1
        assert "/login required" in pro_buttons[0].text

    async def test_no_pro_label_for_connected_user(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await db.set_user_api_key(12345, "some-api-key")
        await model_command(update, context)
        kwargs = update.message.reply_text.call_args[1]
        markup = kwargs["reply_markup"]
        buttons = [btn for row in markup.inline_keyboard for btn in row]
        pro_buttons = [b for b in buttons if "Pro" in b.text and "/login" in b.text]
        assert len(pro_buttons) == 0


# ── model_callback ─────────────────────────────────────────────────────


class TestModelCallback:
    async def test_switches_model(self, db, rate_limiter):
        await db.ensure_user(12345)
        update = make_callback_update(data="model:hermes-3-8b-tee")
        context = make_context(db, rate_limiter)
        await model_callback(update, context)
        user = await db.get_user(12345)
        assert user["default_model"] == "hermes-3-8b-tee"
        update.callback_query.edit_message_text.assert_called_once()
        call_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Hermes" in call_text

    async def test_rejects_unknown_model(self, db, rate_limiter):
        await db.ensure_user(12345)
        update = make_callback_update(data="model:nonexistent-model")
        context = make_context(db, rate_limiter)
        await model_callback(update, context)
        update.callback_query.edit_message_text.assert_called_once()
        call_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Unknown" in call_text or "unknown" in call_text.lower()

    async def test_rejects_pro_model_without_api_key(self, db, rate_limiter):
        await db.ensure_user(12345)
        update = make_callback_update(data="model:glm-4.7")
        context = make_context(db, rate_limiter)
        await model_callback(update, context)
        user = await db.get_user(12345)
        assert user["default_model"] == "gemma-3-27b"  # unchanged
        call_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "/login" in call_text

    async def test_allows_pro_model_with_api_key(self, db, rate_limiter):
        await db.ensure_user(12345)
        await db.set_user_api_key(12345, "my-key")
        update = make_callback_update(data="model:glm-4.7")
        context = make_context(db, rate_limiter)
        await model_callback(update, context)
        user = await db.get_user(12345)
        assert user["default_model"] == "glm-4.7"

    async def test_answers_callback_query(self, db, rate_limiter):
        await db.ensure_user(12345)
        update = make_callback_update(data="model:gemma-3-27b")
        context = make_context(db, rate_limiter)
        await model_callback(update, context)
        update.callback_query.answer.assert_awaited_once()


# ── /usage ─────────────────────────────────────────────────────────────


class TestUsageCommand:
    async def test_shows_remaining_counts(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await usage_command(update, context)
        update.message.reply_text.assert_called_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "50" in call_text or "messages" in call_text.lower()

    async def test_shows_no_limits_when_connected(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await db.set_user_api_key(12345, "some-key")
        await usage_command(update, context)
        call_text = update.message.reply_text.call_args[0][0]
        assert "no" in call_text.lower() and "limit" in call_text.lower()

    async def test_shows_usage_after_increment(self, db, rate_limiter):
        update = make_update()
        context = make_context(db, rate_limiter)
        await db.ensure_user(12345)
        await db.increment_usage(12345, "message")
        await db.increment_usage(12345, "message")
        await usage_command(update, context)
        call_text = update.message.reply_text.call_args[0][0]
        # Should show 48 remaining messages (50 - 2)
        assert "48" in call_text
