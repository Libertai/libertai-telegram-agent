"""Tests for the chat handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.handlers.chat import handle_message, should_respond_in_group
from libertai_telegram_agent.services.inference import InferenceService
from libertai_telegram_agent.services.rate_limiter import RateLimiter


@pytest.fixture
async def db(tmp_path):
    """Create and initialize a temporary database."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter(db: Database) -> RateLimiter:
    return RateLimiter(db, daily_messages=50, daily_images=5)


@pytest.fixture
def encryption_key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture
def inference() -> MagicMock:
    return MagicMock(spec=InferenceService)


def make_context(db, rate_limiter, inference, encryption_key, max_messages=20):
    context = MagicMock()
    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "inference": inference,
        "encryption_key": encryption_key,
        "max_conversation_messages": max_messages,
    }
    context.bot = MagicMock()
    context.bot.username = "test_bot"
    return context


def make_update(text="Hello", user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.text = text
    update.message.photo = None
    update.message.caption = None
    update.message.reply_to_message = None
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


# ── should_respond_in_group ───────────────────────────────────────────


class TestShouldRespondInGroup:
    def test_responds_when_mentioned(self):
        update = make_update(text="@test_bot what is AI?", chat_type="group")
        assert should_respond_in_group(update, "test_bot") is True

    def test_does_not_respond_without_mention(self):
        update = make_update(text="random message", chat_type="group")
        assert should_respond_in_group(update, "test_bot") is False

    def test_responds_when_replied_to_bot(self):
        update = make_update(text="what do you think?", chat_type="group")
        update.message.reply_to_message = MagicMock()
        update.message.reply_to_message.from_user.is_bot = True
        update.message.reply_to_message.from_user.username = "test_bot"
        assert should_respond_in_group(update, "test_bot") is True

    def test_does_not_respond_when_replied_to_different_bot(self):
        update = make_update(text="what do you think?", chat_type="group")
        update.message.reply_to_message = MagicMock()
        update.message.reply_to_message.from_user.is_bot = True
        update.message.reply_to_message.from_user.username = "other_bot"
        assert should_respond_in_group(update, "test_bot") is False

    def test_responds_when_mentioned_in_caption(self):
        update = make_update(text=None, chat_type="group")
        update.message.text = None
        update.message.caption = "@test_bot describe this"
        assert should_respond_in_group(update, "test_bot") is True


# ── handle_message ────────────────────────────────────────────────────


class TestHandleMessageDM:
    async def test_sends_response(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="Hello")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="Hi there!")

        await handle_message(update, context)

        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Hi there!" in call_text

    async def test_stores_messages_in_db(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="Hello bot")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="Hello human")

        await handle_message(update, context)

        conv = await db.get_or_create_conversation(12345, "private")
        messages = await db.get_messages(conv["id"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello bot"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello human"

    async def test_sends_typing_action(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="Hello")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="response")

        await handle_message(update, context)

        update.message.chat.send_action.assert_called()


class TestHandleMessageRateLimiting:
    async def test_rate_limited_shows_limit_message(self, db, inference, encryption_key):
        limiter = RateLimiter(db, daily_messages=1, daily_images=0)
        update = make_update(text="First")
        context = make_context(db, limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="response")

        await handle_message(update, context)  # Uses the 1 allowed message

        update2 = make_update(text="Second")
        await handle_message(update2, context)

        # Second call should mention the limit
        last_call = update2.message.reply_text.call_args[0][0]
        assert "limit" in last_call.lower() or "console.libertai.io" in last_call

    async def test_no_rate_limit_for_connected_user(self, db, inference, encryption_key):
        limiter = RateLimiter(db, daily_messages=1, daily_images=0)

        # Set up a user with an encrypted API key
        await db.ensure_user(12345)
        encrypted = Fernet(encryption_key).encrypt(b"user-api-key").decode()
        await db.set_user_api_key(12345, encrypted)

        update = make_update(text="First")
        context = make_context(db, limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="response1")

        await handle_message(update, context)

        update2 = make_update(text="Second")
        inference.chat = AsyncMock(return_value="response2")
        await handle_message(update2, context)

        # Both should get real responses, not limit messages
        call1 = update.message.reply_text.call_args[0][0]
        call2 = update2.message.reply_text.call_args[0][0]
        assert "response1" in call1
        assert "response2" in call2


class TestHandleMessageGroups:
    async def test_skips_group_without_mention(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="random chat", chat_type="group")
        context = make_context(db, rate_limiter, inference, encryption_key)

        await handle_message(update, context)

        update.message.reply_text.assert_not_called()
        inference.chat.assert_not_called()

    async def test_responds_in_group_when_mentioned(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="@test_bot hello!", chat_type="group")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="Hi from group!")

        await handle_message(update, context)

        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Hi from group!" in call_text

    async def test_strips_bot_mention_from_group_message(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="@test_bot what is Python?", chat_type="group")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="Python is a language")

        await handle_message(update, context)

        # The message stored in DB and sent to inference should not contain the mention
        conv = await db.get_or_create_conversation(12345, "group")
        messages = await db.get_messages(conv["id"])
        user_msg = messages[0]
        assert "@test_bot" not in user_msg["content"]
        assert "what is Python?" in user_msg["content"]

    async def test_skips_supergroup_without_mention(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="random chat", chat_type="supergroup")
        context = make_context(db, rate_limiter, inference, encryption_key)

        await handle_message(update, context)

        update.message.reply_text.assert_not_called()


class TestHandleMessageErrors:
    async def test_handles_inference_error(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="Hello")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(side_effect=Exception("API error"))

        await handle_message(update, context)

        update.message.reply_text.assert_called()
        call_text = update.message.reply_text.call_args[0][0]
        assert "sorry" in call_text.lower() or "try again" in call_text.lower()

    async def test_returns_early_if_no_message(self, db, rate_limiter, inference, encryption_key):
        update = MagicMock()
        update.message = None
        context = make_context(db, rate_limiter, inference, encryption_key)

        # Should not raise
        await handle_message(update, context)


class TestSplitLongMessages:
    async def test_splits_long_message(self, db, rate_limiter, inference, encryption_key):
        long_response = "A" * 5000  # Over 4096 limit
        update = make_update(text="Hello")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value=long_response)

        await handle_message(update, context)

        # Should have been called multiple times
        assert update.message.reply_text.call_count >= 2

    async def test_short_message_not_split(self, db, rate_limiter, inference, encryption_key):
        update = make_update(text="Hello")
        context = make_context(db, rate_limiter, inference, encryption_key)
        inference.chat = AsyncMock(return_value="Short reply")

        await handle_message(update, context)

        assert update.message.reply_text.call_count == 1
