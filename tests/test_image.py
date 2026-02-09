"""Tests for the /image command handler."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

from telegram.constants import ChatAction

from libertai_telegram_agent.handlers.image import image_command


def _make_update_and_context(
    args: list[str] | None = None,
    telegram_id: int = 1001,
    chat_type: str = "private",
    chat_id: int = 9001,
):
    """Build minimal mock Update and Context objects for the image handler."""
    update = MagicMock()
    update.effective_user.id = telegram_id
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.chat.send_action = AsyncMock()

    context = MagicMock()
    context.args = args

    db = AsyncMock()
    rate_limiter = AsyncMock()
    inference = AsyncMock()
    encryption_key = "test-encryption-key"

    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "inference": inference,
        "encryption_key": encryption_key,
    }

    return update, context


class TestImageCommandNoPrompt:
    """When no prompt is provided, show usage instructions."""

    async def test_no_args_shows_usage(self):
        update, context = _make_update_and_context(args=[])
        await image_command(update, context)
        update.message.reply_text.assert_awaited_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "Usage" in call_text or "usage" in call_text.lower()

    async def test_none_args_shows_usage(self):
        update, context = _make_update_and_context(args=None)
        await image_command(update, context)
        update.message.reply_text.assert_awaited_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "usage" in call_text.lower()


class TestImageCommandSuccess:
    """With a valid prompt, generate and send an image."""

    async def test_sends_photo_with_prompt(self):
        update, context = _make_update_and_context(args=["a", "cat", "in", "space"])
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(return_value=b"fake-image-bytes")

        await image_command(update, context)

        inference.generate_image.assert_awaited_once()
        call_kwargs = inference.generate_image.call_args
        assert "a cat in space" in str(call_kwargs)

        update.message.reply_photo.assert_awaited_once()
        call_kwargs = update.message.reply_photo.call_args
        photo_arg = call_kwargs[1].get("photo") or call_kwargs[0][0]
        assert isinstance(photo_arg, io.BytesIO)
        assert photo_arg.read() == b"fake-image-bytes"

    async def test_sends_upload_photo_action(self):
        update, context = _make_update_and_context(args=["sunset"])
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(return_value=b"img-data")

        await image_command(update, context)

        update.message.chat.send_action.assert_awaited()
        action_arg = update.message.chat.send_action.call_args[0][0]
        assert action_arg == ChatAction.UPLOAD_PHOTO

    async def test_uses_user_api_key_when_set(self):
        update, context = _make_update_and_context(args=["forest"])
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(
            return_value={"telegram_id": 1001, "api_key": "encrypted-key"}
        )
        db.get_user = AsyncMock(
            return_value={"telegram_id": 1001, "api_key": "encrypted-key"}
        )
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(return_value=b"img-data")

        with patch(
            "libertai_telegram_agent.handlers.image.decrypt_api_key",
            return_value="decrypted-key",
        ):
            await image_command(update, context)

        call_kwargs = inference.generate_image.call_args
        assert call_kwargs[1].get("api_key") == "decrypted-key" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "decrypted-key"
        )

    async def test_caption_is_prompt(self):
        update, context = _make_update_and_context(args=["beautiful", "sunset"])
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(return_value=b"img-data")

        await image_command(update, context)

        call_kwargs = update.message.reply_photo.call_args
        caption = call_kwargs[1].get("caption", "")
        assert caption == "beautiful sunset"


class TestImageCommandRateLimited:
    """When rate limited, show a friendly limit message."""

    async def test_rate_limited_shows_message(self):
        update, context = _make_update_and_context(args=["ocean"])
        db = context.bot_data["db"]
        rate_limiter = context.bot_data["rate_limiter"]
        inference = context.bot_data["inference"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        rate_limiter.check_and_increment = AsyncMock(return_value=(False, 0))

        await image_command(update, context)

        update.message.reply_text.assert_awaited_once()
        call_text = update.message.reply_text.call_args[0][0]
        assert "limit" in call_text.lower()

        inference.generate_image.assert_not_awaited()


class TestImageCommandGroupChat:
    """In group chats, check group_settings for admin API key."""

    async def test_uses_group_admin_api_key(self):
        update, context = _make_update_and_context(
            args=["mountains"], chat_type="group", chat_id=5001
        )
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_group_settings = AsyncMock(
            return_value={"chat_id": 5001, "admin_id": 2001}
        )
        # The admin user has an API key
        async def get_user_side_effect(tid):
            if tid == 2001:
                return {"telegram_id": 2001, "api_key": "admin-encrypted-key"}
            return {"telegram_id": 1001, "api_key": None}

        db.get_user = AsyncMock(side_effect=get_user_side_effect)
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(return_value=b"img-data")

        with patch(
            "libertai_telegram_agent.handlers.image.decrypt_api_key",
            return_value="admin-decrypted-key",
        ):
            await image_command(update, context)

        call_kwargs = inference.generate_image.call_args
        assert call_kwargs[1].get("api_key") == "admin-decrypted-key" or (
            len(call_kwargs[0]) > 1 and call_kwargs[0][1] == "admin-decrypted-key"
        )


class TestImageCommandError:
    """When inference fails, send a friendly error message."""

    async def test_inference_error_sends_message(self):
        update, context = _make_update_and_context(args=["broken"])
        db = context.bot_data["db"]
        inference = context.bot_data["inference"]
        rate_limiter = context.bot_data["rate_limiter"]

        db.ensure_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        db.get_user = AsyncMock(return_value={"telegram_id": 1001, "api_key": None})
        rate_limiter.check_and_increment = AsyncMock(return_value=(True, 4))
        inference.generate_image = AsyncMock(side_effect=Exception("API error"))

        await image_command(update, context)

        update.message.reply_text.assert_awaited_once()
        call_text = update.message.reply_text.call_args[0][0]
        # Should be a friendly message, not a traceback
        assert "error" in call_text.lower() or "sorry" in call_text.lower() or "failed" in call_text.lower()
