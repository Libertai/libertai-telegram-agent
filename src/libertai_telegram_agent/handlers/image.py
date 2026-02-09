import io
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.encryption import decrypt_api_key
from libertai_telegram_agent.services.inference import InferenceService
from libertai_telegram_agent.services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /image command: generate an image from a text prompt."""
    if not context.args:
        await update.message.reply_text("Usage: /image <prompt>")
        return

    prompt = " ".join(context.args)
    telegram_id = update.effective_user.id

    db: Database = context.bot_data["db"]
    rate_limiter: RateLimiter = context.bot_data["rate_limiter"]
    inference: InferenceService = context.bot_data["inference"]
    encryption_key: str = context.bot_data["encryption_key"]

    await db.ensure_user(telegram_id)

    # Determine API key: check group settings first (if group chat), then user key
    api_key = None
    chat_type = update.effective_chat.type

    if chat_type in ("group", "supergroup"):
        group_settings = await db.get_group_settings(update.effective_chat.id)
        if group_settings:
            admin_user = await db.get_user(group_settings["admin_id"])
            if admin_user and admin_user.get("api_key"):
                api_key = decrypt_api_key(admin_user["api_key"], encryption_key)

    if api_key is None:
        user = await db.get_user(telegram_id)
        if user and user.get("api_key"):
            api_key = decrypt_api_key(user["api_key"], encryption_key)

    # Rate limit check for free tier (no personal API key)
    if api_key is None:
        allowed, remaining = await rate_limiter.check_and_increment(telegram_id, "image")
        if not allowed:
            await update.message.reply_text(
                "You've reached your daily image generation limit. "
                "Please try again tomorrow or set an API key with /setkey to remove limits."
            )
            return

    # Store in conversation history
    chat_id = update.effective_chat.id
    conv = await db.get_or_create_conversation(chat_id, chat_type)

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        image_bytes = await inference.generate_image(prompt, api_key=api_key)
        await update.message.reply_photo(
            photo=io.BytesIO(image_bytes),
            caption=prompt,
        )
        await db.add_message(conv["id"], telegram_id, "user", f"/image {prompt}")
        await db.add_message(conv["id"], 0, "assistant", f"[Generated image: {prompt}]")
    except Exception:
        logger.exception("Failed to generate image for prompt: %s", prompt)
        await update.message.reply_text(
            "Sorry, I couldn't generate that image. Please try again later."
        )
