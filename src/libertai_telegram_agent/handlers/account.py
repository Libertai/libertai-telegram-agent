"""Handlers for /login, /logout, and /account commands."""

from __future__ import annotations

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.encryption import decrypt_api_key, encrypt_api_key


async def validate_api_key(api_base_url: str, api_key: str) -> float | None:
    """Validate API key by checking credit balance. Returns balance or None if invalid."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/credits/balance",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                data = response.json()
                return data["balance"]
            return None
    except Exception:
        return None


async def login_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /login command. Validates and stores an encrypted API key."""
    chat_type = update.effective_chat.type

    # Block /login in groups to protect API keys
    if chat_type in ("group", "supergroup"):
        await update.message.reply_text(
            "For security, /login is only available in DMs.\n"
            "Send me a private message to connect your API key."
        )
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /login <api_key>\n\n"
            "Get your API key at https://console.libertai.io/api-keys "
            "and send it here.\n"
            "Your message will be deleted immediately for security."
        )
        return

    api_key = context.args[0]

    # Delete the message containing the API key as soon as possible
    try:
        await update.message.delete()
    except Exception:
        pass

    db: Database = context.bot_data["db"]
    encryption_key: str = context.bot_data["encryption_key"]
    inference = context.bot_data["inference"]

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Validate the API key
    balance = await validate_api_key(inference.api_base_url, api_key)
    if balance is None:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Invalid API key. Please check your key and try again.",
        )
        return

    # Ensure user exists, encrypt and store the key
    await db.ensure_user(user_id)
    encrypted = encrypt_api_key(api_key, encryption_key)
    await db.set_user_api_key(user_id, encrypted)

    await context.bot.send_message(
        chat_id=chat_id,
        text=f"API key connected successfully! Your balance: {balance} credits.",
    )


async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /logout command. Clears stored API key."""
    chat_type = update.effective_chat.type

    if chat_type in ("group", "supergroup"):
        await update.message.reply_text(
            "For security, /logout is only available in DMs.\n"
            "Send me a private message to manage your account."
        )
        return

    db: Database = context.bot_data["db"]
    user_id = update.effective_user.id

    await db.ensure_user(user_id)
    await db.set_user_api_key(user_id, None)

    await update.message.reply_text("Logged out. Your API key has been removed.")


async def account_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /account command. Shows account status and balance."""
    db: Database = context.bot_data["db"]
    encryption_key: str = context.bot_data["encryption_key"]
    inference = context.bot_data["inference"]

    user_id = update.effective_user.id
    user = await db.ensure_user(user_id)

    if user["api_key"]:
        # User has an API key - show connected status with balance
        api_key = decrypt_api_key(user["api_key"], encryption_key)
        balance = await validate_api_key(inference.api_base_url, api_key)
        if balance is not None:
            await update.message.reply_text(
                f"Connected with API key.\n"
                f"Balance: {balance} credits.\n\n"
                f"Use /logout to disconnect."
            )
        else:
            await update.message.reply_text(
                "Connected with API key, but could not fetch balance.\n"
                "Your key may have expired. Use /logout and /login to reconnect."
            )
    else:
        # No API key - show free tier info
        usage = await db.get_daily_usage(user_id)
        await update.message.reply_text(
            f"Free tier account.\n"
            f"Today's usage: {usage['message_count']} messages, {usage['image_count']} images.\n\n"
            f"Use /login <api_key> to connect your LibertAI account for unlimited access."
        )
