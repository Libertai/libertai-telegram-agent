"""Command handlers for the Telegram bot."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.inference import AVAILABLE_MODELS
from libertai_telegram_agent.services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


def _get_rate_limiter(context: ContextTypes.DEFAULT_TYPE) -> RateLimiter:
    return context.bot_data["rate_limiter"]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start — ensure user exists, create conversation, send welcome."""
    db = _get_db(context)
    await db.ensure_user(update.effective_user.id)
    await db.get_or_create_conversation(update.effective_chat.id, update.effective_chat.type)

    await update.message.reply_text(
        "Welcome to LibertAI! I'm your gateway to decentralized AI inference.\n\n"
        "Just send me a message and I'll respond using AI. You can also:\n"
        "- Send photos for analysis\n"
        "- Use /image to generate images\n"
        "- Use /model to switch AI models\n\n"
        "Type /help for all commands."
    )


async def new_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /new — create a fresh conversation, clearing previous context."""
    db = _get_db(context)
    await db.create_new_conversation(update.effective_chat.id, update.effective_chat.type)
    await update.message.reply_text("Started a new conversation. Previous context has been cleared.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help — list all available commands."""
    await update.message.reply_text(
        "Available commands:\n\n"
        "/new - Start a new conversation\n"
        "/image <prompt> - Generate an image\n"
        "/model - Switch AI model\n"
        "/usage - Check daily usage\n"
        "/login <api-key> - Connect your LibertAI account\n"
        "/logout - Disconnect account\n"
        "/account - View account status\n"
        "/help - Show this message"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /model — show inline keyboard with available models."""
    db = _get_db(context)
    user = await db.ensure_user(update.effective_user.id)
    current_model = user["default_model"]
    is_connected = user["api_key"] is not None

    keyboard = []
    for model_id, info in AVAILABLE_MODELS.items():
        if info["tier"] == "pro" and not is_connected:
            label = f"{info['name']} (Pro - /login required)"
        elif model_id == current_model:
            label = f"-> {info['name']} (current)"
        else:
            label = info["name"]
        keyboard.append([InlineKeyboardButton(label, callback_data=f"model:{model_id}")])

    await update.message.reply_text(
        "Select a model:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard callback for model selection."""
    query = update.callback_query
    await query.answer()

    model_id = query.data.split(":", 1)[1]
    if model_id not in AVAILABLE_MODELS:
        await query.edit_message_text("Unknown model.")
        return

    db = _get_db(context)
    user = await db.ensure_user(query.from_user.id)
    model_info = AVAILABLE_MODELS[model_id]

    if model_info["tier"] == "pro" and user["api_key"] is None:
        await query.edit_message_text(
            f"{model_info['name']} requires a LibertAI account. Use /login to connect."
        )
        return

    await db.set_user_model(query.from_user.id, model_id)
    await query.edit_message_text(f"Model switched to {model_info['name']}.")


async def usage_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /usage — show daily usage or indicate no limits for connected users."""
    db = _get_db(context)
    user = await db.ensure_user(update.effective_user.id)
    rate_limiter = _get_rate_limiter(context)

    if user["api_key"] is not None:
        await update.message.reply_text(
            "You're connected to a LibertAI account — no daily limits apply.\n"
            "Use /account to check your credit balance."
        )
        return

    summary = await rate_limiter.get_usage_summary(update.effective_user.id)
    await update.message.reply_text(
        f"Daily usage (free tier):\n\n"
        f"Messages: {summary['messages_used']}/{summary['messages_used'] + summary['messages_remaining']} "
        f"({summary['messages_remaining']} remaining)\n"
        f"Images: {summary['images_used']}/{summary['images_used'] + summary['images_remaining']} "
        f"({summary['images_remaining']} remaining)\n\n"
        f"Limits reset at midnight UTC.\n"
        f"Connect your account with /login for unlimited access."
    )
