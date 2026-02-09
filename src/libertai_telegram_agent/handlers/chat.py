"""Chat message handler with vision support, group awareness, and rate limiting."""

from __future__ import annotations

import base64
import io
import json
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.encryption import decrypt_api_key
from libertai_telegram_agent.services.inference import AVAILABLE_MODELS, InferenceService
from libertai_telegram_agent.services.rate_limiter import RateLimiter
from libertai_telegram_agent.services.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = (
    "You are the LibertAI assistant, a helpful AI bot running on the Aleph Cloud "
    "decentralized cloud infrastructure. You are powered by open-source AI models "
    "through the LibertAI decentralized AI inference platform.\n\n"
    "About the projects you are part of:\n"
    "- Aleph Cloud (formerly aleph.im) is a decentralized cloud computing platform. "
    "Its token is $ALEPH, available on Ethereum, Base, BSC, Solana, and Avalanche.\n"
    "- LibertAI is a decentralized AI inference platform built on Aleph Cloud. "
    "Its token is $LTAI, available on Base and Solana. "
    "Users can buy inference credits at https://console.libertai.io and generate API keys "
    "at https://console.libertai.io/api-keys.\n\n"
    "You are friendly, concise, and helpful. When users ask you to create, draw, "
    "or generate an image, use the generate_image tool. The image prompt you pass "
    "to generate_image must ALWAYS be in English, regardless of what language the "
    "user is speaking.\n\n"
    "Use web_search to look up current information when needed. "
    "Use crypto_price to get live cryptocurrency prices. "
    "Use fetch_url to read the content of a webpage.\n\n"
    "Keep your responses concise and to the point unless the user asks for detail."
)


def should_respond_in_group(update: Update, bot_username: str) -> bool:
    """Return True if the bot was @mentioned or replied to in a group chat."""
    text = update.message.text or update.message.caption or ""

    # Check if bot is @mentioned
    if f"@{bot_username}" in text:
        return True

    # Check if message is a reply to the bot
    reply = update.message.reply_to_message
    if reply and reply.from_user and reply.from_user.is_bot and reply.from_user.username == bot_username:
        return True

    return False


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


def _get_inference(context: ContextTypes.DEFAULT_TYPE) -> InferenceService:
    return context.bot_data["inference"]


def _get_rate_limiter(context: ContextTypes.DEFAULT_TYPE) -> RateLimiter:
    return context.bot_data["rate_limiter"]


async def _get_api_key_for_request(
    db: Database,
    context: ContextTypes.DEFAULT_TYPE,
    telegram_id: int,
    chat_id: int,
    chat_type: str,
) -> str | None:
    """Determine which API key to use. Returns decrypted key or None for bot default."""
    encryption_key = context.bot_data["encryption_key"]

    # In groups, check group admin's key first
    if chat_type in ("group", "supergroup"):
        group_settings = await db.get_group_settings(chat_id)
        if group_settings:
            admin = await db.get_user(group_settings["admin_id"])
            if admin and admin["api_key"]:
                return decrypt_api_key(admin["api_key"], encryption_key)

    # Check user's own key
    user = await db.get_user(telegram_id)
    if user and user["api_key"]:
        return decrypt_api_key(user["api_key"], encryption_key)

    return None


async def _split_and_send(update: Update, text: str) -> None:
    """Send a message, splitting into chunks if it exceeds Telegram's limit."""
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(text)
        return

    # Split at paragraph boundaries, then line boundaries, then hard split
    chunks: list[str] = []
    while text:
        if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break

        # Try paragraph boundary first
        split_at = text.rfind("\n\n", 0, TELEGRAM_MAX_MESSAGE_LENGTH)
        if split_at == -1:
            # Try line boundary
            split_at = text.rfind("\n", 0, TELEGRAM_MAX_MESSAGE_LENGTH)
        if split_at == -1:
            # Hard split
            split_at = TELEGRAM_MAX_MESSAGE_LENGTH

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk)


async def _handle_image_tool_call(
    update: Update,
    inference: InferenceService,
    api_key: str | None,
    tool_call,
    conv: dict,
    db: Database,
) -> str:
    """Execute a generate_image tool call and send the result. Returns result string for tool loop."""
    try:
        args = json.loads(tool_call.function.arguments)
        prompt = args.get("prompt", "")
    except (json.JSONDecodeError, KeyError):
        return "Error: could not parse image prompt"

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        image_bytes = await inference.generate_image(prompt=prompt, api_key=api_key)
        await update.message.reply_photo(
            photo=io.BytesIO(image_bytes),
            caption=prompt[:1024],
        )
        await db.add_message(conv["id"], 0, "assistant", f"[Generated image: {prompt}]")
        return f"Image generated successfully with prompt: {prompt}"
    except Exception:
        logger.exception("Image generation error")
        await update.message.reply_text("Sorry, I couldn't generate that image. Please try again.")
        return "Image generation failed"


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages and photos."""
    if not update.message or not update.effective_user:
        return

    bot_username = context.bot.username
    chat_type = update.effective_chat.type

    # In groups, only respond when mentioned or replied to
    if chat_type in ("group", "supergroup"):
        if not should_respond_in_group(update, bot_username):
            return

    db = _get_db(context)
    inference = _get_inference(context)
    rate_limiter = _get_rate_limiter(context)
    telegram_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Ensure user exists
    await db.ensure_user(telegram_id)

    # Determine API key (group admin's, user's own, or None for bot default)
    api_key = await _get_api_key_for_request(db, context, telegram_id, chat_id, chat_type)
    is_connected = api_key is not None

    # Rate limit check for free tier users
    if not is_connected:
        allowed, remaining = await rate_limiter.check_and_increment(telegram_id, "message")
        if not allowed:
            login_hint = "3. Use /login <your-api-key> to connect your account"
            if chat_type in ("group", "supergroup"):
                login_hint = "3. DM me with /login <your-api-key> to connect your account"
            await update.message.reply_text(
                "You've reached your daily free message limit.\n\n"
                "For unlimited access:\n"
                "1. Buy credits at https://console.libertai.io\n"
                "2. Generate an API key at https://console.libertai.io/api-keys\n"
                f"{login_hint}"
            )
            return

    # Get or create conversation
    conv = await db.get_or_create_conversation(chat_id, chat_type)
    max_messages = context.bot_data["max_conversation_messages"]

    # Get user's model
    user = await db.get_user(telegram_id)
    model = user["default_model"] if user else "qwen3-coder-next"

    # Build user content text
    user_content = update.message.text or update.message.caption or ""

    # Strip @bot_username from group messages
    if bot_username:
        user_content = user_content.replace(f"@{bot_username}", "").strip()

    # Handle photo messages (vision)
    message_content = user_content  # Text-only version for DB storage
    openai_messages_content = user_content  # May become vision content for API

    if update.message.photo:
        photo = update.message.photo[-1]  # Highest resolution
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        b64_image = base64.b64encode(photo_bytes).decode()

        # Fall back to vision-capable model if current model doesn't support vision
        if not AVAILABLE_MODELS.get(model, {}).get("vision", False):
            model = "gemma-3-27b"

        openai_messages_content = [
            {"type": "text", "text": user_content or "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]

    # Get sender name for group attribution
    is_group = chat_type in ("group", "supergroup")
    sender_name = update.effective_user.first_name if is_group else None

    # Store user message text in DB
    await db.add_message(conv["id"], telegram_id, "user", message_content, sender_name=sender_name)

    # Build conversation history from DB (last N messages)
    history = await db.get_messages(conv["id"], limit=max_messages)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[:-1]:  # All except the one we just added
        content = msg["content"]
        if msg["sender_name"] and msg["role"] == "user":
            content = f"[{msg['sender_name']}] {content}"
        messages.append({"role": msg["role"], "content": content})
    # Add current message with potential vision content
    if sender_name and isinstance(openai_messages_content, str):
        openai_messages_content = f"[{sender_name}] {openai_messages_content}"
    messages.append({"role": "user", "content": openai_messages_content})

    # Send typing indicator
    await update.message.chat.send_action(ChatAction.TYPING)

    # Tool call loop: LLM may call tools, we execute and feed results back
    for _round in range(MAX_TOOL_ROUNDS + 1):
        try:
            result = await inference.chat(
                messages=messages, model=model, api_key=api_key, tools=TOOL_DEFINITIONS,
            )
        except Exception as e:
            logger.error(f"Inference error for user {telegram_id}: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your request right now. Please try again in a moment."
            )
            return

        # No tool calls - we have the final text response
        if not result.tool_calls:
            response_text = result.content or ""
            await db.add_message(conv["id"], 0, "assistant", response_text)
            await _split_and_send(update, response_text)
            return

        # Process tool calls
        # Append assistant message with tool calls to conversation
        messages.append(result.model_dump(exclude_none=True))

        for tool_call in result.tool_calls:
            tool_name = tool_call.function.name

            if tool_name == "generate_image":
                # Image gen is special - sends photo directly, doesn't need LLM follow-up
                await _handle_image_tool_call(update, inference, api_key, tool_call, conv, db)
                return

            # Execute other tools and feed results back
            await update.message.chat.send_action(ChatAction.TYPING)
            tool_result = await execute_tool(tool_name, tool_call.function.arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    # If we exhausted tool rounds, send whatever we have
    await update.message.reply_text("Sorry, I took too long processing that request. Please try again.")
