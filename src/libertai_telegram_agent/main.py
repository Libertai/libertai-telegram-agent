import logging

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from libertai_telegram_agent.config import Settings
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.handlers.account import account_command, login_command, logout_command
from libertai_telegram_agent.handlers.chat import handle_message
from libertai_telegram_agent.handlers.commands import (
    help_command,
    model_callback,
    model_command,
    new_command,
    start_command,
    usage_command,
)
from libertai_telegram_agent.handlers.image import image_command
from libertai_telegram_agent.services.inference import InferenceService
from libertai_telegram_agent.services.rate_limiter import RateLimiter

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def post_init(application: Application) -> None:
    db = Database()
    await db.initialize()
    application.bot_data["db"] = db
    application.bot_data["rate_limiter"].db = db
    logger.info("Database initialized")


async def post_shutdown(application: Application) -> None:
    db = application.bot_data.get("db")
    if db:
        await db.close()
    logger.info("Database closed")


def create_application(settings: Settings | None = None) -> Application:
    if settings is None:
        settings = Settings()

    app = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    app.bot_data["settings"] = settings
    app.bot_data["encryption_key"] = settings.bot_encryption_key
    app.bot_data["max_conversation_messages"] = settings.max_conversation_messages
    app.bot_data["inference"] = InferenceService(
        api_base_url=settings.libertai_api_base_url,
        default_api_key=settings.libertai_api_key,
    )
    app.bot_data["rate_limiter"] = RateLimiter(
        db=None,  # Set in post_init
        daily_messages=settings.free_tier_daily_messages,
        daily_images=settings.free_tier_daily_images,
    )

    # Command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("new", new_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("usage", usage_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("login", login_command))
    app.add_handler(CommandHandler("logout", logout_command))
    app.add_handler(CommandHandler("account", account_command))

    # Callback query handler for inline keyboards
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))

    # Message handlers (text and photos)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))

    return app


def main():
    app = create_application()
    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
