from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": ""}

    telegram_bot_token: str
    libertai_api_key: str
    libertai_api_base_url: str = "https://api.libertai.io/v1"
    default_model: str = "qwen3-code-next"
    free_tier_daily_messages: int = 50
    free_tier_daily_images: int = 5
    bot_encryption_key: str
    max_conversation_messages: int = 20
