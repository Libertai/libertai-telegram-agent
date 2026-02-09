"""Rate limiter service that enforces per-user daily usage limits."""

from __future__ import annotations

from libertai_telegram_agent.database.db import Database


class RateLimiter:
    """Wraps the Database to check and enforce per-user daily limits."""

    def __init__(self, db: Database, daily_messages: int = 50, daily_images: int = 5) -> None:
        self._db = db
        self._daily_messages = daily_messages
        self._daily_images = daily_images

    async def check_and_increment(self, telegram_id: int, usage_type: str) -> tuple[bool, int]:
        """Check if user is within limits. If yes, increment and return (True, remaining). If no, return (False, 0)."""
        usage = await self._db.get_daily_usage(telegram_id)

        if usage_type == "message":
            current = usage["message_count"]
            limit = self._daily_messages
        elif usage_type == "image":
            current = usage["image_count"]
            limit = self._daily_images
        else:
            raise ValueError(f"usage_type must be 'message' or 'image', got '{usage_type}'")

        if current >= limit:
            return False, 0

        await self._db.increment_usage(telegram_id, usage_type)
        remaining = limit - current - 1
        return True, remaining

    async def get_usage_summary(self, telegram_id: int) -> dict:
        """Return dict with messages_used, messages_remaining, images_used, images_remaining."""
        usage = await self._db.get_daily_usage(telegram_id)

        messages_used = usage["message_count"]
        images_used = usage["image_count"]

        return {
            "messages_used": messages_used,
            "messages_remaining": max(0, self._daily_messages - messages_used),
            "images_used": images_used,
            "images_remaining": max(0, self._daily_images - images_used),
        }
