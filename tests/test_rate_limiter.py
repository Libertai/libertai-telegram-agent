"""Tests for the RateLimiter service."""

from __future__ import annotations

import pytest

from libertai_telegram_agent.database.db import Database
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
def limiter(db: Database) -> RateLimiter:
    """Create a RateLimiter with small limits for easy testing."""
    return RateLimiter(db, daily_messages=3, daily_images=2)


# ── check_and_increment: messages ──────────────────────────────────────


class TestCheckAndIncrementMessages:
    async def test_allows_message_under_limit(self, limiter: RateLimiter):
        allowed, remaining = await limiter.check_and_increment(1001, "message")
        assert allowed is True
        assert remaining == 2  # 3 limit - 1 used = 2 remaining

    async def test_remaining_decreases_each_call(self, limiter: RateLimiter):
        _, r1 = await limiter.check_and_increment(1001, "message")
        _, r2 = await limiter.check_and_increment(1001, "message")
        _, r3 = await limiter.check_and_increment(1001, "message")
        assert r1 == 2
        assert r2 == 1
        assert r3 == 0

    async def test_blocks_message_at_limit(self, limiter: RateLimiter):
        # Exhaust the limit (3 messages)
        for _ in range(3):
            await limiter.check_and_increment(1001, "message")

        allowed, remaining = await limiter.check_and_increment(1001, "message")
        assert allowed is False
        assert remaining == 0

    async def test_blocks_message_beyond_limit(self, limiter: RateLimiter):
        for _ in range(3):
            await limiter.check_and_increment(1001, "message")

        # Try multiple times past the limit
        for _ in range(3):
            allowed, remaining = await limiter.check_and_increment(1001, "message")
            assert allowed is False
            assert remaining == 0


# ── check_and_increment: images ────────────────────────────────────────


class TestCheckAndIncrementImages:
    async def test_allows_image_under_limit(self, limiter: RateLimiter):
        allowed, remaining = await limiter.check_and_increment(2001, "image")
        assert allowed is True
        assert remaining == 1  # 2 limit - 1 used = 1 remaining

    async def test_blocks_image_at_limit(self, limiter: RateLimiter):
        # Exhaust the limit (2 images)
        for _ in range(2):
            await limiter.check_and_increment(2001, "image")

        allowed, remaining = await limiter.check_and_increment(2001, "image")
        assert allowed is False
        assert remaining == 0


# ── Separate limits per user ───────────────────────────────────────────


class TestSeparateLimitsPerUser:
    async def test_separate_message_limits_per_user(self, limiter: RateLimiter):
        # User A exhausts message limit
        for _ in range(3):
            await limiter.check_and_increment(3001, "message")

        # User A is blocked
        allowed_a, _ = await limiter.check_and_increment(3001, "message")
        assert allowed_a is False

        # User B should still be allowed
        allowed_b, remaining_b = await limiter.check_and_increment(3002, "message")
        assert allowed_b is True
        assert remaining_b == 2

    async def test_separate_image_limits_per_user(self, limiter: RateLimiter):
        # User A exhausts image limit
        for _ in range(2):
            await limiter.check_and_increment(4001, "image")

        # User A is blocked
        allowed_a, _ = await limiter.check_and_increment(4001, "image")
        assert allowed_a is False

        # User B should still be allowed
        allowed_b, remaining_b = await limiter.check_and_increment(4002, "image")
        assert allowed_b is True
        assert remaining_b == 1

    async def test_message_and_image_limits_independent(self, limiter: RateLimiter):
        # Exhaust message limit for user
        for _ in range(3):
            await limiter.check_and_increment(5001, "message")

        # Messages blocked
        allowed_msg, _ = await limiter.check_and_increment(5001, "message")
        assert allowed_msg is False

        # Images still allowed
        allowed_img, remaining_img = await limiter.check_and_increment(5001, "image")
        assert allowed_img is True
        assert remaining_img == 1


# ── get_usage_summary ──────────────────────────────────────────────────


class TestGetUsageSummary:
    async def test_summary_initial_state(self, limiter: RateLimiter):
        summary = await limiter.get_usage_summary(6001)
        assert summary == {
            "messages_used": 0,
            "messages_remaining": 3,
            "images_used": 0,
            "images_remaining": 2,
        }

    async def test_summary_after_some_usage(self, limiter: RateLimiter):
        await limiter.check_and_increment(6002, "message")
        await limiter.check_and_increment(6002, "message")
        await limiter.check_and_increment(6002, "image")

        summary = await limiter.get_usage_summary(6002)
        assert summary == {
            "messages_used": 2,
            "messages_remaining": 1,
            "images_used": 1,
            "images_remaining": 1,
        }

    async def test_summary_at_limit(self, limiter: RateLimiter):
        for _ in range(3):
            await limiter.check_and_increment(6003, "message")
        for _ in range(2):
            await limiter.check_and_increment(6003, "image")

        summary = await limiter.get_usage_summary(6003)
        assert summary == {
            "messages_used": 3,
            "messages_remaining": 0,
            "images_used": 2,
            "images_remaining": 0,
        }

    async def test_summary_does_not_count_rejected_attempts(self, limiter: RateLimiter):
        # Exhaust messages
        for _ in range(3):
            await limiter.check_and_increment(6004, "message")

        # Try (and fail) two more times
        await limiter.check_and_increment(6004, "message")
        await limiter.check_and_increment(6004, "message")

        summary = await limiter.get_usage_summary(6004)
        # Should still show 3 used, not 5
        assert summary["messages_used"] == 3
        assert summary["messages_remaining"] == 0


# ── Default limits ─────────────────────────────────────────────────────


class TestDefaultLimits:
    async def test_default_limits(self, db: Database):
        limiter = RateLimiter(db)
        summary = await limiter.get_usage_summary(7001)
        assert summary["messages_remaining"] == 50
        assert summary["images_remaining"] == 5
