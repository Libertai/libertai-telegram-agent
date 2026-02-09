"""Tests for the Database class."""

from __future__ import annotations

import pytest

from libertai_telegram_agent.database.db import Database


@pytest.fixture
async def db(tmp_path):
    """Create and initialize a temporary in-memory-like database."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.initialize()
    yield database
    await database.close()


# ── Initialization & connection ────────────────────────────────────────


class TestInitialization:
    async def test_initialize_creates_tables(self, db: Database):
        tables = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = sorted(t["name"] for t in tables)
        assert "conversations" in table_names
        assert "daily_usage" in table_names
        assert "group_settings" in table_names
        assert "messages" in table_names
        assert "users" in table_names

    async def test_initialize_creates_indexes(self, db: Database):
        indexes = await db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        )
        index_names = [i["name"] for i in indexes]
        assert "idx_conversations_chat_active" in index_names
        assert "idx_messages_conversation" in index_names

    async def test_wal_mode_enabled(self, db: Database):
        result = await db.fetch_one("PRAGMA journal_mode")
        assert result is not None
        assert result["journal_mode"] == "wal"

    async def test_close_sets_db_to_none(self, tmp_path):
        database = Database(str(tmp_path / "close_test.db"))
        await database.initialize()
        await database.close()
        assert database._db is None

    async def test_db_property_raises_when_not_initialized(self):
        database = Database(":memory:")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = database.db


# ── Generic query helpers ──────────────────────────────────────────────


class TestQueryHelpers:
    async def test_fetch_all_returns_list_of_dicts(self, db: Database):
        await db.ensure_user(1001)
        await db.ensure_user(1002)
        rows = await db.fetch_all("SELECT * FROM users ORDER BY telegram_id")
        assert len(rows) == 2
        assert isinstance(rows[0], dict)
        assert rows[0]["telegram_id"] == 1001
        assert rows[1]["telegram_id"] == 1002

    async def test_fetch_all_empty(self, db: Database):
        rows = await db.fetch_all("SELECT * FROM users")
        assert rows == []

    async def test_fetch_one_returns_dict(self, db: Database):
        await db.ensure_user(42)
        row = await db.fetch_one("SELECT * FROM users WHERE telegram_id = ?", (42,))
        assert isinstance(row, dict)
        assert row["telegram_id"] == 42

    async def test_fetch_one_returns_none_when_missing(self, db: Database):
        result = await db.fetch_one("SELECT * FROM users WHERE telegram_id = ?", (999,))
        assert result is None


# ── User methods ───────────────────────────────────────────────────────


class TestUserMethods:
    async def test_ensure_user_creates_new(self, db: Database):
        user = await db.ensure_user(100)
        assert user["telegram_id"] == 100
        assert user["default_model"] == "gemma-3-27b"
        assert user["api_key"] is None
        assert user["created_at"] is not None
        assert user["updated_at"] is not None

    async def test_ensure_user_returns_existing(self, db: Database):
        user1 = await db.ensure_user(100)
        user2 = await db.ensure_user(100)
        assert user1["telegram_id"] == user2["telegram_id"]
        assert user1["created_at"] == user2["created_at"]

    async def test_get_user_existing(self, db: Database):
        await db.ensure_user(200)
        user = await db.get_user(200)
        assert user is not None
        assert user["telegram_id"] == 200

    async def test_get_user_nonexistent(self, db: Database):
        user = await db.get_user(999)
        assert user is None

    async def test_set_user_api_key(self, db: Database):
        await db.ensure_user(300)
        await db.set_user_api_key(300, "secret-key-123")
        user = await db.get_user(300)
        assert user is not None
        assert user["api_key"] == "secret-key-123"

    async def test_set_user_api_key_to_none_clears(self, db: Database):
        await db.ensure_user(300)
        await db.set_user_api_key(300, "some-key")
        await db.set_user_api_key(300, None)
        user = await db.get_user(300)
        assert user is not None
        assert user["api_key"] is None

    async def test_set_user_api_key_updates_updated_at(self, db: Database):
        user = await db.ensure_user(300)
        original_updated = user["updated_at"]
        await db.set_user_api_key(300, "new-key")
        user = await db.get_user(300)
        assert user is not None
        # updated_at should be changed (or same if test runs within the same second)
        assert user["updated_at"] is not None
        assert user["updated_at"] >= original_updated

    async def test_set_user_model(self, db: Database):
        await db.ensure_user(400)
        await db.set_user_model(400, "llama-3.1-70b")
        user = await db.get_user(400)
        assert user is not None
        assert user["default_model"] == "llama-3.1-70b"

    async def test_set_user_model_updates_updated_at(self, db: Database):
        user = await db.ensure_user(400)
        original_updated = user["updated_at"]
        await db.set_user_model(400, "llama-3.1-70b")
        user = await db.get_user(400)
        assert user is not None
        assert user["updated_at"] >= original_updated


# ── Conversation methods ───────────────────────────────────────────────


class TestConversationMethods:
    async def test_get_or_create_creates_new(self, db: Database):
        conv = await db.get_or_create_conversation(1000, "private")
        assert conv["chat_id"] == 1000
        assert conv["chat_type"] == "private"
        assert conv["active"] == 1
        assert conv["id"] is not None

    async def test_get_or_create_returns_existing(self, db: Database):
        conv1 = await db.get_or_create_conversation(1000, "private")
        conv2 = await db.get_or_create_conversation(1000, "private")
        assert conv1["id"] == conv2["id"]

    async def test_create_new_conversation_deactivates_old(self, db: Database):
        conv1 = await db.get_or_create_conversation(2000, "group")
        conv2 = await db.create_new_conversation(2000, "group")

        assert conv2["id"] != conv1["id"]
        assert conv2["active"] == 1

        # Old conversation should be deactivated
        old = await db.fetch_one("SELECT * FROM conversations WHERE id = ?", (conv1["id"],))
        assert old is not None
        assert old["active"] == 0

    async def test_create_new_conversation_standalone(self, db: Database):
        conv = await db.create_new_conversation(3000, "supergroup")
        assert conv["chat_id"] == 3000
        assert conv["chat_type"] == "supergroup"
        assert conv["active"] == 1

    async def test_multiple_chats_independent(self, db: Database):
        conv_a = await db.get_or_create_conversation(5000, "private")
        conv_b = await db.get_or_create_conversation(6000, "group")
        assert conv_a["id"] != conv_b["id"]
        assert conv_a["chat_id"] == 5000
        assert conv_b["chat_id"] == 6000


# ── Message methods ────────────────────────────────────────────────────


class TestMessageMethods:
    async def test_add_and_get_messages(self, db: Database):
        conv = await db.get_or_create_conversation(7000, "private")
        await db.add_message(conv["id"], 100, "user", "Hello")
        await db.add_message(conv["id"], 0, "assistant", "Hi there!")

        messages = await db.get_messages(conv["id"])
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[0]["telegram_id"] == 100
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    async def test_get_messages_ordered_asc(self, db: Database):
        conv = await db.get_or_create_conversation(8000, "private")
        await db.add_message(conv["id"], 100, "user", "First")
        await db.add_message(conv["id"], 0, "assistant", "Second")
        await db.add_message(conv["id"], 100, "user", "Third")

        messages = await db.get_messages(conv["id"])
        assert [m["content"] for m in messages] == ["First", "Second", "Third"]

    async def test_get_messages_with_limit(self, db: Database):
        conv = await db.get_or_create_conversation(9000, "private")
        for i in range(10):
            await db.add_message(conv["id"], 100, "user", f"Message {i}")

        messages = await db.get_messages(conv["id"], limit=3)
        assert len(messages) == 3
        # Should be the first 3 since ordered by created_at ASC
        assert messages[0]["content"] == "Message 0"
        assert messages[2]["content"] == "Message 2"

    async def test_get_messages_empty(self, db: Database):
        conv = await db.get_or_create_conversation(10000, "private")
        messages = await db.get_messages(conv["id"])
        assert messages == []

    async def test_messages_scoped_to_conversation(self, db: Database):
        conv1 = await db.get_or_create_conversation(11000, "private")
        conv2 = await db.get_or_create_conversation(12000, "private")
        await db.add_message(conv1["id"], 100, "user", "In conv1")
        await db.add_message(conv2["id"], 200, "user", "In conv2")

        msgs1 = await db.get_messages(conv1["id"])
        msgs2 = await db.get_messages(conv2["id"])
        assert len(msgs1) == 1
        assert len(msgs2) == 1
        assert msgs1[0]["content"] == "In conv1"
        assert msgs2[0]["content"] == "In conv2"


# ── Daily usage methods ────────────────────────────────────────────────


class TestDailyUsageMethods:
    async def test_get_daily_usage_defaults(self, db: Database):
        usage = await db.get_daily_usage(500)
        assert usage["message_count"] == 0
        assert usage["image_count"] == 0

    async def test_increment_message_usage(self, db: Database):
        await db.increment_usage(500, "message")
        usage = await db.get_daily_usage(500)
        assert usage["message_count"] == 1
        assert usage["image_count"] == 0

    async def test_increment_image_usage(self, db: Database):
        await db.increment_usage(500, "image")
        usage = await db.get_daily_usage(500)
        assert usage["message_count"] == 0
        assert usage["image_count"] == 1

    async def test_increment_multiple_times(self, db: Database):
        for _ in range(5):
            await db.increment_usage(600, "message")
        for _ in range(3):
            await db.increment_usage(600, "image")
        usage = await db.get_daily_usage(600)
        assert usage["message_count"] == 5
        assert usage["image_count"] == 3

    async def test_increment_invalid_type_raises(self, db: Database):
        with pytest.raises(ValueError, match="usage_type must be"):
            await db.increment_usage(500, "video")

    async def test_usage_per_user_independent(self, db: Database):
        await db.increment_usage(700, "message")
        await db.increment_usage(701, "message")
        await db.increment_usage(701, "message")

        usage_700 = await db.get_daily_usage(700)
        usage_701 = await db.get_daily_usage(701)
        assert usage_700["message_count"] == 1
        assert usage_701["message_count"] == 2


# ── Group settings methods ─────────────────────────────────────────────


class TestGroupSettingsMethods:
    async def test_set_and_get_group_admin(self, db: Database):
        await db.ensure_user(800)
        await db.set_group_admin(-1001, 800)
        settings = await db.get_group_settings(-1001)
        assert settings is not None
        assert settings["chat_id"] == -1001
        assert settings["admin_id"] == 800
        assert settings["created_at"] is not None

    async def test_get_group_settings_nonexistent(self, db: Database):
        settings = await db.get_group_settings(-9999)
        assert settings is None

    async def test_set_group_admin_upsert(self, db: Database):
        await db.ensure_user(800)
        await db.ensure_user(801)
        await db.set_group_admin(-1002, 800)
        await db.set_group_admin(-1002, 801)
        settings = await db.get_group_settings(-1002)
        assert settings is not None
        assert settings["admin_id"] == 801

    async def test_remove_group_admin(self, db: Database):
        await db.ensure_user(900)
        await db.set_group_admin(-2000, 900)
        await db.remove_group_admin(-2000)
        settings = await db.get_group_settings(-2000)
        assert settings is None

    async def test_remove_group_admin_nonexistent_no_error(self, db: Database):
        # Should not raise
        await db.remove_group_admin(-9999)
