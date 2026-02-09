"""Database layer using aiosqlite for SQLite persistence."""

from __future__ import annotations

from datetime import date, datetime, timezone

import aiosqlite


class Database:
    """Async SQLite database wrapper for the Telegram agent."""

    def __init__(self, db_path: str = "libertai_agent.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the database connection, enable WAL mode, and create tables."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._create_tables()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        """Return the active database connection."""
        if self._db is None:
            raise RuntimeError("Database is not initialized. Call initialize() first.")
        return self._db

    async def _create_tables(self) -> None:
        """Create all required tables and indexes."""
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                api_key TEXT,
                default_model TEXT DEFAULT 'qwen3-coder-next',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                chat_type TEXT NOT NULL,
                active BOOLEAN NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_chat_active
                ON conversations (chat_id, active);

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                telegram_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages (conversation_id);

            CREATE TABLE IF NOT EXISTS daily_usage (
                telegram_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                image_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (telegram_id, date)
            );

            CREATE TABLE IF NOT EXISTS group_settings (
                chat_id INTEGER PRIMARY KEY,
                admin_id INTEGER NOT NULL REFERENCES users(telegram_id),
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)

    # ── Generic query helpers ──────────────────────────────────────────

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a query and return all rows as dicts."""
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one(self, query: str, params: tuple = ()) -> dict | None:
        """Execute a query and return one row as a dict, or None."""
        cursor = await self.db.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row is not None else None

    # ── User methods ───────────────────────────────────────────────────

    async def ensure_user(self, telegram_id: int) -> dict:
        """Get an existing user or create a new one. Returns the user dict."""
        user = await self.get_user(telegram_id)
        if user is not None:
            return user
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT OR IGNORE INTO users (telegram_id, created_at, updated_at) VALUES (?, ?, ?)",
            (telegram_id, now, now),
        )
        await self.db.commit()
        return await self.get_user(telegram_id)  # type: ignore[return-value]

    async def get_user(self, telegram_id: int) -> dict | None:
        """Return a user dict by telegram_id, or None if not found."""
        return await self.fetch_one("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))

    async def set_user_api_key(self, telegram_id: int, api_key: str | None) -> None:
        """Set (or clear when None) the API key for a user."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE users SET api_key = ?, updated_at = ? WHERE telegram_id = ?",
            (api_key, now, telegram_id),
        )
        await self.db.commit()

    async def set_user_model(self, telegram_id: int, model: str) -> None:
        """Set the default model for a user."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE users SET default_model = ?, updated_at = ? WHERE telegram_id = ?",
            (model, now, telegram_id),
        )
        await self.db.commit()

    # ── Conversation methods ───────────────────────────────────────────

    async def get_or_create_conversation(self, chat_id: int, chat_type: str) -> dict:
        """Return the active conversation for a chat, or create a new one."""
        conv = await self.fetch_one(
            "SELECT * FROM conversations WHERE chat_id = ? AND active = 1",
            (chat_id,),
        )
        if conv is not None:
            return conv
        return await self._create_conversation(chat_id, chat_type)

    async def create_new_conversation(self, chat_id: int, chat_type: str) -> dict:
        """Deactivate existing conversations for a chat and create a fresh one."""
        await self.db.execute(
            "UPDATE conversations SET active = 0 WHERE chat_id = ? AND active = 1",
            (chat_id,),
        )
        await self.db.commit()
        return await self._create_conversation(chat_id, chat_type)

    async def _create_conversation(self, chat_id: int, chat_type: str) -> dict:
        """Insert a new active conversation and return it as a dict."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self.db.execute(
            "INSERT INTO conversations (chat_id, chat_type, active, created_at) VALUES (?, ?, 1, ?)",
            (chat_id, chat_type, now),
        )
        await self.db.commit()
        return await self.fetch_one("SELECT * FROM conversations WHERE id = ?", (cursor.lastrowid,))  # type: ignore[return-value]

    # ── Message methods ────────────────────────────────────────────────

    async def add_message(self, conversation_id: int, telegram_id: int, role: str, content: str) -> None:
        """Add a message to a conversation."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO messages (conversation_id, telegram_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, telegram_id, role, content, now),
        )
        await self.db.commit()

    async def get_messages(self, conversation_id: int, limit: int = 50) -> list[dict]:
        """Return messages for a conversation ordered by created_at ASC, limited."""
        return await self.fetch_all(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC LIMIT ?",
            (conversation_id, limit),
        )

    # ── Daily usage methods ────────────────────────────────────────────

    async def get_daily_usage(self, telegram_id: int) -> dict:
        """Return today's usage for a user, defaulting counts to 0."""
        today = date.today().isoformat()
        row = await self.fetch_one(
            "SELECT message_count, image_count FROM daily_usage WHERE telegram_id = ? AND date = ?",
            (telegram_id, today),
        )
        if row is not None:
            return row
        return {"message_count": 0, "image_count": 0}

    async def increment_usage(self, telegram_id: int, usage_type: str) -> None:
        """Increment a usage counter for today using UPSERT.

        Args:
            telegram_id: The Telegram user id.
            usage_type: Either "message" or "image".
        """
        if usage_type not in ("message", "image"):
            raise ValueError(f"usage_type must be 'message' or 'image', got '{usage_type}'")

        today = date.today().isoformat()
        column = f"{usage_type}_count"
        await self.db.execute(
            f"""
            INSERT INTO daily_usage (telegram_id, date, {column})
                VALUES (?, ?, 1)
            ON CONFLICT(telegram_id, date)
                DO UPDATE SET {column} = {column} + 1
            """,
            (telegram_id, today),
        )
        await self.db.commit()

    # ── Group settings methods ─────────────────────────────────────────

    async def set_group_admin(self, chat_id: int, admin_id: int) -> None:
        """Set the admin for a group chat using UPSERT."""
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """
            INSERT INTO group_settings (chat_id, admin_id, created_at)
                VALUES (?, ?, ?)
            ON CONFLICT(chat_id)
                DO UPDATE SET admin_id = excluded.admin_id
            """,
            (chat_id, admin_id, now),
        )
        await self.db.commit()

    async def get_group_settings(self, chat_id: int) -> dict | None:
        """Return group settings for a chat_id, or None."""
        return await self.fetch_one("SELECT * FROM group_settings WHERE chat_id = ?", (chat_id,))

    async def remove_group_admin(self, chat_id: int) -> None:
        """Remove the group admin record for a chat."""
        await self.db.execute("DELETE FROM group_settings WHERE chat_id = ?", (chat_id,))
        await self.db.commit()
