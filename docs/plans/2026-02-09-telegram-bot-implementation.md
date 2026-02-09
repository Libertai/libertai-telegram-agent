# LibertAI Telegram Bot - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Telegram bot that provides free and paid access to LibertAI's decentralized AI inference (chat, image gen, vision) with per-user rate limiting and persistent conversations.

**Architecture:** Single async Python process. `python-telegram-bot` handles Telegram I/O, `openai` SDK talks to LibertAI's OpenAI-compatible API, `aiosqlite` persists conversations/users/usage in SQLite. Bot owns one API key for free tier; connected users supply their own.

**Tech Stack:** Python 3.12+, python-telegram-bot, openai, httpx, aiosqlite, cryptography, pydantic-settings

**Design doc:** `docs/plans/2026-02-09-telegram-bot-design.md`

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/libertai_telegram_agent/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "libertai-telegram-agent"
version = "0.1.0"
description = "Telegram bot for LibertAI decentralized AI inference"
requires-python = ">=3.12"
dependencies = [
    "python-telegram-bot[ext]>=22.0",
    "openai>=1.0",
    "httpx>=0.27",
    "aiosqlite>=0.20",
    "cryptography>=44.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.9",
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create .env.example**

```
TELEGRAM_BOT_TOKEN=your-bot-token-from-botfather
LIBERTAI_API_KEY=your-api-key-from-console-libertai-io
LIBERTAI_API_BASE_URL=https://api.libertai.io/v1
DEFAULT_MODEL=gemma-3-27b
FREE_TIER_DAILY_MESSAGES=50
FREE_TIER_DAILY_IMAGES=5
BOT_ENCRYPTION_KEY=generate-with-python-c-from-cryptography-fernet-import-Fernet-print-Fernet-generate-key-decode
MAX_CONVERSATION_MESSAGES=20
```

**Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.env
*.db
.venv/
dist/
*.egg-info/
.ruff_cache/
.pytest_cache/
```

**Step 4: Create package init**

Create empty `src/libertai_telegram_agent/__init__.py`.

**Step 5: Install dependencies**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv venv && uv pip install -e ".[dev]"`
Expected: All deps installed successfully.

**Step 6: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/
git commit -m "feat: scaffold project with dependencies and config"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/libertai_telegram_agent/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest


def test_config_loads_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("LIBERTAI_API_KEY", "test-key")
    monkeypatch.setenv("BOT_ENCRYPTION_KEY", "dGVzdC1lbmNyeXB0aW9uLWtleS1mb3ItdGVzdGluZzE=")

    from libertai_telegram_agent.config import Settings

    settings = Settings()
    assert settings.telegram_bot_token == "test-token"
    assert settings.libertai_api_key == "test-key"
    assert settings.libertai_api_base_url == "https://api.libertai.io/v1"
    assert settings.default_model == "gemma-3-27b"
    assert settings.free_tier_daily_messages == 50
    assert settings.free_tier_daily_images == 5
    assert settings.max_conversation_messages == 20


def test_config_requires_telegram_token(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("LIBERTAI_API_KEY", raising=False)
    monkeypatch.delenv("BOT_ENCRYPTION_KEY", raising=False)

    from pydantic import ValidationError
    from libertai_telegram_agent.config import Settings

    with pytest.raises(ValidationError):
        Settings()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_config.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    libertai_api_key: str
    libertai_api_base_url: str = "https://api.libertai.io/v1"
    default_model: str = "gemma-3-27b"
    free_tier_daily_messages: int = 50
    free_tier_daily_images: int = 5
    bot_encryption_key: str
    max_conversation_messages: int = 20

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_config.py -v`
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/config.py tests/test_config.py
git commit -m "feat: add config module with env var loading"
```

---

### Task 3: Database Layer

**Files:**
- Create: `src/libertai_telegram_agent/database/__init__.py`
- Create: `src/libertai_telegram_agent/database/db.py`
- Create: `src/libertai_telegram_agent/database/models.py`
- Create: `tests/test_database.py`

**Step 1: Write failing tests**

```python
# tests/test_database.py
import pytest
from libertai_telegram_agent.database.db import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


async def test_initialize_creates_tables(db):
    """Tables should exist after init."""
    rows = await db.fetch_all("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    table_names = [r["name"] for r in rows]
    assert "users" in table_names
    assert "conversations" in table_names
    assert "messages" in table_names
    assert "daily_usage" in table_names
    assert "group_settings" in table_names


async def test_ensure_user_creates_new(db):
    user = await db.ensure_user(12345)
    assert user["telegram_id"] == 12345
    assert user["api_key"] is None
    assert user["default_model"] == "gemma-3-27b"


async def test_ensure_user_returns_existing(db):
    await db.ensure_user(12345)
    user = await db.ensure_user(12345)
    assert user["telegram_id"] == 12345


async def test_set_and_get_api_key(db):
    await db.ensure_user(12345)
    await db.set_user_api_key(12345, "encrypted-key")
    user = await db.get_user(12345)
    assert user["api_key"] == "encrypted-key"


async def test_clear_api_key(db):
    await db.ensure_user(12345)
    await db.set_user_api_key(12345, "encrypted-key")
    await db.set_user_api_key(12345, None)
    user = await db.get_user(12345)
    assert user["api_key"] is None


async def test_create_and_get_active_conversation(db):
    conv = await db.get_or_create_conversation(chat_id=100, chat_type="private")
    assert conv["chat_id"] == 100
    assert conv["active"] == 1

    # Getting again returns same conversation
    conv2 = await db.get_or_create_conversation(chat_id=100, chat_type="private")
    assert conv2["id"] == conv["id"]


async def test_new_conversation_deactivates_old(db):
    conv1 = await db.get_or_create_conversation(chat_id=100, chat_type="private")
    conv2 = await db.create_new_conversation(chat_id=100, chat_type="private")
    assert conv2["id"] != conv1["id"]

    # Old one should be inactive
    old = await db.fetch_one("SELECT active FROM conversations WHERE id = ?", (conv1["id"],))
    assert old["active"] == 0


async def test_add_and_get_messages(db):
    conv = await db.get_or_create_conversation(chat_id=100, chat_type="private")
    await db.add_message(conv["id"], telegram_id=12345, role="user", content="Hello")
    await db.add_message(conv["id"], telegram_id=0, role="assistant", content="Hi there")

    messages = await db.get_messages(conv["id"], limit=20)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


async def test_daily_usage_tracking(db):
    usage = await db.get_daily_usage(12345)
    assert usage["message_count"] == 0
    assert usage["image_count"] == 0

    await db.increment_usage(12345, "message")
    await db.increment_usage(12345, "message")
    await db.increment_usage(12345, "image")

    usage = await db.get_daily_usage(12345)
    assert usage["message_count"] == 2
    assert usage["image_count"] == 1


async def test_group_settings(db):
    await db.ensure_user(12345)
    await db.set_group_admin(chat_id=-100, admin_id=12345)
    settings = await db.get_group_settings(-100)
    assert settings["admin_id"] == 12345


async def test_group_settings_none_when_unset(db):
    settings = await db.get_group_settings(-100)
    assert settings is None


async def test_remove_group_admin(db):
    await db.ensure_user(12345)
    await db.set_group_admin(chat_id=-100, admin_id=12345)
    await db.remove_group_admin(chat_id=-100)
    settings = await db.get_group_settings(-100)
    assert settings is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_database.py -v`
Expected: FAIL — module not found.

**Step 3: Write Database class**

```python
# src/libertai_telegram_agent/database/__init__.py
```

```python
# src/libertai_telegram_agent/database/db.py
import aiosqlite
from datetime import date, datetime, timezone


class Database:
    def __init__(self, db_path: str = "bot.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self):
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._create_tables()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def _create_tables(self):
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id   INTEGER PRIMARY KEY,
                api_key       TEXT,
                default_model TEXT DEFAULT 'gemma-3-27b',
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id       INTEGER NOT NULL,
                chat_type     TEXT NOT NULL,
                active        BOOLEAN DEFAULT 1,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_chat_id_active
                ON conversations(chat_id, active);

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                telegram_id     INTEGER,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id);

            CREATE TABLE IF NOT EXISTS daily_usage (
                telegram_id   INTEGER NOT NULL,
                date          DATE NOT NULL,
                message_count INTEGER DEFAULT 0,
                image_count   INTEGER DEFAULT 0,
                PRIMARY KEY (telegram_id, date)
            );

            CREATE TABLE IF NOT EXISTS group_settings (
                chat_id       INTEGER PRIMARY KEY,
                admin_id      INTEGER NOT NULL REFERENCES users(telegram_id),
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await self._conn.commit()

    async def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def fetch_one(self, query: str, params: tuple = ()) -> dict | None:
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def ensure_user(self, telegram_id: int) -> dict:
        user = await self.get_user(telegram_id)
        if user:
            return user
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO users (telegram_id, created_at, updated_at) VALUES (?, ?, ?)",
            (telegram_id, now, now),
        )
        await self._conn.commit()
        return await self.get_user(telegram_id)

    async def get_user(self, telegram_id: int) -> dict | None:
        return await self.fetch_one("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))

    async def set_user_api_key(self, telegram_id: int, api_key: str | None):
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "UPDATE users SET api_key = ?, updated_at = ? WHERE telegram_id = ?",
            (api_key, now, telegram_id),
        )
        await self._conn.commit()

    async def set_user_model(self, telegram_id: int, model: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "UPDATE users SET default_model = ?, updated_at = ? WHERE telegram_id = ?",
            (model, now, telegram_id),
        )
        await self._conn.commit()

    async def get_or_create_conversation(self, chat_id: int, chat_type: str) -> dict:
        conv = await self.fetch_one(
            "SELECT * FROM conversations WHERE chat_id = ? AND active = 1", (chat_id,)
        )
        if conv:
            return conv
        return await self.create_new_conversation(chat_id, chat_type)

    async def create_new_conversation(self, chat_id: int, chat_type: str) -> dict:
        # Deactivate existing conversations for this chat
        await self._conn.execute(
            "UPDATE conversations SET active = 0 WHERE chat_id = ? AND active = 1", (chat_id,)
        )
        now = datetime.now(timezone.utc).isoformat()
        cursor = await self._conn.execute(
            "INSERT INTO conversations (chat_id, chat_type, created_at) VALUES (?, ?, ?)",
            (chat_id, chat_type, now),
        )
        await self._conn.commit()
        return await self.fetch_one("SELECT * FROM conversations WHERE id = ?", (cursor.lastrowid,))

    async def add_message(self, conversation_id: int, telegram_id: int, role: str, content: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO messages (conversation_id, telegram_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, telegram_id, role, content, now),
        )
        await self._conn.commit()

    async def get_messages(self, conversation_id: int, limit: int = 20) -> list[dict]:
        return await self.fetch_all(
            """SELECT * FROM messages WHERE conversation_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (conversation_id, limit),
        )

    async def get_daily_usage(self, telegram_id: int) -> dict:
        today = date.today().isoformat()
        usage = await self.fetch_one(
            "SELECT * FROM daily_usage WHERE telegram_id = ? AND date = ?",
            (telegram_id, today),
        )
        if usage:
            return usage
        return {"telegram_id": telegram_id, "date": today, "message_count": 0, "image_count": 0}

    async def increment_usage(self, telegram_id: int, usage_type: str):
        today = date.today().isoformat()
        column = "message_count" if usage_type == "message" else "image_count"
        await self._conn.execute(
            f"""INSERT INTO daily_usage (telegram_id, date, {column})
                VALUES (?, ?, 1)
                ON CONFLICT(telegram_id, date) DO UPDATE SET {column} = {column} + 1""",
            (telegram_id, today),
        )
        await self._conn.commit()

    async def set_group_admin(self, chat_id: int, admin_id: int):
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            """INSERT INTO group_settings (chat_id, admin_id, created_at) VALUES (?, ?, ?)
               ON CONFLICT(chat_id) DO UPDATE SET admin_id = ?""",
            (chat_id, admin_id, now, admin_id),
        )
        await self._conn.commit()

    async def get_group_settings(self, chat_id: int) -> dict | None:
        return await self.fetch_one("SELECT * FROM group_settings WHERE chat_id = ?", (chat_id,))

    async def remove_group_admin(self, chat_id: int):
        await self._conn.execute("DELETE FROM group_settings WHERE chat_id = ?", (chat_id,))
        await self._conn.commit()
```

`src/libertai_telegram_agent/database/models.py` is unused for now — the `Database` class handles everything. Keep as empty file or remove from structure.

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_database.py -v`
Expected: All 11 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/database/ tests/test_database.py
git commit -m "feat: add database layer with SQLite persistence"
```

---

### Task 4: Rate Limiter Service

**Files:**
- Create: `src/libertai_telegram_agent/services/__init__.py`
- Create: `src/libertai_telegram_agent/services/rate_limiter.py`
- Create: `tests/test_rate_limiter.py`

**Step 1: Write failing tests**

```python
# tests/test_rate_limiter.py
import pytest
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def limiter(db):
    return RateLimiter(db, daily_messages=3, daily_images=2)


async def test_allows_message_under_limit(limiter):
    allowed, remaining = await limiter.check_and_increment(12345, "message")
    assert allowed is True
    assert remaining == 2


async def test_blocks_message_at_limit(limiter):
    for _ in range(3):
        await limiter.check_and_increment(12345, "message")
    allowed, remaining = await limiter.check_and_increment(12345, "message")
    assert allowed is False
    assert remaining == 0


async def test_allows_image_under_limit(limiter):
    allowed, remaining = await limiter.check_and_increment(12345, "image")
    assert allowed is True
    assert remaining == 1


async def test_blocks_image_at_limit(limiter):
    for _ in range(2):
        await limiter.check_and_increment(12345, "image")
    allowed, remaining = await limiter.check_and_increment(12345, "image")
    assert allowed is False
    assert remaining == 0


async def test_separate_limits_per_user(limiter):
    for _ in range(3):
        await limiter.check_and_increment(111, "message")

    # Different user should still be allowed
    allowed, _ = await limiter.check_and_increment(222, "message")
    assert allowed is True


async def test_get_usage_summary(limiter):
    await limiter.check_and_increment(12345, "message")
    await limiter.check_and_increment(12345, "image")
    summary = await limiter.get_usage_summary(12345)
    assert summary["messages_used"] == 1
    assert summary["messages_remaining"] == 2
    assert summary["images_used"] == 1
    assert summary["images_remaining"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_rate_limiter.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/services/__init__.py
```

```python
# src/libertai_telegram_agent/services/rate_limiter.py
from libertai_telegram_agent.database.db import Database


class RateLimiter:
    def __init__(self, db: Database, daily_messages: int = 50, daily_images: int = 5):
        self.db = db
        self.daily_messages = daily_messages
        self.daily_images = daily_images

    async def check_and_increment(self, telegram_id: int, usage_type: str) -> tuple[bool, int]:
        """Check if user is within limits. If yes, increment and return (True, remaining). If no, return (False, 0)."""
        usage = await self.db.get_daily_usage(telegram_id)

        if usage_type == "message":
            current = usage["message_count"]
            limit = self.daily_messages
        else:
            current = usage["image_count"]
            limit = self.daily_images

        if current >= limit:
            return False, 0

        await self.db.increment_usage(telegram_id, usage_type)
        remaining = limit - current - 1
        return True, remaining

    async def get_usage_summary(self, telegram_id: int) -> dict:
        usage = await self.db.get_daily_usage(telegram_id)
        return {
            "messages_used": usage["message_count"],
            "messages_remaining": max(0, self.daily_messages - usage["message_count"]),
            "images_used": usage["image_count"],
            "images_remaining": max(0, self.daily_images - usage["image_count"]),
        }
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_rate_limiter.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/services/ tests/test_rate_limiter.py
git commit -m "feat: add rate limiter with per-user daily limits"
```

---

### Task 5: Inference Service (Chat + Image)

**Files:**
- Create: `src/libertai_telegram_agent/services/inference.py`
- Create: `tests/test_inference.py`

**Step 1: Write failing tests**

Note: These tests mock the OpenAI client and httpx since we can't hit real APIs in tests.

```python
# tests/test_inference.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from libertai_telegram_agent.services.inference import InferenceService


@pytest.fixture
def service():
    return InferenceService(
        api_base_url="https://api.libertai.io/v1",
        default_api_key="test-bot-key",
    )


def test_creates_openai_client_with_default_key(service):
    client = service.get_client()
    assert client.api_key == "test-bot-key"
    assert client.base_url.host == "api.libertai.io"


def test_creates_openai_client_with_custom_key(service):
    client = service.get_client(api_key="user-key")
    assert client.api_key == "user-key"


async def test_chat_returns_response_text(service):
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Hello! How can I help?"

    with patch.object(service, "get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_get.return_value = mock_client

        result = await service.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemma-3-27b",
        )
        assert result == "Hello! How can I help?"


async def test_chat_passes_api_key(service):
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "response"

    with patch.object(service, "get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_get.return_value = mock_client

        await service.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="gemma-3-27b",
            api_key="user-key",
        )
        mock_get.assert_called_once_with(api_key="user-key")


async def test_generate_image_returns_bytes(service):
    import base64

    fake_image = base64.b64encode(b"fake-image-data").decode()
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"b64_json": fake_image}]}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await service.generate_image("a cat", api_key="test-key")
        assert result == b"fake-image-data"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_inference.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/services/inference.py
import base64
import asyncio
import logging

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = {
    "gemma-3-27b": {"name": "Gemma 3 27B", "tier": "free", "vision": True},
    "hermes-3-8b-tee": {"name": "Hermes 3 8B (TEE)", "tier": "free", "vision": False},
    "glm-4.7": {"name": "GLM 4.7", "tier": "pro", "vision": False},
}

IMAGE_MODEL = "z-image-turbo"


class InferenceService:
    def __init__(self, api_base_url: str, default_api_key: str):
        self.api_base_url = api_base_url
        self.default_api_key = default_api_key

    def get_client(self, api_key: str | None = None) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=api_key or self.default_api_key,
        )

    async def chat(
        self,
        messages: list[dict],
        model: str,
        api_key: str | None = None,
    ) -> str:
        client = self.get_client(api_key=api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Chat API error: {e}")
            # Retry once after 2s
            await asyncio.sleep(2)
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                return response.choices[0].message.content or ""
            except Exception as retry_error:
                logger.error(f"Chat API retry failed: {retry_error}")
                raise

    async def generate_image(
        self,
        prompt: str,
        api_key: str | None = None,
        size: str = "1024x1024",
    ) -> bytes:
        key = api_key or self.default_api_key
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base_url}/images/generations",
                json={
                    "model": IMAGE_MODEL,
                    "prompt": prompt,
                    "size": size,
                    "n": 1,
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            b64_image = data["data"][0]["b64_json"]
            return base64.b64decode(b64_image)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_inference.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/services/inference.py tests/test_inference.py
git commit -m "feat: add inference service for chat and image generation"
```

---

### Task 6: Encryption Utility

**Files:**
- Create: `src/libertai_telegram_agent/services/encryption.py`
- Create: `tests/test_encryption.py`

**Step 1: Write failing tests**

```python
# tests/test_encryption.py
from cryptography.fernet import Fernet
from libertai_telegram_agent.services.encryption import encrypt_api_key, decrypt_api_key


def test_encrypt_decrypt_roundtrip():
    key = Fernet.generate_key().decode()
    original = "sk-test-api-key-12345"
    encrypted = encrypt_api_key(original, key)
    assert encrypted != original
    decrypted = decrypt_api_key(encrypted, key)
    assert decrypted == original


def test_encrypted_value_is_different_each_time():
    key = Fernet.generate_key().decode()
    original = "sk-test-api-key"
    enc1 = encrypt_api_key(original, key)
    enc2 = encrypt_api_key(original, key)
    # Fernet includes a timestamp, so encryptions differ
    assert enc1 != enc2
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_encryption.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/services/encryption.py
from cryptography.fernet import Fernet


def encrypt_api_key(api_key: str, encryption_key: str) -> str:
    f = Fernet(encryption_key.encode())
    return f.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str, encryption_key: str) -> str:
    f = Fernet(encryption_key.encode())
    return f.decrypt(encrypted_key.encode()).decode()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_encryption.py -v`
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/services/encryption.py tests/test_encryption.py
git commit -m "feat: add encryption utility for API key storage"
```

---

### Task 7: Bot Handlers — Commands (/start, /new, /help, /model, /usage)

**Files:**
- Create: `src/libertai_telegram_agent/handlers/__init__.py`
- Create: `src/libertai_telegram_agent/handlers/commands.py`
- Create: `tests/test_commands.py`

**Step 1: Write failing tests**

Tests use `python-telegram-bot`'s test utilities to simulate updates.

```python
# tests/test_commands.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from libertai_telegram_agent.handlers.commands import (
    start_command,
    new_command,
    help_command,
    model_command,
    usage_command,
)
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter(db):
    return RateLimiter(db, daily_messages=50, daily_images=5)


def make_context(db, rate_limiter):
    context = MagicMock()
    context.bot_data = {"db": db, "rate_limiter": rate_limiter}
    context.bot = MagicMock()
    context.bot.username = "test_bot"
    return context


def make_update(user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.reply_text = AsyncMock()
    update.message.reply_markdown_v2 = AsyncMock()
    return update


async def test_start_command_sends_welcome(db, rate_limiter):
    update = make_update()
    context = make_context(db, rate_limiter)
    await start_command(update, context)
    update.message.reply_text.assert_called_once()
    call_text = update.message.reply_text.call_args[0][0]
    assert "LibertAI" in call_text


async def test_new_command_creates_conversation(db, rate_limiter):
    update = make_update()
    context = make_context(db, rate_limiter)
    # Create initial conversation
    await db.get_or_create_conversation(12345, "private")
    await new_command(update, context)
    update.message.reply_text.assert_called_once()
    call_text = update.message.reply_text.call_args[0][0]
    assert "new" in call_text.lower() or "conversation" in call_text.lower()


async def test_help_command_lists_commands(db, rate_limiter):
    update = make_update()
    context = make_context(db, rate_limiter)
    await help_command(update, context)
    update.message.reply_text.assert_called_once()
    call_text = update.message.reply_text.call_args[0][0]
    assert "/image" in call_text
    assert "/login" in call_text


async def test_usage_command_shows_remaining(db, rate_limiter):
    update = make_update()
    context = make_context(db, rate_limiter)
    await usage_command(update, context)
    update.message.reply_text.assert_called_once()
    call_text = update.message.reply_text.call_args[0][0]
    assert "50" in call_text or "messages" in call_text.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_commands.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/handlers/__init__.py
```

```python
# src/libertai_telegram_agent/handlers/commands.py
import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter
from libertai_telegram_agent.services.inference import AVAILABLE_MODELS

logger = logging.getLogger(__name__)


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


def _get_rate_limiter(context: ContextTypes.DEFAULT_TYPE) -> RateLimiter:
    return context.bot_data["rate_limiter"]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    db = _get_db(context)
    await db.create_new_conversation(update.effective_chat.id, update.effective_chat.type)
    await update.message.reply_text("Started a new conversation. Previous context has been cleared.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_commands.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/handlers/ tests/test_commands.py
git commit -m "feat: add command handlers (start, new, help, model, usage)"
```

---

### Task 8: Bot Handlers — Account (/login, /logout, /account)

**Files:**
- Create: `src/libertai_telegram_agent/handlers/account.py`
- Create: `tests/test_account.py`

**Step 1: Write failing tests**

```python
# tests/test_account.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.fernet import Fernet
from libertai_telegram_agent.handlers.account import login_command, logout_command, account_command
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def encryption_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def rate_limiter(db):
    return RateLimiter(db)


def make_context(db, rate_limiter, encryption_key):
    context = MagicMock()
    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "encryption_key": encryption_key,
        "inference": MagicMock(),
    }
    context.args = []
    context.bot = MagicMock()
    return context


def make_update(user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    return update


async def test_login_no_key_shows_usage(db, rate_limiter, encryption_key):
    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    context.args = []
    await login_command(update, context)
    call_text = update.message.reply_text.call_args[0][0]
    assert "Usage" in call_text or "/login" in call_text


async def test_login_deletes_message(db, rate_limiter, encryption_key):
    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    context.args = ["test-api-key"]

    # Mock the validation to succeed
    with patch("libertai_telegram_agent.handlers.account.validate_api_key", new_callable=AsyncMock, return_value=10.5):
        await login_command(update, context)

    update.message.delete.assert_called_once()


async def test_login_stores_encrypted_key(db, rate_limiter, encryption_key):
    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    context.args = ["test-api-key"]

    with patch("libertai_telegram_agent.handlers.account.validate_api_key", new_callable=AsyncMock, return_value=10.5):
        await login_command(update, context)

    user = await db.get_user(12345)
    assert user["api_key"] is not None
    assert user["api_key"] != "test-api-key"  # Should be encrypted


async def test_logout_clears_key(db, rate_limiter, encryption_key):
    await db.ensure_user(12345)
    await db.set_user_api_key(12345, "some-encrypted-key")

    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    await logout_command(update, context)

    user = await db.get_user(12345)
    assert user["api_key"] is None


async def test_account_shows_connected(db, rate_limiter, encryption_key):
    await db.ensure_user(12345)
    await db.set_user_api_key(12345, "some-encrypted-key")

    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    await account_command(update, context)

    call_text = update.message.reply_text.call_args[0][0]
    assert "Connected" in call_text or "connected" in call_text


async def test_account_shows_free_tier(db, rate_limiter, encryption_key):
    await db.ensure_user(12345)

    update = make_update()
    context = make_context(db, rate_limiter, encryption_key)
    await account_command(update, context)

    call_text = update.message.reply_text.call_args[0][0]
    assert "free" in call_text.lower() or "Free" in call_text
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_account.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/handlers/account.py
import logging

import httpx
from telegram import Update
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.encryption import encrypt_api_key, decrypt_api_key

logger = logging.getLogger(__name__)


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


async def validate_api_key(api_base_url: str, api_key: str) -> float | None:
    """Validate an API key by checking the credit balance. Returns balance or None if invalid."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{api_base_url}/credits/balance",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("balance", 0.0)
            return None
    except Exception as e:
        logger.error(f"API key validation error: {e}")
        return None


async def login_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = _get_db(context)
    encryption_key = context.bot_data["encryption_key"]

    if not context.args:
        await update.message.reply_text(
            "Usage: /login <your-api-key>\n\n"
            "Get your API key at https://console.libertai.io\n"
            "Your message will be deleted immediately to protect your key."
        )
        return

    api_key = context.args[0]

    # Delete the message containing the API key
    try:
        await update.message.delete()
    except Exception:
        logger.warning("Could not delete message with API key")

    await db.ensure_user(update.effective_user.id)

    # Validate the key
    inference = context.bot_data["inference"]
    balance = await validate_api_key(inference.api_base_url, api_key)

    if balance is None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Invalid API key. Please check your key at https://console.libertai.io",
        )
        return

    # Encrypt and store
    encrypted = encrypt_api_key(api_key, encryption_key)
    await db.set_user_api_key(update.effective_user.id, encrypted)

    # If in a group, set as group admin
    if update.effective_chat.type in ("group", "supergroup"):
        await db.set_group_admin(update.effective_chat.id, update.effective_user.id)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Connected successfully! Credit balance: {balance:.4f}\n"
        f"You now have unlimited access and can use pro models.",
    )


async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = _get_db(context)
    await db.ensure_user(update.effective_user.id)
    await db.set_user_api_key(update.effective_user.id, None)

    # If in a group, remove group admin link
    if update.effective_chat.type in ("group", "supergroup"):
        await db.remove_group_admin(update.effective_chat.id)

    await update.message.reply_text("Disconnected from LibertAI. You're back on the free tier.")


async def account_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = _get_db(context)
    user = await db.ensure_user(update.effective_user.id)

    if user["api_key"] is not None:
        # Try to fetch balance
        encryption_key = context.bot_data["encryption_key"]
        inference = context.bot_data["inference"]
        try:
            decrypted_key = decrypt_api_key(user["api_key"], encryption_key)
            balance = await validate_api_key(inference.api_base_url, decrypted_key)
            balance_text = f"\nCredit balance: {balance:.4f}" if balance is not None else ""
        except Exception:
            balance_text = "\n(Could not fetch balance)"

        await update.message.reply_text(
            f"Connected to LibertAI.{balance_text}\n"
            f"Model: {user['default_model']}\n"
            f"No daily limits apply.\n\n"
            f"Use /logout to disconnect."
        )
    else:
        rate_limiter = context.bot_data["rate_limiter"]
        summary = await rate_limiter.get_usage_summary(update.effective_user.id)
        await update.message.reply_text(
            f"Free tier (not connected).\n"
            f"Model: {user['default_model']}\n"
            f"Messages today: {summary['messages_used']} ({summary['messages_remaining']} remaining)\n"
            f"Images today: {summary['images_used']} ({summary['images_remaining']} remaining)\n\n"
            f"Use /login <api-key> to connect your account."
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_account.py -v`
Expected: All 7 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/handlers/account.py tests/test_account.py
git commit -m "feat: add account handlers (login, logout, account)"
```

---

### Task 9: Bot Handlers — Chat (message handling + vision)

**Files:**
- Create: `src/libertai_telegram_agent/handlers/chat.py`
- Create: `tests/test_chat.py`

**Step 1: Write failing tests**

```python
# tests/test_chat.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cryptography.fernet import Fernet
from libertai_telegram_agent.handlers.chat import handle_message, should_respond_in_group
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter
from libertai_telegram_agent.services.inference import InferenceService


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter(db):
    return RateLimiter(db, daily_messages=50, daily_images=5)


@pytest.fixture
def encryption_key():
    return Fernet.generate_key().decode()


@pytest.fixture
def inference():
    return MagicMock(spec=InferenceService)


def make_context(db, rate_limiter, inference, encryption_key, max_messages=20):
    context = MagicMock()
    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "inference": inference,
        "encryption_key": encryption_key,
        "max_conversation_messages": max_messages,
    }
    context.bot = MagicMock()
    context.bot.username = "test_bot"
    return context


def make_update(text="Hello", user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.first_name = "Test"
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.text = text
    update.message.photo = None
    update.message.caption = None
    update.message.reply_to_message = None
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


def test_should_respond_in_group_when_mentioned():
    update = make_update(text="@test_bot what is AI?", chat_type="group")
    assert should_respond_in_group(update, "test_bot") is True


def test_should_not_respond_in_group_without_mention():
    update = make_update(text="random message", chat_type="group")
    assert should_respond_in_group(update, "test_bot") is False


def test_should_respond_in_group_when_replied_to():
    update = make_update(text="what do you think?", chat_type="group")
    update.message.reply_to_message = MagicMock()
    update.message.reply_to_message.from_user.is_bot = True
    update.message.reply_to_message.from_user.username = "test_bot"
    assert should_respond_in_group(update, "test_bot") is True


async def test_handle_message_sends_response(db, rate_limiter, inference, encryption_key):
    update = make_update(text="Hello")
    context = make_context(db, rate_limiter, inference, encryption_key)
    inference.chat = AsyncMock(return_value="Hi there!")

    await handle_message(update, context)

    update.message.reply_text.assert_called()
    call_text = update.message.reply_text.call_args[0][0]
    assert "Hi there!" in call_text


async def test_handle_message_rate_limited(db, rate_limiter, inference, encryption_key):
    limiter = RateLimiter(db, daily_messages=1, daily_images=0)
    update = make_update(text="First")
    context = make_context(db, limiter, inference, encryption_key)
    inference.chat = AsyncMock(return_value="response")

    await handle_message(update, context)  # Uses the 1 allowed message

    update2 = make_update(text="Second")
    await handle_message(update2, context)

    # Second call should mention limit
    last_call = update2.message.reply_text.call_args[0][0]
    assert "limit" in last_call.lower() or "console.libertai.io" in last_call


async def test_handle_message_skips_group_without_mention(db, rate_limiter, inference, encryption_key):
    update = make_update(text="random chat", chat_type="group")
    context = make_context(db, rate_limiter, inference, encryption_key)

    await handle_message(update, context)

    update.message.reply_text.assert_not_called()
    inference.chat.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_chat.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/handlers/chat.py
import base64
import logging

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.encryption import decrypt_api_key
from libertai_telegram_agent.services.inference import InferenceService, AVAILABLE_MODELS
from libertai_telegram_agent.services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def should_respond_in_group(update: Update, bot_username: str) -> bool:
    """Check if bot should respond in a group chat."""
    text = update.message.text or update.message.caption or ""

    # Check if bot is mentioned
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
    db: Database, context: ContextTypes.DEFAULT_TYPE, telegram_id: int, chat_id: int, chat_type: str,
) -> str | None:
    """Determine which API key to use. Returns decrypted key or None for bot default."""
    # Check group settings first
    if chat_type in ("group", "supergroup"):
        group_settings = await db.get_group_settings(chat_id)
        if group_settings:
            admin = await db.get_user(group_settings["admin_id"])
            if admin and admin["api_key"]:
                encryption_key = context.bot_data["encryption_key"]
                return decrypt_api_key(admin["api_key"], encryption_key)

    # Check user's own key
    user = await db.get_user(telegram_id)
    if user and user["api_key"]:
        encryption_key = context.bot_data["encryption_key"]
        return decrypt_api_key(user["api_key"], encryption_key)

    return None


async def _split_and_send(update: Update, text: str):
    """Send a message, splitting into chunks if it exceeds Telegram's limit."""
    if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
        await update.message.reply_text(text)
        return

    # Split on paragraph boundaries first, then hard split
    chunks = []
    while text:
        if len(text) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            chunks.append(text)
            break
        split_at = text.rfind("\n\n", 0, TELEGRAM_MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = text.rfind("\n", 0, TELEGRAM_MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = TELEGRAM_MAX_MESSAGE_LENGTH
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()

    for chunk in chunks:
        await update.message.reply_text(chunk)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    await db.ensure_user(telegram_id)

    # Get API key (user's own, group admin's, or None for bot default)
    api_key = await _get_api_key_for_request(db, context, telegram_id, chat_id, chat_type)
    is_connected = api_key is not None

    # Rate limit check for free tier users
    if not is_connected:
        allowed, remaining = await rate_limiter.check_and_increment(telegram_id, "message")
        if not allowed:
            await update.message.reply_text(
                "You've reached your daily message limit.\n"
                "Come back tomorrow, or get unlimited access at https://console.libertai.io"
            )
            return

    # Get or create conversation
    conv = await db.get_or_create_conversation(chat_id, chat_type)
    max_messages = context.bot_data["max_conversation_messages"]

    # Build message content (text or vision)
    user = await db.get_user(telegram_id)
    model = user["default_model"] if user else "gemma-3-27b"

    user_content = update.message.text or update.message.caption or ""
    # Remove bot mention from group messages
    if bot_username:
        user_content = user_content.replace(f"@{bot_username}", "").strip()

    # Handle photo messages (vision)
    message_content = user_content
    openai_messages_content = user_content

    if update.message.photo:
        photo = update.message.photo[-1]  # Highest resolution
        file = await context.bot.get_file(photo.file_id)
        photo_bytes = await file.download_as_bytearray()
        b64_image = base64.b64encode(photo_bytes).decode()

        # Use vision-capable model
        if not AVAILABLE_MODELS.get(model, {}).get("vision", False):
            model = "gemma-3-27b"  # Fall back to vision model

        openai_messages_content = [
            {"type": "text", "text": user_content or "What's in this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]

    # Store user message (text only for DB)
    await db.add_message(conv["id"], telegram_id, "user", message_content)

    # Build conversation history
    history = await db.get_messages(conv["id"], limit=max_messages)
    messages = []
    for msg in history[:-1]:  # All except the one we just added
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Add current message with potential vision content
    messages.append({"role": "user", "content": openai_messages_content})

    # Send typing indicator and generate response
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        response = await inference.chat(messages=messages, model=model, api_key=api_key)
    except Exception as e:
        logger.error(f"Inference error for user {telegram_id}: {e}")
        await update.message.reply_text(
            "Sorry, I couldn't process your request right now. Please try again in a moment."
        )
        return

    # Store and send response
    await db.add_message(conv["id"], 0, "assistant", response)
    await _split_and_send(update, response)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_chat.py -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/handlers/chat.py tests/test_chat.py
git commit -m "feat: add chat handler with vision, groups, and rate limiting"
```

---

### Task 10: Bot Handlers — Image Generation

**Files:**
- Create: `src/libertai_telegram_agent/handlers/image.py`
- Create: `tests/test_image.py`

**Step 1: Write failing tests**

```python
# tests/test_image.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from cryptography.fernet import Fernet
from libertai_telegram_agent.handlers.image import image_command
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.rate_limiter import RateLimiter
from libertai_telegram_agent.services.inference import InferenceService


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "test.db"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def rate_limiter(db):
    return RateLimiter(db, daily_messages=50, daily_images=5)


@pytest.fixture
def inference():
    return MagicMock(spec=InferenceService)


@pytest.fixture
def encryption_key():
    return Fernet.generate_key().decode()


def make_context(db, rate_limiter, inference, encryption_key):
    context = MagicMock()
    context.bot_data = {
        "db": db,
        "rate_limiter": rate_limiter,
        "inference": inference,
        "encryption_key": encryption_key,
    }
    context.args = []
    context.bot = MagicMock()
    return context


def make_update(user_id=12345, chat_id=12345, chat_type="private"):
    update = MagicMock()
    update.effective_user.id = user_id
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.message.text = "/image a cat"
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


async def test_image_no_prompt_shows_usage(db, rate_limiter, inference, encryption_key):
    update = make_update()
    context = make_context(db, rate_limiter, inference, encryption_key)
    context.args = []
    await image_command(update, context)
    call_text = update.message.reply_text.call_args[0][0]
    assert "Usage" in call_text or "/image" in call_text


async def test_image_sends_photo(db, rate_limiter, inference, encryption_key):
    update = make_update()
    context = make_context(db, rate_limiter, inference, encryption_key)
    context.args = ["a", "cute", "cat"]
    inference.generate_image = AsyncMock(return_value=b"fake-png-bytes")

    await image_command(update, context)

    update.message.reply_photo.assert_called_once()


async def test_image_rate_limited(db, rate_limiter, inference, encryption_key):
    limiter = RateLimiter(db, daily_messages=50, daily_images=1)
    update = make_update()
    context = make_context(db, limiter, inference, encryption_key)
    context.args = ["a", "cat"]
    inference.generate_image = AsyncMock(return_value=b"fake-png-bytes")

    await image_command(update, context)  # Uses the 1 allowed image

    update2 = make_update()
    context2 = make_context(db, limiter, inference, encryption_key)
    context2.args = ["another", "cat"]
    await image_command(update2, context2)

    last_call = update2.message.reply_text.call_args[0][0]
    assert "limit" in last_call.lower() or "console.libertai.io" in last_call
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_image.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/handlers/image.py
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


def _get_db(context: ContextTypes.DEFAULT_TYPE) -> Database:
    return context.bot_data["db"]


def _get_inference(context: ContextTypes.DEFAULT_TYPE) -> InferenceService:
    return context.bot_data["inference"]


def _get_rate_limiter(context: ContextTypes.DEFAULT_TYPE) -> RateLimiter:
    return context.bot_data["rate_limiter"]


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /image <prompt>\n\nExample: /image a sunset over mountains")
        return

    prompt = " ".join(context.args)
    db = _get_db(context)
    inference = _get_inference(context)
    rate_limiter = _get_rate_limiter(context)
    telegram_id = update.effective_user.id
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type

    await db.ensure_user(telegram_id)
    user = await db.get_user(telegram_id)

    # Determine API key
    api_key = None
    is_connected = False
    if chat_type in ("group", "supergroup"):
        group_settings = await db.get_group_settings(chat_id)
        if group_settings:
            admin = await db.get_user(group_settings["admin_id"])
            if admin and admin["api_key"]:
                encryption_key = context.bot_data["encryption_key"]
                api_key = decrypt_api_key(admin["api_key"], encryption_key)
                is_connected = True

    if not is_connected and user and user["api_key"]:
        encryption_key = context.bot_data["encryption_key"]
        api_key = decrypt_api_key(user["api_key"], encryption_key)
        is_connected = True

    # Rate limit check
    if not is_connected:
        allowed, remaining = await rate_limiter.check_and_increment(telegram_id, "image")
        if not allowed:
            await update.message.reply_text(
                "You've reached your daily image generation limit.\n"
                "Come back tomorrow, or get unlimited access at https://console.libertai.io"
            )
            return

    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)

    try:
        image_bytes = await inference.generate_image(prompt, api_key=api_key)
        await update.message.reply_photo(photo=io.BytesIO(image_bytes), caption=prompt)
    except Exception as e:
        logger.error(f"Image generation error for user {telegram_id}: {e}")
        await update.message.reply_text("Couldn't generate that image. Try a different prompt.")
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_image.py -v`
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/handlers/image.py tests/test_image.py
git commit -m "feat: add image generation handler"
```

---

### Task 11: Main Entry Point (wire everything together)

**Files:**
- Create: `src/libertai_telegram_agent/main.py`
- Create: `tests/test_main.py`

**Step 1: Write a smoke test**

```python
# tests/test_main.py
from unittest.mock import patch, MagicMock
from libertai_telegram_agent.main import create_application


def test_create_application_returns_app():
    with patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "LIBERTAI_API_KEY": "test-key",
        "BOT_ENCRYPTION_KEY": "dGVzdC1lbmNyeXB0aW9uLWtleS1mb3ItdGVzdGluZzE=",
    }):
        app = create_application()
        assert app is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_main.py -v`
Expected: FAIL — module not found.

**Step 3: Write implementation**

```python
# src/libertai_telegram_agent/main.py
import logging

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

from libertai_telegram_agent.config import Settings
from libertai_telegram_agent.database.db import Database
from libertai_telegram_agent.services.inference import InferenceService
from libertai_telegram_agent.services.rate_limiter import RateLimiter
from libertai_telegram_agent.handlers.commands import (
    start_command,
    new_command,
    help_command,
    model_command,
    model_callback,
    usage_command,
)
from libertai_telegram_agent.handlers.account import login_command, logout_command, account_command
from libertai_telegram_agent.handlers.chat import handle_message
from libertai_telegram_agent.handlers.image import image_command

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_application() -> Application:
    settings = Settings()

    app = Application.builder().token(settings.telegram_bot_token).build()

    # Store shared resources in bot_data
    app.bot_data["settings"] = settings
    app.bot_data["encryption_key"] = settings.bot_encryption_key
    app.bot_data["max_conversation_messages"] = settings.max_conversation_messages
    app.bot_data["inference"] = InferenceService(
        api_base_url=settings.libertai_api_base_url,
        default_api_key=settings.libertai_api_key,
    )
    app.bot_data["rate_limiter"] = RateLimiter(
        db=None,  # Will be set in post_init
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


async def post_init(application: Application):
    """Initialize database after application starts."""
    db = Database()
    await db.initialize()
    application.bot_data["db"] = db
    application.bot_data["rate_limiter"].db = db
    logger.info("Database initialized")


async def post_shutdown(application: Application):
    """Clean up database on shutdown."""
    db = application.bot_data.get("db")
    if db:
        await db.close()
    logger.info("Database closed")


def main():
    settings = Settings()
    app = Application.builder().token(settings.telegram_bot_token).post_init(post_init).post_shutdown(post_shutdown).build()

    # Store shared resources
    app.bot_data["settings"] = settings
    app.bot_data["encryption_key"] = settings.bot_encryption_key
    app.bot_data["max_conversation_messages"] = settings.max_conversation_messages
    app.bot_data["inference"] = InferenceService(
        api_base_url=settings.libertai_api_base_url,
        default_api_key=settings.libertai_api_key,
    )
    app.bot_data["rate_limiter"] = RateLimiter(
        db=None,
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

    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model:"))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_message))

    logger.info("Starting bot...")
    app.run_polling()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/test_main.py -v`
Expected: 1 test PASS.

**Step 5: Commit**

```bash
git add src/libertai_telegram_agent/main.py tests/test_main.py
git commit -m "feat: add main entry point wiring all handlers together"
```

---

### Task 12: Dockerfile

**Files:**
- Create: `Dockerfile`

**Step 1: Write Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/

CMD ["python", "-m", "libertai_telegram_agent.main"]
```

**Step 2: Verify it builds**

Run: `cd /home/jon/repos/libertai-telegram-agent && docker build -t libertai-telegram-agent .`
Expected: Build succeeds.

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for deployment"
```

---

### Task 13: Run Full Test Suite + Lint

**Step 1: Run all tests**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run pytest tests/ -v`
Expected: All tests PASS.

**Step 2: Run linter**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run ruff check src/ tests/`
Expected: No errors (or fix any that appear).

**Step 3: Run formatter**

Run: `cd /home/jon/repos/libertai-telegram-agent && uv run ruff format src/ tests/`

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix lint and formatting issues"
```
