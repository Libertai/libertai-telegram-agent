# LibertAI Telegram Bot - Design Document

## Overview

A Telegram bot that provides access to LibertAI's decentralized AI inference platform. Users can chat with AI models, generate images, and analyze documents directly in Telegram. The bot offers a free tier with daily rate limits and supports connecting a LibertAI account for unlimited access.

## Architecture

**Stack:**
- Python 3.12+ with `python-telegram-bot` (async)
- `openai` Python SDK for LibertAI API (OpenAI-compatible)
- SQLite via `aiosqlite` for persistence
- `httpx` for image generation API calls

**Flow:**
```
Telegram User -> Telegram Bot API -> Bot Process -> LibertAI API (api.libertai.io)
                                         |
                                      SQLite DB
                                   (conversations,
                                    user rate limits,
                                    usage tracking)
```

Single Python async process. Holds one LibertAI API key for free tier requests. Connected users' own API keys are used for their requests.

## Configuration (env vars)

| Variable | Description | Default |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | From BotFather | required |
| `LIBERTAI_API_KEY` | Bot's API key from console.libertai.io | required |
| `LIBERTAI_API_BASE_URL` | API base URL | `https://api.libertai.io/v1` |
| `DEFAULT_MODEL` | Default chat model | `gemma-3-27b` |
| `FREE_TIER_DAILY_MESSAGES` | Daily message limit per user | `50` |
| `FREE_TIER_DAILY_IMAGES` | Daily image gen limit per user | `5` |
| `BOT_ENCRYPTION_KEY` | Fernet key for encrypting stored API keys | required |
| `MAX_CONVERSATION_MESSAGES` | Max messages in context window | `20` |

## Bot Commands

| Command | Description |
|---|---|
| `/start` | Welcome message, intro to LibertAI |
| `/new` | Start a new conversation (clears context) |
| `/image <prompt>` | Generate an image using `z-image-turbo` |
| `/model` | Show/switch current model (inline keyboard) |
| `/usage` | Show daily usage (messages/images remaining) |
| `/login <api-key>` | Connect LibertAI account |
| `/logout` | Disconnect account, return to free tier |
| `/account` | Show connection status, credit balance |
| `/help` | List commands and capabilities |

## Chat Behavior

- Any text message (not a command) continues the current conversation
- Bot shows "typing..." indicator while generating, sends full response when complete
- Photos/images sent with a caption are analyzed using vision-capable models (`gemma-3-27b`)
- Conversation context maintained per chat with configurable max history length

### Group Chats

- Bot only responds when @mentioned or replied to
- One conversation context per group chat
- If a group admin linked their API key via `/login`, the group uses their credits
- Otherwise each user in the group gets their own free tier quota

## Auth & Account Linking

### `/login <api-key>` flow:
1. User sends `/login sk-abc123...`
2. Bot immediately deletes the message (to remove key from chat history)
3. Bot validates key against LibertAI API (`GET /credits/balance`)
4. On success: stores key (Fernet-encrypted) in SQLite, confirms connection, shows balance
5. Connected users: no daily rate limits, billed to their credits, access to pro models

### Group admin linking:
- When a group admin runs `/login` in a group, the group is linked to their account
- All group usage goes through the admin's API key/credits

## Rate Limiting (Free Tier)

- Daily message limit per Telegram user ID (default: 50/day)
- Daily image generation limit (default: 5/day)
- Resets at midnight UTC
- Friendly message when limit reached with link to console.libertai.io

## Models

| Model | Tier | Capabilities |
|---|---|---|
| `gemma-3-27b` | Free | Chat, vision |
| `hermes-3-8b-tee` | Free | Chat (lighter/faster) |
| `glm-4.7` | Pro (connected) | Chat, deep thinking |
| `z-image-turbo` | Free (rate limited) | Image generation |

## Database Schema

```sql
users (
    telegram_id   INTEGER PRIMARY KEY,
    api_key       TEXT,              -- Fernet-encrypted, NULL for free tier
    default_model TEXT DEFAULT 'gemma-3-27b',
    created_at    TIMESTAMP,
    updated_at    TIMESTAMP
)

conversations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER,          -- Telegram chat ID (DM or group)
    chat_type     TEXT,             -- 'private', 'group', 'supergroup'
    active        BOOLEAN DEFAULT TRUE,
    created_at    TIMESTAMP
)

messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER REFERENCES conversations,
    telegram_id     INTEGER,        -- who sent it (tracks per-user in groups)
    role            TEXT,            -- 'user', 'assistant', 'system'
    content         TEXT,
    created_at      TIMESTAMP
)

daily_usage (
    telegram_id   INTEGER,
    date          DATE,
    message_count INTEGER DEFAULT 0,
    image_count   INTEGER DEFAULT 0,
    PRIMARY KEY (telegram_id, date)
)

group_settings (
    chat_id       INTEGER PRIMARY KEY,
    admin_id      INTEGER REFERENCES users,
    created_at    TIMESTAMP
)
```

## Error Handling

- **API errors**: Retry once after 2s, then user-friendly message
- **Invalid API key**: Direct user to console.libertai.io
- **Insufficient credits**: Prompt to top up
- **Model unavailable**: Fall back to default model, inform user
- **Message too long** (>4096 chars): Split into multiple messages
- **Image generation failure**: Suggest trying a different prompt
- All errors logged with context (user ID, chat ID, error type)

## Dependencies

**Runtime:**
- `python-telegram-bot[ext]` - Telegram bot framework
- `openai` - LibertAI chat API client
- `httpx` - Image generation API calls
- `aiosqlite` - Async SQLite
- `cryptography` - Fernet encryption for API keys
- `pydantic-settings` - Env var config

**Dev:**
- `ruff` - Linting/formatting
- `pytest` - Testing
- `pytest-asyncio` - Async test support

## Project Structure

```
libertai-telegram-agent/
├── pyproject.toml
├── .env.example
├── Dockerfile
├── src/
│   └── libertai_telegram_agent/
│       ├── __init__.py
│       ├── main.py             # Entry point, bot setup
│       ├── config.py           # Env var loading, defaults
│       ├── handlers/
│       │   ├── __init__.py
│       │   ├── chat.py         # Message handling, AI responses
│       │   ├── commands.py     # /start, /new, /help, /model, /usage
│       │   ├── image.py        # /image command
│       │   └── account.py      # /login, /logout, /account
│       ├── services/
│       │   ├── __init__.py
│       │   ├── inference.py    # LibertAI API client (chat + image)
│       │   └── rate_limiter.py # Per-user daily limits
│       └── database/
│           ├── __init__.py
│           ├── db.py           # SQLite connection, init
│           └── models.py       # Data access
└── tests/
```
