"""Tool implementations for the chat handler."""

from __future__ import annotations

import json
import logging

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# ── Tool definitions (OpenAI function calling format) ─────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image from a text description. "
                "Use this when the user asks you to draw, create, or generate an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image to generate, always in English",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. "
                "Use this for recent events, facts, or anything you're unsure about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch and read the text content of a webpage. "
                "Use this when the user shares a URL or you need to read a specific page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crypto_price",
            "description": (
                "Get the current price and market data for a cryptocurrency. "
                "Use this when users ask about crypto prices, market cap, or trading volume."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin_id": {
                        "type": "string",
                        "description": (
                            "The CoinGecko coin ID (e.g. 'bitcoin', 'ethereum', 'solana', 'aleph-zero'). "
                            "Use lowercase with hyphens."
                        ),
                    },
                },
                "required": ["coin_id"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────

MAX_CONTENT_LENGTH = 4000


async def web_search(query: str) -> str:
    """Search the web and return top results."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Web search error")
        return f"Search failed: {e}"


async def fetch_url(url: str) -> str:
    """Fetch a URL and return its text content."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "LibertAI-Bot/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            if len(text) > MAX_CONTENT_LENGTH:
                text = text[:MAX_CONTENT_LENGTH] + "\n\n[Content truncated]"
            return text
    except Exception as e:
        logger.exception("URL fetch error")
        return f"Failed to fetch URL: {e}"


async def crypto_price(coin_id: str) -> str:
    """Get crypto price and market data from CoinGecko."""
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            f"?localization=false&tickers=false&community_data=false&developer_data=false"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return f"Coin '{coin_id}' not found on CoinGecko. Check the coin ID."
            resp.raise_for_status()
            data = resp.json()

        market = data.get("market_data", {})
        price_usd = market.get("current_price", {}).get("usd")
        price_eur = market.get("current_price", {}).get("eur")
        change_24h = market.get("price_change_percentage_24h")
        change_7d = market.get("price_change_percentage_7d")
        market_cap = market.get("market_cap", {}).get("usd")
        volume_24h = market.get("total_volume", {}).get("usd")
        name = data.get("name", coin_id)
        symbol = data.get("symbol", "").upper()

        lines = [f"**{name} ({symbol})**"]
        if price_usd is not None:
            lines.append(f"Price: ${price_usd:,.2f} / {price_eur:,.2f}€")
        if change_24h is not None:
            lines.append(f"24h change: {change_24h:+.2f}%")
        if change_7d is not None:
            lines.append(f"7d change: {change_7d:+.2f}%")
        if market_cap is not None:
            lines.append(f"Market cap: ${market_cap:,.0f}")
        if volume_24h is not None:
            lines.append(f"24h volume: ${volume_24h:,.0f}")

        return "\n".join(lines)
    except Exception as e:
        logger.exception("CoinGecko API error")
        return f"Failed to fetch crypto data: {e}"


# ── Tool dispatcher ───────────────────────────────────────────────────

TOOL_HANDLERS = {
    "web_search": web_search,
    "fetch_url": fetch_url,
    "crypto_price": crypto_price,
}


async def execute_tool(name: str, arguments: str) -> str:
    """Parse arguments and execute a tool by name. Returns result string."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"Invalid arguments: {arguments}"
    return await handler(**args)
