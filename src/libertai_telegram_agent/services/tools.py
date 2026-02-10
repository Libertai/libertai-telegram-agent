"""Tool implementations for the chat handler."""

from __future__ import annotations

import json
import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"

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
                "Search Google for current information. "
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
            "name": "news_search",
            "description": (
                "Search Google News for recent news articles. "
                "Use this when the user asks about current events, breaking news, or recent developments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "youtube_search",
            "description": (
                "Search YouTube for videos. "
                "Use this when the user asks for video content, tutorials, or wants to find a video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The YouTube search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scholar_search",
            "description": (
                "Search Google Scholar for academic papers and research. "
                "Use this when the user asks about scientific research, academic papers, or citations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The academic search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "maps_search",
            "description": (
                "Search Google Maps for places and businesses. "
                "Use this when the user asks about locations, restaurants, shops, or local businesses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The place or business search query (e.g. 'best pizza in Paris')",
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


# ── SearchAPI.com helpers ────────────────────────────────────────────

_searchapi_key: str = ""


def configure(searchapi_api_key: str) -> None:
    """Set the SearchAPI.com API key. Called once at startup."""
    global _searchapi_key  # noqa: PLW0603
    _searchapi_key = searchapi_api_key


async def _searchapi_request(engine: str, params: dict) -> dict:
    """Make a request to SearchAPI.com and return the JSON response."""
    params = {"engine": engine, "api_key": _searchapi_key, **params}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(SEARCHAPI_BASE, params=params)
        resp.raise_for_status()
        return resp.json()


# ── Tool implementations ──────────────────────────────────────────────

MAX_CONTENT_LENGTH = 4000


async def web_search(query: str) -> str:
    """Search Google via SearchAPI.com and return top results."""
    try:
        data = await _searchapi_request("google", {"q": query})
        results = data.get("organic_results", [])[:5]
        if not results:
            return "No results found."
        lines = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            lines.append(f"**{title}**\n{snippet}\nURL: {link}")
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Web search error")
        return f"Search failed: {e}"


async def news_search(query: str) -> str:
    """Search Google News via SearchAPI.com and return recent articles."""
    try:
        data = await _searchapi_request("google_news", {"q": query})
        results = data.get("organic_results", [])[:5]
        if not results:
            return "No news found."
        lines = []
        for r in results:
            title = r.get("title", "")
            source = r.get("source", {}).get("name", "") if isinstance(r.get("source"), dict) else r.get("source", "")
            date = r.get("date", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            header = f"**{title}**"
            if source or date:
                header += f"\n_{source}_ — {date}" if source else f"\n{date}"
            if snippet:
                header += f"\n{snippet}"
            if link:
                header += f"\nURL: {link}"
            lines.append(header)
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("News search error")
        return f"News search failed: {e}"


async def youtube_search(query: str) -> str:
    """Search YouTube via SearchAPI.com and return top videos."""
    try:
        data = await _searchapi_request("youtube", {"q": query})
        results = data.get("videos", data.get("video_results", []))[:5]
        if not results:
            return "No videos found."
        lines = []
        for r in results:
            title = r.get("title", "")
            channel = r.get("channel", {}).get("name", "") if isinstance(r.get("channel"), dict) else ""
            views = r.get("views", "")
            length = r.get("length", "")
            link = r.get("link", "")
            line = f"**{title}**"
            meta = []
            if channel:
                meta.append(channel)
            if views:
                meta.append(f"{views} views")
            if length:
                meta.append(length)
            if meta:
                line += f"\n{' · '.join(meta)}"
            if link:
                line += f"\nURL: {link}"
            lines.append(line)
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("YouTube search error")
        return f"YouTube search failed: {e}"


async def scholar_search(query: str) -> str:
    """Search Google Scholar via SearchAPI.com and return academic results."""
    try:
        data = await _searchapi_request("google_scholar", {"q": query})
        results = data.get("organic_results", [])[:5]
        if not results:
            return "No academic results found."
        lines = []
        for r in results:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            pub_info = r.get("publication_info", {}).get("summary", "") if isinstance(
                r.get("publication_info"), dict
            ) else ""
            cited_by = r.get("inline_links", {}).get("cited_by", {}).get("total", "") if isinstance(
                r.get("inline_links"), dict
            ) else ""
            line = f"**{title}**"
            if pub_info:
                line += f"\n_{pub_info}_"
            if snippet:
                line += f"\n{snippet}"
            if cited_by:
                line += f"\nCited by: {cited_by}"
            if link:
                line += f"\nURL: {link}"
            lines.append(line)
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Scholar search error")
        return f"Scholar search failed: {e}"


async def maps_search(query: str) -> str:
    """Search Google Maps via SearchAPI.com and return places."""
    try:
        data = await _searchapi_request("google_maps", {"q": query})
        results = data.get("local_results", [])[:5]
        if not results:
            return "No places found."
        lines = []
        for r in results:
            title = r.get("title", "")
            rating = r.get("rating", "")
            reviews = r.get("reviews", "")
            address = r.get("address", "")
            phone = r.get("phone", "")
            line = f"**{title}**"
            if rating:
                stars = f" ({reviews} reviews)" if reviews else ""
                line += f"\nRating: {rating}/5{stars}"
            if address:
                line += f"\nAddress: {address}"
            if phone:
                line += f"\nPhone: {phone}"
            lines.append(line)
        return "\n\n".join(lines)
    except Exception as e:
        logger.exception("Maps search error")
        return f"Maps search failed: {e}"


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
    "news_search": news_search,
    "youtube_search": youtube_search,
    "scholar_search": scholar_search,
    "maps_search": maps_search,
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
