import asyncio
import base64

import httpx
from openai import AsyncOpenAI

AVAILABLE_MODELS = {
    "qwen3-coder-next": {"name": "Qwen 3 Coder Next", "tier": "free", "vision": False},
    "gemma-3-27b": {"name": "Gemma 3 27B", "tier": "free", "vision": True},
    "hermes-3-8b-tee": {"name": "Hermes 3 8B (TEE)", "tier": "free", "vision": False},
    "glm-4.7": {"name": "GLM 4.7", "tier": "pro", "vision": False},
    "glm-4.7-flash": {"name": "GLM 4.7 Flash", "tier": "free", "vision": False},
}
IMAGE_MODEL = "z-image-turbo"


class InferenceService:
    """Service for interacting with the LibertAI inference API."""

    def __init__(self, api_base_url: str, default_api_key: str):
        self.api_base_url = api_base_url
        self.default_api_key = default_api_key

    def get_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Create an AsyncOpenAI client with the given or default API key."""
        return AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=api_key or self.default_api_key,
        )

    async def chat(self, messages, model, api_key=None, tools=None):
        """Send a chat completion request, retrying once after 2s on failure.

        Returns the full message object from the API response.
        """
        client = self.get_client(api_key)
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception:
            await asyncio.sleep(2)
            response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def generate_image(self, prompt, api_key=None, size="1024x1024") -> bytes:
        """Generate an image and return the decoded bytes."""
        headers = {
            "Authorization": f"Bearer {api_key or self.default_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": IMAGE_MODEL,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resp = await http_client.post(
                f"{self.api_base_url}/images/generations",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            b64_data = data["data"][0]["b64_json"]
            return base64.b64decode(b64_data)
