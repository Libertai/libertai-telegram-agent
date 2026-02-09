import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libertai_telegram_agent.services.inference import InferenceService

API_BASE = "https://api.test.io/v1"
DEFAULT_KEY = "default-key-123"
CUSTOM_KEY = "custom-key-456"


@pytest.fixture
def service():
    return InferenceService(api_base_url=API_BASE, default_api_key=DEFAULT_KEY)


class TestGetClient:
    """get_client creates an AsyncOpenAI client with correct parameters."""

    def test_uses_default_key(self, service):
        client = service.get_client()

        assert client.api_key == DEFAULT_KEY
        assert str(client.base_url).rstrip("/") == API_BASE

    def test_uses_custom_key_when_provided(self, service):
        client = service.get_client(api_key=CUSTOM_KEY)

        assert client.api_key == CUSTOM_KEY
        assert str(client.base_url).rstrip("/") == API_BASE


class TestChat:
    """chat calls the OpenAI completions API and returns content."""

    async def test_returns_response_text(self, service):
        mock_message = MagicMock()
        mock_message.content = "Hello from the model"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(service, "get_client", return_value=mock_client):
            result = await service.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="gemma-3-27b",
            )

        assert result == "Hello from the model"
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gemma-3-27b",
            messages=[{"role": "user", "content": "Hi"}],
        )

    async def test_passes_api_key_through(self, service):
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(service, "get_client", return_value=mock_client) as mock_get:
            await service.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="gemma-3-27b",
                api_key=CUSTOM_KEY,
            )

        mock_get.assert_called_once_with(CUSTOM_KEY)


class TestGenerateImage:
    """generate_image POSTs to the images endpoint and returns decoded bytes."""

    async def test_returns_decoded_bytes(self, service):
        expected_bytes = b"fake-image-data"
        b64_encoded = base64.b64encode(expected_bytes).decode()
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"b64_json": b64_encoded}]}
        mock_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch("libertai_telegram_agent.services.inference.httpx.AsyncClient", return_value=mock_http_client):
            result = await service.generate_image(prompt="a cat in space")

        assert result == expected_bytes
        mock_http_client.post.assert_awaited_once()
        call_args = mock_http_client.post.call_args
        assert call_args[0][0] == f"{API_BASE}/images/generations"
        assert call_args[1]["json"]["prompt"] == "a cat in space"
        assert call_args[1]["json"]["model"] == "z-image-turbo"
        assert call_args[1]["json"]["response_format"] == "b64_json"
