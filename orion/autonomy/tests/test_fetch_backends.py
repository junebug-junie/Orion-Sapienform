from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orion.autonomy.fetch_backends import firecrawl_search_backend


@pytest.mark.asyncio
async def test_firecrawl_search_backend_parses_urls() -> None:
    response_body = json.dumps(
        {
            "success": True,
            "data": [
                {"url": "https://example.com/a", "title": "A"},
                {"url": "https://example.com/b", "title": "B"},
            ],
        }
    )

    def _fake_urlopen(req, timeout=30.0):  # noqa: ARG001
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body.encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        assert req.get_header("Authorization") == "Bearer test-key"
        return mock_resp

    with patch("orion.autonomy.fetch_backends.urllib.request.urlopen", side_effect=_fake_urlopen):
        result = await firecrawl_search_backend("gpu news", max_articles=2, api_key="test-key")

    assert result["success"] is True
    assert result["urls"] == ["https://example.com/a", "https://example.com/b"]


@pytest.mark.asyncio
async def test_firecrawl_search_backend_http_error() -> None:
    import urllib.error

    def _raise_http(*_args, **_kwargs):
        err = urllib.error.HTTPError("https://api.firecrawl.dev/v1/search", 401, "unauthorized", {}, None)
        err.read = MagicMock(return_value=b"invalid key")
        raise err

    with patch("orion.autonomy.fetch_backends.urllib.request.urlopen", side_effect=_raise_http):
        result = await firecrawl_search_backend("gpu news", max_articles=2, api_key="bad")

    assert result["success"] is False
    assert result["error"] == "http_401"
