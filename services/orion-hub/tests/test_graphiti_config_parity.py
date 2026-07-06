from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from scripts import crystallization_routes as routes


@pytest.fixture
def adapter_url_only_settings(monkeypatch):
    fake = SimpleNamespace(
        GRAPHITI_ENABLED=False,
        GRAPHITI_ADAPTER_URL="http://orion-athena-graphiti-adapter:8000",
        GRAPHITI_URL="",
        FALKORDB_URI="",
        CRYSTALLIZER_VECTOR_COLLECTION="orion_memory_crystallizations",
        CRYSTALLIZER_EMBED_HOST_URL="",
        CRYSTALLIZER_EMBED_MODE="http",
        CRYSTALLIZER_EMBED_TIMEOUT_MS=8000,
        SERVICE_NAME="orion-hub",
        SERVICE_VERSION="0.1.0",
        NODE_NAME="hub",
    )
    monkeypatch.setattr(routes, "_settings", lambda: fake)
    return fake


def test_projection_config_url_matches_graphiti_adapter(adapter_url_only_settings):
    cfg = routes._projection_config()
    request = MagicMock(spec=Request)
    adapter = routes._graphiti(request)

    assert cfg.graphiti_url == "http://orion-athena-graphiti-adapter:8000"
    assert adapter.url == "http://orion-athena-graphiti-adapter:8000"
    assert cfg.graphiti_enabled is True
    assert adapter.enabled is True
