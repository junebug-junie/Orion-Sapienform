from types import SimpleNamespace

from orion.memory.crystallization.graphiti_config import resolve_graphiti_adapter_url


def test_resolve_prefers_adapter_url_over_graphiti_url():
    settings = SimpleNamespace(
        GRAPHITI_ADAPTER_URL="http://adapter:8000",
        GRAPHITI_URL="http://legacy:8000",
    )
    assert resolve_graphiti_adapter_url(settings) == "http://adapter:8000"


def test_resolve_falls_back_to_graphiti_url():
    settings = SimpleNamespace(GRAPHITI_ADAPTER_URL="", GRAPHITI_URL="http://legacy:8000")
    assert resolve_graphiti_adapter_url(settings) == "http://legacy:8000"


def test_resolve_empty_when_both_unset():
    settings = SimpleNamespace(GRAPHITI_ADAPTER_URL="", GRAPHITI_URL="")
    assert resolve_graphiti_adapter_url(settings) == ""
