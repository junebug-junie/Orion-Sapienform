from __future__ import annotations

import pytest

import app.llm_profile_resolver as resolver
from app.llm_profile_resolver import (
    LLMProfileValidationError,
    normalize_llm_profile,
    resolve_llm_profile,
    resolve_llm_profile_default,
)


def _live_settings():
    from app.settings import settings

    return settings


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live_settings()
    monkeypatch.setattr(cfg, "context_exec_default_llm_profile", "chat")


def test_normalize_rejects_invalid_profile() -> None:
    with pytest.raises(LLMProfileValidationError):
        normalize_llm_profile("http://evil")


def test_resolve_default_when_omitted() -> None:
    sel = resolve_llm_profile_default(None)
    assert sel.requested is None
    assert sel.selected == "chat"
    assert sel.route_used == "chat"


@pytest.mark.asyncio
async def test_resolve_profile_quick() -> None:
    sel = await resolve_llm_profile("quick")
    assert sel.requested == "quick"
    assert sel.selected == "quick"
    assert sel.route_used == "quick"
    assert sel.fallback_used is False


def test_gateway_routes_reader_is_gone() -> None:
    """GPU pool stage 6.3: context-exec no longer reads the gateway's retiring GET /routes view.

    Pins the deletion so a revert (or a copy-paste from an old branch) cannot quietly put a
    /routes reader back and hold stage 6.5's zero-read window open."""
    assert not hasattr(resolver, "fetch_route_status_map")
    assert not hasattr(resolver, "LLMProfileUnavailableError")
    import inspect

    source = inspect.getsource(resolver)
    assert "httpx" not in source and "urlopen" not in source


@pytest.mark.asyncio
async def test_resolve_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway URL being configured must not turn resolution into an HTTP read."""
    cfg = _live_settings()
    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "http://gateway.test")
    import httpx

    def _boom(*_a, **_k):
        raise AssertionError("resolve_llm_profile must not open an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    sel = await resolve_llm_profile("agent")
    assert (sel.selected, sel.route_used, sel.fallback_used) == ("agent", "agent", False)
