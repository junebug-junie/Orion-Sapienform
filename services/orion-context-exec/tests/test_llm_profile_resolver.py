from __future__ import annotations

import pytest

from app.llm_profile_resolver import (
    LLMProfileUnavailableError,
    LLMProfileValidationError,
    normalize_llm_profile,
    resolve_llm_profile,
    resolve_llm_profile_default,
)


def _live_settings():
    from app.settings import settings

    return settings


@pytest.fixture(autouse=True)
def _clear_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live_settings()
    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "")
    monkeypatch.setattr(cfg, "context_exec_default_llm_profile", "chat")
    monkeypatch.setattr(cfg, "context_exec_llm_profile_fallback_enabled", False)


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


@pytest.mark.asyncio
async def test_route_unavailable_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live_settings()
    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "http://gateway.test")

    async def _down_map(_url: str, *, timeout_sec: float = 1.5) -> dict[str, str]:
        return {"agent": "down", "chat": "up"}

    monkeypatch.setattr("app.llm_profile_resolver.fetch_route_status_map", _down_map)

    with pytest.raises(LLMProfileUnavailableError, match="agent"):
        await resolve_llm_profile("agent")


@pytest.mark.asyncio
async def test_route_unavailable_fallback_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _live_settings()
    monkeypatch.setattr(cfg, "context_exec_llm_gateway_url", "http://gateway.test")
    monkeypatch.setattr(cfg, "context_exec_llm_profile_fallback_enabled", True)

    async def _down_map(_url: str, *, timeout_sec: float = 1.5) -> dict[str, str]:
        return {"quick": "down", "chat": "up"}

    monkeypatch.setattr("app.llm_profile_resolver.fetch_route_status_map", _down_map)

    sel = await resolve_llm_profile("quick")
    assert sel.selected == "chat"
    assert sel.route_used == "chat"
    assert sel.fallback_used is True
    assert sel.fallback_reason and "quick" in sel.fallback_reason
