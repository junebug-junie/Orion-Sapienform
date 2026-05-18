from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


@pytest.fixture()
def hub_settings(monkeypatch):
    _ensure_hub_scripts_import_path()
    import scripts.settings as ss

    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", "brain", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_FALLBACK_ROUTE", "quick", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ENABLE_FALLBACK", True, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 60.0, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 60.0, raising=False)
    monkeypatch.setattr(ss.settings, "HUB_AUTO_DEFAULT_ENABLED", False, raising=False)
    monkeypatch.setattr(ss.settings, "HUB_CONTEXT_TURNS", 10, raising=False)
    monkeypatch.setattr(ss.settings, "NODE_NAME", "athena", raising=False)
    return ss.settings


@pytest.fixture()
def fixture_draft_text() -> str:
    p = REPO_ROOT / "tests/fixtures/memory_graph/joey_cats_draft.json"
    if not p.is_file():
        pytest.skip("fixture missing")
    return p.read_text(encoding="utf-8").strip()


def _msg_payload(content: str) -> Dict[str, Any]:
    return {
        "mode": "brain",
        "verbs": ["memory_graph_suggest"],
        "messages": [{"role": "user", "content": content}],
        "use_recall": False,
        "no_write": True,
    }


def _client_result(text: str, *, ok: bool = True, status: str = "ok") -> Any:
    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

    cr = CortexClientResult(
        ok=ok,
        mode="brain",
        verb="memory_graph_suggest",
        status=status,
        final_text=text,
    )
    return CortexChatResult(cortex_result=cr, final_text=text)


def test_brain_success_quick_not_called(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("draft memory graph"),
            session_id="sid-1",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())
    assert out["ok"] is True
    assert out["suggest_route_used"] == "brain"
    assert calls == [None]
    assert "appendix_c_json" in out and out["draft"]["ontology_version"] == "orionmem-2026-05"


def test_brain_timeout_then_quick_success(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if calls[-1] is None:
            raise TimeoutError("simulated")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("x"),
            session_id="sid-2",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())
    assert out["ok"] is True
    assert out["suggest_route_used"] == "quick"
    assert calls == [None, "quick"]


def test_brain_invalid_json_then_quick_success(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if calls[-1] is None:
            return _client_result("this is not json {")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("y"),
            session_id="sid-3",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())
    assert out["ok"] is True
    assert out["suggest_route_used"] == "quick"


def test_both_routes_fail_returns_ok_false(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    async def chat(req, correlation_id=None):
        raise ValueError("bus down")

    client = AsyncMock()
    client.chat.side_effect = chat

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("z"),
            session_id="sid-4",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())
    assert out["ok"] is False
    assert out.get("error") == "memory_graph_suggest_exhausted"
    attempts = out["suggest_attempts"]
    assert len(attempts) == 2
    assert all(a.get("error_summary") for a in attempts)


def test_rdf_violations_surface_no_fallback_to_quick(hub_settings, fixture_draft_text, monkeypatch) -> None:
    _ensure_hub_scripts_import_path()
    import scripts.memory_graph_suggest as mgs
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    def fake_preview(draft):
        return (False, ["shacl:test_violation"], {"card_count": 0, "edge_count": 0, "titles": []})

    monkeypatch.setattr(mgs, "preview_validate_only", fake_preview)

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("z"),
            session_id="sid-5",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())

    assert out["ok"] is False
    assert out["violations"] == ["shacl:test_violation"]
    assert calls == [None]


def test_cortex_prose_returns_no_json_object_without_draft(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    prose = "There's a kind of sacred tension in those moments when trust and fear meet."

    async def chat(req, correlation_id=None):
        return _client_result(prose)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload=_msg_payload("extract memory graph"),
            session_id="sid-prose",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    assert out.get("error") == "memory_graph_suggest_exhausted"
    assert out.get("draft") is None
    phases = [str(a.get("phase") or "") for a in out.get("suggest_attempts") or []]
    assert "no_json_object" in phases


def test_diagnostic_includes_raw_text(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import run_memory_graph_suggest_with_fallback

    client = AsyncMock()
    client.chat.return_value = _client_result(fixture_draft_text)

    async def _body():
        return await run_memory_graph_suggest_with_fallback(
            cortex_client=client,
            payload={**_msg_payload("d"), "diagnostic": True},
            session_id="sid-6",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )

    out = asyncio.run(_body())
    assert out["ok"] is True
    assert "diagnostic_raw" in out
    assert "ontology_version" in out["diagnostic_raw"]
