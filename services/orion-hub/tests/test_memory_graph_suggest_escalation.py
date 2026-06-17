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

    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", "quick", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE", "brain", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION", True, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING", True, raising=False)
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


def _client_result(
    text: str,
    *,
    ok: bool = True,
    status: str = "ok",
    finish_reason: str = "stop",
) -> Any:
    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult

    cr = CortexClientResult(
        ok=ok,
        mode="brain",
        verb="memory_graph_suggest",
        status=status,
        final_text=text,
        metadata={
            "model": "test-model",
            "provider_finish_reason": finish_reason,
            "provider_completion_tokens": 42,
        },
    )
    return CortexChatResult(cortex_result=cr, final_text=text)


def test_quick_valid_json_returns_without_brain(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper last week about cats"),
            session_id="sid-1",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "quick"
    assert calls == ["quick"]
    assert len(out["attempts"]) == 1
    assert out["attempts"][0].get("finish_reason") == "stop"
    assert out["attempts"][0].get("completion_tokens") == 42
    assert out["grounding_included"] is True


def test_quick_empty_response_escalates_to_brain(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if len(calls) == 1:
            return _client_result("")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-2",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert len(calls) == 2


def test_quick_invalid_json_escalates_to_brain(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if len(calls) == 1:
            return _client_result("not json {")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-3",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert len(calls) == 2


def test_quick_unknown_predicate_escalates_to_brain(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    bad = json.loads(fixture_draft_text)
    bad["edges"][0]["p"] = "orionmem:totallyMadeUp"
    bad_text = json.dumps(bad)

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if len(calls) == 1:
            return _client_result(bad_text)
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-4",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert any("unknown_predicate" in str(e) for e in out["attempts"][0].get("validation_errors", []))


def test_brain_success_after_quick_failure_returns_brain_draft_and_metadata(
    hub_settings, fixture_draft_text
) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if len(calls) == 1:
            return _client_result("")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-5",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert len(out["attempts"]) == 2
    assert out["draft"]["ontology_version"] == "orionmem-2026-05"
    assert out["attempts"][1].get("phase") == "success"


def test_quick_timeout_does_not_escalate_to_brain(hub_settings, fixture_draft_text) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        raise TimeoutError("simulated quick timeout")

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-timeout",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    assert out.get("route_used") is None
    assert len(calls) == 1
    assert len(out["attempts"]) == 1
    assert "hub_wait_for_timeout" in str(out["attempts"][0].get("validation_errors", []))


def test_truncated_json_escalates_to_brain(hub_settings, fixture_draft_text: str) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    truncated = (
        '{"ontology_version":"orionmem-2026-05","utterance_ids":["u1"],'
        '"entities":[{"id":"urn:uuid:10000001-0000-4000-8000-000000000001"'
    )
    calls: List[str] = []

    async def chat(req, correlation_id=None):
        route = str((req.options or {}).get("llm_route") or "brain")
        calls.append(route)
        if len(calls) == 1:
            return _client_result(truncated, finish_reason="length")
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("extract memory graph"),
            session_id="sid-trunc",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert len(calls) == 2
    phases = [str(a.get("phase") or "") for a in out.get("attempts") or []]
    assert "json_truncated" in phases


def test_cortex_prose_returns_no_json_object_without_draft(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    prose = "There's a kind of sacred tension in those moments when trust and fear meet."

    async def chat(req, correlation_id=None):
        return _client_result(prose)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("extract memory graph"),
            session_id="sid-prose",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    assert out.get("error") == "memory_graph_suggest_failed"
    assert out.get("draft") is None
    validation_errors = [str(e) for e in out.get("validation_errors") or []]
    assert "no_json_object" in validation_errors
    phases = [str(a.get("phase") or "") for a in out.get("attempts") or []]
    assert "no_json_object" in phases


def test_both_fail_returns_error_and_attempts(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    async def chat(req, correlation_id=None):
        raise ValueError("bus down")

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("z"),
            session_id="sid-6",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    assert out.get("error") == "memory_graph_suggest_failed"
    assert len(out["attempts"]) == 2


def test_rdf_violations_do_not_escalate_to_brain(hub_settings, fixture_draft_text, monkeypatch) -> None:
    _ensure_hub_scripts_import_path()
    import scripts.memory_graph_suggest as mgs
    from scripts.memory_graph_suggest import suggest_with_escalation

    def fake_preview(draft):
        return (False, ["shacl:test_violation"], {"card_count": 0, "edge_count": 0, "titles": []})

    monkeypatch.setattr(mgs, "preview_validate_only", fake_preview)

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        return _client_result(fixture_draft_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("Joey angered Juniper"),
            session_id="sid-7",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )

    assert out["ok"] is False
    assert out["violations"] == ["shacl:test_violation"]
    assert calls == ["quick"]


SHOWER_PROMPT = """Structured transcript evidence for memory graph extraction (do not invent turns).

--- turn 1 id=u1 role=user ---
k, off to shower. Be back soon!

--- turn 2 id=a1 role=assistant ---
Shower well. I'll be here when you're back.

Emit utterance_ids matching the ids above; fill utterance_text_by_id with excerpts."""


def test_quick_empty_role_grounded_draft_escalates_to_brain(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    empty_draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }
    shower_path = REPO_ROOT / "tests/fixtures/memory_graph/shower_role_grounded_draft.json"
    if not shower_path.is_file():
        pytest.skip("shower fixture missing")
    shower_text = shower_path.read_text(encoding="utf-8").strip()

    calls: List[Optional[str]] = []

    async def chat(req, correlation_id=None):
        calls.append((req.options or {}).get("llm_route"))
        if len(calls) == 1:
            return _client_result(json.dumps(empty_draft))
        return _client_result(shower_text)

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload(SHOWER_PROMPT),
            session_id="sid-shower",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out["route_used"] == "brain"
    assert len(calls) == 2
    quick_errors = out["attempts"][0].get("validation_errors") or []
    assert any(
        "no_entities_when_role_grounded_subjects_expected" in str(e)
        or "no_situations_when_role_grounded_context_expected" in str(e)
        for e in quick_errors
    )
    assert len(out["draft"].get("entities") or []) >= 1
    assert len(out["draft"].get("situations") or []) >= 1


def test_both_routes_empty_role_grounded_returns_failed(hub_settings) -> None:
    _ensure_hub_scripts_import_path()
    from scripts.memory_graph_suggest import suggest_with_escalation

    empty_draft = {
        "ontology_version": "orionmem-2026-05",
        "utterance_ids": ["u1", "a1"],
        "entities": [],
        "situations": [],
        "edges": [],
        "dispositions": [],
    }

    async def chat(req, correlation_id=None):
        return _client_result(json.dumps(empty_draft))

    client = AsyncMock()
    client.chat.side_effect = chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload(SHOWER_PROMPT),
            session_id="sid-shower-fail",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    assert out.get("error") == "memory_graph_suggest_failed"
    validation_errors = [str(e) for e in out.get("validation_errors") or []]
    assert any(
        "no_entities_when_role_grounded_subjects_expected" in e
        or "no_situations_when_role_grounded_context_expected" in e
        for e in validation_errors
    )


def test_suggest_succeeds_when_final_text_empty_but_raw_choices_have_json(
    hub_settings, fixture_draft_text
) -> None:
    _ensure_hub_scripts_import_path()
    import json

    from orion.schemas.cortex.contracts import CortexChatResult, CortexClientResult
    from orion.schemas.cortex.schemas import StepExecutionResult
    from scripts.memory_graph_suggest import suggest_with_escalation

    raw = {
        "choices": [
            {
                "message": {"role": "assistant", "content": fixture_draft_text},
                "finish_reason": "stop",
            }
        ]
    }
    step = StepExecutionResult(
        status="success",
        verb_name="memory_graph_suggest",
        step_name="llm_memory_graph_suggest",
        order=0,
        result={"LLMGatewayService": {"content": "", "raw": raw}},
        latency_ms=1,
        node="n",
        logs=[],
        error=None,
    )
    cr = CortexClientResult(
        ok=True,
        mode="brain",
        verb="memory_graph_suggest",
        status="success",
        final_text="",
        steps=[step],
    )
    resp = CortexChatResult(cortex_result=cr, final_text="")

    client = AsyncMock()
    client.chat = AsyncMock(return_value=resp)

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload=_msg_payload("role=user\nTest"),
            session_id="sid-raw-choices",
            user_id=None,
            settings=hub_settings,
            mutation_context={},
        )
    )
    assert out["ok"] is True
    assert out.get("draft")
    assert json.loads(fixture_draft_text)["ontology_version"] == out["draft"]["ontology_version"]


def test_approve_route_module_has_no_llm_client_imports() -> None:
    import ast

    path = HUB_ROOT / "scripts" / "memory_graph_routes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"openai", "anthropic", "litellm"}
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    assert not (roots & banned)
    text = path.read_text(encoding="utf-8")
    assert "cortex_client" not in text.split("memory_graph_approve", 1)[-1]
