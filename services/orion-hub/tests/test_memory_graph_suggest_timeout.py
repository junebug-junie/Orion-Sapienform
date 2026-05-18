from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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


def test_verb_timeout_ms_matches_yaml_default() -> None:
    from scripts.memory_graph_suggest_timeout import memory_graph_verb_timeout_ms

    assert memory_graph_verb_timeout_ms() == 180_000


def test_resolve_timeouts_derives_from_verb_when_route_secs_zero() -> None:
    from scripts.memory_graph_suggest_timeout import resolve_memory_graph_suggest_timeouts

    settings = SimpleNamespace(
        MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS=180_000,
        MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC=0.0,
        MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC=0.0,
    )
    verb_sec, quick, brain, verb_ms = resolve_memory_graph_suggest_timeouts(settings)
    assert verb_ms == 180_000
    assert verb_sec == 180.0
    assert quick == 72.0
    assert brain == 180.0


def test_resolve_timeouts_honors_explicit_route_overrides() -> None:
    from scripts.memory_graph_suggest_timeout import resolve_memory_graph_suggest_timeouts

    settings = SimpleNamespace(
        MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS=180_000,
        MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC=90.0,
        MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC=120.0,
    )
    _, quick, brain, _ = resolve_memory_graph_suggest_timeouts(settings)
    assert quick == 90.0
    assert brain == 120.0


def test_suggest_attempt_includes_timeout_diagnostics_on_hub_wait(monkeypatch) -> None:
    _ensure_hub_scripts_import_path()
    import scripts.settings as ss
    from scripts.memory_graph_suggest import suggest_with_escalation

    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE", "quick", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE", "brain", raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION", False, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING", False, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS", 180_000, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC", 0.05, raising=False)
    monkeypatch.setattr(ss.settings, "MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC", 0.0, raising=False)
    monkeypatch.setattr(ss.settings, "TIMEOUT_SEC", 400, raising=False)
    monkeypatch.setattr(ss.settings, "HUB_AUTO_DEFAULT_ENABLED", False, raising=False)
    monkeypatch.setattr(ss.settings, "NODE_NAME", "athena", raising=False)

    client = AsyncMock()

    async def slow_chat(*_a, **_k):
        await asyncio.sleep(2.0)
        return None

    client.chat.side_effect = slow_chat

    out = asyncio.run(
        suggest_with_escalation(
            cortex_client=client,
            payload={
                "verbs": ["memory_graph_suggest"],
                "messages": [{"role": "user", "content": "x"}],
                "use_recall": False,
                "no_write": True,
            },
            session_id="sid-timeout-diag",
            user_id=None,
            settings=ss.settings,
            mutation_context={},
        )
    )
    assert out["ok"] is False
    budget = out.get("suggest_timeout_budget") or {}
    assert budget.get("verb_timeout_ms") == 180_000
    assert budget.get("quick_timeout_sec") == 0.05
    att = out["attempts"][0]
    assert att["timeout_sec"] == 0.05
    assert att["timeout_layer"] == "hub_asyncio_wait_for"
    assert att["configured_timeout_sec"] == 0.05
    assert att["error_summary"] == "hub_wait_for_timeout"
    assert float(att["elapsed_sec"]) >= 0.04
