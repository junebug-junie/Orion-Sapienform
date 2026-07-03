"""Agent lane builds an agent_repl request with no keyword classification."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


def _hub_imports():
    for key in list(sys.modules):
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    sys.path = [p for p in sys.path if "orion-context-exec" not in p.replace("\\", "/")]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))
    from scripts.context_exec_agent_bridge import build_context_exec_request
    from scripts.settings import settings
    from orion.schemas.cortex.contracts import CortexChatRequest

    return build_context_exec_request, settings, CortexChatRequest


def _req(CortexChatRequest, text: str):
    return CortexChatRequest(
        prompt=text,
        mode="agent",
        session_id="s1",
        user_id="u1",
        trace_id="t1",
    )


def test_agent_repl_request_is_built_when_enabled(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    body = build_context_exec_request(req=_req(CortexChatRequest, "x"), prompt="what breaks if I change the runtime?", llm_profile="agent")
    assert body.mode == "agent_repl"
    # Ceiling permissions granted, write/network stay off.
    assert body.permissions.read_repo is True
    assert body.permissions.read_recall is True
    assert body.permissions.write_repo is False
    assert body.permissions.network_enabled is False
    # Loop-sized budget.
    assert body.budget.max_seconds >= 600


def test_agent_repl_does_not_use_keyword_mode(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    # A prompt full of legacy keyword triggers must still route to agent_repl.
    body = build_context_exec_request(
        req=_req(CortexChatRequest, "x"),
        prompt="trace corr fail open repo impact patch proposal memory correction",
        llm_profile="agent",
    )
    assert body.mode == "agent_repl"


def test_agent_lane_never_builds_investigation_v2(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    # Even if the legacy investigation_v2 flag is on, the agent lane must not use it.
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "CONTEXT_EXEC_INVESTIGATION_V2_ENABLED", True, raising=False)
    body = build_context_exec_request(req=_req(CortexChatRequest, "x"), prompt="anything", llm_profile="agent")
    assert body.mode == "agent_repl"


def test_curiosity_hint_off_by_default(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    body = build_context_exec_request(
        req=_req(CortexChatRequest, "x"), prompt="what am I missing?", llm_profile="agent"
    )
    assert body.text == "what am I missing?"


def test_curiosity_hint_prepends_when_enabled(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    import scripts.curiosity_hint as curiosity_hint

    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_AGENT_CURIOSITY_HINT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        curiosity_hint,
        "_fetch_fresh_candidates",
        lambda: [{"signal_strength": 0.8, "evidence_summary": "open loop in transport"}],
    )
    body = build_context_exec_request(
        req=_req(CortexChatRequest, "x"), prompt="what am I missing?", llm_profile="agent"
    )
    assert body.mode == "agent_repl"
    assert body.text.startswith("[curiosity focus] Self-observed gaps: open loop in transport")
    assert body.text.endswith("\n\nwhat am I missing?")


def test_curiosity_hint_failure_leaves_prompt_untouched(monkeypatch):
    build_context_exec_request, settings, CortexChatRequest = _hub_imports()
    import scripts.curiosity_hint as curiosity_hint

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(settings, "HUB_AGENT_REPL_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "HUB_AGENT_CURIOSITY_HINT_ENABLED", True, raising=False)
    monkeypatch.setattr(curiosity_hint, "_fetch_fresh_candidates", _boom)
    body = build_context_exec_request(
        req=_req(CortexChatRequest, "x"), prompt="what am I missing?", llm_profile="agent"
    )
    assert body.text == "what am I missing?"
