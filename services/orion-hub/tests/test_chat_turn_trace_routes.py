"""Fused correlation-first chat-turn trace lookup (join logic only).

Each of the three underlying sources (CognitionTraceCache, Grammar Atlas,
ExecutionTrajectoryProjectionV1) already has its own coverage elsewhere
(test_cognition_trace_api.py, test_grammar_atlas_api.py). These tests cover
only the new logic this module adds: the join, the gap reporting, and the
route_signal_inferred labeling.
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

import scripts.chat_turn_trace_routes as chat_turn_trace_routes


def _patch_sources(monkeypatch, *, cognition=None, grammar=None, execution=None):
    async def _fake_cognition(_corr):
        return cognition

    async def _fake_grammar(_trace_id):
        return grammar

    def _fake_execution(_trace_id):
        return execution

    monkeypatch.setattr(chat_turn_trace_routes, "_load_cognition_trace", _fake_cognition)
    monkeypatch.setattr(chat_turn_trace_routes, "_load_grammar_trace", _fake_grammar)
    monkeypatch.setattr(chat_turn_trace_routes, "_load_execution_run", _fake_execution)


def test_fused_trace_reports_all_gaps_when_nothing_found(monkeypatch) -> None:
    _patch_sources(monkeypatch)
    out = asyncio.run(chat_turn_trace_routes.get_fused_chat_turn_trace("corr-empty"))
    assert out["correlation_id"] == "corr-empty"
    assert out["sources"] == {}
    assert out["complete"] is False
    assert set(out["gaps"]) == {
        "no_cognition_trace",
        "no_grammar_trace",
        "no_execution_run_pressure_signal",
    }
    assert out["route_signal_inferred"] == "none"


def test_fused_trace_classic_route_when_only_cognition_trace_present(monkeypatch) -> None:
    _patch_sources(monkeypatch, cognition={"verb": "chat_general"})
    out = asyncio.run(chat_turn_trace_routes.get_fused_chat_turn_trace("corr-classic"))
    assert out["sources"]["cognition_trace"] == {"verb": "chat_general"}
    assert out["complete"] is True
    assert out["route_signal_inferred"] == "classic_planrunner"
    assert "no_cognition_trace" not in out["gaps"]
    assert "no_grammar_trace" in out["gaps"]


def test_fused_trace_unified_turn_route_when_only_harness_sources_present(monkeypatch) -> None:
    _patch_sources(monkeypatch, grammar={"trace_id": "t1"}, execution={"step_count": 3})
    out = asyncio.run(chat_turn_trace_routes.get_fused_chat_turn_trace("corr-unified"))
    assert out["sources"]["grammar_trace"] == {"trace_id": "t1"}
    assert out["sources"]["execution_run"] == {"step_count": 3}
    assert out["route_signal_inferred"] == "unified_turn_harness"
    assert "no_cognition_trace" in out["gaps"]


def test_fused_trace_ambiguous_when_both_paths_present(monkeypatch) -> None:
    _patch_sources(monkeypatch, cognition={"verb": "chat_general"}, grammar={"trace_id": "t1"})
    out = asyncio.run(chat_turn_trace_routes.get_fused_chat_turn_trace("corr-ambiguous"))
    assert out["route_signal_inferred"] == "ambiguous_both_paths_present"


def test_fused_trace_derives_harness_trace_id_from_correlation_id(monkeypatch) -> None:
    from orion.substrate.execution_loop.ids import cortex_exec_trace_id

    _patch_sources(monkeypatch)
    monkeypatch.setattr(chat_turn_trace_routes.settings, "NODE_NAME", "athena", raising=False)
    out = asyncio.run(chat_turn_trace_routes.get_fused_chat_turn_trace("corr-xyz"))
    assert out["harness_trace_id"] == cortex_exec_trace_id("athena", "corr-xyz", lane="harness_motor")


def test_api_chat_turn_trace_route_returns_fused_body(monkeypatch) -> None:
    _patch_sources(monkeypatch, cognition={"verb": "chat_general"})
    out = asyncio.run(chat_turn_trace_routes.api_chat_turn_trace("corr-route"))
    assert out["correlation_id"] == "corr-route"
    assert out["route_signal_inferred"] == "classic_planrunner"
