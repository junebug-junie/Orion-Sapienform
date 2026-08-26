"""2026-08-25: the unified turn passes the browser's surface_context through.

Confirmed live the same day, against the real running Hub: Juniper's
microphone ALREADY reached the Orion-mode unified turn end to end
(`voice.ws.audio_received` -> `voice.stt.done transcript_len=50` -> pre-turn
appraisal -> `run_unified_turn`), because websocket_handler's STT block runs
ahead of the Orion-mode branch. What did not work was subtler and is what
this file locks down: `_build_situation_prompt_fragment` built its
`situation_ctx` from only session_id / raw_user_text / presence_context, so
`_build_surface_context` fell through to its `"typed"` default on every
single unified turn. The words arrived; the fact that they had been SPOKEN
did not.

The `metadata` nesting is the part worth a regression test rather than a
comment: `_build_surface_context` reads `ctx["metadata"]["surface_context"]`,
not `ctx["surface_context"]`, so putting it at the top level looks correct,
type-checks fine, and silently does nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import orion.hub.turn_orchestrator as turn_orchestrator


def _run(payload):
    """Call the real _build_situation_prompt_fragment with build_situation_for_ctx
    stubbed, so this asserts on the ctx actually handed to the situation
    builder rather than on a reimplementation of it."""
    captured = {}

    async def _fake_build(ctx, runtime_ns):
        captured["ctx"] = ctx
        return None, {"compact_text": "Situation: stub"}

    orig = turn_orchestrator.build_situation_for_ctx
    turn_orchestrator.build_situation_for_ctx = _fake_build
    try:
        asyncio.run(
            turn_orchestrator._build_situation_prompt_fragment(
                session_id="s1",
                user_message="hello",
                payload=payload,
                settings=SimpleNamespace(),
                correlation_id="corr-1",
            )
        )
    finally:
        turn_orchestrator.build_situation_for_ctx = orig
    return captured["ctx"]


def test_spoken_surface_context_reaches_the_situation_builder():
    ctx = _run({"surface_context": {"surface": "hub_desktop", "input_modality": "spoken"}})
    # Nested under "metadata" -- top level would be silently ignored.
    assert ctx["metadata"]["surface_context"]["input_modality"] == "spoken"


def test_nesting_matches_what_build_surface_context_actually_reads():
    """Guards the exact lookup in orion/situational/context.py rather than
    trusting that this test's own expectation and the reader agree."""
    from orion.situational.context import _build_surface_context

    ctx = _run({"surface_context": {"surface": "hub_desktop", "input_modality": "spoken"}})
    surface = _build_surface_context(ctx)
    assert surface.input_modality == "spoken"
    assert surface.surface == "hub_desktop"


def test_typed_surface_context_is_carried_verbatim():
    ctx = _run({"surface_context": {"surface": "hub_desktop", "input_modality": "typed"}})
    assert ctx["metadata"]["surface_context"]["input_modality"] == "typed"


@pytest.mark.parametrize("payload", [{}, {"surface_context": None}, {"surface_context": {}}, {"surface_context": "nonsense"}])
def test_missing_or_malformed_surface_context_adds_no_metadata(payload):
    """Absent/garbage must degrade to "no metadata key at all", which lets
    _build_surface_context apply its own documented defaults -- not to a
    half-built metadata dict that shadows them."""
    ctx = _run(payload)
    assert "metadata" not in ctx


def test_payload_surface_context_is_copied_not_aliased():
    """The orchestrator mutates its own `payload` dict elsewhere in the turn;
    handing the situation builder a live reference to a caller-owned nested
    dict is how a later unrelated mutation silently changes what was already
    reported."""
    surface = {"surface": "hub_desktop", "input_modality": "spoken"}
    ctx = _run({"surface_context": surface})
    surface["input_modality"] = "typed"
    assert ctx["metadata"]["surface_context"]["input_modality"] == "spoken"
