"""Golden-prompt coverage for the reverie perception-context seam
(feat/reverie-perception-context).

Renders the real reverie_narrate.j2 through build_reverie_context so the actual
prompt text the LLM sees is exercised, not just the context dict shape --
mirrors test_reverie_narrate_prompt_outcomes.py's approach. Also covers the
actual settings-gated fetch inside run_reverie_once end-to-end, so a future
refactor that drops the flag check / asyncio.to_thread call / kwarg wiring
would fail a test, not just silently no-op the feature.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1

GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)

_PROMPT = (
    Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
)


def _render(**ctx) -> str:
    template = Environment().from_string(_PROMPT.read_text())
    return template.render(**ctx)


def _broadcast(loops=(("ol-1", {}),)):
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id=oid, description="d", **s) for oid, s in loops]),
        attended_node_ids=["n-1"],
        selected_open_loop_id=loops[0][0] if loops else None,
    )


def test_no_percept_block_when_no_recent_percepts():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    assert "recent_percepts" not in ctx
    out = _render(**ctx)
    assert "WHAT YOU CAN CURRENTLY SEE" not in out
    assert "what you can currently see" not in out


def test_percept_block_present_in_plain_coalition_mode():
    from app import reverie

    percepts = [{"narrative": "a person walks into frame near the door", "age_sec": 12.0}]
    ctx = reverie.build_reverie_context(_broadcast(), recent_percepts=percepts)
    assert ctx["recent_percepts"] == percepts
    out = _render(**ctx)
    assert "what you can currently see" in out
    assert "a person walks into frame near the door" in out
    # discipline: percepts must not be pitched as evidence_refs-eligible
    assert "never substitutes for a" in out


def test_percept_block_present_in_concern_cards_mode():
    from app import reverie

    card = ConcernCardV1(
        card_id="c1",
        coalition_ref="ol-1",
        human_text="You said the deploy might slip.",
        thread_label="deploy",
    )
    percepts = [{"narrative": "the kitchen light is on", "age_sec": 40.0}]
    ctx = reverie.build_reverie_context(
        _broadcast(), concern_cards=[card], recent_percepts=percepts
    )
    assert ctx["recent_percepts"] == percepts
    out = _render(**ctx)
    assert "WHAT YOU CAN CURRENTLY SEE" in out
    assert "the kitchen light is on" in out
    assert "do not treat it as evidence_refs" in out


def test_empty_percept_list_renders_no_block():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast(), recent_percepts=[])
    assert "recent_percepts" not in ctx
    out = _render(**ctx)
    assert "WHAT YOU CAN CURRENTLY SEE" not in out


# --- end-to-end: the settings-gated fetch inside run_reverie_once -------------


@pytest.mark.asyncio
async def test_tick_fetches_percepts_when_enabled_and_threads_into_prompt(monkeypatch):
    """The tick must call the vision reader and pass its result through to the
    plan request context the LLM actually sees -- the real regression target
    for a future refactor that drops the flag check, the asyncio.to_thread
    call, or the kwarg wiring into build_reverie_plan_request."""
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", True)
    monkeypatch.setattr(
        vision_reader,
        "read_recent_vision_events",
        lambda **kwargs: [{"narrative": "a person walks into frame", "age_sec": 5.0}],
    )
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(
        bus,
        broadcast_reader=lambda: _broadcast(),
        cortex_client=cortex,
    )
    assert result is not None
    plan_request = cortex.execute_plan.call_args.kwargs["req"]
    assert plan_request.context["recent_percepts"] == [
        {"narrative": "a person walks into frame", "age_sec": 5.0}
    ]


@pytest.mark.asyncio
async def test_tick_skips_percept_fetch_when_disabled(monkeypatch):
    """Default-off must mean the reader is never even called, not just that its
    result gets dropped -- a disabled feature that still hits Postgres on every
    tick is not actually off."""
    from app import reverie, vision_reader
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", False)

    def boom(**kwargs):
        raise AssertionError("vision reader must not be called when disabled")

    monkeypatch.setattr(vision_reader, "read_recent_vision_events", boom)
    bus = AsyncMock()
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(
        bus,
        broadcast_reader=lambda: _broadcast(),
        cortex_client=cortex,
    )
    assert result is not None
    plan_request = cortex.execute_plan.call_args.kwargs["req"]
    assert "recent_percepts" not in plan_request.context
