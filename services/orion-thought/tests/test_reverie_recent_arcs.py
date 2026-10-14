"""Temporal Self rev 4: reverie gets the last few closed arcs.

Covers the context dict, the real reverie_narrate.j2 render, the bounded shared frame fetch,
and the end-to-end tick (block present when the fetch works, absent when it fails or is off).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from jinja2 import Environment

from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.context_provenance import classify
from orion.schemas.reverie import ConcernCardV1

_REPO = Path(__file__).resolve().parents[3]
_PROMPT = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
_FRAME = json.loads((_REPO / "orion/temporal_self/tests/patch4_rows/frame_2026-10-11_talking.json").read_text())
_NOW = datetime(2026, 10, 11, 23, 58, 34, tzinfo=timezone.utc)
GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)
BLOCK = "Recently closed today (most recent first):\n- 17:00-17:20 (20 min) attention on x [arc:abc]"


def _render(**ctx) -> str:
    return Environment().from_string(_PROMPT.read_text()).render(**ctx)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"],
        selected_open_loop_id="ol-1",
    )


def _mock(monkeypatch, handler):
    import app.temporal_self_cue as cue_mod

    monkeypatch.setattr(cue_mod, "_transport", lambda: httpx.MockTransport(handler))


def test_context_and_prompt_carry_block_only_when_given():
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    assert "recent_arcs" not in ctx
    assert "EARLIER TODAY" not in _render(**ctx)
    assert "recent_arcs" not in reverie.build_reverie_context(_broadcast(), recent_arcs="")

    ctx = reverie.build_reverie_context(_broadcast(), recent_arcs=BLOCK)
    assert ctx["recent_arcs"] == BLOCK
    out = _render(**ctx)
    assert "EARLIER TODAY" in out and "[arc:abc]" in out
    assert "NOT evidence_refs" in out


def test_concern_card_mode_also_carries_block():
    from app import reverie

    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="deploy may slip", thread_label="deploy")
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], recent_arcs=BLOCK)
    assert ctx["recent_arcs"] == BLOCK
    assert "EARLIER TODAY" in _render(**ctx)


def test_recent_arcs_ctx_key_is_classified():
    assert classify("recent_arcs") == "derived_summary"


@pytest.mark.asyncio
async def test_fetch_ok_renders_block(monkeypatch):
    from app import reverie

    _mock(monkeypatch, lambda r: httpx.Response(200, json=_FRAME))
    text = await reverie.build_recent_arcs_text("c-1", now=_NOW)
    assert text.startswith("Recently closed today")
    assert text.count("[arc:") == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(503), httpx.Response(200, text="not json"),
                                      httpx.Response(200, json={"garbage": True})])
async def test_fetch_failure_renders_nothing(monkeypatch, response):
    from app import reverie

    _mock(monkeypatch, lambda r: response)
    assert await reverie.build_recent_arcs_text("c-1", now=_NOW) == ""


@pytest.mark.asyncio
async def test_slow_fetch_is_bounded(monkeypatch):
    import app.temporal_self_cue as cue_mod
    from app import reverie
    from app.settings import settings

    async def slow(*a, **k):
        await asyncio.sleep(5)
        return _FRAME

    monkeypatch.setattr(cue_mod, "fetch_frame", slow)
    monkeypatch.setattr(settings, "reverie_recent_arcs_fetch_timeout_sec", 0.05)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    assert await reverie.build_recent_arcs_text("c-1", now=_NOW) == ""
    assert loop.time() - t0 < 1.0


@pytest.mark.asyncio
async def test_disabled_never_fetches(monkeypatch):
    from app import reverie
    from app.settings import settings

    def boom(request):
        raise AssertionError("must not fetch when disabled")

    _mock(monkeypatch, boom)
    monkeypatch.setattr(settings, "reverie_recent_arcs_enabled", False)
    assert await reverie.build_recent_arcs_text("c-1", now=_NOW) == ""


async def _tick(monkeypatch, handler):
    from app import reverie
    from app.settings import settings

    monkeypatch.setattr(settings, "reverie_perception_enabled", False)
    monkeypatch.setattr(settings, "reverie_recent_arcs_enabled", True)
    # The tick uses wall-clock now: make the fixture frame current.
    frame = dict(_FRAME)
    now = datetime.now(timezone.utc)
    from orion.temporal_self.day import day_id_for

    frame.update(as_of=now.isoformat(), day_id=day_id_for(now))
    def fresh(request):
        resp = handler(request)
        return httpx.Response(200, json=frame) if resp.status_code == 200 else resp

    _mock(monkeypatch, fresh)
    cortex = AsyncMock()
    cortex.execute_plan = AsyncMock(return_value={
        "final_text": json.dumps({"interpretation": GROUNDED_TEXT, "evidence_refs": ["ol-1"]}),
    })
    result = await reverie.run_reverie_once(AsyncMock(), broadcast_reader=_broadcast, cortex_client=cortex)
    assert result is not None
    return cortex.execute_plan.call_args.kwargs["req"].context


@pytest.mark.asyncio
async def test_tick_threads_block_into_plan_request(monkeypatch):
    ctx = await _tick(monkeypatch, lambda r: httpx.Response(200, json=_FRAME))
    assert ctx["recent_arcs"].startswith("Recently closed today")


@pytest.mark.asyncio
async def test_tick_omits_block_on_fetch_failure(monkeypatch):
    def down(request):
        raise httpx.ConnectError("down")

    ctx = await _tick(monkeypatch, down)
    assert "recent_arcs" not in ctx
