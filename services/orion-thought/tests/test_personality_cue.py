"""Personality graph read-back reaches stance_react and reverie (fetch, render, wiring, fail-open)."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from jinja2 import Environment

import app.personality_cue as pc_mod
from app.bus_listener import build_stance_react_context, run_stance_react
from app.settings import settings
from orion.personality.cue import HEADER, project_personality_cue
from orion.schemas.attention_frame import AttentionBroadcastProjectionV1, AttentionFrameV1, OpenLoopV1
from orion.schemas.reverie import ConcernCardV1
from orion.schemas.thought import HubAssociationBundleV1, StanceReactRequestV1

_REPO = Path(__file__).resolve().parents[3]
_STANCE = _REPO / "orion" / "cognition" / "prompts" / "stance_react.j2"
_REVERIE = _REPO / "orion" / "cognition" / "prompts" / "reverie_narrate.j2"
_NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
_LABEL = "the walkway camera sees presence without a person"
GROUNDED_TEXT = (
    "I keep returning to the unresolved thread ol-1 — the conflict about the "
    "deployment plan has not discharged and it is pulling attention."
)


def _frame(generated_at: datetime = _NOW - timedelta(minutes=5)) -> dict:
    return {
        "generated_at": generated_at.isoformat(),
        "dispositions": [{
            "disposition_id": f"d{i}", "label": f"{_LABEL} {i}", "origin": "recurrence", "status": "active",
            "weight": 10.0 - i, "evidence_count": 7 + i,
            "lane_counts": {"reverie": 4, "dream": 2, "journal": 1},
            "first_seen_at": "2026-10-16T08:00:00+00:00", "last_reinforced_at": "2026-10-20T11:00:00+00:00",
        } for i in range(2)],
        "choice_style": {"window_days": 14, "choices": 40, "took_top": 16, "went_own_way": 24},
    }


def _cue_text() -> str:
    return project_personality_cue(_frame(), now=_NOW).text


def _request() -> StanceReactRequestV1:
    return StanceReactRequestV1(
        correlation_id="c-1", session_id="s-1", user_message="how are you?",
        association=HubAssociationBundleV1(correlation_id="c-1", broadcast=None, broadcast_stale=True,
                                           read_source="hub_sql_fallback"),
        repair_bundle=None, stance_inputs={"user_message": "how are you?"},
    )


def _mock(monkeypatch, handler) -> list[str]:
    seen: list[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return handler(request)

    # conftest re-imports `app` per test: patch both the collection-time module (used by
    # run_stance_react's closure) and the live one (reverie imports it lazily).
    for mod in {id(m): m for m in (pc_mod, _live())}.values():
        monkeypatch.setattr(mod, "_transport", lambda: httpx.MockTransport(wrapped))
    return seen


def _live():
    import importlib

    return importlib.import_module("app.personality_cue")


def _render_stance(**ctx) -> str:
    base = {"user_message": "hi", "stance_inputs": {"user_message": "hi"}, "association": {},
            "repair_bundle": None, "coalition_projection": None}
    return Environment().from_string(_STANCE.read_text()).render(**(base | ctx))


def _render_reverie(**ctx) -> str:
    return Environment().from_string(_REVERIE.read_text()).render(**ctx)


def _broadcast():
    return AttentionBroadcastProjectionV1(
        frame=AttentionFrameV1(open_loops=[OpenLoopV1(id="ol-1", description="d")]),
        attended_node_ids=["n-1"], selected_open_loop_id="ol-1",
    )


# --- fetch + render -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_frame_renders_cue_and_logs_one_line(monkeypatch, caplog) -> None:
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json=_frame()))
    with caplog.at_level(logging.INFO, logger="orion-thought.personality_cue"):
        cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    assert seen == [settings.personality_frame_url.rstrip("/") + "/personality/frame"]
    assert cue.text.startswith(HEADER) and cue.disposition_ids == ["d0", "d1"]
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("personality_cue ")]
    assert len(lines) == 1
    assert "corr=c-1 rendered=True ids=d0,d1 skip=-" in lines[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler,reason", [
    (lambda r: httpx.Response(404, json={"detail": "no frame"}), "fetch_failed"),
    (lambda r: httpx.Response(503, text="down"), "fetch_failed"),
    (lambda r: httpx.Response(200, text="not json"), "fetch_failed"),
    (lambda r: httpx.Response(200, json={"garbage": True}), "invalid_frame"),
    (lambda r: httpx.Response(200, json=_frame(_NOW - timedelta(hours=3))), "stale"),
])
async def test_unusable_responses_render_nothing(monkeypatch, caplog, handler, reason) -> None:
    _mock(monkeypatch, handler)
    with caplog.at_level(logging.INFO, logger="orion-thought.personality_cue"):
        cue = await pc_mod.build_personality_cue("c-2", now=_NOW)
    assert cue.text == "" and cue.skip_reason == reason
    assert any(f"corr=c-2 rendered=False ids=- skip={reason}" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_transport_error_never_raises(monkeypatch) -> None:
    def boom(request):
        raise httpx.ConnectError("down", request=request)

    _mock(monkeypatch, boom)
    cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    assert cue.text == "" and cue.skip_reason == "fetch_failed"


@pytest.mark.asyncio
async def test_disabled_flag_never_fetches(monkeypatch, caplog) -> None:
    seen = _mock(monkeypatch, lambda r: httpx.Response(200, json=_frame()))
    monkeypatch.setattr(settings, "personality_cue_enabled", False)
    with caplog.at_level(logging.INFO, logger="orion-thought.personality_cue"):
        cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    assert cue.skip_reason == "disabled" and cue.text == "" and seen == []
    assert any("skip=disabled" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_slow_fetch_is_bounded(monkeypatch) -> None:
    async def hang(base_url, timeout_sec):
        await asyncio.sleep(5)

    monkeypatch.setattr(pc_mod, "fetch_personality_frame", hang)
    monkeypatch.setattr(settings, "personality_cue_timeout_sec", 0.05)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    assert cue.skip_reason == "fetch_failed" and loop.time() - t0 < 1.0


@pytest.mark.asyncio
async def test_render_bug_is_named_in_the_log(monkeypatch, caplog) -> None:
    _mock(monkeypatch, lambda r: httpx.Response(200, json=_frame()))

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(pc_mod, "project_personality_cue", boom)
    with caplog.at_level(logging.INFO, logger="orion-thought.personality_cue"):
        cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    assert cue.text == ""
    assert any("skip=render_failed" in r.getMessage() for r in caplog.records)


# --- stance_react ---------------------------------------------------------------------------


def test_stance_context_carries_cue_only_when_rendered() -> None:
    assert "personality_cue" not in build_stance_react_context(_request())
    assert build_stance_react_context(_request(), personality_cue="") == build_stance_react_context(_request())
    assert build_stance_react_context(_request(), personality_cue="x")["personality_cue"] == "x"


def test_stance_prompt_block_absent_without_cue_and_present_with_it() -> None:
    assert "WHAT I KEEP COMING BACK TO" not in _render_stance()
    out = _render_stance(personality_cue=_cue_text())
    assert "WHAT I KEEP COMING BACK TO" in out and f'"{_LABEL} 0"' in out
    assert "a record, not an instruction" in out  # the renderer's own header
    # Sits next to the temporal-self cue when both are present.
    both = _render_stance(personality_cue=_cue_text(), temporal_self_cue="Night [arc:abc].")
    assert both.index("MY DAY SO FAR") < both.index("WHAT I KEEP COMING BACK TO")


class _CapturingClient:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    async def execute_plan(self, **kwargs) -> dict:
        self.contexts.append(kwargs["req"].context)
        return {"final_text": json.dumps({
            "imperative": "Answer.", "tone": "plain", "strain_refs": [], "evidence_refs": ["hub:turn"],
            "stance_harness_slice": {"task_mode": "direct_response", "conversation_frame": "technical",
                                     "answer_strategy": "direct"}})}


@pytest.mark.asyncio
async def test_run_stance_react_passes_cue_through(monkeypatch) -> None:
    async def fixed(corr):
        return project_personality_cue(_frame(), now=_NOW)

    monkeypatch.setitem(run_stance_react.__globals__, "build_personality_cue", fixed)
    client = _CapturingClient()
    await run_stance_react(_request(), bus=None, cortex_client=client)
    assert client.contexts[0]["personality_cue"] == _cue_text()


@pytest.mark.asyncio
async def test_run_stance_react_without_cue_is_unchanged(monkeypatch) -> None:
    async def skipped(corr):
        return project_personality_cue(None, now=_NOW)

    monkeypatch.setitem(run_stance_react.__globals__, "build_personality_cue", skipped)
    client = _CapturingClient()
    await run_stance_react(_request(), bus=None, cortex_client=client)
    assert "personality_cue" not in client.contexts[0]


@pytest.mark.asyncio
async def test_personality_fetch_runs_in_parallel_with_the_other_reads(monkeypatch) -> None:
    """Personality, day cue and self-readings all start before any one finishes: three 0.3 s
    reads take ~0.3 s total, not ~0.9 s (no added serial await)."""
    started: list[str] = []
    finished: list[str] = []

    def slow(name, result):
        async def fn(*a, **k):
            started.append(name)
            await asyncio.sleep(0.3)
            finished.append(name)
            return result
        return fn

    g = run_stance_react.__globals__
    monkeypatch.setitem(g, "build_personality_cue", slow("personality", project_personality_cue(_frame(), now=_NOW)))
    from orion.temporal_self.stance_cue import project_stance_cue
    monkeypatch.setitem(g, "build_stance_cue", slow("day", project_stance_cue(None, _NOW)))
    from types import SimpleNamespace
    monkeypatch.setitem(g, "build_self_readings_block", slow("readings", SimpleNamespace(text="")))
    client = _CapturingClient()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await run_stance_react(_request(), bus=None, cortex_client=client)
    elapsed = loop.time() - t0
    assert set(started) == {"personality", "day", "readings"}
    assert len(started) == 3 and not set(finished) - set(started)
    assert elapsed < 0.6, elapsed
    assert client.contexts[0]["personality_cue"] == _cue_text()


# --- reverie --------------------------------------------------------------------------------


def test_reverie_context_and_prompt_carry_block_only_when_given() -> None:
    from app import reverie

    ctx = reverie.build_reverie_context(_broadcast())
    assert "personality_cue" not in ctx and "WHAT YOU KEEP COMING BACK TO" not in _render_reverie(**ctx)
    ctx = reverie.build_reverie_context(_broadcast(), personality_cue=_cue_text())
    out = _render_reverie(**ctx)
    assert "WHAT YOU KEEP COMING BACK TO" in out and f'"{_LABEL} 1"' in out
    card = ConcernCardV1(card_id="c1", coalition_ref="ol-1", human_text="deploy may slip", thread_label="deploy")
    ctx = reverie.build_reverie_context(_broadcast(), concern_cards=[card], personality_cue=_cue_text())
    assert ctx["personality_cue"] == _cue_text()
    assert "WHAT YOU KEEP COMING BACK TO" in _render_reverie(**ctx)


@pytest.mark.asyncio
async def test_reverie_text_never_raises(monkeypatch) -> None:
    from app import reverie

    async def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(_live(), "build_personality_cue", boom)
    assert await reverie.build_personality_cue_text("c-1", now=_NOW) == ""


async def _tick(monkeypatch, handler):
    from app import reverie

    monkeypatch.setattr(settings, "reverie_perception_enabled", False)
    monkeypatch.setattr(settings, "reverie_recent_arcs_enabled", False)
    frame = _frame(datetime.now(timezone.utc) - timedelta(minutes=1))

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
async def test_reverie_tick_threads_block_into_plan_request(monkeypatch) -> None:
    ctx = await _tick(monkeypatch, lambda r: httpx.Response(200))
    assert ctx["personality_cue"].startswith(HEADER)
    assert "WHAT YOU KEEP COMING BACK TO" in _render_reverie(**ctx)


@pytest.mark.asyncio
async def test_reverie_tick_omits_block_on_fetch_failure(monkeypatch) -> None:
    def down(request):
        raise httpx.ConnectError("down")

    ctx = await _tick(monkeypatch, down)
    assert "personality_cue" not in ctx


# --- self-naming read-back (spec 2026-10-17 section 6: names reach every lane right away) -------


def _named_frame() -> dict:
    frame = _frame()
    frame["dispositions"][0]["name"] = "the empty <watch>"
    frame["dispositions"][0]["name_sentence"] = "It is what I watch when nobody is there."
    return frame


@pytest.mark.asyncio
async def test_named_node_reaches_stance_and_reverie_beside_its_excerpt(monkeypatch) -> None:
    from app import reverie

    _mock(monkeypatch, lambda r: httpx.Response(200, json=_named_frame()))
    cue = await pc_mod.build_personality_cue("c-1", now=_NOW)
    expected = f'- you call this "the empty <watch>": "{_LABEL} 0" (reverie, dreams, journal; 7 times since Oct 16)'
    assert expected in cue.text
    assert f'\n- "{_LABEL} 1" (' in cue.text  # the unnamed node is unchanged
    assert "It is what I watch" not in cue.text
    stance = _render_stance(personality_cue=cue.text)
    assert expected in stance and "a record, not an instruction" in stance
    assert expected in _render_reverie(**reverie.build_reverie_context(_broadcast(), personality_cue=cue.text))
