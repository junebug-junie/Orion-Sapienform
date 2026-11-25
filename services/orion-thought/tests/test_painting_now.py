"""Integration patch 7b: a waking painting reads Orion's now before its interpret call and records its cites.

The interpret call (visual_context_interpret) is the painting's one model step, so it is the only place the now
is shown. Cites are the [experience:<id>] tags the call wrote; they are posted to /temporal-self/now/use and never
reach the diffusion prompt.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orion.schemas.temporal_self import TemporalNowV1

REPO = Path(__file__).resolve().parents[3]
AT = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _now(*event_ids: str) -> TemporalNowV1:
    return TemporalNowV1.model_validate({
        "shown_id": "shown-1", "reader_organ": "painting", "since": AT.isoformat(), "as_of": AT.isoformat(),
        "items": [{"event_id": e, "organ": "camera", "source_kind": "office_session", "label": f"label {e}",
                   "occurred_at": AT.isoformat()} for e in event_ids],
    })


class _Cortex:
    def __init__(self, text: Any):
        self.text = text
        self.calls: list[Any] = []

    async def execute_plan(self, *, source, req, correlation_id, timeout_sec):
        self.calls.append(req)
        if isinstance(self.text, BaseException):
            raise self.text
        return {"final_text": self.text}


def _wire(monkeypatch, *, read, closes: list, enabled: bool = True, interpret: bool = True, slot: str | None = "a reverie"):
    from app import temporal_self_cue as cue
    from app import visual_chain as vc

    monkeypatch.setattr(vc.settings, "visual_chain_interpretation_enabled", interpret)
    monkeypatch.setattr(vc.settings, "painting_now_enabled", enabled)
    monkeypatch.setattr(vc, "load_latest_visual_chain_continuity_state", lambda **kw: (None, 0, 0))
    monkeypatch.setattr(vc, "load_latest_reverie_question_context", lambda **kw: slot)
    monkeypatch.setattr(vc, "load_latest_self_study_reflection", lambda **kw: None)
    monkeypatch.setattr(vc, "load_episode_memory_seed", lambda **kw: None)
    monkeypatch.setattr(vc, "load_latest_memory_crystallization", lambda **kw: None)
    fetched: list = []

    async def fake_fetch(timeout_sec, *, organ, run_ref=None, correlation_id=""):
        fetched.append((organ, run_ref))
        return read

    async def fake_post(shown_id, cites, citing_ref, timeout_sec, *, correlation_id=""):
        closes.append((shown_id, list(cites), citing_ref))
        return {"uses": [{"event_id": c} for c in cites]}

    monkeypatch.setattr(cue, "fetch_now_bounded", fake_fetch)
    monkeypatch.setattr(cue, "post_now_use", fake_post)
    return vc, cue, fetched


@pytest.mark.asyncio
async def test_the_interpret_call_sees_the_now_and_its_cites_are_closed_and_stripped(monkeypatch):
    from app.temporal_self_cue import NowRead

    closes: list = []
    vc, _cue, fetched = _wire(monkeypatch, read=NowRead(reason="ok", now=_now("camera:e1", "dream:e2")), closes=closes)
    cortex = _Cortex("A lantern over black water by an office window [experience:camera:e1] [experience:bogus]")

    plan = await vc.compute_visual_plan(AsyncMock(), chain_id="chain-1", cortex_client=cortex)

    assert fetched == [("painting", "visual_run:chain-1")]
    shown_to_model = cortex.calls[0].context["orion_now"]
    assert "[experience:camera:e1]" in shown_to_model and "[experience:dream:e2]" in shown_to_model
    assert plan.context_slot_interpreted == "A lantern over black water by an office window"
    assert "experience" not in plan.prompt
    assert plan.now_shown_id == "shown-1" and plan.now_cites == ["camera:e1"]
    assert closes == [("shown-1", ["camera:e1"], "visual_run:chain-1")]
    assert plan.chain_fields()["now_cites"] == ["camera:e1"]
    # The frozen plan (durable prepare) round-trips the new fields; an old frozen plan without them still loads.
    assert vc.VisualPlan.from_json(plan.to_json()).now_cites == ["camera:e1"]
    old = {k: v for k, v in plan.to_json().items() if k not in ("now_shown_id", "now_cites")}
    assert vc.VisualPlan.from_json(old).now_shown_id is None


@pytest.mark.asyncio
async def test_no_tag_still_closes_the_read_with_no_cites(monkeypatch):
    from app.temporal_self_cue import NowRead

    closes: list = []
    vc, _cue, _f = _wire(monkeypatch, read=NowRead(reason="ok", now=_now("camera:e1")), closes=closes)
    plan = await vc.compute_visual_plan(AsyncMock(), chain_id="c2", cortex_client=_Cortex("A lantern on a pier"))
    assert closes == [("shown-1", [], "visual_run:c2")]
    assert plan.now_cites == [] and plan.context_slot_interpreted == "A lantern on a pier"


@pytest.mark.asyncio
async def test_a_failed_interpret_call_leaves_the_read_open(monkeypatch):
    import asyncio

    from app.temporal_self_cue import NowRead

    closes: list = []
    vc, _cue, fetched = _wire(monkeypatch, read=NowRead(reason="ok", now=_now("camera:e1")), closes=closes)
    plan = await vc.compute_visual_plan(AsyncMock(), chain_id="c3", cortex_client=_Cortex(asyncio.TimeoutError()))
    assert fetched and closes == [] and plan.now_shown_id is None and plan.now_cites is None


@pytest.mark.asyncio
async def test_unavailable_now_runs_the_painting_as_before(monkeypatch):
    from app.temporal_self_cue import NowRead

    closes: list = []
    vc, _cue, _f = _wire(monkeypatch, read=NowRead(reason="not_found"), closes=closes)
    cortex = _Cortex("A lantern on a pier [experience:camera:e1]")
    plan = await vc.compute_visual_plan(AsyncMock(), chain_id="c4", cortex_client=cortex)
    assert cortex.calls[0].context["orion_now"] is None
    assert closes == [] and plan.now_shown_id is None
    assert plan.context_slot_interpreted == "A lantern on a pier"  # a stray tag never reaches diffusion


@pytest.mark.asyncio
async def test_switch_off_or_no_interpret_call_reads_nothing(monkeypatch):
    from app.temporal_self_cue import NowRead

    for kw in ({"enabled": False}, {"interpret": False}, {"slot": None}):
        closes: list = []
        vc, _cue, fetched = _wire(monkeypatch, read=NowRead(reason="ok", now=_now("camera:e1")), closes=closes, **kw)
        await vc.compute_visual_plan(AsyncMock(), chain_id="c5", cortex_client=_Cortex("A pier"))
        assert fetched == [] and closes == [], kw


def test_painting_cites_keeps_shown_ids_only_and_caps_at_three():
    from app.temporal_self_cue import painting_cites

    text = "A pier [experience:a] at dusk [experience:b], [experience:c] [experience:d] [experience:zzz]."
    clean, cites = painting_cites(text, ["a", "b", "c", "d"])
    assert cites == ["a", "b", "c"]
    assert "experience" not in clean and clean == "A pier at dusk,."


def test_the_template_shows_the_now_only_when_there_is_one():
    from jinja2 import Template

    tpl = Template((REPO / "orion/cognition/prompts/visual_context_interpret.j2").read_text())
    base = {"source_label": "Orion notices", "source_text": "x", "prior_description": None}
    assert "WHAT YOUR OTHER ORGANS LIVED" not in tpl.render(**base, orion_now=None)
    shown = tpl.render(**base, orion_now="- [experience:camera:e1] office visit (camera, 5 min ago)")
    assert "WHAT YOUR OTHER ORGANS LIVED" in shown and "[experience:camera:e1]" in shown
