"""Endogenous outreach shows the personality read-back right after the situation block, read
concurrently with the situation and logged next to it; absent cue leaves the prompt unchanged."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx

import scripts.endogenous_outreach as eo
from orion.personality.cue import HEADER
from scripts.endogenous_outreach import (
    EndogenousOutreach,
    OutreachContext,
    apply_content_novelty,
    build_outreach_prompt,
    situation_lines,
)

CUE = HEADER + '\n- "the walkway camera sees presence without a person 0" (reverie, dreams; 7 times since Oct 16)'


def frame() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "generated_at": (now - timedelta(minutes=5)).isoformat(),
        "dispositions": [{
            "disposition_id": "d0", "label": "the walkway camera sees presence without a person",
            "origin": "recurrence", "status": "active", "weight": 10.0, "evidence_count": 7,
            "lane_counts": {"reverie": 4, "dream": 2}, "first_seen_at": "2026-10-16T08:00:00+00:00",
            "last_reinforced_at": (now - timedelta(hours=1)).isoformat(),
        }],
    }


def _situation() -> dict:
    until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    return {"juniper": {"whereabouts": {"gist": "Juniper is in Chicago", "valid_until": until,
                                        "until_source": "stated"}}}


def _ctx(**kw) -> OutreachContext:
    base = dict(curiosity_summaries=["sustained prediction error on node:x"], recent_turns=[],
                presence=None, situation=_situation(), situation_status="ok")
    base.update(kw)
    return OutreachContext(**base)


def _outreach(**kw) -> EndogenousOutreach:
    base = dict(enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=4,
                quiet_start_hour=-1, quiet_end_hour=-1, timeout_sec=5.0,
                notify_channel="orion:notify:in_app", fallback_session_id="orion_outreach")
    base.update(kw)
    return EndogenousOutreach(**base)


def test_prompt_puts_the_block_right_after_the_situation():
    prompt = build_outreach_prompt(_ctx(personality_cue=CUE))
    assert CUE in prompt
    sit_block = "\n".join(situation_lines(_situation(), datetime.now(timezone.utc), timezone.utc))
    assert f"{sit_block}\n\n{CUE}\n\n" in prompt  # its own block, right after the situation


def test_prompt_without_a_cue_is_unchanged():
    base = build_outreach_prompt(_ctx())
    assert HEADER not in base
    assert build_outreach_prompt(_ctx(personality_cue="")) == base
    assert build_outreach_prompt(_ctx(personality_cue="  \n")) == base


def test_cue_alone_is_not_a_reason_to_speak():
    assert build_outreach_prompt(_ctx(curiosity_summaries=[], personality_cue=CUE)) == ""


def test_novelty_filter_keeps_the_cue():
    ctx = _ctx(curiosity_content_ids=["c1"], personality_cue=CUE)
    assert apply_content_novelty(ctx, used_prior_ids=set(), used_curiosity_ids=set()).personality_cue == CUE


def _quiet_context_reads(monkeypatch, *, situation_delay=0.0):
    for name in ("_fetch_curiosity_content", "_fetch_recent_turns", "_fetch_embodied_presence",
                 "_fetch_current_daydream", "_fetch_open_priors"):
        monkeypatch.setattr(eo, name, lambda *a, **k: None)

    async def read_situation(bus):
        await asyncio.sleep(situation_delay)
        return _situation(), "ok"

    monkeypatch.setattr(eo, "_read_situation", read_situation)


REAL_CLIENT = httpx.AsyncClient


def _patch_http(monkeypatch, handler):
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: REAL_CLIENT(transport=httpx.MockTransport(handler)))


def test_gather_context_reads_the_cue_and_logs_it(monkeypatch, caplog):
    _quiet_context_reads(monkeypatch)
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=frame())

    _patch_http(monkeypatch, handler)
    outreach = _outreach(personality_cue_enabled=True, durable_runs_url="http://durable.test/")
    with caplog.at_level(logging.INFO):
        ctx = asyncio.run(outreach._gather_context("s"))
    assert seen == ["http://durable.test/personality/frame"]
    assert ctx.personality_cue.startswith(HEADER) and ctx.situation_status == "ok"
    assert "endogenous_outreach_personality_cue rendered=True ids=d0 skip=-" in caplog.text


def test_bare_construction_is_off_and_makes_no_call(monkeypatch, caplog):
    _quiet_context_reads(monkeypatch)

    def handler(request):
        raise AssertionError("no call expected")

    _patch_http(monkeypatch, handler)
    with caplog.at_level(logging.INFO):
        ctx = asyncio.run(_outreach()._gather_context("s"))
    assert ctx.personality_cue == ""
    assert "endogenous_outreach_personality_cue rendered=False ids=- skip=disabled" in caplog.text


def test_fetch_failure_is_no_block(monkeypatch):
    _quiet_context_reads(monkeypatch)

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    _patch_http(monkeypatch, handler)
    ctx = asyncio.run(_outreach(personality_cue_enabled=True,
                                durable_runs_url="http://durable.test")._gather_context("s"))
    assert ctx.personality_cue == "" and ctx.situation is not None


def test_situation_and_cue_are_read_concurrently(monkeypatch):
    _quiet_context_reads(monkeypatch, situation_delay=0.3)

    async def handler(request):
        await asyncio.sleep(0.3)
        return httpx.Response(200, json=frame())

    _patch_http(monkeypatch, handler)
    outreach = _outreach(personality_cue_enabled=True, durable_runs_url="http://durable.test")

    async def run():
        ev = asyncio.get_running_loop()
        t = ev.time()
        ctx = await outreach._gather_context("s")
        return ctx, ev.time() - t

    ctx, elapsed = asyncio.run(run())
    assert ctx.personality_cue and ctx.situation_status == "ok"
    assert elapsed < 0.55, elapsed


def test_grounding_summary_records_cue_node_ids_not_text():
    from scripts.endogenous_outreach import OutreachContext, grounding_summary

    ctx = OutreachContext(
        curiosity_summaries=[],
        recent_turns=[],
        presence=None,
        personality_cue="Things you keep coming back to ...",
        personality_cue_ids=["disp-a", "disp-b"],
    )
    g = grounding_summary(ctx)
    assert g["personality_cue"] is True
    assert g["personality_cue_ids"] == ["disp-a", "disp-b"]
    assert g["personality_cue_skip"] is None
    assert "Things you keep" not in str(g)
