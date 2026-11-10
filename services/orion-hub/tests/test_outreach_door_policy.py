"""Door policy (2026-11-10): which composed messages may reach Juniper.

Pure seams plus the wiring at each door: curiosity (line / Juniper-asked focus), the tick (no
investigate-line lanes, no unsolicited turns, daydream novelty), the Why Juniper line, and the
already-told guard.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from orion.hub.why_juniper import WHY_JUNIPER_INSTRUCTION, split_why_juniper
from scripts.outreach_door_policy import (
    ALREADY_TOLD_REASON,
    NOT_FOR_JUNIPER_REASON,
    curiosity_finding_key,
    curiosity_reaches_juniper,
    daydream_content_id,
)


# --- curiosity rule ----------------------------------------------------------------------------


def test_self_inquiry_reaches_juniper_without_a_lookup():
    assert curiosity_reaches_juniper("self_inquiry") == (True, "line:self_inquiry")


@pytest.mark.parametrize("line", ["investigate", "self_sense_eval", "", None])
def test_other_lines_do_not_reach_juniper_unasked(line):
    ok, why = curiosity_reaches_juniper(line, focus_origin="self", focus_chat_correlation_id="c-1")
    assert ok is False and why.startswith("line:")


def test_investigate_reaches_juniper_only_when_she_asked_with_her_chat_turn():
    # Live 2026-11-05: run f8eaf82252e8, "Did the 05:00 MST backup run succeed?", focus origin
    # 'juniper' with her chat turn. The only investigate send of the 14 days she asked for.
    assert curiosity_reaches_juniper(
        "investigate", focus_origin="juniper", focus_chat_correlation_id="chat-corr"
    ) == (True, "asked_by_juniper")
    # Juniper origin without the chat turn it came from is not proof she asked.
    assert curiosity_reaches_juniper("investigate", focus_origin="juniper", focus_chat_correlation_id="")[0] is False
    assert curiosity_reaches_juniper("investigate", focus_origin="juniper")[0] is False


# --- Why Juniper line --------------------------------------------------------------------------


def test_why_juniper_line_is_stripped_and_returned():
    text = "I noticed I keep quoting myself.\n\nWhy Juniper: you asked how I see my own likes."
    assert split_why_juniper(text) == (
        "I noticed I keep quoting myself.", "you asked how I see my own likes.")


@pytest.mark.parametrize("trailer", [
    "**Why Juniper:** it changes how I answer you.",
    "why juniper - it changes how I answer you.",
    "Why_Juniper: it changes how I answer you.",
    "> *Why Juniper*: it changes how I answer you.",
])
def test_why_juniper_markdown_shapes(trailer):
    body, why = split_why_juniper(f"Message body.\n---\n{trailer}\n\n")
    assert body == "Message body."
    assert why == "it changes how I answer you."


@pytest.mark.parametrize("text", [
    "Message with no reason line.",
    "Why Juniper: it matters.\nBut then more text after it.",
    "Message.\nWhy Juniper:",
    "Message.\nWhy Juniper:   **",
    "",
])
def test_missing_or_empty_why_juniper_gives_empty_reason(text):
    assert split_why_juniper(text)[1] == ""


def test_every_composing_prompt_asks_for_the_line():
    from orion.curiosity.outreach_prompt import build_outreach_composition_prompt

    assert WHY_JUNIPER_INSTRUCTION in build_outreach_composition_prompt(finding_text="f", reach_out_why="w")


# --- identities --------------------------------------------------------------------------------


def test_daydream_id_is_stable_over_whitespace_and_distinct_by_content():
    a = daydream_content_id("A  star chart\nover water")
    assert a == daydream_content_id("A star chart over water") and a.startswith("daydream:")
    assert a != daydream_content_id("A Roman aqueduct")
    assert daydream_content_id("  ") == ""


def test_curiosity_finding_key():
    assert curiosity_finding_key("abc123") == "curiosity_run:abc123"


# --- curiosity door wiring ---------------------------------------------------------------------


class _Outreach:
    def __init__(self, told: bool = False) -> None:
        self.recorded: list[tuple[str, dict]] = []
        self.offered: list[dict] = []
        self._told = told

    def record_blocked(self, reason, *, correlation_id, source, extra=None):
        self.recorded.append((reason, dict(extra or {})))
        return {"reason": reason}

    def blocked_reason(self, **_kw):
        return None

    async def already_told(self, key):
        return self._told

    async def offer_message(self, **kw):
        self.offered.append(kw)
        return {"outreach": True, "reason": "sent"}


def _loop(monkeypatch, outreach, focus=None):
    import scripts.curiosity_investigation as ci

    loop = ci.CuriosityInvestigation.__new__(ci.CuriosityInvestigation)
    loop.outreach_enabled = True
    loop._outreach_provider = lambda: outreach
    loop._reader = None
    loop._pool_provider = (lambda: object()) if focus is not False else (lambda: None)

    async def read_for_runs(pool, run_ids):
        return {} if focus in (None, False) else {run_ids[0]: focus}

    async def generate(*a, **kw):
        return "Text.\nWhy Juniper: because.", {}

    monkeypatch.setattr(ci, "read_focus_for_runs", read_for_runs)
    loop._generate = generate
    return loop


def _reach(loop, line):
    from scripts.curiosity_investigation import TurnOutcome

    outcome = TurnOutcome(run_id="r1", continue_line=False, continue_note="", reach_out=True, reach_out_why="w")
    return asyncio.run(loop._maybe_reach_out_inner(
        outcome=outcome, finding_text="finding", run_id="r1", hop_notes=[], line=line, correlation_id="c"))


def test_investigate_run_is_refused_before_composing(monkeypatch):
    outreach = _Outreach()
    loop = _loop(monkeypatch, outreach, focus=SimpleNamespace(origin="self", chat_correlation_id="prime:x"))
    assert _reach(loop, "investigate") == NOT_FOR_JUNIPER_REASON
    assert outreach.offered == []
    reason, extra = outreach.recorded[0]
    assert reason == NOT_FOR_JUNIPER_REASON and extra["door_policy"] == "line:investigate"
    assert extra["run_id"] == "r1" and extra["line"] == "investigate"


def test_unreadable_focus_is_unasked(monkeypatch):
    outreach = _Outreach()
    loop = _loop(monkeypatch, outreach, focus=False)
    assert _reach(loop, "investigate") == NOT_FOR_JUNIPER_REASON
    assert outreach.recorded[0][1]["door_policy"] == "line:investigate:focus_unreadable"


def test_juniper_asked_investigate_and_self_inquiry_are_offered_with_why_and_key(monkeypatch):
    for line, focus in (("investigate", SimpleNamespace(origin="juniper", chat_correlation_id="chat-1")),
                        ("self_inquiry", None)):
        outreach = _Outreach()
        loop = _loop(monkeypatch, outreach, focus=focus)
        assert _reach(loop, line) is None
        offered = outreach.offered[0]
        assert offered["require_why_juniper"] is True
        assert offered["finding_key"] == "curiosity_run:r1"
        assert offered["meta"]["door_policy"] in ("asked_by_juniper", "line:self_inquiry")


def test_already_told_run_is_not_composed_again(monkeypatch):
    outreach = _Outreach(told=True)
    loop = _loop(monkeypatch, outreach)
    assert _reach(loop, "self_inquiry") == ALREADY_TOLD_REASON
    assert outreach.offered == []
    assert outreach.recorded[0][1]["finding_key"] == "curiosity_run:r1"


# --- offer_message gates -----------------------------------------------------------------------


def _endogenous(monkeypatch, told: Any = False):
    import scripts.endogenous_outreach as eo
    import scripts.endogenous_outreach_decisions as dec

    rows: list[dict] = []
    monkeypatch.setattr(dec, "record_decision", lambda result, **kw: rows.append(dict(result)))
    monkeypatch.setattr(dec, "finding_already_told", lambda key, **kw: told)
    o = eo.EndogenousOutreach(
        enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=4, quiet_start_hour=-1,
        quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0, notify_channel="orion:notify:in_app",
        fallback_session_id="orion_outreach", trigger_evaluator=lambda: None,
    )
    delivered: list[str] = []

    async def deliver(*, text, **kw):
        delivered.append(text)

    async def fresh():
        return None

    async def recover():
        return None

    o._deliver = deliver
    o._fetch_fresh_tension_reason = fresh
    o._recover_sent_today = recover
    return o, rows, delivered


def test_offer_message_without_why_juniper_is_not_sent(monkeypatch):
    o, rows, delivered = _endogenous(monkeypatch)
    out = asyncio.run(o.offer_message(text="A finding with no reason.", correlation_id="c", tag="t",
                                      skip_schedule_gates=True, require_why_juniper=True))
    assert out["reason"] == "no_why_juniper" and delivered == []
    assert rows[-1]["reason"] == "no_why_juniper"


def test_offer_message_strips_and_records_why_juniper(monkeypatch):
    o, rows, delivered = _endogenous(monkeypatch)
    out = asyncio.run(o.offer_message(text="My likes were not self-quotation.\nWhy Juniper: you asked.",
                                      correlation_id="c", tag="t", skip_schedule_gates=True,
                                      require_why_juniper=True, finding_key="curiosity_run:r1"))
    assert out["reason"] == "sent"
    assert delivered == ["My likes were not self-quotation."]
    assert rows[-1]["why_juniper"] == "you asked." and rows[-1]["finding_key"] == "curiosity_run:r1"


def test_offer_message_does_not_send_a_finding_already_told(monkeypatch):
    o, rows, delivered = _endogenous(monkeypatch, told=True)
    out = asyncio.run(o.offer_message(text="Again.\nWhy Juniper: x.", correlation_id="c", tag="t",
                                      skip_schedule_gates=True, require_why_juniper=True,
                                      finding_key="curiosity_run:r1"))
    assert out["reason"] == ALREADY_TOLD_REASON and delivered == []


def test_offer_message_unknown_ledger_does_not_invent_silence(monkeypatch):
    o, _rows, delivered = _endogenous(monkeypatch, told=None)
    out = asyncio.run(o.offer_message(text="New.\nWhy Juniper: x.", correlation_id="c", tag="t",
                                      skip_schedule_gates=True, require_why_juniper=True,
                                      finding_key="curiosity_run:r2"))
    assert out["reason"] == "sent" and delivered == ["New."]


def test_office_style_offer_without_the_flag_is_unchanged(monkeypatch):
    o, _rows, delivered = _endogenous(monkeypatch)
    out = asyncio.run(o.offer_message(text="What are you up to?", correlation_id="c", tag="office_reach_out",
                                      skip_rate_gates=True))
    assert out["reason"] == "sent" and delivered == ["What are you up to?"]
