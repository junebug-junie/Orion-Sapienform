"""The consult lifecycle, end to end, with an in-memory store: offer -> Orion turn -> Claude turn ->
ending -> brief (chat + email) or answer -> Juniper's decision -> Orion told once.

Tests the LIFECYCLE, not the arithmetic: each step is driven by the same calls the live loop makes
(tick, the relay's utterance handler, the decision route's store call)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from orion.consult.core import consult_room_id, orion_session_id
from orion.schemas.claude_limit import ClaudeLimitObservationV1
from orion.schemas.room_claude import ExternalRoomResponderV1, RoomClaudeUtteranceV1
from scripts import consult_loop as cl

NOW = datetime(2026, 11, 11, 18, 0, tzinfo=timezone.utc)


class FakeStore:
    """In-memory stand-ins for the module's SQL functions (same signatures)."""

    def __init__(self) -> None:
        self.rows: Dict[str, Dict[str, Any]] = {}
        self.turn_rows: Dict[str, List[Dict[str, Any]]] = {}
        self.n = 0

    def install(self, monkeypatch) -> None:
        for name in ("offer", "get", "open_consult", "next_pending", "expire_old_pending", "set_status", "turns",
                     "add_turn", "briefs_sent_since", "decide", "take_outcomes"):
            monkeypatch.setattr(cl, name, getattr(self, name))

    async def offer(self, pool, *, source_kind, source_ref, question, detail=""):
        if any(r["source_ref"] == source_ref for r in self.rows.values()):
            return None
        self.n += 1
        cid = f"c{self.n}"
        self.rows[cid] = dict(consult_id=cid, source_kind=source_kind, source_ref=source_ref, source_detail=detail,
                              question=question, status="pending", status_reason=None, ending=None, turns=0,
                              claude_cost_usd=0.0, answer=None, brief_json=None, brief_text=None, brief_sent_at=None,
                              email_status=None, decision=None, decision_reason=None, orion_told_run_id=None,
                              created_at=NOW, updated_at=NOW)
        self.turn_rows[cid] = []
        return cid

    async def get(self, pool, cid):
        row = self.rows.get(cid)
        return dict(row) if row else None

    async def open_consult(self, pool):
        rows = [r for r in self.rows.values() if r["status"] == "open"]
        return dict(rows[0]) if rows else None

    async def next_pending(self, pool, *, max_age):
        rows = [r for r in self.rows.values() if r["status"] == "pending"]
        return dict(rows[0]) if rows else None

    async def expire_old_pending(self, pool, *, max_age):
        return 0

    async def set_status(self, pool, cid, status, reason=None, **fields):
        self.rows[cid].update(status=status, status_reason=reason, updated_at=NOW, **fields)

    async def turns(self, pool, cid):
        return [dict(t) for t in self.turn_rows[cid]]

    async def add_turn(self, pool, cid, *, turn_no, speaker, text, ending=None, model=None, cost_usd=None, error=None):
        assert turn_no == len(self.turn_rows[cid]) + 1
        self.turn_rows[cid].append(dict(turn_no=turn_no, speaker=speaker, text=text, ending=ending, error=error))
        self.rows[cid]["turns"] += 1
        self.rows[cid]["claude_cost_usd"] += float(cost_usd or 0.0)

    async def briefs_sent_since(self, pool, since):
        return sum(1 for r in self.rows.values() if r["brief_sent_at"] is not None)

    async def decide(self, pool, cid, *, decision, reason):
        row = self.rows[cid]
        if row["status"] != "brief_sent" or row["decision"]:
            return None
        row.update(decision=decision, decision_reason=reason)
        return dict(row)

    async def take_outcomes(self, pool, *, run_id, limit=5):
        out = []
        for row in self.rows.values():
            ready = row["status"] == "answered" or (row["status"] == "brief_sent" and row["decision"])
            if ready and not row["orion_told_run_id"]:
                row["orion_told_run_id"] = run_id
                out.append(dict(row))
        return out


class FakeOutreach:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def compose(self, prompt, correlation_id, retrieval_query=None, lane="agent_first", session_id=None):
        self.calls.append(dict(prompt=prompt, session_id=session_id, lane=lane))
        return (self.replies.pop(0) if self.replies else ""), {"model": "MODEL_X"}


class FakeRelay:
    def __init__(self):
        self.invites = []

    async def invite(self, **kw):
        self.invites.append(kw)


class FakeNotify:
    def __init__(self):
        self.sent = []

    def send(self, request):
        self.sent.append(request)
        return SimpleNamespace(ok=True, email_status="sent")


def meter(kind, state="clear", observed=True):
    return ClaudeLimitObservationV1(
        observed_at=NOW - timedelta(seconds=30), kind=kind, window_hours=5.0 if kind == "session_limit" else 168.0,
        window_start=NOW - timedelta(hours=5), window_end=NOW, state=state, observed=observed, event_count=0,
        staleness_sec=20.0 if observed else None, observed_message_count=10 if observed else 0, scanned_file_count=3,
    )


def brief_block(ending="agreed_proposal", **over):
    brief = {
        "why_it_matters": "Journal runs stopped.",
        "checked": [{"claim": "The lease is never released.", "proof": "orion/curiosity/journal.py:88"}],
        "disagreement": "none",
        "impact": {"what_changes": "Journal runs again.", "number_to_watch": "journal rows/day",
                   "how_we_see_it": "python -P -m orion.dig journal"},
        "cost": {"doing_the_work": {"value": "2 GPU-min/day", "source": "orion.dig energy"},
                 "doing_nothing": {"unknown_reason": "no dig measures it"}},
        "recommendation": "Release the lease.",
        "question_for_juniper": "Should a dev session make this change?",
    }
    brief.update(over)
    return "I checked the code.\n```json\n" + json.dumps({"ending": ending, "brief": brief}) + "\n```"


def utt(cid, text, ok=True, cost=0.25):
    return RoomClaudeUtteranceV1(request_id="r", room_id=consult_room_id(cid), text=text, ok=ok, cost_usd=cost,
                                 model="claude-sonnet-5-5", error=None if ok else "boom",
                                 responder=ExternalRoomResponderV1(participant_id="claude"))


@pytest.fixture()
def world(monkeypatch):
    store = FakeStore()
    store.install(monkeypatch)
    pushed: List[dict] = []
    queue = SimpleNamespace(put_nowait=pushed.append)
    outreach = FakeOutreach(["Why does journal stop?", "I agree: release the lease."])
    relay, notify = FakeRelay(), FakeNotify()
    loop = cl.ConsultLoop(
        enabled=True, pool_provider=lambda: object(), outreach_provider=lambda: outreach,
        relay_provider=lambda: relay, notify_client=notify,
        connections_provider=lambda: {"conn1": {"queue": queue}}, clock=lambda: NOW,
    )
    loop.meter = {"session_limit": meter("session_limit"), "weekly_limit": meter("weekly_limit")}
    monkeypatch.setattr(loop, "_spawn", lambda coro: asyncio.get_event_loop().create_task(coro))

    async def _history(cid, text):
        pushed.append({"history": cid})

    monkeypatch.setattr(loop, "_history", _history)
    return SimpleNamespace(store=store, loop=loop, outreach=outreach, relay=relay, notify=notify, pushed=pushed)


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_full_consult_to_brief_then_decision_then_orion_told(world):
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r1", question="Journal broke?")
    assert await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r1", question="again") is None

    await world.loop.tick()  # opens, Orion turn 1, Claude invited
    row = world.store.rows[cid]
    assert row["status"] == "open" and row["status_reason"] == cl.AWAIT_CLAUDE
    assert world.outreach.calls[0]["session_id"] == orion_session_id(cid), "Orion's turn must not carry Juniper's chats"
    assert world.relay.invites[0]["room_id"] == consult_room_id(cid)
    assert "Journal broke?" in world.relay.invites[0]["prompt"]

    await world.loop.on_claude_utterance(utt(cid, "What does the lease log say?"))  # continue
    await _settle()
    assert len(world.outreach.calls) == 2 and "Claude: What does the lease log say?" in world.outreach.calls[1]["prompt"]
    assert world.store.rows[cid]["status_reason"] == cl.AWAIT_CLAUDE

    await world.loop.on_claude_utterance(utt(cid, brief_block()))
    row = world.store.rows[cid]
    assert row["status"] == "brief_sent" and row["ending"] == "agreed_proposal"
    assert row["brief_text"].startswith("@Juniper") and "$0.50" in row["brief_text"]
    assert row["email_status"] == "sent" and world.notify.sent[0].dedupe_key == f"consult:{cid}"
    kinds = [p.get("kind") for p in world.pushed]
    assert kinds.count(cl.CONSULT_TURN_KIND) == 4 and cl.CONSULT_BRIEF_KIND in kinds
    assert {"history": cid} in world.pushed, "the brief, and only the brief, goes to chat history"

    assert await cl.decide(None, cid, decision="yes", reason=None)
    assert await cl.decide(None, cid, decision="no", reason=None) is None, "answered once"
    told = await cl.take_outcomes(None, run_id="run-9")
    assert [r["consult_id"] for r in told] == [cid]
    assert await cl.take_outcomes(None, run_id="run-10") == [], "Orion is told once"


@pytest.mark.asyncio
async def test_answered_sends_nothing_to_juniper(world):
    cid = await world.loop.offer(source_kind="stuck_prior", source_ref="prior:p1", question="Why?")
    await world.loop.tick()
    await world.loop.on_claude_utterance(utt(cid, 'By design.\n```json\n{"ending": "answered", "answer": "It is by design: x.py:3"}\n```'))
    row = world.store.rows[cid]
    assert row["status"] == "answered" and row["answer"] == "It is by design: x.py:3"
    assert world.notify.sent == [] and cl.CONSULT_BRIEF_KIND not in [p.get("kind") for p in world.pushed]


@pytest.mark.asyncio
async def test_a_refused_brief_gets_one_retry_then_is_never_sent(world):
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r2", question="q")
    await world.loop.tick()
    bad = brief_block(cost={"doing_the_work": {"value": "3 kWh"}, "doing_nothing": {"unknown_reason": "n/a"}})
    await world.loop.on_claude_utterance(utt(cid, bad))
    assert world.store.rows[cid]["status_reason"] == f"{cl.GATE_RETRY}cost.doing_the_work"
    assert "cost.doing_the_work" in world.relay.invites[-1]["prompt"]
    await world.loop.on_claude_utterance(utt(cid, bad))
    row = world.store.rows[cid]
    assert row["status"] == "brief_refused" and row["status_reason"] == "missing:cost.doing_the_work"
    assert world.notify.sent == []


@pytest.mark.asyncio
async def test_retry_with_a_complete_brief_is_sent(world):
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r3", question="q")
    await world.loop.tick()
    await world.loop.on_claude_utterance(utt(cid, brief_block(recommendation=" ")))
    await world.loop.on_claude_utterance(utt(cid, brief_block()))
    assert world.store.rows[cid]["status"] == "brief_sent"


@pytest.mark.asyncio
async def test_budget_refusal_holds_the_claude_turn_and_retries(world):
    world.loop.meter["session_limit"] = meter("session_limit", state="limited")
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r4", question="q")
    await world.loop.tick()
    assert world.store.rows[cid]["status"] == "pending", "no consult opens while the budget refuses"
    world.loop.meter["session_limit"] = meter("session_limit")
    await world.loop.tick()
    assert world.relay.invites, "opens once the budget clears"
    # Budget closes mid-consult: the Claude turn waits, nothing is lost.
    world.loop.meter["weekly_limit"] = meter("weekly_limit", state="limited")
    await world.loop.on_claude_utterance(utt(cid, "And the lease log?"))
    await _settle()
    assert world.store.rows[cid]["status_reason"] == "budget:weekly_limited"
    sent = len(world.relay.invites)
    world.loop.meter["weekly_limit"] = meter("weekly_limit")
    await world.loop.tick()
    assert len(world.relay.invites) == sent + 1 and world.store.rows[cid]["status_reason"] == cl.AWAIT_CLAUDE


@pytest.mark.asyncio
async def test_last_turn_without_an_ending_is_refused(world):
    world.loop.max_turns = 2
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r5", question="q")
    await world.loop.tick()
    await world.loop.on_claude_utterance(utt(cid, "still thinking"))
    assert world.store.rows[cid]["status"] == "brief_refused"


@pytest.mark.asyncio
async def test_empty_orion_turn_fails_the_consult(world):
    world.outreach.replies = [""]
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r6", question="q")
    await world.loop.tick()
    assert world.store.rows[cid]["status"] == "failed" and world.relay.invites == []


@pytest.mark.asyncio
async def test_claude_error_fails_and_unexpected_replies_are_ignored(world):
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r7", question="q")
    await world.loop.on_claude_utterance(utt(cid, "too early"))  # pending: ignored
    assert world.store.turn_rows[cid] == []
    await world.loop.tick()
    await world.loop.on_claude_utterance(utt(cid, "", ok=False))
    assert world.store.rows[cid]["status"] == "failed"


@pytest.mark.asyncio
async def test_weekly_cap_stops_new_consults(world):
    world.loop.weekly_brief_cap = 0
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r8", question="q")
    tick = await world.loop.tick()
    assert tick["skipped"] == "weekly_brief_cap" and world.store.rows[cid]["status"] == "pending"


@pytest.mark.asyncio
async def test_claude_silence_times_out(world, monkeypatch):
    cid = await world.loop.offer(source_kind="curiosity_run", source_ref="curiosity_run:r9", question="q")
    await world.loop.tick()
    world.store.rows[cid]["updated_at"] = NOW - timedelta(seconds=world.loop.claude_timeout_sec + 1)
    await world.loop.tick()
    assert world.store.rows[cid]["status"] == "failed" and world.store.rows[cid]["status_reason"] == "claude_no_reply"


def test_meter_messages_are_kept_per_kind(world):
    obs = meter("weekly_limit", state="limited")
    world.loop.note_meter({"data": json.dumps({"payload": obs.model_dump(mode="json")})})
    world.loop.note_meter({"data": "not json"})
    assert world.loop.meter["weekly_limit"].state == "limited"
    assert world.loop.budget() == "weekly_limited"
