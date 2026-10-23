"""The dream cycle's `make` act (body economy A5): opened once, closed once, on every path."""
from __future__ import annotations

import asyncio
import json

import pytest


def _rows():
    return {
        "metacog": [
            {"id": "m1", "summary": "Recall returned empty for three turns in a row", "severity": "critical",
             "trigger_kind": "recall_empty", "tags": ["recall"]},
            {"id": "m2", "summary": "Latency spike on the chat lane", "severity": "degraded",
             "trigger_kind": "latency", "tags": ["latency"]},
        ],
        "compaction_request": [{"request_id": "r1", "theme": "juniper's sleep schedule", "reason": "recurs"}],
        "resonance": [{"alert_id": "a1", "theme_key": "being watched", "violation_count": 4}],
        "crystallization": [
            {"crystallization_id": "c1", "subject": "Juniper prefers plain English",
             "summary": "dense answers get re-said", "salience": 0.8, "tags": ["style"]},
        ],
    }


LINK = json.dumps({"link": True, "claim": "These two recur together more often than chance", "why": "w"})


class _Fakes:
    def __init__(self, rows=None, *, answer=LINK, fail_llm=False, persist=None, act_on=True, publish_raises=False):
        self.rows = _rows() if rows is None else rows
        self.answer, self.fail_llm = answer, fail_llm
        self.persist = persist
        self.act_on, self.publish_raises = act_on, publish_raises
        self.events: list = []
        self.call_corrs: list = []
        self.persisted: list = []

    def deps(self):
        from app import act_emit
        from app.cycle import CycleDeps

        async def complete(prompt):
            self.call_corrs.append(act_emit.current_lease_correlation())
            if self.fail_llm:
                raise RuntimeError("gateway down")
            return self.answer

        async def publish(event):
            if self.publish_raises:
                raise ConnectionError("bus gone")
            self.events.append(event)

        def persist(cycle):
            if self.persist is not None:
                return self.persist(cycle)
            self.persisted.append(cycle)
            return True

        return CycleDeps(
            load_source_rows=lambda since, limit, until=None: {} if until is not None else self.rows,
            load_idle_minutes=lambda: 120.0,
            load_last_window_start=lambda: None,
            load_last_attempt_end=lambda: None,
            persist_cycle=persist,
            complete=complete,
            publish_act=publish if self.act_on else None,
        )


def _kinds(events):
    return [e.event_kind for e in events]


def test_completed_sleep_opens_once_and_closes_once_produced():
    from app import act_emit
    from app.cycle import run_cycle_once
    from orion.grammar.acts import stable_act_id

    f = _Fakes()
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "completed" and cycle.hypotheses
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    opened, closed = f.events
    assert opened.act.act_id == closed.act.act_id == stable_act_id("dream.cycle", "make", cycle.cycle_id)
    assert opened.act.actor == "dream.cycle" and opened.act.verb == "make"
    assert opened.provenance.source_service == "orion-dream"
    assert closed.act.outcome.value == 1.0 and closed.act.outcome.source == "producer"
    assert closed.act.cost.seconds is not None and closed.act.cost.seconds >= 0.0
    # V1 cost only: the ledger fills the V2 fields itself.
    assert type(closed.act.cost).__name__ == "ActCostV1"
    corr = act_emit.cycle_correlation_id(cycle.cycle_id)
    assert opened.correlation_id == closed.correlation_id == corr
    # Every recombination call carried the act's correlation id (the GPU lease join) ...
    assert f.call_corrs and set(f.call_corrs) == {corr}
    # ... and the binding ends with the cycle.
    assert act_emit.current_lease_correlation() is None
    assert opened.act.expect.signal_id == "artifact_produced"
    assert closed.act.expect == opened.act.expect


def test_all_llm_calls_failing_closes_zero():
    from app.cycle import run_cycle_once

    f = _Fakes(fail_llm=True)
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "failed"
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    assert f.events[1].act.outcome.value == 0.0
    assert "status=failed" in f.events[1].act.outcome.note


def test_completed_with_no_hypotheses_closes_zero():
    from app.cycle import run_cycle_once

    f = _Fakes(answer=json.dumps({"link": False, "claim": "", "why": "nothing"}))
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "completed" and not cycle.hypotheses
    assert f.events[-1].act.outcome.value == 0.0


def test_unpersisted_sleep_closes_zero():
    from app.cycle import run_cycle_once

    f = _Fakes(persist=lambda c: False)
    asyncio.run(run_cycle_once(f.deps()))
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    assert f.events[1].act.outcome.value == 0.0 and "persisted=False" in f.events[1].act.outcome.note


def test_exception_after_open_closes_zero_and_reraises():
    from app import act_emit
    from app.cycle import run_cycle_once

    def boom(cycle):
        raise RuntimeError("db gone")

    f = _Fakes(persist=boom)
    with pytest.raises(RuntimeError, match="db gone"):
        asyncio.run(run_cycle_once(f.deps()))
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    assert f.events[1].act.outcome.value == 0.0
    assert f.events[1].act.outcome.note.startswith("raised:RuntimeError")
    assert act_emit.current_lease_correlation() is None


def test_cancellation_after_open_closes_zero():
    from app.cycle import run_cycle_once

    f = _Fakes()
    deps = f.deps()

    async def hang(prompt):
        await asyncio.sleep(3600)

    deps.complete = hang

    async def go():
        task = asyncio.create_task(run_cycle_once(deps))
        while not f.events:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    assert f.events[1].act.outcome.note.startswith("raised:CancelledError")


def test_no_act_when_nothing_to_replay_or_not_due():
    from app.cycle import run_cycle_once

    empty = _Fakes(rows={})
    cycle = asyncio.run(run_cycle_once(empty.deps(), trigger="manual", force=True))
    assert cycle.status == "empty" and empty.events == []

    not_due = _Fakes()
    deps = not_due.deps()
    deps.load_idle_minutes = lambda: 1.0
    assert asyncio.run(run_cycle_once(deps)) is None and not_due.events == []


def test_flag_off_no_act_and_calls_keep_their_own_ids():
    from app.cycle import run_cycle_once

    f = _Fakes(act_on=False)
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "completed" and f.events == []
    assert f.call_corrs and set(f.call_corrs) == {None}


def test_a_failing_publish_never_fails_the_sleep():
    from app.cycle import run_cycle_once

    f = _Fakes(publish_raises=True)
    cycle = asyncio.run(run_cycle_once(f.deps()))
    assert cycle.status == "completed" and f.persisted


def test_close_is_once_and_act_id_is_stable_for_the_cycle():
    from app import act_emit
    from app.cycle import CycleActTrace

    f = _Fakes()
    trace = CycleActTrace(f.deps(), "dc-abc")
    asyncio.run(trace.open())
    asyncio.run(trace.close(True, "first"))
    asyncio.run(trace.close(False, "second"))
    assert _kinds(f.events) == ["act_opened", "act_closed"]
    assert f.events[1].act.outcome.note == "first"
    # Same cycle id -> same act id and correlation on any rebuild (a re-sent close is first-close-wins).
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    a = act_emit.opened_event("dc-abc", horizon=480.0, at=now)
    b = act_emit.opened_event("dc-abc", horizon=480.0, at=now)
    assert a.act.act_id == b.act.act_id == f.events[0].act.act_id
    assert a.correlation_id == b.correlation_id
    assert a.event_id != b.event_id


def test_horizon_is_the_cycles_own_deadline():
    from app import act_emit

    assert act_emit.horizon_sec(hypotheses_per_cycle=3, control_per_cycle=1, llm_timeout_sec=90) == 480.0


def test_llm_complete_uses_the_cycle_correlation_with_a_unique_reply_channel():
    from app import act_emit, llm

    sent = []

    class _Decoded:
        ok = True
        error = None

        class envelope:
            payload = {"content": "hello"}

    class _Codec:
        def decode(self, data):
            return _Decoded()

    class _Bus:
        codec = _Codec()

        async def rpc_request(self, channel, env, *, reply_channel, timeout_sec):
            sent.append((str(env.correlation_id), reply_channel))
            return {"data": b""}

    corr = act_emit.cycle_correlation_id("dc-xyz")

    async def go():
        token = act_emit.set_lease_correlation(corr)
        try:
            await llm.complete(_Bus(), "p")
            await llm.complete(_Bus(), "p")
        finally:
            act_emit.reset_lease_correlation(token)
        await llm.complete(_Bus(), "p")

    asyncio.run(go())
    assert sent[0][0] == sent[1][0] == corr
    assert sent[0][1] != sent[1][1]  # reply channels stay per call
    assert sent[2][0] != corr  # outside a cycle: a fresh id, as before
