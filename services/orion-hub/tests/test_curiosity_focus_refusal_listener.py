"""Hub's responder for the curiosity replica's focus refusals (patch 4, design 2026-11-04 section 9).

The focus table is ``orion.curiosity.tests.focus_fixtures.FocusTable`` (the same REFUSE_SQL guard); the real
Postgres lifecycle of ``refuse_focus`` is orion/curiosity/tests/test_focus_lifecycle.py."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import scripts.curiosity_focus_refusal_listener as rl
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity import focus as cf
from orion.curiosity.tests.focus_fixtures import FocusPool, FocusTable
from orion.schemas.curiosity_replica import (
    FOCUS_REFUSAL_REQUEST_KIND,
    FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX,
    FOCUS_REFUSAL_RESULT_KIND,
    CuriosityFocusRefusalResultV1,
)

REASON = "the last three runs died on cabinet questions"


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def queued(table, *, origin="self", corr="prime:thought1"):
    pool = FocusPool(table)
    q = asyncio.run(cf.queue_focus(pool, question="What heats the cabinet after 14:00?", why="heat", origin=origin,
                                   chat_correlation_id=corr, keep_pending=origin == "self"))
    return q.focus


def listener(table, *, enabled=True, pool="default"):
    lis = rl.CuriosityFocusRefusalListener(pool_provider=lambda: FocusPool(table) if pool == "default" else pool,
                                           source_ref=ServiceRef(name="orion-hub"), enabled=enabled)
    lis.bus = Bus()
    return lis


def envelope(focus_id, *, reply_to=None, kind=FOCUS_REFUSAL_REQUEST_KIND, reason=REASON):
    corr = uuid4()
    return BaseEnvelope(kind=kind, correlation_id=corr, source=ServiceRef(name="orion-mind"),
                        reply_to=reply_to or f"{FOCUS_REFUSAL_RESULT_CHANNEL_PREFIX}{corr}",
                        payload={"focus_id": str(focus_id), "reason": reason, "act_id": "act_answer1",
                                 "prime_act_id": "act_prime_q"})


def answer(lis):
    [(channel, env)] = lis.bus.published
    assert env.kind == FOCUS_REFUSAL_RESULT_KIND
    return channel, env, CuriosityFocusRefusalResultV1.model_validate(env.payload)


def test_a_pending_prime_focus_is_refused_with_its_reason_and_act():
    table = FocusTable()
    focus = queued(table)
    lis = listener(table)
    env = envelope(focus.focus_id)
    asyncio.run(lis.handle(env))
    channel, out, result = answer(lis)
    assert channel == env.reply_to and out.correlation_id == env.correlation_id
    assert result.ok and result.focus_id == str(focus.focus_id) and result.error is None
    pending = asyncio.run(cf.read_pending(FocusPool(table)))
    assert pending is None                                      # no run will be shown it
    [row] = table.rows
    assert row["refused_by"] == "curiosity.replica" and row["refused_reason"] == REASON
    assert row["refused_act_id"] == "act_answer1"


def test_juniper_focus_and_orions_own_are_never_refused():
    for origin, corr, words in (("juniper", "chat-corr-1", "Juniper asked for this focus"),
                                ("self", "chat-corr-2", "only Prime's focus can be refused")):
        table = FocusTable()
        focus = queued(table, origin=origin, corr=corr)
        lis = listener(table)
        asyncio.run(lis.handle(envelope(focus.focus_id)))
        _, _, result = answer(lis)
        assert not result.ok and words in result.error
        assert asyncio.run(cf.read_pending(FocusPool(table))) is not None   # still waiting for its run


def test_the_wrong_reply_channel_or_kind_gets_no_answer():
    table = FocusTable()
    focus = queued(table)
    lis = listener(table)
    asyncio.run(lis.handle(envelope(focus.focus_id, reply_to="orion:somewhere:else")))
    asyncio.run(lis.handle(envelope(focus.focus_id, kind="introspect.tool.request.v1")))
    assert lis.bus.published == [] and asyncio.run(cf.read_pending(FocusPool(table))) is not None


def test_switched_off_no_pool_a_db_error_or_a_bad_request_write_nothing_and_say_so():
    table = FocusTable()
    focus = queued(table)
    for lis, words in ((listener(table, enabled=False), "switched off"), (listener(table, pool=None), "could not be written")):
        asyncio.run(lis.handle(envelope(focus.focus_id)))
        _, _, result = answer(lis)
        assert not result.ok and words in result.error

    class Boom:
        def acquire(self):
            raise RuntimeError("pool closed")

    lis = listener(table, pool=Boom())
    asyncio.run(lis.handle(envelope(focus.focus_id)))
    assert answer(lis)[2].error == rl.UNAVAILABLE
    lis = listener(table)
    asyncio.run(lis.handle(envelope(focus.focus_id, reason="")))
    assert answer(lis)[2].error.startswith("invalid refusal request")
    assert asyncio.run(cf.read_pending(FocusPool(table))) is not None


def test_a_request_past_its_not_after_writes_nothing():
    """The replica stopped waiting and reads the row instead: a late write would make its record wrong."""
    from datetime import datetime, timedelta, timezone

    table = FocusTable()
    focus = queued(table)
    lis = listener(table)
    env = envelope(focus.focus_id)
    env = env.model_copy(update={"payload": {**env.payload,
                                             "not_after": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}})
    asyncio.run(lis.handle(env))
    _, _, result = answer(lis)
    assert not result.ok and result.error == rl.EXPIRED
    assert asyncio.run(cf.read_pending(FocusPool(table))) is not None
