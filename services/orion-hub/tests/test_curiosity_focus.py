"""Curiosity focus in Hub: the queue_curiosity_focus responder (producer), the investigate loop
that takes it and stamps the outcome (consumer), the `curiosity` read tool's kind=focus, and the
Curiosity tab's pending tile and Clear route.

The focus table is ``orion.curiosity.tests.focus_fixtures.FocusTable``; the same lifecycle runs
against real Postgres in orion/curiosity/tests/test_focus_lifecycle.py.
"""
from __future__ import annotations

import ast
import asyncio
import logging
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import scripts.curiosity_focus_listener as fl
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity import focus as cf
from orion.curiosity.tests.focus_fixtures import FocusPool, FocusTable
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1

HUB = Path(__file__).resolve().parents[1]
CHAT = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="chat-run", parent_trace_id="chat-corr-1", memory_allowed=False,
)
ARGS = {"question": "How much of the bus traffic is heartbeats?", "why": "I keep guessing at it in chat"}


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


SCHEDULE = {
    "available": True, "tz": "America/Denver",
    "lines": {"investigate": {"enabled": True, "runs_today": 7, "daily_cap": 7, "next_eligible_at": None}},
}


def _listener(*, table=None, enabled=True, pool="default", schedule=SCHEDULE):
    table = table if table is not None else FocusTable()
    the_pool = FocusPool(table) if pool == "default" else pool

    async def _schedule():
        if isinstance(schedule, Exception):
            raise schedule
        return schedule

    listener = fl.CuriosityFocusListener(
        pool_provider=lambda: the_pool, source_ref=ServiceRef(name="orion-hub"), enabled=enabled,
        schedule_provider=_schedule,
    )
    listener.bus = Bus()
    return listener, table


def _ask(listener, args, *, binding=CHAT, reply_to=None, operation="queue_curiosity_focus", kind=REQUEST_KIND):
    corr = uuid4()
    env = BaseEnvelope(
        kind=kind, correlation_id=corr, source=ServiceRef(name="orion-harness-governor"),
        reply_to=reply_to if reply_to is not None else f"{RESULT_PREFIX}{corr}",
        payload={"operation": operation, "binding": binding.model_dump(mode="json"), "args": args},
    )
    before = len(listener.bus.published)
    asyncio.run(listener.handle(env))
    if len(listener.bus.published) == before:
        return None
    channel, out = listener.bus.published[-1]
    assert channel == f"{RESULT_PREFIX}{corr}" and out.kind == RESULT_KIND
    return IntrospectResultV1.model_validate(out.payload)


# --- the responder (producer) --------------------------------------------------------------------


def test_a_chat_turn_queues_one_pending_focus_and_says_when_the_run_can_start(caplog):
    listener, table = _listener()
    with caplog.at_level(logging.INFO, logger="orion-hub.curiosity_focus"):
        out = _ask(listener, ARGS)
    assert out.ok is True and out.operation == "queue_curiosity_focus"
    [item] = out.items
    assert item.kind == "curiosity_focus" and item.text == ARGS["question"]
    assert item.extra["status"] == "pending" and item.extra["replaced"] is None
    assert item.extra["chat_correlation_id"] == "chat-corr-1" and item.extra["origin"] == "self"
    assert "used up (7 of 7)" in item.extra["next_run"]["words"]
    assert "nothing starts now" in item.extra["what_happens_next"]
    [row] = table.rows
    assert str(row["focus_id"]) == item.id and row["status"] == "pending"
    # The trace line the design names, with the chat correlation id.
    assert any(f"curiosity_focus_queued focus_id={item.id}" in r.message and "chat_corr=chat-corr-1" in r.message
               for r in caplog.records)


def test_only_a_chat_turn_queues():
    listener, table = _listener()
    out = _ask(listener, ARGS, binding=CHAT.model_copy(update={"invocation_context": "curiosity"}))
    assert out.ok is False and "conversation with Juniper" in out.error and table.rows == []


def test_kill_switch_off_refuses_and_writes_nothing():
    listener, table = _listener(enabled=False)
    out = _ask(listener, ARGS)
    assert out.ok is False and "switched off" in out.error and table.rows == [] and table.statements == []


def test_the_same_call_twice_is_one_row_and_the_same_id():
    listener, table = _listener()
    first, again = _ask(listener, ARGS), _ask(listener, ARGS)
    assert first.items[0].id == again.items[0].id and len(table.rows) == 1
    assert again.items[0].extra["already_queued"] is True


def test_a_new_focus_replaces_the_pending_one_and_the_reply_names_it():
    listener, table = _listener()
    first = _ask(listener, ARGS)
    second = _ask(listener, {"question": "What is the GPU idle floor?", "why": "the fans"})
    assert second.items[0].extra["replaced"] == {
        "focus_id": first.items[0].id, "question": ARGS["question"], "origin": "self"}
    assert [r["status"] for r in table.rows] == ["replaced", "pending"]


def test_orion_cannot_replace_a_focus_juniper_asked_for():
    listener, table = _listener()
    hers = _ask(listener, {**ARGS, "origin": "juniper"})
    mine = _ask(listener, {"question": "my own idea", "why": "curious"})
    assert hers.ok and mine.ok is False
    assert cf.JUNIPER_PROTECTED in mine.error
    assert [r["status"] for r in table.rows] == ["pending"] and table.rows[0]["origin"] == "juniper"
    # Juniper's new ask replaces hers.
    again = _ask(listener, {"question": "the fans instead", "why": "she said", "origin": "juniper"})
    assert again.ok and again.items[0].extra["replaced"]["focus_id"] == hers.items[0].id


def test_invalid_args_are_refused_before_any_write():
    listener, table = _listener()
    for bad in ({**ARGS, "priority": "urgent"}, {**ARGS, "question": "   "}, {**ARGS, "origin": "hub"},
                {**ARGS, "why": "x" * 501}):
        out = _ask(listener, bad)
        assert out.ok is False and out.error.startswith("invalid queue_curiosity_focus request"), bad
    assert table.statements == []


def test_db_failure_is_unknown_never_queued():
    class _Boom(FocusPool):
        def acquire(self):
            raise RuntimeError("password=hunter2")

    listener, _ = _listener(pool=_Boom(None))
    out = _ask(listener, ARGS)
    assert out.ok is False and out.error == fl.UNAVAILABLE and "hunter2" not in out.error
    listener2, _ = _listener(pool=None)
    assert _ask(listener2, ARGS).error == fl.UNAVAILABLE


def test_an_unreadable_schedule_still_queues_and_says_the_time_is_unknown():
    listener, table = _listener(schedule=RuntimeError("redis down"))
    out = _ask(listener, ARGS)
    assert out.ok and out.items[0].extra["next_run"]["known"] is False and len(table.rows) == 1


def test_wrong_reply_channel_or_kind_is_ignored():
    listener, table = _listener()
    assert _ask(listener, ARGS, reply_to="orion:somewhere:else") is None
    assert _ask(listener, ARGS, kind="introspect.tool.result.v1") is None
    assert _ask(listener, ARGS, operation="curiosity").ok is False
    assert table.rows == []


def test_next_run_words():
    assert fl.next_run_words(None)["known"] is False
    off = {"available": True, "lines": {"investigate": {"enabled": False}}}
    assert "switched off" in fl.next_run_words(off)["words"]
    cool = {"available": True, "lines": {"investigate": {
        "enabled": True, "runs_today": 2, "daily_cap": 7, "next_eligible_at": "2026-10-17T06:40:00+00:00"}}}
    assert "no earlier than 2026-10-17T06:40:00+00:00" in fl.next_run_words(cool)["words"]


def test_the_responder_never_touches_the_graph():
    src = (HUB / "scripts" / "curiosity_focus_listener.py").read_text()
    for forbidden in ("GRAPH.QUERY", "execute_command", "MERGE", "reader"):
        assert forbidden not in src, forbidden


# --- the investigate loop (consumer) -------------------------------------------------------------


def _loop(monkeypatch, *, enabled=True, text="found it", graph_down=False, answers=None, **over):
    from test_curiosity_investigation import _FakeBus, _FakeReader, _graph_loop

    table = FocusTable()
    reader = _FakeReader(raises=True) if graph_down else _FakeReader(answers=answers or {})
    loop = _graph_loop(_FakeBus(), reader=reader, text=text, focus_enabled=enabled, **over)
    pool = FocusPool(table)
    g = type(loop)._take_focus.__globals__
    for name, real in (("take_focus", cf.take_focus), ("release_focus", cf.release_focus),
                       ("stamp_focus_outcome", cf.stamp_focus_outcome), ("fail_focus", cf.fail_focus),
                       ("read_focus_for_runs", cf.read_for_runs)):
        monkeypatch.setitem(g, name, (lambda real: lambda _pool, *a, **k: real(pool, *a, **k))(real))
    return loop, table, pool


def _queue(pool, question="How much of the bus is heartbeats?", origin="self"):
    return asyncio.run(cf.queue_focus(pool, question=question, why="I keep guessing", origin=origin,
                                      chat_correlation_id="chat-corr-1")).focus


def test_the_next_run_takes_the_focus_shows_it_first_and_recalls_it(monkeypatch, caplog):
    loop, table, pool = _loop(monkeypatch)
    focus = _queue(pool)
    with caplog.at_level(logging.INFO):
        asyncio.run(loop.tick())
    prompt = loop.seen_prompt
    assert cf.FOCUS_HEADER in prompt and focus.question in prompt
    assert prompt.index(cf.FOCUS_HEADER) < prompt.index("LAST, IF YOU WANT EITHER OF THESE")
    assert 'focus_disposition: "explored"' in prompt
    [row] = table.rows
    assert row["status"] in ("taken", "ran") and row["offered_run_id"] and row["offered_run_id"] in prompt
    assert loop.seen_generate_kwargs.get("retrieval_query") == focus.question
    assert any(f"curiosity_focus_offered run={row['offered_run_id']} focus_id={focus.focus_id}" in r.message
               for r in caplog.records)


def test_turn_end_stamps_ran_with_the_disposition_the_run_wrote(monkeypatch, caplog):
    loop, table, pool = _loop(monkeypatch, answers={
        "focus_disposition IS NOT NULL": [{"focus_disposition": "set_aside: the logs were empty"}]})
    _queue(pool)
    with caplog.at_level(logging.INFO):
        asyncio.run(loop.tick())
    [row] = table.rows
    assert row["status"] == "ran" and row["turn_ok"] is True
    assert row["disposition"] == "set_aside:the logs were empty"
    assert any("curiosity_focus_outcome run=" in r.message and "turn_ok=True" in r.message for r in caplog.records)


def test_a_run_that_wrote_no_disposition_reads_unknown(monkeypatch):
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)
    asyncio.run(loop.tick())
    [row] = table.rows
    assert row["status"] == "ran" and row["disposition"] is None
    assert "unknown" in cf.disposition_words(cf._row({**row, "effective_status": row["status"]}))


def test_an_empty_turn_fails_the_focus_and_never_requeues_it(monkeypatch):
    loop, table, pool = _loop(monkeypatch, text="")
    _queue(pool)
    asyncio.run(loop.tick())
    [row] = table.rows
    assert row["status"] == "failed" and row["turn_ok"] is False
    assert asyncio.run(cf.take_focus(pool, run_id="b2b2b2b2b2b2")) is None


def test_kill_switch_off_takes_nothing(monkeypatch):
    loop, table, pool = _loop(monkeypatch, enabled=False)
    _queue(pool)
    asyncio.run(loop.tick())
    assert cf.FOCUS_HEADER not in loop.seen_prompt and table.rows[0]["status"] == "pending"


def test_an_unreadable_graph_takes_nothing(monkeypatch):
    """No graph write, no disposition: the focus waits for a run that can answer it."""
    loop, table, pool = _loop(monkeypatch, graph_down=True)
    _queue(pool)
    asyncio.run(loop.tick())
    assert cf.FOCUS_HEADER not in loop.seen_prompt and table.rows[0]["status"] == "pending"


def test_a_pending_focus_is_no_gate_bypass(monkeypatch):
    """The cooldown blocks the second tick exactly as without a focus, and the focus waits."""
    loop, table, pool = _loop(monkeypatch)
    assert asyncio.run(loop.tick()) is None  # spends the cooldown
    _queue(pool)
    seen = len(table.statements)
    assert asyncio.run(loop.tick()) == "cooldown"
    assert table.rows[0]["status"] == "pending"
    assert cf.TAKE_SQL not in table.statements[seen:]


def test_a_run_cancelled_before_its_turn_gives_the_focus_back(monkeypatch):
    loop, table, pool = _loop(monkeypatch)
    loop.kickoff_via_cortex = True
    _queue(pool)

    async def _cancelled(**k):
        raise asyncio.CancelledError()

    loop._dispatch_durable_run = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.tick())
    assert cf.TAKE_SQL in table.statements and cf.RELEASE_SQL in table.statements
    assert table.rows[0]["status"] == "pending" and table.rows[0]["offered_run_id"] is None


def test_a_turn_cancelled_after_it_started_fails_the_focus_never_releases_it(monkeypatch):
    """Orion may have seen it, so it is never re-offered; and no durable run will settle it, so it
    is failed now rather than reading "taken" forever (review finding, 2026-10-16)."""
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)

    async def _cancelled(*a, **k):
        raise asyncio.CancelledError()

    loop._generate = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.tick())
    assert cf.RELEASE_SQL not in table.statements and table.rows[0]["status"] == "failed"


def test_a_durable_run_that_ended_before_it_started_gives_the_focus_back(monkeypatch):
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id="c3c3c3c3c3c3"))

    async def _not_started(run_id):
        return False

    loop._durable_run_started = _not_started  # type: ignore[assignment]
    asyncio.run(loop._settle_focus_after_run_end("c3c3c3c3c3c3"))
    assert table.rows[0]["status"] == "pending"


def test_a_durable_run_that_failed_after_it_started_fails_the_focus(monkeypatch):
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id="c3c3c3c3c3c3"))

    async def _started(run_id):
        return True

    loop._durable_run_started = _started  # type: ignore[assignment]
    asyncio.run(loop._settle_focus_after_run_end("c3c3c3c3c3c3"))
    assert table.rows[0]["status"] == "failed"


def test_a_durable_run_in_doubt_keeps_the_claim(monkeypatch):
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id="c3c3c3c3c3c3"))

    async def _unknown(run_id):
        return None

    loop._durable_run_started = _unknown  # type: ignore[assignment]
    asyncio.run(loop._settle_focus_after_run_end("c3c3c3c3c3c3"))
    assert table.rows[0]["status"] == "taken"


def test_only_investigate_takes_and_both_turn_end_paths_stamp():
    """Static: the focus is taken only by `_investigate` (after every gate in `tick`), every path
    that ends a measured turn stamps it, and a failed durable run settles it."""
    tree = ast.parse((HUB / "scripts" / "curiosity_investigation.py").read_text())
    takers: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name == "_take_focus":
                    takers.add(fn.name)
    assert takers == {"_investigate"}
    src = (HUB / "scripts" / "curiosity_investigation.py").read_text()
    assert "await self._settle_focus_after_run_end(state.run_id)" in src


def _turn_request(run_id, *, corr="turn-1", attempt=1, tag=None):
    from orion.schemas.durable_run import CuriosityTurnRequestV1

    extra = {"source_tag": tag} if tag else {}
    return CuriosityTurnRequestV1(run_id=run_id, correlation_id=corr, prompt="p", timeout_sec=60.0,
                                  attempt=attempt, **extra)


def _scripted(loop, outcomes):
    """Each _generate call takes the next outcome: text, or an exception to raise."""
    calls = iter(outcomes)

    async def _gen(*a, **k):
        out = next(calls)
        if isinstance(out, BaseException):
            raise out
        return out, {}

    loop._generate = _gen  # type: ignore[assignment]


def test_durable_turns_stamp_failed_then_ran_and_a_cache_hit_does_not_restamp(monkeypatch):
    run_id = "d4d4d4d4d4d4"
    loop, table, pool = _loop(monkeypatch, answers={
        "focus_disposition IS NOT NULL": [{"focus_disposition": "explored"}]})
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id=run_id))
    _scripted(loop, ["", "found it"])
    first = asyncio.run(loop._turn_result_for(_turn_request(run_id), hold_lock=False))
    assert first.ok is False and table.rows[0]["status"] == "failed"
    second = asyncio.run(loop._turn_result_for(_turn_request(run_id, attempt=2), hold_lock=False))
    assert second.ok and table.rows[0]["status"] == "ran" and table.rows[0]["disposition"] == "explored"
    stamped = sum(1 for st in table.statements if st == cf.STAMP_SQL)
    asyncio.run(loop._turn_result_for(_turn_request(run_id, attempt=3), hold_lock=False))  # cache hit
    assert sum(1 for st in table.statements if st == cf.STAMP_SQL) == stamped


def test_a_raised_turn_fails_the_focus_and_another_line_never_stamps(monkeypatch):
    run_id = "e5e5e5e5e5e5"
    loop, table, pool = _loop(monkeypatch)
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id=run_id))
    _scripted(loop, ["found it", RuntimeError("harness down")])
    asyncio.run(loop._turn_result_for(_turn_request(run_id, tag="curiosity_self_inquiry"), hold_lock=False))
    assert table.rows[0]["status"] == "taken" and cf.STAMP_SQL not in table.statements
    out = asyncio.run(loop._turn_result_for(_turn_request(run_id, corr="turn-2"), hold_lock=False))
    assert out.ok is False and table.rows[0]["status"] == "failed"


def test_a_completed_durable_run_stamps_a_focus_its_turn_end_missed(monkeypatch):
    loop, table, pool = _loop(monkeypatch, answers={
        "focus_disposition IS NOT NULL": [{"focus_disposition": "explored"}]})
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id="f6f6f6f6f6f6"))
    asyncio.run(loop._settle_focus_after_run_end("f6f6f6f6f6f6", completed=True))
    assert table.rows[0]["status"] == "ran" and table.rows[0]["disposition"] == "explored"


# --- the read tool (curiosity kind=focus) --------------------------------------------------------


def test_curiosity_kind_focus_reads_the_story_back():
    from scripts.curiosity_introspect_listener import CuriosityIntrospectListener

    table = FocusTable()
    pool = FocusPool(table)
    old = _queue(pool, "first question")
    new = _queue(pool, "second question")
    asyncio.run(cf.take_focus(pool, run_id="a1a1a1a1a1a1"))
    asyncio.run(cf.stamp_focus_outcome(pool, run_id="a1a1a1a1a1a1", turn_ok=True, disposition=None,
                                       disposition_raw="partly"))
    listener = CuriosityIntrospectListener(pool_provider=lambda: pool, reader_provider=lambda: None,
                                           source_ref=ServiceRef(name="orion-hub"), search=None)
    listener.bus = Bus()
    corr = uuid4()
    asyncio.run(listener.handle(BaseEnvelope(
        kind=REQUEST_KIND, correlation_id=corr, source=ServiceRef(name="orion-harness-governor"),
        reply_to=f"{RESULT_PREFIX}{corr}",
        payload={"operation": "curiosity", "binding": CHAT.model_dump(mode="json"), "args": {"kind": "focus"}},
    )))
    out = IntrospectResultV1.model_validate(listener.bus.published[-1][1].payload)
    assert out.ok and out.total_available == 2
    by_id = {i.id: i for i in out.items}
    assert by_id[new.focus_id].extra["status"] == "ran"
    story = by_id[new.focus_id].extra["story"]
    assert "taken by run a1a1a1a1a1a1" in story and "unknown (the run wrote 'partly'" in story
    assert by_id[old.focus_id].extra["status"] == "replaced"
    assert by_id[new.focus_id].extra["disposition"] is None


def test_curiosity_kind_focus_takes_only_limit_and_since():
    from orion.schemas.introspect import CuriosityArguments

    with pytest.raises(ValueError):
        CuriosityArguments.model_validate({"kind": "focus", "query": "bus"})
    assert CuriosityArguments.model_validate({"kind": "focus", "limit": 2}).kind == "focus"


# --- the Curiosity tab -----------------------------------------------------------------------------


def test_schedule_carries_the_pending_focus_and_unreadable_is_not_none(monkeypatch):
    import scripts.curiosity_routes as cr

    table = FocusTable()
    pool = FocusPool(table)
    monkeypatch.setattr(cr, "_get_memory_pg_pool", lambda: pool)
    assert asyncio.run(cr._read_pending_focus())["focus"] is None
    focus = _queue(pool)
    got = asyncio.run(cr._read_pending_focus())
    assert got["available"] is True and got["focus"]["focus_id"] == focus.focus_id
    assert got["focus"]["question"] == focus.question
    table.fail_with = RuntimeError("relation does not exist")
    assert asyncio.run(cr._read_pending_focus()) == {"available": False, "focus": None, "enabled": got["enabled"]}


def test_runs_that_took_a_focus_are_marked(monkeypatch):
    import scripts.curiosity_routes as cr

    pool = FocusPool(FocusTable())
    monkeypatch.setattr(cr, "_get_memory_pg_pool", lambda: pool)
    _queue(pool)
    asyncio.run(cf.take_focus(pool, run_id="a1a1a1a1a1a1"))
    runs = [{"run_id": "a1a1a1a1a1a1"}, {"run_id": "b2b2b2b2b2b2"}]
    asyncio.run(cr._attach_run_focus(runs))
    assert runs[0]["focus"]["status"] == "taken" and "focus" not in runs[1]


def test_clear_route_clears_only_a_pending_focus(monkeypatch):
    import scripts.curiosity_routes as cr

    pool = FocusPool(FocusTable())
    monkeypatch.setattr(cr, "_get_memory_pg_pool", lambda: pool)
    focus = _queue(pool)
    assert asyncio.run(cr.curiosity_focus_clear("not-a-uuid")).status_code == 400
    ok = asyncio.run(cr.curiosity_focus_clear(focus.focus_id))
    assert ok.status_code == 200
    again = asyncio.run(cr.curiosity_focus_clear(focus.focus_id))
    assert again.status_code == 409
    assert asyncio.run(cf.read_pending(pool)) is None
    monkeypatch.setattr(cr, "_get_memory_pg_pool", lambda: None)
    assert asyncio.run(cr.curiosity_focus_clear(str(UUID(int=1)))).status_code == 503


def test_the_listener_is_wired_with_the_kill_switch():
    src = (HUB / "scripts" / "main.py").read_text()
    assert "CuriosityFocusListener(" in src and "enabled=settings.HUB_CURIOSITY_FOCUS_ENABLED" in src
    assert "focus_enabled=settings.HUB_CURIOSITY_FOCUS_ENABLED" in src
    assert "await curiosity_focus_listener.stop()" in src


# --- the MCP result budget (review finding, 2026-10-16) -------------------------------------------

NASTY = '"\\\n<>&\u2028'
MCP_TOOL_RESULT_MAX_CHARS = 12000


def _worst_focus(n):
    from datetime import datetime, timezone

    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return cf.FocusRow(
        focus_id=str(UUID(int=n)), question=(NASTY * 200)[:500], why=(NASTY * 200)[:500], origin="juniper",
        status="ran", created_at=now, expires_at=now, status_at=now, chat_correlation_id="c" * 300,
        replaced_by=str(UUID(int=n + 1)), offered_at=now, offered_run_id="a" * 32, outcome_stamped_at=now,
        turn_ok=True, disposition="set_aside:" + (NASTY * 100)[:300], disposition_raw=(NASTY * 100)[:300],
    )


def _size(result):
    """The text orion/introspect/mcp_server.py sends: json.dumps(result, default=str,
    ensure_ascii=False). Pinned by the source check below."""
    import json

    return len(json.dumps(result.model_dump(mode="json"), default=str, ensure_ascii=False))


def test_the_budget_measure_matches_the_mcp_server():
    src = (HUB.parents[1] / "orion" / "introspect" / "mcp_server.py").read_text()
    assert "json.dumps(result, default=str, ensure_ascii=False)" in src


def test_five_worst_case_focus_items_stay_under_the_mcp_cap():
    from datetime import datetime, timezone

    from scripts.curiosity_introspect_listener import focus_item

    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    items = [focus_item(_worst_focus(n), now) for n in range(5)]
    result = IntrospectResultV1(ok=True, operation="curiosity", as_of=now, total_available=99999, items=items)
    assert _size(result) < MCP_TOOL_RESULT_MAX_CHARS, _size(result)
    assert all("set aside" in i.extra["story"] for i in items)


def test_five_worst_case_run_items_with_a_focus_stay_under_the_mcp_cap():
    from test_curiosity_introspect import NOW, _worst_outcome, _worst_story

    from scripts.curiosity_introspect import run_item
    from scripts.curiosity_introspect_listener import run_focus_extra

    for with_write_up in (True, False):
        items = [run_item(_worst_story(n, body=with_write_up), _worst_outcome(),
                          extra={"similarity": 0.987654321, "graph_read": False,
                                 "focus": run_focus_extra(_worst_focus(n))})
                 for n in range(5)]
        result = IntrospectResultV1(ok=True, operation="curiosity", as_of=NOW, total_available=99999, items=items)
        assert _size(result) < MCP_TOOL_RESULT_MAX_CHARS, (_size(result), with_write_up)


# --- Orion Prime 2A: the resident loop in orion-mind -----------------------------------------------

PRIME = IntrospectToolBindingV1(
    invocation_context="prime", parent_run_id="t1", parent_trace_id="prime:t1", memory_allowed=False,
)


def test_prime_queues_into_an_empty_slot_and_the_row_is_marked_prime():
    listener, table = _listener()
    out = _ask(listener, ARGS, binding=PRIME)
    assert out.ok is True
    [row] = table.rows
    assert row["status"] == "pending" and row["origin"] == "self" and row["chat_correlation_id"] == "prime:t1"
    # Prime can compute the id before Hub answers (orion/prime/acts.py prime_focus_id).
    from orion.prime.acts import prime_focus_id
    assert out.items[0].id == prime_focus_id(question=ARGS["question"], why=ARGS["why"], thought_id="t1")


def test_prime_never_replaces_a_pending_focus():
    listener, table = _listener()
    chat = _ask(listener, ARGS)
    mine = _ask(listener, {"question": "prime's own", "why": "a concern"}, binding=PRIME)
    assert chat.ok and mine.ok is False and cf.PENDING_KEPT in mine.error
    assert [r["status"] for r in table.rows] == ["pending"]


def test_prime_can_never_claim_juniper_asked():
    listener, table = _listener()
    out = _ask(listener, {**ARGS, "origin": "juniper"}, binding=PRIME)
    assert out.ok is False and "never juniper" in out.error and table.rows == []
