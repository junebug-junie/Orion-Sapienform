"""Hub record_correction responder, and corrections inside the curiosity loop and the email.

The fake connection answers the letter read with a real ``OrionDayLetterV1`` (the fixture
day's reread letter) and every correction statement with the in-memory table in
``orion.orion_day.tests.fixtures``; the graph is ``FakePriorReader``. So the responder runs the
real ref parsing, part lookup, prior-id read and insert underneath.
"""
from __future__ import annotations

import ast
import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import scripts.orion_day_correction_listener as cl
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.curiosity.value import PriorState
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.orion_day import corrections as oc
from orion.orion_day import store
from orion.orion_day.tests import fixtures as fx
from orion.schemas.introspect import IntrospectResultV1, IntrospectToolBindingV1

HUB = Path(__file__).resolve().parents[1]
CHAT = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="chat-run", parent_trace_id="chat-corr-1", memory_allowed=False,
)
DAY = fx.LETTER_DATE.isoformat()
ARGS = {
    "ref": f"{DAY} ¶4",
    "claim": "I said it dropped 4,059 requests",
    "correction": "It dropped 12; 4,059 was the queue's lifetime total.",
    "evidence": ["Juniper said so in this conversation", "chat_compactor:d5f4c123"],
    "prior_ids": [fx.CORRECTION_PRIOR],
}


class _Conn(fx.CorrectionTable):
    def __init__(self, letters):
        super().__init__()
        self.letters = letters

    async def fetchrow(self, sql, *args):
        self.statements.append(sql)
        assert sql == store.SELECT_LETTER_SQL, sql[:60]
        letter = self.letters.get(args[0])
        return (letter.model_dump(mode="json") | {"letter_date": letter.letter_date}) if letter else None


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(*, reader=None, letters=None, pool="default"):
    letter = fx.reread_letter()
    conn = _Conn({letter.letter_date: letter} if letters is None else letters)
    the_pool = fx.TablePool(conn) if pool == "default" else pool
    reader = reader if reader is not None else fx.FakePriorReader()
    listener = cl.OrionDayCorrectionListener(
        pool_provider=lambda: the_pool, reader_provider=lambda: reader, source_ref=ServiceRef(name="orion-hub"),
    )
    listener.bus = Bus()
    return listener, conn


def _ask(listener, args, *, binding=CHAT, reply_to=None, operation="record_correction", kind=REQUEST_KIND):
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


# --- the responder ------------------------------------------------------------------------------


def test_a_valid_correction_is_one_row_and_names_its_id(caplog):
    listener, conn = _listener()
    with caplog.at_level(logging.INFO, logger="orion-hub.orion_day_correction"):
        out = _ask(listener, ARGS)
    assert out.ok and out.operation == "record_correction" and out.total_available == 1
    [item] = out.items
    [row] = conn.rows
    assert item.id == str(row["correction_id"]) == item.extra["correction_id"]
    assert UUID(item.id) == oc.correction_id_for(ARGS["ref"], ARGS["claim"], ARGS["correction"], "chat-corr-1")
    assert (row["letter_date"], row["ref"], row["claim"], row["correction"]) == (
        fx.LETTER_DATE, ARGS["ref"], ARGS["claim"], ARGS["correction"])
    assert row["prior_ids"] == [fx.CORRECTION_PRIOR] and row["chat_correlation_id"] == "chat-corr-1"
    assert row["offered_at"] is None and row["priors_retested"] is None
    # Evidence: free text kept as text, a record ref of that day marked as one.
    import json
    assert json.loads(row["evidence"]) == [
        {"text": "Juniper said so in this conversation", "in_letter_records": False},
        {"text": "chat_compactor:d5f4c123", "in_letter_records": True},
    ]
    assert item.extra["claim_quoted"] is True  # ¶4's own words
    assert item.extra["already_recorded"] is False
    assert f"correction_id={item.id} ref={ARGS['ref']} priors=1" in caplog.text
    assert "orion_day_correction_recorded corr=" in caplog.text


def test_quoting_ignores_case_and_spacing_and_a_paraphrase_is_marked():
    listener, _ = _listener()
    assert _ask(listener, {**ARGS, "claim": "i said it  dropped 4,059\nrequests"}).items[0].extra["claim_quoted"] is True
    assert _ask(listener, {**ARGS, "claim": "it lost about four thousand"}).items[0].extra["claim_quoted"] is False


def test_the_same_correction_twice_is_one_row():
    listener, conn = _listener()
    first = _ask(listener, ARGS)
    second = _ask(listener, ARGS)
    assert len(conn.rows) == 1
    assert first.items[0].id == second.items[0].id and second.items[0].extra["already_recorded"] is True


def test_a_correction_without_priors_needs_no_graph():
    listener, conn = _listener(reader=fx.FakePriorReader(raises=True))
    out = _ask(listener, {**ARGS, "prior_ids": []})
    assert out.ok and conn.rows[0]["prior_ids"] == []


def test_a_carry_item_ref_resolves():
    listener, conn = _listener()
    assert _ask(listener, {**ARGS, "ref": f"{DAY} carry 1"}).ok
    assert conn.rows[0]["ref"] == f"{DAY} carry 1"


@pytest.mark.parametrize("args, words", [
    ({"ref": "2026-01-01 ¶1"}, "unknown letter 2026-01-01"),
    ({"ref": f"{DAY} ¶40"}, f"unknown part {DAY} ¶40: that letter has 5 numbered note paragraphs"),
    ({"ref": f"{DAY} carry 9"}, "that letter has 3 numbered carry items"),
    ({"prior_ids": [fx.CORRECTION_PRIOR, "self:never_held"]}, "unknown prior_id self:never_held"),
])
def test_unresolvable_ref_or_unknown_prior_is_a_refusal_and_nothing_is_written(args, words):
    listener, conn = _listener()
    out = _ask(listener, {**ARGS, **args})
    assert out.ok is False and words in out.error and out.items == []
    assert conn.rows == [] and oc.INSERT_CORRECTION_SQL not in conn.statements


def test_an_unreadable_graph_refuses_rather_than_skipping_the_check():
    listener, conn = _listener(reader=fx.FakePriorReader(raises=True))
    out = _ask(listener, ARGS)
    assert out.ok is False and out.error == cl.WORLDVIEW_UNREADABLE and conn.rows == []
    listener2, conn2 = _listener()
    listener2.reader_provider = lambda: None  # graph half not configured
    assert _ask(listener2, ARGS).error == cl.WORLDVIEW_UNREADABLE and conn2.rows == []


def test_an_impossible_date_is_a_refusal_not_an_outage():
    listener, conn = _listener()
    out = _ask(listener, {**ARGS, "ref": "2026-13-45 ¶1"})
    assert out.ok is False and "names no real date" in out.error and conn.rows == []


def test_an_oversized_graph_says_so_rather_than_unreadable():
    from orion.curiosity.worldview import LIVE_PRIORS_LIMIT

    rows = [{"prior_id": f"p{i}", "claim": "c", "confidence": 0.5} for i in range(LIVE_PRIORS_LIMIT)]
    listener, conn = _listener(reader=fx.FakePriorReader(rows))
    out = _ask(listener, {**ARGS, "prior_ids": ["p1"]})
    assert out.ok is False and out.error == cl.WORLDVIEW_TRUNCATED and conn.rows == []


def test_only_a_chat_turn_records():
    listener, conn = _listener()
    curiosity = CHAT.model_copy(update={"invocation_context": "curiosity"})
    out = _ask(listener, ARGS, binding=curiosity)
    assert out.ok is False and "conversation with Juniper" in out.error and conn.rows == []


def test_invalid_args_are_refused_before_any_read():
    listener, conn = _listener()
    out = _ask(listener, {**ARGS, "confidence": 0.1})
    assert out.ok is False and out.error.startswith("invalid record_correction request")
    assert conn.statements == []


def test_db_failure_is_unavailable_never_success():
    class _Boom(fx.TablePool):
        def acquire(self):
            raise RuntimeError("password=hunter2")

    listener, _ = _listener(pool=_Boom(None))
    out = _ask(listener, ARGS)
    assert out.ok is False and out.error == cl.UNAVAILABLE and "hunter2" not in out.error
    assert "answer unknown" in out.error  # a lost connection may follow a committed insert
    listener2, _ = _listener(pool=None)
    assert _ask(listener2, ARGS).error == cl.UNAVAILABLE


def test_wrong_reply_channel_or_kind_is_ignored():
    listener, conn = _listener()
    assert _ask(listener, ARGS, reply_to="orion:somewhere:else") is None
    assert _ask(listener, ARGS, kind="introspect.tool.result.v1") is None
    assert _ask(listener, ARGS, operation="orion_day").ok is False
    assert conn.rows == []


def test_the_responder_never_writes_the_graph():
    """Static: the module reads the worldview only through read_named_priors (GRAPH.RO_QUERY)."""
    src = (HUB / "scripts" / "orion_day_correction_listener.py").read_text()
    for forbidden in ("GRAPH.QUERY", "execute_command", ".client()", "MERGE", "SET p."):
        assert forbidden not in src, forbidden


# --- the curiosity loop -------------------------------------------------------------------------


def _curiosity_loop(monkeypatch, *, enabled=True, text="found it", graph_down=False):
    from test_curiosity_investigation import _FakeBus, _FakeReader, _graph_loop

    table = fx.CorrectionTable()
    cid = oc.correction_id_for(ARGS["ref"], ARGS["claim"], ARGS["correction"], "chat-corr-1")
    asyncio.run(table.fetchval(oc.INSERT_CORRECTION_SQL, cid, fx.LETTER_DATE, ARGS["ref"], ARGS["claim"],
                               ARGS["correction"], "[]", [fx.CORRECTION_PRIOR, fx.UNTOUCHED_PRIOR], "chat-corr-1"))
    from orion.curiosity.worldview import ALL_PRIORS_CYPHER

    reader = _FakeReader(raises=True) if graph_down else _FakeReader(answers={ALL_PRIORS_CYPHER: fx.FIXTURE_PRIORS})
    loop = _graph_loop(_FakeBus(), reader=reader, text=text, orion_day_corrections_enabled=enabled)
    # The loop's own pool serves curiosity's tables; the correction statements go to the
    # in-memory table. Patch the globals the loop's class actually runs with (conftest
    # re-imports `scripts.*` per test).
    table_pool = fx.TablePool(table)
    g = type(loop)._take_corrections.__globals__
    for name in ("take_corrections", "release_corrections", "offered_corrections_for_run", "stamp_correction_outcome"):
        real = getattr(oc, name)
        monkeypatch.setitem(g, name, (lambda real: lambda pool, *a, **k: real(table_pool, *a, **k))(real))
    return loop, table, str(cid)


def test_regular_investigation_offers_corrections_once_with_current_numbers(monkeypatch):
    loop, table, cid = _curiosity_loop(monkeypatch)
    asyncio.run(loop.tick())
    prompt = loop.seen_prompt
    assert oc.CORRECTIONS_HEADER in prompt
    assert "[confidence=0.80, tested 2x]" in prompt and f"prior_id: {fx.CORRECTION_PRIOR}" in prompt
    assert ARGS["correction"] in prompt
    [row] = table.rows
    assert row["offered_run_id"] and row["offered_run_id"] in prompt


def test_corrections_kill_switch(monkeypatch):
    loop, table, _ = _curiosity_loop(monkeypatch, enabled=False)
    asyncio.run(loop.tick())
    assert oc.CORRECTIONS_HEADER not in loop.seen_prompt and table.rows[0]["offered_at"] is None


def test_an_unreadable_graph_claims_nothing(monkeypatch):
    loop, table, _ = _curiosity_loop(monkeypatch, graph_down=True)
    asyncio.run(loop.tick())
    assert "COULD NOT BE READ" in loop.seen_prompt
    assert oc.CORRECTIONS_HEADER not in loop.seen_prompt and table.rows[0]["offered_at"] is None


def test_a_turn_cancelled_after_it_started_keeps_the_claim(monkeypatch):
    """Orion may already have the prompt (and have stamped priors): re-offering would let a
    second run re-test the same correction."""
    loop, table, _ = _curiosity_loop(monkeypatch)

    async def _cancelled(*a, **k):
        raise asyncio.CancelledError()

    loop._generate = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.tick())
    assert oc.TAKE_CORRECTIONS_SQL in table.statements and oc.RELEASE_CORRECTIONS_SQL not in table.statements
    assert table.rows[0]["offered_run_id"] is not None


def test_a_run_cancelled_before_its_turn_releases_the_corrections(monkeypatch):
    loop, table, _ = _curiosity_loop(monkeypatch)
    loop.kickoff_via_cortex = True

    async def _cancelled(**k):
        raise asyncio.CancelledError()

    loop._dispatch_durable_run = _cancelled  # type: ignore[assignment]
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(loop.tick())
    assert oc.TAKE_CORRECTIONS_SQL in table.statements and oc.RELEASE_CORRECTIONS_SQL in table.statements
    assert table.rows[0]["offered_at"] is None and table.rows[0]["offered_run_id"] is None


def test_turn_end_stamps_retested_and_leaves_the_untouched_prior_out(monkeypatch):
    loop, table, cid = _curiosity_loop(monkeypatch)
    run_id = "feedfacecafe"
    table.rows[0]["offered_at"], table.rows[0]["offered_run_id"] = table.now, run_id
    before = {r["prior_id"]: PriorState.from_json(r) for r in fx.FIXTURE_PRIORS}
    after_rows = [dict(r) for r in fx.FIXTURE_PRIORS]
    after_rows[0].update(times_tested=3, confidence=0.35, last_run_id=run_id)
    g = type(loop)._stamp_corrections.__globals__

    async def fake_snapshot(pool, rid):
        assert rid == run_id
        return True, before

    monkeypatch.setitem(g, "load_turn_snapshot", fake_snapshot)
    monkeypatch.setitem(g, "read_prior_states", lambda reader: {
        r["prior_id"]: PriorState.from_json(r) for r in after_rows})
    asyncio.run(loop._stamp_corrections(run_id))
    row = table.rows[0]
    assert row["priors_retested"] == [fx.CORRECTION_PRIOR]
    assert fx.UNTOUCHED_PRIOR not in row["priors_retested"]
    assert row["priors_moved"] == [fx.CORRECTION_PRIOR] and row["outcome_stamped_at"] is not None


def test_turn_end_without_a_graph_read_leaves_the_row_unstamped(monkeypatch):
    loop, table, _ = _curiosity_loop(monkeypatch)
    table.rows[0]["offered_at"], table.rows[0]["offered_run_id"] = table.now, "r1"
    loop._reader = None
    asyncio.run(loop._stamp_corrections("r1"))
    assert table.rows[0]["outcome_stamped_at"] is None


def test_both_turn_end_paths_stamp_and_only_investigate_takes():
    """Static: corrections are claimed only by `_investigate` (the regular line), and every
    path that scores a turn's end also stamps its corrections."""
    tree = ast.parse((HUB / "scripts" / "curiosity_investigation.py").read_text())
    takers: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name == "_take_corrections":
                    takers.add(fn.name)
    assert takers == {"_investigate"}
    src = (HUB / "scripts" / "curiosity_investigation.py").read_text()
    ended = src.count("await self._spend_turn_ended(")
    assert ended == 2 and src.count("await self._stamp_corrections(") == ended


# --- the email ----------------------------------------------------------------------------------


def test_durable_brief_omits_empty_corrections_for_an_older_durable_runs():
    from orion.orion_day.brief import build_orion_day_request
    from orion.schemas.orion_day import OrionDayRunBriefV1
    from scripts.orion_day_letter import brief_request_payload

    material = asyncio.run(fx.gather.gather_orion_day(
        fx.FakeConn(), fx.LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)))
    from orion.orion_day.budget import build_llm_view

    def payload(m):
        brief = OrionDayRunBriefV1(letter_date=m.letter_date, window_start=m.window_start,
                                   window_end=m.window_end, material=m, llm_view=build_llm_view(m))
        return brief_request_payload(build_orion_day_request(brief, attempt=1))

    empty = payload(material)
    assert "corrections" not in empty["brief"]["material"]
    with_rows = payload(asyncio.run(fx.gather.gather_orion_day(
        fx.FakeConn(corrections=fx.CORRECTIONS), fx.LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc))))
    assert with_rows["brief"]["material"]["corrections"][0]["correction_id"] == fx.CORRECTION_ID
    # What durable-runs receives still validates on the new model.
    OrionDayRunBriefV1.model_validate(empty["brief"])


def test_email_shows_corrections_from_conversations():
    from scripts import orion_day_email as email

    material = asyncio.run(fx.gather.gather_orion_day(
        fx.FakeConn(corrections=fx.CORRECTIONS), fx.LETTER_DATE, now=datetime(2026, 9, 30, 14, 30, tzinfo=timezone.utc)))
    letter = fx.reread_letter(material=material)
    html = email.render_html(letter, [])
    text = email.render_text(letter, [])
    for doc in (html, text):
        assert "Corrections from conversations" in doc
        assert "The timeout dropped 12 requests" in doc and "2026-09-28 ¶4" in doc
        assert "re-tested: self:reading_queue_drops_most_requests" in doc
    assert 'data-section="corrections"' in html
    plain = fx.reread_letter()
    assert "Corrections from conversations" not in email.render_html(plain, [])
