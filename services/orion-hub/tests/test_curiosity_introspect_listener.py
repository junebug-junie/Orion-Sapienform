"""Hub curiosity responder: trusted reply path, modes, read-only, empty vs unknown.

The fake pool answers the run store's own SQL, so these tests run the real
`curiosity_run_store` + `run_story` join underneath the responder.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import scripts.curiosity_introspect_listener as il
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.semantic_index import IndexPass, SearchConfig
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

NOW = datetime.now(timezone.utc).replace(microsecond=0)
BINDING = IntrospectToolBindingV1(
    invocation_context="curiosity", parent_run_id="r", parent_trace_id="t", memory_allowed=False,
)
SEARCH = SearchConfig(chroma_url="http://c", embed_url="http://e", collection="orion_curiosity", min_similarity=0.65)


def _admission(run_id, workflow, at, terminal="completed", line=None):
    request = {"workflow": workflow, "brief": {"line": line}} if line else {"workflow": workflow}
    return {"run_id": run_id, "request": json.dumps(request), "created_at": at, "control": None,
            "terminal": terminal, "updated_at": at + timedelta(minutes=30)}


def _events(run_id, at, end_event="run.completed", detail=None):
    return [
        {"entry_id": f"accepted:{run_id}", "run_id": run_id, "event": "run.accepted", "generated_at": at,
         "payload": "{}"},
        {"entry_id": f"end:{run_id}", "run_id": run_id, "event": end_event,
         "generated_at": at + timedelta(minutes=30), "payload": json.dumps({"detail": detail or {}})},
    ]


def _journal(run_id, at, body):
    return {"entry_id": f"j-{run_id}", "source_ref": f"curiosity:{run_id}", "title": "Curiosity",
            "body": body, "created_at": at + timedelta(minutes=29)}


def _db():
    t = NOW - timedelta(days=1)
    return {
        "admission": [
            _admission("old1", "curiosity.investigate", t - timedelta(hours=5)),
            _admission("fail1", "curiosity.investigate", t - timedelta(hours=3), terminal="failed"),
            _admission("read1", "reading.turn", t - timedelta(hours=2)),
            _admission("live1", "curiosity.investigate", t - timedelta(hours=1), terminal=""),
            _admission("new1", "curiosity.investigate", t),
        ],
        "events": (
            _events("old1", t - timedelta(hours=5))
            + _events("fail1", t - timedelta(hours=3), "run.failed", {"error": "lane_timeout"})
            + _events("read1", t - timedelta(hours=2))
            + _events("new1", t)
        ),
        "journals": [
            _journal("old1", t - timedelta(hours=5), "Old prose about bees."),
            _journal("new1", t, "Preamble.\n\n## Answer\n\nThe gate is not algorithmic."),
        ],
        "outcomes": [{"run_id": "new1", "turn_ok": True, "n_tested": 2, "n_moved": 1, "n_formed": 0,
                      "unknown_reason": None}],
        "questions": [
            {"question_id": "lived.orion.a", "text": "Q a?", "family": "lived", "pinned": False,
             "ask_count": 1, "last_asked_at": None, "created_at": NOW - timedelta(days=2)},
            {"question_id": "lived.orion.b", "text": "Q b?", "family": "lived", "pinned": True,
             "ask_count": 0, "last_asked_at": None, "created_at": NOW - timedelta(days=9)},
        ],
    }


class _Conn:
    def __init__(self, db, fail=False):
        self.db, self.fail = db, fail
        self.calls: list[str] = []
        self.readonly_flags: list[bool] = []

    @asynccontextmanager
    async def transaction(self, readonly=False):
        self.readonly_flags.append(readonly)
        yield

    async def fetch(self, sql, *args):
        self.calls.append(sql)
        if self.fail:
            raise RuntimeError("password=hunter2 host=db")
        db = self.db
        if "FROM durable_admission_runs WHERE run_id = $1" in sql:
            return [r for r in db["admission"] if r["run_id"] == args[0]]
        if "FROM durable_admission_runs" in sql:
            return list(db["admission"])
        if "FROM durable_resource_events" in sql and "ORDER BY" in sql:
            return [e for e in db["events"] if e["run_id"] in args[0]]
        if "DISTINCT ON (j.source_ref)" in sql:
            return list(db["journals"])
        if "FROM journal_entries" in sql:
            return [j for j in db["journals"] if j["source_ref"] in args[0]]
        if "FROM curiosity_run_outcomes" in sql:
            return [o for o in db["outcomes"] if o["run_id"] in args[0]]
        if "FROM curiosity_self_questions" in sql:
            since, limit = args
            rows = [q for q in db["questions"] if since is None or q["created_at"] >= since]
            rows.sort(key=lambda q: q["created_at"], reverse=True)
            return [{**q, "total": len(rows)} for q in rows[:limit]]
        return []

    async def fetchval(self, sql, *args):
        self.calls.append(sql)
        return any(j["created_at"] >= args[0] for j in self.db["journals"])


class _Pool:
    def __init__(self, db=None, fail=False):
        self.conn = _Conn(db if db is not None else _db(), fail=fail)
        self.acquired = 0

    @asynccontextmanager
    async def acquire(self):
        self.acquired += 1
        yield self.conn


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(pool="default", search=SEARCH, index_complete_as_of=NOW + timedelta(minutes=1)):
    pool = _Pool() if pool == "default" else pool
    listener = il.CuriosityIntrospectListener(
        pool_provider=lambda: pool, reader_provider=lambda: None,
        source_ref=ServiceRef(name="orion-hub"), search=search,
    )
    listener.bus = Bus()
    listener.index_complete_as_of = index_complete_as_of
    return listener, pool


def _envelope(args, *, reply=None, kind=REQUEST_KIND, operation="curiosity"):
    corr = uuid4()
    payload = IntrospectRequestV1(operation=operation, binding=BINDING, args=args).model_dump(mode="json")
    return BaseEnvelope(
        kind=kind, correlation_id=corr, reply_to=reply or f"{RESULT_PREFIX}{corr}",
        source=ServiceRef(name="orion-harness-governor"), payload=payload,
    )


def _ask(listener, args, **kw):
    env = _envelope(args, **kw)
    asyncio.run(listener.handle(env))
    channel, out = listener.bus.published[-1]
    assert channel == f"{RESULT_PREFIX}{env.correlation_id}"
    assert out.kind == RESULT_KIND and out.correlation_id == env.correlation_id
    return IntrospectResultV1.model_validate(out.payload)


def test_only_a_trusted_reply_subject_is_answered():
    listener, _ = _listener()
    asyncio.run(listener.handle(_envelope({}, reply="orion:somewhere:else")))
    asyncio.run(listener.handle(_envelope({}, kind="introspect.tool.result.v1")))
    assert listener.bus.published == []


def test_invalid_arguments_are_unknown_not_empty():
    listener, _ = _listener()
    result = _ask(listener, {"run_id": "r1", "query": "bees"})
    assert result.ok is False and result.error.startswith("invalid curiosity request:")
    other = _ask(listener, {}, operation="dreams")
    assert other.ok is False and "not answered here" in other.error


def test_recent_returns_curiosity_runs_newest_first_with_failures_and_no_leaks(caplog):
    listener, pool = _listener()
    with caplog.at_level(logging.INFO, logger="orion-hub.curiosity_introspect"):
        result = _ask(listener, {"limit": 5})
    assert result.ok and result.operation == "curiosity"
    assert [i.id for i in result.items] == ["new1", "fail1", "old1"], "reading.turn and running runs never appear"
    assert result.total_available == 3
    new1, fail1, old1 = result.items
    assert new1.epistemic_status == "unsettled" and new1.text == "The gate is not algorithmic."
    assert (new1.extra["n_tested"], new1.extra["n_moved"]) == (2, 1)
    assert new1.extra["graph_read"] is False, "hop counts without the graph are not zeros"
    assert fail1.epistemic_status == "record" and fail1.extra["status"] == "failed"
    assert fail1.text == "World question run failed with no write-up; error: lane_timeout."
    assert old1.text == "Old prose about bees."
    assert pool.conn.readonly_flags and all(pool.conn.readonly_flags), "every read is a READ ONLY transaction"
    assert pool.acquired == len(pool.conn.readonly_flags)
    assert any(
        r.getMessage().startswith("introspect op=curiosity corr=") and "mode=recent items=3 total=3" in r.getMessage()
        for r in caplog.records
    )


def test_recent_honours_limit_since_and_line():
    listener, _ = _listener()
    limited = _ask(listener, {"limit": 1})
    assert [i.id for i in limited.items] == ["new1"] and limited.total_available == 3
    since = _ask(listener, {"since": (NOW - timedelta(days=1, hours=4)).isoformat()})
    assert [i.id for i in since.items] == ["new1", "fail1"]
    other_line = _ask(listener, {"line": "self_sense_eval"})
    assert other_line.ok and other_line.items == [] and other_line.total_available == 0


def test_one_run_returns_the_full_write_up_and_unknown_id_is_empty():
    listener, _ = _listener()
    listener.reader_provider = lambda: _GraphReader({})  # graph read and empty: "not found" is trusted
    one = _ask(listener, {"run_id": "new1"})
    assert [i.id for i in one.items] == ["new1"]
    assert one.items[0].text == "Preamble.\n\n## Answer\n\nThe gate is not algorithmic."
    missing = _ask(listener, {"run_id": "nope"})
    assert missing.ok and missing.items == [] and missing.total_available == 0
    leaked = _ask(listener, {"run_id": "read1"})
    assert leaked.ok and leaked.items == [], "a reading run is not a curiosity run"


def test_postgres_failure_or_no_pool_is_unknown_and_leaks_nothing():
    for listener, _ in (_listener(pool=_Pool(fail=True)), _listener(pool=None)):
        for args in ({}, {"run_id": "new1"}, {"kind": "self_question"}):
            result = _ask(listener, args)
            assert result.ok is False and result.error == il.QUERY_UNAVAILABLE
            assert "hunter2" not in result.error and "SELECT" not in result.error


def test_self_questions_are_open_questions_newest_first():
    listener, _ = _listener()
    result = _ask(listener, {"kind": "self_question", "limit": 1})
    assert [i.id for i in result.items] == ["lived.orion.a"] and result.total_available == 2
    assert result.items[0].kind == "self_question" and result.items[0].epistemic_status == "record"
    recent = _ask(listener, {"kind": "self_question", "since": (NOW - timedelta(days=3)).isoformat()})
    assert [i.id for i in recent.items] == ["lived.orion.a"] and recent.total_available == 1


def test_search_rereads_hits_through_the_run_join_in_ranked_order(monkeypatch):
    async def fake_rank(client, cfg, query, *, since=None, line=None):
        return [("old1", 0.81), ("read1", 0.79), ("new1", 0.70)]

    monkeypatch.setattr(il, "rank", fake_rank)
    listener, _ = _listener()
    listener.reader_provider = lambda: _GraphReader({})
    result = _ask(listener, {"query": "bees"})
    assert [i.id for i in result.items] == ["old1", "new1"], "ranked order; a non-curiosity hit is dropped"
    assert [i.extra["similarity"] for i in result.items] == [0.81, 0.7]
    assert result.total_available == 2
    filtered = _ask(listener, {"query": "bees", "since": (NOW - timedelta(days=1, hours=1)).isoformat()})
    assert [i.id for i in filtered.items] == ["new1"]


def test_empty_search_is_unknown_until_the_index_has_caught_up(monkeypatch):
    async def no_hits(client, cfg, query, *, since=None, line=None):
        return []

    monkeypatch.setattr(il, "rank", no_hits)
    behind, _ = _listener(index_complete_as_of=None)
    assert _ask(behind, {"query": "bees"}).error == il.SEARCH_UNAVAILABLE
    stale, _ = _listener(index_complete_as_of=NOW - timedelta(days=3))
    assert _ask(stale, {"query": "bees"}).error == il.SEARCH_UNAVAILABLE
    caught_up, _ = _listener()
    empty = _ask(caught_up, {"query": "bees"})
    assert empty.ok and empty.items == [] and empty.total_available == 0


def test_search_not_configured_or_failing_is_unknown(monkeypatch):
    off, _ = _listener(search=None)
    assert _ask(off, {"query": "bees"}).error == il.SEARCH_UNAVAILABLE

    async def broken(client, cfg, query, *, since=None, line=None):
        raise il.SearchUnavailableError("embedder unavailable: ConnectError")

    monkeypatch.setattr(il, "rank", broken)
    listener, _ = _listener()
    assert _ask(listener, {"query": "bees"}).error == il.SEARCH_UNAVAILABLE


def test_index_docs_keep_only_valid_runs_with_text():
    t = NOW
    docs = il.index_docs_from_rows([
        {"source_ref": "curiosity:new1", "body": "x\n## Answer\nshort answer", "created_at": t},
        {"source_ref": "curiosity:bad id", "body": "text", "created_at": t},
        {"source_ref": "curiosity:empty", "body": "   ", "created_at": t},
        {"source_ref": "reading:1", "body": "text", "created_at": t},
    ])
    assert docs == [("new1", "short answer",
                     {"occurred_at": t.isoformat(), "occurred_ts": t.timestamp(), "line": "investigate"})]


def test_index_once_marks_complete_only_when_nothing_is_pending(monkeypatch):
    passes = iter([IndexPass(indexed=10, pending=3), IndexPass(indexed=0, pending=0)])
    seen = []

    async def fake_index_docs(docs, cfg, **kw):
        seen.append((docs, kw["hash_keys"]))
        return next(passes)

    monkeypatch.setattr(il, "index_docs", fake_index_docs)
    listener, pool = _listener(index_complete_as_of=None)
    asyncio.run(listener.index_once())
    assert listener.index_complete_as_of is None
    asyncio.run(listener.index_once())
    assert listener.index_complete_as_of is not None
    assert [d[0] for d in seen[0][0]] == ["old1", "new1"] and seen[0][1] == ("occurred_ts", "line")
    assert all(pool.conn.readonly_flags)
    none_pool, _ = _listener(pool=None)
    assert asyncio.run(none_pool.index_once()) is None


# --- graph-clocked runs (review fix 1) ----------------------------------------
# 46 of 371 live write-ups (08-26..09-14) predate the admission path: their only
# start clock is a graph node, and some graph nodes carry no clock at all.

from orion.curiosity.worldview import WorldviewReader, WorldviewUnavailable  # noqa: E402


class _GraphReader(WorldviewReader):
    def __init__(self, answers=None, down=False):
        super().__init__(host="x", port=1, graph_name="g", client=object())
        self.answers, self.down = answers or {}, down

    def query(self, cypher):
        if self.down:
            raise WorldviewUnavailable("ConnectionError: graph down")
        hits = [rows for needle, rows in self.answers.items() if needle in cypher]
        return hits[0] if hits else []


def _graph_db(journal_at):
    db = _db()
    db["journals"].append(_journal("g1", journal_at - timedelta(minutes=29), "Graph-era prose."))
    return db


def _graph_listener(reader, db=None, **kw):
    listener, pool = _listener(pool=_Pool(db if db is not None else _graph_db(NOW - timedelta(hours=1))), **kw)
    listener.reader_provider = lambda: reader
    return listener, pool


CLOCKLESS_GRAPH = _GraphReader({
    "RETURN DISTINCT n.run_id": [{"run_id": "g1"}],
    "MATCH (n:InvestigationRole)": [{"run_id": "g1", "choice": "local_crawl", "why": "w", "written_at": None}],
})


def test_one_graph_run_without_a_start_clock_uses_its_write_up_clock():
    listener, _ = _graph_listener(CLOCKLESS_GRAPH)
    result = _ask(listener, {"run_id": "g1"})
    assert [i.id for i in result.items] == ["g1"], "a found run is never dropped as 'nothing matched'"
    item = result.items[0]
    assert item.occurred_at == NOW - timedelta(hours=1) and item.extra["clock_from"] == "write_up"


def test_recent_includes_a_graph_run_with_only_a_write_up_clock():
    listener, _ = _graph_listener(CLOCKLESS_GRAPH)
    result = _ask(listener, {"limit": 5})
    assert [i.id for i in result.items] == ["g1", "new1", "fail1", "old1"]
    assert result.total_available == 4


def test_graph_clock_is_used_when_the_graph_has_one():
    start = NOW - timedelta(hours=3)
    reader = _GraphReader({
        "RETURN DISTINCT n.run_id": [{"run_id": "g1"}],
        "MATCH (n:InvestigationRole)": [{"run_id": "g1", "choice": "c", "why": "w",
                                         "written_at": int(start.timestamp() * 1000)}],
    })
    listener, _ = _graph_listener(reader)
    [item] = _ask(listener, {"run_id": "g1"}).items
    assert item.occurred_at == start and "clock_from" not in item.extra


def test_graph_down_not_found_is_unknown_and_found_uses_the_write_up_clock():
    listener, _ = _graph_listener(_GraphReader(down=True))
    assert _ask(listener, {"run_id": "nope"}).error == il.QUERY_UNAVAILABLE, "graph-only runs cannot be ruled out"
    [item] = _ask(listener, {"run_id": "g1"}).items
    assert item.extra["graph_read"] is False and item.extra["clock_from"] == "write_up"


def test_a_found_run_with_no_clock_at_all_is_unknown_not_empty():
    db = _db()  # g1 exists only as a clockless graph node: no write-up to borrow a clock from
    listener, _ = _graph_listener(CLOCKLESS_GRAPH, db=db)
    assert _ask(listener, {"run_id": "g1"}).error == il.QUERY_UNAVAILABLE


def test_search_where_every_hit_was_dropped_for_a_non_filter_reason_is_unknown(monkeypatch):
    async def stale_hits(client, cfg, query, *, since=None, line=None):
        return [("gone1", 0.9), ("gone2", 0.8)]

    monkeypatch.setattr(il, "rank", stale_hits)
    listener, _ = _graph_listener(_GraphReader(down=True))
    assert _ask(listener, {"query": "bees"}).error == il.QUERY_UNAVAILABLE
    up, _ = _graph_listener(_GraphReader({}))
    assert _ask(up, {"query": "bees"}).error == il.QUERY_UNAVAILABLE


# --- line filter before the candidate cut (review fix 2) ----------------------

from orion.introspect.tests.fake_chroma import FakeChroma  # noqa: E402

LIVE_SEARCH = SearchConfig(chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
                           collection="orion_curiosity", min_similarity=0.65)


def test_index_docs_carry_the_run_line_from_postgres_signals():
    t = NOW
    docs = il.index_docs_from_rows([
        {"source_ref": "curiosity:a", "title": "Curiosity", "body": "x", "created_at": t,
         "request": json.dumps({"workflow": "curiosity.investigate", "brief": {"line": "self_inquiry"}})},
        {"source_ref": "curiosity:b", "title": "Self-inquiry", "body": "x", "created_at": t},
        {"source_ref": "curiosity:c", "title": "Curiosity", "body": "x", "created_at": t,
         "detail": json.dumps({"line": "self_inquiry"})},
        {"source_ref": "curiosity:d", "title": "Curiosity", "body": "x", "created_at": t},
    ])
    assert {d[0]: d[2]["line"] for d in docs} == {
        "a": "self_inquiry", "b": "self_inquiry", "c": "self_inquiry", "d": "investigate",
    }
    assert "line" in il._HASH_KEYS, "a relabelled doc must re-upsert"


def test_line_filter_reaches_the_index_before_the_candidate_cut():
    db = _db()
    db["admission"].append(_admission("si1", "curiosity.investigate", NOW - timedelta(hours=2), line="self_inquiry"))
    db["events"] += _events("si1", NOW - timedelta(hours=2))
    db["journals"].append(_journal("si1", NOW - timedelta(hours=2), "What I am made of."))
    chroma = FakeChroma("orion_curiosity")
    for n in range(10):
        chroma.put(f"inv{n}", {"line": "investigate", "occurred_ts": NOW.timestamp()}, score=0.9 - n * 0.01)
    chroma.put("si1", {"line": "self_inquiry", "occurred_ts": NOW.timestamp()}, score=0.7)
    listener, _ = _graph_listener(_GraphReader({}), db=db, search=LIVE_SEARCH)
    listener.client_factory = chroma.client
    result = _ask(listener, {"query": "what am I made of", "line": "self_inquiry"})
    assert [i.id for i in result.items] == ["si1"]
    assert chroma.queries[-1]["where"] == {"line": "self_inquiry"}


# --- index completeness needs stored, not published, docs (review fix 3) ------

from orion.introspect.semantic_index import INDEX_LAG_MARGIN  # noqa: E402


def test_published_but_unstored_upserts_never_mark_the_curiosity_index_complete():
    chroma = FakeChroma("orion_curiosity")
    listener, _ = _listener(search=LIVE_SEARCH, index_complete_as_of=None)
    listener.client_factory = chroma.client
    first = asyncio.run(listener.index_once())
    assert first.indexed == 2 and first.pending == 0
    assert listener.index_complete_as_of is None, "a pass that only published upserts proves nothing is stored"
    asyncio.run(listener.index_once())
    assert listener.index_complete_as_of is None, "vector-writer has not stored them yet"
    assert chroma.apply(listener.bus) == 4
    started = datetime.now(timezone.utc)
    confirmed = asyncio.run(listener.index_once())
    assert (confirmed.indexed, confirmed.pending) == (0, 0)
    as_of = listener.index_complete_as_of
    assert started - INDEX_LAG_MARGIN - timedelta(seconds=1) <= as_of <= started - INDEX_LAG_MARGIN + timedelta(seconds=1)
