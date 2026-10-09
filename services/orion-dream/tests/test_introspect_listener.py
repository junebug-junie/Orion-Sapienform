"""Dream responder: trusted reply path, modes, read-only, empty vs unknown."""
import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app import introspect_listener as il
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.semantic_index import IndexPass, SearchConfig, SearchUnavailableError
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

NOW = datetime(2026, 9, 29, 6, 0, tzinfo=timezone.utc)
BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
)
SEARCH = SearchConfig(chroma_url="http://c", embed_url="http://e", collection="orion_dreams", min_similarity=0.6)
NARR = {"id": 19, "dream_date": None, "tldr": "A dream.", "narrative": "", "themes": [], "occurred_at": NOW}


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def execute(self, clause, params=None):
        sql = str(clause)
        self.calls.append(sql)
        if "FROM dreams" in sql:
            since = (params or {}).get("since")
            rows = [r for r in self.rows if since is None or r["occurred_at"] >= since]
            return _Result([{**r, "total": len(rows)} for r in rows])
        return _Result([])


class FakeEngine:
    def __init__(self, rows=(), fail=False, fail_message="db down"):
        self.conn, self.fail, self.fail_message = FakeConn(list(rows)), fail, fail_message
        self.opened = 0

    @contextmanager
    def connect(self):
        self.opened += 1
        if self.fail:
            raise ConnectionError(self.fail_message)
        yield self.conn


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(engine=None, search=SEARCH, index_complete_as_of=NOW + timedelta(minutes=1)):
    listener = il.DreamIntrospectListener(
        bus_url="redis://x", engine_provider=lambda: engine or FakeEngine([NARR]),
        source=ServiceRef(name="orion-dream"), search=search,
    )
    listener.bus = Bus()
    listener.index_complete_as_of = index_complete_as_of
    return listener


def _envelope(args, *, reply=None, kind=REQUEST_KIND):
    corr = uuid4()
    payload = IntrospectRequestV1(operation="dreams", binding=BINDING, args=args).model_dump(mode="json")
    return BaseEnvelope(
        kind=kind, correlation_id=corr, reply_to=reply or f"{RESULT_PREFIX}{corr}",
        source=ServiceRef(name="orion-harness-governor"), payload=payload,
    )


def _handle(listener, envelope):
    asyncio.run(listener.handle(envelope))
    return listener.bus.published


def test_untrusted_reply_or_kind_is_ignored():
    listener = _listener()
    assert _handle(listener, _envelope({}, reply="orion:somewhere:else")) == []
    assert _handle(listener, _envelope({}, kind="reading.tool.request.v1")) == []


def test_recent_replies_on_derived_channel_read_only():
    engine = FakeEngine([NARR])
    listener = _listener(engine)
    env = _envelope({"kind": "narrative"})
    [(channel, reply)] = _handle(listener, env)
    assert channel == f"{RESULT_PREFIX}{env.correlation_id}" and reply.kind == RESULT_KIND
    assert str(reply.correlation_id) == str(env.correlation_id)
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and [i.id for i in result.items] == ["dream:19"]
    assert engine.conn.calls[0] == "SET TRANSACTION READ ONLY"


def test_invalid_arguments_are_a_failed_result():
    [(_, reply)] = _handle(_listener(), _envelope({"arm": "dream"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error.startswith("invalid dreams request")


def test_database_failure_is_unknown_not_empty():
    [(_, reply)] = _handle(_listener(FakeEngine(fail=True)), _envelope({}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.QUERY_UNAVAILABLE


def test_database_failure_log_redacts_dsn_credentials(caplog):
    engine = FakeEngine(fail=True, fail_message="could not connect: postgresql://postgres:secret@host/db")
    with caplog.at_level(logging.WARNING, logger="orion-dream.introspect"):
        [(_, reply)] = _handle(_listener(engine), _envelope({}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.QUERY_UNAVAILABLE
    [record] = [r for r in caplog.records if r.getMessage().startswith("introspect_failed")]
    logged = record.getMessage()
    assert "secret" not in logged and "postgresql://[REDACTED]@host/db" in logged


def test_search_not_configured_is_unknown():
    off = SearchConfig(chroma_url="", embed_url="", collection="orion_dreams", min_similarity=0.6)
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine, search=off), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.SEARCH_UNAVAILABLE
    assert engine.opened == 0 and engine.conn.calls == []


def test_search_backend_failure_is_unknown(monkeypatch):
    async def down(client, cfg, query, **filters):
        raise SearchUnavailableError("chroma unreachable")
    monkeypatch.setattr(il, "rank", down)
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.SEARCH_UNAVAILABLE
    assert engine.opened == 0


def test_search_hits_then_database_failure_is_query_unknown(monkeypatch):
    async def hits(client, cfg, query, **filters):
        return [("dream:19", 0.8)]
    monkeypatch.setattr(il, "rank", hits)
    engine = FakeEngine(fail=True)
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.QUERY_UNAVAILABLE
    assert engine.opened == 1


def _no_hits(monkeypatch):
    async def no_hits(client, cfg, query, **filters):
        return []
    monkeypatch.setattr(il, "rank", no_hits)


def test_search_with_no_hits_on_a_caught_up_index_is_empty(monkeypatch):
    _no_hits(monkeypatch)
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and result.items == [] and result.total_available == 0
    assert not any("ANY(:ids)" in c for c in engine.conn.calls)


def test_empty_search_before_any_complete_index_pass_is_unknown(monkeypatch):
    _no_hits(monkeypatch)
    [(_, reply)] = _handle(_listener(index_complete_as_of=None), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.SEARCH_UNAVAILABLE


def test_empty_search_with_a_dream_newer_than_the_index_is_unknown(monkeypatch):
    """A just-offered hypothesis, or an indexer outage: "no match" would be a lie."""
    _no_hits(monkeypatch)
    listener = _listener(FakeEngine([NARR]), index_complete_as_of=NOW - timedelta(minutes=1))
    [(_, reply)] = _handle(listener, _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.SEARCH_UNAVAILABLE


def test_newer_dream_outside_since_window_does_not_block_empty(monkeypatch):
    _no_hits(monkeypatch)
    listener = _listener(FakeEngine([NARR]), index_complete_as_of=NOW - timedelta(minutes=1))
    later = (NOW + timedelta(hours=1)).isoformat()
    [(_, reply)] = _handle(listener, _envelope({"query": "vision", "since": later}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and result.items == []


def test_search_hits_are_regated_with_similarity(monkeypatch):
    async def hits(client, cfg, query, **filters):
        return [("dream:19", 0.8)]
    monkeypatch.setattr(il, "rank", hits)

    class ByIdConn(FakeConn):
        def execute(self, clause, params=None):
            sql = str(clause)
            self.calls.append(sql)
            return _Result([NARR] if "ANY(:ids)" in sql and "FROM dreams" in sql else [])

    engine = FakeEngine()
    engine.conn = ByIdConn([])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    [item] = IntrospectResultV1.model_validate(reply.payload).items
    assert item.id == "dream:19" and item.extra["similarity"] == 0.8


def test_search_passes_kind_and_since_to_rank(monkeypatch):
    seen = {}

    async def spy(client, cfg, query, **filters):
        seen.update(filters)
        return []
    monkeypatch.setattr(il, "rank", spy)
    _handle(_listener(), _envelope({"query": "vision", "kind": "narrative", "since": "2026-09-20T00:00:00Z"}))
    assert seen == {"kind": "narrative", "since": datetime(2026, 9, 20, tzinfo=timezone.utc)}


def _chroma_search(monkeypatch, indexed):
    """Real rank over a fake vector-db that filters then takes the n nearest, like Chroma."""
    import json

    import httpx

    sent = []

    def handler(request):
        if request.url.host == "e":
            body = json.loads(request.content)
            return httpx.Response(200, json={"doc_id": body["doc_id"], "embedding": [1.0, 0.0]})
        if request.url.path.endswith("/orion_dreams"):
            return httpx.Response(200, json={"id": "cid", "metadata": None})
        if request.url.path.endswith("/cid/count"):
            return httpx.Response(200, json=len(indexed))
        body = json.loads(request.content)
        sent.append(body.get("where"))
        kind = (body.get("where") or {}).get("kind")
        hits = sorted((d for d in indexed if kind in (None, d[2])), key=lambda d: d[1])[:body["n_results"]]
        return httpx.Response(200, json={"ids": [[h[0] for h in hits]], "distances": [[h[1] for h in hits]]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return sent


class ByIdConn(FakeConn):
    def execute(self, clause, params=None):
        sql = str(clause)
        self.calls.append(sql)
        if "ANY(:ids)" in sql and "FROM dreams" in sql:
            return _Result([r for r in self.rows if r["id"] in params["ids"]])
        return _Result([])


def test_kind_filtered_search_finds_narrative_behind_20_nearer_hypotheses(monkeypatch):
    indexed = [(f"dh-{n:06d}", 0.1, "hypothesis") for n in range(21)] + [("dream:19", 0.6, "narrative")]
    sent = _chroma_search(monkeypatch, indexed)
    engine = FakeEngine()
    engine.conn = ByIdConn([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision", "kind": "narrative"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and [i.id for i in result.items] == ["dream:19"]
    assert result.items[0].extra["similarity"] == 0.7
    assert sent == [{"kind": "narrative"}]


def test_kind_filtered_search_with_no_such_kind_is_empty_not_unknown(monkeypatch):
    sent = _chroma_search(monkeypatch, [(f"dh-{n:06d}", 0.1, "hypothesis") for n in range(3)])
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision", "kind": "narrative"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and result.items == [] and result.total_available == 0
    assert sent == [{"kind": "narrative"}]
    assert not any("ANY(:ids)" in c for c in engine.conn.calls)


def test_kind_filtered_search_on_empty_index_is_unknown(monkeypatch):
    _chroma_search(monkeypatch, [])
    [(_, reply)] = _handle(_listener(), _envelope({"query": "vision", "kind": "narrative"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert not result.ok and result.error == il.SEARCH_UNAVAILABLE


def test_one_mode():
    [(_, reply)] = _handle(_listener(), _envelope({"dream_id": "dream:19"}))
    assert IntrospectResultV1.model_validate(reply.payload).items[0].id == "dream:19"


def test_index_once_reads_read_only_and_hands_rows_to_indexer(monkeypatch):
    seen = {}

    async def fake_index(pairs, cfg, *, client, bus, source, batch=None):
        seen.update(pairs=pairs, cfg=cfg, bus=bus, timeout=client.timeout.read)
        return IndexPass(indexed=len(pairs), pending=0)

    monkeypatch.setattr(il, "index_missing", fake_index)
    monkeypatch.setattr(il, "HTTP_TIMEOUT_SEC", 1.25)
    engine = FakeEngine([NARR])
    listener = _listener(engine)
    result = asyncio.run(listener.index_once())
    assert result.indexed == 1 and seen["pairs"][0][0] == "narrative"
    assert seen["cfg"] is SEARCH and seen["bus"] is listener.bus
    assert seen["timeout"] == 1.25
    assert engine.conn.calls[0] == "SET TRANSACTION READ ONLY"


def test_index_complete_as_of_advances_only_when_nothing_is_pending(monkeypatch):
    passes = iter([IndexPass(indexed=10, pending=5), IndexPass(indexed=5, pending=0)])

    async def fake_index(pairs, cfg, *, client, bus, source, batch=None):
        return next(passes)

    monkeypatch.setattr(il, "index_missing", fake_index)
    listener = _listener(FakeEngine([NARR]), index_complete_as_of=None)
    asyncio.run(listener.index_once())
    assert listener.index_complete_as_of is None
    before = datetime.now(timezone.utc)
    asyncio.run(listener.index_once())
    assert listener.index_complete_as_of is not None and listener.index_complete_as_of <= before + timedelta(seconds=1)


def test_start_stop_survives_a_dead_bus_and_skips_index_without_search():
    class DeadBus:
        def __init__(self, url):
            self.closed = False

        async def connect(self):
            raise ConnectionError("bus down")

        async def close(self):
            self.closed = True

    async def cycle(search):
        listener = il.DreamIntrospectListener(
            bus_url="redis://x", engine_provider=FakeEngine, source=ServiceRef(name="orion-dream"),
            search=search, bus_factory=DeadBus,
        )
        await listener.start()
        started = (listener.task is not None, listener.index_task is not None)
        await asyncio.sleep(0.05)
        await listener.stop()
        return started, (listener.task, listener.index_task)

    off = SearchConfig(chroma_url="", embed_url="", collection="orion_dreams", min_similarity=0.6)
    assert asyncio.run(cycle(SEARCH)) == ((True, True), (None, None))
    assert asyncio.run(cycle(off)) == ((True, False), (None, None))


def test_lifespan_starts_and_stops_the_responder_only_when_enabled(monkeypatch):
    # conftest re-imports `app` per test, so patch the module main will import.
    from app import introspect_listener as live_il
    from app import main

    events = []

    class Stub:
        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")

    monkeypatch.setattr(live_il, "build_listener", Stub)
    monkeypatch.setattr(main.settings, "ORION_DREAM_CYCLE_ENABLED", False)
    monkeypatch.setattr(main.settings, "ORION_BUS_ENABLED", True)

    async def run_lifespan():
        async with main.lifespan(main.app):
            events.append("serving")

    monkeypatch.setattr(main.settings, "DREAM_INTROSPECT_ENABLED", True)
    asyncio.run(run_lifespan())
    assert events == ["start", "serving", "stop"]

    events.clear()
    monkeypatch.setattr(main.settings, "DREAM_INTROSPECT_ENABLED", False)
    asyncio.run(run_lifespan())
    assert events == ["serving"]


def test_build_listener_with_default_settings_has_search_off(monkeypatch):
    import sqlalchemy

    from app import introspect_listener as live_il
    from app import settings as settings_mod

    for key in ("DREAM_SEARCH_CHROMA_URL", "DREAM_SEARCH_EMBED_URL", "DREAM_SEARCH_COLLECTION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings_mod, "settings", settings_mod.Settings(_env_file=None))
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda *a, **k: FakeEngine())
    listener = live_il.build_listener()
    assert listener.search is not None and not listener.search.enabled
    assert listener.search.chroma_url == "" and listener.search.embed_url == ""
