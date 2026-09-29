"""Dream responder: trusted reply path, modes, read-only, empty vs unknown."""
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from app import introspect_listener as il
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.semantic_index import IndexPass, SearchConfig
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
            return _Result([{**r, "total": len(self.rows)} for r in self.rows])
        return _Result([])


class FakeEngine:
    def __init__(self, rows=(), fail=False):
        self.conn, self.fail = FakeConn(list(rows)), fail

    @contextmanager
    def connect(self):
        if self.fail:
            raise ConnectionError("db down")
        yield self.conn


class Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(engine=None, search=SEARCH):
    listener = il.DreamIntrospectListener(
        bus_url="redis://x", engine_provider=lambda: engine or FakeEngine([NARR]),
        source=ServiceRef(name="orion-dream"), search=search,
    )
    listener.bus = Bus()
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


def test_search_not_configured_is_unknown():
    off = SearchConfig(chroma_url="", embed_url="", collection="orion_dreams", min_similarity=0.6)
    [(_, reply)] = _handle(_listener(search=off), _envelope({"query": "vision"}))
    assert IntrospectResultV1.model_validate(reply.payload).error == il.SEARCH_UNAVAILABLE


def test_search_with_no_hits_is_empty_and_skips_postgres(monkeypatch):
    async def no_hits(client, cfg, query):
        return []
    monkeypatch.setattr(il, "rank", no_hits)
    engine = FakeEngine([NARR])
    [(_, reply)] = _handle(_listener(engine), _envelope({"query": "vision"}))
    result = IntrospectResultV1.model_validate(reply.payload)
    assert result.ok and result.items == [] and result.total_available == 0
    assert engine.conn.calls == []


def test_search_hits_are_regated_with_similarity(monkeypatch):
    async def hits(client, cfg, query):
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
