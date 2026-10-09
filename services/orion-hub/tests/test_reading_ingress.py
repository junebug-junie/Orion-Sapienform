"""Real domain/codec and runtime adapters; fake SQL/DNS/bus for fast local gates."""
import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.reading import DurableReadingReceiptV1, ReadingRequestedV1, ReadingStatusArguments, ReadingStatusReceiptV1, ReadingToolBindingV1, ReadingToolRequestV1
from orion.schemas.registry import resolve
from orion.world_pulse_read.events import REQUESTED_CHANNEL, TOOL_RESULT_PREFIX
from orion.world_pulse_read.tools import ReadingTools
from orion.world_pulse_read.urls import normalize_source_url, validate_source_url
from scripts.reading_listener import ReadingListener
from test_world_pulse_read_pipeline import _FakePool
from test_world_pulse_read_queue import _FakeConn


class RpcBus:
    codec = OrionCodec()

    def __init__(self, conn):
        self.conn = conn
        self.published = []
        self.commands = []
        self.listener = ReadingListener(lambda: _FakePool(conn), ServiceRef(name="orion-hub"))
        self.listener.bus = self

    async def publish(self, channel, envelope):
        if channel == REQUESTED_CHANNEL:
            # The accepted event can only follow the durable queue insertion.
            assert any(str(r["request_id"]) == envelope.payload["request_id"] for r in self.conn.rows.values())
        self.published.append((channel, envelope))

    async def rpc_request(self, channel, envelope, **kwargs):
        self.commands.append(envelope)
        await self.listener.handle(envelope)
        receipt = next(e for c, e in reversed(self.published) if c == kwargs["reply_channel"])
        return {"data": self.codec.encode(receipt)}


@pytest.mark.usefixtures("reading_dns")
@pytest.mark.parametrize("context,requester", [("unified_chat", "juniper"), ("curiosity", "orion")])
def test_real_tool_and_listener_share_queue_and_emit_typed_acceptance(context, requester):
    conn = _FakeConn()
    bus = RpcBus(conn)
    binding = ReadingToolBindingV1(invocation_context=context, parent_run_id="actual-run", parent_trace_id="actual-trace")
    tools = ReadingTools(bus, binding)

    async def run():
        response = await tools.invoke("recommend_reading", {"url": "https://example.org/article", "why_now": "Compare this with our prior source"})
        assert response["ok"] is True
        receipt = response["result"]
        assert receipt["status"] == "queued"
        assert receipt["request"]["requested_by"] == requester
        assert receipt["request"]["invocation_context"] == context
        assert receipt["request"]["parent_run_id"] == "actual-run"
        assert receipt["request"]["parent_trace_id"] == "actual-trace"
        status_response = await tools.invoke("reading_status", {"request_id": receipt["request_id"]})
        assert status_response["ok"] is True
        assert status_response["result"] == receipt
        by_url = await tools.invoke("reading_status", {"url": "https://EXAMPLE.org/article#section"})
        assert by_url["result"]["request_id"] == receipt["request_id"]
        assert by_url["result"]["status"] == "queued"
        assert by_url["result"]["matched_request_count"] == 1
        assert by_url["result"]["lookup_url"] == "https://example.org/article"
        # Exact duplicate Pub/Sub delivery and MCP retry remain one active row.
        await bus.listener.handle(bus.commands[0])
        retry = await tools.invoke("recommend_reading", {"url": "https://example.org/article", "why_now": "Compare this with our prior source"})
        assert retry["result"]["request_id"] == receipt["request_id"]
        assert len(conn.rows) == 1
        event = next(e for c, e in bus.published if c == REQUESTED_CHANNEL)
        assert resolve("ReadingRequestedV1").model_validate(event.payload).requested_by == requester

    asyncio.run(run())


@pytest.mark.usefixtures("reading_dns")
def test_chat_ask_for_an_already_read_url_is_blocked_and_the_receipt_says_so():
    from orion.world_pulse_read.tools import RECOMMEND_DESCRIPTION, reading_brief_lines

    conn = _FakeConn()
    bus = RpcBus(conn)
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"))
    url = "https://example.org/rubin"

    async def run():
        first = (await tools.invoke("recommend_reading", {"url": url, "why_now": "GPU roadmap"}))["result"]
        assert first["duplicate"] is None
        conn.rows[first["seed_id"]].update(status="done", stage2_status="done", handoff_json={
            "read_evidence": [{"tool_name": "WebFetch", "url": url, "content_chars": 900}]})
        again = (await tools.invoke("recommend_reading", {"url": url, "why_now": "Read it again"}))["result"]
        assert again["duplicate"] == "already_read"
        assert again["duplicate_of"] == first["seed_id"]
        assert again["request_id"] != first["request_id"]
        # Nothing new waits to be read.
        assert [r["seed_id"] for r in conn.rows.values() if r["status"] == "pending"] == []

    asyncio.run(run())
    # The model is told to report the block, not to quietly re-queue.
    for text in (RECOMMEND_DESCRIPTION, " ".join(reading_brief_lines())):
        assert "duplicate='already_read'" in text
        assert "duplicate by design" in text


@pytest.mark.parametrize("selectors", [{}, {"url": "https://example.org/a", "request_id": str(uuid4())}])
def test_status_requires_exactly_one_selector(selectors):
    with pytest.raises(ValidationError):
        ReadingStatusArguments.model_validate(selectors)
    with pytest.raises(ValidationError):
        ReadingToolRequestV1(operation="reading_status", **selectors)


def test_url_status_not_found_is_read_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("status must not resolve DNS or enqueue")

    monkeypatch.setattr("socket.getaddrinfo", forbidden)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "enqueue_reading", forbidden)
    conn = _FakeConn()
    bus = RpcBus(conn)
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"))
    response = asyncio.run(tools.invoke("reading_status", {"url": "https://arxiv.org/abs/2310.19279"}))
    assert response["ok"] is True
    assert response["result"] == {
        "request_id": None, "status": "not_found",
        "lookup_url": "https://arxiv.org/abs/2310.19279", "matched_request_count": 0,
    }
    assert not conn.rows
    assert not any(c == REQUESTED_CHANNEL for c, _ in bus.published)


def test_legacy_status_requires_a_locator_and_cannot_prove_recommendation_acceptance():
    legacy = {"request_id": None, "seed_id": "legacy", "status": "queued"}
    ReadingStatusReceiptV1.model_validate(legacy)
    with pytest.raises(ValidationError):
        DurableReadingReceiptV1.model_validate(legacy)
    with pytest.raises(ValidationError):
        ReadingStatusReceiptV1.model_validate({"request_id": None, "status": "queued"})


@pytest.mark.parametrize("field", ["requested_by", "invocation_context", "parent_run_id", "parent_trace_id", "request_id", "headers", "sql", "cypher"])
def test_model_cannot_supply_provenance_or_execution_arguments(field):
    tools = ReadingTools(None, ReadingToolBindingV1(invocation_context="curiosity", parent_run_id="r", parent_trace_id="t"))
    with pytest.raises(ValidationError):
        asyncio.run(tools.invoke("recommend_reading", {"url": "https://example.org/a", "why_now": "Read", field: "juniper"}))


@pytest.mark.parametrize("url", ["", "garbage", "http://", "file:///etc/passwd", "data:text/plain,hi", "ftp://example.org/a", "http://127.0.0.1/a", "http://127.1/", "http://2130706433/", "http://10.1.2.3", "https://172.16.1.1", "http://192.168.1.1", "http://169.254.169.254", "http://100.92.216.81", "http://[::1]", "http://[::ffff:127.0.0.1]", "http://[fe80::1]", "http://[ff02::1]", "http://localhost", "http://hub-app", "http://host.internal/a", "http://user:pass@example.org", "https://example.org\\@127.0.0.1", "https://a.localhost."])
def test_reject_invalid_or_internal_urls(url):
    with pytest.raises(ValueError):
        normalize_source_url(url)


def test_reject_public_hostname_with_private_or_mixed_dns(monkeypatch):
    async def mixed(self, *args, **kwargs):
        return [(2, 1, 6, "", (ip, 443)) for ip in ("93.184.216.34", "10.0.0.1")]
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", mixed)
    with pytest.raises(ValueError):
        asyncio.run(validate_source_url("https://example.org/article"))


def test_rpc_unavailable_never_claims_queued(caplog):
    conn = _FakeConn()
    bus = RpcBus(conn)
    bus.listener.pool_provider = lambda: None
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"))
    with pytest.raises(RuntimeError, match="acceptance unknown"):
        asyncio.run(tools.invoke("recommend_reading", {"url": "https://example.org/a", "why_now": "Read"}))
    assert not conn.rows
    assert not any(c == REQUESTED_CHANNEL for c, _ in bus.published)
    assert "category=no_pool phase=pool_lookup" in caplog.text


@pytest.mark.usefixtures("reading_dns")
@pytest.mark.parametrize(
    "error,category",
    [
        (type("UndefinedColumn", (Exception,), {"sqlstate": "42703"})("request_json missing"), "schema_incompatible"),
        (RuntimeError("insert exploded"), "enqueue_failure"),
        (ValueError("public URL resolved to a private address"), "enqueue_failure"),
    ],
)
def test_listener_logs_sanitized_enqueue_failure_category(monkeypatch, caplog, error, category):
    async def fail_enqueue(*args, **kwargs):
        raise error

    conn = _FakeConn()
    bus = RpcBus(conn)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "enqueue_reading", fail_enqueue)
    tools = ReadingTools(
        bus,
        ReadingToolBindingV1(
            invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"
        ),
    )

    with pytest.raises(RuntimeError, match="acceptance unknown"):
        asyncio.run(
            tools.invoke(
                "recommend_reading",
                {"url": "https://example.org/a", "why_now": "Read"},
            )
        )

    assert f"category={category} phase=enqueue" in caplog.text
    assert f"exc_type={type(error).__name__}" in caplog.text


def test_listener_classifies_pool_acquire_failure_without_leaking_dsn(caplog):
    class BrokenPool:
        def acquire(self):
            raise ConnectionError("postgresql://operator:secret@db.internal/memory unavailable")

    conn = _FakeConn()
    bus = RpcBus(conn)
    bus.listener.pool_provider = BrokenPool
    tools = ReadingTools(
        bus,
        ReadingToolBindingV1(
            invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"
        ),
    )

    with pytest.raises(RuntimeError, match="acceptance unknown"):
        asyncio.run(
            tools.invoke(
                "recommend_reading",
                {"url": "https://example.org/a", "why_now": "Read"},
            )
        )

    assert "category=connection_failure phase=pool_acquire" in caplog.text
    assert "postgresql://[REDACTED]@db.internal/memory unavailable" in caplog.text
    assert "operator:secret" not in caplog.text


def test_listener_redacts_quoted_keyword_dsn_password(caplog):
    class BrokenPool:
        def acquire(self):
            raise ConnectionError(
                "host=db password = 'secret with spaces' user=operator unavailable"
            )

    bus = RpcBus(_FakeConn())
    bus.listener.pool_provider = BrokenPool
    tools = ReadingTools(
        bus,
        ReadingToolBindingV1(
            invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"
        ),
    )

    with pytest.raises(RuntimeError, match="acceptance unknown"):
        asyncio.run(
            tools.invoke(
                "recommend_reading",
                {"url": "https://example.org/a", "why_now": "Read"},
            )
        )

    assert "password=[REDACTED]" in caplog.text
    assert "secret with spaces" not in caplog.text


@pytest.mark.usefixtures("reading_dns")
def test_event_failure_after_commit_still_returns_durable_queue_state():
    conn = _FakeConn()
    bus = RpcBus(conn)
    original = bus.publish
    async def fail_event(channel, envelope):
        if channel == REQUESTED_CHANNEL:
            raise ConnectionError("offline")
        await original(channel, envelope)
    bus.publish = fail_event
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="curiosity", parent_run_id="r", parent_trace_id="t"))
    result = asyncio.run(tools.invoke("recommend_reading", {"url": "https://example.org/a", "why_now": "Read"}))
    assert result["ok"] is True
    assert result["result"]["status"] == "queued"
    assert len(conn.rows) == 1


def test_new_bus_subjects_resolve_registered_contracts():
    from orion.core.bus.enforce import ChannelCatalogEnforcer
    enforcer = ChannelCatalogEnforcer(enforce=True)
    for channel, schema in {
        "orion:reading:requested": "ReadingRequestedV1",
        "orion:reading:lifecycle": "ReadingLifecycleV1",
        "orion:reading:tool:request": "ReadingToolRequestV1",
        "orion:reading:tool:result:123": "ReadingToolResultV1",
    }.items():
        enforcer.validate(channel)
        assert enforcer.entry_for(channel)["schema_id"] == schema
        assert resolve(schema) is not None


def _introspect_tools(bus):
    from orion.introspect.tools import IntrospectTools
    from orion.schemas.introspect import IntrospectToolBindingV1

    return IntrospectTools(
        bus,
        IntrospectToolBindingV1(
            invocation_context="unified_chat",
            parent_run_id="r",
            parent_trace_id="t",
            memory_allowed=True,
        ),
    )


def test_reading_result_round_trips_through_real_listener(monkeypatch, caplog):
    import logging
    from datetime import datetime, timezone

    from orion.schemas.introspect import IntrospectResultV1

    seen = {}

    async def fake_results(conn, **kwargs):
        seen.update(kwargs)
        return IntrospectResultV1(
            ok=True,
            operation="reading_result",
            as_of=datetime.now(timezone.utc),
            total_available=0,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("reading_result must never enqueue")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", fake_results)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "enqueue_reading", forbidden)
    bus = RpcBus(_FakeConn())
    caplog.set_level(logging.INFO, logger="scripts.reading_listener")
    out = asyncio.run(
        _introspect_tools(bus).invoke(
            "reading_results",
            {"url": "https://EXAMPLE.org/a#frag", "limit": 3},
        )
    )
    assert out["ok"] is True and out["items"] == [] and out["total_available"] == 0
    assert seen == {"request_id": None, "url": "https://example.org/a", "limit": 3, "since": None}
    assert "introspect op=reading_result" in caplog.text
    assert f"corr={bus.commands[0].correlation_id}" in caplog.text


def test_reading_result_failure_is_unknown_and_sanitized(monkeypatch, caplog):
    from orion.introspect.tools import IntrospectUnknownError

    async def boom(conn, **kwargs):
        raise RuntimeError("lost postgres://orion:hunter2@db/orion")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", boom)
    bus = RpcBus(_FakeConn())
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        asyncio.run(_introspect_tools(bus).invoke("reading_results", {}))
    assert "hunter2" not in caplog.text
    assert "category=reading_result_failure phase=reading_result" in caplog.text


def _search_cfg(**overrides):
    from orion.world_pulse_read.search import ReadingSearchConfig

    base = dict(chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
                collection="orion_reading_results", min_similarity=0.6)
    base.update(overrides)
    return ReadingSearchConfig(**base)


def test_query_ranks_before_taking_a_connection(monkeypatch, caplog):
    import logging
    from datetime import datetime, timezone

    from orion.schemas.introspect import IntrospectResultV1

    events = []
    seen = {}

    async def fake_rank(client, cfg, query):
        events.append("rank")
        seen["query"] = query
        return [("seed-a", 0.8)]

    async def fake_gate(conn, scored, **kwargs):
        events.append("gate")
        seen.update(kwargs, scored=scored)
        return IntrospectResultV1(ok=True, operation="reading_result",
                                  as_of=datetime.now(timezone.utc), total_available=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("query mode must not use exact lookup")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "rank_readings", fake_rank)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "gated_results", fake_gate)
    monkeypatch.setitem(ReadingListener.handle.__globals__, "reading_results", forbidden)
    bus = RpcBus(_FakeConn())
    conn = bus.conn

    class RecordingPool:
        def acquire(self):
            events.append("acquire")
            return conn

    bus.listener.pool_provider = lambda: RecordingPool()
    bus.listener.search = _search_cfg()
    with caplog.at_level(logging.INFO):
        out = asyncio.run(_introspect_tools(bus).invoke("reading_results", {"query": "graphics cards", "limit": 2}))
    assert out["ok"] is True and out["items"] == []
    assert events == ["rank", "acquire", "gate"]
    assert seen["query"] == "graphics cards" and seen["limit"] == 2
    assert seen["scored"] == [("seed-a", 0.8)]
    assert "introspect op=reading_result" in caplog.text and "mode=query" in caplog.text


@pytest.mark.parametrize("configured", [False, True])
def test_query_without_working_search_is_unknown(monkeypatch, configured):
    from orion.introspect.tools import IntrospectUnknownError
    from orion.world_pulse_read.search import SearchUnavailableError

    async def down(client, cfg, query):
        raise SearchUnavailableError("reading index not built yet")

    monkeypatch.setitem(ReadingListener.handle.__globals__, "rank_readings", down)
    bus = RpcBus(_FakeConn())
    bus.listener.search = _search_cfg() if configured else _search_cfg(chroma_url="")
    with pytest.raises(IntrospectUnknownError, match="answer unknown"):
        asyncio.run(_introspect_tools(bus).invoke("reading_results", {"query": "gpus"}))


def test_index_once_uses_released_rows_and_logs(monkeypatch, caplog):
    import logging

    from orion.world_pulse_read.search import IndexPass

    calls = {}

    async def fake_rows(conn, *args, **kwargs):
        return ["row"]

    async def fake_index(rows, cfg, **kwargs):
        calls["rows"] = rows
        calls["source"] = kwargs["source"]
        return IndexPass(indexed=1, pending=2)

    monkeypatch.setitem(ReadingListener.index_once.__globals__, "verified_rows", fake_rows)
    monkeypatch.setitem(ReadingListener.index_once.__globals__, "index_missing_readings", fake_index)
    bus = RpcBus(_FakeConn())
    bus.listener.search = _search_cfg()
    with caplog.at_level(logging.INFO):
        result = asyncio.run(bus.listener.index_once())
    assert result == IndexPass(indexed=1, pending=2)
    assert calls["rows"] == ["row"] and calls["source"].name == "orion-hub"
    assert "reading_search_index indexed=1 pending=2" in caplog.text


def test_index_loop_starts_only_when_search_enabled():
    async def run(search):
        listener = ReadingListener(lambda: None, ServiceRef(name="orion-hub"), search=search)

        class Bus:
            async def publish(self, *a):
                pass

        listener._run = lambda: asyncio.sleep(3600)
        await listener.start(Bus())
        started = listener.index_task is not None
        await listener.stop()
        return started

    assert asyncio.run(run(_search_cfg())) is True
    assert asyncio.run(run(_search_cfg(chroma_url=""))) is False
    assert asyncio.run(run(None)) is False


def test_index_once_without_pool_says_so(caplog):
    import logging

    listener = ReadingListener(lambda: None, ServiceRef(name="orion-hub"), search=_search_cfg())
    with caplog.at_level(logging.INFO):
        assert asyncio.run(listener.index_once()) is None
    assert "reading_search_index skipped reason=no_pool" in caplog.text


@pytest.mark.parametrize("pool_ready,expected", [(False, 15.0), (True, 300.0)])
def test_index_loop_retries_soon_only_while_pool_is_missing(monkeypatch, pool_ready, expected):
    from orion.world_pulse_read.search import IndexPass

    delays = []

    async def fake_sleep(delay):
        delays.append(delay)
        raise asyncio.CancelledError

    async def fake_index_once(self):
        return IndexPass(indexed=0, pending=0) if pool_ready else None

    monkeypatch.setattr(ReadingListener, "index_once", fake_index_once)
    monkeypatch.setattr(ReadingListener._index_loop.__globals__["asyncio"], "sleep", fake_sleep)
    listener = ReadingListener(lambda: None, ServiceRef(name="orion-hub"), search=_search_cfg())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(listener._index_loop())
    assert delays == [expected]


def test_chat_recommends_a_document_path_and_versions_dedup(tmp_path):
    from orion.world_pulse_read.documents import DocumentPolicy

    conn = _FakeConn()
    bus = RpcBus(conn)
    bus.listener.documents = DocumentPolicy.from_values(roots=str(tmp_path), extensions=None, max_bytes=4096)
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"))
    doc = tmp_path / "spec.md"
    doc.write_text("# Spec v1\n\nFirst version of the design.\n")

    async def run():
        first = await tools.invoke("recommend_reading", {"url": str(doc), "why_now": "Review the spec"})
        assert first["ok"] is True
        row = conn.rows[first["result"]["seed_id"]]
        assert row["url"].startswith(f"file://{doc}?sha256=")
        assert conn.snapshots[row["url"].rsplit("=", 1)[1]]["content"].startswith("# Spec v1")
        event = next(e for c, e in bus.published if c == REQUESTED_CHANNEL)
        assert resolve("ReadingRequestedV1").model_validate(event.payload).url == row["url"]

        # Unchanged file: joins the read already waiting, not a second read.
        same = await tools.invoke("recommend_reading", {"url": f"file://{doc}", "why_now": "Again, please"})
        assert same["result"]["duplicate"] == "already_queued"
        assert same["result"]["duplicate_of"] == first["result"]["seed_id"]

        # Edited file: a new version is a new read.
        doc.write_text("# Spec v2\n\nThe design changed.\n")
        edited = await tools.invoke("recommend_reading", {"url": str(doc), "why_now": "It changed"})
        assert edited["result"]["duplicate"] is None
        assert conn.rows[edited["result"]["seed_id"]]["url"] != row["url"]

        # A bare path looks up the latest captured version of that file.
        status = await tools.invoke("reading_status", {"url": str(doc)})
        assert status["result"]["request_id"] == edited["result"]["request_id"]
        assert status["result"]["matched_request_count"] == 3

        outside = tmp_path.parent / "elsewhere.md"
        outside.write_text("not allowed")
        before = len(conn.rows)
        with pytest.raises(RuntimeError, match="document_outside_allowed_roots"):
            await tools.invoke("recommend_reading", {"url": str(outside), "why_now": "Try it"})
        assert len(conn.rows) == before

    asyncio.run(run())


def test_a_pinned_document_ref_still_needs_policy_and_provenance(tmp_path):
    from orion.world_pulse_read.documents import DocumentPolicy

    conn = _FakeConn()
    bus = RpcBus(conn)
    bus.listener.documents = DocumentPolicy.from_values(roots=str(tmp_path), extensions=None, max_bytes=4096)
    tools = ReadingTools(bus, ReadingToolBindingV1(invocation_context="unified_chat", parent_run_id="r", parent_trace_id="t"))
    doc, other = tmp_path / "spec.md", tmp_path / "other.md"
    doc.write_text("# Spec\n\nThe real design.\n")
    other.write_text("# Other\n")

    async def run():
        first = await tools.invoke("recommend_reading", {"url": str(doc), "why_now": "Review"})
        pinned = conn.rows[first["result"]["seed_id"]]["url"]
        sha = pinned.rsplit("=", 1)[1]
        # The exact ref Hub captured is accepted again without touching the file.
        doc.write_text("# Spec\n\nEdited after capture.\n")
        again = await tools.invoke("recommend_reading", {"url": pinned, "why_now": "Again"})
        assert again["result"]["duplicate_of"] == first["result"]["seed_id"]
        before = len(conn.rows)
        # Someone else's hash cannot vouch for a different path.
        for forged, code in [
            (f"file:///etc/shadow?sha256={sha}", "document_outside_allowed_roots"),
            (f"file://{other}?sha256={sha}", "document_snapshot_missing"),
        ]:
            with pytest.raises(RuntimeError, match=code):
                await tools.invoke("recommend_reading", {"url": forged, "why_now": "Forged"})
        # The kill switch covers pinned refs too.
        bus.listener.documents = DocumentPolicy.from_values(roots="", extensions=None, max_bytes=4096)
        with pytest.raises(RuntimeError, match="document_reading_disabled"):
            await tools.invoke("recommend_reading", {"url": pinned, "why_now": "Disabled"})
        assert len(conn.rows) == before

    asyncio.run(run())
