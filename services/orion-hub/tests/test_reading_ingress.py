"""Real domain/codec and runtime adapters; fake SQL/DNS/bus for fast local gates."""
import asyncio
from uuid import uuid4

import pytest
from pydantic import ValidationError

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.schemas.reading import ReadingRequestedV1, ReadingToolBindingV1, ReadingToolRequestV1
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
        # Exact duplicate Pub/Sub delivery and MCP retry remain one active row.
        await bus.listener.handle(bus.commands[0])
        retry = await tools.invoke("recommend_reading", {"url": "https://example.org/article", "why_now": "Compare this with our prior source"})
        assert retry["result"]["request_id"] == receipt["request_id"]
        assert len(conn.rows) == 1
        event = next(e for c, e in bus.published if c == REQUESTED_CHANNEL)
        assert resolve("ReadingRequestedV1").model_validate(event.payload).requested_by == requester

    asyncio.run(run())


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
