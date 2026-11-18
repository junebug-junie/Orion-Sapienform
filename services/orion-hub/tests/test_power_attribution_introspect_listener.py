"""Hub power_attribution responder, and the unified-chat path that reaches it.

The last test is the end-to-end proof Juniper asked for: a unified chat turn's
orion-introspect MCP server (the same `build_server` + `IntrospectTools` the harness spawns,
bound to invocation_context="unified_chat") is asked "why is my cabinet running hot?" as a
`power_attribution` tool call; the bus request reaches this Hub responder, which reads the
fixture rows through a READ ONLY transaction and returns the attribution with its evidence.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

import scripts.power_attribution_introspect_listener as pl
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.introspect.tests.power_attribution_fake import FakePool, FakePowerConn
from orion.introspect.transport import POWER_ATTRIBUTION_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.physiology.tests import heat_fixture as fx
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

BINDING = IntrospectToolBindingV1(
    invocation_context="unified_chat", parent_run_id="chat-1", parent_trace_id="t", memory_allowed=True,
)
# The fixture's window ends 2026-10-21 03:00; "now" for the listener is pinned there.
NOW = fx.END


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(conn):
    listener = pl.PowerAttributionIntrospectListener(
        pool_provider=(lambda: FakePool(conn)) if conn is not None else (lambda: None),
        source_ref=ServiceRef(name="orion-hub"),
    )
    listener.bus = _Bus()
    return listener


@pytest.fixture(autouse=True)
def _pinned_now(monkeypatch):
    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(pl, "datetime", _Now)
    import orion.schemas.introspect as si
    monkeypatch.setattr(si, "datetime", _Now)


def _ask(listener, args, *, reply_to=None, operation="power_attribution"):
    corr = uuid4()
    envelope = BaseEnvelope(
        kind=REQUEST_KIND, correlation_id=corr,
        reply_to=reply_to or f"{RESULT_PREFIX}{corr}",
        source=ServiceRef(name="orion-harness-governor"),
        payload=IntrospectRequestV1(operation=operation, binding=BINDING, args=args).model_dump(mode="json"),
    )
    asyncio.run(listener.handle(envelope))
    if not listener.bus.published:
        return None
    channel, out = listener.bus.published[-1]
    assert channel == f"{RESULT_PREFIX}{corr}" and out.kind == RESULT_KIND
    return IntrospectResultV1.model_validate(out.payload)


def test_answers_with_the_step_attribution_and_the_sql_read_only():
    conn = FakePowerConn(fx.db_rows())
    result = _ask(_listener(conn), {"days": 14})
    assert result.ok and result.operation == "power_attribution"
    circe = next(i for i in result.items if i.extra["node"] == "circe")
    assert "since around 2026-10-14" in circe.text and "CPU-load regime" in circe.text
    assert "UNVERIFIED" in circe.text and circe.extra["verified"] is False
    assert result.items[-1].kind == "power_attribution_queries"
    assert conn.readonly_flags == [True]


def test_no_wall_reading_is_empty_not_unknown():
    conn = FakePowerConn({"energy_hourly": [], "biometrics_hourly": [], "pool_busy_hours": []})
    result = _ask(_listener(conn), {})
    assert result.ok and result.items == [] and result.total_available == 0


def test_database_failure_is_unknown_and_leaks_no_detail():
    result = _ask(_listener(FakePowerConn({}, fail=True)), {})
    assert not result.ok and result.error == pl.QUERY_UNAVAILABLE
    assert "hunter2" not in (result.error or "")


def test_no_pool_is_unknown():
    result = _ask(_listener(None), {})
    assert not result.ok and "answer unknown" in result.error


def test_invalid_args_are_refused_with_reason():
    result = _ask(_listener(FakePowerConn(fx.db_rows())), {"days": 400})
    assert not result.ok and result.error.startswith("invalid power_attribution request")


def test_wrong_reply_channel_or_operation_is_ignored_or_refused():
    listener = _listener(FakePowerConn(fx.db_rows()))
    assert _ask(listener, {}, reply_to="orion:somewhere:else") is None
    result = _ask(listener, {}, operation="node_history")
    assert not result.ok and "not answered here" in result.error


class _BridgeBus:
    """The governor-side bus, wired straight to the Hub responder: rpc_request publishes the
    request to the listener and returns whatever the listener publishes on the reply channel."""

    codec = OrionCodec()

    def __init__(self, listener):
        self.listener = listener
        self.channels = []

    async def rpc_request(self, channel, envelope, *, reply_channel, timeout_sec):
        self.channels.append(channel)
        assert channel == POWER_ATTRIBUTION_REQUEST_CHANNEL
        await self.listener.handle(envelope)
        out_channel, out = self.listener.bus.published[-1]
        assert out_channel == reply_channel
        return {"data": self.codec.encode(out)}


def test_unified_chat_turn_reaches_power_attribution_through_the_mcp_server():
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session

    from orion.introspect.brief import POWER_ATTRIBUTION, introspect_brief_lines
    from orion.introspect.mcp_server import build_server
    from orion.introspect.tools import IntrospectTools

    # The unified turn's harness brief names the tool for exactly this question.
    brief = " ".join(introspect_brief_lines(BINDING))
    assert POWER_ATTRIBUTION in brief and "cabinet runs hot" in brief

    conn = FakePowerConn(fx.db_rows())
    bridge = _BridgeBus(_listener(conn))
    server = build_server(IntrospectTools(bridge, BINDING))

    async def run():
        async with create_connected_server_and_client_session(server) as client:
            listing = await client.list_tools()
            assert "power_attribution" in {t.name for t in listing.tools}
            # "why is my cabinet running hot?" -> the model's tool call:
            response = await client.call_tool("power_attribution", {"days": 14})
            assert not response.isError
            return json.loads(response.content[0].text)

    payload = asyncio.run(run())
    assert bridge.channels == [POWER_ATTRIBUTION_REQUEST_CHANNEL]
    assert payload["ok"] is True
    circe = next(i for i in payload["items"] if i["extra"].get("node") == "circe")
    assert "CPU-load regime" in circe["text"] and "since around 2026-10-14" in circe["text"]
    assert {e["query"] for e in circe["extra"]["evidence"]} == {"energy_hourly", "biometrics_hourly", "pool_busy_hours"}
    assert payload["items"][-1]["extra"]["queries"]["pool_busy_hours"].startswith("with ends as")
    assert conn.readonly_flags == [True]
