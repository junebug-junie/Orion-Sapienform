"""Hub node_history responder: read-only, empty node is items=[], failures are unknown."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import scripts.node_history_introspect_listener as nl
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.introspect.tests.node_history_fake import FakeConn
from orion.introspect.transport import REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, IntrospectToolBindingV1

NODE = "node:substrate.bus_synaptic"
BINDING = IntrospectToolBindingV1(
    invocation_context="curiosity", parent_run_id="r", parent_trace_id="t", memory_allowed=True,
)


class _Conn(FakeConn):
    def __init__(self, rows, fail=False):
        super().__init__(rows)
        self.fail = fail
        self.readonly_flags: list[bool] = []

    @asynccontextmanager
    async def transaction(self, readonly=False):
        self.readonly_flags.append(readonly)
        yield

    async def fetchrow(self, sql, *args):
        if self.fail:
            raise RuntimeError("password=hunter2 host=db")
        return await super().fetchrow(sql, *args)


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class _Bus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, envelope):
        self.published.append((channel, envelope))


def _listener(conn):
    listener = nl.NodeHistoryIntrospectListener(
        pool_provider=(lambda: _Pool(conn)) if conn is not None else (lambda: None),
        source_ref=ServiceRef(name="orion-hub"),
    )
    listener.bus = _Bus()
    return listener


def _ask(listener, args, *, reply_to=None, operation="node_history"):
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


def _recent_rows(node=NODE):
    # FakeConn rows are anchored to the real clock here: the listener windows on now().
    now = datetime.now(timezone.utc)
    return [(node, now - timedelta(minutes=m), 0.02 + 0.001 * (m % 5)) for m in range(1, 600, 2)]


def test_history_answered_in_a_read_only_transaction():
    conn = _Conn(_recent_rows())
    result = _ask(_listener(conn), {"node_id": NODE, "hours": 12})
    assert result.ok and result.total_available == 1
    [item] = result.items
    assert item.kind == "node_prediction_error_history" and item.extra["count"] == 300
    assert conn.readonly_flags == [True]


def test_node_with_no_reading_in_window_is_ok_with_no_items():
    old = [(NODE, datetime.now(timezone.utc) - timedelta(hours=100), 0.3)]
    result = _ask(_listener(_Conn(old)), {"node_id": NODE, "hours": 24})
    assert result.ok is True and result.items == [] and result.total_available == 0


def test_node_with_no_stored_reading_is_not_found_not_empty():
    result = _ask(_listener(_Conn(_recent_rows())), {"node_id": "node:substrate.never_written", "hours": 72})
    assert result.ok is False and "(not found)" in result.error and "no node_id" in result.error


def test_no_node_id_lists_nodes():
    rows = _recent_rows() + _recent_rows("node:substrate.cabinet")
    result = _ask(_listener(_Conn(rows)), {"hours": 24})
    assert [n["node_id"] for n in result.items[0].extra["nodes"]] == ["node:substrate.bus_synaptic",
                                                                     "node:substrate.cabinet"]


def test_window_past_retention_is_an_invalid_request():
    result = _ask(_listener(_Conn([])), {"node_id": NODE, "hours": 200})
    assert result.ok is False and result.error.startswith("invalid node_history request")


def test_query_failure_is_unknown_and_does_not_leak_detail():
    result = _ask(_listener(_Conn([], fail=True)), {"node_id": NODE})
    assert result.ok is False and result.error == nl.QUERY_UNAVAILABLE
    assert "hunter2" not in result.error


def test_no_pool_is_unknown():
    result = _ask(_listener(None), {"node_id": NODE})
    assert result.ok is False and "answer unknown" in result.error


def test_wrong_reply_channel_is_ignored():
    listener = _listener(_Conn(_recent_rows()))
    assert _ask(listener, {"node_id": NODE}, reply_to="orion:somewhere:else") is None


def test_other_operation_is_refused():
    result = _ask(_listener(_Conn([])), {}, operation="dreams")
    assert result.ok is False and "not answered here" in result.error
