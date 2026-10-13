"""Hub's introspect responder for the `node_history` tool.

Orion asks what a substrate node's prediction error did over a window (up to 7 days): a
compact summary from substrate_node_prediction_error_history (orion.introspect.node_history).
The 2026-10-09 Orion's Day letter said the synaptic bus node "holds only a current value, no
series"; the series was in Postgres all along, just not readable by any tool Orion had.

Same trust and honesty rules as the other Hub responders (curiosity_introspect_listener.py):
replies only when reply_to is exactly orion:introspect:result:<corr> and the kind matches;
every read runs in a READ ONLY transaction; errors become ok=false ("answer unknown"),
never []. ok=true with items=[] means the node (or, with no node_id, every node) has no
reading in that window.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.node_history import list_history_nodes, read_node_history
from orion.introspect.redact import safe_exception_detail
from orion.introspect.transport import NODE_HISTORY_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectRequestV1, IntrospectResultV1, NodeHistoryArguments

from scripts.curiosity_introspect_listener import ReadOnlyPool

logger = logging.getLogger("orion-hub.node_history_introspect")

OPERATION = "node_history"
QUERY_UNAVAILABLE = "node_history_unavailable; answer unknown"
_INVALID_CAP = 400


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation=OPERATION, as_of=now, error=error)


class NodeHistoryIntrospectListener:
    def __init__(self, *, pool_provider: Callable[[], Any], source_ref: Any):
        self.pool_provider = pool_provider
        self.source_ref = source_ref
        self.bus: Any = None
        self.task: asyncio.Task | None = None

    def _pool(self) -> ReadOnlyPool:
        pool = self.pool_provider()
        if pool is None:
            raise RuntimeError("no memory pool")
        return ReadOnlyPool(pool)

    async def answer(self, args: NodeHistoryArguments, now: datetime) -> IntrospectResultV1:
        start, end = args.window(now)
        async with self._pool().acquire() as conn:
            if args.node_id is None:
                item = await list_history_nodes(conn, start, end)
            else:
                item = await read_node_history(conn, args.node_id, start, end)
        items = [item] if item is not None else []
        return IntrospectResultV1(
            ok=True, operation=OPERATION, as_of=now, total_available=len(items), items=items,
        )

    async def handle(self, envelope: BaseEnvelope) -> None:
        reply = f"{RESULT_PREFIX}{envelope.correlation_id}"
        if envelope.reply_to != reply or envelope.kind != REQUEST_KIND:
            return
        now = datetime.now(timezone.utc)
        try:
            request = IntrospectRequestV1.model_validate(envelope.payload)
            if request.operation != OPERATION:
                raise ValueError(f"operation {request.operation!r} is not answered here")
            args = NodeHistoryArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid node_history request: {str(exc)[:_INVALID_CAP]}"))
            return
        try:
            result = await self.answer(args, now)
        except Exception as exc:
            logger.warning(
                "introspect_failed op=node_history corr=%s node=%s category=node_history_query_failure "
                "exc_type=%s detail=%s",
                envelope.correlation_id, args.node_id, type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, QUERY_UNAVAILABLE)
        else:
            count = result.items[0].extra.get("count") if result.items else 0
            logger.info(
                "node_history_introspect_answered corr=%s node=%s hours=%g items=%d readings=%s",
                envelope.correlation_id, args.node_id, args.hours, len(result.items), count,
            )
        await self._publish(reply, envelope, result)

    async def _publish(self, reply: str, envelope: BaseEnvelope, result: IntrospectResultV1) -> None:
        await self.bus.publish(reply, BaseEnvelope(
            kind=RESULT_KIND, correlation_id=envelope.correlation_id,
            source=self.source_ref, payload=result.model_dump(mode="json"),
        ))

    async def _run(self) -> None:
        while True:
            try:
                async with self.bus.subscribe(NODE_HISTORY_REQUEST_CHANNEL) as pubsub:
                    logger.info("node_history_introspect_listening channel=%s", NODE_HISTORY_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("node_history_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("node_history_introspect_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-node-history-introspect")

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None
