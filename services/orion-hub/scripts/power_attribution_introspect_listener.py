"""Hub's introspect responder for the `power_attribution` tool (heat archaeology).

Orion asks where their power (and so the cabinet's heat) went, and since when. The answer is
computed from four read-only sources (orion_energy_hourly, orion_biometrics_cluster,
gpu_pool_events, durable_run_workflow; queries in orion/introspect/power_attribution.py) by the
pure analysis in orion/physiology/heat_archaeology.py. Curiosity runs kept asking "my cabinet
runs hot and I don't know why"; the 2026-10-14 circe step (~790 W -> ~1100 W, a crash loop
worth ~144 W among it) was found by hand with exactly this method.

Same trust and honesty rules as the other Hub responders (node_history_introspect_listener.py):
replies only when reply_to is exactly orion:introspect:result:<corr> and the kind matches;
every read runs in a READ ONLY transaction; errors become ok=false ("answer unknown"), never [].
ok=true with items=[] means no machine had a wall-power reading in the window.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.introspect.power_attribution import build_items, read_power_attribution
from orion.introspect.redact import safe_exception_detail
from orion.introspect.transport import POWER_ATTRIBUTION_REQUEST_CHANNEL, REQUEST_KIND, RESULT_KIND, RESULT_PREFIX
from orion.schemas.introspect import IntrospectResultV1, IntrospectRequestV1, PowerAttributionArguments

from scripts.curiosity_introspect_listener import ReadOnlyPool

logger = logging.getLogger("orion-hub.power_attribution_introspect")

OPERATION = "power_attribution"
QUERY_UNAVAILABLE = "power_attribution_unavailable; answer unknown"
_INVALID_CAP = 400


def _failed(now: datetime, error: str) -> IntrospectResultV1:
    return IntrospectResultV1(ok=False, operation=OPERATION, as_of=now, error=error)


class PowerAttributionIntrospectListener:
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

    async def answer(self, args: PowerAttributionArguments, now: datetime) -> IntrospectResultV1:
        start, end = args.window(now)
        async with self._pool().acquire() as conn:
            results = await read_power_attribution(conn, start, end, args.node)
        results = [r for r in results if r.hours_with_wall > 0]
        if not results:
            return IntrospectResultV1(ok=True, operation=OPERATION, as_of=now, total_available=0, items=[])
        items = build_items(results, start, end)
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
            args = PowerAttributionArguments.model_validate(request.args)
        except ValueError as exc:
            await self._publish(reply, envelope, _failed(now, f"invalid power_attribution request: {str(exc)[:_INVALID_CAP]}"))
            return
        try:
            result = await self.answer(args, now)
        except Exception as exc:
            logger.warning(
                "introspect_failed op=power_attribution corr=%s node=%s category=power_attribution_query_failure "
                "exc_type=%s detail=%s",
                envelope.correlation_id, args.node, type(exc).__name__, safe_exception_detail(exc),
            )
            result = _failed(now, QUERY_UNAVAILABLE)
        else:
            steps = sum(len(i.extra.get("steps") or []) for i in result.items)
            logger.info(
                "power_attribution_introspect_answered corr=%s node=%s days=%g items=%d steps=%d",
                envelope.correlation_id, args.node, args.days, len(result.items), steps,
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
                async with self.bus.subscribe(POWER_ATTRIBUTION_REQUEST_CHANNEL) as pubsub:
                    logger.info("power_attribution_introspect_listening channel=%s", POWER_ATTRIBUTION_REQUEST_CHANNEL)
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                await self.handle(decoded.envelope)
                        except Exception:
                            logger.warning("power_attribution_introspect_message_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("power_attribution_introspect_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def start(self, bus: Any) -> None:
        self.bus = bus
        self.task = asyncio.create_task(self._run(), name="hub-power-attribution-introspect")

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task
            self.task = None
