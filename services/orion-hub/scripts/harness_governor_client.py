from __future__ import annotations

import asyncio
import logging
import uuid
from time import perf_counter
from typing import Callable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.llm.routes import is_agent_route_model_label
from orion.schemas.harness_finalize import HarnessRunCancelV1, HarnessRunRequestV1, HarnessRunV1
from scripts.settings import settings

logger = logging.getLogger("hub.bus.harness_governor")

LivenessCheckFn = Callable[[float], bool]


async def _get_message_within(pubsub, timeout: float) -> dict | None:
    """pubsub.get_message() performs exactly one read per call: if that single read
    consumes a non-publish message (most commonly the subscribe confirmation, which
    arrives immediately after pubsub.subscribe()), it returns None almost instantly
    instead of waiting out `timeout`. Loop on the remaining budget so a spurious single
    read can't cut the wait short.
    """
    deadline = perf_counter() + timeout
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return None
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=remaining)
        if msg is not None:
            return msg


class HarnessGovernorClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

    def _liveness_alive(
        self,
        liveness_check: LivenessCheckFn | None,
        liveness_window_sec: float,
        correlation_id: str,
    ) -> bool:
        if liveness_check is None:
            return False
        try:
            return bool(liveness_check(liveness_window_sec))
        except Exception:
            logger.warning(
                "[%s] liveness_check raised, treating as not-alive",
                correlation_id,
                exc_info=True,
            )
            return False

    async def run(
        self,
        request: HarnessRunRequestV1,
        *,
        correlation_id: Optional[str] = None,
        timeout_sec: float | None = None,
        liveness_check: LivenessCheckFn | None = None,
    ) -> HarnessRunV1 | None:
        """Dispatch a harness run and wait for its reply.

        Which governor dispatch queue the request goes out on
        (CHANNEL_HARNESS_RUN_REQUEST vs CHANNEL_HARNESS_RUN_REQUEST_AGENT) is
        derived here from `request.fcc_model_label` -- the SAME field that
        already picked the turn's model -- via
        `orion.llm.routes.is_agent_route_model_label`, not passed in
        separately. That is deliberate: it is the one fact every caller
        already has to set correctly (the model would be wrong otherwise), so
        there is nothing left for a second, independently-computed flag to
        get out of sync with. Both queues are consumed by the same governor
        code (two independent loops, see bus_listener.run_bus_worker) -- this
        only changes which queue a turn waits in, never how it runs. See
        is_agent_route_model_label's own docstring for why curiosity and a
        manual Mode=Agent+Compute=Agent chat turn both land on the agent
        queue (they already share one physical GPU) while ordinary chat keeps
        its own queue untouched by either.
        """
        correlation_id = correlation_id or request.correlation_id or str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_HARNESS_RESULT_PREFIX}{correlation_id}"
        is_agent_lane = is_agent_route_model_label(request.fcc_model_label)
        request_channel = (
            settings.CHANNEL_HARNESS_RUN_REQUEST_AGENT if is_agent_lane else settings.CHANNEL_HARNESS_RUN_REQUEST
        )
        poll_sec = max(
            0.1,
            float(
                timeout_sec
                if timeout_sec is not None
                else settings.HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC
            ),
        )
        # HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC is a hard ceiling: clamp the poll size down
        # to it rather than letting an oversized poll_sec silently expand the ceiling.
        max_wait_sec = float(settings.HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC)
        poll_sec = min(poll_sec, max_wait_sec)
        # Fixed recency threshold for "is the governor still active" — deliberately NOT
        # poll_sec, which shrinks over the loop for reasons unrelated to step cadence and
        # would otherwise make the check either too lenient (early, large poll_sec) or too
        # strict (late, poll_sec shrunk toward the ceiling).
        liveness_window_sec = max(0.1, float(settings.HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC))
        envelope = BaseEnvelope(
            kind="harness.run.request.v1",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request.model_dump(mode="json"),
        )

        started = perf_counter()
        worker_task = getattr(self.bus, "_rpc_worker_task", None)
        if worker_task is not None and not worker_task.done():
            # Reuse the bus's shared, pooled RPC-worker connection (fork_rpc_client /
            # start_rpc_worker=True) instead of opening a dedicated ad-hoc pubsub
            # connection per turn — with many concurrent long-running turns, ad-hoc
            # connections held open for up to max_wait_sec risk exhausting Redis
            # maxclients / hub file descriptors.
            msg = await self._run_via_worker(
                envelope,
                request_channel=request_channel,
                reply_to=reply_to,
                poll_sec=poll_sec,
                max_wait_sec=max_wait_sec,
                liveness_window_sec=liveness_window_sec,
                liveness_check=liveness_check,
                correlation_id=correlation_id,
                started=started,
            )
        else:
            msg = await self._run_via_ad_hoc_subscribe(
                envelope,
                request_channel=request_channel,
                reply_to=reply_to,
                poll_sec=poll_sec,
                max_wait_sec=max_wait_sec,
                liveness_window_sec=liveness_window_sec,
                liveness_check=liveness_check,
                correlation_id=correlation_id,
                started=started,
            )
        if msg is None:
            return None
        logger.info(
            "[%s] harness governor reply received elapsed_sec=%.1f",
            correlation_id,
            perf_counter() - started,
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            return None
        payload = decoded.envelope.payload
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning(
                "[%s] harness governor RPC error payload=%s",
                correlation_id,
                payload.get("error"),
            )
            return None
        if isinstance(payload, dict):
            return HarnessRunV1.model_validate(payload)
        return None

    async def _run_via_worker(
        self,
        envelope: BaseEnvelope,
        *,
        request_channel: str,
        reply_to: str,
        poll_sec: float,
        max_wait_sec: float,
        liveness_window_sec: float,
        liveness_check: LivenessCheckFn | None,
        correlation_id: str,
        started: float,
    ) -> dict | None:
        corr = str(envelope.correlation_id)
        key = (reply_to, corr)
        fut = asyncio.get_running_loop().create_future()
        self.bus._pending_rpc[key] = fut
        try:
            async with self.bus._rpc_lock:
                await self.bus._rpc_subscribe(reply_to)
            await self.bus.publish(request_channel, envelope)
            wait = poll_sec
            while True:
                try:
                    # shield: a per-chunk timeout must not cancel `fut` itself — the
                    # worker's dispatcher (_handle_rpc_result) may still resolve it on
                    # a later retry.
                    return await asyncio.wait_for(asyncio.shield(fut), timeout=wait)
                except asyncio.TimeoutError:
                    elapsed = perf_counter() - started
                    remaining = max_wait_sec - elapsed
                    alive = self._liveness_alive(liveness_check, liveness_window_sec, correlation_id)
                    if remaining <= 0 or not alive:
                        logger.warning(
                            "[%s] harness governor RPC timeout elapsed_sec=%.1f alive=%s path=worker",
                            correlation_id,
                            elapsed,
                            alive,
                        )
                        return None
                    logger.info(
                        "[%s] harness governor still active after %.1fs, extending wait "
                        "(remaining_sec=%.1f) path=worker",
                        correlation_id,
                        elapsed,
                        remaining,
                    )
                    wait = min(poll_sec, remaining)
        finally:
            self.bus._pending_rpc.pop(key, None)

    async def _run_via_ad_hoc_subscribe(
        self,
        envelope: BaseEnvelope,
        *,
        request_channel: str,
        reply_to: str,
        poll_sec: float,
        max_wait_sec: float,
        liveness_window_sec: float,
        liveness_check: LivenessCheckFn | None,
        correlation_id: str,
        started: float,
    ) -> dict | None:
        async with self.bus.subscribe(reply_to) as pubsub:
            await self.bus.publish(request_channel, envelope)
            wait = poll_sec
            while True:
                # _get_message_within's own timeout (not asyncio.wait_for cancelling an
                # in-flight read) so a reply landing right at the poll boundary can't be
                # silently dropped by task cancellation racing message delivery.
                msg = await _get_message_within(pubsub, wait)
                if msg is not None:
                    return msg
                elapsed = perf_counter() - started
                remaining = max_wait_sec - elapsed
                alive = self._liveness_alive(liveness_check, liveness_window_sec, correlation_id)
                if remaining <= 0 or not alive:
                    logger.warning(
                        "[%s] harness governor RPC timeout elapsed_sec=%.1f alive=%s path=inline",
                        correlation_id,
                        elapsed,
                        alive,
                    )
                    return None
                logger.info(
                    "[%s] harness governor still active after %.1fs, extending wait "
                    "(remaining_sec=%.1f) path=inline",
                    correlation_id,
                    elapsed,
                    remaining,
                )
                wait = min(poll_sec, remaining)

    async def cancel(
        self,
        *,
        correlation_id: str,
        reason: str = "client_disconnect",
    ) -> None:
        """Fire-and-forget cancel for an in-flight FCC motor (no reply expected)."""
        channel = str(
            getattr(settings, "CHANNEL_HARNESS_RUN_CANCEL", None) or "orion:harness:run:cancel"
        )
        cancel = HarnessRunCancelV1(correlation_id=str(correlation_id), reason=str(reason or "client_disconnect"))
        envelope = BaseEnvelope(
            kind="harness.run.cancel.v1",
            source=self._source,
            correlation_id=str(correlation_id),
            payload=cancel.model_dump(mode="json"),
        )
        try:
            await self.bus.publish(channel, envelope)
            logger.info("[%s] harness run cancel published reason=%s", correlation_id, cancel.reason)
        except Exception:
            logger.warning("[%s] harness run cancel publish failed", correlation_id, exc_info=True)
