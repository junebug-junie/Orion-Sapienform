"""orion-dream's dream.carry step responder (orion:dream:carry:step:request).

Trust: answers only envelopes of kind dream.carry.step.request.v1 whose reply_to is under
orion:dream:carry:step:reply:. The reply echoes the request envelope's correlation id and the
payload's run_id / correlation_id / step, so durable-runs can check identity.

Each request runs as its own task (a text hop is one LLM call, up to minutes), bounded by a
small semaphore, so a slow hop never blocks the subscription. None of this touches the sleep
loop: it has its own bus connection.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any, Callable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.dream_carry import (
    DREAM_CARRY_LLM_ROUTE,
    DREAM_CARRY_STEP_CHANNEL,
    DREAM_CARRY_STEP_REPLY_PREFIX,
    DREAM_CARRY_STEP_REQUEST_KIND,
    DREAM_CARRY_STEP_RESULT_KIND,
    DreamCarryStepRequestV1,
    DreamCarryStepResultV1,
)
from orion.schemas.telemetry.dream import DreamResultV1

from app import carry, llm

logger = logging.getLogger("orion-dream.carry.listener")

MAX_CONCURRENT_STEPS = 2
DREAM_RESULT_KIND = "dream.result.v1"


class DreamCarryListener:
    def __init__(
        self, *, bus_url: str, source: Any, dream_log_channel: str,
        already_recorded: Optional[carry.AlreadyRecorded] = None,
        bus_factory: Callable[[str], Any] = OrionBusAsync,
    ):
        self.bus_url = bus_url
        self.source = source
        self.dream_log_channel = dream_log_channel
        self.already_recorded = already_recorded
        self.bus_factory = bus_factory
        self.bus: Any = None
        self.ledger = carry.FinishLedger()
        self.failures = carry.ReplyFailures()
        self._handler_errors: dict[tuple[str, str, int | None], int] = {}
        self._finish_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_STEPS)
        self._inflight: set[asyncio.Task] = set()
        self.task: asyncio.Task | None = None

    # --- seams the handlers use -------------------------------------------------
    async def complete(self, prompt: str, gpu_lease: dict, timeout_sec: float) -> str:
        return await llm.complete(
            self.bus, prompt, max_tokens=carry.PASSAGE_MAX_TOKENS, purpose=carry.CARRY_PURPOSE,
            route=DREAM_CARRY_LLM_ROUTE, gpu_lease=gpu_lease, timeout_sec=timeout_sec,
        )

    async def publish_dream(self, dream: DreamResultV1) -> None:
        await self.bus.publish(self.dream_log_channel, BaseEnvelope(
            kind=DREAM_RESULT_KIND, source=self.source, correlation_id=_corr(dream.correlation_id),
            payload=dream.model_dump(mode="json"),
        ))

    # --- one request ----------------------------------------------------------------
    async def handle(self, envelope: BaseEnvelope) -> Optional[DreamCarryStepResultV1]:
        received = time.monotonic()
        reply_to = envelope.reply_to or ""
        if envelope.kind != DREAM_CARRY_STEP_REQUEST_KIND or not reply_to.startswith(f"{DREAM_CARRY_STEP_REPLY_PREFIX}:"):
            return None
        try:
            request = DreamCarryStepRequestV1.model_validate(envelope.payload)
        except ValueError as exc:
            result = _invalid_result(envelope.payload, exc)
            if result is None:
                logger.warning("dream_carry_step_unanswerable corr=%s err=%s", envelope.correlation_id, str(exc)[:300])
                return None
        else:
            async with self._slots:
                waited = time.monotonic() - received
                if request.step == "finish":
                    # One finish at a time: the ledger check and the publish must not interleave.
                    async with self._finish_lock:
                        result = await self._step(request, waited)
                else:
                    result = await self._step(request, waited)
        await self.bus.publish(reply_to, BaseEnvelope(
            kind=DREAM_CARRY_STEP_RESULT_KIND, source=self.source, correlation_id=envelope.correlation_id,
            payload=result.model_dump(mode="json"),
        ))
        return result

    async def _step(self, request: DreamCarryStepRequestV1, waited_sec: float = 0.0) -> DreamCarryStepResultV1:
        try:
            return await carry.handle_step(
                request, complete=self.complete, publish=self.publish_dream,
                ledger=self.ledger, already_recorded=self.already_recorded,
                failures=self.failures, waited_sec=waited_sec,
            )
        except Exception as exc:  # a handler bug answers retry (then terminal), never silence
            logger.exception("dream_carry_step_failed run=%s step=%s", request.run_id, request.step)
            key = (request.run_id, request.step, request.hop_index)
            self._handler_errors[key] = self._handler_errors.get(key, 0) + 1
            reason = f"handler_{type(exc).__name__}"[:280]
            if self._handler_errors[key] >= carry.MAX_REPLY_FAILURES:
                return DreamCarryStepResultV1(
                    run_id=request.run_id, correlation_id=request.correlation_id, step=request.step,
                    status="terminal", reason=f"{reason} x{carry.MAX_REPLY_FAILURES}",
                )
            return DreamCarryStepResultV1(
                run_id=request.run_id, correlation_id=request.correlation_id, step=request.step,
                status="retry", reason=reason, retry_after_sec=carry.RETRY_AFTER_SEC,
            )

    def _spawn(self, envelope: BaseEnvelope) -> None:
        task = asyncio.create_task(self._safe_handle(envelope))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _safe_handle(self, envelope: BaseEnvelope) -> None:
        try:
            await self.handle(envelope)
        except Exception:
            logger.warning("dream_carry_message_failed corr=%s", envelope.correlation_id, exc_info=True)

    async def _run(self) -> None:
        while True:
            bus = self.bus_factory(self.bus_url)
            try:
                await bus.connect()
                self.bus = bus
                async with bus.subscribe(DREAM_CARRY_STEP_CHANNEL) as pubsub:
                    logger.info("dream_carry_listening channel=%s", DREAM_CARRY_STEP_CHANNEL)
                    async for raw in bus.iter_messages(pubsub):
                        try:
                            decoded = bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                self._spawn(decoded.envelope)
                        except Exception:
                            logger.warning("dream_carry_decode_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("dream_carry_reconnecting", exc_info=True)
            finally:
                with suppress(Exception):
                    await bus.close()
            await asyncio.sleep(1)

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="dream-carry")

    async def stop(self) -> None:
        tasks = [t for t in (self.task, *self._inflight) if t]
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await t
        self.task = None


def _corr(value: Optional[str]) -> Any:
    from uuid import UUID, uuid4
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return uuid4()


def _invalid_result(payload: Any, exc: Exception) -> Optional[DreamCarryStepResultV1]:
    """A terminal answer for a request we cannot run, when it names enough to be echoed."""
    if not isinstance(payload, dict):
        return None
    run_id, corr, step = payload.get("run_id"), payload.get("correlation_id"), payload.get("step")
    if not (isinstance(run_id, str) and isinstance(corr, str) and step in ("text", "finish")):
        return None
    return DreamCarryStepResultV1(run_id=run_id, correlation_id=corr, step=step, status="terminal",
                                  reason=f"invalid_request: {str(exc)[:240]}")


def _already_recorded_factory(postgres_uri: str) -> carry.AlreadyRecorded:
    """dreams row exists for this dream_id? sql-writer stores it in metrics._dream_audit.dream_id."""
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_uri, pool_pre_ping=True, pool_size=1, max_overflow=1)
    # Unindexed JSONB lookup: a full scan of dreams, fine at its size (tens of rows a month).
    query = text("SELECT 1 FROM dreams WHERE metrics->'_dream_audit'->>'dream_id' = :id LIMIT 1")

    def _check(dream_id: str) -> bool:
        with engine.connect() as conn:
            return conn.execute(query, {"id": dream_id}).first() is not None

    async def already_recorded(dream_id: str) -> bool:
        return await asyncio.to_thread(_check, dream_id)

    return already_recorded


def build_carry_listener() -> DreamCarryListener:
    from orion.core.bus.bus_schemas import ServiceRef

    from app.settings import settings

    return DreamCarryListener(
        bus_url=settings.ORION_BUS_URL,
        source=ServiceRef(name="orion-dream", version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        dream_log_channel=settings.CHANNEL_DREAM_LOG,
        already_recorded=_already_recorded_factory(settings.POSTGRES_URI),
    )
