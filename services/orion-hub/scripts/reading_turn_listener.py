"""Held, read-only reading turns requested by the durable runner."""

import asyncio
import logging

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.gpu_pool.client import (
    LeaseUnavailable,
    durable_run_holder,
    validate_hold_ref,
)
from orion.schemas.reading_turn import (
    READING_TURN_CHANNEL,
    READING_TURN_REPLY_PREFIX,
    READING_TURN_REQUEST_KIND,
    READING_TURN_RESULT_KIND,
    ReadingTurnRequestV1,
    ReadingTurnResultV1,
)
from orion.world_pulse_read.read_evidence import parse_source_fetches
from scripts.world_pulse_read_pipeline import (
    _reason_from_non_final_frame,
    _turn_payload,
)

logger = logging.getLogger(__name__)


class ReadingTurnListener:
    def __init__(self, source_ref, step_relay_provider=None):
        self.source_ref = source_ref
        self.step_relay_provider = step_relay_provider
        self.bus = None
        self.rpc_bus = None
        self.task = None
        self.turns = {}
        self.handlers = set()

    async def start(self, bus, rpc_bus=None):
        self.bus = bus
        self.rpc_bus = rpc_bus or bus
        self.task = asyncio.create_task(self._run(), name="hub-reading-turns")

    async def stop(self):
        tasks = [
            task for task in [self.task, *self.turns.values(), *self.handlers] if task
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.turns.clear()
        self.handlers.clear()
        self.task = None

    async def _run(self):
        while True:
            try:
                async with self.bus.subscribe(READING_TURN_CHANNEL) as pubsub:
                    async for raw in self.bus.iter_messages(pubsub):
                        try:
                            decoded = self.bus.codec.decode(raw.get("data"))
                            if decoded.ok:
                                task = asyncio.create_task(
                                    self._handle_safely(decoded.envelope)
                                )
                                self.handlers.add(task)
                                task.add_done_callback(self.handlers.discard)
                        except Exception:
                            logger.warning("reading_turn_rpc_failed", exc_info=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("reading_turn_listener_reconnecting", exc_info=True)
                await asyncio.sleep(1)

    async def _handle_safely(self, envelope):
        try:
            await self.handle(envelope)
        except Exception:
            logger.warning("reading_turn_rpc_failed", exc_info=True)

    async def handle(self, envelope):
        reply = f"{READING_TURN_REPLY_PREFIX}:{envelope.correlation_id}"
        if envelope.kind != READING_TURN_REQUEST_KIND or envelope.reply_to != reply:
            return
        request = ReadingTurnRequestV1.model_validate(envelope.payload)
        if request.correlation_id != str(envelope.correlation_id):
            raise ValueError("reading turn correlation mismatch")
        try:
            await validate_hold_ref(
                self.bus,
                request.gpu_lease,
                source="orion-hub",
                expected_holder=durable_run_holder(request.run_id),
            )
        except LeaseUnavailable as exc:
            result = ReadingTurnResultV1(
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                ok=False,
                error=f"turn_deferred:reading_admission:{exc.reason}",
            )
            await self.bus.publish(
                reply,
                BaseEnvelope(
                    kind=READING_TURN_RESULT_KIND,
                    correlation_id=envelope.correlation_id,
                    source=self.source_ref,
                    payload=result.model_dump(mode="json"),
                ),
            )
            return
        key = (
            request.run_id,
            request.correlation_id,
            request.gpu_lease.lease_id,
            request.gpu_lease.generation,
        )
        # Keep settled replies long enough for duplicate RPC delivery, without
        # retaining an unbounded prompt/result history in Hub memory.
        for old in list(self.turns):
            if len(self.turns) < 128:
                break
            if self.turns[old].done():
                del self.turns[old]
        task = self.turns.get(key)
        if task is None:
            task = asyncio.create_task(self._execute(request))
            self.turns[key] = task
        result = await asyncio.shield(task)
        await self.bus.publish(
            reply,
            BaseEnvelope(
                kind=READING_TURN_RESULT_KIND,
                correlation_id=envelope.correlation_id,
                source=self.source_ref,
                payload=result.model_dump(mode="json"),
            ),
        )

    async def _execute(self, request):
        from orion.cognition.cortex_payload_extract import looks_like_error_text
        from orion.hub.turn_orchestrator import execute_unified_turn

        brief = request.brief

        def failure(reason):
            return ReadingTurnResultV1(
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                ok=False,
                error=reason,
            )

        if self.bus is None:
            return failure("bus_unavailable")
        payload = _turn_payload(
            "world_pulse_read" if brief.stage == 1 else "world_pulse_read_stage2",
            brief.fcc_model_label,
        )
        payload["gpu_lease"] = request.gpu_lease.model_dump(mode="json")
        try:
            frames = await asyncio.wait_for(
                execute_unified_turn(
                    reading_only=True,
                    bus=self.bus,
                    correlation_id=request.correlation_id,
                    session_id=brief.session_id,
                    user_message=brief.prompt,
                    # "<source title> — <stage-1 claim>", set by the pipeline on
                    # the stored brief: recall searches that, not the prompt.
                    retrieval_query=brief.retrieval_query,
                    payload=payload,
                    continuity_messages=None,
                    harness_rpc_bus=self.rpc_bus,
                    harness_step_relay=self.step_relay_provider()
                    if self.step_relay_provider
                    else None,
                    harness_step_queue=None,
                ),
                timeout=brief.timeout_sec,
            )
            final = next(
                (f for f in frames if isinstance(f, dict) and f.get("type") == "final"),
                None,
            )
            if final is None:
                return failure(_reason_from_non_final_frame(frames))
            text = str(final.get("llm_response") or "").strip()
            if not text:
                return failure("blank_final_response")
            if looks_like_error_text(text):
                return failure("looks_like_error_text")
            return ReadingTurnResultV1(
                run_id=request.run_id,
                correlation_id=request.correlation_id,
                ok=True,
                text=text,
                source_fetches=parse_source_fetches(
                    final.get("harness_source_fetches")
                ),
            )
        except TimeoutError:
            return failure(f"stage{brief.stage}_turn_timeout")
        except Exception as exc:
            logger.warning(
                "reading_turn_failed run=%s corr=%s",
                request.run_id,
                request.correlation_id,
                exc_info=True,
            )
            return failure(
                f"turn_exception:{str(exc)[:160]}" if str(exc) else "turn_exception"
            )
