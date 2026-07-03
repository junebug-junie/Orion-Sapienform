import asyncio
import uuid
from typing import Any, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.llm.openai_message_content import join_openai_message_content
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, ChatRequestPayload, LLMMessage
from orion.schemas.vision import (
    VisionWindowPayload,
    VisionEventPayload,
    VisionCouncilRequestPayload,
    VisionCouncilResultPayload,
    VisionSceneInterpretationV1,
)

# Check if CortexChatRequest is available, otherwise assume dict for now to avoid cross-service import issues if not in schemas yet
try:
    from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
except ImportError:
    CortexChatRequest = None
    logger.warning("Cortex schemas not found, using dicts")

from .evidence_grounding import (
    build_person_presence_fallback,
    edge_person_hits,
    enforce_evidence_grounding,
)
from .interpretation import (
    InterpretationParseOutcome,
    build_interpretation_llm_options,
    build_interpretation_prompt,
    parse_llm_content,
    project_interpretation_to_events,
)

_MAX_DEBUG_INTERPRETATIONS = 20
from .settings import Settings

settings = Settings()


def _extract_chat_result_text(payload: Any) -> str:
    if payload is None:
        return ""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        return join_openai_message_content(payload)

    for key in ("content", "text"):
        text = join_openai_message_content(payload.get(key))
        if text:
            return text

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = join_openai_message_content(msg.get("content"))
        if text:
            return text
        text = join_openai_message_content(first.get("text"))
        if text:
            return text

    raw = payload.get("raw")
    if isinstance(raw, dict):
        raw_choices = raw.get("choices")
        if isinstance(raw_choices, list) and raw_choices:
            first = raw_choices[0] if isinstance(raw_choices[0], dict) else {}
            msg = first.get("message") if isinstance(first.get("message"), dict) else {}
            text = join_openai_message_content(msg.get("content"))
            if text:
                return text

    return ""


class CouncilService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._rpc_bus: OrionBusAsync | None = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._recent_interpretations: list[dict] = []
        self._llm_semaphore = asyncio.Semaphore(1)

    def _record_interpretation(
        self,
        interpretation: VisionSceneInterpretationV1,
        outcome: InterpretationParseOutcome,
    ) -> None:
        record = interpretation.model_dump()
        record["parse_mode"] = outcome.parse_mode
        if outcome.salvage_warnings:
            record["salvage_warnings"] = list(outcome.salvage_warnings)
        self._recent_interpretations.append(record)
        if len(self._recent_interpretations) > _MAX_DEBUG_INTERPRETATIONS:
            self._recent_interpretations = self._recent_interpretations[-_MAX_DEBUG_INTERPRETATIONS:]

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        await self.bus.connect()
        from orion.core.bus.rpc_fork import fork_rpc_client

        self._rpc_bus = await fork_rpc_client(self.bus)
        self._consumer_task = asyncio.create_task(self._consume())
        self._rpc_task = asyncio.create_task(self._consume_rpc())
        logger.info(f"[COUNCIL] Started. Listening on {settings.CHANNEL_COUNCIL_INTAKE} and {settings.CHANNEL_COUNCIL_REQUEST}")

    async def stop(self):
        self._shutdown_event.set()
        for t in [self._consumer_task, self._rpc_task]:
            if t:
                try:
                    t.cancel()
                    await t
                except asyncio.CancelledError:
                    pass
        if self._rpc_bus is not None:
            await self._rpc_bus.close()
            self._rpc_bus = None
        await self.bus.close()

    async def _consume(self):
        async with self.bus.subscribe(settings.CHANNEL_COUNCIL_INTAKE) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break

                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue

                env = decoded.envelope
                asyncio.create_task(self._process_window(env, is_rpc=False))

    async def _consume_rpc(self):
        async with self.bus.subscribe(settings.CHANNEL_COUNCIL_REQUEST) as pubsub:
             async for msg in self.bus.iter_messages(pubsub):
                if self._shutdown_event.is_set():
                    break

                data = msg.get("data")
                decoded = self.bus.codec.decode(data)
                if not decoded.ok:
                    continue

                env = decoded.envelope
                asyncio.create_task(self._handle_rpc(env))

    async def _handle_rpc(self, env: BaseEnvelope):
        try:
            if isinstance(env.payload, dict):
                req = VisionCouncilRequestPayload(**env.payload)
            else:
                req = env.payload
        except Exception as e:
            logger.error(f"[COUNCIL] RPC invalid payload: {e}")
            return

        interpretation, parse_outcome = await self._generate_interpretation(req.window, env)
        interpretation, parse_outcome = self._finalize_interpretation(
            interpretation, parse_outcome, req.window
        )
        if interpretation is not None:
            self._record_interpretation(interpretation, parse_outcome)

        event_payload = (
            self._project_interpretation_to_events(interpretation, req.window)
            if interpretation is not None
            else None
        )
        if not event_payload:
            return

        # 1. Reply to caller
        res_payload = VisionCouncilResultPayload(events=event_payload)

        reply_env = env.derive_child(
            kind="vision.council.result",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=res_payload,
            reply_to=None
        )

        if env.reply_to:
            await self.bus.publish(env.reply_to, reply_env)

        # 2. Broadcast for consistency
        broadcast_env = env.derive_child(
            kind="vision.event.bundle",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=event_payload
        )
        await self.bus.publish(settings.CHANNEL_COUNCIL_PUB, broadcast_env)

    async def _process_window(self, env: BaseEnvelope, is_rpc: bool = False):
        try:
            if isinstance(env.payload, dict):
                payload = VisionWindowPayload(**env.payload)
            else:
                payload = env.payload
        except Exception as e:
            logger.error(f"[COUNCIL] Invalid payload: {e}")
            return

        interpretation, parse_outcome = await self._generate_interpretation(payload, env)
        interpretation, parse_outcome = self._finalize_interpretation(
            interpretation, parse_outcome, payload
        )

        if interpretation is not None:
            self._record_interpretation(interpretation, parse_outcome)

        event_payload = (
            self._project_interpretation_to_events(interpretation, payload)
            if interpretation is not None
            else None
        )
        if not event_payload:
            return

        out_env = env.derive_child(
            kind="vision.event.bundle",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=event_payload
        )

        await self.bus.publish(settings.CHANNEL_COUNCIL_PUB, out_env)
        logger.info(f"[COUNCIL] Published {len(event_payload.events)} events")

    def _finalize_interpretation(
        self,
        interpretation: VisionSceneInterpretationV1 | None,
        parse_outcome: InterpretationParseOutcome,
        window: VisionWindowPayload,
    ) -> tuple[VisionSceneInterpretationV1 | None, InterpretationParseOutcome]:
        if interpretation is not None:
            interpretation, grounding_notes = enforce_evidence_grounding(interpretation, window)
            for note in grounding_notes:
                logger.info(f"[COUNCIL] grounding {note}")
            if not interpretation.event_candidates and edge_person_hits(window) > 0:
                interpretation = build_person_presence_fallback(window)
                warnings = list(parse_outcome.salvage_warnings)
                warnings.append("edge_fallback_after_grounding")
                parse_outcome = InterpretationParseOutcome(
                    interpretation=interpretation,
                    parse_mode=parse_outcome.parse_mode,
                    salvage_warnings=warnings,
                )
        elif edge_person_hits(window) > 0:
            interpretation = build_person_presence_fallback(window)
            parse_outcome = InterpretationParseOutcome(
                interpretation=interpretation,
                parse_mode="edge_fallback",
            )
        return interpretation, parse_outcome

    async def _generate_interpretation(
        self,
        window: VisionWindowPayload,
        source_env: BaseEnvelope,
    ) -> tuple[VisionSceneInterpretationV1 | None, InterpretationParseOutcome]:
        prompt = build_interpretation_prompt(window)
        content = await self._call_llm_raw(prompt, source_env)
        if not content:
            return None, InterpretationParseOutcome(interpretation=None, parse_mode="parse_failed")

        outcome = parse_llm_content(content, window)
        return outcome.interpretation, outcome

    def _project_interpretation_to_events(
        self,
        interpretation: VisionSceneInterpretationV1,
        window: VisionWindowPayload,
    ) -> VisionEventPayload | None:
        payload = project_interpretation_to_events(interpretation, window)
        if not payload.events:
            return None
        return payload

    async def _call_llm_raw(self, prompt: str, source_env: BaseEnvelope) -> str | None:
        req_id = str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_LLM_REPLY_PREFIX}:{req_id}"

        chat_request = ChatRequestPayload(
            model=settings.COUNCIL_MODEL,
            route=settings.COUNCIL_LLM_ROUTE,
            messages=[
                LLMMessage(role="system", content="You are a visual analysis AI. Output strict JSON."),
                LLMMessage(role="user", content=prompt)
            ],
            options=build_interpretation_llm_options(
                structured_output_method=settings.COUNCIL_STRUCTURED_OUTPUT_METHOD,
                max_tokens=settings.COUNCIL_LLM_MAX_TOKENS,
            ),
        )

        envelope = source_env.derive_child(
            kind="llm.chat.request",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=chat_request,
            reply_to=reply_to
        )

        if self._rpc_bus is None:
            logger.error("[COUNCIL] RPC bus not initialized; cannot call LLM gateway")
            return None

        try:
            async with self._llm_semaphore:
                reply = await self._rpc_bus.rpc_request(
                    settings.CHANNEL_LLM_REQUEST,
                    envelope,
                    reply_channel=reply_to,
                    timeout_sec=settings.COUNCIL_LLM_TIMEOUT_SEC,
                )

            decoded = self._rpc_bus.codec.decode(reply.get("data"))
            if not decoded.ok:
                logger.error(f"[COUNCIL] LLM decode error: {decoded.error}")
                return None

            res_env = decoded.envelope
            content = _extract_chat_result_text(res_env.payload)

            if not content:
                logger.warning(f"[COUNCIL] Empty LLM response: {res_env.payload}")
                return None
            if content.startswith("[Error:"):
                logger.error(f"[COUNCIL] LLM gateway error response: {content[:240]}")
                return None

            return content

        except TimeoutError:
            logger.error("[COUNCIL] LLM timeout")
            return None
        except Exception as e:
            logger.error(f"[COUNCIL] LLM error: {e}")
            return None


service = CouncilService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Council", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/debug/last-interpretation")
async def debug_last_interpretation():
    if not service._recent_interpretations:
        return {"interpretation": None}
    return {"interpretation": service._recent_interpretations[-1]}


@app.get("/debug/recent-interpretations")
async def debug_recent_interpretations(limit: int = 10):
    limit = min(max(1, limit), _MAX_DEBUG_INTERPRETATIONS)
    return {"interpretations": service._recent_interpretations[-limit:]}
