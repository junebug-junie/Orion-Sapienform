import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, ChatRequestPayload, LLMMessage
from orion.schemas.vision import (
    VisionWindowPayload,
    VisionEventPayload,
    VisionEventBundleItem,
    VisionCouncilRequestPayload,
    VisionCouncilResultPayload
)

# Check if CortexChatRequest is available, otherwise assume dict for now to avoid cross-service import issues if not in schemas yet
try:
    from orion.schemas.cortex.contracts import CortexChatRequest, CortexChatResult
except ImportError:
    CortexChatRequest = None
    logger.warning("Cortex schemas not found, using dicts")

from .settings import Settings

settings = Settings()

class CouncilService:
    def __init__(self):
        self.bus = OrionBusAsync(url=settings.ORION_BUS_URL)
        self._rpc_bus: OrionBusAsync | None = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._rpc_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

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

        event_payload = await self._generate_events(req.window, env)
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

        event_payload = await self._generate_events(payload, env)
        if not event_payload:
            return

        out_env = env.derive_child(
            kind="vision.event.bundle",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=event_payload
        )

        await self.bus.publish(settings.CHANNEL_COUNCIL_PUB, out_env)
        logger.info(f"[COUNCIL] Published {len(event_payload.events)} events")

    async def _generate_events(self, window: VisionWindowPayload, source_env: BaseEnvelope) -> Optional[VisionEventPayload]:
        # Prepare prompt for LLM
        summary = window.summary
        prompt = f"""
        Analyze this visual window summary (30s):
        - Objects: {summary.get('top_labels')}
        - Captions: {summary.get('captions')}

        Identify key events. Return JSON list of events with fields:
        - event_type (str)
        - narrative (str)
        - entities (list[str])
        - tags (list[str])
        - confidence (float 0-1)
        - salience (float 0-1)
        """

        events_data = await self._call_llm(prompt, source_env)
        if not events_data:
            return None

        bundle_items = []
        for evt in events_data:
            item = VisionEventBundleItem(
                event_id=str(uuid.uuid4()),
                event_type=evt.get("event_type", "unknown"),
                narrative=evt.get("narrative", ""),
                entities=evt.get("entities", []),
                tags=evt.get("tags", []),
                confidence=float(evt.get("confidence", 0.5)),
                salience=float(evt.get("salience", 0.5)),
                evidence_refs=window.artifact_ids
            )
            bundle_items.append(item)

        return VisionEventPayload(events=bundle_items)

    async def _call_llm(self, prompt: str, source_env: BaseEnvelope) -> List[Dict[str, Any]]:
        # RPC to LLM Gateway
        req_id = str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_LLM_REPLY_PREFIX}:{req_id}"

        chat_request = ChatRequestPayload(
            model=settings.COUNCIL_MODEL,
            messages=[
                LLMMessage(role="system", content="You are a visual analysis AI. Output strict JSON."),
                LLMMessage(role="user", content=prompt)
            ],
            options={"return_json": True}
        )

        envelope = source_env.derive_child(
            kind="llm.chat.request",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION),
            payload=chat_request,
            reply_to=reply_to
        )

        if self._rpc_bus is None:
            logger.error("[COUNCIL] RPC bus not initialized; cannot call LLM gateway")
            return []

        try:
            reply = await self._rpc_bus.rpc_request(
                settings.CHANNEL_LLM_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=30.0
            )

            decoded = self._rpc_bus.codec.decode(reply.get("data"))
            if not decoded.ok:
                logger.error(f"[COUNCIL] LLM decode error: {decoded.error}")
                return []

            res_env = decoded.envelope

            content = ""
            if isinstance(res_env.payload, dict):
                 if "content" in res_env.payload:
                     content = res_env.payload["content"]
                 elif "choices" in res_env.payload:
                     content = res_env.payload["choices"][0]["message"]["content"]

            if not content:
                logger.warning(f"[COUNCIL] Empty LLM response: {res_env.payload}")
                return []

            try:
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]

                data = json.loads(content)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "events" in data:
                    return data["events"]
                return [data] if isinstance(data, dict) else []
            except Exception as e:
                logger.error(f"[COUNCIL] JSON parse error: {e} | Content: {content}")
                return []

        except TimeoutError:
            logger.error("[COUNCIL] LLM timeout")
            return []
        except Exception as e:
            logger.error(f"[COUNCIL] LLM error: {e}")
            return []

service = CouncilService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await service.start()
    yield
    await service.stop()

app = FastAPI(title="Orion Vision Council", version="0.1.0", lifespan=lifespan)
