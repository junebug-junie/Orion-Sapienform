import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from loguru import logger

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionWindowPayload, VisionEventPayload, VisionEventBundleItem
# Check if CortexChatRequest is available, otherwise assume dict for now to avoid cross-service import issues if not in schemas yet
# Memory said: "The CortexChatRequest schema is canonically defined in orion/schemas/cortex/contracts.py"
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
        self._consumer_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        logger.remove()
        logger.add(lambda m: print(m, end=""), level=settings.LOG_LEVEL)

        await self.bus.connect()
        self._consumer_task = asyncio.create_task(self._consume())
        logger.info(f"[COUNCIL] Started. Listening on {settings.CHANNEL_COUNCIL_INTAKE}")

    async def stop(self):
        self._shutdown_event.set()
        if self._consumer_task:
            try:
                self._consumer_task.cancel()
                await self._consumer_task
            except asyncio.CancelledError:
                pass
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
                asyncio.create_task(self._process_window(env))

    async def _process_window(self, env: BaseEnvelope):
        try:
            if isinstance(env.payload, dict):
                payload = VisionWindowPayload(**env.payload)
            else:
                payload = env.payload
        except Exception as e:
            logger.error(f"[COUNCIL] Invalid payload: {e}")
            return

        # Prepare prompt for LLM
        summary = payload.summary
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

        # Call LLM via RPC
        events = await self._call_llm(prompt)

        if not events:
            return

        # Bundle events
        bundle_items = []
        for evt in events:
            # Map to schema
            item = VisionEventBundleItem(
                event_id=str(uuid.uuid4()),
                event_type=evt.get("event_type", "unknown"),
                narrative=evt.get("narrative", ""),
                entities=evt.get("entities", []),
                tags=evt.get("tags", []),
                confidence=float(evt.get("confidence", 0.5)),
                salience=float(evt.get("salience", 0.5)),
                evidence_refs=payload.artifact_ids # Link back to artifacts
            )
            bundle_items.append(item)

        event_payload = VisionEventPayload(events=bundle_items)

        out_env = BaseEnvelope(
            schema_id="vision.event.bundle",
            schema_version="1.0.0",
            kind="vision.event.bundle",
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=env.correlation_id or str(uuid.uuid4()),
            causality_chain=env.causality_chain + [env.correlation_id] if env.correlation_id else [],
            payload=event_payload
        )

        await self.bus.publish(settings.CHANNEL_COUNCIL_PUB, out_env)
        logger.info(f"[COUNCIL] Published {len(bundle_items)} events")

    async def _call_llm(self, prompt: str) -> List[Dict[str, Any]]:
        # RPC to LLM Gateway
        # Using dict payload if schema not available or simple construction

        req_id = str(uuid.uuid4())
        reply_to = f"{settings.CHANNEL_LLM_REPLY_PREFIX}:{req_id}"

        # Build payload based on contracts.md or memory
        # Assuming "messages" list format
        chat_request = {
            "model": settings.COUNCIL_MODEL,
            "messages": [
                {"role": "system", "content": "You are a visual analysis AI. Output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "json_mode": True
        }

        # If we have the typed schema, use it, but for now dict is safer if we are unsure of imports

        envelope = BaseEnvelope(
            schema_id="cortex.chat.request",
            schema_version="1.0.0",
            kind="llm.chat.request", # From memory: orion-exec:request:LLMGatewayService (llm.chat.request)
            source=f"{settings.SERVICE_NAME}:{settings.SERVICE_VERSION}",
            correlation_id=req_id,
            reply_to=reply_to,
            payload=chat_request # Codec handles dicts if they match schema roughly or if relaxed
        )

        try:
            # We use rpc_request
            reply = await self.bus.rpc_request(
                settings.CHANNEL_LLM_REQUEST,
                envelope,
                reply_channel=reply_to,
                timeout_sec=30.0
            )

            decoded = self.bus.codec.decode(reply.get("data"))
            if not decoded.ok:
                logger.error(f"[COUNCIL] LLM decode error: {decoded.error}")
                return []

            res_env = decoded.envelope
            # Parse result. Assuming standard chat result structure
            # result -> choices[0] -> message -> content
            # But we need to see what standard result schema is.
            # Assuming it returns something we can parse as JSON.

            content = ""
            if isinstance(res_env.payload, dict):
                 # Try to find content
                 if "content" in res_env.payload:
                     content = res_env.payload["content"]
                 elif "choices" in res_env.payload:
                     content = res_env.payload["choices"][0]["message"]["content"]

            if not content:
                logger.warning(f"[COUNCIL] Empty LLM response: {res_env.payload}")
                return []

            # Parse JSON from content
            try:
                # content might be wrapped in ```json ... ```
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
app = FastAPI(title="Orion Vision Council", version="0.1.0", lifespan=None)

@app.on_event("startup")
async def startup():
    await service.start()

@app.on_event("shutdown")
async def shutdown():
    await service.stop()
