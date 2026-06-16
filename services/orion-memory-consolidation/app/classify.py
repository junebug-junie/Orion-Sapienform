from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.memory.consolidation_classify import build_classify_prompt
from orion.schemas.memory_consolidation import MemoryTurnPersistedV1

from app.boundary import scores_from_llm_result

logger = logging.getLogger(__name__)


async def classify_turn(bus: OrionBusAsync, *, turn: MemoryTurnPersistedV1, settings) -> dict:
    prompt = build_classify_prompt(prompt=turn.prompt, response=turn.response, spark_meta=turn.spark_meta)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            rpc_corr = str(uuid4())
            reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
            payload = ChatRequestPayload(
                messages=[LLMMessage(role="user", content=prompt)],
                route="quick",
                options={
                    "return_logprobs": True,
                    "logprobs_top_k": 2,
                    "logprob_summary_only": False,
                    "max_tokens": 8,
                    "llm_route": "quick",
                },
            )
            env = BaseEnvelope(
                kind="llm.chat.request",
                source=ServiceRef(
                    name=settings.SERVICE_NAME,
                    version=settings.SERVICE_VERSION,
                    node=settings.NODE_NAME,
                ),
                correlation_id=rpc_corr,
                reply_to=reply_channel,
                payload=payload.model_dump(mode="json"),
            )
            msg = await bus.rpc_request(
                settings.CHANNEL_LLM_INTAKE,
                env,
                reply_channel=reply_channel,
                timeout_sec=float(settings.MEMORY_CLASSIFY_TIMEOUT_SEC),
            )
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                raise RuntimeError(decoded.error)
            result_payload = decoded.envelope.payload
            content = str(result_payload.get("content") or result_payload.get("text") or "")
            raw = result_payload.get("raw") if isinstance(result_payload.get("raw"), dict) else {}
            mem_score, bnd_score = scores_from_llm_result(content, raw)
            return {
                "memory_significance_score": mem_score,
                "conversation_boundary_score": bnd_score,
                "memory_classify_status": "ok",
                "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            last_error = exc
            logger.warning("memory_classify_attempt_failed attempt=%s corr=%s err=%s", attempt + 1, turn.correlation_id, exc)
            await asyncio.sleep(0.2)
    logger.error("memory_classify_degraded corr=%s err=%s", turn.correlation_id, last_error)
    return {
        "memory_classify_status": "degraded",
        "memory_classify_ts": datetime.now(timezone.utc).isoformat(),
    }
