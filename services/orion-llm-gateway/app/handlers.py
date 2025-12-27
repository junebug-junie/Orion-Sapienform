from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from orion.core.bus.bus_schemas import ExecutionEnvelopeV1

from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .llm_backend import (
    run_llm_chat,
    run_llm_generate,
    run_llm_exec_step,
    run_llm_embeddings,
)
from .settings import settings


async def handle_llm_request(envelope: ExecutionEnvelopeV1) -> Dict[str, Any]:
    """Typed request handler.

    Guarantees:
      - Pydantic validation happens before calling any LLM backend
      - Returns the exact legacy-compatible reply shapes used across the mesh
    """

    # ─────────────────────────────
    # CHAT
    # ─────────────────────────────
    if envelope.event == "chat":
        raw = envelope.payload.get("body", envelope.payload) or {}

        # Backwards-compat: allow prompt-only payloads
        if "messages" not in raw and "prompt" in raw:
            prompt = raw.get("prompt") or ""
            raw = {
                "model": raw.get("model"),
                "messages": [{"role": "user", "content": prompt}],
                "options": raw.get("options") or {},
                "stream": raw.get("stream", False),
                "return_json": raw.get("return_json", False),
                "trace_id": raw.get("trace_id", envelope.correlation_id),
                "user_id": raw.get("user_id"),
                "session_id": raw.get("session_id"),
                "source": raw.get("source", "brain-cortex"),
                "profile_name": raw.get("profile_name"),
            }

        body = ChatBody(**raw)

        # run_llm_chat is sync (httpx.Client); do not block the event loop.
        result = await asyncio.to_thread(run_llm_chat, body)

        if isinstance(result, dict):
            text = result.get("text") or ""
            spark_meta = result.get("spark_meta")
            raw_llm = result.get("raw")
        else:
            text = str(result)
            spark_meta = None
            raw_llm = None

        return {
            "event": "chat_result",
            "service": settings.llm_service_name,
            "correlation_id": envelope.correlation_id,
            "payload": {"text": text, "spark_meta": spark_meta, "raw": raw_llm},
        }

    # ─────────────────────────────
    # GENERATE
    # ─────────────────────────────
    if envelope.event == "generate":
        body = GenerateBody(**envelope.payload.get("body", envelope.payload))
        text = await asyncio.to_thread(run_llm_generate, body)

        return {
            "event": "generate_result",
            "service": settings.llm_service_name,
            "correlation_id": envelope.correlation_id,
            "payload": {"text": text},
        }

    # ─────────────────────────────
    # CORTEX EXEC STEP
    # ─────────────────────────────
    if envelope.event == "exec_step":
        t0 = time.time()
        body = ExecStepPayload(**envelope.payload)
        result = await asyncio.to_thread(run_llm_exec_step, body)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Legacy contract used by cortex-exec aggregation
        return {
            "trace_id": envelope.correlation_id,
            "service": envelope.service,
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "result": result,
            "artifacts": {},
            "status": "success",
        }

    # ─────────────────────────────
    # EMBEDDINGS
    # ─────────────────────────────
    if envelope.event == "embeddings":
        body = EmbeddingsBody(**envelope.payload.get("body", envelope.payload))
        data_out = await asyncio.to_thread(run_llm_embeddings, body)

        return {
            "event": "embeddings_result",
            "service": settings.llm_service_name,
            "correlation_id": envelope.correlation_id,
            "payload": data_out,
        }

    return {
        "event": "llm_gateway_error",
        "service": settings.llm_service_name,
        "correlation_id": envelope.correlation_id,
        "payload": {"error": f"Unknown event: {envelope.event}"},
    }
