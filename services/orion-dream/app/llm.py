"""One LLM gateway RPC for REM recombination.

Same bus pattern as orion-memory-consolidation/app/classify.py: publish an
`llm.chat.request` to the gateway intake, await one reply on a per-call channel.
Background lane, no chat fallback: a dream must never compete with a live turn.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef

from app.settings import settings


class GatewayRefused(RuntimeError):
    """The gateway answered with an error instead of text (e.g. the GPU pool shed the call).

    Raised, not returned as "", so recombination counts it as a failed call: an empty
    string would read as an unparseable answer and mark a sleep with no model work
    'completed', which advances the replay window past items nothing looked at.
    """


async def complete(
    bus: Any,
    prompt: str,
    *,
    max_tokens: int = 320,
    purpose: str = "dream_recombine",
    route: str | None = None,
    gpu_lease: dict[str, Any] | None = None,
    timeout_sec: float | None = None,
) -> str:
    """Return the gateway's text. Raises on transport/decode failure or a gateway error reply.

    `gpu_lease` (a GpuLeaseRefV1 dump) attaches the call to a durable run's existing hold
    (options.gpu_lease) instead of the gateway taking a lease of its own: the dream.carry text
    hop runs under its run's LLM hold."""
    rpc_corr = str(uuid4())
    reply_channel = f"orion:exec:result:LLMGatewayService:{rpc_corr}"
    route = route or settings.DREAM_LLM_ROUTE
    timeout = float(timeout_sec if timeout_sec is not None else settings.DREAM_LLM_TIMEOUT_SEC)
    options: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "llm_route": route,
        "llm_lane": "background",
        "allow_chat_fallback": False,
        "purpose": purpose,
        "skip_spark_candidate_publish": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "gateway_read_timeout_sec": timeout,
    }
    if gpu_lease is not None:
        options["gpu_lease"] = gpu_lease
    payload = ChatRequestPayload(
        messages=[LLMMessage(role="user", content=prompt)],
        route=route,
        options=options,
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
        timeout_sec=timeout,
    )
    decoded = bus.codec.decode(msg.get("data"))
    if not decoded.ok:
        raise RuntimeError(decoded.error)
    result = decoded.envelope.payload or {}
    text = str(result.get("content") or result.get("text") or "")
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    # Same three failure signals the gateway's own _result_error reads: raw.error, or an
    # upstream "[Error: ...]" text (e.g. backend URL unset), plus no text at all.
    if raw.get("error") or not text.strip() or text.startswith("[Error:"):
        details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
        reason = raw.get("error") or ("upstream_error" if text.strip() else "empty_content")
        raise GatewayRefused(f"{reason}:{details.get('reason') or ''}")
    return text
