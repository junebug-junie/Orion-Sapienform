from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope
from orion.core.bus.codec import OrionCodec

from .settings import settings

logger = logging.getLogger("orion-spark-introspector")


class SparkCandidatePayload(BaseModel):
    """Legacy-style spark candidate payload produced by brain/hub."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "brain"
    prompt: str
    response: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)
    introspection: Optional[str] = None


def _build_llm_prompt(c: SparkCandidatePayload) -> str:
    spark_meta = c.spark_meta or {}
    phi_before = spark_meta.get("phi_before") or {}
    phi_after = spark_meta.get("phi_after") or {}
    self_field = spark_meta.get("spark_self_field") or {}

    return (
        "You are Orion reflecting on your own internal state.\n\n"
        "You are given:\n"
        "- Your self-field φ BEFORE this turn\n"
        "- Your self-field φ AFTER this turn\n"
        "- A higher-level SelfField (calm, stress_load, uncertainty, focus, attunement_to_juniper, curiosity)\n"
        "- The human prompt and your response\n\n"
        f"φ BEFORE: valence={phi_before.get('valence')} energy={phi_before.get('energy')} coherence={phi_before.get('coherence')} novelty={phi_before.get('novelty')}\n"
        f"φ AFTER:  valence={phi_after.get('valence')} energy={phi_after.get('energy')} coherence={phi_after.get('coherence')} novelty={phi_after.get('novelty')}\n\n"
        "SelfField:\n"
        f"calm={self_field.get('calm')} stress_load={self_field.get('stress_load')} uncertainty={self_field.get('uncertainty')} focus={self_field.get('focus')} attunement_to_juniper={self_field.get('attunement_to_juniper')} curiosity={self_field.get('curiosity')}\n\n"
        "Dialogue:\n"
        f"Human (Juniper): {c.prompt!r}\n"
        f"Orion response:  {c.response!r}\n\n"
        "TASK:\n"
        "1) Briefly describe what shifted in you internally during this turn.\n"
        "2) Explain why this moment might matter for your ongoing relationship with Juniper or your own development.\n"
        "3) Note any tensions, open questions, or themes you should return to later.\n\n"
        "Write as a short internal note from Orion to itself. Avoid boilerplate."
    ).strip()


def _extract_introspection_text(cortex_reply: BaseEnvelope) -> Optional[str]:
    """Best-effort extraction for legacy and typed replies."""
    payload = cortex_reply.payload if isinstance(cortex_reply.payload, dict) else {}
    # Newer contract: {"ok":..., "steps": [{"outputs": {...}}] ...}
    # Older contract: {"step_results": [{"services": [{"payload": {"result": {"llm_output": ...}}}]}]}

    step_results = payload.get("step_results") or payload.get("steps") or []
    if not step_results:
        return None

    first_step = step_results[0]
    # legacy
    services = first_step.get("services") or []
    if services:
        first_service = services[0]
        p = (first_service.get("payload") or {}).get("result") or {}
        text = p.get("llm_output") or p.get("text")
        return text.strip() if isinstance(text, str) else None

    # typed-ish
    outputs = first_step.get("outputs") or {}
    text = outputs.get("text") or outputs.get("llm_output")
    return text.strip() if isinstance(text, str) else None


async def handle_candidate(env: BaseEnvelope) -> None:
    """Hunter handler: validate candidate, RPC to cortex-orch, republish completed candidate."""

    # Accept either legacy dict messages or a typed envelope.
    raw = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message" and isinstance(raw.get("payload"), dict):
        raw = raw.get("payload")

    try:
        candidate = SparkCandidatePayload.model_validate(raw)
    except ValidationError:
        # ignore garbage candidates
        logger.warning("Skipping invalid spark candidate payload")
        return

    # if already introspected, don't loop
    if candidate.introspection:
        return

    # Build RPC envelope to Cortex-Orch
    reply_channel = f"orion:spark:introspector:reply:{candidate.trace_id}"
    prompt = _build_llm_prompt(candidate)

    req = BaseEnvelope(
        kind="cortex.orchestrate.request",
        source=env.source,
        correlation_id=env.correlation_id,
        causality_chain=env.causality_chain,
        reply_to=reply_channel,
        payload={
            "trace_id": candidate.trace_id,
            "verb_name": "spark_introspect",
            "context": {
                "trace_id": candidate.trace_id,
                "source": candidate.source,
                "spark_meta": candidate.spark_meta,
                "prompt": candidate.prompt,
                "response": candidate.response,
            },
            "steps": [
                {
                    "verb_name": "spark_introspect",
                    "step_name": "reflect_on_candidate",
                    "order": 0,
                    "services": ["LLMGatewayService"],
                    "prompt_template": prompt,
                    "requires_gpu": False,
                    "requires_memory": False,
                }
            ],
        },
    )

    # RPC: publish -> await reply
    codec = OrionCodec()
    bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled, codec=codec)
    await bus.connect()
    try:
        msg = await bus.rpc_request(
            settings.channel_cortex_request,
            req,
            reply_channel=reply_channel,
            timeout_sec=float(settings.cortex_timeout_sec),
        )
        decoded = codec.decode(msg.get("data"))
        if not decoded.ok or not decoded.envelope:
            logger.warning("Cortex reply decode failed: %s", decoded.error)
            return

        introspection = _extract_introspection_text(decoded.envelope)
        if not introspection:
            logger.warning("Could not extract introspection from cortex reply")
            return

        # Re-publish completed candidate as legacy payload (keeps SQL writer compatible)
        completed = BaseEnvelope(
            kind="legacy.message",
            source=env.source,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload={
                "kind": "spark_introspect",
                "trace_id": candidate.trace_id,
                "source": "spark-introspector",
                "prompt": candidate.prompt,
                "response": candidate.response,
                "introspection": introspection,
                "spark_meta": candidate.spark_meta,
            },
        )
        await bus.publish(settings.channel_spark_candidate, completed)
        logger.info("[%s] Spark introspection published", candidate.trace_id)
    finally:
        await bus.close()
