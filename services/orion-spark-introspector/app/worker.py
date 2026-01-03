from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope
from orion.core.bus.codec import OrionCodec
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.spark import SparkTelemetryPayload

from orion.spark.orion_tissue import OrionTissue
from orion.spark.surface_encoding import SurfaceEncoding
from orion.spark.signal_mapper import SignalMapper

from .settings import settings
import time
import numpy as np

logger = logging.getLogger("orion-spark-introspector")

# Global publisher bus (set by main.py)
_pub_bus: Optional[OrionBusAsync] = None

def set_publisher_bus(bus: OrionBusAsync):
    global _pub_bus
    _pub_bus = bus

# --- Tissue State (Persistent in Memory for Worker) ---
# We initialize it once. In a real persistent service, this should be properly singleton-managed.
# The class loads from disk on init.
from pathlib import Path
TISSUE = OrionTissue(snapshot_path=Path(settings.orion_tissue_snapshot_path) if settings.orion_tissue_snapshot_path else None)
MAPPER = SignalMapper(TISSUE.H, TISSUE.W, TISSUE.C)

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


async def handle_trace(env: BaseEnvelope) -> None:
    """
    Consumes CognitionTrace, updates Tissue, emits SparkTelemetry.
    """
    try:
        # 1. Decode Payload
        trace = CognitionTracePayload.model_validate(env.payload)

        # 2. Derive Surface Encoding
        # Simple heuristic: map verb/status to valence/arousal
        # This is "Dumb" mapping for now, mirroring `orion/spark/strategies.py` logic if we imported it.
        # We construct a synthetic encoding based on trace success/fail/complexity.

        valence = 0.5
        arousal = 0.5

        # Heuristic: Failures drop valence
        success_count = sum(1 for s in trace.steps if s.status == "success")
        fail_count = sum(1 for s in trace.steps if s.status == "fail")

        if fail_count > 0:
            valence -= (0.1 * fail_count)
        else:
            valence += 0.1

        # Heuristic: More steps = higher arousal (energy)
        arousal += (len(trace.steps) * 0.05)

        # Clamp
        valence = max(0.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))

        encoding = SurfaceEncoding(
            valence=valence,
            arousal=arousal,
            dominance=0.5, # Neutral
            text_hash=hash(trace.final_text or "")
        )

        # 3. Calculate Novelty (Predictive Coding)
        # We need to map the encoding to a stimulus first
        stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
        novelty = TISSUE.calculate_novelty(stimulus)

        # 4. Propagate (Update Tissue)
        # We inject the stimulus and evolve the field
        # "Inject Surface" does step() internally.
        # But we want to separate learning (expectation) from physics if possible,
        # but OrionTissue.propagate combines them.
        # Let's use propagate to update expectation + tissue.
        TISSUE.propagate(stimulus, steps=2, learning_rate=0.1)

        # 5. Compute Phi (Self State)
        phi_stats = TISSUE.phi()

        # 6. Snapshot
        TISSUE.snapshot()

        # 7. Publish Telemetry
        telem = SparkTelemetryPayload(
            correlation_id=trace.correlation_id,
            phi=phi_stats.get("coherence", 0.0), # Using coherence as proxy for 'phi' scalar if needed, or novelty?
            # actually spark schema asks for 'phi' as float. Usually phi is integrated information, or coherence.
            # Let's map phi -> coherence for now as per "SelfField" logic in spark_engine.
            novelty=novelty,
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            stimulus_summary=f"v={valence:.2f} a={arousal:.2f}",
            timestamp=time.time(),
            metadata=phi_stats # Include full phi stats in metadata
        )

        # Publish via shared bus
        if _pub_bus and _pub_bus.is_connected:
            out_env = BaseEnvelope(
                kind="spark.introspection.log", # Aligned with SQL Writer map
                source=env.source,
                correlation_id=env.correlation_id,
                causality_chain=env.causality_chain,
                payload=telem
            )
            await _pub_bus.publish(settings.channel_spark_telemetry, out_env)
            logger.info(f"Tissue updated (novelty={novelty:.3f}, phi={telem.phi:.3f}) for trace {trace.correlation_id}")
        else:
            logger.error("Publisher bus not connected; skipping telemetry emit")

    except Exception as e:
        logger.error(f"Error processing trace for tissue: {e}", exc_info=True)


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
