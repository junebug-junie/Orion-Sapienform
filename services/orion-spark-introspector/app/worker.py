from __future__ import annotations

import logging
from uuid import uuid4
from typing import Any, Dict, Optional
import time
import numpy as np
from pathlib import Path

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

logger = logging.getLogger("orion-spark-introspector")

# Global publisher bus (set by main.py)
_pub_bus: Optional[OrionBusAsync] = None

def set_publisher_bus(bus: OrionBusAsync):
    global _pub_bus
    _pub_bus = bus

# --- Tissue State (Persistent in Memory for Worker) ---
TISSUE = OrionTissue(snapshot_path=Path(settings.orion_tissue_snapshot_path) if settings.orion_tissue_snapshot_path else None)
MAPPER = SignalMapper(TISSUE.H, TISSUE.W, TISSUE.C)

# --- Typed Envelopes ---
class SparkTelemetryEnvelope(Envelope[SparkTelemetryPayload]):
    """Typed contract for Spark Telemetry logs."""
    kind: str = Field("spark.telemetry", frozen=True)

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

        # 2. Derive Heuristics (Valence/Arousal)
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
        dominance = 0.5

        # 3. Construct Proper SurfaceEncoding
        # Waveform: Simple activation bump scaled by arousal
        wave_len = 64
        x = np.linspace(-3, 3, wave_len)
        waveform = np.exp(-x**2) * arousal 

        # Feature vector: [valence, arousal, dominance, ...padding]
        feat_dim = 32
        feature_vec = np.zeros(feat_dim, dtype=np.float32)
        feature_vec[0] = valence
        feature_vec[1] = arousal
        feature_vec[2] = dominance

        encoding = SurfaceEncoding(
            event_id=str(trace.correlation_id),
            modality="system",
            timestamp=time.time(),
            source=trace.source_service or "orion",
            channel_tags=["cognition", trace.mode, trace.verb],
            waveform=waveform.astype(np.float32),
            feature_vec=feature_vec.astype(np.float32),
            meta={
                "text_hash": str(hash(trace.final_text or "")),
                "verb": trace.verb
            }
        )

        # 4. Calculate Novelty (Predictive Coding)
        stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
        novelty = TISSUE.calculate_novelty(stimulus)

        # 5. Propagate (Update Tissue)
        TISSUE.propagate(stimulus, steps=2, learning_rate=0.1)
        phi_stats = TISSUE.phi()
        TISSUE.snapshot()

        # 6. Publish Telemetry
        telem = SparkTelemetryPayload(
            correlation_id=trace.correlation_id,
            phi=phi_stats.get("coherence", 0.0), 
            novelty=novelty,
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            stimulus_summary=f"v={valence:.2f} a={arousal:.2f}",
            timestamp=time.time(),
            metadata=phi_stats
        )

        # Publish via shared bus
        if _pub_bus and _pub_bus.enabled:
            # FIX: Use Typed Envelope
            out_env = SparkTelemetryEnvelope(
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
