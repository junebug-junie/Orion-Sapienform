# services/orion-spark-introspector/app/worker.py
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, Envelope
from orion.core.bus.codec import OrionCodec
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.spark import SparkTelemetryPayload, SparkStateSnapshotV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1

from orion.spark.orion_tissue import OrionTissue
from orion.spark.signal_mapper import SignalMapper
from orion.spark.surface_encoding import SurfaceEncoding

from .settings import settings
from .conn_manager import manager

logger = logging.getLogger("orion-spark-introspector")

_pub_bus: Optional[OrionBusAsync] = None

# Restart semantics: new UUID per producer (service) boot.
_PRODUCER_BOOT_ID = str(uuid4())
_SEQ: int = 0
_ACTIVE_SIGNALS: List[Dict[str, Any]] = []


def set_publisher_bus(bus: OrionBusAsync):
    global _pub_bus
    _pub_bus = bus


TISSUE = OrionTissue(
    snapshot_path=Path(settings.orion_tissue_snapshot_path) if settings.orion_tissue_snapshot_path else None
)
MAPPER = SignalMapper(TISSUE.H, TISSUE.W, TISSUE.C)


class SparkTelemetryEnvelope(Envelope[SparkTelemetryPayload]):
    kind: str = Field("spark.telemetry", frozen=True)


class SparkStateSnapshotEnvelope(Envelope[SparkStateSnapshotV1]):
    kind: str = Field("spark.state.snapshot.v1", frozen=True)


class SparkCandidatePayload(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "brain"
    prompt: str
    response: str
    spark_meta: Dict[str, Any] = Field(default_factory=dict)
    introspection: Optional[str] = None


def _register_signal(signal: SparkSignalV1) -> None:
    expires_at = time.time() + (float(signal.ttl_ms) / 1000.0)
    _ACTIVE_SIGNALS.append(
        {
            "intensity": float(signal.intensity),
            "valence_delta": float(signal.valence_delta or 0.0),
            "arousal_delta": float(signal.arousal_delta or 0.0),
            "coherence_delta": float(signal.coherence_delta or 0.0),
            "novelty_delta": float(signal.novelty_delta or 0.0),
            "expires_at": expires_at,
        }
    )


def _apply_signal_deltas(phi_stats: Dict[str, float]) -> Dict[str, float]:
    now = time.time()
    active = [s for s in _ACTIVE_SIGNALS if s.get("expires_at", 0) > now]
    # prune expired
    _ACTIVE_SIGNALS[:] = active
    if not active:
        return phi_stats

    total_delta = {"valence": 0.0, "arousal": 0.0, "coherence": 0.0, "novelty": 0.0}
    for sig in active:
        total_delta["valence"] += sig["valence_delta"]
        total_delta["arousal"] += sig["arousal_delta"]
        total_delta["coherence"] += sig["coherence_delta"]
        total_delta["novelty"] += sig["novelty_delta"]

    adjusted = dict(phi_stats)
    adjusted["valence"] = float(phi_stats.get("valence", 0.0) + total_delta["valence"])
    adjusted["energy"] = float(phi_stats.get("energy", 0.0))
    adjusted["coherence"] = float(phi_stats.get("coherence", 0.0) + total_delta["coherence"])
    adjusted["novelty"] = float(phi_stats.get("novelty", 0.0) + total_delta["novelty"])
    return adjusted


def _coerce_epoch_ts(v: Any) -> float:
    if v is None:
        return time.time()
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        dt = v
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            pass
        try:
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return time.time()
    return time.time()


def _to_iso_utc(ts_epoch: float) -> str:
    return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat()


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
    payload = cortex_reply.payload if isinstance(cortex_reply.payload, dict) else {}
    step_results = payload.get("step_results") or payload.get("steps") or []
    if not step_results:
        return None
    first_step = step_results[0]
    services = first_step.get("services") or []
    if services:
        first_service = services[0]
        p = (first_service.get("payload") or {}).get("result") or {}
        text = p.get("llm_output") or p.get("text")
        return text.strip() if isinstance(text, str) else None
    outputs = first_step.get("outputs") or {}
    text = outputs.get("text") or outputs.get("llm_output")
    return text.strip() if isinstance(text, str) else None


def _next_seq() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


async def handle_trace(env: BaseEnvelope) -> None:
    """Consume cognition.trace events, update Tissue, and emit:

    1) spark.telemetry (durable log via sql-writer)
    2) spark.state.snapshot.v1 (real-time stream for state-service/UI)

    The durable row embeds the canonical snapshot under metadata["spark_state_snapshot"].
    """

    try:
        trace = CognitionTracePayload.model_validate(env.payload)

        # Priority: Trace payload > Envelope Header > Generate New (error case)
        corr_id = (trace.correlation_id or str(getattr(env, "correlation_id", "") or "")).strip()
        if not corr_id:
            corr_id = str(uuid4())
            logger.warning("Missing correlation_id in trace/envelope; generated new one: %s", corr_id)

        ts_epoch = _coerce_epoch_ts(getattr(trace, "timestamp", None))
        iso_ts = _to_iso_utc(ts_epoch)

        # --- Derive basic affect heuristics from step outcomes (placeholder until richer signal comes in) ---
        valence = 0.5
        arousal = 0.5
        success_count = sum(1 for s in trace.steps if s.status == "success")
        fail_count = sum(1 for s in trace.steps if s.status == "fail")

        if fail_count > 0:
            valence -= (0.1 * fail_count)
        else:
            valence += 0.1

        arousal += (len(trace.steps) * 0.05)
        valence = max(0.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = 0.5

        # Find spark_vector if any
        spark_vector: Optional[List[float]] = None
        for step in trace.steps:
            if step.spark_vector:
                spark_vector = step.spark_vector
                break

        # --- Tissue update ---
        wave_len = 64
        x = np.linspace(-3, 3, wave_len)
        waveform = (np.exp(-x**2) * arousal).astype(np.float32)

        feat_dim = 32
        feature_vec = np.zeros(feat_dim, dtype=np.float32)
        feature_vec[0] = float(valence)
        feature_vec[1] = float(arousal)
        feature_vec[2] = float(dominance)

        encoding = SurfaceEncoding(
            event_id=corr_id,
            modality="system",
            timestamp=float(ts_epoch),
            source=trace.source_service or "orion",
            channel_tags=["cognition", trace.mode, trace.verb],
            waveform=waveform,
            feature_vec=feature_vec,
            spark_vector=spark_vector,
            meta={
                "text_hash": str(hash(trace.final_text or "")),
                "verb": trace.verb,
                "mode": trace.mode,
                "source_node": trace.source_node or "",
            },
        )

        stimulus = MAPPER.surface_to_stimulus(encoding, magnitude=1.0)
        channel_key = trace.mode or "chat"
        embedding_vec = None
        if spark_vector:
            try:
                embedding_vec = np.array(spark_vector, dtype=np.float32)
            except Exception:
                embedding_vec = None
        if embedding_vec is None:
            embedding_vec = feature_vec

        novelty = float(TISSUE.calculate_novelty(stimulus, channel_key=channel_key))
        TISSUE.propagate(stimulus, steps=2, learning_rate=0.1, channel_key=channel_key, embedding=embedding_vec, distress=0.0)
        phi_stats = {k: float(v) for k, v in (TISSUE.phi() or {}).items()}
        phi_stats = _apply_signal_deltas(phi_stats)
        valence = float(phi_stats.get("valence", valence))
        arousal = float(phi_stats.get("energy", arousal))
        TISSUE.snapshot()

        # --- Canonical snapshot (restart-proof + ordering-safe) ---
        seq = _next_seq()
        snap = SparkStateSnapshotV1(
            source_service=settings.service_name,
            source_node=settings.node_name,
            producer_boot_id=_PRODUCER_BOOT_ID,
            seq=seq,
            snapshot_ts=datetime.now(timezone.utc),
            valid_for_ms=int(settings.spark_state_valid_for_ms),
            correlation_id=corr_id,
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            phi=phi_stats,
            valence=float(valence),
            arousal=float(arousal),
            dominance=float(dominance),
            vector_present=bool(spark_vector),
            vector_ref=None,
            metadata={
                "stimulus_summary": f"v={valence:.2f} a={arousal:.2f} vec={'yes' if spark_vector else 'no'}",
                "trace_source_service": trace.source_service,
                "trace_source_node": trace.source_node,
                "success_count": int(success_count),
                "fail_count": int(fail_count),
            },
        )

        # --- Durable telemetry row (compact columns) ---
        telem_meta = {
            # keep the legacy/visualizer-friendly shape
            "phi": phi_stats,
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "vector_present": bool(spark_vector),
            "source_service": trace.source_service,
            "source_node": trace.source_node,

            # canonical snapshot payload (typed contract) for restart-proof consumers
            "spark_state_snapshot": snap.model_dump(mode="json"),
        }

        telem = SparkTelemetryPayload(
            correlation_id=corr_id,
            phi=float(phi_stats.get("coherence", 0.0)),
            novelty=float(novelty),
            trace_mode=trace.mode,
            trace_verb=trace.verb,
            stimulus_summary=snap.metadata.get("stimulus_summary"),
            timestamp=iso_ts,
            metadata=telem_meta,
            state_snapshot=snap,
        )

        # Broadcast to Web UI
        try:
            telemetry_id = str(uuid4())
            ws_payload = {
                "type": "tissue.update",
                "telemetry_id": telemetry_id,
                "correlation_id": corr_id,
                "timestamp": iso_ts,
                "stats": {
                    "phi": telem.phi,
                    "novelty": telem.novelty,
                    "valence": valence,
                    "arousal": arousal,
                },
                "grid": [],
                "metadata": telem.metadata,
            }
            await manager.broadcast(ws_payload)
        except Exception as e:
            logger.warning("Failed to broadcast tissue update: %s", e)

        # Publish to bus
        if _pub_bus and _pub_bus.enabled:
            # 1) Durable telemetry stream (sql-writer)
            out_env = SparkTelemetryEnvelope(
                source=env.source,
                correlation_id=corr_id,
                causality_chain=env.causality_chain,
                payload=telem,
            )
            await _pub_bus.publish(settings.channel_spark_telemetry, out_env)

            # 2) Real-time snapshot stream (state-service/UI)
            snap_env = SparkStateSnapshotEnvelope(
                source=env.source,
                correlation_id=corr_id,
                causality_chain=env.causality_chain,
                payload=snap,
            )
            await _pub_bus.publish(settings.channel_spark_state_snapshot, snap_env)

            logger.info(
                "Emitted Spark telemetry + snapshot corr_id=%s coherence=%0.3f novelty=%0.3f seq=%s",
                corr_id,
                telem.phi or 0.0,
                telem.novelty or 0.0,
                seq,
            )
        else:
            logger.error("Publisher bus not connected; skipping telemetry emit")

    except Exception as e:
        logger.error("Error processing trace for tissue: %s", e, exc_info=True)


async def handle_candidate(env: BaseEnvelope) -> None:
    raw = env.payload if isinstance(env.payload, dict) else {}
    if env.kind == "legacy.message" and isinstance(raw.get("payload"), dict):
        raw = raw.get("payload")

    try:
        candidate = SparkCandidatePayload.model_validate(raw)
    except ValidationError:
        logger.warning("Skipping invalid spark candidate payload")
        return

    if candidate.introspection:
        return

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

    codec = OrionCodec()
    bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled, codec=codec)
    await bus.connect()

    introspection = "Introspection yielded no text."

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
            introspection = "Error: Introspection decode failed."
        else:
            found_text = _extract_introspection_text(decoded.envelope)
            if found_text:
                introspection = found_text

    except Exception as rpc_error:
        logger.error("Introspection RPC failed: %s", rpc_error)
        introspection = f"Introspection unavailable (RPC Error: {rpc_error})"

    # Always proceed so UI gets *something*
    if introspection:
        # Payload for both Bus and UI
        final_payload = {
            "kind": "spark_introspect",
            "trace_id": candidate.trace_id,
            "source": "spark-introspector",
            "prompt": candidate.prompt,
            "response": candidate.response,
            "introspection": introspection,
            "spark_meta": candidate.spark_meta,
        }

        # 1) Publish to Bus
        completed = BaseEnvelope(
            kind="legacy.message",
            source=env.source,
            correlation_id=env.correlation_id,
            causality_chain=env.causality_chain,
            payload=final_payload,
        )
        await bus.publish(settings.channel_spark_candidate, completed)

        # 2) Broadcast to UI with METADATA
        try:
            ws_introspection = {
                "type": "introspection.update",
                "correlation_id": candidate.trace_id,
                "text": introspection,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": candidate.spark_meta or {},
            }
            await manager.broadcast(ws_introspection)
        except Exception as ex:
            logger.warning("Failed to broadcast introspection: %s", ex)

        logger.info("[%s] Spark introspection published", candidate.trace_id)

    await bus.close()


async def handle_signal(env: BaseEnvelope) -> None:
    payload = env.payload if isinstance(env.payload, dict) else {}
    try:
        sig = SparkSignalV1.model_validate(payload)
    except ValidationError:
        logger.warning("Ignoring malformed spark.signal payload")
        return
    _register_signal(sig)
