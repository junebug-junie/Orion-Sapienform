from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _delta_from_turn_effect(effect: Dict[str, Any], key: str) -> Optional[float]:
    turn = effect.get("turn") if isinstance(effect, dict) else None
    if not isinstance(turn, dict):
        return None
    val = turn.get(key)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_turn_effect(payload: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("turn_effect",):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    spark_meta = payload.get("spark_meta")
    if isinstance(spark_meta, dict):
        eff = spark_meta.get("turn_effect")
        if isinstance(eff, dict):
            return eff
        if all(k in spark_meta for k in ("phi_before", "phi_post_after")):
            before = spark_meta.get("phi_before") or {}
            after = spark_meta.get("phi_post_after") or {}
            if isinstance(before, dict) and isinstance(after, dict):
                return {
                    "turn": {
                        "coherence": float(after.get("coherence", 0.0)) - float(before.get("coherence", 0.0)),
                        "valence": float(after.get("valence", 0.0)) - float(before.get("valence", 0.0)),
                        "novelty": float(after.get("novelty", 0.0)) - float(before.get("novelty", 0.0)),
                        "energy": float(after.get("energy", 0.0)) - float(before.get("energy", 0.0)),
                    }
                }
    return {}


def extract_tensions(
    *,
    envelope: BaseEnvelope,
    intake_channel: str,
    subject: str,
    model_layer: str,
    entity_id: str,
) -> List[TensionEventV1]:
    payload = envelope.payload if isinstance(envelope.payload, dict) else {}
    turn_effect = _extract_turn_effect(payload)

    ts = envelope.created_at if envelope.created_at.tzinfo else envelope.created_at.replace(tzinfo=timezone.utc)
    trace_id = None
    if isinstance(envelope.trace, dict):
        trace_id = envelope.trace.get("trace_id") or envelope.trace.get("trace")

    spark_meta = payload.get("spark_meta") if isinstance(payload.get("spark_meta"), dict) else {}
    introspection_text = payload.get("final_text") if isinstance(payload.get("final_text"), str) else None
    if not introspection_text and isinstance(spark_meta, dict):
        introspection_text = spark_meta.get("introspect_spark") or spark_meta.get("introspection")

    prov = ArtifactProvenance(
        intake_channel=intake_channel,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        evidence_text=introspection_text,
    )

    events: List[TensionEventV1] = []

    coherence_delta = _delta_from_turn_effect(turn_effect, "coherence")
    if coherence_delta is not None and coherence_delta < 0:
        events.append(
            TensionEventV1(
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.contradiction.v1",
                ts=ts,
                confidence=0.8,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:coherence", "drive:predictive"],
                magnitude=clamp01(abs(coherence_delta)),
                drive_impacts={"coherence": 1.0, "predictive": 0.65},
            )
        )

    valence_delta = _delta_from_turn_effect(turn_effect, "valence")
    if valence_delta is not None and valence_delta < 0:
        events.append(
            TensionEventV1(
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.distress.v1",
                ts=ts,
                confidence=0.75,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:relational", "drive:continuity"],
                magnitude=clamp01(abs(valence_delta)),
                drive_impacts={"relational": 0.95, "continuity": 0.55},
            )
        )

    novelty_delta = _delta_from_turn_effect(turn_effect, "novelty")
    if novelty_delta is not None and novelty_delta > 0:
        events.append(
            TensionEventV1(
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.identity_drift.v1",
                ts=ts,
                confidence=0.7,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:continuity", "drive:autonomy"],
                magnitude=clamp01(novelty_delta),
                drive_impacts={"continuity": 0.8, "autonomy": 0.6, "predictive": 0.4},
            )
        )

    energy_delta = _delta_from_turn_effect(turn_effect, "energy")
    if energy_delta is not None and energy_delta < 0:
        events.append(
            TensionEventV1(
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.cognitive_load.v1",
                ts=ts,
                confidence=0.74,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:capability", "drive:coherence"],
                magnitude=clamp01(abs(energy_delta)),
                drive_impacts={"capability": 0.9, "coherence": 0.4},
            )
        )

    return events
