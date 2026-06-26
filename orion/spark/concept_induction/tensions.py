from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.self_state import SelfStateV1

from .dossier import build_evidence_items, build_source_event_ref, extract_trace_id, extract_turn_id


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
        nested_meta = spark_meta.get("metadata")
        if isinstance(nested_meta, dict):
            nested_eff = nested_meta.get("turn_effect")
            if isinstance(nested_eff, dict):
                return nested_eff
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


def _artifact_id(envelope: BaseEnvelope, entity_id: str, kind: str) -> str:
    material = f"{envelope.id}|{entity_id}|{kind}"
    return f"tension-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


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
    trace_id = extract_trace_id(envelope)
    turn_id = extract_turn_id(envelope)

    spark_meta = payload.get("spark_meta") if isinstance(payload.get("spark_meta"), dict) else {}
    introspection_text = payload.get("final_text") if isinstance(payload.get("final_text"), str) else None
    if not introspection_text and isinstance(spark_meta, dict):
        introspection_text = spark_meta.get("introspect_spark") or spark_meta.get("introspection")

    source_event_ref = build_source_event_ref(envelope, intake_channel)
    evidence_items = build_evidence_items(envelope, intake_channel, introspection_text)
    prov = ArtifactProvenance(
        intake_channel=intake_channel,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        turn_id=turn_id,
        evidence_text=introspection_text,
        evidence_summary=evidence_items[0].summary if evidence_items else None,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
    )

    events: List[TensionEventV1] = []

    coherence_delta = _delta_from_turn_effect(turn_effect, "coherence")
    if coherence_delta is not None and coherence_delta < 0:
        events.append(
            TensionEventV1(
                artifact_id=_artifact_id(envelope, entity_id, "tension.contradiction.v1"),
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.contradiction.v1",
                ts=ts,
                confidence=0.8,
                correlation_id=str(envelope.correlation_id),
                trace_id=str(trace_id) if trace_id else None,
                turn_id=turn_id,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:coherence", "drive:predictive", "tension.contradiction.v1"],
                magnitude=clamp01(abs(coherence_delta)),
                drive_impacts={"coherence": 1.0, "predictive": 0.65},
            )
        )

    valence_delta = _delta_from_turn_effect(turn_effect, "valence")
    if valence_delta is not None and valence_delta < 0:
        events.append(
            TensionEventV1(
                artifact_id=_artifact_id(envelope, entity_id, "tension.distress.v1"),
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.distress.v1",
                ts=ts,
                confidence=0.75,
                correlation_id=str(envelope.correlation_id),
                trace_id=str(trace_id) if trace_id else None,
                turn_id=turn_id,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:relational", "drive:continuity", "tension.distress.v1"],
                magnitude=clamp01(abs(valence_delta)),
                drive_impacts={"relational": 0.95, "continuity": 0.55},
            )
        )

    novelty_delta = _delta_from_turn_effect(turn_effect, "novelty")
    if novelty_delta is not None and novelty_delta > 0:
        events.append(
            TensionEventV1(
                artifact_id=_artifact_id(envelope, entity_id, "tension.identity_drift.v1"),
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.identity_drift.v1",
                ts=ts,
                confidence=0.7,
                correlation_id=str(envelope.correlation_id),
                trace_id=str(trace_id) if trace_id else None,
                turn_id=turn_id,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:continuity", "drive:autonomy", "tension.identity_drift.v1"],
                magnitude=clamp01(novelty_delta),
                drive_impacts={"continuity": 0.8, "autonomy": 0.6, "predictive": 0.4},
            )
        )

    energy_delta = _delta_from_turn_effect(turn_effect, "energy")
    if energy_delta is not None and energy_delta < 0:
        events.append(
            TensionEventV1(
                artifact_id=_artifact_id(envelope, entity_id, "tension.cognitive_load.v1"),
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                kind="tension.cognitive_load.v1",
                ts=ts,
                confidence=0.74,
                correlation_id=str(envelope.correlation_id),
                trace_id=str(trace_id) if trace_id else None,
                turn_id=turn_id,
                provenance=prov,
                related_nodes=["signal:turn_effect", "drive:capability", "drive:coherence", "tension.cognitive_load.v1"],
                magnitude=clamp01(abs(energy_delta)),
                drive_impacts={"capability": 0.9, "coherence": 0.4},
            )
        )

    for event in events:
        event.provenance.tension_refs = [event.artifact_id]
    return events


# Matches DriveEngine keys so pressure spread is comparable across turns.
_DRIVE_KEYS: Sequence[str] = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")

# When turn_effect-derived tensions are empty but drives disagree materially, record a named tension for audits/goals.
_PRESSURE_SPREAD_THRESHOLD = 0.06


def _canonical_pressures_for_spread(pressures: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(pressures.get(k, 0.0)) for k in _DRIVE_KEYS}
    rs = float(pressures.get("relational_stability") or 0.0)
    if rs:
        out["relational"] = max(out["relational"], rs)
    return out


def derive_pressure_competition_tensions(
    *,
    envelope: BaseEnvelope,
    intake_channel: str,
    subject: str,
    model_layer: str,
    entity_id: str,
    pressures: Dict[str, float],
) -> List[TensionEventV1]:
    """Emit at most one tension when max-min drive pressure spread exceeds threshold (no turn_effect required)."""
    canon = _canonical_pressures_for_spread(pressures)
    vals = list(canon.values())
    if len(vals) < 2:
        return []
    spread = max(vals) - min(vals)
    if spread < _PRESSURE_SPREAD_THRESHOLD:
        raw = [float(v) for v in pressures.values() if isinstance(v, (int, float))]
        if len(raw) >= 2:
            spread = max(raw) - min(raw)
        if spread < _PRESSURE_SPREAD_THRESHOLD:
            return []

    ranked = sorted(_DRIVE_KEYS, key=lambda k: float(canon.get(k, 0.0)), reverse=True)
    top = ranked[0]
    runner = ranked[1]
    ts = envelope.created_at if envelope.created_at.tzinfo else envelope.created_at.replace(tzinfo=timezone.utc)
    trace_id = extract_trace_id(envelope)
    turn_id = extract_turn_id(envelope)
    source_event_ref = build_source_event_ref(envelope, intake_channel)
    evidence_items = build_evidence_items(envelope, intake_channel, None)
    prov = ArtifactProvenance(
        intake_channel=intake_channel,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        turn_id=turn_id,
        evidence_text=None,
        evidence_summary=evidence_items[0].summary if evidence_items else None,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
    )
    event = TensionEventV1(
        artifact_id=_artifact_id(envelope, entity_id, "tension.drive_competition.v1"),
        subject=subject,
        model_layer=model_layer,
        entity_id=entity_id,
        kind="tension.drive_competition.v1",
        ts=ts,
        confidence=0.68,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        turn_id=turn_id,
        provenance=prov,
        related_nodes=[f"drive:{top}", f"drive:{runner}", "tension.drive_competition.v1"],
        magnitude=clamp01(spread),
        drive_impacts={top: 0.9, runner: 0.75},
    )
    event.provenance.tension_refs = [event.artifact_id]
    return [event]


def extract_tensions_from_self_state(
    *,
    envelope: BaseEnvelope,
    intake_channel: str,
    self_state: SelfStateV1,
    previous_self_state: Optional[SelfStateV1] = None,
) -> List[TensionEventV1]:
    """Derive tensions from substrate self-state ticks, using dimension deltas where a previous state exists."""
    subject = "orion"
    model_layer = "self-model"
    entity_id = "self:orion"

    d = self_state.dimensions

    def _s(key: str, default: float = 0.5) -> float:
        dim = d.get(key)
        return float(dim.score) if dim is not None else default

    def _delta(key: str) -> Optional[float]:
        if previous_self_state is None:
            return None
        prev = previous_self_state.dimensions.get(key)
        return _s(key) - (float(prev.score) if prev is not None else _s(key))

    ts = envelope.created_at if envelope.created_at.tzinfo else envelope.created_at.replace(tzinfo=timezone.utc)
    trace_id = extract_trace_id(envelope)
    turn_id = extract_turn_id(envelope)
    source_event_ref = build_source_event_ref(envelope, intake_channel)
    evidence_text = f"condition={self_state.overall_condition} trajectory={self_state.trajectory_condition}"
    evidence_items = build_evidence_items(envelope, intake_channel, evidence_text)
    prov = ArtifactProvenance(
        intake_channel=intake_channel,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        turn_id=turn_id,
        evidence_text=evidence_text,
        evidence_summary=evidence_items[0].summary if evidence_items else None,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
    )
    traj_mul = 1.25 if self_state.trajectory_condition == "degrading" else 1.0

    events: List[TensionEventV1] = []

    # Coherence drop → contradiction (integration is breaking down)
    coh_delta = _delta("coherence")
    coh_now = _s("coherence")
    fire_coh = coh_delta < -0.04 if coh_delta is not None else coh_now < 0.35
    mag_coh = clamp01(abs(coh_delta) * traj_mul) if coh_delta is not None else clamp01((0.5 - coh_now) * traj_mul)
    if fire_coh and mag_coh > 0.0:
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.contradiction.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.contradiction.v1", ts=ts, confidence=0.82,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["substrate:coherence", "drive:coherence", "drive:predictive"],
            magnitude=mag_coh,
            drive_impacts={"coherence": 1.0, "predictive": 0.7},
        ))

    # Agency/social-ease drop → distress
    agency_delta = _delta("agency_readiness")
    social_delta = _delta("social_pressure")  # rising pressure = distress signal
    agency_now = _s("agency_readiness")
    if agency_delta is not None or social_delta is not None:
        combined = (-(agency_delta or 0.0) + (social_delta or 0.0)) / 2.0
        fire_dist = combined > 0.04
        mag_dist = clamp01(combined * traj_mul)
    else:
        fire_dist = agency_now < 0.30
        mag_dist = clamp01((0.5 - agency_now) * traj_mul)
    if fire_dist and mag_dist > 0.0:
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.distress.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.distress.v1", ts=ts, confidence=0.78,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["substrate:agency_readiness", "substrate:social_pressure", "drive:relational", "drive:continuity"],
            magnitude=mag_dist,
            drive_impacts={"relational": 0.9, "continuity": 0.55},
        ))

    # Uncertainty rise → identity_drift (ground shifting)
    unc_delta = _delta("uncertainty")
    unc_now = _s("uncertainty")
    fire_unc = unc_delta > 0.05 if unc_delta is not None else unc_now > 0.75
    mag_unc = clamp01(unc_delta * traj_mul) if unc_delta is not None else clamp01((unc_now - 0.5) * traj_mul)
    if fire_unc and mag_unc > 0.0:
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.identity_drift.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.identity_drift.v1", ts=ts, confidence=0.72,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["substrate:uncertainty", "drive:continuity", "drive:autonomy"],
            magnitude=mag_unc,
            drive_impacts={"continuity": 0.8, "autonomy": 0.6, "predictive": 0.4},
        ))

    # Resource + execution pressure rise → cognitive_load
    res_delta = _delta("resource_pressure")
    exe_delta = _delta("execution_pressure")
    res_now = _s("resource_pressure")
    exe_now = _s("execution_pressure")
    if res_delta is not None or exe_delta is not None:
        load_delta = ((res_delta or 0.0) + (exe_delta or 0.0)) / 2.0
        fire_load = load_delta > 0.04
        mag_load = clamp01(load_delta * traj_mul)
    else:
        overall_load = (res_now + exe_now) / 2.0
        fire_load = overall_load > 0.65
        mag_load = clamp01((overall_load - 0.5) * traj_mul)
    if fire_load and mag_load > 0.0:
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.cognitive_load.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.cognitive_load.v1", ts=ts, confidence=0.76,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["substrate:resource_pressure", "substrate:execution_pressure", "drive:capability"],
            magnitude=mag_load,
            drive_impacts={"capability": 0.9, "coherence": 0.45},
        ))

    for event in events:
        event.provenance.tension_refs = [event.artifact_id]
    return events


def extract_tensions_from_feedback(
    *,
    envelope: BaseEnvelope,
    intake_channel: str,
    feedback_frame: FeedbackFrameV1,
) -> List[TensionEventV1]:
    """Derive tensions from a scored feedback frame — real outcome signal closing the autonomy loop."""
    subject = "orion"
    model_layer = "self-model"
    entity_id = "self:orion"

    ts = envelope.created_at if envelope.created_at.tzinfo else envelope.created_at.replace(tzinfo=timezone.utc)
    trace_id = extract_trace_id(envelope)
    turn_id = extract_turn_id(envelope)
    source_event_ref = build_source_event_ref(envelope, intake_channel)
    evidence_text = (
        f"outcome_status={feedback_frame.outcome_status} "
        f"score={feedback_frame.outcome_score:.3f} "
        f"confidence={feedback_frame.confidence_score:.3f}"
    )
    evidence_items = build_evidence_items(envelope, intake_channel, evidence_text)
    prov = ArtifactProvenance(
        intake_channel=intake_channel,
        correlation_id=str(envelope.correlation_id),
        trace_id=str(trace_id) if trace_id else None,
        turn_id=turn_id,
        evidence_text=evidence_text,
        evidence_summary=evidence_items[0].summary if evidence_items else None,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
    )

    events: List[TensionEventV1] = []
    conf = clamp01(feedback_frame.confidence_score)

    # Failed/absent/blocked outcome → contradiction (predicted success, got nothing)
    if feedback_frame.outcome_status in ("failed", "absent", "blocked"):
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.contradiction.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.contradiction.v1", ts=ts, confidence=conf,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["feedback:outcome_status", "drive:predictive", "drive:coherence"],
            magnitude=clamp01(1.0 - feedback_frame.outcome_score),
            drive_impacts={"predictive": 1.0, "coherence": 0.65},
        ))

    # Low outcome score → distress (the action didn't land; autonomy frustrated)
    if feedback_frame.outcome_score < 0.4:
        mag = clamp01((0.5 - feedback_frame.outcome_score) * 2.0)
        events.append(TensionEventV1(
            artifact_id=_artifact_id(envelope, entity_id, "tension.distress.v1"),
            subject=subject, model_layer=model_layer, entity_id=entity_id,
            kind="tension.distress.v1", ts=ts, confidence=conf,
            correlation_id=str(envelope.correlation_id),
            trace_id=str(trace_id) if trace_id else None, turn_id=turn_id,
            provenance=prov,
            related_nodes=["feedback:outcome_score", "drive:relational", "drive:continuity", "drive:autonomy"],
            magnitude=mag,
            drive_impacts={"relational": 0.8, "continuity": 0.55, "autonomy": 0.5},
        ))

    for event in events:
        event.provenance.tension_refs = [event.artifact_id]
    return events
