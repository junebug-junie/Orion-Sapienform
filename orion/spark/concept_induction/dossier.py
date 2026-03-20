from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import ArtifactEventRef, ArtifactEvidence, GraphReadyArtifact, TurnDossierV1


def _payload_dict(payload: Any) -> Dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    return payload if isinstance(payload, dict) else {}


def extract_trace_id(env: BaseEnvelope) -> Optional[str]:
    if isinstance(env.trace, dict):
        trace_id = env.trace.get("trace_id") or env.trace.get("trace")
        if trace_id:
            return str(trace_id)
    return None


def extract_turn_id(env: BaseEnvelope) -> Optional[str]:
    payload = _payload_dict(env.payload)
    for key in ("turn_id", "id", "chat_turn_ref", "message_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    corr = payload.get("correlation_id")
    if isinstance(corr, str) and corr.strip():
        return corr
    return str(env.correlation_id) if env.correlation_id else None


def build_source_event_ref(env: BaseEnvelope, intake_channel: str) -> ArtifactEventRef:
    return ArtifactEventRef(
        event_id=str(env.id),
        kind=env.kind,
        channel=intake_channel,
        correlation_id=str(env.correlation_id) if env.correlation_id else None,
        trace_id=extract_trace_id(env),
        turn_id=extract_turn_id(env),
        created_at=env.created_at,
        source_service=getattr(env.source, "name", None),
    )


def build_evidence_items(env: BaseEnvelope, intake_channel: str, evidence_text: Optional[str]) -> List[ArtifactEvidence]:
    payload = _payload_dict(env.payload)
    summary = None
    for key in ("summary", "content", "text", "message", "final_text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()
            break
    if not summary and evidence_text:
        summary = evidence_text.strip()
    if summary:
        summary = summary[:280]
    return [
        ArtifactEvidence(
            event_ref=build_source_event_ref(env, intake_channel),
            summary=summary,
            text=evidence_text[:500] if isinstance(evidence_text, str) and evidence_text else None,
            source_summary=f"{env.kind} on {intake_channel}",
        )
    ]


def dossier_ref_key(channel: str) -> Optional[str]:
    lowered = (channel or "").lower()
    if "chat:history" in lowered:
        return "chat_turn_ref"
    if "spark:telemetry" in lowered:
        return "spark_telemetry_ref"
    if "metacognition:tick" in lowered:
        return "metacognition_tick_ref"
    if "collapse:sql-write" in lowered:
        return "collapse_sql_write_ref"
    if "cognition:trace" in lowered:
        return "cognition_trace_ref"
    return None


def build_turn_dossier(
    *,
    env: BaseEnvelope,
    intake_channel: str,
    subject: str,
    model_layer: str,
    entity_id: str,
    published: List[GraphReadyArtifact],
    suppressed_goal_signatures: List[str],
) -> TurnDossierV1:
    event_ref = build_source_event_ref(env, intake_channel)
    dossier = TurnDossierV1(
        correlation_id=str(env.correlation_id) if env.correlation_id else None,
        trace_id=extract_trace_id(env),
        turn_id=event_ref.turn_id,
        subject=subject,
        model_layer=model_layer,
        entity_id=entity_id,
        source_event_refs=[event_ref],
        tension_refs=[artifact.artifact_id for artifact in published if artifact.kind.startswith("tension.")],
        drive_audit_ref=next((artifact.artifact_id for artifact in published if artifact.kind == "memory.drives.audit.v1"), None),
        identity_snapshot_ref=next((artifact.artifact_id for artifact in published if artifact.kind == "memory.identity.snapshot.v1"), None),
        goal_proposal_ref=next((artifact.artifact_id for artifact in published if artifact.kind == "memory.goals.proposed.v1"), None),
        suppressed_goal_signatures=suppressed_goal_signatures,
    )
    ref_key = dossier_ref_key(intake_channel)
    if ref_key:
        setattr(dossier, ref_key, event_ref.event_id)
    return dossier
