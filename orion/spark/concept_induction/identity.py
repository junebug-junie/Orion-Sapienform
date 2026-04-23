from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, Optional

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import DriveStateV1, IdentitySnapshotV1, TensionEventV1

SELF_SUBJECT = "orion"
USER_SUBJECT = "juniper"
RELATIONSHIP_SUBJECT = "relationship"
WORLD_FALLBACK_SUBJECT = "world:external-fallback"


def _slug(value: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return norm or "unknown"


def _payload_dict(payload: Any) -> Dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    return payload if isinstance(payload, dict) else {}


def _source_name(env: BaseEnvelope) -> str:
    return (getattr(env.source, "name", None) or "").strip().lower()


def infer_external_subject(env: BaseEnvelope, intake_channel: str) -> str:
    payload = _payload_dict(env.payload)
    for key in ("service", "service_name", "component", "component_name", "subsystem"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return f"service:{_slug(value)}"

    tool_value = payload.get("tool_name") or payload.get("tool") or payload.get("tool_id") or payload.get("verb")
    if isinstance(tool_value, str) and tool_value.strip():
        return f"tooling:{_slug(tool_value)}"

    channel = (intake_channel or "").lower()
    source_name = _source_name(env)
    if "telemetry" in channel or env.kind.endswith("telemetry") or "telemetry" in source_name:
        return "telemetry"
    if any(token in channel for token in ("system", "hardware", "equilibrium", "infra")):
        return "infra"
    if source_name:
        return f"service:{_slug(source_name)}"
    return WORLD_FALLBACK_SUBJECT


def resolve_subject_identity(env: BaseEnvelope, intake_channel: str) -> str:
    payload = _payload_dict(env.payload)
    explicit = payload.get("subject") or payload.get("entity_id")
    if isinstance(explicit, str) and explicit.strip():
        norm = explicit.strip().lower()
        if norm in {SELF_SUBJECT, USER_SUBJECT, RELATIONSHIP_SUBJECT}:
            return norm
        if norm == "world":
            return infer_external_subject(env, intake_channel)
        if norm.startswith("service:"):
            return f"service:{_slug(norm.split(':', 1)[1])}"
        if norm.startswith("tooling:"):
            return f"tooling:{_slug(norm.split(':', 1)[1])}"
        if norm in {"infra", "telemetry"}:
            return norm

    user = payload.get("user") or payload.get("speaker")
    role = payload.get("role")
    if isinstance(user, str) and user.lower().startswith("juniper"):
        return USER_SUBJECT
    # Dyadic chat turns (Hub WS/HTTP, GPT log, etc.) are relationship-scoped: drives/tensions must
    # land on `relationship` so Graph autonomy + Hub cards match what chat_stance queries.
    ch = (intake_channel or "").lower()
    if any(tok in ch for tok in ("chat:history", "chat:gpt", "chat:social")):
        if isinstance(payload.get("prompt"), str) and isinstance(payload.get("response"), str):
            return RELATIONSHIP_SUBJECT
    role_norm = str(role or "").strip().lower()
    if role_norm == "assistant":
        return SELF_SUBJECT
    # Avoid classifying Hub as "self": source names like "orion-hub" contain "orion" but are not
    # the assistant agent identity for autonomy graph keys.
    source_name = _source_name(env)
    if "orion" in source_name and "hub" not in source_name:
        return SELF_SUBJECT
    if any(token in ch for token in ("telemetry", "system", "hardware", "infra")):
        return infer_external_subject(env, intake_channel)
    return RELATIONSHIP_SUBJECT


def model_layer_for_subject(subject: str) -> str:
    if subject == SELF_SUBJECT:
        return "self-model"
    if subject == USER_SUBJECT:
        return "user-model"
    if subject == RELATIONSHIP_SUBJECT:
        return "relationship-model"
    return "world-model"


def entity_id_for_subject(subject: str, model_layer: str) -> str:
    if model_layer == "self-model":
        return "self:orion"
    if model_layer == "user-model":
        return f"user:{_slug(subject)}"
    if model_layer == "relationship-model":
        return "relationship:orion|juniper"
    if subject == WORLD_FALLBACK_SUBJECT:
        return "world:fallback:external-unknown"
    return f"world:{subject}"


def anchor_strategy_for_subject(subject: str, model_layer: str) -> str:
    if model_layer != "world-model":
        return "canonical-subject"
    if subject == WORLD_FALLBACK_SUBJECT:
        return "explicit-world-fallback"
    return "concrete-world-entity"


def _artifact_id(subject: str, correlation_id: str | None, suffix: str) -> str:
    base = f"{subject}|{correlation_id or 'na'}|{suffix}"
    return f"{suffix}-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def build_identity_snapshot(
    *,
    drive_state: DriveStateV1,
    source_event_ref,
    evidence_items,
    tensions: Iterable[TensionEventV1],
) -> IdentitySnapshotV1:
    tension_list = list(tensions)
    anchor_strategy = anchor_strategy_for_subject(drive_state.subject, drive_state.model_layer)
    summary = f"{drive_state.subject} anchored as {drive_state.entity_id} in {drive_state.model_layer}"
    return IdentitySnapshotV1(
        artifact_id=_artifact_id(drive_state.subject, drive_state.correlation_id, "identity-snapshot"),
        subject=drive_state.subject,
        model_layer=drive_state.model_layer,
        entity_id=drive_state.entity_id,
        kind="memory.identity.snapshot.v1",
        ts=drive_state.updated_at,
        confidence=drive_state.confidence,
        correlation_id=drive_state.correlation_id,
        trace_id=drive_state.trace_id,
        turn_id=drive_state.turn_id,
        provenance=drive_state.provenance.model_copy(update={
            "source_event_refs": [source_event_ref],
            "evidence_items": evidence_items,
            "tension_refs": [tension.artifact_id for tension in tension_list],
            "evidence_summary": evidence_items[0].summary if evidence_items else drive_state.provenance.evidence_summary,
        }),
        related_nodes=drive_state.related_nodes + [tension.artifact_id for tension in tension_list],
        anchor_strategy=anchor_strategy,
        summary=summary,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
        tension_kinds=[tension.kind for tension in tension_list],
        drive_pressures=drive_state.pressures,
    )
