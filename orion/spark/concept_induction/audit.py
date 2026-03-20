from __future__ import annotations

import hashlib
from typing import Iterable, List

from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.schemas.drives import DriveAuditV1, DriveStateV1, TensionEventV1
from .dossier import build_evidence_items, build_source_event_ref, extract_trace_id, extract_turn_id


def _artifact_id(subject: str, correlation_id: str | None, suffix: str) -> str:
    base = f"{subject}|{correlation_id or 'na'}|{suffix}"
    return f"{suffix}-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def build_drive_audit(
    *,
    env: BaseEnvelope,
    intake_channel: str,
    drive_state: DriveStateV1,
    tensions: Iterable[TensionEventV1],
) -> DriveAuditV1:
    tension_list = list(tensions)
    dominant_drive = None
    if drive_state.pressures:
        dominant_drive = max(sorted(drive_state.pressures), key=lambda key: drive_state.pressures.get(key, 0.0))
    active_drives = [key for key, active in sorted(drive_state.activations.items()) if active]
    evidence_items = build_evidence_items(env, intake_channel, drive_state.provenance.evidence_text)
    source_event_ref = build_source_event_ref(env, intake_channel)
    tension_refs = [tension.artifact_id for tension in tension_list]
    tension_kinds = [tension.kind for tension in tension_list]
    summary = None
    if dominant_drive:
        summary = f"{drive_state.subject} pressure concentrates on {dominant_drive}"
    return DriveAuditV1(
        artifact_id=_artifact_id(drive_state.subject, drive_state.correlation_id, "drive-audit"),
        subject=drive_state.subject,
        model_layer=drive_state.model_layer,
        entity_id=drive_state.entity_id,
        kind="memory.drives.audit.v1",
        ts=drive_state.updated_at,
        confidence=drive_state.confidence,
        correlation_id=drive_state.correlation_id,
        trace_id=drive_state.trace_id or extract_trace_id(env),
        turn_id=drive_state.turn_id or extract_turn_id(env),
        provenance=drive_state.provenance.model_copy(update={
            "source_event_refs": [source_event_ref],
            "evidence_items": evidence_items,
            "tension_refs": tension_refs,
        }),
        related_nodes=drive_state.related_nodes + tension_refs,
        drive_pressures=drive_state.pressures,
        drive_activations=drive_state.activations,
        active_drives=active_drives,
        dominant_drive=dominant_drive,
        tension_kinds=tension_kinds,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
        summary=summary,
    )
