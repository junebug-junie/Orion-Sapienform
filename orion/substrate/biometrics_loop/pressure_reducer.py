from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.biometrics.node_catalog import NodeCatalog
from orion.substrate.ids import stable_delta_id, stable_receipt_id
from orion.schemas.biometrics_projection import ActiveNodePressureProjectionV1, ActiveNodePressureStateV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1

from .candidate_events import PRESSURE_SOURCE_COMPONENT
from .ids import parse_pressure_trace_id
from .pressure_organ import ALLOWED_PRESSURE_ROLES

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MERGE_WINDOW_SEC = 300

ROLE_TO_PRESSURE_KIND = {
    "node_pressure_detected": "strain",
    "node_pressure_reinforced": "strain",
    "node_pressure_decayed": "strain",
    "node_availability_concern": "availability",
    "node_availability_recovered": "availability",
    "node_pressure_suppressed": "strain",
    "node_capability_impact": "capability",
}

ROLE_TO_OPERATION = {
    "node_pressure_detected": "create",
    "node_pressure_reinforced": "reinforce",
    "node_pressure_decayed": "decay",
    "node_availability_concern": "update",
    # Reuses "decay" (StateDeltaV1.operation is a closed Literal with no
    # "recover" value) -- semantically this delta is removing a pressure
    # kind from active_pressures, the same shape as node_pressure_decayed.
    "node_availability_recovered": "decay",
    "node_pressure_suppressed": "suppress",
    "node_capability_impact": "update",
}


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _extract_candidate_traces(candidates: list) -> list[list[GrammarEventV1]]:
    traces: list[list[GrammarEventV1]] = []
    for item in candidates:
        if isinstance(item, list):
            traces.append(item)
        else:
            traces.append([item])
    return traces


def _evidence_valid(evidence_event_ids: list[str]) -> bool:
    if not evidence_event_ids:
        return False
    return any(
        event_id.startswith("gev_") or event_id.startswith("biometrics.node:")
        for event_id in evidence_event_ids
    )


def _collect_evidence(trace: list[GrammarEventV1]) -> list[str]:
    evidence: list[str] = []
    for event in trace:
        if event.edge and event.edge.evidence_event_ids:
            evidence.extend(event.edge.evidence_event_ids)
    return evidence


def _confidence_for_trace(trace: list[GrammarEventV1]) -> float:
    for event in trace:
        if event.atom and event.atom.confidence is not None:
            return float(event.atom.confidence)
    return 0.0


def _semantic_role_for_trace(trace: list[GrammarEventV1]) -> str | None:
    for event in trace:
        if event.atom:
            return event.atom.semantic_role
    return None


def _node_id_for_trace(trace: list[GrammarEventV1]) -> str | None:
    for event in trace:
        if event.trace_id:
            parsed = parse_pressure_trace_id(event.trace_id)
            if parsed:
                return parsed
        if event.atom and event.atom.text_value:
            return event.atom.text_value.strip().lower()
    return None


def _atom_event_id(trace: list[GrammarEventV1]) -> str | None:
    for event in trace:
        if event.event_kind == "atom_emitted" and event.atom:
            return event.event_id
    return None


def reduce_node_pressure_candidates(
    *,
    candidates: list[list[GrammarEventV1]],
    projection: ActiveNodePressureProjectionV1,
    catalog: NodeCatalog,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    merge_window_sec: int = DEFAULT_MERGE_WINDOW_SEC,
    reducer_id: str = "node_pressure_reducer",
    organ_id: str = "biometrics_pressure",
    emission_id: str | None = None,
    now: datetime | None = None,
) -> tuple[ActiveNodePressureProjectionV1, ReductionReceiptV1]:
    clock = _utc_now(now)
    updated = deepcopy(projection)
    updated.generated_at = clock

    accepted: list[str] = []
    rejected: list[str] = []
    merged: list[str] = []
    state_deltas: list[StateDeltaV1] = []
    projection_updates: list[ProjectionUpdateV1] = []

    traces = _extract_candidate_traces(candidates)

    for trace in traces:
        atom_event_id = _atom_event_id(trace)
        if atom_event_id is None:
            continue

        provenance = next((ev.provenance for ev in trace if ev.provenance), None)
        if provenance is None or provenance.source_component != PRESSURE_SOURCE_COMPONENT:
            rejected.append(atom_event_id)
            continue

        role = _semantic_role_for_trace(trace)
        if role not in ALLOWED_PRESSURE_ROLES:
            rejected.append(atom_event_id)
            continue

        node_id = _node_id_for_trace(trace)
        profile = catalog.resolve(node_id or "")
        if not profile.known:
            rejected.append(atom_event_id)
            continue
        canonical_node_id = profile.node_id

        evidence = _collect_evidence(trace)
        if not _evidence_valid(evidence):
            rejected.append(atom_event_id)
            continue

        confidence = _confidence_for_trace(trace)
        if confidence < min_confidence:
            rejected.append(atom_event_id)
            continue

        pressure_kind = ROLE_TO_PRESSURE_KIND.get(role, "strain")
        existing = updated.nodes.get(canonical_node_id)

        # Merge-window dedup against the projection's own persisted
        # last_accepted_at (per node+pressure_kind) -- not a function-local
        # dict. This function is called once per trigger event (see
        # orion/substrate/biometrics_loop/pipeline.py's per-event loop), so
        # a call-scoped dict could never accumulate dedup history across
        # events -- confirmed live 2026-07-22: merge_window_sec=300 was a
        # complete no-op in production, node:atlas alone accepted 767
        # "reinforce" deltas in 2 hours (~6.4/min) instead of the ~24 a
        # working 5-minute window would allow.
        if existing is not None:
            prior = existing.last_accepted_at.get(pressure_kind)
            if prior is not None:
                prior_aware = prior if prior.tzinfo else prior.replace(tzinfo=timezone.utc)
                if (clock - prior_aware).total_seconds() <= merge_window_sec:
                    merged.append(atom_event_id)
                    continue

        before = existing.model_dump(mode="json") if existing else None
        if existing is None:
            node_state = ActiveNodePressureStateV1(
                node_id=canonical_node_id,
                availability_status="unknown",
                last_updated_at=clock,
            )
            projection_operation = "create"
        else:
            node_state = existing.model_copy(deep=True)
            projection_operation = "update"

        node_state.last_updated_at = clock
        node_state.last_accepted_at[pressure_kind] = clock
        # Cap to 200 most-recent IDs — this list is append-only and grows to 400K+ events
        # without a bound, causing deepcopy/serialize to become O(N) per tick.
        all_ids = sorted(set(node_state.evidence_event_ids + evidence))
        node_state.evidence_event_ids = all_ids[-200:]

        if role == "node_pressure_detected":
            if pressure_kind not in node_state.active_pressures:
                node_state.active_pressures.append(pressure_kind)
            node_state.pressure_score = max(node_state.pressure_score, confidence)
        elif role == "node_pressure_reinforced":
            if pressure_kind not in node_state.active_pressures:
                node_state.active_pressures.append(pressure_kind)
            node_state.pressure_score = max(node_state.pressure_score, confidence)
        elif role == "node_pressure_decayed":
            node_state.active_pressures = [p for p in node_state.active_pressures if p != pressure_kind]
            node_state.pressure_score = max(0.0, node_state.pressure_score - 0.2)
        elif role == "node_availability_concern":
            if "availability" not in node_state.active_pressures:
                node_state.active_pressures.append("availability")
            node_state.availability_status = "stale"
        elif role == "node_availability_recovered":
            node_state.active_pressures = [p for p in node_state.active_pressures if p != pressure_kind]
            node_state.availability_status = "online"
        elif role == "node_pressure_suppressed":
            for pressure in list(node_state.active_pressures):
                if pressure == pressure_kind:
                    node_state.active_pressures.remove(pressure)
                    if pressure not in node_state.suppressed_pressures:
                        node_state.suppressed_pressures.append(pressure)
        elif role == "node_capability_impact":
            label = f"capability:{pressure_kind}"
            if label not in node_state.capability_impacts:
                node_state.capability_impacts.append(label)

        updated.nodes[canonical_node_id] = node_state
        accepted.append(atom_event_id)

        state_deltas.append(
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=updated.projection_id,
                    target_kind="active_node_pressure",
                    target_id=canonical_node_id,
                    operation=ROLE_TO_OPERATION[role],
                    caused_by_event_ids=[atom_event_id],
                ),
                target_projection=updated.projection_id,
                target_kind="active_node_pressure",
                target_id=canonical_node_id,
                operation=ROLE_TO_OPERATION[role],
                before=before,
                after=node_state.model_dump(mode="json"),
                caused_by_event_ids=[atom_event_id],
                reducer_id=reducer_id,
            )
        )
        projection_updates.append(
            ProjectionUpdateV1(
                projection_kind="active_node_pressure",
                projection_id=updated.projection_id,
                node_id=canonical_node_id,
                operation=projection_operation,
            )
        )

    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=accepted,
            rejected_event_ids=rejected,
            merged_event_ids=merged,
            noop_event_ids=[],
            emission_id=emission_id,
        ),
        emission_id=emission_id,
        organ_id=organ_id,
        accepted_event_ids=accepted,
        rejected_event_ids=rejected,
        merged_event_ids=merged,
        state_deltas=state_deltas,
        projection_updates=projection_updates,
        created_at=clock,
    )
    return updated, receipt
