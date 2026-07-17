from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)
from orion.schemas.telemetry.biometrics import (
    BiometricsInductionV1,
    BiometricsSampleV1,
    BiometricsSummaryV1,
)

from .node_catalog import NodeProfile


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _safe_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _availability_summary(
    node_id: str,
    node_profile: NodeProfile,
    summary: BiometricsSummaryV1,
) -> str:
    if summary.telemetry_error_rate and summary.telemetry_error_rate > 0.1:
        status = "DEGRADED"
    else:
        status = "OK"
    online = "expected online" if node_profile.expected_online else "expected offline"
    known = "known node" if node_profile.known else "unknown node"
    return f"{node_id} telemetry status {status} ({online}, {known})"


def _event(
    *,
    event_kind: str,
    trace_id: str,
    emitted_at: datetime,
    observed_at: datetime,
    provenance: GrammarProvenanceV1,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else uuid4().hex
    return GrammarEventV1(
        event_id=_hash_id(trace_id, event_kind, body_key, prefix="gev"),
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        layer=layer,
        dimensions=dimensions or [],
        atom=atom,
        edge=edge,
        provenance=provenance,
    )


def build_biometrics_node_grammar_events(
    *,
    sample: BiometricsSampleV1,
    summary: BiometricsSummaryV1,
    induction: BiometricsInductionV1,
    node_profile: NodeProfile,
    source_channel: str,
    code_version: str | None = None,
) -> list[GrammarEventV1]:
    observed_at = _dt(summary.timestamp or sample.timestamp or induction.timestamp)
    emitted_at = datetime.now(timezone.utc)
    ts = _safe_ts(observed_at)

    node_id = node_profile.node_id
    trace_id = f"biometrics.node:{node_id}:{ts}"

    provenance = GrammarProvenanceV1(
        source_service="orion-biometrics",
        source_component="biometrics_grammar_emit",
        source_event_id=f"{node_id}:{ts}",
        source_trace_id=trace_id,
        source_payload_ref=source_channel,
        code_version=code_version,
    )

    root_event = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="biometrics",
        dimensions=["node", "telemetry", "capability"],
    )
    root_id = root_event.event_id

    def atom_id(role: str) -> str:
        return f"{trace_id}:{role}"

    atoms = {
        "node_context": GrammarAtomV1(
            atom_id=atom_id("node_context"),
            trace_id=trace_id,
            atom_type="entity",
            semantic_role="node_context",
            layer="context",
            dimensions=["node", "role", "capability"],
            summary=f"{node_id} node context: role={node_profile.role}",
            text_value=node_id,
            confidence=1.0 if node_profile.known else 0.45,
            salience=1.0,
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.node_profile:{node_id}",
        ),
        "telemetry_sample": GrammarAtomV1(
            atom_id=atom_id("telemetry_sample"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="telemetry_sample",
            layer="raw_input",
            dimensions=["telemetry", "node"],
            summary=f"Raw biometrics sample observed for {node_id}",
            confidence=0.95,
            salience=0.7,
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.sample:{node_id}:{ts}",
        ),
        "body_state": GrammarAtomV1(
            atom_id=atom_id("body_state"),
            trace_id=trace_id,
            atom_type="observation",
            semantic_role="body_state",
            layer="organ_signal",
            dimensions=["physiology", "telemetry", "node"],
            summary=f"Biometrics summary/induction observed for {node_id}",
            confidence=0.95,
            salience=float((summary.composites or {}).get("strain", 0.5)),
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.summary:{node_id}:{ts}",
        ),
        "capability_surface": GrammarAtomV1(
            atom_id=atom_id("capability_surface"),
            trace_id=trace_id,
            atom_type="action_candidate",
            semantic_role="capability_surface",
            layer="capability",
            dimensions=["capability", "boundary", "node"],
            summary=(
                f"{node_id} capability surface: "
                f"{', '.join(sorted(k for k, v in node_profile.capabilities.items() if v)) or 'none'}"
            ),
            confidence=1.0 if node_profile.known else 0.45,
            salience=0.8,
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.capabilities:{node_id}",
        ),
        "node_availability": GrammarAtomV1(
            atom_id=atom_id("node_availability"),
            trace_id=trace_id,
            atom_type="uncertainty_marker",
            semantic_role="node_availability",
            layer="substrate",
            dimensions=["freshness", "availability", "node"],
            summary=_availability_summary(node_id, node_profile, summary),
            confidence=0.9,
            salience=0.7,
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.availability:{node_id}:{ts}",
        ),
        # Individually-computed hardware pressures (memory/thermal/disk).
        # These are already computed as named intermediates in
        # orion/telemetry/biometrics_pipeline.py's `pressures` dict (keys
        # "mem"/"thermal"/"disk") but were previously only folded into the
        # composite "strain" salience on `body_state` above -- no downstream
        # consumer could ever see them individually. Additive: `strain` and
        # `capability_surface`'s "gpu" hint are untouched.
        "memory_pressure_signal": GrammarAtomV1(
            atom_id=atom_id("memory_pressure_signal"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="memory_pressure_signal",
            layer="organ_signal",
            dimensions=["physiology", "telemetry", "node", "resource"],
            summary=f"{node_id} memory pressure observed",
            confidence=0.9,
            salience=float((summary.pressures or {}).get("mem", 0.0)),
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.pressure.memory:{node_id}:{ts}",
        ),
        "thermal_pressure_signal": GrammarAtomV1(
            atom_id=atom_id("thermal_pressure_signal"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="thermal_pressure_signal",
            layer="organ_signal",
            dimensions=["physiology", "telemetry", "node", "resource"],
            summary=f"{node_id} thermal pressure observed",
            confidence=0.9,
            salience=float((summary.pressures or {}).get("thermal", 0.0)),
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.pressure.thermal:{node_id}:{ts}",
        ),
        "disk_pressure_signal": GrammarAtomV1(
            atom_id=atom_id("disk_pressure_signal"),
            trace_id=trace_id,
            atom_type="signal",
            semantic_role="disk_pressure_signal",
            layer="organ_signal",
            dimensions=["physiology", "telemetry", "node", "resource"],
            summary=f"{node_id} disk pressure observed",
            confidence=0.9,
            salience=float((summary.pressures or {}).get("disk", 0.0)),
            source_event_id=f"{node_id}:{ts}",
            payload_ref=f"biometrics.pressure.disk:{node_id}:{ts}",
        ),
    }

    edge_specs = [
        ("node_context", "telemetry_sample", "references"),
        ("telemetry_sample", "body_state", "derived_from"),
        ("node_context", "body_state", "contains"),
        ("body_state", "capability_surface", "influenced"),
        ("node_availability", "capability_surface", "influenced"),
        ("node_context", "capability_surface", "supports"),
        ("telemetry_sample", "memory_pressure_signal", "derived_from"),
        ("telemetry_sample", "thermal_pressure_signal", "derived_from"),
        ("telemetry_sample", "disk_pressure_signal", "derived_from"),
        ("memory_pressure_signal", "capability_surface", "influenced"),
        ("thermal_pressure_signal", "capability_surface", "influenced"),
        ("disk_pressure_signal", "capability_surface", "influenced"),
    ]

    events: list[GrammarEventV1] = [root_event]

    for atom in atoms.values():
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                atom=atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=atom.layer,
                dimensions=atom.dimensions,
            )
        )

    for from_key, to_key, relation_type in edge_specs:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{from_key}:{relation_type}:{to_key}",
            trace_id=trace_id,
            from_atom_id=atoms[from_key].atom_id,
            to_atom_id=atoms[to_key].atom_id,
            relation_type=relation_type,  # type: ignore[arg-type]
            confidence=0.9,
            salience=0.7,
            layer_from=atoms[from_key].layer,
            layer_to=atoms[to_key].layer,
            evidence_event_ids=[f"{node_id}:{ts}"],
        )
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                edge=edge,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=edge.layer_to,
                dimensions=["node", "telemetry", "capability"],
            )
        )

    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=observed_at,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="biometrics",
            dimensions=["node", "telemetry", "capability"],
        )
    )

    return events
