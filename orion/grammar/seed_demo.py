"""Deterministic vision_observation demo trace for Substrate Atlas."""

from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarCompactionV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProjectionV1,
    GrammarProvenanceV1,
    TimeRangeV1,
)

TRACE_ID = "trace:vision:demo"
_DEMO_TS = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
_TIME_RANGE = TimeRangeV1(start=_DEMO_TS, end=_DEMO_TS)
_PROVENANCE = GrammarProvenanceV1(
    source_service="orion-grammar-seed",
    source_component="vision_observation",
    code_version="seed_demo.v1",
)


def _event(
    *,
    event_id: str,
    event_kind: str,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    compaction: GrammarCompactionV1 | None = None,
    projection: GrammarProjectionV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str = "evt:vision:demo:000",
    offset_ms: int = 0,
) -> GrammarEventV1:
    emitted_at = _DEMO_TS.replace(microsecond=offset_ms * 1000)
    return GrammarEventV1(
        event_id=event_id,
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=TRACE_ID,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        session_id="sess:vision:demo",
        emitted_at=emitted_at,
        observed_at=emitted_at,
        layer=atom.layer if atom is not None else None,
        dimensions=list(atom.dimensions) if atom is not None else [],
        atom=atom,
        edge=edge,
        compaction=compaction,
        projection=projection,
        provenance=_PROVENANCE,
    )


def _atom(
    *,
    atom_id: str,
    semantic_role: str,
    atom_type: str,
    layer: str,
    summary: str,
    dimensions: list[str],
    confidence: float | None = None,
    uncertainty: float | None = None,
    source_event_id: str | None = None,
) -> GrammarAtomV1:
    return GrammarAtomV1(
        atom_id=atom_id,
        trace_id=TRACE_ID,
        atom_type=atom_type,  # type: ignore[arg-type]
        semantic_role=semantic_role,
        layer=layer,
        dimensions=dimensions,
        summary=summary,
        confidence=confidence,
        uncertainty=uncertainty,
        time_range=_TIME_RANGE,
        source_event_id=source_event_id,
    )


def build_vision_demo_events() -> list[GrammarEventV1]:
    """Build a fixed vision_observation trace (frame → motion → person → uncertainty → scene → projection)."""
    root = "evt:vision:demo:000"

    atom_frame = "atom:vision:demo:frame"
    atom_motion = "atom:vision:demo:motion"
    atom_person = "atom:vision:demo:person"
    atom_uncertainty = "atom:vision:demo:uncertainty"
    atom_spatial = "atom:vision:demo:spatial"
    atom_scene = "atom:vision:demo:scene"

    events: list[GrammarEventV1] = [
        _event(
            event_id=root,
            event_kind="trace_started",
            offset_ms=0,
        ),
        _event(
            event_id="evt:vision:demo:001",
            event_kind="atom_emitted",
            offset_ms=1,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_frame,
                semantic_role="frame_observed",
                atom_type="observation",
                layer="sensor_raw",
                summary="Camera frame captured (demo)",
                dimensions=["visual", "temporal"],
                confidence=1.0,
                source_event_id="evt:vision:demo:001",
            ),
        ),
        _event(
            event_id="evt:vision:demo:002",
            event_kind="atom_emitted",
            offset_ms=2,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_motion,
                semantic_role="motion_detected",
                atom_type="observation",
                layer="sensor_raw",
                summary="Motion delta detected in frame ROI",
                dimensions=["visual", "temporal", "spatial"],
                confidence=0.91,
                source_event_id="evt:vision:demo:002",
            ),
        ),
        _event(
            event_id="evt:vision:demo:003",
            event_kind="atom_emitted",
            offset_ms=3,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_person,
                semantic_role="object_candidate_person",
                atom_type="observation",
                layer="sensor_semantic",
                summary="Possible person near doorway",
                dimensions=["visual", "spatial", "epistemic", "social"],
                confidence=0.72,
                source_event_id="evt:vision:demo:003",
            ),
        ),
        _event(
            event_id="evt:vision:demo:004",
            event_kind="atom_emitted",
            offset_ms=4,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_uncertainty,
                semantic_role="uncertainty_marker",
                atom_type="uncertainty_marker",
                layer="metacognitive",
                summary="Person detection confidence below threshold",
                dimensions=["epistemic"],
                uncertainty=0.41,
                source_event_id="evt:vision:demo:004",
            ),
        ),
        _event(
            event_id="evt:vision:demo:005",
            event_kind="atom_emitted",
            offset_ms=5,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_spatial,
                semantic_role="spatial_region",
                atom_type="observation",
                layer="sensor_semantic",
                summary="Doorway region of interest",
                dimensions=["spatial", "visual"],
                confidence=0.88,
                source_event_id="evt:vision:demo:005",
            ),
        ),
        _event(
            event_id="evt:vision:demo:006",
            event_kind="edge_emitted",
            offset_ms=6,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:motion-from-frame",
                trace_id=TRACE_ID,
                from_atom_id=atom_motion,
                to_atom_id=atom_frame,
                relation_type="derived_from",
                confidence=0.91,
                layer_from="sensor_raw",
                layer_to="sensor_raw",
                evidence_event_ids=["evt:vision:demo:002"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:007",
            event_kind="edge_emitted",
            offset_ms=7,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:person-from-motion",
                trace_id=TRACE_ID,
                from_atom_id=atom_person,
                to_atom_id=atom_motion,
                relation_type="derived_from",
                confidence=0.72,
                layer_from="sensor_semantic",
                layer_to="sensor_raw",
                evidence_event_ids=["evt:vision:demo:003"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:008",
            event_kind="edge_emitted",
            offset_ms=8,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:uncertainty-supports-person",
                trace_id=TRACE_ID,
                from_atom_id=atom_uncertainty,
                to_atom_id=atom_person,
                relation_type="supports",
                confidence=0.65,
                layer_from="metacognitive",
                layer_to="sensor_semantic",
                evidence_event_ids=["evt:vision:demo:004"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:009",
            event_kind="edge_emitted",
            offset_ms=9,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:spatial-supports-person",
                trace_id=TRACE_ID,
                from_atom_id=atom_spatial,
                to_atom_id=atom_person,
                relation_type="supports",
                confidence=0.8,
                layer_from="sensor_semantic",
                layer_to="sensor_semantic",
                evidence_event_ids=["evt:vision:demo:005"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:010",
            event_kind="atom_emitted",
            offset_ms=10,
            parent_event_id=root,
            atom=_atom(
                atom_id=atom_scene,
                semantic_role="scene_state",
                atom_type="scene_state",
                layer="semantic_interpretation",
                summary="Doorway scene: motion, possible person, spatial ROI",
                dimensions=["visual", "spatial", "epistemic", "world"],
                confidence=0.68,
                source_event_id="evt:vision:demo:010",
            ),
        ),
        _event(
            event_id="evt:vision:demo:011",
            event_kind="edge_emitted",
            offset_ms=11,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:person-compacted-into-scene",
                trace_id=TRACE_ID,
                from_atom_id=atom_person,
                to_atom_id=atom_scene,
                relation_type="compacted_into",
                confidence=0.68,
                layer_from="sensor_semantic",
                layer_to="semantic_interpretation",
                evidence_event_ids=["evt:vision:demo:011"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:012",
            event_kind="edge_emitted",
            offset_ms=12,
            parent_event_id=root,
            edge=GrammarEdgeV1(
                edge_id="edge:vision:demo:motion-compacted-into-scene",
                trace_id=TRACE_ID,
                from_atom_id=atom_motion,
                to_atom_id=atom_scene,
                relation_type="compacted_into",
                confidence=0.68,
                layer_from="sensor_raw",
                layer_to="semantic_interpretation",
                evidence_event_ids=["evt:vision:demo:012"],
            ),
        ),
        _event(
            event_id="evt:vision:demo:013",
            event_kind="compaction_emitted",
            offset_ms=13,
            parent_event_id=root,
            compaction=GrammarCompactionV1(
                compaction_id="cmp:vision:demo:scene",
                trace_id=TRACE_ID,
                source_atom_ids=[atom_frame, atom_motion, atom_person, atom_spatial],
                output_atom_id=atom_scene,
                compaction_type="scene_summary",
                method="deterministic_seed",
                summary="Compact doorway observation into scene_state",
                preserves=["visual", "spatial", "epistemic"],
                drops=["raw_pixel_refs"],
                confidence=0.68,
            ),
        ),
        _event(
            event_id="evt:vision:demo:014",
            event_kind="projection_emitted",
            offset_ms=14,
            parent_event_id=root,
            projection=GrammarProjectionV1(
                projection_id="proj:vision:demo:presence",
                trace_id=TRACE_ID,
                source_atom_ids=[atom_person, atom_scene],
                projection_type="persistent_presence_candidate",
                summary="Candidate persistent presence (not confirmed fact)",
                confidence=0.55,
                projected_atom_id="atom:vision:demo:presence_candidate",
            ),
        ),
        _event(
            event_id="evt:vision:demo:015",
            event_kind="trace_ended",
            offset_ms=15,
            parent_event_id=root,
        ),
    ]
    return events
