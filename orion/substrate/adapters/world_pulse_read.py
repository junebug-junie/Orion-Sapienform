from __future__ import annotations

import hashlib
from datetime import datetime

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateGraphRecordV1,
    SubstrateSignalBundleV1,
)
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1

from ._common import make_provenance, make_temporal


def _concept_node_id(trace_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{trace_id}:{label}".encode()).hexdigest()[:16]
    return f"sub-concept-wp-read-{digest}"


def map_world_pulse_read_handoff_to_substrate(
    handoff: WorldPulseReadHandoffV1,
    *,
    observed_at: datetime | None = None,
) -> SubstrateGraphRecordV1:
    observed = observed_at or handoff.created_at
    seed = handoff.seed_ref
    evidence = [seed.url, seed.run_id, handoff.trace_id]
    nodes = []
    for cand in handoff.concept_candidates:
        nodes.append(
            ConceptNodeV1(
                node_id=_concept_node_id(handoff.trace_id, cand.label),
                anchor_scope="orion",
                subject_ref="world_pulse",
                temporal=make_temporal(observed_at=observed),
                provenance=make_provenance(
                    source_kind="world_pulse.read",
                    source_channel="orion:world_pulse:read",
                    producer="world_pulse_read_pipeline",
                    correlation_id=seed.run_id,
                    trace_id=handoff.trace_id,
                    evidence_refs=evidence,
                ),
                label=cand.label,
                definition=cand.definition,
                signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.5),
                metadata={
                    "seed_id": seed.seed_id,
                    "seed_kind": seed.kind,
                    "section": seed.section,
                },
            )
        )
    return SubstrateGraphRecordV1(
        anchor_scope="orion",
        subject_ref="world_pulse",
        nodes=nodes,
        edges=[],
    )
