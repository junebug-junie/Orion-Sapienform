from datetime import datetime, timezone

from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate


def _handoff(**over):
    base = dict(
        seed_ref=WorldPulseReadSeedV1(
            seed_id="finding:r1:x",
            kind="finding",
            run_id="r1",
            url="https://ex.com/a",
            title="A",
            section="ai_technology",
        ),
        what_i_learned="Learned about packaging.",
        concept_candidates=[
            WorldPulseReadConceptCandidateV1(label="advanced packaging", definition="chip pkg")
        ],
        trace_id="tr-9",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    base.update(over)
    return WorldPulseReadHandoffV1(**base)


def test_mapper_sets_locked_provenance():
    record = map_world_pulse_read_handoff_to_substrate(_handoff())
    concepts = [n for n in record.nodes if n.node_kind == "concept"]
    assert len(concepts) == 1
    p = concepts[0].provenance
    assert p.producer == "world_pulse_read_pipeline"
    assert p.source_kind == "world_pulse.read"
    assert p.trace_id == "tr-9"
    assert "https://ex.com/a" in p.evidence_refs
    assert "r1" in p.evidence_refs
    assert "tr-9" in p.evidence_refs
    assert concepts[0].label == "advanced packaging"


def test_mapper_skips_empty_labels_only_via_schema(min_length_enforced=True):
    record = map_world_pulse_read_handoff_to_substrate(
        _handoff(concept_candidates=[])
    )
    assert record.nodes == []
