from datetime import datetime, timezone

from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadPriorCandidateV1,
    WorldPulseReadSeedV1,
)


def test_seed_round_trip():
    seed = WorldPulseReadSeedV1(
        seed_id="finding:run-1:abcd",
        kind="finding",
        run_id="run-1",
        url="https://example.com/a",
        title="A",
        section="ai_technology",
    )
    assert WorldPulseReadSeedV1.model_validate(seed.model_dump()).seed_id == seed.seed_id


def test_handoff_requires_seed_ref_and_trace():
    handoff = WorldPulseReadHandoffV1(
        seed_ref=WorldPulseReadSeedV1(
            seed_id="finding:run-1:abcd",
            kind="finding",
            run_id="run-1",
            url="https://example.com/a",
            title="A",
            section="ai_technology",
        ),
        what_i_learned="Chip fab news.",
        candidate_priors=[
            WorldPulseReadPriorCandidateV1(claim="TSMC capacity is tight", confidence=0.6)
        ],
        concept_candidates=[
            WorldPulseReadConceptCandidateV1(label="advanced packaging", definition="chip packaging")
        ],
        open_threads=["Who supplies the lasers?"],
        trace_id="tr-1",
        created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )
    re = WorldPulseReadHandoffV1.model_validate(handoff.model_dump(mode="json"))
    assert re.producer_hint == "world_pulse_read_pipeline"
    assert re.concept_candidates[0].label == "advanced packaging"
    assert re.trace_id == "tr-1"
