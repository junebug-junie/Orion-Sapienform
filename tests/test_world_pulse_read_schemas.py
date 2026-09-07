from datetime import datetime, timezone

from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadPriorCandidateV1,
    WorldPulseReadSeedV1,
    WorldPulseReadStage2ResultV1,
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


def test_handoff_coerces_string_priors_and_concepts():
    """Live FCC returned bare strings for priors — must not burn the seed."""
    seed = {
        "seed_id": "finding:run-1:abcd",
        "kind": "finding",
        "run_id": "run-1",
        "url": "https://example.com/a",
        "title": "A",
        "section": "ai_technology",
    }
    handoff = WorldPulseReadHandoffV1.model_validate(
        {
            "seed_ref": seed,
            "what_i_learned": "Learned from a YouTube teaser.",
            "candidate_priors": [
                "Nvidia is returning to MSRP Founders Edition sales",
                {"claim": "RAM shortages constrain GPUs", "confidence": 0.4},
            ],
            "concept_candidates": [
                "RTX 50-series",
                {"label": "Vera Rubin", "definition": "platform"},
            ],
            "open_threads": ["Need full article body"],
            "trace_id": "tr-coerce",
            "created_at": "2026-09-07T00:00:00+00:00",
        }
    )
    assert handoff.candidate_priors[0].claim.startswith("Nvidia")
    assert handoff.candidate_priors[0].confidence == 0.5
    assert handoff.candidate_priors[1].confidence == 0.4
    assert handoff.concept_candidates[0].label == "RTX 50-series"
    assert handoff.concept_candidates[1].definition == "platform"


def test_stage2_result_round_trip():
    result = WorldPulseReadStage2ResultV1(
        summary="Priors formed.",
        need_stage1_urls=["https://example.com/more"],
        trace_id="tr-s2",
        created_at=datetime(2026, 9, 7, tzinfo=timezone.utc),
        seed_id="finding:run-1:abcd",
    )
    re = WorldPulseReadStage2ResultV1.model_validate(result.model_dump(mode="json"))
    assert re.producer_hint == "world_pulse_read_stage2"
    assert re.need_stage1_urls == ["https://example.com/more"]
