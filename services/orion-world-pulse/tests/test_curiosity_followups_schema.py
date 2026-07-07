from datetime import datetime, timezone

from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
)


def _minimal_digest(**overrides) -> DailyWorldPulseV1:
    now = datetime.now(timezone.utc)
    base = dict(
        run_id="run-1",
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="",
        created_at=now,
    )
    base.update(overrides)
    return DailyWorldPulseV1(**base)


def test_digest_defaults_to_empty_followups():
    digest = _minimal_digest()
    assert digest.curiosity_followups == []


def test_followup_round_trips_with_findings():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="hardware compute gpu recent news coverage",
        articles=[
            CuriosityFindingV1(url="https://x/1", title="A", description="d", salience=0.5)
        ],
        action_id="fetch-abc",
        correlation_id="run-1",
    )
    digest = _minimal_digest(curiosity_followups=[followup])
    rehydrated = DailyWorldPulseV1.model_validate(digest.model_dump(mode="json"))
    assert rehydrated.curiosity_followups[0].section == "hardware_compute_gpu"
    assert rehydrated.curiosity_followups[0].articles[0].salience == 0.5
