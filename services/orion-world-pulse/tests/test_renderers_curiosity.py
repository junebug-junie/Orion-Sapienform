from datetime import datetime, timezone

from app.services.renderers import render_hub_digest, render_plaintext_digest
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
)


def _digest(followups):
    now = datetime.now(timezone.utc)
    return DailyWorldPulseV1(
        run_id="run-1",
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="summary",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="",
        created_at=now,
        curiosity_followups=followups,
    )


def test_block_absent_when_no_followups():
    text = render_plaintext_digest(_digest([]))
    assert "Orion went looking" not in text


def test_block_renders_section_query_and_article():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="hardware compute gpu recent news coverage",
        articles=[CuriosityFindingV1(url="https://ex/1", title="GPU news", salience=0.67)],
        action_id="fetch-x",
        correlation_id="run-1",
    )
    text = render_plaintext_digest(_digest([followup]))
    assert "Orion went looking" in text
    assert "hardware compute gpu" in text
    assert "hardware compute gpu recent news coverage" in text
    assert "GPU news" in text
    assert "0.67" in text


def test_hub_structured_payload_carries_followups():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="q",
        articles=[CuriosityFindingV1(url="https://ex/1", title="t", salience=0.1)],
    )
    hub = render_hub_digest(_digest([followup]))
    assert hub.structured_payload["curiosity_followups"][0]["section"] == "hardware_compute_gpu"
    assert "Orion went looking" in hub.rendered_markdown
