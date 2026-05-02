from __future__ import annotations

from datetime import datetime, timezone

from app.services.analysis import compare_situation
from orion.schemas.world_pulse import (
    ClaimRecordV1,
    EventRecordV1,
    SituationEvidenceV1,
    SituationObservationV1,
    SituationRelevanceV1,
    TopicSituationBriefV1,
)


def _previous(now: datetime) -> TopicSituationBriefV1:
    return TopicSituationBriefV1(
        topic_id="t1",
        title="topic",
        scope="world",
        category="general_world",
        current_assessment="Data center buildout proceeds",
        known_facts=["data center buildout proceeds"],
        source_mix={"source_count": 1},
        relevance=SituationRelevanceV1(local_utah=0.1, orion_lab=0.1),
        last_updated=now,
        first_seen_at=now,
        watch_conditions=["monitor"],
        created_at=now,
        updated_at=now,
    )


def _obs(now: datetime) -> SituationObservationV1:
    return SituationObservationV1(
        observation_id="o1",
        topic_id="t1",
        run_id="r1",
        observation_summary="obs",
        confidence=0.9,
        observed_at=now,
        status="active",
    )


def _evidence(now: datetime, source_id: str = "reuters", trust_tier: int = 1) -> SituationEvidenceV1:
    return SituationEvidenceV1(
        evidence_id=f"e:{source_id}:{trust_tier}",
        run_id="r1",
        source_id=source_id,
        article_id="a1",
        evidence_summary="ev",
        trust_tier=trust_tier,
        confidence=0.9,
        captured_at=now,
    )


def _claim(now: datetime, text: str, confidence: float = 0.9) -> ClaimRecordV1:
    return ClaimRecordV1(
        claim_id=f"c:{text}",
        run_id="r1",
        article_id="a1",
        claim_text=text,
        confidence=confidence,
        source_trust_tier=1,
        source_ids=["reuters"],
        extracted_at=now,
    )


def test_new_topic_branch():
    now = datetime.now(timezone.utc)
    _, changes, _ = compare_situation(None, [_obs(now)], [_claim(now, "new item")], [], [_evidence(now)])
    assert changes[0].change_type == "new_topic"


def test_new_development_branch():
    now = datetime.now(timezone.utc)
    previous = _previous(now)
    event = EventRecordV1(event_id="new-event", run_id="r1", title="e", event_type="other", summary="s", detected_at=now)
    _, changes, priors = compare_situation(previous, [_obs(now)], [_claim(now, "fresh update", 0.95)], [event], [_evidence(now)])
    assert changes[0].change_type == "new_development"
    assert priors


def test_confirmation_branch():
    now = datetime.now(timezone.utc)
    previous = _previous(now)
    _, changes, _ = compare_situation(previous, [_obs(now)], [_claim(now, "data center buildout proceeds")], [], [_evidence(now)])
    assert changes[0].change_type == "confirmation"


def test_contradiction_branch():
    now = datetime.now(timezone.utc)
    previous = _previous(now)
    _, changes, priors = compare_situation(previous, [_obs(now)], [_claim(now, "claim was denied by officials")], [], [_evidence(now)])
    assert changes[0].change_type == "contradiction"
    assert priors


def test_stale_no_change_branch():
    now = datetime.now(timezone.utc)
    previous = _previous(now)
    _, changes, _ = compare_situation(previous, [_obs(now)], [], [], [])
    assert changes[0].change_type == "stale_no_change"


def test_source_quality_shift_branch():
    now = datetime.now(timezone.utc)
    previous = _previous(now)
    _, changes, _ = compare_situation(
        previous,
        [_obs(now)],
        [_claim(now, "novel detail", 0.6)],
        [],
        [_evidence(now, "reuters", 1), _evidence(now, "ap", 1)],
    )
    assert changes[0].change_type == "source_quality_shift"
