from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.world_pulse import (
    ClaimRecordV1,
    EventRecordV1,
    SituationChangeV1,
    SituationEvidenceV1,
    SituationObservationV1,
    SituationPriorUpdateCandidateV1,
    TopicSituationBriefV1,
)


def _contains_any(text: str, tokens: list[str]) -> bool:
    lowered = text.lower()
    return any(t in lowered for t in tokens)


def _determine_change_type(
    previous_brief: TopicSituationBriefV1 | None,
    current_claims: list[ClaimRecordV1],
    current_events: list[EventRecordV1],
    current_evidence: list[SituationEvidenceV1],
) -> str:
    if previous_brief is None:
        return "new_topic"
    if not current_claims and not current_events:
        return "stale_no_change"

    prior_text = " ".join([previous_brief.current_assessment, *previous_brief.known_facts]).lower()
    claim_texts = [c.claim_text for c in current_claims]
    claim_blob = " ".join(claim_texts).lower()
    high_conf = any(c.confidence >= 0.75 for c in current_claims)
    new_event_ids = [e.event_id for e in current_events if e.event_id not in set(previous_brief.event_ids)]
    avg_tier_now = sum(e.trust_tier for e in current_evidence) / max(1, len(current_evidence))
    prev_source_count = max(1, int(previous_brief.source_mix.get("source_count", 1)))
    source_quality_improved = (avg_tier_now <= 2 and prev_source_count == 1 and len({e.source_id for e in current_evidence}) >= 2)

    if _contains_any(claim_blob, ["contradict", "denied", "retracted", "false"]):
        return "contradiction"
    if prior_text and any(c.lower() in prior_text for c in claim_texts):
        return "confirmation"
    if source_quality_improved:
        return "source_quality_shift"
    if _contains_any(claim_blob, ["utah", "local", "county", "city"]) and previous_brief.relevance.local_utah < 0.5:
        return "local_relevance_shift"
    if _contains_any(claim_blob, ["ai", "gpu", "model", "orion"]) and previous_brief.relevance.orion_lab < 0.5:
        return "orion_relevance_shift"
    if high_conf or new_event_ids:
        return "new_development"
    return "stale_no_change"


def compare_situation(
    previous_brief: TopicSituationBriefV1 | None,
    current_observations: list[SituationObservationV1],
    current_claims: list[ClaimRecordV1],
    current_events: list[EventRecordV1],
    current_evidence: list[SituationEvidenceV1],
) -> tuple[TopicSituationBriefV1, list[SituationChangeV1], list[SituationPriorUpdateCandidateV1]]:
    now = datetime.now(timezone.utc)
    first = current_observations[0]
    known_facts = [c.claim_text for c in current_claims[:5]]
    change_type = _determine_change_type(previous_brief, current_claims, current_events, current_evidence)

    change = SituationChangeV1(
        change_id=f"change:{first.topic_id}:{first.run_id}",
        topic_id=first.topic_id,
        run_id=first.run_id,
        change_type=change_type,
        change_summary=f"{change_type} from {len(current_claims)} claims and {len(current_events)} events",
        previous_state=previous_brief.current_assessment if previous_brief else None,
        new_state=current_observations[0].observation_summary,
        evidence_ids=[e.evidence_id for e in current_evidence],
        claim_ids=[c.claim_id for c in current_claims],
        event_ids=[e.event_id for e in current_events],
        confidence=max([o.confidence for o in current_observations] or [0.0]),
        importance=0.7 if change_type != "stale_no_change" else 0.2,
        requires_followup=change_type in {"new_development", "contradiction"},
        expires_or_recheck_after=now + timedelta(days=1),
        created_at=now,
    )

    prior_candidates: list[SituationPriorUpdateCandidateV1] = []
    if change_type in {"new_development", "contradiction", "source_quality_shift"}:
        prior_candidates.append(
            SituationPriorUpdateCandidateV1(
                candidate_id=f"prior:{first.topic_id}:{first.run_id}",
                topic_id=first.topic_id,
                run_id=first.run_id,
                existing_prior=previous_brief.current_assessment if previous_brief else "No prior",
                new_evidence=current_observations[0].observation_summary,
                proposed_update=f"Update working assumption for {first.topic_id}",
                confidence=0.66,
                requires_review=True,
                affected_orion_contexts=["planning", "advice"],
                evidence_ids=[e.evidence_id for e in current_evidence],
                source_ids=list({e.source_id for e in current_evidence}),
                created_at=now,
                status="candidate",
            )
        )

    brief = TopicSituationBriefV1(
        topic_id=first.topic_id,
        title=first.observation_summary,
        scope="world",
        category="general_world",
        current_assessment=first.observation_summary,
        previous_assessment=previous_brief.current_assessment if previous_brief else None,
        last_updated=now,
        first_seen_at=previous_brief.first_seen_at if previous_brief else now,
        status="volatile" if change_type in {"new_development", "contradiction"} else "stable",
        confidence=max([o.confidence for o in current_observations] or [0.0]),
        source_agreement="mixed",
        known_facts=known_facts,
        open_questions=[],
        recent_changes=[change.change_id],
        prior_assumptions=[],
        prior_update_candidates=[c.candidate_id for c in prior_candidates],
        source_mix={"source_count": len({e.source_id for e in current_evidence})},
        evidence_ids=[e.evidence_id for e in current_evidence],
        claim_ids=[c.claim_id for c in current_claims],
        event_ids=[e.event_id for e in current_events],
        article_ids=list({e.article_id for e in current_evidence}),
        watch_conditions=["recheck daily"],
        next_recheck_at=now + timedelta(days=1),
        stance_eligible=True,
        epistemic_posture={"confidence_label": "working"},
        tracking_status="tracked",
        created_at=previous_brief.created_at if previous_brief else now,
        updated_at=now,
    )
    return brief, [change], prior_candidates
