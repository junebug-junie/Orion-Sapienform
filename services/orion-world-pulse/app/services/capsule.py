from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.world_pulse import DailyWorldPulseV1, WorldContextCapsuleV1, WorldContextTopicV1


def build_world_context_capsule(
    digest: DailyWorldPulseV1,
    *,
    locality: str,
    max_topics: int,
    min_confidence: float,
) -> WorldContextCapsuleV1:
    now = datetime.now(timezone.utc)
    topics: list[WorldContextTopicV1] = []
    for item in digest.items:
        if not item.stance_eligible or item.confidence < min_confidence:
            continue
        topics.append(
            WorldContextTopicV1(
                topic_id=item.topic_ids[0] if item.topic_ids else item.item_id,
                topic=item.title,
                summary=item.summary,
                relevance_tags=[item.category],
                expires_at=now + timedelta(hours=36),
                confidence=item.confidence,
                use_when=["user asks current events", "safety/planning relevance"],
                do_not_volunteer=True,
            )
        )
        if len(topics) >= max_topics:
            break
    return WorldContextCapsuleV1(
        capsule_id=f"capsule:{digest.run_id}",
        run_id=digest.run_id,
        date=digest.date,
        locality=locality,
        generated_at=now,
        salient_topics=topics,
        politics_context={
            "available": True,
            "default_use": "only_when_relevant_or_requested",
            "do_not_volunteer": True,
        },
        use_policy={
            "mention_only_when_relevant": True,
            "avoid_unsolicited_politics": True,
            "do_not_present_as_personal_experience": True,
            "cite_uncertainty_when_needed": True,
        },
        expires_at=now + timedelta(hours=36),
        stance_eligible_item_ids=[t.topic_id for t in topics],
        created_at=now,
    )
