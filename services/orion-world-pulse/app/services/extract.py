from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from orion.schemas.world_pulse import (
    ArticleRecordV1,
    ClaimRecordV1,
    EntityRecordV1,
    EventRecordV1,
    SituationEvidenceV1,
    SituationObservationV1,
    TopicRecordV1,
)


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def extract_topic(article: ArticleRecordV1, category: str) -> TopicRecordV1:
    key = f"{category}:{article.title.lower().split(' ')[0]}"
    now = datetime.now(timezone.utc)
    return TopicRecordV1(
        topic_id=_stable_id("topic", key),
        title=article.title,
        normalized_key=key,
        category=category,
        region_scope=article.region_scope,
        relevance_tags=[f"{category}_relevance"],
        description=article.text_excerpt or article.title,
        created_at=now,
        last_seen_at=now,
        tracking_status="candidate",
    )


def extract_claim(
    article: ArticleRecordV1,
    topic_id: str,
    *,
    requires_corroboration: bool,
    claim_extraction_allowed: bool,
) -> ClaimRecordV1:
    now = datetime.now(timezone.utc)
    base_caveats: list[str] = []
    if requires_corroboration:
        base_caveats.append("requires_corroboration")
    if not claim_extraction_allowed:
        base_caveats.append("claim_extraction_disabled")
    return ClaimRecordV1(
        claim_id=f"claim:{article.article_id}",
        run_id=article.run_id,
        article_id=article.article_id,
        topic_id=topic_id,
        claim_text=article.title,
        claim_type="factual",
        region_scope=article.region_scope,
        valid_as_of=article.published_at or now,
        expires_at=(article.published_at or now) + timedelta(days=7),
        confidence=0.6,
        source_trust_tier=article.source_trust_tier,
        corroboration_status="uncorroborated",
        promotion_status="candidate" if requires_corroboration else "observed",
        controversy_level="medium" if "election" in article.title.lower() else "low",
        source_ids=[article.source_id],
        caveats=base_caveats,
        extracted_at=now,
    )


def extract_event(article: ArticleRecordV1, claim_id: str) -> EventRecordV1:
    now = datetime.now(timezone.utc)
    return EventRecordV1(
        event_id=f"event:{article.article_id}",
        run_id=article.run_id,
        title=article.title,
        event_type="other",
        summary=article.text_excerpt or article.title,
        occurred_at=article.published_at,
        detected_at=now,
        claim_ids=[claim_id],
        article_ids=[article.article_id],
        source_ids=[article.source_id],
        confidence=0.6,
        status="developing",
        provenance={"article_id": article.article_id},
    )


def extract_entity(article: ArticleRecordV1) -> EntityRecordV1:
    now = datetime.now(timezone.utc)
    head = article.title.split(" ")[0]
    return EntityRecordV1(
        entity_id=_stable_id("entity", head.lower()),
        canonical_name=head,
        entity_type="topic",
        source_ids=[article.source_id],
        first_seen_at=now,
        last_seen_at=now,
        confidence=0.4,
        provenance={"article_id": article.article_id},
    )


def build_evidence(article: ArticleRecordV1, claim_id: str, event_id: str) -> SituationEvidenceV1:
    now = datetime.now(timezone.utc)
    return SituationEvidenceV1(
        evidence_id=f"evidence:{article.article_id}",
        run_id=article.run_id,
        source_id=article.source_id,
        article_id=article.article_id,
        claim_id=claim_id,
        event_id=event_id,
        evidence_summary=article.text_excerpt or article.title,
        trust_tier=article.source_trust_tier,
        confidence=0.6,
        url=article.url,
        canonical_url=article.canonical_url,
        published_at=article.published_at,
        captured_at=now,
    )


def build_observation(article: ArticleRecordV1, topic_id: str, claim_id: str, event_id: str, evidence_id: str) -> SituationObservationV1:
    now = datetime.now(timezone.utc)
    return SituationObservationV1(
        observation_id=f"observation:{article.article_id}",
        topic_id=topic_id,
        run_id=article.run_id,
        observation_summary=article.title,
        supporting_claim_ids=[claim_id],
        supporting_event_ids=[event_id],
        supporting_article_ids=[article.article_id],
        evidence_ids=[evidence_id],
        confidence=0.6,
        observed_at=now,
        expires_at=now + timedelta(days=2),
        status="active",
    )
