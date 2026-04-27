from __future__ import annotations

from datetime import datetime, timezone

from app.services.emit_sql import build_sql_envelopes
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.world_pulse import (
    ArticleClusterV1,
    ArticleRecordV1,
    ClaimRecordV1,
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    EntityRecordV1,
    EventRecordV1,
    SituationChangeV1,
    TopicSituationBriefV1,
    WorldLearningDeltaV1,
    WorldPulseAllowedUsesV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def test_build_sql_envelopes_include_article_cluster_digest_and_status() -> None:
    now = datetime.now(timezone.utc)
    run = WorldPulseRunV1(
        run_id="run-1",
        date="2026-04-27",
        started_at=now,
        completed_at=now,
        requested_by="manual",
        dry_run=False,
        status="completed",
        sql_emit_status="published",
    )
    item = DailyWorldPulseItemV1(
        item_id="item-1",
        run_id="run-1",
        title="Item",
        category="us_politics",
        summary="summary",
        why_it_matters="why",
        what_changed="changed",
        orion_read="read",
        created_at=now,
    )
    digest = DailyWorldPulseV1(
        run_id="run-1",
        date="2026-04-27",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="summary",
        sections=DailyWorldPulseSectionsV1(),
        items=[item],
        orion_analysis_layer="layer",
        created_at=now,
    )
    run_result = WorldPulseRunResultV1(
        run=run,
        digest=digest,
        publish_status={
            "articles": [
                ArticleRecordV1(
                    article_id="a1",
                    run_id="run-1",
                    source_id="s1",
                    source_name="Source 1",
                    url="https://example.com/a1",
                    title="Article 1",
                    fetched_at=now,
                    normalized_text_hash="h1",
                    content_hash="h2",
                    source_trust_tier=1,
                    allowed_uses=WorldPulseAllowedUsesV1(),
                    dedupe_key="d1",
                ).model_dump(mode="json")
            ],
            "clusters": [
                ArticleClusterV1(
                    cluster_id="c1",
                    run_id="run-1",
                    title="Cluster 1",
                    summary="Cluster summary",
                    category="us_politics",
                    article_ids=["a1"],
                    topic_ids=["t1"],
                    article_count=1,
                    source_ids=["s1"],
                    source_count=1,
                    source_tiers_present=[1],
                    source_agreement=1.0,
                    confidence=0.8,
                    created_at=now,
                ).model_dump(mode="json")
            ],
        },
    )
    claim = ClaimRecordV1(
        claim_id="claim-1",
        run_id="run-1",
        article_id="a1",
        claim_text="fact",
        source_trust_tier=1,
        extracted_at=now,
    )
    event = EventRecordV1(
        event_id="event-1",
        run_id="run-1",
        title="event",
        event_type="policy",
        summary="summary",
        detected_at=now,
    )
    entity = EntityRecordV1(
        entity_id="entity-1",
        canonical_name="Entity",
        entity_type="org",
        first_seen_at=now,
        last_seen_at=now,
    )
    brief = TopicSituationBriefV1(
        topic_id="topic-1",
        title="Topic",
        scope="daily",
        category="us_politics",
        current_assessment="assessment",
        last_updated=now,
        first_seen_at=now,
        created_at=now,
        updated_at=now,
    )
    change = SituationChangeV1(
        change_id="change-1",
        topic_id="topic-1",
        run_id="run-1",
        change_type="new_development",
        change_summary="summary",
        created_at=now,
    )
    learning = WorldLearningDeltaV1(
        learning_id="learn-1",
        run_id="run-1",
        topic_id="topic-1",
        category="us_politics",
        summary="summary",
        why_it_matters="matters",
        created_at=now,
    )
    envelopes = build_sql_envelopes(
        source_ref=ServiceRef(name="orion-world-pulse", version="0.1.0", node="n1"),
        run_result=run_result,
        claims=[claim],
        events=[event],
        entities=[entity],
        briefs=[brief],
        changes=[change],
        learning=[learning],
    )
    kinds = {envelope.kind for _, envelope in envelopes}
    assert "world.pulse.run.result.v1" in kinds
    assert "world.pulse.digest.created.v1" in kinds
    assert "world.pulse.digest.item.v1" in kinds
    assert "world.pulse.article.emit.v1" in kinds
    assert "world.pulse.cluster.emit.v1" in kinds
    assert "world.pulse.publish.status.v1" in kinds
