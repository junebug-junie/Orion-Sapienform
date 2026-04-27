from __future__ import annotations

from datetime import datetime, timezone

from app.services.analysis import compare_situation
from app.services.capsule import build_world_context_capsule
from app.services.classify import classify_article
from app.services.corroboration import apply_corroboration
from app.services.dedupe import dedupe_articles
from app.services.digest import build_digest
from app.services.emit_graph import build_graph_delta
from app.services.extract import extract_entity, extract_topic
from app.services.renderers import render_plaintext_digest
from orion.schemas.world_pulse import (
    ArticleRecordV1,
    ClaimRecordV1,
    DailyWorldPulseItemV1,
    EventRecordV1,
    SituationRelevanceV1,
    SituationEvidenceV1,
    SituationObservationV1,
    TopicSituationBriefV1,
    WorldContextCapsuleV1,
    WorldPulseAllowedUsesV1,
)


def _article(article_id: str, title: str) -> ArticleRecordV1:
    now = datetime.now(timezone.utc)
    return ArticleRecordV1(
        article_id=article_id,
        run_id="run-1",
        source_id="reuters",
        source_name="Reuters",
        url="https://example.com/a",
        title=title,
        fetched_at=now,
        normalized_text_hash=f"h-{article_id}",
        content_hash=f"c-{article_id}",
        categories=["global_politics"],
        source_trust_tier=1,
        allowed_uses=WorldPulseAllowedUsesV1(),
        dedupe_key=article_id,
        extraction_status="normalized",
        provenance={},
        raw_metadata={},
    )


def test_dedupe_articles_keeps_first():
    a1 = _article("a1", "title")
    a2 = _article("a2", "title")
    a2 = a2.model_copy(update={"dedupe_key": a1.dedupe_key})
    deduped = dedupe_articles([a1, a2])
    assert len(deduped) == 1
    assert deduped[0].article_id == "a1"


def test_classify_article_politics():
    article = _article("a1", "Utah election update")
    assert classify_article(article) in {"local_politics", "global_politics"}


def test_apply_corroboration_promotes_when_two_sources():
    now = datetime.now(timezone.utc)
    c1 = ClaimRecordV1(
        claim_id="c1",
        run_id="r1",
        article_id="a1",
        claim_text="Data center energy demand is rising",
        source_trust_tier=1,
        source_ids=["reuters"],
        extracted_at=now,
    )
    c2 = c1.model_copy(update={"claim_id": "c2", "article_id": "a2", "source_ids": ["ap"]})
    out = apply_corroboration([c1, c2])
    assert all(c.corroboration_status == "corroborated" for c in out)


def test_apply_corroboration_respects_requires_corroboration():
    now = datetime.now(timezone.utc)
    c = ClaimRecordV1(
        claim_id="c1",
        run_id="r1",
        article_id="a1",
        claim_text="single source claim",
        source_trust_tier=1,
        source_ids=["reuters"],
        caveats=["requires_corroboration"],
        extracted_at=now,
    )
    out = apply_corroboration([c])
    assert out[0].promotion_status == "candidate"


def test_compare_situation_new_topic():
    now = datetime.now(timezone.utc)
    obs = SituationObservationV1(
        observation_id="o1",
        topic_id="t1",
        run_id="r1",
        observation_summary="New development",
        confidence=0.8,
        observed_at=now,
        status="active",
    )
    evidence = SituationEvidenceV1(
        evidence_id="e1",
        run_id="r1",
        source_id="reuters",
        article_id="a1",
        evidence_summary="Evidence",
        trust_tier=1,
        confidence=0.8,
        captured_at=now,
    )
    claim = ClaimRecordV1(
        claim_id="c1",
        run_id="r1",
        article_id="a1",
        claim_text="Claim",
        source_trust_tier=1,
        source_ids=["reuters"],
        extracted_at=now,
    )
    brief, changes, _ = compare_situation(None, [obs], [claim], [], [evidence])
    assert brief.tracking_status == "tracked"
    assert changes[0].change_type == "new_topic"


def test_compare_situation_change_branches():
    now = datetime.now(timezone.utc)
    obs = SituationObservationV1(
        observation_id="o1",
        topic_id="t1",
        run_id="r1",
        observation_summary="Utah AI policy changed",
        confidence=0.9,
        observed_at=now,
        status="active",
    )
    evidence = SituationEvidenceV1(
        evidence_id="e1",
        run_id="r1",
        source_id="reuters",
        article_id="a1",
        evidence_summary="Evidence",
        trust_tier=1,
        confidence=0.9,
        captured_at=now,
    )
    event = EventRecordV1(event_id="ev1", run_id="r1", title="event", event_type="other", summary="s", detected_at=now)
    previous = TopicSituationBriefV1(
        topic_id="t1",
        title="Old",
        scope="world",
        category="general_world",
        current_assessment="Data center buildout proceeds",
        last_updated=now,
        first_seen_at=now,
        confidence=0.4,
        source_mix={"source_count": 1},
        relevance=SituationRelevanceV1(orion_lab=0.1, local_utah=0.1),
        watch_conditions=["monitor"],
        created_at=now,
        updated_at=now,
    )
    claim_new_dev = ClaimRecordV1(
        claim_id="cdev",
        run_id="r1",
        article_id="a1",
        claim_text="New high-confidence deployment",
        confidence=0.9,
        source_trust_tier=1,
        source_ids=["reuters"],
        extracted_at=now,
    )
    _, changes, _ = compare_situation(previous, [obs], [claim_new_dev], [event], [evidence])
    assert changes[0].change_type in {
        "new_development",
        "confirmation",
        "contradiction",
        "source_quality_shift",
        "local_relevance_shift",
        "orion_relevance_shift",
    }


def test_compare_situation_stale_no_change():
    now = datetime.now(timezone.utc)
    obs = SituationObservationV1(
        observation_id="o1",
        topic_id="t1",
        run_id="r1",
        observation_summary="No update",
        confidence=0.2,
        observed_at=now,
        status="active",
    )
    previous = TopicSituationBriefV1(
        topic_id="t1",
        title="Old",
        scope="world",
        category="general_world",
        current_assessment="Stable",
        last_updated=now,
        first_seen_at=now,
        watch_conditions=["monitor"],
        created_at=now,
        updated_at=now,
    )
    _, changes, _ = compare_situation(previous, [obs], [], [], [])
    assert changes[0].change_type == "stale_no_change"


def test_digest_render_and_capsule():
    now = datetime.now(timezone.utc)
    item = DailyWorldPulseItemV1(
        item_id="i1",
        run_id="r1",
        title="GPU policy update",
        category="hardware_compute_gpu",
        summary="summary",
        why_it_matters="matters",
        what_changed="changed",
        orion_read="read",
        source_ids=["reuters"],
        confidence=0.9,
        stance_eligible=True,
        created_at=now,
    )
    digest = build_digest("r1", [item], [], [])
    text = render_plaintext_digest(digest)
    assert "GPU policy update" in text
    assert "Why it matters:" in text
    assert "Orion's read:" in text
    capsule = build_world_context_capsule(digest, locality="Utah", max_topics=5, min_confidence=0.65)
    assert capsule.salient_topics


def test_extract_ids_are_deterministic():
    article = _article("a1", "Deterministic Title")
    topic1 = extract_topic(article, "ai_technology")
    topic2 = extract_topic(article, "ai_technology")
    entity1 = extract_entity(article)
    entity2 = extract_entity(article)
    assert topic1.topic_id == topic2.topic_id
    assert entity1.entity_id == entity2.entity_id


def test_schema_json_serialization():
    now = datetime.now(timezone.utc)
    capsule = WorldContextCapsuleV1(
        capsule_id="c1",
        run_id="r1",
        date=now.date().isoformat(),
        generated_at=now,
        created_at=now,
    )
    payload = capsule.model_dump(mode="json")
    assert payload["capsule_id"] == "c1"


def test_graph_delta_escapes_quotes_in_titles():
    now = datetime.now(timezone.utc)
    item = DailyWorldPulseItemV1(
        item_id="i-quote",
        run_id="r-quote",
        title='State says "new policy"',
        category="local_politics",
        summary="summary",
        why_it_matters="matters",
        what_changed="changed",
        orion_read="read",
        source_ids=["source-1"],
        created_at=now,
    )
    digest = build_digest("r-quote", [item], [], [])
    delta = build_graph_delta(digest, dry_run=True, allowed_item_ids={"i-quote"})
    assert '\\"new policy\\"' in delta.triples
