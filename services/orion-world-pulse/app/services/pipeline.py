from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.world_pulse import (
    CoverageStatus,
    DailyWorldPulseItemV1,
    SectionCoverageV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
    WorldLearningDeltaV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)

from app.models import now_utc
from app.settings import settings
from app.services.analysis import compare_situation
from app.services.capsule import build_world_context_capsule
from app.services.classify import classify_article
from app.services.cluster import build_article_clusters
from app.services.corroboration import apply_corroboration
from app.services.dedupe import dedupe_articles
from app.services.digest import build_digest, build_section_rollups, curate_digest_items
from app.services.emit_graph import build_graph_delta
from app.services.extract import (
    build_evidence,
    build_observation,
    extract_claim,
    extract_entity,
    extract_event,
    extract_topic,
)
from app.services.ingest.registry import fetch_source_candidates
from app.services.normalize_article import normalize_article
from app.services.source_registry import load_source_registry
from app.services.trust import assess_source, source_allowed_for_fetch, source_allowed_for_graph

logger = logging.getLogger("orion-world-pulse.pipeline")
SECTION_NAMES = [
    "us_politics",
    "global_politics",
    "local_politics",
    "ai_technology",
    "science_climate_energy",
    "healthcare_mental_health",
    "security_infrastructure_software",
    "hardware_compute_gpu",
    "local_conditions",
]


def _finalize_digest_aggregates(
    *,
    digest,
    run,
    all_articles,
    clusters,
    curated_items,
    coverage_status,
    section_coverage,
    section_rollups,
    max_digest_items_total: int,
) -> None:
    digest.coverage_status = coverage_status
    digest.section_coverage = section_coverage
    digest.section_rollups = section_rollups
    digest.accepted_article_count = max(int(digest.accepted_article_count or 0), len(all_articles))
    digest.article_cluster_count = max(int(digest.article_cluster_count or 0), len(clusters))
    digest.max_digest_items_total = max(int(digest.max_digest_items_total or 0), int(max_digest_items_total))
    if not digest.source_ids:
        digest.source_ids = sorted({a.source_id for a in all_articles})
    run.metrics["coverage_status"] = str(digest.coverage_status)
    run.metrics["digest_items"] = len(curated_items)
    run.metrics["accepted_article_count"] = digest.accepted_article_count
    run.metrics["article_cluster_count"] = digest.article_cluster_count


def _compute_coverage(
    *,
    registry,  # SourceRegistryV1
    all_articles,
    digest_items: list[DailyWorldPulseItemV1],
) -> tuple[CoverageStatus, dict[str, SectionCoverageV1], list[str], list[str], list[str]]:
    accepted_source_ids = {a.source_id for a in all_articles}
    per_section: dict[str, SectionCoverageV1] = {}
    covered_sections: list[str] = []
    for section in SECTION_NAMES:
        enabled_sources = [
            s for s in registry.sources if s.enabled and s.approved and section in (s.categories or [])
        ]
        sources_enabled = len(enabled_sources)
        fetched_sources = len([s for s in enabled_sources if s.source_id in accepted_source_ids])
        accepted_articles = len([a for a in all_articles if section in (a.categories or [])])
        digest_count = len([i for i in digest_items if i.category == section])
        if digest_count > 0 or accepted_articles > 0:
            section_status = "covered"
            covered_sections.append(section)
        elif sources_enabled == 0:
            section_status = "source_unavailable"
        elif fetched_sources > 0:
            section_status = "no_articles"
        else:
            section_status = "missing"
        per_section[section] = SectionCoverageV1(
            sources_enabled=sources_enabled,
            sources_fetched=fetched_sources,
            articles_accepted=accepted_articles,
            digest_items=digest_count,
            status=section_status,
        )
    required = list(registry.required_sections or [])
    recommended = list(registry.recommended_sections or [])
    missing_required = [s for s in required if per_section.get(s) and per_section[s].status != "covered"]
    missing_recommended = [s for s in recommended if per_section.get(s) and per_section[s].status != "covered"]
    if not covered_sections:
        coverage_status: CoverageStatus = "empty"
    elif not missing_required and not missing_recommended:
        coverage_status = "complete"
    elif not missing_required:
        coverage_status = "partial"
    else:
        coverage_status = "sparse"
    return coverage_status, per_section, missing_required, missing_recommended, covered_sections


def run_world_pulse(
    *,
    run_id: str | None = None,
    date: str | None = None,
    requested_by: str = "manual",
    dry_run: bool | None = None,
    fixture_items: list[dict[str, Any]] | None = None,
) -> WorldPulseRunResultV1:
    rid = run_id or str(uuid4())
    now = now_utc()
    dry = settings.world_pulse_dry_run if dry_run is None else dry_run
    run = WorldPulseRunV1(
        run_id=rid,
        date=date or now.date().isoformat(),
        started_at=now,
        status="running",
        requested_by=requested_by,
        dry_run=dry,
    )
    if not settings.world_pulse_enabled and not dry:
        run.status = "failed"
        run.errors.append("world pulse is disabled")
        run.completed_at = datetime.now(timezone.utc)
        return WorldPulseRunResultV1(run=run, publish_status={"detail": "disabled"})
    logger.info("world_pulse_run_start run_id=%s dry_run=%s", rid, dry)
    try:
        registry = load_source_registry(settings.world_pulse_sources_config_path)
    except Exception as exc:
        run.status = "failed"
        run.errors.append(str(exc))
        run.completed_at = datetime.now(timezone.utc)
        return WorldPulseRunResultV1(run=run, publish_status={"detail": "source_registry_error"})
    logger.info("world_pulse_source_loaded run_id=%s source_count=%s", rid, len(registry.sources))
    source_index = {s.source_id: s for s in registry.sources}
    all_articles = []
    fetched_count = 0
    sources_skipped = 0
    sources_failed = 0
    source_skip_details: list[str] = []
    source_error_details: list[str] = []
    required_source_failures: list[str] = []
    if fixture_items:
        source = next((s for s in registry.sources if s.enabled and s.approved), None)
        if source is None:
            run.status = "failed"
            run.errors.append("no approved source available for fixture mode")
            run.completed_at = datetime.now(timezone.utc)
            return WorldPulseRunResultV1(run=run, publish_status={"detail": "fixture_source_missing"})
        for row in fixture_items[: settings.world_pulse_max_articles_per_run]:
            all_articles.append(normalize_article(run_id=rid, source=source, item=row, fetched_at=now))
    elif settings.world_pulse_fetch_enabled:
        for source in registry.sources:
            assessment = assess_source(source)
            if not source_allowed_for_fetch(source=source, requested_by=requested_by):
                skip_reason = f"source_skipped:{source.source_id}"
                run.warnings.append(skip_reason)
                source_skip_details.append(skip_reason)
                sources_skipped += 1
                continue
            strategy = source.strategy or "rss"
            logger.info(
                "world_pulse_fetch_source_start run_id=%s source_id=%s strategy=%s",
                rid,
                source.source_id,
                strategy,
            )
            try:
                candidates = fetch_source_candidates(source, settings.world_pulse_fetch_timeout_seconds)
            except Exception as exc:
                logger.warning(
                    "world_pulse_fetch_source_result run_id=%s source_id=%s strategy=%s status=error error=%s",
                    rid,
                    source.source_id,
                    strategy,
                    exc,
                )
                err = f"source_fetch_error:{source.source_id}"
                source_error_details.append(err)
                sources_failed += 1
                if source.required:
                    run.errors.append(err)
                    required_source_failures.append(source.source_id)
                else:
                    run.warnings.append(f"optional_{err}")
                continue
            for candidate in candidates[: min(source.max_articles_per_day, settings.world_pulse_max_articles_per_source)]:
                if not assessment.allowed_uses.digest:
                    continue
                all_articles.append(
                    normalize_article(
                        run_id=rid,
                        source=source,
                        item=candidate.model_dump(mode="python"),
                        fetched_at=now,
                    )
                )
            fetched_count += len(candidates)
            logger.info(
                "world_pulse_fetch_source_result run_id=%s source_id=%s strategy=%s fetched=%s accepted=%s",
                rid,
                source.source_id,
                strategy,
                len(candidates),
                min(source.max_articles_per_day, settings.world_pulse_max_articles_per_source, len(candidates)),
            )
    all_articles = dedupe_articles(all_articles)[: settings.world_pulse_max_articles_per_run]
    run.sources_considered = len(registry.sources)
    run.sources_fetched = len({a.source_id for a in all_articles})
    run.sources_failed = sources_failed
    run.sources_skipped = sources_skipped
    run.articles_fetched = fetched_count
    run.articles_accepted = len(all_articles)
    run.metrics["source_errors"] = source_error_details
    run.metrics["source_skips"] = source_skip_details
    run.metrics["required_source_failures"] = required_source_failures
    if not all_articles:
        run.status = "failed"
        run.warnings.append("no enabled sources or no articles fetched")
        run.completed_at = datetime.now(timezone.utc)
        return WorldPulseRunResultV1(run=run, publish_status={"detail": "no sources"})
    clusters = build_article_clusters(all_articles, clustering_policy=registry.clustering_policy)
    run.metrics["article_clusters"] = len(clusters)
    singleton_cluster_count = len([c for c in clusters if c.article_count <= 1])
    multi_article_cluster_count = len([c for c in clusters if c.article_count > 1])
    run.metrics["singleton_cluster_count"] = singleton_cluster_count
    run.metrics["multi_article_cluster_count"] = multi_article_cluster_count
    run.metrics["average_articles_per_cluster"] = round(
        len(all_articles) / max(1, len(clusters)),
        3,
    )

    digest_policy = registry.digest_policy or {}
    max_digest_items_total = int(digest_policy.get("max_digest_items_total", 12))
    max_digest_items_per_section = int(digest_policy.get("max_digest_items_per_section", 2))
    min_digest_items_per_required = int(digest_policy.get("min_digest_items_per_required_section", 1))
    max_worth_reading = int(digest_policy.get("max_worth_reading", 10))
    max_worth_watching = int(digest_policy.get("max_worth_watching", 8))
    situation_policy = registry.situation_policy or {}
    max_situation_changes_per_run = int(situation_policy.get("max_situation_changes_per_run", 25))
    min_articles_for_corroborated_topic = int(situation_policy.get("min_articles_for_corroborated_topic", 2))

    claims = []
    events = []
    entities = []
    briefs = []
    changes = []
    learning = []
    items = []
    worth_reading = []
    worth_watching = []
    article_index = {a.article_id: a for a in all_articles}
    for idx, cluster in enumerate(clusters):
        cluster_articles = [article_index[aid] for aid in cluster.article_ids if aid in article_index]
        if not cluster_articles:
            continue
        representative = cluster_articles[0]
        category = cluster.category or classify_article(representative)
        topic = extract_topic(representative, category).model_copy(
            update={"topic_id": cluster.cluster_id.replace("cluster:", "topic:")}
        )
        cluster_claims = []
        cluster_events = []
        cluster_evidence = []
        cluster_obs = []
        cluster_entities = []
        for article in cluster_articles:
            if category in {"us_politics", "global_politics", "local_politics"} and not article.allowed_uses.digest:
                continue
            claim = extract_claim(
                article,
                topic.topic_id,
                requires_corroboration=bool(
                    source_index.get(article.source_id).requires_corroboration if source_index.get(article.source_id) else False
                ),
                claim_extraction_allowed=bool(article.allowed_uses.claim_extraction),
            ).model_copy(update={"cluster_id": cluster.cluster_id, "topic_id": topic.topic_id})
            event = extract_event(article, claim.claim_id)
            entity = extract_entity(article)
            evidence = build_evidence(article, claim.claim_id, event.event_id)
            obs = build_observation(article, topic.topic_id, claim.claim_id, event.event_id, evidence.evidence_id)
            cluster_claims.append(claim)
            cluster_events.append(event)
            cluster_entities.append(entity)
            cluster_evidence.append(evidence)
            cluster_obs.append(obs)

        if not cluster_claims:
            continue
        brief, delta_changes, prior_candidates = compare_situation(None, cluster_obs, cluster_claims, cluster_events, cluster_evidence)
        consolidated_change = delta_changes[0].model_copy(
            update={
                "change_id": f"{delta_changes[0].change_id}:{cluster.cluster_id}",
                "change_summary": f"{delta_changes[0].change_type} across {len(cluster_articles)} articles and {cluster.source_count} sources",
            }
        )
        claims.extend(cluster_claims)
        events.extend(cluster_events)
        entities.extend(cluster_entities)
        briefs.append(brief)
        changes.append(consolidated_change)
        # Prior candidates are retained on briefs; do not emit them as extra situation changes.
        _ = prior_candidates
        source_count = len({a.source_id for a in cluster_articles})
        cluster_confidence = max([c.confidence for c in cluster_claims] or [0.0])
        if source_count >= min_articles_for_corroborated_topic:
            cluster_confidence = min(1.0, cluster_confidence + 0.1)
        learning.append(
            WorldLearningDeltaV1(
                learning_id=f"learning:{cluster.cluster_id}",
                run_id=rid,
                topic_id=topic.topic_id,
                category=category,
                summary=cluster.title,
                why_it_matters=cluster.summary,
                entities=[e.entity_id for e in cluster_entities[:3]],
                claims=[c.claim_id for c in cluster_claims[:5]],
                confidence=cluster_confidence,
                source_count=source_count,
                relevance_tags=[category],
                stance_eligible=False,
                graph_eligible=all(
                    source_allowed_for_graph(source=source_index[a.source_id])
                    for a in cluster_articles
                    if a.source_id in source_index
                ),
                created_at=now,
            )
        )
        item = DailyWorldPulseItemV1(
            item_id=f"item:{cluster.cluster_id}",
            run_id=rid,
            title=cluster.title,
            category=category,
            region_scope=representative.region_scope,
            summary=cluster.summary,
            why_it_matters=cluster.summary,
            what_changed=consolidated_change.change_summary,
            context_bullets=[cluster.summary],
            by_the_numbers=[f"Articles: {len(cluster_articles)}", f"Sources: {source_count}"],
            what_theyre_saying=[],
            caveats=list({cv for c in cluster_claims for cv in c.caveats}),
            orion_read=brief.current_assessment,
            what_to_watch=brief.watch_conditions,
            worth_reading=[a.title for a in cluster_articles[:2]],
            source_ids=sorted({a.source_id for a in cluster_articles}),
            article_ids=[a.article_id for a in cluster_articles],
            claim_ids=[c.claim_id for c in cluster_claims],
            event_ids=[e.event_id for e in cluster_events],
            topic_ids=[topic.topic_id],
            situation_change_ids=[consolidated_change.change_id],
            confidence=cluster_confidence,
            volatility=brief.volatility,
            source_agreement=brief.source_agreement,
            stance_eligible=False,
            created_at=now,
        )
        items.append(item)
        best_articles = sorted(cluster_articles, key=lambda a: (a.source_trust_tier, -(a.published_at or a.fetched_at).timestamp()))[:2]
        for i2, article in enumerate(best_articles):
            worth_reading.append(
                WorthReadingItemV1(
                    reading_id=f"read:{idx}:{i2}:{article.article_id}",
                    title=article.title,
                    source_id=article.source_id,
                    article_id=article.article_id,
                    url=article.url,
                    reason_selected="representative link from curated cluster",
                    reading_type="quick_read",
                    trust_tier=article.source_trust_tier,
                    category=category,
                    topic_ids=[topic.topic_id],
                    priority=1,
                    created_at=now,
                )
            )
        worth_watching.append(
            WorthWatchingItemV1(
                watch_id=f"watch:{idx}:{topic.topic_id}",
                topic_id=topic.topic_id,
                title=cluster.title,
                reason=f"cluster with {len(cluster_articles)} articles",
                watch_condition="recheck in 24h",
                category=category,
                region_scope=representative.region_scope,
                confidence=cluster_confidence,
                volatility=brief.volatility,
                priority=1,
                created_at=now,
            )
        )
    capped_situation_changes = len(changes) > max_situation_changes_per_run
    changes = changes[:max_situation_changes_per_run]
    run.metrics["capped_situation_changes"] = capped_situation_changes
    claims = apply_corroboration(claims)
    curated_items = curate_digest_items(
        items=items,
        required_sections=list(registry.required_sections or []),
        max_total=max_digest_items_total,
        max_per_section=max_digest_items_per_section,
        min_per_required=min_digest_items_per_required,
    )
    digest = build_digest(
        rid,
        curated_items,
        worth_reading[:max_worth_reading],
        worth_watching[:max_worth_watching],
    )
    capsule = build_world_context_capsule(
        digest,
        locality=settings.world_pulse_locality,
        max_topics=settings.world_pulse_stance_max_topics,
        min_confidence=settings.world_pulse_min_stance_confidence,
    )
    graph_eligible_item_ids = {
        l.learning_id.replace("learning:", "item:")
        for l in learning
        if l.graph_eligible and l.confidence >= 0.5
    }
    graph_delta = build_graph_delta(digest, dry_run=settings.world_pulse_graph_dry_run, allowed_item_ids=graph_eligible_item_ids)
    coverage_status, section_coverage, missing_required, missing_recommended, covered_sections = _compute_coverage(
        registry=registry,
        all_articles=all_articles,
        digest_items=items,
    )
    section_rollups = build_section_rollups(section_coverage=section_coverage, digest_items=curated_items)
    _finalize_digest_aggregates(
        digest=digest,
        run=run,
        all_articles=all_articles,
        clusters=clusters,
        curated_items=curated_items,
        coverage_status=coverage_status,
        section_coverage=section_coverage,
        section_rollups=section_rollups,
        max_digest_items_total=max_digest_items_total,
    )
    run.metrics["missing_required_sections"] = missing_required
    run.metrics["missing_recommended_sections"] = missing_recommended
    run.metrics["covered_sections"] = covered_sections
    run.metrics["worth_reading_count"] = len(digest.things_worth_reading)
    run.metrics["worth_watching_count"] = len(digest.things_worth_watching)
    logger.info(
        "world_pulse_coverage_result run_id=%s coverage_status=%s missing_required_sections=%s missing_recommended_sections=%s covered_sections=%s",
        rid,
        coverage_status,
        missing_required,
        missing_recommended,
        covered_sections,
    )
    run.completed_at = datetime.now(timezone.utc)
    run.status = "partial" if required_source_failures else "completed"
    run.claims_extracted = len(claims)
    run.events_extracted = len(events)
    run.entities_extracted = len(entities)
    run.situation_briefs_updated = len(briefs)
    run.situation_changes_created = len(changes)
    run.digest_created = True
    run.sql_emit_status = "pending"
    run.graph_emit_status = "pending"
    run.stance_capsule_status = "created"
    logger.info("world_pulse_digest_created run_id=%s items=%s", rid, len(items))
    return WorldPulseRunResultV1(
        run=run,
        digest=digest,
        capsule=capsule,
        graph_delta_plan=graph_delta,
        publish_status={
            "claims_count": len(claims),
            "events_count": len(events),
            "entities_count": len(entities),
            "briefs_count": len(briefs),
            "learning_count": len(learning),
            "changes_count": len(changes),
            "articles": [a.model_dump(mode="json") for a in all_articles],
            "clusters": [c.model_dump(mode="json") for c in clusters],
            "claims": [c.model_dump(mode="json") for c in claims],
            "events": [e.model_dump(mode="json") for e in events],
            "entities": [e.model_dump(mode="json") for e in entities],
            "briefs": [b.model_dump(mode="json") for b in briefs],
            "changes": [c.model_dump(mode="json") for c in changes],
            "learning": [l.model_dump(mode="json") for l in learning],
        },
    )
