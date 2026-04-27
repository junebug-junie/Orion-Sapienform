from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.schemas.world_pulse import ArticleClusterV1, ArticleRecordV1
from app.services.topic_normalization import TopicNormalizationResult, normalize_topic


@dataclass
class _ClusterState:
    normalization: TopicNormalizationResult
    articles: list[ArticleRecordV1] = field(default_factory=list)
    terms: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score_match(
    *,
    normalization: TopicNormalizationResult,
    state: _ClusterState,
    preserve_section_boundaries: bool,
) -> float:
    if preserve_section_boundaries and normalization.section != state.normalization.section:
        return 0.0
    token_jaccard = _jaccard(set(normalization.topic_terms), state.terms)
    entity_overlap = _jaccard(set(normalization.entities_hint), state.entities)
    section_match = 1.0 if normalization.section == state.normalization.section else 0.0
    region_match = 1.0 if normalization.region_scope == state.normalization.region_scope else 0.0
    source_category_match = 1.0 if normalization.topic_bucket == state.normalization.topic_bucket else 0.0
    return (
        0.45 * token_jaccard
        + 0.25 * entity_overlap
        + 0.15 * section_match
        + 0.10 * region_match
        + 0.05 * source_category_match
    )


def build_article_clusters(articles: list[ArticleRecordV1], *, clustering_policy: dict | None = None) -> list[ArticleClusterV1]:
    policy = clustering_policy or {}
    enabled = bool(policy.get("enabled", True))
    threshold = float(policy.get("similarity_threshold", 0.45))
    max_articles_per_cluster = int(policy.get("max_articles_per_cluster", 8))
    min_strong_terms = int(policy.get("min_strong_terms", 2))
    preserve_section_boundaries = bool(policy.get("preserve_section_boundaries", True))

    sorted_articles = sorted(articles, key=lambda a: a.published_at or a.fetched_at, reverse=True)
    states: list[_ClusterState] = []
    for article in sorted_articles:
        section = article.categories[0] if article.categories else "general_world"
        normalization = normalize_topic(article, section)
        if len(normalization.topic_terms) < min_strong_terms:
            normalization = TopicNormalizationResult(
                normalized_title_tokens=normalization.normalized_title_tokens,
                topic_terms=(normalization.topic_terms + [section, article.source_id])[:min_strong_terms],
                canonical_topic_key=normalization.canonical_topic_key,
                entities_hint=normalization.entities_hint,
                section=normalization.section,
                region_scope=normalization.region_scope,
                topic_bucket=normalization.topic_bucket,
            )

        if not enabled:
            state = _ClusterState(normalization=normalization)
            state.articles.append(article)
            state.terms.update(normalization.topic_terms)
            state.entities.update(normalization.entities_hint)
            states.append(state)
            continue

        best_idx = -1
        best_score = 0.0
        for idx, state in enumerate(states):
            if len(state.articles) >= max_articles_per_cluster:
                continue
            score = _score_match(
                normalization=normalization,
                state=state,
                preserve_section_boundaries=preserve_section_boundaries,
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score >= threshold:
            state = states[best_idx]
            state.articles.append(article)
            state.terms.update(normalization.topic_terms)
            state.entities.update(normalization.entities_hint)
        else:
            state = _ClusterState(normalization=normalization)
            state.articles.append(article)
            state.terms.update(normalization.topic_terms)
            state.entities.update(normalization.entities_hint)
            states.append(state)

    now = datetime.now(timezone.utc)
    out: list[ArticleClusterV1] = []
    for state in states:
        rows_sorted = sorted(state.articles, key=lambda a: a.published_at or a.fetched_at, reverse=True)
        representative = rows_sorted[0]
        category = representative.categories[0] if representative.categories else "general_world"
        top_terms = sorted(state.terms)[:5]
        key = f"{category}|{representative.region_scope}|{','.join(top_terms[:4])}"
        cluster_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        source_ids = {r.source_id for r in rows_sorted}
        avg_trust = sum(r.source_trust_tier for r in rows_sorted) / max(1, len(rows_sorted))
        out.append(
            ArticleClusterV1(
                cluster_id=f"cluster:{cluster_hash}",
                run_id=representative.run_id,
                title=representative.title,
                summary=f"{representative.title} ({len(rows_sorted)} articles across {len(source_ids)} sources)",
                article_ids=[r.article_id for r in rows_sorted],
                representative_article_id=representative.article_id,
                topic_ids=[],
                category=category,
                categories=sorted({c for r in rows_sorted for c in r.categories}) or [category],
                region_scope=representative.region_scope,
                topic_terms=top_terms,
                article_count=len(rows_sorted),
                source_ids=sorted(source_ids),
                source_count=len(source_ids),
                source_tiers_present=sorted({r.source_trust_tier for r in rows_sorted}),
                source_agreement=min(1.0, len(source_ids) / max(1, len(rows_sorted))),
                confidence=max(0.25, min(1.0, 1.0 - (avg_trust * 0.11) + (0.08 * (len(rows_sorted) - 1)))),
                created_at=now,
            )
        )
    return sorted(out, key=lambda c: (-c.confidence, -c.article_count, c.category, c.cluster_id))

