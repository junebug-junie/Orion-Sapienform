from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.world_pulse import (
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionCoverageV1,
    SectionRollupV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
)


def build_digest(
    run_id: str,
    items: list[DailyWorldPulseItemV1],
    worth_reading: list[WorthReadingItemV1],
    worth_watching: list[WorthWatchingItemV1],
) -> DailyWorldPulseV1:
    now = datetime.now(timezone.utc)
    section_map: dict[str, list[str]] = {
        "us_politics": [],
        "global_politics": [],
        "local_politics": [],
        "ai_technology": [],
        "science_climate_energy": [],
        "healthcare_mental_health": [],
        "security_infrastructure_software": [],
        "hardware_compute_gpu": [],
        "local_conditions": [],
    }
    for item in items:
        if item.category in section_map:
            section_map[item.category].append(item.item_id)
    sections = DailyWorldPulseSectionsV1(**section_map)
    return DailyWorldPulseV1(
        run_id=run_id,
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary=f"{len(items)} tracked items with {len(worth_reading)} worth-reading and {len(worth_watching)} worth-watching entries.",
        sections=sections,
        items=items,
        things_worth_reading=worth_reading,
        things_worth_watching=worth_watching,
        orion_analysis_layer="Compared against previous situation briefs and captured deterministic changes.",
        source_notes=[],
        confidence_summary="Working confidence based on trust tiers and corroboration.",
        source_ids=sorted({sid for it in items for sid in it.source_ids}),
        article_count=len({aid for it in items for aid in it.article_ids}),
        claim_count=len({cid for it in items for cid in it.claim_ids}),
        event_count=len({eid for it in items for eid in it.event_ids}),
        situation_change_count=len({sid for it in items for sid in it.situation_change_ids}),
        created_at=now,
    )


def curate_digest_items(
    *,
    items: list[DailyWorldPulseItemV1],
    required_sections: list[str],
    max_total: int,
    max_per_section: int,
    min_per_required: int,
) -> list[DailyWorldPulseItemV1]:
    def _score(item: DailyWorldPulseItemV1) -> float:
        article_count = len(item.article_ids)
        source_count = len(set(item.source_ids))
        section_bonus = 0.2 if item.category in required_sections else 0.0
        consolidation_bonus = min(0.25, 0.08 * max(0, article_count - 1)) + min(0.2, 0.1 * max(0, source_count - 1))
        safety_bonus = 0.1 if item.category in {"security_infrastructure_software", "local_conditions"} else 0.0
        return item.confidence + section_bonus + consolidation_bonus + safety_bonus

    ranked = sorted(
        items,
        key=lambda i: (
            _score(i),
            i.created_at.timestamp(),
        ),
        reverse=True,
    )
    chosen: list[DailyWorldPulseItemV1] = []
    section_counts: dict[str, int] = {}

    # Reserve at least one slot per required section when available.
    for section in required_sections:
        if section_counts.get(section, 0) >= min_per_required:
            continue
        for item in ranked:
            if item.category != section:
                continue
            if item in chosen:
                continue
            chosen.append(item)
            section_counts[section] = section_counts.get(section, 0) + 1
            break

    for item in ranked:
        if len(chosen) >= max_total:
            break
        if item in chosen:
            continue
        current = section_counts.get(item.category, 0)
        if current >= max_per_section:
            continue
        chosen.append(item)
        section_counts[item.category] = current + 1
    return chosen


def build_section_rollups(
    *,
    section_coverage: dict[str, SectionCoverageV1],
    digest_items: list[DailyWorldPulseItemV1],
) -> list[SectionRollupV1]:
    rollups: list[SectionRollupV1] = []
    for section, coverage in section_coverage.items():
        section_items = [i for i in digest_items if i.category == section]
        top_topics = []
        for item in section_items[:3]:
            top_topics.extend(item.topic_ids[:1])
        summary = (
            f"{coverage.articles_accepted} accepted articles, "
            f"{len(section_items)} curated digest cards."
        )
        rollups.append(
            SectionRollupV1(
                section=section,
                status=coverage.status,
                article_count=coverage.articles_accepted,
                cluster_count=max(0, min(coverage.articles_accepted, coverage.sources_fetched)),
                digest_item_count=len(section_items),
                top_topic_ids=top_topics[:3],
                summary=summary,
                source_notes=[f"sources_enabled={coverage.sources_enabled}", f"sources_fetched={coverage.sources_fetched}"],
                confidence=1.0 if coverage.status == "covered" else 0.35,
            )
        )
    return rollups
