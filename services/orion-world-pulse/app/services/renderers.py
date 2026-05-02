from __future__ import annotations

from collections import defaultdict

from orion.schemas.world_pulse import DailyWorldPulseV1, EmailWorldPulseRenderV1, HubWorldPulseMessageV1


def render_plaintext_digest(digest: DailyWorldPulseV1) -> str:
    lines = [
        digest.title,
        "",
        digest.executive_summary,
        "",
        f"Coverage status: {digest.coverage_status}",
        f"Accepted articles: {digest.accepted_article_count}",
        f"Curated digest cards: {len(digest.items)} (cap: {digest.max_digest_items_total})",
        "",
    ]
    if digest.section_rollups:
        lines.extend(["Section rollups:"])
        for rollup in digest.section_rollups:
            lines.append(
                f"- {rollup.section}: status={rollup.status}, articles={rollup.article_count}, digest_cards={rollup.digest_item_count}"
            )
        lines.append("")
    for item in digest.items:
        caveat_text = "; ".join(item.caveats) if item.caveats else "No major caveats noted."
        lines.extend(
            [
                item.category.upper(),
                "",
                f"{item.title}.",
                "",
                f"Why it matters: {item.why_it_matters}",
                "",
                f"What changed: {item.what_changed}",
                "",
                "Context:",
                *([f"- {x}" for x in item.context_bullets] or ["- No additional context"]),
                "",
                "By the numbers:",
                *([f"- {x}" for x in item.by_the_numbers] or ["- None reported"]),
                "",
                "What they're saying:",
                *([f"- {x}" for x in item.what_theyre_saying] or ["- No attributed commentary"]),
                "",
                f"Caveat: {caveat_text}",
                "",
                f"Orion's read: {item.orion_read}",
                "",
                "What to watch:",
                *([f"- {x}" for x in item.what_to_watch] or ["- Continue monitoring"]),
                "",
                "Worth reading:",
                *([f"- {x}" for x in item.worth_reading] or ["- No supplemental links"]),
                "",
                "Sources:",
                *([f"- {x}" for x in item.source_ids] or ["- Unknown"]),
                "",
            ]
        )
    if digest.things_worth_reading:
        lines.extend(["Worth reading:", *[f"- {w.title} ({w.url or 'no-url'})" for w in digest.things_worth_reading], ""])
    if digest.things_worth_watching:
        lines.extend(["Things worth watching:", *[f"- {w.title}: {w.watch_condition}" for w in digest.things_worth_watching], ""])
    return "\n".join(lines).strip()


def render_email_digest(digest: DailyWorldPulseV1, *, subject_prefix: str, to: list[str], from_email: str | None, dry_run: bool) -> EmailWorldPulseRenderV1:
    body = render_plaintext_digest(digest)
    return EmailWorldPulseRenderV1(
        run_id=digest.run_id,
        subject=f"{subject_prefix} — {digest.date}",
        opening="Good morning, Juniper.",
        plaintext_body=body,
        to=to,
        from_email=from_email,
        dry_run=dry_run,
        created_at=digest.created_at,
    )


def _normalized_structured_payload(digest: DailyWorldPulseV1) -> dict:
    structured = digest.model_dump(mode="json")
    item_article_ids = {
        aid
        for item in digest.items
        for aid in (item.article_ids or [])
        if isinstance(aid, str) and aid
    }
    item_source_ids = {
        sid
        for item in digest.items
        for sid in (item.source_ids or [])
        if isinstance(sid, str) and sid
    }
    inferred_section_coverage: dict[str, dict] = {}
    if isinstance(structured.get("section_coverage"), dict):
        inferred_section_coverage = dict(structured["section_coverage"])
    rollups = structured.get("section_rollups") if isinstance(structured.get("section_rollups"), list) else []
    rollup_map = {r.get("section"): r for r in rollups if isinstance(r, dict) and r.get("section")}
    counts_by_category: dict[str, int] = defaultdict(int)
    for item in digest.items:
        if item.category:
            counts_by_category[item.category] += 1

    for section, count in counts_by_category.items():
        existing = inferred_section_coverage.get(section, {}) if isinstance(inferred_section_coverage.get(section), dict) else {}
        roll = rollup_map.get(section, {})
        inferred_section_coverage[section] = {
            "sources_enabled": existing.get("sources_enabled", 0),
            "sources_fetched": existing.get("sources_fetched", 0),
            "articles_accepted": existing.get("articles_accepted", max(count, int(roll.get("article_count", 0) or 0))),
            "digest_items": existing.get("digest_items", max(count, int(roll.get("digest_item_count", 0) or 0))),
            "status": existing.get("status") or roll.get("status") or "covered",
        }

    accepted_count = structured.get("accepted_article_count")
    if not isinstance(accepted_count, int) or accepted_count <= 0:
        accepted_count = len(item_article_ids)
        if accepted_count <= 0:
            accepted_count = max((int((r or {}).get("article_count", 0) or 0) for r in rollups if isinstance(r, dict)), default=0)
    cluster_count = structured.get("article_cluster_count")
    if not isinstance(cluster_count, int) or cluster_count <= 0:
        cluster_count = max((int((r or {}).get("cluster_count", 0) or 0) for r in rollups if isinstance(r, dict)), default=0)
    if cluster_count <= 0 and digest.items:
        cluster_count = len(digest.items)
    max_digest_items_total = structured.get("max_digest_items_total")
    if not isinstance(max_digest_items_total, int) or max_digest_items_total <= 0:
        max_digest_items_total = max(len(digest.items), 12)
    coverage_status = structured.get("coverage_status")
    if (not coverage_status or coverage_status == "empty") and inferred_section_coverage:
        covered_sections = [k for k, v in inferred_section_coverage.items() if isinstance(v, dict) and v.get("status") == "covered"]
        coverage_status = "partial" if covered_sections else "empty"

    structured["accepted_article_count"] = accepted_count
    structured["article_cluster_count"] = cluster_count
    structured["max_digest_items_total"] = max_digest_items_total
    structured["section_coverage"] = inferred_section_coverage
    structured["coverage_status"] = coverage_status or "unknown"
    if not structured.get("source_ids"):
        structured["source_ids"] = sorted(item_source_ids)
    return structured


def render_hub_digest(digest: DailyWorldPulseV1) -> HubWorldPulseMessageV1:
    structured = _normalized_structured_payload(digest)
    structured["message_type"] = "daily_world_pulse"
    return HubWorldPulseMessageV1(
        message_id=f"world-pulse:{digest.run_id}",
        run_id=digest.run_id,
        title=digest.title,
        date=digest.date,
        executive_summary=digest.executive_summary,
        cards=digest.items,
        worth_reading=digest.things_worth_reading,
        worth_watching=digest.things_worth_watching,
        rendered_markdown=render_plaintext_digest(digest),
        structured_payload=structured,
        created_at=digest.created_at,
    )
