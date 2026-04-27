from __future__ import annotations

from app.services.ingest.models import ArticleCandidate
from app.services.ingest import html_section_adapter
from app.services.ingest import manual_urls_adapter
from app.services.ingest import rss_adapter
from app.services.ingest import sitemap_adapter
from orion.schemas.world_pulse import WorldPulseSourceV1


def fetch_source_candidates(source: WorldPulseSourceV1, fallback_timeout_seconds: int) -> list[ArticleCandidate]:
    strategy = source.strategy or "rss"
    if strategy in {"rss", "atom"}:
        return rss_adapter.fetch(source, fallback_timeout_seconds)
    if strategy == "sitemap":
        return sitemap_adapter.fetch(source, fallback_timeout_seconds)
    if strategy == "html_section":
        return html_section_adapter.fetch(source, fallback_timeout_seconds)
    if strategy == "manual_urls":
        return manual_urls_adapter.fetch(source, fallback_timeout_seconds)
    if strategy == "api":
        # API adapters are intentionally not enabled in this pass.
        return []
    raise ValueError(f"unsupported source strategy: {strategy}")

