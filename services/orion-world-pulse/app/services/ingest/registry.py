from __future__ import annotations

from urllib.parse import urlsplit

from app.services.ingest.models import ArticleCandidate
from app.services.ingest import arxiv_api_adapter
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
        # API adapters are keyed by the API host; an api source on any other
        # host has no adapter and fetches nothing.
        if _api_host(source) == arxiv_api_adapter.ARXIV_API_HOST:
            return arxiv_api_adapter.fetch(source, fallback_timeout_seconds)
        return []
    raise ValueError(f"unsupported source strategy: {strategy}")



def _api_host(source: WorldPulseSourceV1) -> str:
    urls = ([source.url] if source.url else []) + list(source.urls)
    hosts = {(urlsplit(u).hostname or "").lower() for u in urls if u}
    return hosts.pop() if len(hosts) == 1 else ""
