from __future__ import annotations

from app.services.ingest.base import bounded_urls, extract_links, http_get_text, to_candidate
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1


def fetch(source: WorldPulseSourceV1, fallback_timeout_seconds: int) -> list[ArticleCandidate]:
    if not source.url:
        return []
    html = http_get_text(
        source.url,
        timeout_seconds=source.fetch_timeout_seconds or fallback_timeout_seconds,
        user_agent=source.user_agent or "orion-world-pulse/0.1 (+approved-ingest)",
    )
    links = bounded_urls(extract_links(html, base_url=source.url), source.html_link_limit)
    out: list[ArticleCandidate] = []
    for link in links:
        title = link.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ").strip() or "Section link"
        candidate = to_candidate(
            source=source,
            url=link,
            title=title,
            summary=None,
            published_at=None,
            discovered_via="html_section",
            metadata={"strategy": "html_section"},
        )
        if candidate:
            out.append(candidate)
    return out

