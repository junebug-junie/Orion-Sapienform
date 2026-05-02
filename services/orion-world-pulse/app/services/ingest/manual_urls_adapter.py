from __future__ import annotations

from app.services.ingest.base import bounded_urls, to_candidate
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1


def fetch(source: WorldPulseSourceV1, _: int) -> list[ArticleCandidate]:
    out: list[ArticleCandidate] = []
    for url in bounded_urls(source.urls, source.max_articles_per_day):
        title = url.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ").strip() or "Manual URL"
        candidate = to_candidate(
            source=source,
            url=url,
            title=title,
            summary="Manually curated approved-source URL.",
            published_at=None,
            discovered_via="manual_urls",
            metadata={"strategy": "manual_urls"},
        )
        if candidate:
            out.append(candidate)
    return out

