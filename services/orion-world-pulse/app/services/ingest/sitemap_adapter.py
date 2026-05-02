from __future__ import annotations

from xml.etree import ElementTree

from app.services.ingest.base import (
    bounded_urls,
    http_get_text,
    is_allowed_url,
    parse_xml,
    source_timeout_seconds,
    source_user_agent,
    to_candidate,
)
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1


def _loc_values(root: ElementTree.Element) -> list[str]:
    return [(node.text or "").strip() for node in root.findall(".//{*}loc") if (node.text or "").strip()]


def fetch(source: WorldPulseSourceV1, fallback_timeout_seconds: int) -> list[ArticleCandidate]:
    if not source.url:
        return []
    timeout = source_timeout_seconds(source, fallback_timeout_seconds)
    user_agent = source_user_agent(source)
    root = parse_xml(http_get_text(source.url, timeout_seconds=timeout, user_agent=user_agent))
    url_candidates: list[str] = []
    if root.tag.endswith("sitemapindex"):
        child_sitemaps = bounded_urls(_loc_values(root), source.sitemap_max_child_sitemaps)
        for sitemap_url in child_sitemaps:
            if not is_allowed_url(source, sitemap_url):
                continue
            child_root = parse_xml(http_get_text(sitemap_url, timeout_seconds=timeout, user_agent=user_agent))
            url_candidates.extend(_loc_values(child_root))
    else:
        url_candidates.extend(_loc_values(root))
    out: list[ArticleCandidate] = []
    for url in bounded_urls(url_candidates, source.sitemap_max_urls):
        candidate = to_candidate(
            source=source,
            url=url,
            title=url.rsplit("/", 1)[-1].replace("-", " ").strip() or "Sitemap item",
            summary=None,
            published_at=None,
            discovered_via="sitemap",
            metadata={"strategy": "sitemap"},
        )
        if candidate:
            out.append(candidate)
    return out

