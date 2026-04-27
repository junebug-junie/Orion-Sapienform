from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests

from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def source_timeout_seconds(source: WorldPulseSourceV1, fallback: int) -> int:
    return source.fetch_timeout_seconds or fallback


def source_user_agent(source: WorldPulseSourceV1) -> str:
    return source.user_agent or "orion-world-pulse/0.1 (+approved-ingest)"


def source_domains(source: WorldPulseSourceV1) -> set[str]:
    parsed = urlparse(source.url or "")
    domains = {d.lower() for d in source.domains if d}
    if parsed.hostname:
        domains.add(parsed.hostname.lower())
    return domains


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return parsed.geturl()


def is_allowed_url(source: WorldPulseSourceV1, url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    allowed_domains = source_domains(source)
    if allowed_domains and host not in allowed_domains:
        return False
    prefixes = [p.strip() for p in source.allowed_path_prefixes if p.strip()]
    if prefixes and not any(parsed.path.startswith(prefix) for prefix in prefixes):
        return False
    return True


def bounded_urls(urls: Iterable[str], max_count: int) -> list[str]:
    out: list[str] = []
    for url in urls:
        normalized = normalize_url(url)
        if not normalized:
            continue
        out.append(normalized)
        if len(out) >= max_count:
            break
    return out


def http_get_text(url: str, *, timeout_seconds: int, user_agent: str) -> str:
    resp = requests.get(url, timeout=timeout_seconds, headers={"User-Agent": user_agent})
    resp.raise_for_status()
    return resp.text


def parse_xml(text: str) -> ElementTree.Element:
    return ElementTree.fromstring(text)


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def extract_links(html: str, *, base_url: str) -> list[str]:
    parser = _LinkCollector()
    parser.feed(html)
    return [urljoin(base_url, href) for href in parser.links]


def to_candidate(
    *,
    source: WorldPulseSourceV1,
    url: str,
    title: str,
    summary: str | None,
    published_at: datetime | None,
    discovered_via: str,
    fetched_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArticleCandidate | None:
    clean_title = title.strip()
    clean_url = normalize_url(url)
    if not clean_title or not clean_url:
        return None
    if not is_allowed_url(source, clean_url):
        return None
    return ArticleCandidate(
        source_id=source.source_id,
        source_name=source.name,
        url=clean_url,
        canonical_url=clean_url,
        title=clean_title,
        summary=(summary or "").strip() or None,
        excerpt=(summary or "").strip() or None,
        published_at=published_at,
        fetched_at=fetched_at or now_utc(),
        discovered_via=discovered_via,  # type: ignore[arg-type]
        metadata=metadata or {},
        trust_tier=source.trust_tier,
        categories=list(source.categories),
        region_scope=source.region_scope,
    )

