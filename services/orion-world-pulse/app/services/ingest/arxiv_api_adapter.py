"""arXiv papers for the daily world pulse, via the official export API.

One Atom query per configured category (``source.urls``, newest submissions
first). Papers are picked round-robin across those queries so no single
category fills the day, deduped by bare arXiv id against every arXiv URL Orion
has already queued or read (``world_pulse_read_seed``), and capped at
``source.max_articles_per_day``.

arXiv's API terms ask for no more than one request every three seconds and an
honest client identity; both are enforced here, not left to config:
https://info.arxiv.org/help/api/tou.html
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

from app.services.ingest.base import http_get_text, source_timeout_seconds, to_candidate
from app.services.ingest.models import ArticleCandidate
from orion.schemas.world_pulse import WorldPulseSourceV1

logger = logging.getLogger("orion-world-pulse.arxiv")

ARXIV_API_HOST = "export.arxiv.org"
# arXiv API terms of use: "no more than one request every three seconds".
MIN_REQUEST_SPACING_SECONDS = 3.0
# Honest identity: what this is and where it lives. Not a browser string.
DEFAULT_USER_AGENT = (
    "orion-world-pulse/0.1 (Orion-Sapienform daily research reading; "
    "+https://github.com/junebug-junie/Orion-Sapienform)"
)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# New-style ids (0704.0001 onward) and old-style archive ids (hep-th/9901001).
_NEW_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")
_OLD_ID = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")
_ARXIV_HOSTS = ("arxiv.org", "ar5iv.org")
_ID_PATH_PREFIXES = ("/abs/", "/pdf/", "/html/", "/format/")


def arxiv_id_from_url(url: str) -> str | None:
    """Bare arXiv id for any arXiv page URL, or None.

    abs/pdf/html/ar5iv and a ``vN`` version or ``.pdf`` suffix all name the
    same paper: ``https://arxiv.org/pdf/2509.07526v2.pdf``,
    ``https://ar5iv.labs.arxiv.org/html/2509.07526`` and
    ``http://arxiv.org/abs/2509.07526`` all return ``2509.07526``.
    """
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if not any(host == h or host.endswith("." + h) for h in _ARXIV_HOSTS):
        return None
    path = parts.path
    for prefix in _ID_PATH_PREFIXES:
        if path.startswith(prefix):
            rest = path[len(prefix):]
            if rest.endswith(".pdf"):
                rest = rest[: -len(".pdf")]
            rest = rest.strip("/")
            m = _NEW_ID.fullmatch(rest)
            if m:
                return m.group(1)
            m = _OLD_ID.fullmatch(rest)
            if m:
                return m.group(1)
            return None
    return None


@dataclass(frozen=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    summary: str
    published_at: datetime | None
    primary_category: str | None
    categories: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_arxiv_atom(xml_text: str) -> list[ArxivEntry]:
    """Entries from one arXiv API Atom response, in feed order. An entry whose
    id is not a recognisable arXiv paper id (the API's error entry, for one)
    is dropped."""
    root = ElementTree.fromstring(xml_text)
    out: list[ArxivEntry] = []
    for entry in root.findall(f"{_ATOM}entry"):
        arxiv_id = arxiv_id_from_url(entry.findtext(f"{_ATOM}id") or "")
        title = _clean(entry.findtext(f"{_ATOM}title"))
        if not arxiv_id or not title:
            continue
        primary = entry.find(f"{_ARXIV}primary_category")
        out.append(
            ArxivEntry(
                arxiv_id=arxiv_id,
                title=title,
                summary=_clean(entry.findtext(f"{_ATOM}summary")),
                published_at=_parse_iso(entry.findtext(f"{_ATOM}published")),
                primary_category=primary.attrib.get("term") if primary is not None else None,
                categories=[c.attrib.get("term", "") for c in entry.findall(f"{_ATOM}category") if c.attrib.get("term")],
                authors=[_clean(a.findtext(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")],
            )
        )
    return out


def select_papers(
    feeds: list[list[ArxivEntry]], *, seen_ids: set[str], cap: int
) -> list[ArxivEntry]:
    return [e for e, _ in select_papers_with_feed(feeds, seen_ids=seen_ids, cap=cap)]


def select_papers_with_feed(
    feeds: list[list[ArxivEntry]], *, seen_ids: set[str], cap: int
) -> list[tuple[ArxivEntry, int]]:
    """Round-robin over the per-category feeds (each newest first), skipping
    ids already seen and papers already picked via another feed, until
    ``cap`` papers are chosen. Each pick carries the index of the feed it was
    actually taken from."""
    chosen: list[tuple[ArxivEntry, int]] = []
    taken = set(seen_ids)
    cursors = [0] * len(feeds)
    while len(chosen) < cap:
        progressed = False
        for i, feed in enumerate(feeds):
            if len(chosen) >= cap:
                break
            while cursors[i] < len(feed):
                entry = feed[cursors[i]]
                cursors[i] += 1
                if entry.arxiv_id in taken:
                    continue
                taken.add(entry.arxiv_id)
                chosen.append((entry, i))
                progressed = True
                break
        if not progressed:
            break
    return chosen


class RequestSpacer:
    """Process-wide floor on time between arXiv API requests. Shared across
    runs, not per call, so two back-to-back runs still honour the 3 s rule."""

    def __init__(
        self,
        min_interval: float = MIN_REQUEST_SPACING_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._last is not None:
                remaining = self.min_interval - (self._clock() - self._last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last = self._clock()


_SPACER = RequestSpacer()

_history_engine: object | None = None
_history_engine_lock = threading.Lock()

# ar5iv.org does not contain "arxiv.org"; both patterns are needed.
SEEN_ARXIV_URLS_SQL = (
    "SELECT url FROM world_pulse_read_seed WHERE url ILIKE '%arxiv.org%' OR url ILIKE '%ar5iv.org%'"
)


def _get_history_engine():
    from app.settings import settings

    if not settings.world_pulse_read_history_db_url:
        return None
    global _history_engine
    if _history_engine is None:
        with _history_engine_lock:
            if _history_engine is None:
                from sqlalchemy import create_engine

                _history_engine = create_engine(settings.world_pulse_read_history_db_url, pool_pre_ping=True)
    return _history_engine


def seen_ids_from_urls(urls: Iterable[str]) -> set[str]:
    return {i for i in (arxiv_id_from_url(u) for u in urls) if i}


def load_seen_arxiv_ids() -> set[str] | None:
    """Bare ids of every arXiv URL already in the reading queue (any status:
    read, in flight, skipped). None when the history DB is unconfigured or
    unreachable -- the caller logs that; Hub's own exact-URL already_read gate
    still stops a true re-read of an abs URL."""
    engine = _get_history_engine()
    if engine is None:
        return None
    try:
        from sqlalchemy import text

        with engine.connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(text(SEEN_ARXIV_URLS_SQL)).fetchall()
        return seen_ids_from_urls(r[0] for r in rows)
    except Exception:
        logger.warning("world_pulse_arxiv_seen_ids_unavailable", exc_info=True)
        return None


def _query_category(url: str) -> str | None:
    q = parse_qs(urlsplit(url).query).get("search_query", [""])[0]
    m = re.fullmatch(r"cat:([\w.\-]+)", q.strip())
    return m.group(1) if m else None


def fetch(
    source: WorldPulseSourceV1,
    fallback_timeout_seconds: int,
    *,
    seen_ids_loader: Callable[[], set[str] | None] = load_seen_arxiv_ids,
    http_get: Callable[..., str] | None = None,
    spacer: RequestSpacer | None = None,
) -> list[ArticleCandidate]:
    query_urls = [u for u in ([source.url] if source.url else []) + list(source.urls) if u]
    if not query_urls:
        return []
    getter = http_get or http_get_text
    gate = spacer or _SPACER
    user_agent = source.user_agent or DEFAULT_USER_AGENT
    timeout = source_timeout_seconds(source, fallback_timeout_seconds)

    feeds: list[list[ArxivEntry]] = []
    feed_category: list[str | None] = []
    last_error: Exception | None = None
    failures = 0
    for url in query_urls:
        gate.wait()
        try:
            xml_text = getter(url, timeout_seconds=timeout, user_agent=user_agent)
            feeds.append(parse_arxiv_atom(xml_text))
        except Exception as exc:
            # One category's failure must not cost the others.
            logger.warning("world_pulse_arxiv_query_failed url=%s", url, exc_info=True)
            feeds.append([])
            failures += 1
            last_error = exc
        feed_category.append(_query_category(url))
    if last_error is not None and failures == len(query_urls):
        # Every query failed: raise so the pipeline records source_fetch_error
        # instead of an empty-but-"ok" source.
        raise last_error
    if not any(feeds):
        # HTTP 200 with zero usable entries everywhere (arXiv returns empty
        # feeds under load): not an error, but must not look like a quiet day.
        logger.warning("world_pulse_arxiv_all_feeds_empty source_id=%s queries=%s", source.source_id, len(query_urls))

    seen = seen_ids_loader()
    if seen is None:
        logger.warning("world_pulse_arxiv_dedupe_skipped reason=read_history_unavailable")
        seen = set()
    cap = int(source.max_articles_per_day)
    picks_with_feed = select_papers_with_feed(feeds, seen_ids=seen, cap=cap)
    picks = [e for e, _ in picks_with_feed]
    category_of = {e.arxiv_id: feed_category[i] for e, i in picks_with_feed}

    out: list[ArticleCandidate] = []
    for entry in picks:
        candidate = to_candidate(
            source=source,
            url=entry.abs_url,
            title=entry.title,
            summary=entry.summary,
            published_at=entry.published_at,
            discovered_via="api",
            metadata={
                "strategy": "api",
                "api": "arxiv",
                "arxiv_id": entry.arxiv_id,
                "query_category": category_of.get(entry.arxiv_id),
                "primary_category": entry.primary_category,
                "arxiv_categories": entry.categories,
                "authors": entry.authors[:8],
            },
        )
        if candidate:
            if entry.authors:
                candidate = candidate.model_copy(update={"author": ", ".join(entry.authors[:3])})
            out.append(candidate)
    logger.info(
        "world_pulse_arxiv_selected source_id=%s feeds=%s entries=%s seen=%s picked=%s ids=%s",
        source.source_id,
        len(feeds),
        sum(len(f) for f in feeds),
        len(seen),
        len(out),
        ",".join(c.metadata["arxiv_id"] for c in out),
    )
    return out
