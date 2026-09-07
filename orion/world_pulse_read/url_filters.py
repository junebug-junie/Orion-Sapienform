"""URL heuristics for world-pulse Stage 1 seeds.

Skip section-index pages before spending Wallet A — live 2026-09-07 burned a
debit on ``tomshardware.com/news`` (no article body → prose, no JSON handoff).
"""

from __future__ import annotations

from urllib.parse import urlparse

# Last path segment that usually means "listing", not "article".
_INDEX_SEGMENTS = frozenset(
    {
        "news",
        "blog",
        "articles",
        "stories",
        "posts",
        "index",
        "category",
        "categories",
        "tag",
        "tags",
        "topics",
        "section",
        "sections",
        "latest",
        "headlines",
    }
)


def url_looks_like_section_index(url: str) -> bool:
    """True when the URL is a section/listing page, not a single article."""
    raw = (url or "").strip()
    if not raw:
        return True
    try:
        parsed = urlparse(raw)
    except Exception:  # noqa: BLE001
        return False
    path = (parsed.path or "").rstrip("/")
    if not path:
        return True
    segments = [s for s in path.split("/") if s]
    if not segments:
        return True
    cleaned: list[str] = []
    for seg in segments:
        last = seg.lower()
        if "." in last:
            last = last.rsplit(".", 1)[0]
        cleaned.append(last)
    leaf = cleaned[-1]
    if leaf in _INDEX_SEGMENTS:
        return True
    # /category/ai — parent is a listing bucket and the leaf is not an article slug.
    parents = cleaned[:-1]
    if any(p in _INDEX_SEGMENTS for p in parents):
        if "-" in leaf or len(leaf) >= 24:
            return False
        return True
    return False
