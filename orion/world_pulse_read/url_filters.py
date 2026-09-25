"""URL heuristics for world-pulse Stage 1 seeds.

Skip listing / roundup / data-feed pages before spending Wallet A -- live
2026-09-07 burned a debit on ``tomshardware.com/news`` (no article body ->
prose, no JSON handoff), and live 2026-09-13..25 spent three Wallet A slots on
``networkworld.com/article/3562856/nvidia-latest-news-and-insights.html``, a
recurring topic hub that sits under ``/article/`` with an article-shaped slug.

Every rule below is justified by a live URL from ``world_pulse_read_seed``
(pulled 2026-09-25); tests/test_world_pulse_read_url_filters.py carries them.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# Path segment that usually means "listing", not "article".
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
        "topic",  # wccftech.com/topic/hardware/
        "topics",
        "section",
        "sections",
        "latest",
        "headlines",
        "media",  # silicondata.com/media
    }
)

# Machine-readable feeds, not a page a reader can read as an article
# (cisa.gov/.../known_exploited_vulnerabilities.json, 4 pending digest items).
_FEED_EXTENSIONS = frozenset({"json", "xml", "rss", "atom", "csv"})

# Slug words that, taken together, only ever name a listing. A leaf built
# entirely from these is a listing (usgs.gov/news/national-news-release).
_LISTING_WORDS = frozenset(
    {"national", "news", "release", "releases", "press", "latest", "all", "recent"}
)

# "latest" next to a news-ish word marks a roundup hub even inside an
# article-shaped slug (networkworld ".../nvidia-latest-news-and-insights") --
# but only when the slug is otherwise just listing words plus at most one topic
# word, so a real article slug ("nvidia_latest_gpu_news",
# "latest-updates-on-merger-with-x") is not caught.
_ROUNDUP_PARTNERS = frozenset({"news", "updates", "insights", "headlines", "stories"})
_ROUNDUP_FILLER = _LISTING_WORDS | _ROUNDUP_PARTNERS | {"and"}
_ROUNDUP_MAX_TOPIC_WORDS = 1

# An immediate parent that names a collection of *items* makes the leaf one
# item, whatever it looks like (bbc.co.uk/news/articles/cvgykzgljlyo,
# bbc.co.uk/news/videos/czxzwge59w8o). Before this, 36 BBC article URLs were
# skipped (5) or queued to be skipped (31) as `section_index_url`: "articles" is a
# listing word and the opaque ID has no "-" and is < 24 chars. Taxonomy
# parents (category/tag/topic/section) are deliberately NOT here: their leaf
# is another listing (/category/ai).
_ITEM_PARENTS = frozenset(
    {"articles", "article", "stories", "story", "posts", "post", "videos", "video", "item"}
)


def _leaf_tokens(leaf: str) -> list[str]:
    return [t for t in re.split(r"[-_]+", leaf) if t]


def url_looks_like_section_index(url: str) -> bool:
    """True when the URL is a section/listing/roundup page or a data feed,
    not a single readable article."""
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
    ext = ""
    for i, seg in enumerate(segments):
        last = seg.lower()
        if "." in last:
            last, seg_ext = last.rsplit(".", 1)
            if i == len(segments) - 1:
                ext = seg_ext
        cleaned.append(last)
    if ext in _FEED_EXTENSIONS:
        return True
    leaf = cleaned[-1]
    if leaf in _INDEX_SEGMENTS:
        return True
    tokens = _leaf_tokens(leaf)
    if (
        "latest" in tokens
        and any(t in _ROUNDUP_PARTNERS for t in tokens)
        and sum(t not in _ROUNDUP_FILLER for t in tokens) <= _ROUNDUP_MAX_TOPIC_WORDS
    ):
        return True
    parents = cleaned[:-1]
    if parents and parents[-1] in _ITEM_PARENTS:
        return False
    # /category/ai -- parent is a listing bucket and the leaf is not an article slug.
    if any(p in _INDEX_SEGMENTS for p in parents):
        if tokens and all(t in _LISTING_WORDS for t in tokens):
            return True
        if "-" in leaf or len(leaf) >= 24:
            return False
        return True
    return False
