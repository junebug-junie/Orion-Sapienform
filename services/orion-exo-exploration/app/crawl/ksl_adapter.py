"""KSL classifieds category-page + listing-detail-page fetch and parse.

Mirrors services/orion-world-pulse/app/services/ingest/base.py's fetch-helper
shape (honest user-agent, bounded fetch via `requests`) without importing
from orion-world-pulse -- that service's `ArticleCandidate`/`trust_tier`/
`region_scope` schema is for world-news-into-Orion's-cognition, a different
domain than classifieds (checked and rejected as a host for this, see the
service README).

Real structure confirmed live 2026-09-04 against
https://classifieds.ksl.com/search/cat/Electronics (and Computers, FREE):
category cards render as plain server-rendered HTML (no JS execution
required), each one an `<a>` tag carrying everything needed to build a
candidate:

    <a class="... search-result" aria-label="TITLE" data-item-id="ID"
       href="https://classifieds.ksl.com/listing/ID" role="listitem" ...>
      ...
      <div ... aria-label="Price $100.00">$100.00</div>

Detail pages additionally carry the seller-written description as one or
more `<p class="mb-4 last-of-type:mb-0">...</p>` tags inside the
"Description" tab panel, and a `Posted <Mon DD, YYYY>` stat in the page-stats
block. See tests/fixtures/ksl_category_sample.html for a trimmed real sample
(3 Electronics + 3 Computers + 2 FREE cards, byte-for-byte from the live
fetch that informed this parser).
"""
from __future__ import annotations

import html as html_module
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from app.crawl.dedup import external_listing_id_from_url
from app.settings import settings

logger = logging.getLogger("orion-exo-exploration.ksl_adapter")

_CARD_RE = re.compile(
    r'aria-label="(?P<title>[^"]*)"\s+data-item-id="(?P<item_id>\d+)"\s+href="(?P<href>[^"]*)"',
)
_PRICE_RE = re.compile(r'aria-label="Price\s+([^"]*)"')
_DESCRIPTION_P_RE = re.compile(r'<p class="mb-4 last-of-type:mb-0">(.*?)</p>', re.DOTALL)
_POSTED_RE = re.compile(r'aria-label="Posted ([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")

# How far past each card's opening tag to look for its price -- generous
# enough for the real markup's card body, tight enough to never bleed into
# the next card.
_PRICE_LOOKAHEAD_CHARS = 3000
_DESCRIPTION_LOOKAHEAD_CHARS = 4000


@dataclass(frozen=True)
class KslCandidate:
    external_listing_id: str
    url: str
    title: str
    price: Optional[float]
    price_raw: str
    source_category: str


def parse_price(price_raw: str) -> Optional[float]:
    """"$100.00" -> 100.0, "$17,500.00" -> 17500.0, "FREE" -> 0.0,
    "Call for quote" (or anything else non-numeric) -> None.
    """
    raw = price_raw.strip()
    if raw.upper() == "FREE":
        return 0.0
    cleaned = raw.replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_category_page(html: str, *, category_url: str) -> list[KslCandidate]:
    candidates: list[KslCandidate] = []
    for match in _CARD_RE.finditer(html):
        href = match.group("href")
        listing_id = match.group("item_id")
        parsed_id = external_listing_id_from_url(href)
        if parsed_id and parsed_id != listing_id:
            # data-item-id and the URL disagree -- trust neither blindly,
            # skip rather than guess.
            logger.warning(
                "ksl_card_id_mismatch data_item_id=%s url_id=%s href=%s",
                listing_id, parsed_id, href,
            )
            continue
        title = html_module.unescape(match.group("title")).strip()
        if not title or not listing_id:
            continue
        window = html[match.end(): match.end() + _PRICE_LOOKAHEAD_CHARS]
        price_match = _PRICE_RE.search(window)
        price_raw = price_match.group(1).strip() if price_match else ""
        price = parse_price(price_raw) if price_raw else None
        candidates.append(
            KslCandidate(
                external_listing_id=listing_id,
                url=href,
                title=title,
                price=price,
                price_raw=price_raw,
                source_category=category_url,
            )
        )
    return candidates


def parse_detail_description(html: str) -> Optional[str]:
    idx = html.find(">Description<")
    if idx == -1:
        return None
    window = html[idx: idx + _DESCRIPTION_LOOKAHEAD_CHARS]
    paragraphs = _DESCRIPTION_P_RE.findall(window)
    if not paragraphs:
        return None
    cleaned = [html_module.unescape(_TAG_RE.sub("", p)).strip() for p in paragraphs]
    cleaned = [c for c in cleaned if c]
    return "\n".join(cleaned) if cleaned else None


def parse_posted_date(html: str) -> Optional[datetime]:
    match = _POSTED_RE.search(html)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parsed = datetime.strptime(raw, "%b %d, %Y")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def fetch_category_html(category_url: str) -> str:
    resp = requests.get(
        category_url,
        timeout=settings.exo_exploration_fetch_timeout_seconds,
        headers={"User-Agent": settings.exo_exploration_user_agent},
    )
    resp.raise_for_status()
    return resp.text


def fetch_detail_html(listing_url: str) -> str:
    resp = requests.get(
        listing_url,
        timeout=settings.exo_exploration_fetch_timeout_seconds,
        headers={"User-Agent": settings.exo_exploration_user_agent},
    )
    resp.raise_for_status()
    return resp.text


def polite_delay() -> None:
    time.sleep(settings.exo_exploration_request_delay_seconds)
