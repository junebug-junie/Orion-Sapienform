"""Dedup by KSL's own listing ID, plus a same-run possible-duplicate flag.

Two distinct notions of "the same listing" are handled here, and they must
stay distinct:

1. Renewal -- the *same* `external_listing_id` reappears (KSL lets a seller
   bump/renew a listing, which changes nothing about its identity). This is
   a normal UPDATE of the existing current row, never a new INSERT. That is
   enforced structurally by `ON CONFLICT (external_listing_id) DO UPDATE` in
   `app/storage/repository.py::upsert_current`, not by anything in this file.

2. Possible duplicate -- a *different* `external_listing_id` whose
   normalized (title, price, category) matches an existing current row.
   KSL listing IDs are assigned per-post, so a seller who deletes and
   reposts, or cross-posts the same item into two categories, produces two
   different IDs for what is plausibly the same physical item. This module
   only *flags* that (`possible_duplicate_of`) -- it never merges the rows,
   because a false positive here would silently make one real listing
   disappear from `/finds`.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

_LISTING_ID_RE = re.compile(r"/listing/(\d+)")
_WHITESPACE_RE = re.compile(r"\s+")


def external_listing_id_from_url(url: str) -> str | None:
    """Parse the canonical KSL listing id out of a listing URL.

    KSL's canonical listing URL shape, confirmed live 2026-09-04:
    https://classifieds.ksl.com/listing/<digits>. The id also appears as
    `data-item-id` on the category-page card and as `sku` in the listing's
    own ld+json block, but the URL is the one thing every caller in this
    service already has in hand.
    """
    match = _LISTING_ID_RE.search(urlparse(url).path)
    return match.group(1) if match else None


def normalize_title(title: str) -> str:
    """Lowercase, collapse whitespace. Must match the SQL-side normalization
    in `app/storage/repository.py::list_current_by_normalized`
    (`lower(regexp_replace(title, '\\s+', ' ', 'g'))`) exactly, or the
    possible-duplicate check silently stops matching anything.
    """
    return _WHITESPACE_RE.sub(" ", title).strip().lower()


def content_hash(*, external_listing_id: str, title: str, price_raw: str | None, description: str | None) -> str:
    """A stable fingerprint of the fields that matter for "did this listing
    change since we last saw it," stored on every observed row
    (`raw_content_hash`) for later inspection/dedup tooling.

    Review finding, 2026-09-04: an earlier version of this docstring claimed
    this hash is "used only to decide whether an observed-row insert is
    worth writing" -- that gating does not exist. `app/crawl/daemon.py`
    calls `insert_observed` unconditionally for every candidate on every
    crawl; nothing reads or compares `raw_content_hash` before that call.
    `listings_observed` is genuinely append-only by design (one row per
    observation, not deduped), so that is not itself a bug -- the bug was
    this docstring describing behavior the code does not have. Never used
    as the dedup key itself; that is always `external_listing_id`.
    """
    parts = [external_listing_id, title, price_raw or "", description or ""]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
