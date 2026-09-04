from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from psycopg2.extras import Json, RealDictCursor

from app.storage.ddl import (
    CRAWL_RUNS_DDL,
    INTEREST_RULES_DDL,
    LISTINGS_CURRENT_DDL,
    LISTINGS_OBSERVED_DDL,
)
from app.storage.pg import pg_conn


logger = logging.getLogger("orion-exo-exploration.repository")

# Starter tech/compute keyword list -- the extensibility seam described in the
# design doc: a future "add keywords from Hub" feature is a new row in
# exo_exploration_interest_rules, not a schema change. Seeded once; re-running
# ensure_tables() must never duplicate these on restart.
_SEED_KEYWORDS: list[str] = [
    "gpu", "rtx", "gtx", "radeon", "cpu", "ryzen", "threadripper", "xeon",
    "server", "poweredge", "rack", "nas", "synology", "motherboard", "mobo",
    "ram", "ddr4", "ddr5", "ssd", "nvme", "raid", "network switch",
    "workstation", "mining rig", "psu", "power supply", "4k monitor",
    "ultrawide monitor", "thinkpad", "docking station",
]

# Documents which category URLs this crawl covers. These rows are
# informational only -- `category_url` set, `keyword` NULL, `weight=0` --
# and `rules_from_rows()` (app/crawl/interest.py) deliberately skips any row
# with no keyword, so they never contribute to a score. Reviewed 2026-09-04:
# without this comment, a `weight=0` row reads as a broken/dead rule rather
# than the scope-documentation row it is.
_SEED_CATEGORIES: list[str] = [
    "https://classifieds.ksl.com/search/cat/Electronics",
    "https://classifieds.ksl.com/search/cat/Computers",
    "https://classifieds.ksl.com/search/cat/FREE",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_tables() -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(LISTINGS_OBSERVED_DDL)
            cur.execute(LISTINGS_CURRENT_DDL)
            cur.execute(CRAWL_RUNS_DDL)
            cur.execute(INTEREST_RULES_DDL)
            cur.execute("SELECT COUNT(*) FROM exo_exploration_interest_rules")
            (existing_count,) = cur.fetchone()
            if existing_count == 0:
                now = utc_now()
                for category_url in _SEED_CATEGORIES:
                    cur.execute(
                        """
                        INSERT INTO exo_exploration_interest_rules (
                            rule_id, category_url, keyword, min_price, max_price, weight, added_by, created_at
                        ) VALUES (%s, %s, NULL, NULL, NULL, 0, 'seed', %s)
                        """,
                        (str(uuid4()), category_url, now),
                    )
                for keyword in _SEED_KEYWORDS:
                    cur.execute(
                        """
                        INSERT INTO exo_exploration_interest_rules (
                            rule_id, category_url, keyword, min_price, max_price, weight, added_by, created_at
                        ) VALUES (%s, NULL, %s, NULL, NULL, 1.0, 'seed', %s)
                        """,
                        (str(uuid4()), keyword, now),
                    )
                logger.info(
                    "exo_exploration_interest_rules_seeded categories=%d keywords=%d",
                    len(_SEED_CATEGORIES),
                    len(_SEED_KEYWORDS),
                )


# --- interest rules ---------------------------------------------------------


def list_interest_rules() -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM exo_exploration_interest_rules ORDER BY created_at ASC"
            )
            return cur.fetchall() or []


# --- observed (append-only) --------------------------------------------------


def insert_observed(
    *,
    source: str,
    source_category: str,
    external_listing_id: str,
    url: str,
    title: str,
    price: Optional[float],
    price_raw: Optional[str],
    description: Optional[str],
    posted_or_renewed_at: Optional[datetime],
    raw_content_hash: str,
    crawl_id: UUID,
    observed_at: Optional[datetime] = None,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO exo_exploration_listings_observed (
                    observed_id, source, source_category, external_listing_id, url, title,
                    price, price_raw, description, posted_or_renewed_at, raw_content_hash,
                    crawl_id, observed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid4()),
                    source,
                    source_category,
                    external_listing_id,
                    url,
                    title,
                    price,
                    price_raw,
                    description,
                    posted_or_renewed_at,
                    raw_content_hash,
                    str(crawl_id),
                    observed_at or utc_now(),
                ),
            )


# --- current (deduped, one row per external_listing_id) ---------------------


def upsert_current(
    *,
    external_listing_id: str,
    source: str,
    source_category: str,
    url: str,
    title: str,
    price: Optional[float],
    price_raw: Optional[str],
    description: Optional[str],
    posted_or_renewed_at: Optional[datetime],
    interest_score: float,
    interest_reasons: list[str],
    possible_duplicate_of: Optional[str],
    retention_days: int,
    seen_at: Optional[datetime] = None,
) -> None:
    """A renewal (same external_listing_id, new posted_or_renewed_at) updates
    the existing row -- it never inserts a second one. `times_seen` only
    increments on an actual conflict (a real second observation), never on
    the first insert.
    """
    now = seen_at or utc_now()
    expires_at = now + timedelta(days=retention_days)
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO exo_exploration_listings_current (
                    external_listing_id, source, source_category, url, title, price, price_raw,
                    description, posted_or_renewed_at, first_seen_at, last_seen_at, times_seen,
                    is_currently_listed, interest_score, interest_reasons, possible_duplicate_of,
                    expires_at
                ) VALUES (
                    %(external_listing_id)s, %(source)s, %(source_category)s, %(url)s, %(title)s,
                    %(price)s, %(price_raw)s, %(description)s, %(posted_or_renewed_at)s, %(now)s,
                    %(now)s, 1, TRUE, %(interest_score)s, %(interest_reasons)s,
                    %(possible_duplicate_of)s, %(expires_at)s
                )
                ON CONFLICT (external_listing_id) DO UPDATE SET
                    source = EXCLUDED.source,
                    source_category = EXCLUDED.source_category,
                    url = EXCLUDED.url,
                    title = EXCLUDED.title,
                    price = EXCLUDED.price,
                    price_raw = EXCLUDED.price_raw,
                    description = EXCLUDED.description,
                    posted_or_renewed_at = EXCLUDED.posted_or_renewed_at,
                    last_seen_at = EXCLUDED.last_seen_at,
                    times_seen = exo_exploration_listings_current.times_seen + 1,
                    is_currently_listed = TRUE,
                    interest_score = EXCLUDED.interest_score,
                    interest_reasons = EXCLUDED.interest_reasons,
                    possible_duplicate_of = EXCLUDED.possible_duplicate_of,
                    expires_at = EXCLUDED.expires_at
                """,
                {
                    "external_listing_id": external_listing_id,
                    "source": source,
                    "source_category": source_category,
                    "url": url,
                    "title": title,
                    "price": price,
                    "price_raw": price_raw,
                    "description": description,
                    "posted_or_renewed_at": posted_or_renewed_at,
                    "now": now,
                    "interest_score": interest_score,
                    "interest_reasons": Json(interest_reasons),
                    "possible_duplicate_of": possible_duplicate_of,
                    "expires_at": expires_at,
                },
            )


def get_current(external_listing_id: str) -> Optional[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM exo_exploration_listings_current WHERE external_listing_id = %s",
                (external_listing_id,),
            )
            return cur.fetchone()


def list_current_by_normalized(
    title_norm: str, price: Optional[float], source_category: str
) -> List[Dict[str, Any]]:
    """Candidates whose normalized (title, price, category) already match a
    current row -- used to flag possible duplicates under a *different*
    external_listing_id, never to merge them silently.
    """
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM exo_exploration_listings_current
                WHERE source_category = %s
                  AND lower(regexp_replace(title, '\\s+', ' ', 'g')) = %s
                  AND price IS NOT DISTINCT FROM %s
                """,
                (source_category, title_norm, price),
            )
            return cur.fetchall() or []


def list_finds(
    *,
    category: Optional[str] = None,
    min_interest: Optional[float] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM exo_exploration_listings_current"
    filters: List[str] = []
    params: List[Any] = []
    if category:
        filters.append("source_category = %s")
        params.append(category)
    if min_interest is not None:
        filters.append("interest_score >= %s")
        params.append(min_interest)
    if status == "active":
        filters.append("is_currently_listed = TRUE")
    elif status == "inactive":
        filters.append("is_currently_listed = FALSE")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY interest_score DESC, last_seen_at DESC LIMIT %s"
    params.append(limit)
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall() or []


def mark_not_seen_since_crawl(source_category: str, crawl_started_at: datetime) -> int:
    """The real "went off KSL" transition -- called once per successfully
    crawled category, right after that category's candidates are processed
    (app/crawl/daemon.py::run_crawl). Any current row in this category whose
    last_seen_at is older than THIS crawl's own start time was not touched
    by it, i.e. it no longer appeared on the category page.

    Review finding, confirmed live 2026-09-04: `mark_expired` used to set
    `is_currently_listed = FALSE` and then DELETE the same rows
    (`expires_at <= NOW()`) inside one transaction that only commits at the
    end -- no reader could ever observe a persisted FALSE row, so the
    "no longer listed" badge and the `status=inactive` filter in
    routers/finds.py were dead code paths. This function is the real
    lifecycle transition; `mark_expired` below now only ever deletes.
    """
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE exo_exploration_listings_current
                SET is_currently_listed = FALSE
                WHERE source_category = %s
                  AND last_seen_at < %s
                  AND is_currently_listed = TRUE
                """,
                (source_category, crawl_started_at),
            )
            return cur.rowcount


def mark_expired(retention_days: int) -> int:
    """Rows past 14 days since last_seen_at: delete both the current row and
    its observed history. Returns the number of rows deleted.

    Does NOT touch is_currently_listed -- see mark_not_seen_since_crawl for
    that transition. A row reaching this function's DELETE has already had
    every chance to be read as "no longer listed" for up to
    `retention_days` days; this is pure cleanup, not a lifecycle change.
    """
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM exo_exploration_listings_current WHERE expires_at <= NOW()"
            )
            deleted_current = cur.rowcount
            cur.execute(
                "DELETE FROM exo_exploration_listings_observed WHERE observed_at <= NOW() - (%s || ' days')::interval",
                (retention_days,),
            )
            deleted_observed = cur.rowcount
    return deleted_current + deleted_observed


# --- crawl runs ---------------------------------------------------------


def create_crawl_run(crawl_id: UUID, started_at: datetime) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO exo_exploration_crawl_runs (
                    crawl_id, started_at, categories_crawled, listings_seen, new_listings, errors, status
                ) VALUES (%s, %s, '[]'::jsonb, 0, 0, 0, 'running')
                """,
                (str(crawl_id), started_at),
            )


def finish_crawl_run(
    crawl_id: UUID,
    *,
    finished_at: datetime,
    categories_crawled: list[str],
    listings_seen: int,
    new_listings: int,
    errors: int,
    status: str,
) -> None:
    with pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE exo_exploration_crawl_runs
                SET finished_at = %s,
                    categories_crawled = %s,
                    listings_seen = %s,
                    new_listings = %s,
                    errors = %s,
                    status = %s
                WHERE crawl_id = %s
                """,
                (
                    finished_at,
                    Json(categories_crawled),
                    listings_seen,
                    new_listings,
                    errors,
                    status,
                    str(crawl_id),
                ),
            )


def list_crawl_runs(*, limit: int = 50) -> List[Dict[str, Any]]:
    with pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM exo_exploration_crawl_runs ORDER BY started_at DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall() or []
