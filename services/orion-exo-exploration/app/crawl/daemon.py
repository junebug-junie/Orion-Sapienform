"""Two daemon loops, mirroring orion-topic-foundry's `drift_daemon_loop`
shape (`while True` / `try-except` / `await asyncio.sleep(interval)`),
started as `asyncio.create_task`s in `app/main.py`'s lifespan and cancelled
on shutdown:

- `crawl_loop`: once per `EXO_EXPLORATION_CRAWL_INTERVAL_SECONDS` (default
  86400 -- once a day), crawl all configured categories.
- `retention_sweep_loop`: once per
  `EXO_EXPLORATION_RETENTION_SWEEP_INTERVAL_SECONDS` (default hourly),
  expire rows past their `expires_at`.

The actual crawl-and-score-and-persist work lives in `run_crawl()` so the
docker-entrypoint daemon loop and a one-shot manual trigger (used by tests
and by an operator wanting a run right now) share one code path.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.crawl import ksl_adapter
from app.crawl.dedup import content_hash, normalize_title
from app.crawl.interest import rules_from_rows, score_candidate
from app.settings import settings
from app.storage import repository

logger = logging.getLogger("orion-exo-exploration.daemon")


def _category_urls() -> list[str]:
    return [c.strip() for c in settings.exo_exploration_categories.split(",") if c.strip()]


def run_crawl() -> dict:
    """One full crawl across every configured category. Synchronous (uses
    `requests`, not an async HTTP client) -- run via `asyncio.to_thread` from
    the async loop below so a slow KSL response never blocks the event loop.
    """
    crawl_id: UUID = uuid4()
    started_at = datetime.now(timezone.utc)
    repository.create_crawl_run(crawl_id, started_at)

    rules = rules_from_rows(repository.list_interest_rules())
    categories_crawled: list[str] = []
    listings_seen = 0
    new_listings = 0
    # Review finding, 2026-09-04: a whole-category fetch failure and a
    # single routine per-listing 403 (a listing removed between the category
    # fetch and its detail fetch is normal churn, not a run-level problem)
    # used to be counted together, so any one detail-fetch hiccup downgraded
    # a run where every category succeeded to status="partial". Tracked
    # separately now; only category_errors affects `status`. `errors` in
    # crawl_runs stays the combined total for visibility.
    category_errors = 0
    listing_errors = 0
    detail_fetches_done = 0

    for category_url in _category_urls():
        try:
            ksl_adapter.polite_delay()
            category_html = ksl_adapter.fetch_category_html(category_url)
        except Exception as exc:  # noqa: BLE001 -- one bad category must not kill the run
            logger.warning("exo_exploration_category_fetch_failed url=%s err=%s", category_url, exc)
            category_errors += 1
            continue

        categories_crawled.append(category_url)
        candidates = ksl_adapter.parse_category_page(category_html, category_url=category_url)

        for candidate in candidates:
            listings_seen += 1
            title_score, title_reasons = score_candidate(
                title=candidate.title, description=None, price=candidate.price, rules=rules
            )

            description = None
            posted_at = None
            final_score, final_reasons = title_score, title_reasons
            if (
                title_score > 0
                and detail_fetches_done < settings.exo_exploration_max_detail_fetches_per_run
            ):
                try:
                    ksl_adapter.polite_delay()
                    detail_html = ksl_adapter.fetch_detail_html(candidate.url)
                    detail_fetches_done += 1
                    description = ksl_adapter.parse_detail_description(detail_html)
                    posted_at = ksl_adapter.parse_posted_date(detail_html)
                    final_score, final_reasons = score_candidate(
                        title=candidate.title,
                        description=description,
                        price=candidate.price,
                        rules=rules,
                    )
                except Exception as exc:  # noqa: BLE001 -- keep the crawl going
                    logger.warning(
                        "exo_exploration_detail_fetch_failed url=%s err=%s", candidate.url, exc
                    )
                    listing_errors += 1

            # Review finding, 2026-09-04: these DB writes used to be
            # unguarded -- a transient Postgres error here would raise out
            # of run_crawl() before finish_crawl_run() ever ran, leaving
            # this crawl_runs row stuck at status="running" with no
            # finished_at forever. One bad candidate must not orphan the
            # whole run's own bookkeeping.
            try:
                existing_before = repository.get_current(candidate.external_listing_id)
                if existing_before is None:
                    new_listings += 1

                possible_duplicate_of = None
                if existing_before is None:
                    dupes = repository.list_current_by_normalized(
                        normalize_title(candidate.title), candidate.price, category_url
                    )
                    dupes = [d for d in dupes if d["external_listing_id"] != candidate.external_listing_id]
                    if dupes:
                        possible_duplicate_of = dupes[0]["external_listing_id"]
                        logger.info(
                            "exo_exploration_possible_duplicate new_id=%s existing_id=%s title=%r",
                            candidate.external_listing_id, possible_duplicate_of, candidate.title,
                        )

                repository.insert_observed(
                    source="ksl",
                    source_category=category_url,
                    external_listing_id=candidate.external_listing_id,
                    url=candidate.url,
                    title=candidate.title,
                    price=candidate.price,
                    price_raw=candidate.price_raw,
                    description=description,
                    posted_or_renewed_at=posted_at,
                    raw_content_hash=content_hash(
                        external_listing_id=candidate.external_listing_id,
                        title=candidate.title,
                        price_raw=candidate.price_raw,
                        description=description,
                    ),
                    crawl_id=crawl_id,
                )
                repository.upsert_current(
                    external_listing_id=candidate.external_listing_id,
                    source="ksl",
                    source_category=category_url,
                    url=candidate.url,
                    title=candidate.title,
                    price=candidate.price,
                    price_raw=candidate.price_raw,
                    description=description,
                    posted_or_renewed_at=posted_at,
                    interest_score=final_score,
                    interest_reasons=final_reasons,
                    possible_duplicate_of=possible_duplicate_of,
                    retention_days=settings.exo_exploration_retention_days,
                )
            except Exception as exc:  # noqa: BLE001 -- keep the crawl going
                logger.warning(
                    "exo_exploration_persist_failed external_listing_id=%s err=%s",
                    candidate.external_listing_id, exc,
                )
                listing_errors += 1

        # The real "went off KSL" transition for this category -- any
        # current row here whose last_seen_at predates this crawl's own
        # started_at was not touched above, i.e. it no longer appeared on
        # the category page. Only run for a category that was actually
        # fetched successfully (this line is unreached otherwise), so a
        # category-fetch failure can never be mistaken for real delisting.
        try:
            repository.mark_not_seen_since_crawl(category_url, started_at)
        except Exception as exc:  # noqa: BLE001 -- keep the crawl going
            logger.warning(
                "exo_exploration_mark_not_seen_failed category=%s err=%s", category_url, exc
            )
            listing_errors += 1

    finished_at = datetime.now(timezone.utc)
    total_errors = category_errors + listing_errors
    # Status reflects whether the crawl itself succeeded at the category
    # level -- a routine per-listing 403 (a listing pulled between the
    # category fetch and its own detail fetch) must not downgrade a run
    # where every configured category was actually reached.
    if category_errors == 0:
        status = "success"
    elif categories_crawled:
        status = "partial"
    else:
        status = "failed"
    repository.finish_crawl_run(
        crawl_id,
        finished_at=finished_at,
        categories_crawled=categories_crawled,
        listings_seen=listings_seen,
        new_listings=new_listings,
        errors=total_errors,
        status=status,
    )
    logger.info(
        "exo_exploration_crawl_finished crawl_id=%s status=%s listings_seen=%d new_listings=%d "
        "category_errors=%d listing_errors=%d",
        crawl_id, status, listings_seen, new_listings, category_errors, listing_errors,
    )
    return {
        "crawl_id": str(crawl_id),
        "status": status,
        "listings_seen": listings_seen,
        "new_listings": new_listings,
        "errors": total_errors,
    }


async def crawl_loop() -> None:
    if settings.exo_exploration_crawl_run_at_startup:
        try:
            await asyncio.to_thread(run_crawl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("exo_exploration_crawl_startup_failed err=%s", exc)
    while True:
        await asyncio.sleep(settings.exo_exploration_crawl_interval_seconds)
        try:
            await asyncio.to_thread(run_crawl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("exo_exploration_crawl_loop_failed err=%s", exc)


async def retention_sweep_loop() -> None:
    while True:
        try:
            deleted = await asyncio.to_thread(
                repository.mark_expired, settings.exo_exploration_retention_days
            )
            if deleted:
                logger.info("exo_exploration_retention_swept rows_deleted=%d", deleted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("exo_exploration_retention_sweep_failed err=%s", exc)
        await asyncio.sleep(settings.exo_exploration_retention_sweep_interval_seconds)
