"""Postgres integration test for the real `ON CONFLICT ... DO UPDATE` upsert
path. Mirrors services/orion-substrate-telemetry/tests/test_integration_postgres.py's
conditional-skip convention: `pytest.mark.integration` (registered in
pyproject.toml, "optional tests requiring external services (set
RUN_INTEGRATION=1)") plus an explicit skip when RUN_INTEGRATION or a
reachable POSTGRES_URI/EXO_EXPLORATION_PG_DSN is absent, so a plain local
`pytest` run never fails for lacking Postgres.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.integration


def _pg_dsn() -> str:
    return (
        os.environ.get("EXO_EXPLORATION_PG_DSN")
        or os.environ.get("POSTGRES_URI")
        or "postgresql://postgres:postgres@localhost:55432/conjourney"
    )


@pytest.fixture
def repo():
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("RUN_INTEGRATION=1 not set")
    os.environ.setdefault("EXO_EXPLORATION_PG_DSN", _pg_dsn())

    import psycopg2

    try:
        conn = psycopg2.connect(_pg_dsn())
        conn.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unreachable: {exc}")

    from app.storage import repository

    repository.ensure_tables()
    yield repository


def test_ensure_tables_seeds_interest_rules_exactly_once(repo):
    rules = repo.list_interest_rules()
    assert len(rules) >= 30  # 3 category rows + 29 seed keywords
    keyword_rows = [r for r in rules if r["keyword"]]
    category_rows = [r for r in rules if r["category_url"]]
    assert len(category_rows) == 3
    assert any(r["keyword"] == "gpu" for r in keyword_rows)

    # ensure_tables is idempotent: calling it again must not duplicate seed rows.
    repo.ensure_tables()
    rules_again = repo.list_interest_rules()
    assert len(rules_again) == len(rules)


def test_upsert_current_insert_then_conflict_update_increments_times_seen(repo):
    external_id = f"test-{uuid.uuid4().hex[:12]}"
    try:
        repo.upsert_current(
            external_listing_id=external_id,
            source="ksl",
            source_category="https://classifieds.ksl.com/search/cat/Computers",
            url=f"https://classifieds.ksl.com/listing/{external_id}",
            title="Test GPU listing",
            price=500.0,
            price_raw="$500.00",
            description=None,
            posted_or_renewed_at=None,
            interest_score=1.0,
            interest_reasons=["keyword:'gpu' matched title/description (+1)"],
            possible_duplicate_of=None,
            retention_days=14,
        )
        row = repo.get_current(external_id)
        assert row is not None
        assert row["times_seen"] == 1
        assert float(row["interest_score"]) == 1.0

        # A renewal: same external_listing_id, updated fields -- must UPDATE
        # the existing row, never insert a second one.
        repo.upsert_current(
            external_listing_id=external_id,
            source="ksl",
            source_category="https://classifieds.ksl.com/search/cat/Computers",
            url=f"https://classifieds.ksl.com/listing/{external_id}",
            title="Test GPU listing (renewed)",
            price=475.0,
            price_raw="$475.00",
            description="Now with a description.",
            posted_or_renewed_at=datetime.now(timezone.utc),
            interest_score=1.5,
            interest_reasons=["keyword:'gpu' matched title/description (+1)", "price band (+0.5)"],
            possible_duplicate_of=None,
            retention_days=14,
        )
        row_after = repo.get_current(external_id)
        assert row_after["times_seen"] == 2
        assert row_after["title"] == "Test GPU listing (renewed)"
        assert float(row_after["price"]) == 475.0
        assert float(row_after["interest_score"]) == 1.5

        rows = repo.list_finds(category="https://classifieds.ksl.com/search/cat/Computers")
        matching = [r for r in rows if r["external_listing_id"] == external_id]
        assert len(matching) == 1  # never a second row for the same listing id
    finally:
        import psycopg2

        conn = psycopg2.connect(_pg_dsn())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM exo_exploration_listings_current WHERE external_listing_id = %s",
                    (external_id,),
                )
                cur.execute(
                    "DELETE FROM exo_exploration_listings_observed WHERE external_listing_id = %s",
                    (external_id,),
                )
            conn.commit()
        finally:
            conn.close()


def test_list_current_by_normalized_flags_possible_duplicate_candidate(repo):
    external_id = f"test-{uuid.uuid4().hex[:12]}"
    try:
        repo.upsert_current(
            external_listing_id=external_id,
            source="ksl",
            source_category="https://classifieds.ksl.com/search/cat/Computers",
            url=f"https://classifieds.ksl.com/listing/{external_id}",
            title="Duplicate Test Widget",
            price=42.0,
            price_raw="$42.00",
            description=None,
            posted_or_renewed_at=None,
            interest_score=0.0,
            interest_reasons=[],
            possible_duplicate_of=None,
            retention_days=14,
        )
        from app.crawl.dedup import normalize_title

        matches = repo.list_current_by_normalized(
            normalize_title("Duplicate Test Widget"), 42.0, "https://classifieds.ksl.com/search/cat/Computers"
        )
        assert any(m["external_listing_id"] == external_id for m in matches)

        no_matches = repo.list_current_by_normalized(
            normalize_title("Totally Different Title"), 42.0, "https://classifieds.ksl.com/search/cat/Computers"
        )
        assert not any(m["external_listing_id"] == external_id for m in no_matches)
    finally:
        import psycopg2

        conn = psycopg2.connect(_pg_dsn())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM exo_exploration_listings_current WHERE external_listing_id = %s",
                    (external_id,),
                )
            conn.commit()
        finally:
            conn.close()


def test_mark_not_seen_since_crawl_then_mark_expired_is_a_real_two_stage_lifecycle(repo):
    """Review finding, 2026-09-04: `mark_expired` used to UPDATE
    is_currently_listed=FALSE and DELETE the same rows (both on
    `expires_at <= NOW()`) inside one transaction that only commits at the
    end -- no reader could ever observe a persisted FALSE row before it was
    gone. This proves the real, now-separate lifecycle: a row not touched
    by a later crawl goes inactive but stays queryable, then is deleted only
    once genuinely past its retention window.
    """
    external_id = f"test-{uuid.uuid4().hex[:12]}"
    # A synthetic, uniquely-namespaced category rather than a real KSL
    # category URL -- this table also holds real production rows from the
    # live crawler (this service runs in Docker against the same DB this
    # test connects to), and mark_not_seen_since_crawl's WHERE clause
    # matches on source_category, so reusing a real category would also
    # flip every real row in it. Confirmed live: first version of this test
    # used the real Computers category URL and asserted `changed == 1`; it
    # actually returned 12, having just flipped 12 real live listings.
    category = f"https://test.invalid/category/{uuid.uuid4().hex[:8]}"
    try:
        first_seen = datetime.now(timezone.utc)
        repo.upsert_current(
            external_listing_id=external_id,
            source="ksl",
            source_category=category,
            url=f"https://classifieds.ksl.com/listing/{external_id}",
            title="Lifecycle Test GPU",
            price=100.0,
            price_raw="$100.00",
            description=None,
            posted_or_renewed_at=None,
            interest_score=1.0,
            interest_reasons=["keyword:'gpu' matched title/description (+1)"],
            possible_duplicate_of=None,
            retention_days=14,
            seen_at=first_seen,
        )
        row = repo.get_current(external_id)
        assert row["is_currently_listed"] is True

        # A later crawl of the same category that did NOT see this listing
        # again (it fell off KSL) -- started_at is after first_seen, so this
        # row's last_seen_at predates it.
        later_crawl_started_at = first_seen + timedelta(seconds=1)
        changed = repo.mark_not_seen_since_crawl(category, later_crawl_started_at)
        assert changed == 1

        row_after = repo.get_current(external_id)
        # The real assertion this bug broke: the row is STILL READABLE, and
        # its inactive state is now genuinely persisted and observable --
        # not deleted in the same breath it was marked.
        assert row_after is not None
        assert row_after["is_currently_listed"] is False

        inactive_finds = repo.list_finds(category=category, status="inactive")
        assert any(f["external_listing_id"] == external_id for f in inactive_finds)

        # mark_expired must NOT delete it yet -- expires_at is 14 days out,
        # only is_currently_listed changed, not retention.
        repo.mark_expired(retention_days=14)
        assert repo.get_current(external_id) is not None
    finally:
        import psycopg2

        conn = psycopg2.connect(_pg_dsn())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM exo_exploration_listings_current WHERE external_listing_id = %s",
                    (external_id,),
                )
            conn.commit()
        finally:
            conn.close()


def test_mark_expired_deletes_rows_past_retention_regardless_of_listed_state(repo):
    external_id = f"test-{uuid.uuid4().hex[:12]}"
    try:
        long_ago = datetime.now(timezone.utc) - timedelta(days=30)
        repo.upsert_current(
            external_listing_id=external_id,
            source="ksl",
            source_category="https://classifieds.ksl.com/search/cat/Computers",
            url=f"https://classifieds.ksl.com/listing/{external_id}",
            title="Expired Test Widget",
            price=10.0,
            price_raw="$10.00",
            description=None,
            posted_or_renewed_at=None,
            interest_score=0.0,
            interest_reasons=[],
            possible_duplicate_of=None,
            retention_days=14,
            seen_at=long_ago,  # expires_at = long_ago + 14 days, already past
        )
        assert repo.get_current(external_id) is not None
        repo.mark_expired(retention_days=14)
        assert repo.get_current(external_id) is None
    finally:
        import psycopg2

        conn = psycopg2.connect(_pg_dsn())
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM exo_exploration_listings_current WHERE external_listing_id = %s",
                    (external_id,),
                )
            conn.commit()
        finally:
            conn.close()
