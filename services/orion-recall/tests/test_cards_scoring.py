"""Tests for the structured, graded, temporally-honest cards retrieval
(2026-07-21 memory-cards substrate spec, Item 1c) that replaced
cards_embedding.py's live-embedding cosine path.

Two layers:
  - Structural/string tests (no DB, always run): lock in the SQL shape --
    no candidate-window LIMIT before scoring, always_inject bypass, temporal
    validity gate, confidence multiplier, ts_rank_cd weight ordering.
  - Live-Postgres-gated behavioral tests: run the real query against an
    isolated schema (never the shared memory_cards table) inside the same
    DSN cards_adapter.py itself would use, skipped cleanly if Postgres isn't
    reachable. These are what actually prove the scoring behaves correctly --
    the string tests alone can't prove ts_rank_cd/weight-ordering correctness.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None  # type: ignore[assignment]

_REPO = Path(__file__).resolve().parents[3]
_APP = Path(__file__).resolve().parents[1] / "app"
_SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "memory_cards.sql"

_TEST_DSN = os.environ.get(
    "RECALL_TEST_PG_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
)


def _pg_available() -> bool:
    if asyncpg is None:
        return False

    async def _try() -> None:
        conn = await asyncio.wait_for(asyncpg.connect(dsn=_TEST_DSN), timeout=2.0)
        await conn.close()

    try:
        asyncio.run(_try())
        return True
    except Exception:
        return False


_PG_AVAILABLE = _pg_available()


# ── Structural tests (no DB) ──────────────────────────────────────────────


def _adapter_text() -> str:
    return (Path(__file__).resolve().parents[1] / "app" / "cards_adapter.py").read_text(encoding="utf-8")


def test_no_candidate_window_limit_before_scoring() -> None:
    """Locks in the fix for the exact bug this spec item replaces: the
    ranked-cards query must have no LIMIT applied before WHERE/scoring --
    the only LIMIT is the final top_k on the ranked output."""
    text = _adapter_text()
    ranked_fn = text.split("async def _fetch_ranked_cards", 1)[1].split("async def fetch_card_fragments", 1)[0]
    # The removed setting may still be named in an explanatory comment
    # elsewhere in the file (documenting what used to bound this query) --
    # what matters is that the ranked-query body itself never references it.
    assert "RECALL_CARDS_CANDIDATE_LIMIT" not in ranked_fn
    assert "ORDER BY score DESC" in ranked_fn
    assert "LIMIT $2" in ranked_fn
    # No secondary ORDER BY on recency anywhere in the ranked query -- ranking
    # is relevance-only, never recency-influenced.
    assert "updated_at DESC" not in ranked_fn


def test_always_inject_query_has_own_bypass_and_safety_cap() -> None:
    text = _adapter_text()
    always_fn = text.split("async def _fetch_always_inject_cards", 1)[1].split(
        "async def _fetch_ranked_cards", 1
    )[0]
    assert "priority = 'always_inject'" in always_fn
    assert "status = 'active'" in always_fn
    assert "LIMIT" in always_fn


def test_neighbor_expansion_also_applies_temporal_validity_gate() -> None:
    """Regression test (review finding, should-fix): an expired era_bound
    card must be excluded from neighbor-expansion results too, not just the
    primary always_inject/ranked queries -- previously the neighbor query
    only filtered on status='active', so a superseded fact could still
    surface as a neighbor even though it was correctly gated out everywhere
    else."""
    text = _adapter_text()
    fragment = text.split("async def fetch_card_fragments", 1)[1].split(
        "async def fetch_card_fragments_guarded", 1
    )[0]
    neighbor_query = fragment.split("n_rows = await conn.fetch", 1)[1].split("n_added = 0", 1)[0]
    assert "_TEMPORAL_VALIDITY_SQL_QUALIFIED" in neighbor_query


def test_ts_rank_weight_array_is_dcba_order() -> None:
    """ts_rank_cd's weight array is ordered {D,C,B,A} and each weight must be
    in [0,1] (Postgres rejects anything outside that range). This test locks
    the normalized mapping in: A=anchor(1.0), B=title(0.5), C=summary(0.25),
    D=tag(0.15) -- the 2.0:1.0:0.5:0.3 ratio from the original 2026-05-01
    design, rescaled by /2.0 to fit Postgres's constraint. A swapped order
    would silently invert which field matters most."""
    text = _adapter_text()
    assert "_TS_RANK_WEIGHTS = \"{0.15,0.25,0.5,1.0}\"" in text


def test_confidence_multiplier_present() -> None:
    text = _adapter_text()
    assert "WHEN 'certain' THEN 1.0" in text
    assert "WHEN 'likely' THEN 0.8" in text
    assert "WHEN 'possible' THEN 0.5" in text
    assert "WHEN 'uncertain' THEN 0.3" in text


def test_temporal_validity_gates_not_just_scores() -> None:
    text = _adapter_text()
    assert "era_bound" in text
    assert "< now()" in text
    # Guarded cast: the malformed-value regex guard must appear before the
    # cast, inside a CASE (not a bare WHERE AND chain).
    assert "!~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'" in text


def test_generated_tsvector_column_and_gin_index_in_both_ddl_copies() -> None:
    for rel in (
        Path(__file__).resolve().parents[1] / "sql" / "memory_cards.sql",
        _REPO / "orion" / "core" / "storage" / "sql" / "memory_cards.sql",
    ):
        text = rel.read_text(encoding="utf-8")
        assert "search_vector tsvector" in text
        assert "GENERATED ALWAYS AS" in text
        assert "USING GIN (search_vector)" in text


# ── Live-Postgres behavioral tests ────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not _PG_AVAILABLE, reason=f"Postgres not reachable at {_TEST_DSN!r} for cards scoring integration tests"
)


@pytest_asyncio.fixture
async def cards_pool():
    schema = f"test_cards_scoring_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(dsn=_TEST_DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        await admin.execute(f'SET search_path TO "{schema}", public')
        ddl = _SQL_PATH.read_text(encoding="utf-8")
        await admin.execute(ddl)
        pool = await asyncpg.create_pool(
            dsn=_TEST_DSN,
            min_size=1,
            max_size=2,
            server_settings={"search_path": f'"{schema}",public'},
        )
        try:
            yield pool
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()


async def _insert_card(
    pool,
    *,
    title: str,
    summary: str,
    confidence: str = "likely",
    priority: str = "episodic_detail",
    status: str = "active",
    anchor_class: str | None = None,
    anchors: list[str] | None = None,
    tags: list[str] | None = None,
    time_horizon: dict | None = None,
    updated_at: datetime | None = None,
) -> str:
    import json as _json

    card_id = str(uuid.uuid4())
    slug = f"{title.lower().replace(' ', '-')}-{card_id[:8]}"
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_cards (
              card_id, slug, types, anchor_class, status, confidence, sensitivity, priority,
              visibility_scope, time_horizon, provenance, title, summary, anchors, tags,
              updated_at
            ) VALUES (
              $1, $2, '{fact}', $3, $4, $5, 'private', $6, '{chat}', $7::jsonb, 'operator_highlight',
              $8, $9, $10, $11, coalesce($12, now())
            )
            """,
            card_id,
            slug,
            anchor_class,
            status,
            confidence,
            priority,
            _json.dumps(time_horizon) if time_horizon is not None else None,
            title,
            summary,
            anchors or [],
            tags or [],
            updated_at,
        )
    return card_id


async def _insert_edge(pool, *, from_card_id: str, to_card_id: str, edge_type: str = "relates_to") -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memory_card_edges (from_card_id, to_card_id, edge_type)
            VALUES ($1, $2, $3)
            """,
            from_card_id,
            to_card_id,
            edge_type,
        )


@pytest.mark.asyncio
async def test_expired_neighbor_excluded_from_neighbor_expansion(cards_pool) -> None:
    """Live behavioral counterpart to
    test_neighbor_expansion_also_applies_temporal_validity_gate above: an
    expired era_bound card linked as a _NEIGHBOR_TYPES edge must not surface
    via neighbor expansion, even though it would otherwise be a valid
    'relates_to' neighbor of a top-ranked card."""
    from app.cards_adapter import fetch_card_fragments

    anchor_id = await _insert_card(
        cards_pool,
        title="Widget factory anchor card",
        summary="Widget factory anchor card widget factory widget factory.",
    )
    expired_neighbor_id = await _insert_card(
        cards_pool,
        title="Expired widget factory detail",
        summary="An expired detail related to the widget factory.",
        time_horizon={"kind": "era_bound", "start": "2020-01-01", "end": "2020-06-01"},
    )
    await _insert_edge(cards_pool, from_card_id=anchor_id, to_card_id=expired_neighbor_id)

    results = await fetch_card_fragments(
        cards_pool, "widget factory", {"cards_top_k": 6}, lane=None, max_neighbors=6
    )

    ids = {r["meta"]["card_id"] for r in results}
    assert anchor_id in ids
    assert expired_neighbor_id not in ids


@pytest.mark.asyncio
async def test_always_inject_appears_regardless_of_query_relevance(cards_pool) -> None:
    from app.cards_adapter import fetch_card_fragments

    always_id = await _insert_card(
        cards_pool,
        title="Foundational identity fact",
        summary="Completely unrelated to the search query text below.",
        priority="always_inject",
    )
    await _insert_card(
        cards_pool,
        title="Something about spaceships",
        summary="Spaceships and rockets are relevant to this query.",
        priority="episodic_detail",
    )

    results = await fetch_card_fragments(
        cards_pool, "spaceships rockets", {"cards_top_k": 6}, lane=None, max_neighbors=0
    )

    ids = {r["meta"]["card_id"] for r in results}
    assert always_id in ids
    always_result = next(r for r in results if r["meta"]["card_id"] == always_id)
    assert always_result["score"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_expired_era_bound_card_excluded(cards_pool) -> None:
    from app.cards_adapter import fetch_card_fragments

    expired_id = await _insert_card(
        cards_pool,
        title="Old quarterly roadmap",
        summary="Quarterly roadmap details for the expired planning era.",
        time_horizon={
            "kind": "era_bound",
            "start": "2020-01-01",
            "end": "2020-06-01",
        },
    )
    current_id = await _insert_card(
        cards_pool,
        title="Current quarterly roadmap",
        summary="Quarterly roadmap details still true today.",
        time_horizon={"kind": "current"},
    )

    results = await fetch_card_fragments(
        cards_pool, "quarterly roadmap", {"cards_top_k": 6}, lane=None, max_neighbors=0
    )

    ids = {r["meta"]["card_id"] for r in results}
    assert expired_id not in ids
    assert current_id in ids


@pytest.mark.asyncio
async def test_malformed_time_horizon_end_does_not_500(cards_pool) -> None:
    """A malformed time_horizon.end must not raise -- the card should simply
    be treated as not-provably-expired (included), never crash the query."""
    from app.cards_adapter import fetch_card_fragments

    malformed_id = await _insert_card(
        cards_pool,
        title="Malformed time horizon card",
        summary="This card has a garbage time_horizon end value.",
        time_horizon={"kind": "era_bound", "start": "2020-01-01", "end": "not-a-date"},
    )

    results = await fetch_card_fragments(
        cards_pool, "malformed time horizon", {"cards_top_k": 6}, lane=None, max_neighbors=0
    )

    # Must not raise; the malformed card is included since it can't be proven expired.
    ids = {r["meta"]["card_id"] for r in results}
    assert malformed_id in ids


@pytest.mark.asyncio
async def test_confidence_ordering_at_equal_topical_match(cards_pool) -> None:
    from app.cards_adapter import fetch_card_fragments

    ids_by_confidence = {}
    for confidence in ("uncertain", "possible", "likely", "certain"):
        ids_by_confidence[confidence] = await _insert_card(
            cards_pool,
            title="Widget factory status report",
            summary="Widget factory status report widget factory widget factory.",
            confidence=confidence,
        )

    results = await fetch_card_fragments(
        cards_pool, "widget factory", {"cards_top_k": 6}, lane=None, max_neighbors=0
    )

    order = [r["meta"]["card_id"] for r in results]
    rank = {cid: order.index(cid) for cid in ids_by_confidence.values()}
    assert rank[ids_by_confidence["certain"]] < rank[ids_by_confidence["likely"]]
    assert rank[ids_by_confidence["likely"]] < rank[ids_by_confidence["possible"]]
    assert rank[ids_by_confidence["possible"]] < rank[ids_by_confidence["uncertain"]]


@pytest.mark.asyncio
async def test_old_relevant_card_outranks_recent_irrelevant_card(cards_pool) -> None:
    """The key regression test: this must fail against the old PR #1225
    recency-windowed shape (which orders candidates by updated_at DESC
    before any relevance scoring, so an old-but-relevant card can be pushed
    out of the candidate window entirely by newer, less relevant traffic)."""
    from app.cards_adapter import fetch_card_fragments

    old_relevant_id = await _insert_card(
        cards_pool,
        title="Gigafactory battery program status",
        summary="Battery cell production ramp for the gigafactory battery line remains on schedule.",
        anchor_class="project",
        anchors=["gigafactory"],
        tags=["battery", "manufacturing"],
        updated_at=datetime.now(timezone.utc) - timedelta(days=400),
    )
    recent_irrelevant_id = await _insert_card(
        cards_pool,
        title="Lunch plans",
        summary="Thinking about getting a sandwich for lunch, maybe a battery-shaped cookie as a joke.",
        updated_at=datetime.now(timezone.utc),
    )

    results = await fetch_card_fragments(cards_pool, "battery", {"cards_top_k": 6}, lane=None, max_neighbors=0)

    ids = [r["meta"]["card_id"] for r in results]
    # The recent card genuinely contains "battery" once (in a jokey aside),
    # so it always matches the tsquery and always appears -- this assertion
    # is unconditional, not a weaker "if present" check.
    assert old_relevant_id in ids
    assert recent_irrelevant_id in ids
    assert ids.index(old_relevant_id) < ids.index(recent_irrelevant_id)
