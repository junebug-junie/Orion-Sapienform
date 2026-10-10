"""Link-accepted curiosity seeds, end to end on a disposable Postgres journal.

Real journal -> real AssertionProjector -> ACCEPTED_UNSEEDED_LINKS_SQL -> seed ->
neighborhood attach on the same store -> focal_edge_refs holds the projection edge.

ORION_SUBSTRATE_TEST_DATABASE_URL: admin DSN of a THROWAWAY Postgres server (same lane as
test_graph_journal_pg.py; CI: .github/workflows/substrate-neighborhood.yml, must not skip).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
from orion.substrate.assertion_projector import AssertionProjector, assertion_node_id
from orion.substrate.curiosity_seed_neighborhood import attach_seed_neighborhoods
from orion.substrate.evals.neighborhood_fixture import concept
from orion.substrate.graph_journal import SubstrateGraphJournal
from orion.substrate.link_accepted_seeds import (
    ACCEPTED_UNSEEDED_LINKS_SQL,
    accepted_links_params,
    link_accepted_seeds,
    link_seed_key,
    links_from_rows,
    projection_edge_id,
)
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reader_capability import ALWAYS_READY
from orion.substrate.store import InMemorySubstrateGraphStore

ADMIN_DSN = os.environ.get("ORION_SUBSTRATE_TEST_DATABASE_URL")
SQL = Path(__file__).resolve().parents[3] / "services" / "orion-sql-db"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_SUBSTRATE_TEST_DATABASE_URL not set")

READER = "world_pulse_read_stage2"
NOW = datetime.now(timezone.utc) - timedelta(minutes=30)


_DSN: dict[str, str] = {}


def _links_sync(lookback_hours: float, limit: int) -> list[dict]:
    """Exactly the store's path: SQLAlchemy text() + psycopg2 binds + mappings()."""
    from sqlalchemy import create_engine, text

    engine = create_engine(_DSN["current"].replace("postgresql://", "postgresql+psycopg2://", 1))
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(ACCEPTED_UNSEEDED_LINKS_SQL),
                                accepted_links_params(lookback_hours=lookback_hours, limit=limit)).mappings().all()
    finally:
        engine.dispose()
    return [{**dict(r), "edge_ids": json.loads(r["edge_ids"]) if isinstance(r["edge_ids"], str) else r["edge_ids"]}
            for r in rows]


async def _links(pool, *, lookback_hours: float = 24.0, limit: int = 10):
    return await asyncio.to_thread(_links_sync, lookback_hours, limit)


async def _store_seed_row(pool, seeds) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO substrate_endogenous_curiosity_candidates (candidate_set_id, generated_at, candidates_json)"
            " VALUES ($1, now(), $2::jsonb)",
            f"curiosity-{uuid.uuid4().hex[:12]}", json.dumps([s.model_dump(mode="json") for s in seeds]))


async def _with_db(fn):
    import asyncpg

    name = f"lks_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    _DSN["current"] = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = await asyncpg.create_pool(dsn=_DSN["current"], min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            for path in ("manual_migration_substrate_graph_journal_v1.sql",
                         "manual_migration_endogenous_curiosity_candidates_v1.sql",
                         "manual_migration_endogenous_curiosity_gate_json_v1.sql"):
                await conn.execute((SQL / path).read_text())
        await fn(pool)
    finally:
        await pool.close()
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()


def _store(*ids: str) -> InMemorySubstrateGraphStore:
    store = InMemorySubstrateGraphStore()
    for node_id in ids:
        node = concept(node_id).model_copy(update={"promotion_state": "proposed"})
        store.upsert_node(identity_key=f"concept|{node_id}", node=node)
    return store


class Claim:
    def __init__(self, subject: str, obj: str, *, actor: str = READER):
        self.key = f"{subject}|associated_with|{obj}|"
        self.target = assertion_node_id(self.key)
        self.proposal_id = f"prop-{uuid.uuid4().hex[:8]}"
        self.subject, self.obj, self.actor = subject, obj, actor

    def proposal(self) -> SubstrateGraphProposalV1:
        return SubstrateGraphProposalV1(
            proposal_id=self.proposal_id, proposal_kind="relationship_assertion", target_id=self.target,
            actor=self.actor, subject_node_id=self.subject, subject_kind="concept", object_node_id=self.obj,
            object_kind="concept", predicate="associated_with", statement_key=self.key,
            statement_text=f"the reading says {self.subject} goes with {self.obj}", anchor_scope="world",
            authority="local_inferred", recorded_at=NOW)

    def decision(self, prior: int, state: str, minutes: int = 1) -> SubstrateGraphDecisionV1:
        return SubstrateGraphDecisionV1(
            proposal_id=self.proposal_id, proposal_kind="relationship_assertion", target_id=self.target,
            actor=self.actor, decision_id=f"dec-{uuid.uuid4().hex[:8]}", expected_prior_revision=prior,
            resulting_state=state, policy="reading_quote_rule_v1", authority="local_inferred",
            recorded_at=NOW + timedelta(minutes=minutes))


async def _accept(journal, projector, claim: Claim, state: str = "provisional") -> None:
    await journal.append(claim.proposal())
    await journal.append(claim.decision(0, state))
    report = await projector.run_once()
    assert not report.failed, report.failed


def _projector(pool, store, actors=(READER,)):
    journal = SubstrateGraphJournal(pool)
    return journal, AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY, proposal_actors=actors)


def test_accepted_reading_claim_becomes_a_seed_whose_focal_edge_is_the_link():
    async def body(pool):
        store = _store("read-a", "read-b")
        journal, projector = _projector(pool, store)
        claim = Claim("read-a", "read-b")
        await _accept(journal, projector, claim)

        links, skipped = links_from_rows(await _links(pool))
        assert skipped == 0 and [l.assertion_id for l in links] == [claim.target]
        seeds = link_accepted_seeds(links, cap=2)
        assert seeds[0].focal_node_refs == ["read-a", "read-b"]
        assert link_seed_key(claim.target, 1) in seeds[0].notes

        stored, receipt = attach_seed_neighborhoods(seeds, store=store)
        edge = projection_edge_id(claim.target)
        assert stored[0].focal_edge_refs == [edge]
        assert stored[0].boundary_edge_refs == []
        assert stored[0].projection_endpoint_node_refs == ["read-a", "read-b"]
        assert receipt["nonempty"] == 1 and receipt["internal_edges"] == 1

        # Stored once -> never minted again (a retry/next tick finds nothing).
        await _store_seed_row(pool, stored)
        assert await _links(pool) == []
    asyncio.run(_with_db(body))


@pytest.mark.parametrize("later_state", ["rejected", "deprecated"])
def test_a_later_rejection_or_deprecation_mints_no_seed(later_state):
    async def body(pool):
        store = _store("read-a", "read-b")
        journal, projector = _projector(pool, store)
        claim = Claim("read-a", "read-b")
        await _accept(journal, projector, claim)
        await journal.append(claim.decision(1, later_state, minutes=2))
        await projector.run_once()
        assert await _links(pool) == []
        # And the graph agrees: the projection no longer walks.
        _, receipt = attach_seed_neighborhoods(
            link_accepted_seeds(links_from_rows([{
                "assertion_id": claim.target, "revision": 1, "subject_node_id": "read-a",
                "object_node_id": "read-b", "edge_ids": [projection_edge_id(claim.target)]}])[0], cap=1),
            store=store)
        assert receipt["nonempty"] == 0
    asyncio.run(_with_db(body))


def test_rejected_at_first_decision_mints_no_seed():
    async def body(pool):
        store = _store("read-a", "read-b")
        journal, projector = _projector(pool, store)
        await _accept(journal, projector, Claim("read-a", "read-b"), state="rejected")
        assert await _links(pool) == []
    asyncio.run(_with_db(body))


def test_accepted_but_not_yet_projected_waits():
    async def body(pool):
        journal = SubstrateGraphJournal(pool)
        claim = Claim("read-a", "read-b")
        await journal.append(claim.proposal())
        await journal.append(claim.decision(0, "provisional"))
        assert await _links(pool) == []
    asyncio.run(_with_db(body))


def test_only_reading_claims_and_the_limit_is_oldest_first():
    async def body(pool):
        store = _store("m-a", "m-b", *[f"read-{i}" for i in range(4)])
        journal, projector = _projector(pool, store, actors=None)
        await _accept(journal, projector, Claim("m-a", "m-b", actor="memory.referents"))
        claims = [Claim(f"read-{i}", f"read-{i + 1}") for i in range(3)]
        for claim in claims:
            await _accept(journal, projector, claim)
        rows = await _links(pool, limit=2)
        assert [r["assertion_id"] for r in rows] == [c.target for c in claims[:2]]
        assert len(link_accepted_seeds(links_from_rows(await _links(pool))[0], cap=2)) == 2
    asyncio.run(_with_db(body))


def test_lookback_excludes_old_links():
    async def body(pool):
        store = _store("read-a", "read-b")
        journal, projector = _projector(pool, store)
        await _accept(journal, projector, Claim("read-a", "read-b"))
        async with pool.acquire() as conn:
            # Throwaway DB only: the journal is append-only by trigger.
            await conn.execute("ALTER TABLE substrate_graph_journal DISABLE TRIGGER USER")
            await conn.execute("UPDATE substrate_graph_journal SET recorded_at = now() - interval '10 days'"
                               " WHERE event_kind = 'materialization'")
        assert await _links(pool, lookback_hours=24.0) == []
        assert len(await _links(pool, lookback_hours=24.0 * 30)) == 1
    asyncio.run(_with_db(body))


def test_a_seed_row_without_the_link_note_does_not_count_as_seeded():
    async def body(pool):
        store = _store("read-a", "read-b", "read-c")
        journal, projector = _projector(pool, store)
        first, second = Claim("read-a", "read-b"), Claim("read-b", "read-c")
        await _accept(journal, projector, first)
        await _accept(journal, projector, second)
        links, _ = links_from_rows(await _links(pool))
        await _store_seed_row(pool, link_accepted_seeds(links[:1], cap=1))   # only the first stored
        assert [r["assertion_id"] for r in await _links(pool)] == [second.target]
    asyncio.run(_with_db(body))
