"""Journal + AssertionProjector against a disposable Postgres (and FalkorDB when available).

ORION_SUBSTRATE_TEST_DATABASE_URL: admin DSN of a THROWAWAY Postgres server. Each test
creates and drops its own database. ORION_TEST_FALKOR_URI (optional) adds the real
Falkor lane. CI: .github/workflows/substrate-neighborhood.yml (fails on any skip).
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

from orion.core.schemas.cognitive_substrate import EvidenceNodeV1, SubstrateTemporalWindowV1
from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphProposalV1,
)
from orion.substrate.assertion_projector import AssertionProjector, assertion_node_id
from orion.substrate.graph_journal import RevisionConflict, SubstrateGraphJournal
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.neighborhood import NeighborhoodRequestV1
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.tests.test_assertion_core import FENCED, NOW, entity, prov

ADMIN_DSN = os.environ.get("ORION_SUBSTRATE_TEST_DATABASE_URL")
FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
SQL = Path(__file__).resolve().parents[3] / "services" / "orion-sql-db"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_SUBSTRATE_TEST_DATABASE_URL not set")

STATEMENT = "vincent|co_occurs_with|offsite|"
TARGET = assertion_node_id(STATEMENT)


def _run_sql(text: str) -> list[str]:
    code = "\n".join(re.sub(r"--.*$", "", line) for line in text.splitlines())
    return [stmt for stmt in code.split(";") if stmt.strip()]


async def _with_db(fn):
    import asyncpg

    name = f"sgj_{uuid.uuid4().hex[:10]}"
    admin = await asyncpg.connect(ADMIN_DSN)
    await admin.execute(f'CREATE DATABASE "{name}"')
    await admin.close()
    dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
    pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            for _ in range(2):  # idempotent
                for stmt in _run_sql((SQL / "manual_migration_substrate_graph_journal_v1.sql").read_text()):
                    await conn.execute(stmt)
        await fn(pool)
    finally:
        await pool.close()
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await admin.close()


def _proposal(**kw) -> SubstrateGraphProposalV1:
    return SubstrateGraphProposalV1(
        proposal_id=kw.pop("proposal_id", "prop-1"), proposal_kind="relationship_assertion", target_id=TARGET,
        actor=FENCED, subject_node_id="vincent", subject_kind="entity", object_node_id="offsite",
        object_kind="entity", predicate="co_occurs_with", statement_key=STATEMENT,
        statement_text="Juniper named Vincent and the Austin offsite together", anchor_scope="juniper",
        authority="user_asserted", supporting_evidence_ids=kw.pop("evidence", ["ev-mem-1"]),
        recorded_at=NOW, **kw)


def _decision(decision_id: str, prior: int, state: str, *, minutes: int = 1) -> SubstrateGraphDecisionV1:
    return SubstrateGraphDecisionV1(
        proposal_id="prop-1", proposal_kind="relationship_assertion", target_id=TARGET, actor=FENCED,
        decision_id=decision_id, expected_prior_revision=prior, resulting_state=state,
        policy="source_cooccurrence_v1", authority="local_inferred", recorded_at=NOW + timedelta(minutes=minutes))


def _seed(store) -> None:
    for node in (entity("vincent", "vincent", producer=FENCED, scope="juniper"),
                 entity("offsite", "austin offsite", producer=FENCED, scope="juniper"),
                 EvidenceNodeV1(node_id="ev-mem-1", evidence_type="episode_memory", content_ref="episode_memory:1",
                                anchor_scope="juniper", temporal=SubstrateTemporalWindowV1(observed_at=NOW),
                                provenance=prov(FENCED))):
        store.upsert_node(identity_key=f"fenced|{node.node_kind}|{node.node_id}", node=node)


def _stores():
    stores = [("memory", lambda: InMemorySubstrateGraphStore())]
    if FALKOR_URI:
        def falkor():
            from orion.graph.falkor_client import RedisGraphQueryClient
            from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

            client = RedisGraphQueryClient(uri=FALKOR_URI, graph_name=f"t_proj_{uuid.uuid4().hex[:10]}")
            return FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri=FALKOR_URI, graph_name="unused"),
                                        client=client, hydrate=False)
        stores.append(("falkor", falkor))
    return stores


def _walk(store) -> set[str]:
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("vincent",)))
    assert not result.degraded, result.reason
    return {e.edge_id for e in result.boundary_edges}


def test_journal_append_is_idempotent_and_enforces_the_expected_revision():
    async def body(pool):
        journal = SubstrateGraphJournal(pool)
        assert await journal.append(_proposal()) is True
        assert await journal.append(_proposal()) is False          # replayed producer writes nothing twice
        assert await journal.append(_decision("dec-1", 0, "provisional")) is True
        with pytest.raises(RevisionConflict):                       # a second claim on revision 1
            await journal.append(_decision("dec-1b", 0, "rejected"))
        assert [d.decision_id for d in await journal.pending_decisions()] == ["dec-1"]
        assert (await journal.proposal("prop-1")).statement_key == STATEMENT
    asyncio.run(_with_db(body))


@pytest.mark.parametrize("kind,make_store", _stores())
def test_accept_then_reject_projects_then_retracts_and_replays_idempotently(kind, make_store):
    async def body(pool):
        store = make_store()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store))
        await journal.append(_proposal())
        await journal.append(_decision("dec-1", 0, "provisional"))

        report = await projector.run_once()
        assert report.applied == ["dec-1"] and not report.failed
        node = store.get_node_by_id(TARGET)
        assert (node.node_kind, node.promotion_state, node.revision, node.decision_ref) == (
            "assertion", "provisional", 1, "dec-1")
        walked = _walk(store)
        assert len(walked) == 1
        projection = store.get_edge_by_id(next(iter(walked)))
        assert (projection.edge_role, projection.assertion_id, projection.assertion_revision) == (
            "semantic_projection", TARGET, 1)
        assert await journal.latest_applied_revision(TARGET) == 1
        assert (await projector.run_once()).applied == []           # nothing pending: no rewrite

        await journal.append(_decision("dec-2", 1, "rejected", minutes=5))
        assert (await projector.run_once()).applied == ["dec-2"]
        assert store.get_node_by_id(TARGET).promotion_state == "rejected"
        assert _walk(store) == set()                                # retracted, history kept
        closed = store.get_edge_by_id(projection.edge_id)
        assert closed.temporal.valid_to is not None and closed.assertion_revision == 2
    asyncio.run(_with_db(body))


def test_a_missing_endpoint_fails_closed_once_and_mints_no_placeholder():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store))
        await journal.append(_proposal(proposal_id="prop-1", evidence=["ev-missing"]))
        await journal.append(_decision("dec-1", 0, "provisional"))
        for _ in range(3):
            report = await projector.run_once()
            assert report.failed == {"dec-1": "endpoint_missing:ev-missing"}
        assert store.get_node_by_id(TARGET) is None and store.get_node_by_id("ev-missing") is None
        async with pool.acquire() as conn:
            failed = await conn.fetchval("SELECT count(*) FROM substrate_graph_journal "
                                         "WHERE event_kind='materialization' AND outcome='failed'")
        assert failed == 1                                          # recorded once, not every tick
        # The evidence lands later: the same decision now applies.
        store.upsert_node(identity_key="fenced|evidence|ev-missing", node=EvidenceNodeV1(
            node_id="ev-missing", evidence_type="episode_memory", content_ref="episode_memory:2",
            anchor_scope="juniper", temporal=SubstrateTemporalWindowV1(observed_at=NOW), provenance=prov(FENCED)))
        assert (await projector.run_once()).applied == ["dec-1"]
    asyncio.run(_with_db(body))


def test_revisions_apply_in_order_even_when_the_later_one_is_read_first():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store))
        await journal.append(_proposal())
        # dec-2 is recorded first in wall time but builds on revision 1.
        await journal.append(_decision("dec-2", 1, "canonical", minutes=1))
        await journal.append(_decision("dec-1", 0, "provisional", minutes=2))
        first = await projector.run_once()
        assert first.waiting == ["dec-2"] and first.applied == ["dec-1"]
        assert (await projector.run_once()).applied == ["dec-2"]
        assert store.get_node_by_id(TARGET).promotion_state == "canonical"
    asyncio.run(_with_db(body))


def test_rollback_drops_the_journal():
    async def body(pool):
        async with pool.acquire() as conn:
            for stmt in _run_sql((SQL / "manual_migration_substrate_graph_journal_v1_rollback.sql").read_text()):
                await conn.execute(stmt)
            assert await conn.fetchval("SELECT to_regclass('substrate_graph_journal')") is None
    asyncio.run(_with_db(body))
