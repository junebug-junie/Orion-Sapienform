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
from orion.substrate.reader_capability import ALWAYS_READY, ReadinessV1
from orion.substrate.store import InMemorySubstrateGraphStore
from orion.substrate.tests.test_assertion_core import FENCED, NOW, entity, prov

ADMIN_DSN = os.environ.get("ORION_SUBSTRATE_TEST_DATABASE_URL")
FALKOR_URI = os.getenv("ORION_TEST_FALKOR_URI", "").strip()
SQL = Path(__file__).resolve().parents[3] / "services" / "orion-sql-db"
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_SUBSTRATE_TEST_DATABASE_URL not set")

STATEMENT = "quill|co_occurs_with|retreat|"
TARGET = assertion_node_id(STATEMENT)


def _run_sql(text: str) -> list[str]:
    """The whole file as one script (it holds a plpgsql function body, so no naive split)."""
    return [text]


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
        actor=FENCED, subject_node_id="quill", subject_kind="entity", object_node_id="retreat",
        object_kind="entity", predicate="co_occurs_with", statement_key=STATEMENT,
        statement_text="Juniper named Quill and the spring retreat together", anchor_scope="juniper",
        authority="user_asserted", supporting_evidence_ids=kw.pop("evidence", ["ev-mem-1"]),
        recorded_at=NOW, **kw)


def _decision(decision_id: str, prior: int, state: str, *, minutes: int = 1) -> SubstrateGraphDecisionV1:
    return SubstrateGraphDecisionV1(
        proposal_id="prop-1", proposal_kind="relationship_assertion", target_id=TARGET, actor=FENCED,
        decision_id=decision_id, expected_prior_revision=prior, resulting_state=state,
        policy="source_cooccurrence_v1", authority="local_inferred", recorded_at=NOW + timedelta(minutes=minutes))


def _seed(store) -> None:
    for node in (entity("quill", "quill", producer=FENCED, scope="juniper"),
                 entity("retreat", "spring retreat", producer=FENCED, scope="juniper"),
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
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=("quill",)))
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
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
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
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
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
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
        await journal.append(_proposal())
        # dec-2 is recorded first in wall time but builds on revision 1.
        await journal.append(_decision("dec-2", 1, "canonical", minutes=1))
        await journal.append(_decision("dec-1", 0, "provisional", minutes=2))
        # dec-2 is not even listed until revision 1 has landed.
        first = await projector.run_once()
        assert first.applied == ["dec-1"] and first.waiting == []
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


def _prop(i: int, **kw) -> SubstrateGraphProposalV1:
    key = f"quill|co_occurs_with|retreat|{i}"
    return _proposal(proposal_id=f"prop-{i}", **kw).model_copy(update={
        "target_id": assertion_node_id(key), "statement_key": key})


def _dec(i: int, prior: int = 0, minutes: int = 1) -> SubstrateGraphDecisionV1:
    return _decision(f"dec-{i}", prior, "provisional", minutes=minutes).model_copy(update={
        "proposal_id": f"prop-{i}", "target_id": assertion_node_id(f"quill|co_occurs_with|retreat|{i}")})


def test_101_stuck_decisions_cannot_starve_a_new_one():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
        for i in range(101):  # transient poison: their evidence never lands
            await journal.append(_prop(i, evidence=["ev-never"]))
            await journal.append(_dec(i))
        await projector.run_once(limit=100)
        await projector.run_once(limit=100)  # every poison decision has now been attempted
        await journal.append(_prop(500))
        await journal.append(_dec(500, minutes=30))
        report = await projector.run_once(limit=100)
        assert "dec-500" in report.applied
    asyncio.run(_with_db(body))


def test_terminal_failures_leave_the_queue_and_a_b_a_is_three_records():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
        await journal.append(_decision("dec-orphan", 0, "provisional").model_copy(  # no proposal: terminal
            update={"proposal_id": "prop-missing", "target_id": assertion_node_id("orphan|x|y|")}))
        await projector.run_once()
        assert [d.decision_id for d in await journal.pending_decisions()] == []

        await journal.append(_proposal(proposal_id="prop-1", evidence=["ev-a", "ev-b"]))
        await journal.append(_decision("dec-1", 0, "provisional"))
        reasons = []
        for missing in ("ev-a", "ev-b", "ev-a"):  # A -> B -> A
            store._nodes.pop("ev-a", None)
            store._nodes.pop("ev-b", None)
            if missing == "ev-b":
                store.upsert_node(identity_key="x", node=EvidenceNodeV1(
                    node_id="ev-a", evidence_type="t", content_ref="t:1", anchor_scope="juniper",
                    temporal=SubstrateTemporalWindowV1(observed_at=NOW), provenance=prov(FENCED)))
            await projector.run_once()
            reasons.append((await journal.last_materialization("dec-1")).failure_reason)
        assert reasons == ["endpoint_missing:ev-a", "endpoint_missing:ev-b", "endpoint_missing:ev-a"]
        assert await journal.failed_attempts("dec-1") == 3
    asyncio.run(_with_db(body))


def test_the_journal_is_append_only_and_ids_are_namespaced_by_kind():
    import asyncpg

    async def body(pool):
        journal = SubstrateGraphJournal(pool)
        await journal.append(_proposal(proposal_id="same-id"))
        await journal.append(_decision("same-id", 0, "provisional").model_copy(update={"proposal_id": "same-id"}))
        async with pool.acquire() as conn:
            ids = sorted(r["event_id"] for r in await conn.fetch("SELECT event_id FROM substrate_graph_journal"))
            assert ids == ["decision:same-id", "proposal:same-id"]
            for sql in ("UPDATE substrate_graph_journal SET actor = 'x'", "DELETE FROM substrate_graph_journal"):
                with pytest.raises(asyncpg.RaiseError, match="append-only"):
                    await conn.execute(sql)
    asyncio.run(_with_db(body))


def test_nothing_is_written_until_every_reader_is_ready():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        journal = SubstrateGraphJournal(pool)
        state = {"ready": ReadinessV1(ready=False, missing=("orion-hub",), reason="readers_not_ready")}
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=lambda: state["ready"])
        await journal.append(_proposal())
        await journal.append(_decision("dec-1", 0, "provisional"))
        blocked = await projector.run_once()
        assert blocked.blocked.missing == ("orion-hub",) and blocked.applied == []
        assert store.get_node_by_id(TARGET) is None
        state["ready"] = ReadinessV1(ready=True)
        assert (await projector.run_once()).applied == ["dec-1"]
    asyncio.run(_with_db(body))


def test_a_held_id_is_refused_before_anything_is_written():
    async def body(pool):
        store = InMemorySubstrateGraphStore()
        _seed(store)
        from orion.substrate.reconcile import SubstrateIdentityResolver
        from orion.substrate.tests.test_assertion_core import assertion

        squatter = assertion("someone-else")
        store.upsert_node(identity_key=SubstrateIdentityResolver().canonical_node_key(
            assertion(TARGET)), node=squatter)
        journal = SubstrateGraphJournal(pool)
        projector = AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                       readiness=ALWAYS_READY)
        await journal.append(_proposal())
        await journal.append(_decision("dec-1", 0, "provisional"))
        report = await projector.run_once()
        assert report.failed == {"dec-1": "canonical_id_mismatch"}
        assert store.get_node_by_id(TARGET) is None and len(store._edges) == 0
    asyncio.run(_with_db(body))
