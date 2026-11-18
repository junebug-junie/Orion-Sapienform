"""Eval 5 of the memory Stage 2 spec ("graph discipline"), on throwaway Postgres + FalkorDB.

Label-free, deterministic, and it fails the build on any violation:
1. no walkable semantic edge without an accepted assertion: every edge_role=semantic_projection
   edge names an Assertion that is provisional/canonical at the SAME revision, or it is not
   returned by the neighborhood read;
2. no silent merge: every memory.referents node id is one the referent step minted, and no
   other producer's node gained a memory.referents provenance or alias;
3. rebuildable: ReferentProjector.rebuild() into an EMPTY graph gives identical node and edge
   id sets (assertions included) and writes nothing to the append-only journal.

Runs with the memory-episode CI job (ORION_MEMORY_EPISODE_TEST_DATABASE_URL + ORION_TEST_FALKOR_URI).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVICE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.memory.referents.tests.test_referents_pg import (  # noqa: E402
    NOW, _falkor, _persist, _seed_topic_foundry_circe, _table, _with_dbs,
    pytestmark,  # noqa: F401  (same skip rule)
)
from orion.substrate.reader_capability import ALWAYS_READY  # noqa: E402

spec = importlib.util.spec_from_file_location("mc_referent_projector_eval", SERVICE_ROOT / "app" / "referent_projector.py")
referent_projector = importlib.util.module_from_spec(spec)
sys.modules["mc_referent_projector_eval"] = referent_projector
spec.loader.exec_module(referent_projector)

PRODUCER = "memory.referents"


def _ids(store):
    state = store.snapshot()
    nodes = {n for n, v in state.nodes.items() if v.provenance.producer == PRODUCER}
    edges = {e for e, v in state.edges.items() if v.provenance.producer == PRODUCER}
    return state, nodes, edges


def test_graph_discipline_and_rebuild_from_postgres():
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.neighborhood import ACCEPTED_ASSERTION_STATES, NeighborhoodRequestV1

    async def body(pg, apg):
        _, client_a, store_a = _falkor()
        _, client_b, store_b = _falkor()
        try:
            _seed_topic_foundry_circe(store_a)
            await _persist(pg)
            await referent_projector.ReferentProjector(
                pool=apg, materializer=SubstrateGraphMaterializer(store=store_a), readiness=ALWAYS_READY).run_once(now=NOW)
            state, nodes, edges = _ids(store_a)

            # 1. every projection is backed by an accepted assertion at its revision, or unwalkable
            minted = {r["node_id"] for r in await _table(pg, "SELECT DISTINCT node_id FROM referent_alias")}
            for edge in (e for e in state.edges.values() if e.edge_role == "semantic_projection"):
                assertion = state.nodes[edge.assertion_id]
                accepted = (assertion.promotion_state in ACCEPTED_ASSERTION_STATES
                            and assertion.revision == edge.assertion_revision)
                walked = store_a.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=(edge.source.node_id,)))
                in_walk = edge.edge_id in {e.edge_id for e in walked.boundary_edges}
                assert in_walk <= accepted, f"{edge.edge_id} walkable without an accepted assertion"

            # 2. no silent merge
            referents = {n for n in nodes if state.nodes[n].node_kind in {"entity", "concept"}}
            assert referents <= minted
            assert state.nodes["tf-circe"].provenance.producer == "topic_foundry_adapter"
            assert not getattr(state.nodes["tf-circe"], "aliases", [])

            # 3. rebuild into an EMPTY graph through the one supported path (ReferentProjector.rebuild):
            #    same node and edge ids, accepted assertions included, no new journal rows.
            async with pg.connection() as conn:
                journal_before = (await (await conn.execute("SELECT count(*) AS n FROM substrate_graph_journal"))
                                  .fetchone())["n"]
            _seed_topic_foundry_circe(store_b)
            await referent_projector.ReferentProjector(
                pool=apg, materializer=SubstrateGraphMaterializer(store=store_b), readiness=ALWAYS_READY).rebuild(now=NOW)
            async with pg.connection() as conn:
                journal_after = (await (await conn.execute("SELECT count(*) AS n FROM substrate_graph_journal"))
                                 .fetchone())["n"]
            assert journal_after == journal_before
            _, nodes_b, edges_b = _ids(store_b)
            assert (nodes_b, edges_b) == (nodes, edges)
        finally:
            for client in (client_a, client_b):
                client.graph_query("MATCH (n) DETACH DELETE n")
    asyncio.run(_with_dbs(body))
