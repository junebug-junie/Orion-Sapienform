"""The referent projector (app/referent_projector.py) on a disposable Postgres + FalkorDB.

Shares the fixture of orion/memory/referents/tests/test_referents_pg.py. CI:
.github/workflows/orion-memory-episode-tests.yml (Postgres + FalkorDB services, fails on skip).
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

from orion.substrate.reader_capability import ALWAYS_READY, ReadinessV1  # noqa: E402
from orion.memory.referents.tests.test_referents_pg import (  # noqa: E402
    CIRCE, HECATE, MORGAN, NOW, QUILL, RETREAT, _falkor, _persist, _seed_topic_foundry_circe, _table, _with_dbs,
    pytestmark,  # noqa: F401  (same skip rule)
)

spec = importlib.util.spec_from_file_location("mc_referent_projector", SERVICE_ROOT / "app" / "referent_projector.py")
referent_projector = importlib.util.module_from_spec(spec)
sys.modules["mc_referent_projector"] = referent_projector
spec.loader.exec_module(referent_projector)


def test_projector_builds_the_graph_fences_circe_and_walks_only_accepted_claims():
    from orion.substrate.materializer import SubstrateGraphMaterializer
    from orion.substrate.neighborhood import NeighborhoodRequestV1

    async def body(pg, apg):
        _, client, store = _falkor()
        try:
            _seed_topic_foundry_circe(store)
            await _persist(pg)
            projector = referent_projector.ReferentProjector(pool=apg, materializer=SubstrateGraphMaterializer(store=store),
                                                             readiness=ALWAYS_READY)
            tick = await projector.run_once(now=NOW)
            assert tick.label_questions == [CIRCE] and tick.memories == 2 and tick.assertions_applied == 4

            hecate = store.get_node_by_id(HECATE)
            assert (hecate.entity_type, hecate.promotion_state) == ("project", "provisional")
            assert {"Inspur NF5288M5", "AGX-2 GPU", "8x smx2 gpus"} <= set(hecate.aliases)
            assert "the new server" not in hecate.aliases
            # topic-foundry's "circe" is untouched and separate; ours stays walkable and Orion asks.
            assert store.get_node_by_id("tf-circe").provenance.producer == "topic_foundry_adapter"
            assert store.get_node_by_id(CIRCE).promotion_state == "provisional"
            questions = await _table(pg, "SELECT text, scope, answer_via FROM memory_tension_shadow "
                                         "WHERE source_refs->0->>'reason' = 'label_collision'")
            assert questions == [{"text": "Is the 'circe' from our conversations the same as the 'circe' already "
                                          "in my graph?", "scope": "self", "answer_via": "investigation"}]

            walk = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=(MORGAN,)))
            assert not walk.degraded, walk.reason
            assert {n.node_id for n in walk.neighbor_nodes} == {QUILL, RETREAT}
            assert all(e.edge_role == "semantic_projection" for e in walk.boundary_edges)
            # Hecate-Circe is accepted and Circe stays walkable while Orion asks about the name.
            hecate_walk = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=(HECATE,)))
            assert {n.node_id for n in hecate_walk.neighbor_nodes} == {CIRCE}

            # A full hydration of the result decodes cleanly (new readers accept every shape), every
            # memory is an Evidence node with a real valid_from, linked by provenance edges only.
            state = store.snapshot()
            evidence = [n for n in state.nodes.values() if n.node_kind == "evidence"]
            assert len(evidence) == 2 and all(n.temporal.valid_from == NOW for n in evidence)
            provenance = [e for e in state.edges.values() if e.predicate == "observed_in"]
            assert len(provenance) == 6 and {e.edge_role for e in provenance} == {"provenance"}

            again = await projector.run_once(now=NOW)
            assert (again.nodes, again.memories, again.assertions_applied, again.label_questions) == (0, 0, 0, [])

            # After a restart the projector primes only its own rows, not the whole graph:
            # 6 referents + 2 evidence + 4 assertions; 6 provenance + 12 structure + 4 projections.
            from orion.substrate.falkor_store import FalkorSubstrateStore, FalkorSubstrateStoreConfig

            fresh = FalkorSubstrateStore(FalkorSubstrateStoreConfig(uri="redis://unused", graph_name="unused"),
                                         client=client, hydrate=False)
            assert fresh.prime_cache_for_producers(("memory.referents",)) == (12, 22)
            assert fresh.get_node_by_id("tf-circe") is None
            assert fresh.get_node_by_id(HECATE).aliases == store.get_node_by_id(HECATE).aliases

            async with pg.connection() as conn:
                await conn.execute("UPDATE episode_memory SET status = 'superseded', updated_at = %s "
                                   "WHERE statement LIKE 'Juniper''s boss%%'", (NOW,))
            closed = await projector.run_once(now=NOW)
            assert closed.memories == 1
            ended = [e for e in store.snapshot().edges.values()
                     if e.edge_role == "provenance" and e.source.node_id == MORGAN]
            assert ended and all(e.temporal.valid_to is not None for e in ended)
        finally:
            client.graph_query("MATCH (n) DETACH DELETE n")
    asyncio.run(_with_dbs(body))


def test_projector_writes_nothing_until_every_reader_is_ready():
    """#2520 review 3 / #2515 review 2: one mechanical gate, shared with AssertionProjector."""
    from orion.substrate.materializer import SubstrateGraphMaterializer

    async def body(pg, apg):
        _, client, store = _falkor()
        try:
            await _persist(pg)
            state = {"r": ReadinessV1(ready=False, missing=("orion-hub", "orion-recall"), reason="readers_not_ready")}
            projector = referent_projector.ReferentProjector(
                pool=apg, materializer=SubstrateGraphMaterializer(store=store), readiness=lambda: state["r"])
            tick = await projector.run_once(now=NOW)
            assert tick.blocked.missing == ("orion-hub", "orion-recall") and tick.nodes == 0
            assert store.snapshot().nodes == {}
            assert await _table(pg, "SELECT 1 FROM referent_projection") == []
            state["r"] = ReadinessV1(ready=True)
            assert (await projector.run_once(now=NOW)).nodes == 6
        finally:
            client.graph_query("MATCH (n) DETACH DELETE n")
    asyncio.run(_with_dbs(body))


def test_health_reports_the_missing_readers():
    blocked = ReadinessV1(ready=False, missing=("orion-hub",), reason="readers_not_ready")
    referent_projector.record_status(referent_projector.ProjectionTickV1(blocked=blocked), blocked)
    status = dict(referent_projector.PROJECTOR_STATUS)
    assert status["state"] == "waiting"
    assert status["readiness"] == {"ready": False, "missing": ["orion-hub"], "present": [], "reason": "readers_not_ready"}
