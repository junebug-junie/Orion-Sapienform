"""In-process fakes for the reading-assertion path (tests + evals).

``FakeJournal`` mirrors ``orion.substrate.graph_journal.SubstrateGraphJournal``'s contract:
append is idempotent on event id, a second decision for the same (target, revision)
raises ``RevisionConflict``, ``pending_decisions`` lists unapplied decisions (optionally
only those whose proposal came from ``proposal_actors``). The real SQL is covered by
orion/substrate/tests/test_graph_journal_pg.py against a throwaway Postgres.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphMaterializationV1,
    SubstrateGraphProposalV1,
)
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import (
    WorldPulseReadConceptCandidateV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadSeedV1,
)
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from orion.substrate.graph_journal import TERMINAL_FAILURE_REASONS, RevisionConflict
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.store import InMemorySubstrateGraphStore

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
SOURCE_URL = "https://example.org/heat-pumps"
FETCH_TEXT = (
    "Heat pumps explained. A heat pump transfers heat using a refrigeration cycle, "
    "moving thermal energy from a cold space to a warm one. Unlike a furnace, it does "
    "not burn fuel."
)
OBJECT_ID = "concept-refrigeration-cycle"
UNRELATED_ID = "concept-furnace"


class FakeJournal:
    def __init__(self) -> None:
        self.events: dict[str, Any] = {}
        self.order: list[str] = []
        self.decision_slots: set[tuple[str, int]] = set()

    async def append(self, event) -> bool:
        key = f"{event.event_kind}:{event.event_id}"
        if key in self.events:
            return False
        if isinstance(event, SubstrateGraphDecisionV1):
            slot = (event.target_id, event.resulting_revision)
            if slot in self.decision_slots:
                raise RevisionConflict(f"target {event.target_id} revision {event.resulting_revision}")
            self.decision_slots.add(slot)
        self.events[key] = event
        self.order.append(key)
        return True

    def _of(self, cls) -> list:
        return [self.events[k] for k in self.order if isinstance(self.events[k], cls)]

    def proposals(self) -> list[SubstrateGraphProposalV1]:
        return self._of(SubstrateGraphProposalV1)

    def decisions(self) -> list[SubstrateGraphDecisionV1]:
        return self._of(SubstrateGraphDecisionV1)

    def materializations(self) -> list[SubstrateGraphMaterializationV1]:
        return self._of(SubstrateGraphMaterializationV1)

    async def pending_decisions(self, *, limit: int = 100, proposal_actors=None) -> list:
        applied = {m.decision_id for m in self.materializations() if m.outcome == "applied"}
        out = []
        for d in self.decisions():
            if d.decision_id in applied:
                continue
            last = await self.last_materialization(d.decision_id)
            if last is not None and last.failure_reason and last.failure_reason.split(":")[0] in TERMINAL_FAILURE_REASONS:
                continue
            if d.expected_prior_revision > await self.latest_applied_revision(d.target_id):
                continue
            if proposal_actors is not None:
                proposal = await self.proposal(d.proposal_id)
                if proposal is None or proposal.actor not in proposal_actors:
                    continue
            out.append(d)
        return out[:limit]

    async def proposal(self, proposal_id: str):
        return next((p for p in self.proposals() if p.proposal_id == proposal_id), None)

    async def latest_applied_revision(self, target_id: str) -> int:
        return max([m.revision for m in self.materializations()
                    if m.outcome == "applied" and m.target_id == target_id] or [0])

    async def failed_attempts(self, decision_id: str) -> int:
        return sum(1 for m in self.materializations() if m.outcome == "failed" and m.decision_id == decision_id)

    async def last_materialization(self, decision_id: str):
        rows = [m for m in self.materializations() if m.decision_id == decision_id]
        return rows[-1] if rows else None


class FakeSnapshotConn:
    """``reading_document_snapshot`` as a dict: the two statements fetch_text.py issues."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, str]] = {}
        self.fail = False

    async def execute(self, sql: str, sha: str, text: str, chars: int, source: str) -> str:
        if self.fail:
            raise ConnectionError("down")
        assert "reading_document_snapshot" in sql and chars == len(text)
        self.rows.setdefault(sha, (text, source))
        return "INSERT 0 1"

    async def fetch(self, sql: str, shas: list[str]) -> list[dict]:
        return [{"sha256": s, "content": self.rows[s][0]} for s in shas if s in self.rows]


class FakePool:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def existing_concept(node_id: str, label: str) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id, anchor_scope="orion", subject_ref="world_pulse", label=label,
        temporal=SubstrateTemporalWindowV1(observed_at=NOW),
        provenance=SubstrateProvenanceV1(authority="local_inferred", source_kind="topic_foundry",
                                         source_channel="test", producer="topic_foundry_adapter"),
        signals=SubstrateSignalBundleV1(confidence=0.6, salience=0.6),
    )


def handoff(*, sha: str | None, labels=("Heat pump",)) -> WorldPulseReadHandoffV1:
    evidence = SourceFetchEvidenceV1(url=SOURCE_URL, tool_name="WebFetch", content_chars=len(FETCH_TEXT),
                                     content_sha256=sha)
    return WorldPulseReadHandoffV1(
        seed_ref=WorldPulseReadSeedV1(seed_id="finding:heat-pump", kind="finding", run_id="run-1",
                                      url=SOURCE_URL, title="Heat pumps explained"),
        what_i_learned="Heat pumps move heat with a refrigeration cycle rather than burning fuel.",
        concept_candidates=[WorldPulseReadConceptCandidateV1(label=label) for label in labels],
        trace_id="trace-heat-pump", created_at=NOW, read_evidence=[evidence],
    )


def seeded_store(h: WorldPulseReadHandoffV1) -> tuple[InMemorySubstrateGraphStore, str]:
    """The atlas before Stage 2: two existing concepts, plus this read's Stage 1 concept
    written by the real Stage 1 path (adapter + materializer). Returns the stored subject id."""
    store = InMemorySubstrateGraphStore()
    materializer = SubstrateGraphMaterializer(store=store)
    from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1

    materializer.apply_record(SubstrateGraphRecordV1(anchor_scope="orion", subject_ref="world_pulse", nodes=[
        existing_concept(OBJECT_ID, "Refrigeration cycle"), existing_concept(UNRELATED_ID, "Furnace")]))
    result = materializer.apply_record(map_world_pulse_read_handoff_to_substrate(h))
    return store, result.node_decisions[0].canonical_node_id


class FakeSemanticBackend:
    """vector-host ``/embedding`` + Chroma's v1 REST + orion-vector-writer, in process.

    The real ``orion.introspect.semantic_index`` code talks to it over an
    ``httpx.MockTransport``; ``publish`` stands in for the bus hop to vector-writer.
    ``vectors`` maps an embedded text to its vector; any other text gets an axis no
    fixture vector shares (similarity 0)."""

    EMBED_URL = "http://vector-host.test/embedding"
    CHROMA_URL = "http://chroma.test"
    COLLECTION_ID = "coll-1"

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        import math

        dims = max(len(v) for v in vectors.values()) + 1
        self._dims = dims

        def norm(v):
            v = list(v) + [0.0] * (dims - len(v))
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / n for x in v]

        self._norm = norm
        self.vectors = {k: norm(v) for k, v in vectors.items()}
        self.docs: dict[str, tuple[list[float], dict, str]] = {}
        self.embedded: list[str] = []
        self.fail_embed = False
        # vector-writer down: upserts are published but never land.
        self.drop_upserts = False
        # Chroma answers the whole-index paged read (all_vectors) with an error.
        self.fail_get_all = False
        # The collection exists but holds no rows (created, then emptied).
        self.exists_empty = False

    def vector(self, text: str) -> list[float]:
        return self.vectors.get(text) or self._norm([0.0] * (self._dims - 1) + [1.0])

    def _handler(self, request):
        import json

        import httpx

        url = str(request.url)
        if url == self.EMBED_URL:
            if self.fail_embed:
                return httpx.Response(503, json={"error": "down"})
            body = json.loads(request.content)
            self.embedded.append(body["text"])
            vec = self.vector(body["text"])
            return httpx.Response(200, json={"doc_id": body["doc_id"], "embedding": vec,
                                             "embedding_model": "fake-bge", "embedding_dim": len(vec)})
        base = f"{self.CHROMA_URL}/api/v1/collections"
        if request.method == "GET" and url.startswith(base + "/") and not url.endswith("/count"):
            if not self.docs and not self.exists_empty:
                # Chroma 0.4.24: a collection nothing was ever upserted into does not exist.
                name = url.rsplit("/", 1)[-1]
                return httpx.Response(500, json={"error": f"ValueError('Collection {name} does not exist.')"})
            return httpx.Response(200, json={"id": self.COLLECTION_ID, "metadata": {"hnsw:space": "cosine"}})
        if url == f"{base}/{self.COLLECTION_ID}/count":
            return httpx.Response(200, json=len(self.docs))
        if url == f"{base}/{self.COLLECTION_ID}/get":
            body = json.loads(request.content)
            if "ids" in body:
                ids = [i for i in body["ids"] if i in self.docs]
            elif self.fail_get_all:
                return httpx.Response(500, json={"error": "boom"})
            else:  # paged read of the whole collection (semantic_index.all_vectors)
                every = sorted(self.docs)
                ids = every[body.get("offset", 0): body.get("offset", 0) + body.get("limit", len(every))]
            include = body.get("include") or []
            return httpx.Response(200, json={
                "ids": ids,
                "metadatas": [self.docs[i][1] for i in ids] if "metadatas" in include else None,
                "embeddings": [self.docs[i][0] for i in ids] if "embeddings" in include else None})
        if url == f"{base}/{self.COLLECTION_ID}/query":
            body = json.loads(request.content)
            q = body["query_embeddings"][0]
            scored = sorted(((1.0 - sum(a * b for a, b in zip(q, v)), i) for i, (v, _, _) in self.docs.items()))
            scored = scored[: body["n_results"]]
            return httpx.Response(200, json={"ids": [[i for _, i in scored]], "distances": [[d for d, _ in scored]]})
        return httpx.Response(404, json={"error": f"unexpected {request.method} {url}"})

    def client(self):
        import httpx

        return httpx.AsyncClient(transport=httpx.MockTransport(self._handler))

    def config(self, *, min_similarity: float | None = None, index_batch: int = 200):
        from orion.introspect.semantic_index import SearchConfig
        from orion.world_pulse_read.concept_search import ATLAS_CONCEPT_COLLECTION, OBJECT_MIN_CENTERED_SIMILARITY

        if min_similarity is None:
            min_similarity = OBJECT_MIN_CENTERED_SIMILARITY
        return SearchConfig(chroma_url=self.CHROMA_URL, embed_url=self.EMBED_URL,
                            collection=ATLAS_CONCEPT_COLLECTION, min_similarity=min_similarity,
                            index_interval_sec=300.0, index_batch=index_batch)

    async def publish(self, channel: str, envelope: Any) -> None:
        """What orion-vector-writer does with an upsert: store the vector under its doc id."""
        assert channel == "orion:vector:semantic:upsert"
        if self.drop_upserts:
            return
        p = envelope.payload
        self.docs[p["doc_id"]] = (p["embedding"], p["meta"], p["text"])
