"""Reading claim objects by meaning (orion/world_pulse_read/concept_search.py).

Exact label-in-text offered no object on most live reads (2026-10-13: four reads in a row
logged objects=0). These tests pin the meaning path: an atlas concept whose label is NOT in
the read but is near a subject is offered; fenced memory nodes and the read's own concepts
never are; the cap holds with label hits first; and the quote rule is unchanged, so a claim
to a meaning-only object is journalled ``proposed`` with no link.
"""

from __future__ import annotations

import asyncio

from orion.core.bus.bus_schemas import ServiceRef
from orion.core.schemas.cognitive_substrate import ConceptNodeV1
from orion.schemas.world_pulse_read import WorldPulseReadRelationshipClaimV1, WorldPulseReadStage2ResultV1
from orion.substrate.assertion_projector import AssertionProjector
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reader_capability import ALWAYS_READY
from orion.world_pulse_read.assertions import (
    MAX_OBJECTS,
    READING_ACTOR,
    build_claim_context,
    claim_prompt_section,
    journal_claims,
    plan_claims,
)
from orion.world_pulse_read.concept_search import (
    concept_doc,
    index_atlas_concepts,
    related_objects,
    with_related_objects,
)
from orion.world_pulse_read.fetch_text import load_retained_texts, text_sha256
from reading_assertion_fakes import (
    FETCH_TEXT,
    NOW,
    OBJECT_ID,
    SOURCE_URL,
    UNRELATED_ID,
    FakeJournal,
    FakePool,
    FakeSemanticBackend,
    FakeSnapshotConn,
    existing_concept,
    handoff,
    seeded_store,
)
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline, _build_stage2_prompt

RELATED_ID = "concept-thermal-energy-transfer"
RELATED_LABEL = "Thermal energy transfer"  # not in FETCH_TEXT ("thermal energy" is, "transfer" is not)
FENCED_ID = "concept-private-heat-pump-memory"
FAR_ID = "concept-medieval-poetry"
QUOTE = "A heat pump transfers heat using a refrigeration cycle"

VECTORS = {
    "Heat pump": (1.0, 0.0, 0.0),
    RELATED_LABEL: (0.9, 0.3, 0.0),
    "Refrigeration cycle": (0.9, 0.4, 0.0),
    "Furnace": (0.5, 1.0, 0.0),
    "Heat pump memory": (1.0, 0.0, 0.0),
    "Medieval poetry": (0.0, 0.0, 1.0),
}


def _fenced(node_id: str, label: str) -> ConceptNodeV1:
    node = existing_concept(node_id, label)
    return node.model_copy(update={"provenance": node.provenance.model_copy(update={"producer": "memory.referents"})})


def _setup(extra=()):
    conn = FakeSnapshotConn()
    sha = text_sha256(FETCH_TEXT)
    conn.rows[sha] = (FETCH_TEXT, SOURCE_URL)
    h = handoff(sha=sha)
    store, subject_id = seeded_store(h)
    for node in (existing_concept(RELATED_ID, RELATED_LABEL), existing_concept(FAR_ID, "Medieval poetry"),
                 _fenced(FENCED_ID, "Heat pump memory"), *extra):
        store.upsert_node(identity_key=None, node=node)
    texts = asyncio.run(load_retained_texts(conn, h.read_evidence))
    return store, subject_id, build_claim_context(store, h, texts), conn, h


def _indexed(store, vectors=VECTORS):
    backend = FakeSemanticBackend(vectors)
    async def go():
        async with backend.client() as client:
            return await index_atlas_concepts(store, backend.config(), client=client, bus=backend,
                                              source=ServiceRef(name="hub", version="0", node="t"))
    asyncio.run(go())
    return backend


def _related(store, ctx, backend, **cfg):
    async def go():
        async with backend.client() as client:
            return await related_objects(store, ctx, backend.config(**cfg), client=client)
    return asyncio.run(go())


def test_a_concept_near_the_subject_whose_label_is_not_in_the_read_is_offered():
    store, subject_id, ctx, _, _ = _setup()
    assert RELATED_LABEL.lower() not in ctx.haystack.lower()
    assert RELATED_ID not in {o.node_id for o in ctx.objects}  # exact matching misses it
    backend = _indexed(store)
    related = _related(store, ctx, backend)
    ids = [e.node_id for e, _ in related]
    assert ids == [RELATED_ID]
    assert related[0][0].via == "meaning" and related[0][1] > 0.9
    widened = with_related_objects(ctx, related)
    assert widened.usable
    # Label hits first, unchanged; the meaning hit after them.
    assert [o.node_id for o in widened.objects][: len(ctx.objects)] == [o.node_id for o in ctx.objects]
    section = claim_prompt_section(widened)
    assert "objects related by meaning" in section
    assert section.index(RELATED_ID) > section.index("objects related by meaning")
    assert section.index(OBJECT_ID) < section.index("objects related by meaning")


def test_fenced_own_far_and_already_offered_concepts_are_never_meaning_objects():
    store, subject_id, ctx, _, _ = _setup()
    backend = _indexed(store)
    assert FENCED_ID not in backend.docs  # never indexed
    # A stale index entry for a fenced node (written before it was fenced) is re-gated away.
    backend.docs[FENCED_ID] = (backend.vector("Heat pump memory"), {"node_id": FENCED_ID}, "Heat pump memory")
    backend.docs["concept-deleted"] = (backend.vector("Heat pump"), {"node_id": "concept-deleted"}, "gone")
    ids = {e.node_id for e, _ in _related(store, ctx, backend)}
    assert subject_id in backend.docs  # the read's own concept is indexed (it is an atlas concept)...
    assert subject_id not in ids  # ...but never its own object
    assert FENCED_ID not in ids and "concept-deleted" not in ids
    assert FAR_ID not in ids  # below the floor
    assert OBJECT_ID not in ids and UNRELATED_ID not in ids  # already offered by label


def test_the_cap_holds_with_label_hits_first():
    extra = [existing_concept(f"concept-heat-topic-{i:02d}", f"Heat topic {i:02d}") for i in range(30)]
    vectors = dict(VECTORS, **{f"Heat topic {i:02d}": (1.0, 0.01 * i, 0.0) for i in range(30)})
    store, _, ctx, _, _ = _setup(extra)
    backend = _indexed(store, vectors)
    related = _related(store, ctx, backend)
    assert len(related) == MAX_OBJECTS - len(ctx.objects)  # 31 qualify; only the room is taken
    widened = with_related_objects(ctx, related)
    assert len(widened.objects) == MAX_OBJECTS
    assert [o.node_id for o in widened.objects[: len(ctx.objects)]] == [o.node_id for o in ctx.objects]
    sims = [s for _, s in related]
    assert sims == sorted(sims, reverse=True)


def test_the_floor_is_the_configured_similarity():
    store, _, ctx, _, _ = _setup()
    backend = _indexed(store)
    assert _related(store, ctx, backend, min_similarity=0.99) == []


def test_a_definition_is_part_of_what_a_concept_is_embedded_as():
    node = existing_concept("c", "Heat pump").model_copy(update={"definition": "moves heat"})
    assert concept_doc(node) == "Heat pump: moves heat"
    assert concept_doc(existing_concept("c", "Heat pump")) == "Heat pump"


def test_the_strict_quote_rule_keeps_a_meaning_only_claim_proposed_with_no_link():
    store, subject_id, ctx, _, h = _setup()
    widened = with_related_objects(ctx, _related(store, ctx, _indexed(store)))
    claim = WorldPulseReadRelationshipClaimV1(
        subject_id=subject_id, predicate="associated_with", object_id=RELATED_ID,
        statement_text="A heat pump is a thermal energy transfer device.", quote=QUOTE)
    journal = FakeJournal()
    plans = plan_claims([claim], widened, seed_id=h.seed_ref.seed_id, recorded_at=NOW)
    report = asyncio.run(journal_claims(journal, plans))
    receipt = report.claims[0].receipt
    assert receipt.outcome == "proposed" and receipt.reason == "quote_not_about_endpoints"
    assert len(journal.proposals()) == 1 and journal.decisions() == []
    asyncio.run(AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                   readiness=ALWAYS_READY, proposal_actors=(READING_ACTOR,)).run_once())
    assert store.snapshot().edges == {}


# ── Stage 2 wiring ─────────────────────────────────────────────────────────


def _pipeline(store, conn, backend):
    pipe = WorldPulseReadStage2Pipeline(
        enabled=True, tick_interval_sec=60, timeout_sec=60, session_id="s",
        pool_provider=lambda: FakePool(conn), source_ref=ServiceRef(name="hub", version="0", node="t"),
        store_provider=lambda: store, projector_readiness=ALWAYS_READY, concept_search=backend.config(),
    )
    pipe._http_client = backend.client
    pipe._bus = backend
    return pipe


def test_stage2_indexes_the_atlas_then_offers_meaning_objects_in_the_prompt():
    store, _, _, conn, h = _setup()
    backend = FakeSemanticBackend(VECTORS)
    pipe = _pipeline(store, conn, backend)
    first = asyncio.run(pipe.index_concepts_once())
    assert first.indexed == len(backend.docs) and first.pending == 0
    assert asyncio.run(pipe.index_concepts_once()).indexed == 0  # unchanged concepts are not re-embedded
    ctx = asyncio.run(pipe._build_claim_context(h))
    assert RELATED_ID in {o.node_id for o in ctx.objects if o.via == "meaning"}
    assert "objects related by meaning" in _build_stage2_prompt(h, "t", ctx)


def test_stage2_keeps_label_objects_when_the_embedder_is_down():
    store, _, exact_ctx, conn, h = _setup()
    backend = FakeSemanticBackend(VECTORS)
    pipe = _pipeline(store, conn, backend)
    asyncio.run(pipe.index_concepts_once())
    backend.fail_embed = True
    ctx = asyncio.run(pipe._build_claim_context(h))
    assert [o.node_id for o in ctx.objects] == [o.node_id for o in exact_ctx.objects]
    assert ctx.usable


def test_no_concept_search_configured_means_label_objects_only():
    store, _, exact_ctx, conn, h = _setup()
    pipe = WorldPulseReadStage2Pipeline(
        enabled=True, tick_interval_sec=60, timeout_sec=60, session_id="s",
        pool_provider=lambda: FakePool(conn), source_ref=ServiceRef(name="hub", version="0", node="t"),
        store_provider=lambda: store, projector_readiness=ALWAYS_READY,
    )
    assert pipe.concept_search is None
    ctx = asyncio.run(pipe._build_claim_context(h))
    assert [o.node_id for o in ctx.objects] == [o.node_id for o in exact_ctx.objects]
