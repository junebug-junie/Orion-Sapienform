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
from orion.world_pulse_read import concept_search
from orion.world_pulse_read.concept_search import (
    OBJECT_MIN_CENTERED_SIMILARITY,
    concept_doc,
    rank_centered,
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
    assert related[0][0].via == "meaning" and related[0][1] >= OBJECT_MIN_CENTERED_SIMILARITY
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
    # Floor off: 30 near-duplicates shift the index mean onto themselves, so the cap (not
    # the floor) is what this test pins.
    related = _related(store, ctx, backend, min_similarity=-1.0)
    assert len(related) == MAX_OBJECTS - len(ctx.objects)  # 31+ qualify; only the room is taken
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


def test_a_label_too_short_to_name_anything_is_never_offered():
    # The quote check is a substring test: a concept labelled "AI" would be "named" by any
    # quote containing "said". The label path drops such labels; so does the meaning path.
    store, _, ctx, _, _ = _setup([existing_concept("concept-ai", "AI")])
    backend = _indexed(store, dict(VECTORS, AI=(1.0, 0.0, 0.0)))
    assert "concept-ai" in backend.docs
    assert "concept-ai" not in {e.node_id for e, _ in _related(store, ctx, backend)}


def test_a_low_ranked_concept_named_in_the_text_is_linked_by_the_same_quote_rule():
    """The only way a meaning object can be linked: its label IS in the text but it ranked
    below the label path's region scan. The quote rule treats it exactly like a label hit."""
    store, subject_id, ctx, _, h = _setup()
    low = existing_concept("concept-thermal-energy", "thermal energy")
    store.upsert_node(identity_key=None, node=low)
    backend = _indexed(store, dict(VECTORS, **{"thermal energy": (0.95, 0.2, 0.0)}))
    # Simulate the rank cut: drop it from the label objects the way a 500-node scan would.
    ctx = type(ctx)(subjects=ctx.subjects, objects=tuple(o for o in ctx.objects if o.node_id != low.node_id),
                    texts=ctx.texts, haystack=ctx.haystack)
    widened = with_related_objects(ctx, _related(store, ctx, backend))
    assert any(o.node_id == low.node_id and o.via == "meaning" for o in widened.objects)
    claim = WorldPulseReadRelationshipClaimV1(
        subject_id=subject_id, predicate="associated_with", object_id=low.node_id, statement_text="s",
        quote="moving thermal energy from a cold space to a warm one")
    plan = plan_claims([claim], widened, seed_id=h.seed_ref.seed_id, recorded_at=NOW)[0]
    assert plan.claim.receipt.outcome == "accepted_provisional"


def test_stage2_keeps_label_objects_before_the_index_exists():
    store, _, exact_ctx, conn, h = _setup()
    backend = FakeSemanticBackend(VECTORS)  # nothing indexed: the collection does not exist
    pipe = _pipeline(store, conn, backend)
    ctx = asyncio.run(pipe._build_claim_context(h))
    assert [o.node_id for o in ctx.objects] == [o.node_id for o in exact_ctx.objects]


def test_stage2_records_a_meaning_claim_as_proposed_and_refuses_one_no_longer_offered(monkeypatch):
    store, subject_id, _, conn, h = _setup()
    journal = FakeJournal()
    # The globals the class actually runs with (Hub's conftest re-imports scripts.* per test).
    monkeypatch.setitem(WorldPulseReadStage2Pipeline._record_claims.__globals__, "SubstrateGraphJournal",
                        lambda pool: journal)
    backend = FakeSemanticBackend(VECTORS)
    pipe = _pipeline(store, conn, backend)
    asyncio.run(pipe.index_concepts_once())
    pipe._claim_context = asyncio.run(pipe._build_claim_context(h))
    claim = WorldPulseReadRelationshipClaimV1(
        subject_id=subject_id, predicate="associated_with", object_id=RELATED_ID, statement_text="s", quote=QUOTE)
    result = WorldPulseReadStage2ResultV1(summary="s", trace_id="t", created_at=NOW, seed_id="finding:heat-pump",
                                          relationship_claims=[claim])
    recorded = asyncio.run(pipe._record_claims("finding:heat-pump", result))
    assert recorded.relationship_claims[0].receipt.outcome == "proposed"
    assert recorded.relationship_claims[0].receipt.reason == "quote_not_about_endpoints"
    assert len(journal.proposals()) == 1 and journal.decisions() == []
    # A Hub restart rebuilt the context and the meaning object no longer ranks: refused, no
    # proposal (documented on stored_object_lookup; it could never have been linked).
    pipe._claim_context = type(pipe._claim_context)(
        subjects=pipe._claim_context.subjects, texts=pipe._claim_context.texts,
        objects=tuple(o for o in pipe._claim_context.objects if o.via != "meaning"),
        haystack=pipe._claim_context.haystack)
    again = asyncio.run(pipe._record_claims("finding:heat-pump", result))
    assert again.relationship_claims[0].receipt.reason == "unknown_object"
    assert len(journal.proposals()) == 1


def _run_loop(pipe, passes):
    """Run _concept_index_loop for `passes` index passes, recording each requested delay."""
    delays = []

    async def fake_wait_for(aw, timeout):
        delays.append(timeout)
        aw.close()
        if len(delays) >= passes:
            pipe._stop.set()
        raise asyncio.TimeoutError

    async def go():
        import scripts.world_pulse_read_stage2 as mod
        orig = mod.asyncio.wait_for
        mod.asyncio.wait_for = fake_wait_for
        try:
            await pipe._concept_index_loop()
        finally:
            mod.asyncio.wait_for = orig
    asyncio.run(go())
    return delays


def test_index_loop_hurries_only_while_the_backlog_drains():
    extra = [existing_concept(f"concept-x-{i:02d}", f"Concept x {i:02d}") for i in range(5)]
    store, _, _, conn, _ = _setup(extra)
    backend = FakeSemanticBackend(VECTORS)
    pipe = _pipeline(store, conn, backend)
    pipe.concept_search = backend.config(index_batch=3)
    total = len(asyncio.run(asyncio.to_thread(__import__(
        "orion.world_pulse_read.concept_search", fromlist=["index_rows"]).index_rows, store)))
    delays = _run_loop(pipe, passes=6)
    assert len(backend.docs) == total
    assert delays[0] == 5.0 and delays[-1] == 300.0


def test_index_loop_backs_off_when_upserts_never_land():
    store, _, _, conn, _ = _setup()
    backend = FakeSemanticBackend(VECTORS)
    backend.drop_upserts = True
    pipe = _pipeline(store, conn, backend)
    pipe.concept_search = backend.config(index_batch=2)
    delays = _run_loop(pipe, passes=4)
    # First pass hurries (no history); the backlog never shrinks, so every later pass waits.
    assert delays[0] == 5.0 and delays[1:] == [300.0, 300.0, 300.0]
    embeds_per_pass = 2
    assert len(backend.embedded) == 4 * embeds_per_pass


def test_index_task_starts_only_with_search_and_assertions_and_stop_cancels_it():
    store, _, _, conn, _ = _setup()
    backend = FakeSemanticBackend(VECTORS)

    async def go(assertions):
        pipe = WorldPulseReadStage2Pipeline(
            enabled=True, tick_interval_sec=3600, timeout_sec=60, session_id="s",
            pool_provider=lambda: None, source_ref=ServiceRef(name="hub", version="0", node="t"),
            store_provider=lambda: store, projector_readiness=ALWAYS_READY,
            concept_search=backend.config(), assertions_enabled=assertions)
        pipe._http_client = backend.client
        pipe.tick = lambda **kw: asyncio.sleep(0)
        await pipe.start(backend)
        started = pipe._concept_index_task is not None
        await pipe.stop()
        return started, pipe._concept_index_task

    assert asyncio.run(go(True)) == (True, None)
    assert asyncio.run(go(False)) == (False, None)


# ---- centered ranking (2026-10-14): the raw floor had no calm state ----

def _cone(*topic: float, cone: float = 3.0) -> list[float]:
    """A unit vector with a large shared component, the way every bge vector shares one."""
    import math

    v = list(topic) + [cone]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_centering_gives_unrelated_concepts_a_calm_state_the_raw_floor_lacked():
    # Six topics on separate axes, all inside one narrow cone: raw cosine between ANY two
    # is ~0.9, so the retired raw floor (0.65) offered everything.
    index = [(f"c{i}", _cone(*[1.0 if k == i else 0.0 for k in range(6)])) for i in range(6)]
    near = _cone(*[0.9 if k == 0 else (0.3 if k == 1 else 0.0) for k in range(6)])
    raw = {i: sum(a * b for a, b in zip(v, near)) for i, v in index}
    assert min(raw.values()) > 0.65  # what the raw floor saw: every concept "related"
    ranked = dict(rank_centered(index, [near], -1.0))
    assert ranked["c0"] > OBJECT_MIN_CENTERED_SIMILARITY
    assert all(ranked[f"c{i}"] < 0.1 for i in range(2, 6))  # unrelated rests near/below 0
    kept = [i for i, _ in rank_centered(index, [near], OBJECT_MIN_CENTERED_SIMILARITY)]
    assert kept[0] == "c0" and not set(kept) & {"c2", "c3", "c4", "c5"}


def test_rank_centered_takes_the_best_subject_and_refuses_mismatched_dimensions():
    import pytest

    index = [(f"c{i}", _cone(*[1.0 if k == i else 0.0 for k in range(4)])) for i in range(4)]
    a, b = _cone(1.0, 0, 0, 0), _cone(0, 0, 0, 1.0)
    both = dict(rank_centered(index, [a, b], -1.0))
    assert both["c0"] == pytest.approx(dict(rank_centered(index, [a], -1.0))["c0"])
    assert both["c3"] == pytest.approx(dict(rank_centered(index, [b], -1.0))["c3"])
    assert rank_centered([], [a], 0.0) == [] and rank_centered(index, [], 0.0) == []
    with pytest.raises(ValueError):
        rank_centered(index, [[1.0, 0.0]], 0.0)


def test_all_vectors_pages_through_the_whole_index():
    from orion.introspect.semantic_index import SearchUnavailableError, all_vectors

    backend = FakeSemanticBackend({"x": (1.0, 0.0)})
    for i in range(7):
        backend.docs[f"d{i}"] = ([float(i), 1.0], {}, "t")

    async def go(page):
        async with backend.client() as client:
            return await all_vectors(client, backend.config(), page=page)
    got = asyncio.run(go(3))
    assert [i for i, _ in got] == [f"d{i}" for i in range(7)] and got[4][1] == [4.0, 1.0]
    backend.docs.clear()  # a collection nothing was ever upserted into does not exist
    import pytest
    with pytest.raises(SearchUnavailableError, match="not built"):
        asyncio.run(go(3))
    backend.exists_empty = True  # exists, holds nothing: unknown, not "no match"
    with pytest.raises(SearchUnavailableError, match="is empty"):
        asyncio.run(go(3))


def test_a_failing_whole_index_read_is_search_unavailable_so_label_objects_stand():
    import pytest

    from orion.introspect.semantic_index import SearchUnavailableError

    store, _, ctx, _, _ = _setup()
    backend = _indexed(store)
    backend.fail_get_all = True
    with pytest.raises(SearchUnavailableError):
        _related(store, ctx, backend)


def test_the_raw_floor_is_retired_not_left_beside_the_centered_one():
    assert not hasattr(concept_search, "OBJECT_MIN_SIMILARITY")
    assert not hasattr(concept_search, "NEAREST_PER_SUBJECT")
