"""Reading relationship claims: retained text -> proposal -> rule -> projector -> walkable edge.

#2497 sections 3-4 on the #2515 assertion core. Fixture from the design: retained text
"a heat pump transfers heat using a refrigeration cycle" links the read's new heat-pump
concept to an EXISTING refrigeration-cycle concept, and to nothing else.
"""

from __future__ import annotations

import asyncio

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.world_pulse_read import (
    WorldPulseReadRelationshipClaimV1,
    WorldPulseReadStage2ResultV1,
)
from orion.substrate.assertion_projector import AssertionProjector
from orion.substrate.dynamics import SubstrateDynamicsEngine
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.neighborhood import NeighborhoodRequestV1
from orion.substrate.reader_capability import ALWAYS_READY
from orion.world_pulse_read.assertions import (
    READING_ACTOR,
    ClaimContextV1,
    EndpointV1,
    build_claim_context,
    claim_prompt_section,
    journal_claims,
    plan_claims,
)
from orion.world_pulse_read.fetch_text import load_retained_texts, retain_fetch_texts, text_sha256
from reading_assertion_fakes import (
    FETCH_TEXT,
    NOW,
    OBJECT_ID,
    SOURCE_URL,
    UNRELATED_ID,
    FakeJournal,
    FakePool,
    FakeSnapshotConn,
    handoff,
    seeded_store,
)
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline, _as_stage2_result, _build_stage2_prompt

QUOTE = "A heat pump transfers heat using a refrigeration cycle"
ALL_STATES = ("proposed", "provisional", "canonical")


def _claim(**kw) -> WorldPulseReadRelationshipClaimV1:
    base = dict(subject_id="", predicate="associated_with", object_id=OBJECT_ID,
                statement_text="A heat pump works by running a refrigeration cycle.", quote=QUOTE)
    base.update(kw)
    return WorldPulseReadRelationshipClaimV1(**base)


def _setup():
    conn = FakeSnapshotConn()
    sha = text_sha256(FETCH_TEXT)
    conn.rows[sha] = (FETCH_TEXT, SOURCE_URL)
    h = handoff(sha=sha)
    store, subject_id = seeded_store(h)
    texts = asyncio.run(load_retained_texts(conn, h.read_evidence))
    ctx = build_claim_context(store, h, texts)
    return store, subject_id, ctx, conn, h


def _projector(store, journal, actors=(READING_ACTOR,)):
    return AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                              readiness=ALWAYS_READY, proposal_actors=actors)


def _walk(store, focal):
    result = store.read_neighborhood(NeighborhoodRequestV1(focal_node_ids=(focal,), semantic_states=ALL_STATES))
    assert not result.degraded, result.reason
    return result


def _run(store, ctx, claims, journal=None):
    journal = journal or FakeJournal()
    plans = plan_claims(claims, ctx, seed_id="finding:heat-pump", recorded_at=NOW)
    report = asyncio.run(journal_claims(journal, plans))
    projected = asyncio.run(_projector(store, journal).run_once())
    return journal, report, projected


# ── retained text ──────────────────────────────────────────────────────────


def test_fetch_text_is_retained_by_hash_and_dropped_from_the_evidence():
    conn = FakeSnapshotConn()
    evidence = [SourceFetchEvidenceV1(url=SOURCE_URL, tool_name="WebFetch", content_chars=len(FETCH_TEXT),
                                      content_text=FETCH_TEXT)]
    kept = asyncio.run(retain_fetch_texts(conn, evidence))
    sha = text_sha256(FETCH_TEXT)
    assert kept[0].content_sha256 == sha and kept[0].content_text is None
    assert "content_text" not in kept[0].model_dump(mode="json")
    assert conn.rows[sha] == (FETCH_TEXT, SOURCE_URL)
    (loaded,) = asyncio.run(load_retained_texts(conn, kept))
    assert (loaded.text, loaded.representation) == (FETCH_TEXT, "tool_digest")


def test_a_store_failure_keeps_the_read_but_no_hash():
    conn = FakeSnapshotConn()
    conn.fail = True
    evidence = [SourceFetchEvidenceV1(url=SOURCE_URL, tool_name="WebFetch", content_chars=9, content_text="some text")]
    (kept,) = asyncio.run(retain_fetch_texts(conn, evidence))
    assert kept.content_sha256 is None and kept.content_text is None


def test_a_tampered_retained_digest_is_not_evidence():
    conn = FakeSnapshotConn()
    sha = text_sha256(FETCH_TEXT)
    conn.rows[sha] = (FETCH_TEXT + " (edited)", SOURCE_URL)
    assert asyncio.run(load_retained_texts(conn, handoff(sha=sha).read_evidence)) == []


def test_a_read_without_retained_text_gets_no_claim_context():
    h = handoff(sha=None)
    store, _ = seeded_store(h)
    ctx = build_claim_context(store, h, [])
    assert not ctx.usable and claim_prompt_section(ctx) == ""


# ── context ────────────────────────────────────────────────────────────────


def test_context_uses_the_stored_subject_id_and_existing_objects_named_in_the_read():
    store, subject_id, ctx, _, _ = _setup()
    assert [s.node_id for s in ctx.subjects] == [subject_id]
    assert {o.node_id for o in ctx.objects} == {OBJECT_ID, UNRELATED_ID}
    section = claim_prompt_section(ctx)
    assert subject_id in section and OBJECT_ID in section and QUOTE in section


def test_subject_is_the_node_the_materializer_merged_into_not_a_recomputed_id():
    # A second read of the same label merges into the first read's node (identity index).
    store, first_id, _, conn, _ = _setup()
    second = handoff(sha=text_sha256(FETCH_TEXT)).model_copy(update={"trace_id": "trace-second"})
    result = SubstrateGraphMaterializer(store=store).apply_record(
        __import__("orion.substrate.adapters.world_pulse_read", fromlist=["x"]).map_world_pulse_read_handoff_to_substrate(second))
    assert result.node_decisions[0].canonical_node_id == first_id
    texts = asyncio.run(load_retained_texts(conn, second.read_evidence))
    ctx = build_claim_context(store, second, texts)
    assert [s.node_id for s in ctx.subjects] == [first_id]


# ── the rule ───────────────────────────────────────────────────────────────


def test_heat_pump_fixture_accepts_projects_and_walks_both_ways():
    store, subject_id, ctx, _, _ = _setup()
    journal, report, projected = _run(store, ctx, [_claim(subject_id=subject_id)])
    (claim,) = report.claims
    assert claim.receipt.outcome == "accepted_provisional" and claim.receipt.reason == "accepted"
    assert claim.receipt.representation == "tool_digest"
    start, end = claim.receipt.span_start, claim.receipt.span_end
    assert FETCH_TEXT.encode()[start:end].decode() == QUOTE
    (decision,) = journal.decisions()
    assert (decision.policy, decision.resulting_state, decision.authority) == (
        "reading_quote_rule_v1", "provisional", "local_inferred")
    assert projected.applied == [decision.decision_id]
    out = _walk(store, subject_id)
    assert [(e.source.node_id, e.predicate, e.target.node_id, e.edge_role)
            for e in out.boundary_edges] == [(subject_id, "associated_with", OBJECT_ID, "semantic_projection")]
    back = _walk(store, OBJECT_ID)
    assert [e.source.node_id for e in back.boundary_edges] == [subject_id]


def test_a_quote_not_in_the_retained_text_stays_proposed_with_no_edge():
    store, subject_id, ctx, _, _ = _setup()
    paraphrase = "Heat pumps rely on the refrigeration cycle to move heat"
    journal, report, projected = _run(store, ctx, [_claim(subject_id=subject_id, quote=paraphrase)])
    assert report.claims[0].receipt.outcome == "proposed"
    assert report.claims[0].receipt.reason == "quote_not_found"
    assert len(journal.proposals()) == 1 and journal.decisions() == []
    assert projected.applied == [] and _walk(store, subject_id).boundary_edges == []


def test_a_one_word_quote_is_not_evidence():
    store, subject_id, ctx, _, _ = _setup()
    _, report, _ = _run(store, ctx, [_claim(subject_id=subject_id, quote="heat pump")])
    assert (report.claims[0].receipt.outcome, report.claims[0].receipt.reason) == ("proposed", "quote_too_short")


def test_an_unknown_object_id_writes_nothing_and_mints_no_node():
    store, subject_id, ctx, _, _ = _setup()
    before = set(store.snapshot().nodes)
    journal, report, _ = _run(store, ctx, [_claim(subject_id=subject_id, object_id="concept-invented")])
    assert (report.claims[0].receipt.outcome, report.claims[0].receipt.reason) == ("rejected", "unknown_object")
    assert journal.events == {}
    assert set(store.snapshot().nodes) == before and store.snapshot().edges == {}


def test_a_disallowed_predicate_is_refused_at_validation():
    store, subject_id, ctx, _, _ = _setup()
    journal, report, _ = _run(store, ctx, [_claim(subject_id=subject_id, predicate="activates")])
    assert (report.claims[0].receipt.outcome, report.claims[0].receipt.reason) == ("rejected", "predicate_not_allowed")
    assert journal.events == {}


def test_domain_rule_keeps_a_concept_to_entity_subtype_claim_proposed():
    store, subject_id, ctx, _, _ = _setup()
    entity_ctx = ClaimContextV1(subjects=ctx.subjects, texts=ctx.texts, haystack=ctx.haystack,
                                objects=(EndpointV1(node_id=OBJECT_ID, kind="entity", label="Refrigeration cycle"),))
    plans = plan_claims([_claim(subject_id=subject_id, predicate="subtype_of")], entity_ctx,
                        seed_id="s", recorded_at=NOW)
    assert (plans[0].claim.receipt.outcome, plans[0].claim.receipt.reason) == ("proposed", "domain_rule:subtype_of")
    assert plans[0].proposal is not None and plans[0].decision is None


def test_retry_is_idempotent_no_duplicate_assertion_or_projection():
    store, subject_id, ctx, _, _ = _setup()
    journal, _, _ = _run(store, ctx, [_claim(subject_id=subject_id)])
    events, edges = dict(journal.events), dict(store.snapshot().edges)
    _, report, projected = _run(store, ctx, [_claim(subject_id=subject_id)], journal=journal)
    assert journal.events == events and projected.applied == []
    assert store.snapshot().edges == edges
    assert report.claims[0].receipt.outcome == "accepted_provisional"
    assert sum(1 for n in store.snapshot().nodes.values() if n.node_kind == "assertion") == 1


def test_another_read_of_the_same_statement_adds_support_not_a_second_decision():
    store, subject_id, ctx, _, _ = _setup()
    journal, _, _ = _run(store, ctx, [_claim(subject_id=subject_id)])
    plans = plan_claims([_claim(subject_id=subject_id)], ctx, seed_id="finding:other-read", recorded_at=NOW)
    report = asyncio.run(journal_claims(journal, plans))
    assert (report.claims[0].receipt.outcome, report.claims[0].receipt.reason) == ("proposed", "already_decided")
    assert len(journal.proposals()) == 2 and len(journal.decisions()) == 1


def test_the_memory_projector_never_picks_up_a_reading_decision():
    store, subject_id, ctx, _, _ = _setup()
    journal = FakeJournal()
    asyncio.run(journal_claims(journal, plan_claims([_claim(subject_id=subject_id)], ctx, seed_id="s", recorded_at=NOW)))
    memory = asyncio.run(_projector(store, journal, actors=("memory.referents",)).run_once())
    assert memory.applied == [] and memory.failed == {} and journal.materializations() == []


# ── cognition isolation (#2497 rule 8) ────────────────────────────────────


def _dynamics(store):
    result = SubstrateDynamicsEngine(store=store).tick(now=NOW)
    return (sorted((u.node_id, round(u.new_activation, 9), u.reason) for u in result.activation_updates),
            sorted((u.node_id, round(u.new_pressure, 9), u.reason) for u in result.pressure_updates))


def _surprised(store, node_id):
    """Give the reading concept prediction error, so pressure WOULD flow along any edge
    dynamics admits (otherwise the comparison below is vacuous)."""
    node = store.get_node_by_id(node_id)
    store.upsert_node(identity_key=store.get_identity_key_by_node_id(node_id), node=node.model_copy(update={
        "metadata": {**node.metadata, "prediction_error": 0.8},
        "signals": node.signals.model_copy(update={"salience": 0.9})}))


def test_reading_assertions_and_proposals_do_not_change_dynamics():
    before, subject_id, ctx, _, _ = _setup()
    after, _, _, _, _ = _setup()
    _surprised(before, subject_id)
    _surprised(after, subject_id)
    # One accepted (assertion node + structure + projection) and one proposal-only claim.
    _run(after, ctx, [_claim(subject_id=subject_id),
                      _claim(subject_id=subject_id, object_id=UNRELATED_ID, quote="Unlike a furnace, it does not burn fuel")])
    assert any(e.edge_role == "semantic_projection" for e in after.snapshot().edges.values())
    # One tick each: a tick persists what it computes.
    b, a = _dynamics(before), _dynamics(after)
    assert b[0] and b[1]  # activation and pressure really move in this fixture
    assert a == b


# ── Stage 2 wiring ─────────────────────────────────────────────────────────


def _patch_journal(monkeypatch, journal):
    # Patch the globals the pipeline class actually runs with (Hub's conftest re-imports
    # `scripts.*` per test, so a dotted-path patch can hit a different module object).
    monkeypatch.setitem(WorldPulseReadStage2Pipeline._record_claims.__globals__, "SubstrateGraphJournal",
                        lambda pool: journal)


def _pipeline(store, pool, *, enabled=True):
    return WorldPulseReadStage2Pipeline(
        enabled=True, tick_interval_sec=60, timeout_sec=60, session_id="s",
        pool_provider=lambda: pool, source_ref=ServiceRef(name="hub", version="0", node="t"),
        store_provider=lambda: store, assertions_enabled=enabled, projector_readiness=ALWAYS_READY,
    )


def _result(claims) -> WorldPulseReadStage2ResultV1:
    return WorldPulseReadStage2ResultV1(summary="s", trace_id="t", created_at=NOW, seed_id="finding:heat-pump",
                                        relationship_claims=claims)


def test_stage2_records_claims_with_receipts_and_projects(monkeypatch):
    store, subject_id, _, conn, h = _setup()
    journal = FakeJournal()
    _patch_journal(monkeypatch, journal)
    monkeypatch.setattr("orion.substrate.graph_journal.SubstrateGraphJournal", lambda pool: journal)
    pipe = _pipeline(store, FakePool(conn))
    pipe._claim_context = asyncio.run(pipe._build_claim_context(h))
    assert pipe._claim_context.usable
    assert QUOTE in _build_stage2_prompt(h, "t", pipe._claim_context)
    result = asyncio.run(pipe._record_claims("finding:heat-pump", _result([_claim(subject_id=subject_id)])))
    assert result.relationship_claims[0].receipt.outcome == "accepted_provisional"
    asyncio.run(pipe._project_assertions())
    assert [e.edge_role for e in _walk(store, subject_id).boundary_edges] == ["semantic_projection"]
    # The stored result round-trips with its receipts (the row is re-read by the repair path).
    again = WorldPulseReadStage2ResultV1.model_validate(result.model_dump(mode="json"))
    assert again.relationship_claims[0].receipt.reason == "accepted"


def test_a_model_written_receipt_is_discarded():
    forged = {"summary": "s", "relationship_claims": [{
        "subject_id": "a", "predicate": "causes", "object_id": "b", "statement_text": "x", "quote": "y",
        "receipt": {"outcome": "accepted_provisional", "reason": "accepted"}}]}
    result = _as_stage2_result(forged, fallback_trace="t", seed_id="s")
    assert result.relationship_claims[0].receipt is None


def test_kill_switch_drops_claims_and_prompt_block(monkeypatch):
    store, subject_id, _, conn, h = _setup()
    pipe = _pipeline(store, FakePool(conn), enabled=False)
    assert not asyncio.run(pipe._build_claim_context(h)).usable
    result = asyncio.run(pipe._record_claims("s", _result([_claim(subject_id=subject_id)])))
    assert result.relationship_claims == []
    asyncio.run(pipe._project_assertions())
    assert store.snapshot().edges == {}


def test_a_candidate_that_slipped_in_rank_after_the_prompt_was_bound_still_resolves(monkeypatch):
    store, subject_id, ctx, conn, h = _setup()
    journal = FakeJournal()
    _patch_journal(monkeypatch, journal)
    pipe = _pipeline(store, FakePool(conn))
    # The durable turn was bound with OBJECT_ID offered; by the time it finishes the
    # ranked region no longer lists it.
    pipe._claim_context = ClaimContextV1(subjects=ctx.subjects, objects=(), texts=ctx.texts, haystack=ctx.haystack)
    result = asyncio.run(pipe._record_claims("finding:heat-pump", _result([_claim(subject_id=subject_id)])))
    assert result.relationship_claims[0].receipt.outcome == "accepted_provisional"
    # ...but an id whose label never appears in the read is still refused.
    other = asyncio.run(pipe._record_claims("finding:heat-pump", _result([
        _claim(subject_id=subject_id, object_id="concept-missing")])))
    assert other.relationship_claims[0].receipt.reason == "unknown_object"


@pytest.mark.parametrize("retain", [True, False])
def test_success_frames_carry_fetch_text_only_for_reading_turns(retain):
    from orion.hub.turn_orchestrator import _success_frames
    from orion.schemas.harness_finalize import HarnessRunV1

    run = HarnessRunV1(correlation_id="c", final_text="ok", finalize_ran=True, step_count=3,
                       compliance_verdict="completed", grounding_status="grounded", source_fetches=[
        SourceFetchEvidenceV1(url=SOURCE_URL, tool_name="WebFetch", content_chars=5, content_text="hello")])
    final = next(f for f in _success_frames(run, correlation_id="c", retain_fetch_text=retain)
                 if f.get("type") == "final")
    assert ("content_text" in final["harness_source_fetches"][0]) is retain
