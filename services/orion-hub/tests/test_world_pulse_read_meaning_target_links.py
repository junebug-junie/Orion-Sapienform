"""reading_meaning_target_v1 (Juniper 2026-10-14): accept/reject matrix.

A claim to an object offered ONLY by meaning (concept_search; its label is not in the read)
is accepted to ``provisional`` only when the predicate is ``associated_with``, the verbatim
quote names the SUBJECT, both ids are real, and the object was in THIS read's offered list.
Exact-name objects keep reading_quote_rule_v1 unchanged. The flag off restores the old rule
and the old prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from orion.schemas.world_pulse_read import WorldPulseReadRelationshipClaimV1, WorldPulseReadStage2ResultV1
from orion.substrate.assertion_projector import AssertionProjector
from orion.substrate.materializer import SubstrateGraphMaterializer
from orion.substrate.reader_capability import ALWAYS_READY
from orion.world_pulse_read.assertions import (
    MEANING_POLICY,
    POLICY,
    READING_ACTOR,
    claim_prompt_section,
    journal_claims,
    plan_claims,
    stored_object_lookup,
)
from orion.world_pulse_read.concept_search import with_related_objects
from reading_assertion_fakes import NOW, OBJECT_ID, FakeJournal, FakeSemanticBackend, existing_concept
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline, _build_stage2_prompt
from test_world_pulse_read_claim_related_objects import (
    FAR_ID,
    QUOTE,
    RELATED_ID,
    VECTORS,
    _indexed,
    _pipeline,
    _related,
    _setup,
)

# Verbatim in FETCH_TEXT, but about thermal energy, not the heat pump (the subject).
QUOTE_NOT_SUBJECT = "moving thermal energy from a cold space to a warm one"


def _ctx(*, on=True, extra=()):
    store, subject_id, ctx, _, h = _setup(extra)
    widened = with_related_objects(ctx, _related(store, ctx, _indexed(store)))
    return store, subject_id, replace(widened, meaning_links=on), h


def _claim(subject_id, *, predicate="associated_with", object_id=RELATED_ID, quote=QUOTE):
    return WorldPulseReadRelationshipClaimV1(subject_id=subject_id, predicate=predicate, object_id=object_id,
                                             statement_text="A heat pump relates to thermal energy transfer.",
                                             quote=quote)


def _one(ctx, claim, h, store=None):
    lookup = stored_object_lookup(store, ctx) if store is not None else None
    return plan_claims([claim], ctx, seed_id=h.seed_ref.seed_id, recorded_at=NOW, object_lookup=lookup)[0]


def test_accepts_associated_with_when_the_quote_names_the_subject_and_projects_the_link():
    store, subject_id, ctx, h = _ctx()
    meaning = next(o for o in ctx.objects if o.node_id == RELATED_ID)
    assert meaning.via == "meaning" and meaning.score is not None
    plan = _one(ctx, _claim(subject_id), h, store)
    receipt = plan.claim.receipt
    assert (receipt.outcome, receipt.reason) == ("accepted_provisional", "accepted_meaning_target")
    assert plan.decision.policy == MEANING_POLICY and plan.decision.resulting_state == "provisional"
    assert f"{meaning.score:.3f}" in plan.decision.rationale
    assert receipt.content_sha256 and receipt.span_end > receipt.span_start
    journal = FakeJournal()
    report = asyncio.run(journal_claims(journal, [plan]))
    assert report.accepted == 1
    asyncio.run(AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                   readiness=ALWAYS_READY, proposal_actors=(READING_ACTOR,)).run_once())
    projected = [e for e in store.snapshot().edges.values() if e.edge_role == "semantic_projection"]
    assert [(e.source.node_id, e.predicate, e.target.node_id) for e in projected] == [
        (subject_id, "associated_with", RELATED_ID)]


def test_rejects_any_other_predicate_to_a_meaning_target():
    store, subject_id, ctx, h = _ctx()
    for predicate in ("part_of", "subtype_of", "causes", "co_occurs_with", "refines"):
        plan = _one(ctx, _claim(subject_id, predicate=predicate), h, store)
        assert (plan.claim.receipt.outcome, plan.claim.receipt.reason) == ("proposed", "meaning_target_predicate")
        assert plan.decision is None and plan.proposal is not None


def test_rejects_a_quote_that_does_not_name_the_subject():
    store, subject_id, ctx, h = _ctx()
    plan = _one(ctx, _claim(subject_id, quote=QUOTE_NOT_SUBJECT), h, store)
    assert (plan.claim.receipt.outcome, plan.claim.receipt.reason) == ("proposed", "quote_not_about_subject")
    assert plan.decision is None


def test_rejects_a_paraphrased_or_short_quote():
    store, subject_id, ctx, h = _ctx()
    para = _one(ctx, _claim(subject_id, quote="A heat pump moves heat with a refrigeration loop"), h, store)
    assert para.claim.receipt.reason == "quote_not_found" and para.decision is None
    short = _one(ctx, _claim(subject_id, quote="heat pump"), h, store)
    assert short.claim.receipt.reason == "quote_too_short" and short.decision is None


def test_rejects_a_target_this_read_was_not_offered():
    # FAR_ID is a real stored concept, but neither named in the read nor near its subject.
    store, subject_id, ctx, h = _ctx()
    assert FAR_ID not in {o.node_id for o in ctx.objects}
    plan = _one(ctx, _claim(subject_id, object_id=FAR_ID), h, store)
    assert (plan.claim.receipt.outcome, plan.claim.receipt.reason) == ("rejected", "unknown_object")
    assert plan.proposal is None
    # A context rebuilt without the meaning list (Hub restart) cannot accept it either.
    rebuilt = replace(ctx, objects=tuple(o for o in ctx.objects if o.via != "meaning"))
    again = _one(rebuilt, _claim(subject_id), h, store)
    assert again.claim.receipt.reason == "unknown_object" and again.decision is None


def test_rejects_unknown_subject_ids():
    store, _, ctx, h = _ctx()
    plan = _one(ctx, _claim("concept-not-this-read"), h, store)
    assert plan.claim.receipt.reason == "unknown_subject" and plan.proposal is None


def test_flag_off_restores_the_strict_quote_rule_and_prompt():
    store, subject_id, ctx, h = _ctx(on=False)
    plan = _one(ctx, _claim(subject_id), h, store)
    assert (plan.claim.receipt.outcome, plan.claim.receipt.reason) == ("proposed", "quote_not_about_endpoints")
    assert plan.decision is None
    prompt = claim_prompt_section(ctx)
    assert "recorded for review, not linked, unless the quote names it" in prompt
    assert "ONLY with predicate associated_with" not in prompt


def test_flag_on_prompt_says_meaning_targets_link_only_as_associated_with_naming_the_subject():
    _, _, ctx, h = _ctx()
    prompt = _build_stage2_prompt(h, "t", ctx)
    assert "ONLY with predicate associated_with AND a quote that names the SUBJECT" in prompt
    assert "for an object related by meaning: the subject" in prompt
    # No meaning objects -> no meaning wording at all.
    plain = replace(ctx, objects=tuple(o for o in ctx.objects if o.via != "meaning"))
    assert "related by meaning" not in claim_prompt_section(plain)


def test_exact_name_targets_keep_the_full_quote_rule():
    """The extension never reaches an object the text names: a quote naming only the subject
    is still not evidence for a link to a named object."""
    store, subject_id, ctx, h = _ctx()
    named = next(o for o in ctx.objects if o.via == "label")
    assert named.node_id == OBJECT_ID
    subject_only = _one(ctx, _claim(subject_id, object_id=OBJECT_ID, quote="Heat pumps explained. A heat pump transfers heat"), h, store)
    assert subject_only.claim.receipt.reason == "quote_not_about_endpoints" and subject_only.decision is None
    full = _one(ctx, _claim(subject_id, object_id=OBJECT_ID, predicate="part_of", quote=QUOTE), h, store)
    assert full.claim.receipt.reason == "accepted" and full.decision.policy == POLICY


def test_a_meaning_offered_object_the_text_does_name_is_an_exact_name_target():
    """Offered by meaning only because the region scan ranked it out; its label IS in the text,
    so the full quote rule applies (any allowed predicate), under reading_quote_rule_v1."""
    low = existing_concept("concept-thermal-energy", "thermal energy")
    store, subject_id, ctx0, _, h = _setup([low])
    backend = _indexed(store, dict(VECTORS, **{"thermal energy": (0.95, 0.2, 0.0)}))
    ctx0 = replace(ctx0, objects=tuple(o for o in ctx0.objects if o.node_id != low.node_id))
    ctx = replace(with_related_objects(ctx0, _related(store, ctx0, backend)), meaning_links=True)
    assert any(o.node_id == low.node_id and o.via == "meaning" for o in ctx.objects)
    plan = _one(ctx, _claim(subject_id, object_id=low.node_id, predicate="co_occurs_with", quote=QUOTE_NOT_SUBJECT), h, store)
    assert plan.claim.receipt.reason == "accepted" and plan.decision.policy == POLICY


def test_stage2_binds_the_flag_onto_the_read_and_records_the_policy(monkeypatch):
    for on, outcome, reason in ((True, "accepted_provisional", "accepted_meaning_target"),
                                (False, "proposed", "quote_not_about_endpoints")):
        store, subject_id, _, conn, h = _setup()
        journal = FakeJournal()
        monkeypatch.setitem(WorldPulseReadStage2Pipeline._record_claims.__globals__, "SubstrateGraphJournal",
                            lambda pool, j=journal: j)
        pipe = _pipeline(store, conn, FakeSemanticBackend(VECTORS), meaning_target_links=on)
        asyncio.run(pipe.index_concepts_once())
        pipe._claim_context = asyncio.run(pipe._build_claim_context(h))
        assert pipe._claim_context.meaning_links is on
        result = WorldPulseReadStage2ResultV1(summary="s", trace_id="t", created_at=NOW, seed_id="finding:heat-pump",
                                              relationship_claims=[_claim(subject_id)])
        receipt = asyncio.run(pipe._record_claims("finding:heat-pump", result)).relationship_claims[0].receipt
        assert (receipt.outcome, receipt.reason) == (outcome, reason)
        assert [d.policy for d in journal.decisions()] == ([MEANING_POLICY] if on else [])


def test_the_flag_defaults_on_in_the_pipeline_and_settings():
    import inspect

    from app.settings import Settings

    assert inspect.signature(WorldPulseReadStage2Pipeline).parameters["meaning_target_links"].default is True
    assert Settings.model_fields["HUB_WORLD_PULSE_READ_MEANING_TARGET_LINKS_ENABLED"].default is True


def test_only_an_object_in_this_reads_offered_list_counts_as_offered():
    """Even if a store lookup handed back a meaning endpoint the prompt never showed, the
    meaning rule refuses it: "offered" means present in the bound context, nothing else."""
    store, subject_id, ctx, h = _ctx()
    shown = next(o for o in ctx.objects if o.node_id == RELATED_ID)
    hidden = replace(ctx, objects=tuple(o for o in ctx.objects if o.node_id != RELATED_ID))
    plan = plan_claims([_claim(subject_id)], hidden, seed_id=h.seed_ref.seed_id, recorded_at=NOW,
                       object_lookup=lambda node_id: shown if node_id == RELATED_ID else None)[0]
    assert plan.decision is None and plan.claim.receipt.reason == "quote_not_about_endpoints"
