"""Offline eval: do reading relationship claims land as links only when the read supports them?

Recorded model-shaped Stage 2 claim lists (no live model) run through the real path:
retained text -> plan_claims -> journal -> AssertionProjector -> concept-region read (what
the Concept Atlas and Recall see). Measures, over the whole batch:

- unsupported_links: projected edges whose cited bytes are NOT in the retained text. Must be 0.
- supported_recall: verbatim-quoted, allowlisted claims between offered ids that became
  walkable links. Must be 1.0 (the rule is deterministic; anything less is a wiring bug).
- refused_writes: claims refused at validation that still wrote a journal row or a node.
  Must be 0.

This does not measure live model quality (how often a real Stage 2 quotes exactly); that
needs a completed live read, UNVERIFIED until the reading queue moves.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from orion.schemas.world_pulse_read import WorldPulseReadRelationshipClaimV1  # noqa: E402
from orion.substrate.assertion_projector import AssertionProjector  # noqa: E402
from orion.substrate.materializer import SubstrateGraphMaterializer  # noqa: E402
from orion.substrate.reader_capability import ALWAYS_READY  # noqa: E402
from orion.world_pulse_read.assertions import (  # noqa: E402
    READING_ACTOR,
    build_claim_context,
    journal_claims,
    plan_claims,
)
from orion.world_pulse_read.fetch_text import load_retained_texts, text_sha256  # noqa: E402
from reading_assertion_fakes import (  # noqa: E402
    FETCH_TEXT,
    NOW,
    OBJECT_ID,
    SOURCE_URL,
    UNRELATED_ID,
    FakeJournal,
    FakeSnapshotConn,
    handoff,
    seeded_store,
)

# (label, claim fields with SUBJECT placeholder, should_link)
CASES = [
    ("verbatim", dict(predicate="associated_with", object_id=OBJECT_ID,
                      quote="A heat pump transfers heat using a refrigeration cycle"), True),
    ("verbatim_contrast", dict(predicate="associated_with", object_id=UNRELATED_ID,
                               quote="Unlike a furnace, it does not burn fuel"), True),
    ("paraphrase", dict(predicate="causes", object_id=OBJECT_ID,
                        quote="heat pumps use refrigeration cycles to move heat"), False),
    ("whitespace_drift", dict(predicate="part_of", object_id=OBJECT_ID,
                              quote="A heat pump  transfers heat using a refrigeration cycle"), False),
    ("invented_object", dict(predicate="associated_with", object_id="concept-thermodynamics",
                             quote="A heat pump transfers heat using a refrigeration cycle"), False),
    ("operational_predicate", dict(predicate="activates", object_id=OBJECT_ID,
                                   quote="A heat pump transfers heat using a refrigeration cycle"), False),
    ("quote_about_something_else", dict(predicate="causes", object_id=UNRELATED_ID,
                                        quote="A heat pump transfers heat using a refrigeration cycle"), False),
    ("one_word_quote", dict(predicate="co_occurs_with", object_id=OBJECT_ID, quote="refrigeration"), False),
]
REFUSED = {"invented_object", "operational_predicate"}


def _run_batch():
    conn = FakeSnapshotConn()
    sha = text_sha256(FETCH_TEXT)
    conn.rows[sha] = (FETCH_TEXT, SOURCE_URL)
    h = handoff(sha=sha)
    store, subject = seeded_store(h)
    nodes_before = set(store.snapshot().nodes)
    texts = asyncio.run(load_retained_texts(conn, h.read_evidence))
    ctx = build_claim_context(store, h, texts)
    claims = [WorldPulseReadRelationshipClaimV1(subject_id=subject, statement_text=f"case {label}", **fields)
              for label, fields, _ in CASES]
    journal = FakeJournal()
    plans = plan_claims(claims, ctx, seed_id=h.seed_ref.seed_id, recorded_at=NOW)
    report = asyncio.run(journal_claims(journal, plans))
    asyncio.run(AssertionProjector(journal=journal, materializer=SubstrateGraphMaterializer(store=store),
                                   readiness=ALWAYS_READY, proposal_actors=(READING_ACTOR,)).run_once())
    return store, subject, journal, report, nodes_before


def test_reading_claims_link_only_what_the_retained_text_supports():
    store, subject, journal, report, nodes_before = _run_batch()
    region = store.read_concept_region(limit_nodes=50, limit_edges=50)
    linked = {e.assertion_id: e for e in region.edges if e.edge_role == "semantic_projection"}
    by_label = {label: claim for (label, _, _), claim in zip(CASES, report.claims)}

    unsupported = 0
    for edge in linked.values():
        claim = next(c for c in report.claims if c.receipt.assertion_id == edge.assertion_id)
        span = FETCH_TEXT.encode()[claim.receipt.span_start:claim.receipt.span_end].decode()
        unsupported += span != claim.quote
    expected = [label for label, _, should in CASES if should]
    supported_recall = sum(by_label[label].receipt.outcome == "accepted_provisional"
                           and by_label[label].receipt.assertion_id in linked for label in expected) / len(expected)
    refused_writes = sum(1 for p in journal.proposals() if p.statement_text in {f"case {r}" for r in REFUSED})
    refused_writes += len(set(store.snapshot().nodes) - nodes_before
                          - {n for n, v in store.snapshot().nodes.items() if v.node_kind == "assertion"})

    print(f"\nreading_assertion_eval cases={len(CASES)} linked={len(linked)} "
          f"unsupported_links={unsupported} supported_recall={supported_recall:.2f} "
          f"refused_writes={refused_writes} reasons="
          + ",".join(f"{label}:{c.receipt.reason}" for label, c in by_label.items()))
    assert unsupported == 0
    assert supported_recall == 1.0
    assert refused_writes == 0
    assert {e.target.node_id for e in linked.values()} == {OBJECT_ID, UNRELATED_ID}
    assert len(linked) == len(expected)
