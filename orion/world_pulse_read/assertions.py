"""Reading relationship claims: Stage 2 proposes, this module decides, the journal records.

#2497 sections 3-4, on the shared assertion core (#2515). Before this, a reading landed
concept nodes with ``edges=[]`` (orion/substrate/adapters/world_pulse_read.py) and
nothing ever linked them to the rest of the atlas.

Flow, all deterministic except the model's proposal:

1. ``build_claim_context``: the concepts this read produced, looked up as the ids the
   materializer actually stored (identity index, never recomputed), plus a bounded list of
   EXISTING atlas concepts whose label appears in the read (the same region-rank +
   label-in-text retrieval Recall's concept_region collector uses; no new score), plus the
   text Hub retained for the read (fetch_text.py).
2. Stage 2's model returns ``relationship_claims`` choosing ids from those two lists.
3. ``plan_claims`` validates each claim. Refused outright (not journalled, no node, no
   edge): a predicate outside the reader allowlist, an id not in the lists Hub offered,
   the same id on both ends. Otherwise a ``SubstrateGraphProposalV1`` is journalled, and
   ``reading_quote_rule_v1`` ACCEPTS it to ``provisional`` only when all hold:
   (a) the quote is found verbatim in a text Hub retained for this read;
   (b) both endpoints resolved to stored Concept/Entity ids (step 1);
   (c) the predicate passes its domain rule.
   Anything else stays ``proposed``: journalled, no link. Juniper can reject later.
4. ``AssertionProjector`` (run by Hub with ``proposal_actors=(READING_ACTOR,)``) writes the
   Assertion node and its semantic projection edge, which the neighborhood read walks.

Retained text for a web fetch is the fetch tool's digest (``tool_digest``), not the page;
the receipt says which. Reads stored before retention have no text and get no claims.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
from orion.schemas.world_pulse_read import (
    READING_CLAIM_PREDICATES,
    WorldPulseReadClaimReceiptV1,
    WorldPulseReadHandoffV1,
    WorldPulseReadRelationshipClaimV1,
)
from orion.substrate.adapters.world_pulse_read import map_world_pulse_read_handoff_to_substrate
from orion.substrate.assertion_projector import assertion_node_id
from orion.substrate.graph_journal import RevisionConflict
from orion.substrate.reconcile import SubstrateIdentityResolver, is_identity_fenced

from .fetch_text import RetainedTextV1

logger = logging.getLogger(__name__)

READING_ACTOR = "world_pulse_read_stage2"
POLICY = "reading_quote_rule_v1"
_NS = uuid.UUID("5d0c7e1a-6f0b-4b5e-8a43-2c1d9b7e4f10")

# A cited quote must be a statement, not a word: a single label ("refrigeration") is found
# verbatim in almost any text that names the concept and supports no relationship.
MIN_QUOTE_CHARS = 20
MIN_QUOTE_WORDS = 4
MAX_STATEMENT_CHARS = 400
MAX_SUBJECTS = 12
MAX_OBJECTS = 24
MAX_CLAIMS = 12
# Recall's concept_region collector constants (services/orion-recall/app/collectors/
# concept_region.py): generous region scan, labels under 3 chars match too much.
REGION_SCAN_LIMIT = 500
MIN_LABEL_CHARS = 3
# Retained text shown to the Stage 2 model, across all texts of the read.
PROMPT_TEXT_BUDGET_CHARS = 16000

_SEMANTIC_KINDS = frozenset({"concept", "entity"})


@dataclass(frozen=True)
class EndpointV1:
    node_id: str
    kind: str
    label: str


@dataclass(frozen=True)
class ClaimContextV1:
    subjects: tuple[EndpointV1, ...] = ()
    objects: tuple[EndpointV1, ...] = ()
    texts: tuple[RetainedTextV1, ...] = ()
    # What object labels were matched against (the read's own words + retained text).
    haystack: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.subjects and self.objects and self.texts)


def _stored_endpoint(store: Any, node_id: Optional[str]) -> Optional[EndpointV1]:
    if not node_id:
        return None
    node = store.get_node_by_id(node_id)
    if node is None or node.node_kind not in _SEMANTIC_KINDS:
        return None
    return EndpointV1(node_id=node.node_id, kind=node.node_kind, label=str(getattr(node, "label", "") or ""))


def resolve_reading_subjects(store: Any, handoff: WorldPulseReadHandoffV1) -> list[EndpointV1]:
    """The stored nodes Stage 1 wrote for this read's concepts.

    Looks up what the materializer stored, the way it stored it: the identity index first
    (reconcile may have merged the concept into an older node with the same identity),
    then the incoming id. Both subject refs a reading concept can carry are tried, since
    the seed's request is re-derived between stages. A concept that is not stored yields
    nothing: no placeholder."""
    resolver = SubstrateIdentityResolver()
    out: dict[str, EndpointV1] = {}
    for node in map_world_pulse_read_handoff_to_substrate(handoff).nodes:
        keys = []
        for subject_ref in (node.subject_ref, "world_pulse", "reading"):
            key = resolver.canonical_node_key(node.model_copy(update={"subject_ref": subject_ref}))
            if key and key not in keys:
                keys.append(key)
        found = None
        for key in keys:
            found = _stored_endpoint(store, store.get_node_id_by_identity(key))
            if found is not None:
                break
        found = found or _stored_endpoint(store, node.node_id)
        if found is not None and found.node_id not in out:
            out[found.node_id] = found
        if len(out) >= MAX_SUBJECTS:
            break
    return list(out.values())


def _label_in(label: str, text_lower: str) -> bool:
    norm = str(label or "").strip().lower()
    return len(norm) >= MIN_LABEL_CHARS and norm in text_lower


def object_candidates(store: Any, *, haystack: str, exclude_ids: Iterable[str]) -> list[EndpointV1]:
    """Existing atlas concepts whose label appears in the read, in the store's rank order."""
    text = haystack.lower()
    excluded = set(exclude_ids)

    def keep(label: str) -> bool:
        return _label_in(label, text)

    matching = getattr(store, "read_concept_region_matching", None)
    if callable(matching):
        region = matching(keep_label=keep, limit_nodes=REGION_SCAN_LIMIT, limit_edges=1)
    else:
        region = store.read_concept_region(limit_nodes=REGION_SCAN_LIMIT, limit_edges=1)
    out: list[EndpointV1] = []
    for node in getattr(region, "nodes", None) or []:
        if node.node_id in excluded or node.node_kind not in _SEMANTIC_KINDS:
            continue
        if is_identity_fenced(node):
            # Memory referents (private recall) are never linked from a public reading.
            continue
        label = str(getattr(node, "label", "") or "")
        if not keep(label):
            continue
        out.append(EndpointV1(node_id=node.node_id, kind=node.node_kind, label=label))
        excluded.add(node.node_id)
        if len(out) >= MAX_OBJECTS:
            break
    return out


def build_claim_context(
    store: Any, handoff: WorldPulseReadHandoffV1, texts: list[RetainedTextV1]
) -> ClaimContextV1:
    if not texts:
        return ClaimContextV1()
    subjects = resolve_reading_subjects(store, handoff)
    if not subjects:
        return ClaimContextV1(texts=tuple(texts))
    haystack = "\n".join([handoff.what_i_learned, *(t.text for t in texts)])
    objects = object_candidates(store, haystack=haystack, exclude_ids={s.node_id for s in subjects})
    return ClaimContextV1(subjects=tuple(subjects), objects=tuple(objects), texts=tuple(texts),
                          haystack=haystack)


def stored_object_lookup(store: Any, ctx: ClaimContextV1) -> Callable[[str], Optional[EndpointV1]]:
    """Re-check an object id the model chose against the store with the candidate rule
    (stored concept/entity, not a fenced memory node, label in this read), minus the rank
    cut. A durable Stage 2 turn binds its prompt once and may finish hours later, after the
    region ranking that produced the offered list has moved; a still-valid candidate from
    that prompt must not be refused because it slipped in rank."""
    subject_ids = {s.node_id for s in ctx.subjects}
    text = ctx.haystack.lower()

    def lookup(node_id: str) -> Optional[EndpointV1]:
        if not node_id or node_id in subject_ids:
            return None
        node = store.get_node_by_id(node_id)
        if node is None or node.node_kind not in _SEMANTIC_KINDS or is_identity_fenced(node):
            return None
        label = str(getattr(node, "label", "") or "")
        return EndpointV1(node_id=node.node_id, kind=node.node_kind, label=label) if _label_in(label, text) else None

    return lookup


def claim_prompt_section(ctx: ClaimContextV1) -> str:
    """The extra Stage 2 prompt block; empty when there is nothing to link."""
    if not ctx.usable:
        return ""
    budget = PROMPT_TEXT_BUDGET_CHARS
    excerpts = []
    for t in ctx.texts:
        if budget <= 0:
            break
        shown = t.text[:budget]
        budget -= len(shown)
        excerpts.append(f"--- retained {t.representation} of {t.url} (sha256 {t.sha256[:12]}) ---\n{shown}")

    def listing(items: tuple[EndpointV1, ...]) -> str:
        return "\n".join(f"  {e.node_id} = {e.label!r} ({e.kind})" for e in items)

    return (
        "\n\nRELATIONSHIP CLAIMS (optional; an empty list is a fine answer). You may link a concept "
        "this read produced (subject) to an EXISTING concept (object) only when the text below "
        "states the relationship. Use only these ids.\n"
        f"subjects:\n{listing(ctx.subjects)}\n"
        f"objects:\n{listing(ctx.objects)}\n"
        f"predicate must be one of: {', '.join(READING_CLAIM_PREDICATES)}. subtype_of and refines "
        "relate two concepts; part_of relates two things of the same kind; co_occurs_with means "
        "only that the source mentions both, never causation. If no predicate fits without "
        "distortion, propose nothing.\n"
        "quote: copy the supporting sentence EXACTLY, character for character, from the text "
        "below. A paraphrased quote is not accepted.\n"
        'Add the key "relationship_claims": [{"subject_id": "...", "predicate": "...", '
        '"object_id": "...", "statement_text": "one plain sentence", "quote": "exact text"}]\n'
        + "\n".join(excerpts)
    )


def domain_violation(predicate: str, subject_kind: str, object_kind: str) -> Optional[str]:
    """#2497 reader domain rules. None when the predicate fits these endpoint kinds."""
    if subject_kind not in _SEMANTIC_KINDS or object_kind not in _SEMANTIC_KINDS:
        return f"domain_rule:{predicate}"
    if predicate in {"subtype_of", "refines"} and (subject_kind, object_kind) != ("concept", "concept"):
        return f"domain_rule:{predicate}"
    if predicate == "part_of" and subject_kind != object_kind:
        return f"domain_rule:{predicate}"
    return None


def find_quote(texts: Iterable[RetainedTextV1], quote: str) -> Optional[tuple[RetainedTextV1, int, int]]:
    """First retained text containing the quote verbatim, with its half-open UTF-8 byte span.
    Matching encoded bytes of a whole UTF-8 string always lands on character boundaries."""
    needle = quote.encode("utf-8")
    if not needle:
        return None
    for t in texts:
        start = t.text.encode("utf-8").find(needle)
        if start >= 0:
            return t, start, start + len(needle)
    return None


@dataclass
class ClaimPlanV1:
    claim: WorldPulseReadRelationshipClaimV1
    proposal: Optional[SubstrateGraphProposalV1] = None
    decision: Optional[SubstrateGraphDecisionV1] = None


def _refused(claim: WorldPulseReadRelationshipClaimV1, reason: str) -> ClaimPlanV1:
    receipt = WorldPulseReadClaimReceiptV1(outcome="rejected", reason=reason)
    return ClaimPlanV1(claim=claim.model_copy(update={"receipt": receipt}))


def plan_claims(
    claims: Iterable[WorldPulseReadRelationshipClaimV1],
    ctx: ClaimContextV1,
    *,
    seed_id: str,
    recorded_at: datetime,
    object_lookup: Optional[Callable[[str], Optional[EndpointV1]]] = None,
) -> list[ClaimPlanV1]:
    """The journal events (if any) and receipt for every claim, in input order. Pure unless
    ``object_lookup`` (``stored_object_lookup``) reads the store for an id not in the list."""
    subjects = {e.node_id: e for e in ctx.subjects}
    objects = {e.node_id: e for e in ctx.objects}
    seen: set[str] = set()
    plans: list[ClaimPlanV1] = []
    for raw in list(claims)[:MAX_CLAIMS]:
        claim = raw.model_copy(update={
            "subject_id": raw.subject_id.strip(), "object_id": raw.object_id.strip(),
            "predicate": raw.predicate.strip(), "receipt": None,
        })
        if claim.predicate not in READING_CLAIM_PREDICATES:
            plans.append(_refused(claim, "predicate_not_allowed"))
            continue
        sub, obj = subjects.get(claim.subject_id), objects.get(claim.object_id)
        if obj is None and object_lookup is not None and claim.object_id not in subjects:
            obj = object_lookup(claim.object_id)
        if sub is None:
            plans.append(_refused(claim, "unknown_subject"))
            continue
        if obj is None:
            plans.append(_refused(claim, "unknown_object"))
            continue
        if sub.node_id == obj.node_id:
            plans.append(_refused(claim, "same_endpoint"))
            continue
        statement_key = f"{sub.node_id}|{claim.predicate}|{obj.node_id}|"
        if statement_key in seen:
            plans.append(_refused(claim, "duplicate"))
            continue
        seen.add(statement_key)
        target = assertion_node_id(statement_key)
        proposal_id = f"reading-{uuid.uuid5(_NS, f'{seed_id}|{statement_key}')}"
        proposal = SubstrateGraphProposalV1(
            proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target,
            actor=READING_ACTOR, subject_node_id=sub.node_id, subject_kind=sub.kind,
            object_node_id=obj.node_id, object_kind=obj.kind, predicate=claim.predicate,
            statement_key=statement_key, statement_text=claim.statement_text.strip()[:MAX_STATEMENT_CHARS],
            anchor_scope="orion", subject_ref="reading", authority="local_inferred",
            supporting_evidence_ids=[], recorded_at=recorded_at,
        )
        quote_ok = len(claim.quote.strip()) >= MIN_QUOTE_CHARS and len(claim.quote.split()) >= MIN_QUOTE_WORDS
        found = find_quote(ctx.texts, claim.quote) if quote_ok else None
        span = {}
        if found is not None:
            text, start, end = found
            span = {"content_sha256": text.sha256, "representation": text.representation,
                    "span_start": start, "span_end": end}
        reason = (
            "quote_too_short" if not quote_ok
            else "quote_not_found" if found is None
            else domain_violation(claim.predicate, sub.kind, obj.kind)
        )
        decision = None
        if reason is None:
            text, start, end = found
            decision = SubstrateGraphDecisionV1(
                proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target,
                actor=READING_ACTOR, decision_id=f"reading-{uuid.uuid5(_NS, f'{proposal_id}|accept')}",
                expected_prior_revision=0, resulting_state="provisional", policy=POLICY,
                authority="local_inferred",
                rationale=f"quote found verbatim in the retained {text.representation} of {text.url}",
                evidence_refs=[f"reading_text:{text.sha256}#bytes={start}-{end}", f"reading_seed:{seed_id}"],
                recorded_at=recorded_at,
            )
        receipt = WorldPulseReadClaimReceiptV1(
            outcome="accepted_provisional" if decision else "proposed", reason=reason or "accepted",
            proposal_id=proposal_id, assertion_id=target,
            decision_id=decision.decision_id if decision else None, **span,
        )
        plans.append(ClaimPlanV1(claim=claim.model_copy(update={"receipt": receipt}),
                                 proposal=proposal, decision=decision))
    return plans


@dataclass
class JournalReportV1:
    proposals: int = 0
    accepted: int = 0
    claims: list[WorldPulseReadRelationshipClaimV1] = field(default_factory=list)


async def journal_claims(journal: Any | None, plans: list[ClaimPlanV1]) -> JournalReportV1:
    """Append each plan's proposal, then its decision. Idempotent on retry (event ids are
    deterministic, so a replay writes nothing twice). A decision another read already made
    for the same assertion stays out: this read's proposal is kept as extra support and the
    receipt says ``already_decided`` (that includes a claim Juniper rejected; a reading never
    overrides a review). A journal error is written on the receipt, never swallowed."""
    report = JournalReportV1()

    def amend(claim: WorldPulseReadRelationshipClaimV1, **update: Any) -> WorldPulseReadRelationshipClaimV1:
        return claim.model_copy(update={"receipt": claim.receipt.model_copy(update=update)})

    for plan in plans:
        claim = plan.claim
        if plan.proposal is None:
            report.claims.append(claim)
            continue
        try:
            if journal is None:
                raise RuntimeError("no_journal")
            await journal.append(plan.proposal)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reading_claim_journal_failed proposal_id=%s err=%s",
                           plan.proposal.proposal_id, type(exc).__name__)
            # Not recorded anywhere but this receipt: no proposal, no decision.
            report.claims.append(amend(claim, outcome="rejected", reason="journal_unavailable",
                                       proposal_id=None, decision_id=None))
            continue
        report.proposals += 1
        if plan.decision is not None:
            try:
                await journal.append(plan.decision)
                report.accepted += 1
            except RevisionConflict:
                claim = amend(claim, outcome="proposed", reason="already_decided", decision_id=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reading_claim_decision_journal_failed decision_id=%s err=%s",
                               plan.decision.decision_id, type(exc).__name__)
                claim = amend(claim, outcome="proposed", reason="decision_journal_failed", decision_id=None)
        report.claims.append(claim)
    return report
