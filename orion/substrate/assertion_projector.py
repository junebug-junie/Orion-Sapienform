"""Apply journal decisions about relationship assertions to the substrate graph.

The single writer of Assertion nodes, their structure edges and their semantic
projections (#2497 "Relationships"; memory Stage 2 spec 2.2-2.3). Every write goes
through ``SubstrateGraphMaterializer.apply_record``, and every attempt ends in a
``SubstrateGraphMaterializationV1`` naming the canonical ids the materializer
actually returned.

What it writes for one decision (revision r, state s):
- the Assertion node (state s, revision r, decision_ref = the decision);
- ``assertion_subject`` / ``assertion_object`` edges to the two endpoints and one
  ``supports`` edge per supporting Evidence node (edge_role=assertion_structure);
- when s is provisional/canonical: the semantic edge subject -predicate-> object
  with edge_role=semantic_projection, assertion_id, assertion_revision=r;
- when s is rejected/deprecated and a projection exists: the same edge closed
  (valid_to = decision time, revision r), which also makes it unwalkable because
  the neighborhood checks the assertion's state and revision.

Fails closed, never mints placeholders: a missing endpoint or evidence node, an
out-of-order revision, or a deterministic id already held by a different stored id
(checked before writing) is recorded as ``outcome=failed`` and nothing is written for
that decision. A failure is recorded again only when its reason changes; stale,
mismatched and missing-proposal failures are terminal and never retried.

Nothing is written until ``readiness()`` says every substrate reader can read the new
shapes (orion/substrate/reader_capability.py).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from orion.core.schemas.cognitive_substrate import (
    AssertionNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.core.schemas.substrate_graph_journal import (
    SubstrateGraphDecisionV1,
    SubstrateGraphMaterializationV1,
    SubstrateGraphProposalV1,
)

from .graph_journal import SubstrateGraphJournal
from .materializer import SubstrateGraphMaterializer
from .neighborhood import ACCEPTED_ASSERTION_STATES
from .reader_capability import ReadinessCheck, ReadinessV1
from .reconcile import SubstrateIdentityResolver

logger = logging.getLogger(__name__)

PROJECTOR_ACTOR = "substrate.assertion_projector"
_ID_NAMESPACE = uuid.UUID("9b4f0f86-3a51-4d0e-9a39-1f2f7c3f2d61")


def assertion_node_id(statement_key: str) -> str:
    """Deterministic Assertion id for a resolved statement (one Assertion per statement)."""
    return f"assertion-{uuid.uuid5(_ID_NAMESPACE, statement_key)}"


def _edge_id(*parts: str) -> str:
    return f"edge-{uuid.uuid5(_ID_NAMESPACE, '|'.join(parts))}"


def _materialization_id(decision_id: str, outcome: str, attempt: int = 0) -> str:
    """Applied: one per decision. Failed: one per attempt whose reason CHANGED (A, B, A is three)."""
    return f"mat-{uuid.uuid5(_ID_NAMESPACE, f'{decision_id}|{outcome}|{attempt}')}"


@dataclass
class ProjectionReportV1:
    applied: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    waiting: list[str] = field(default_factory=list)
    blocked: ReadinessV1 | None = None     # set when readers are not ready: nothing was written


class AssertionProjector:
    def __init__(self, *, journal: SubstrateGraphJournal, materializer: SubstrateGraphMaterializer,
                 readiness: ReadinessCheck, proposal_actors: tuple[str, ...] | None = None) -> None:
        """``proposal_actors``: apply only claims proposed by these producers (None = all)."""
        self._readiness = readiness
        self._proposal_actors = proposal_actors
        self._journal = journal
        self._materializer = materializer
        self._store = materializer.store
        self._identity = SubstrateIdentityResolver()

    async def run_once(self, *, limit: int = 100) -> ProjectionReportV1:
        report = ProjectionReportV1()
        ready = await asyncio.to_thread(self._readiness)
        if not ready.ready:
            report.blocked = ready
            logger.info("assertion_projector_waiting reason=%s missing=%s", ready.reason, list(ready.missing))
            return report
        for decision in await self._journal.pending_decisions(limit=limit, proposal_actors=self._proposal_actors):
            if decision.proposal_kind != "relationship_assertion":
                continue
            applied_revision = await self._journal.latest_applied_revision(decision.target_id)
            if decision.expected_prior_revision > applied_revision:
                # An earlier revision of this target has not landed yet (it failed,
                # or is later in this batch). Apply revisions strictly in order.
                report.waiting.append(decision.decision_id)
                continue
            if decision.expected_prior_revision < applied_revision:
                await self._fail(decision, "stale_revision", report)
                continue
            proposal = await self._journal.proposal(decision.proposal_id)
            if proposal is None or proposal.target_id != decision.target_id:
                await self._fail(decision, "proposal_missing_or_mismatched", report)
                continue
            missing = await asyncio.to_thread(self._missing_nodes, proposal)
            if missing:
                await self._fail(decision, f"endpoint_missing:{missing}", report)
                continue
            record, expected_nodes, expected_edges = await asyncio.to_thread(self._record, proposal, decision)
            if await asyncio.to_thread(self._identity_taken, record):
                # Checked BEFORE writing: one of our deterministic ids is already held by a
                # different stored id, so the write would land somewhere we would not report.
                await self._fail(decision, "canonical_id_mismatch", report)
                continue
            try:
                result = await asyncio.to_thread(self._materializer.apply_record, record)
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
                logger.exception("assertion_projection_failed decision_id=%s", decision.decision_id)
                await self._fail(decision, f"materializer_error:{type(exc).__name__}", report)
                continue
            node_ids = [d.canonical_node_id for d in result.node_decisions]
            edge_ids = [d.canonical_edge_id for d in result.edge_decisions]
            if node_ids != expected_nodes or edge_ids != expected_edges:
                # Defense in depth after the pre-check above (should be unreachable): the write
                # happened, but we do not claim a projection that is not where we said it is.
                logger.error("assertion_projection_id_mismatch_after_write decision_id=%s", decision.decision_id)
                await self._fail(decision, "canonical_id_mismatch", report)
                continue
            await self._journal.append(
                SubstrateGraphMaterializationV1(
                    proposal_id=decision.proposal_id,
                    proposal_kind=decision.proposal_kind,
                    target_id=decision.target_id,
                    actor=PROJECTOR_ACTOR,
                    materialization_id=_materialization_id(decision.decision_id, "applied"),
                    decision_id=decision.decision_id,
                    revision=decision.resulting_revision,
                    outcome="applied",
                    node_ids=node_ids,
                    edge_ids=edge_ids,
                )
            )
            report.applied.append(decision.decision_id)
            logger.info(
                "assertion_projected decision_id=%s assertion_id=%s revision=%d state=%s policy=%s edges=%d",
                decision.decision_id, decision.target_id, decision.resulting_revision,
                decision.resulting_state, decision.policy, len(edge_ids),
            )
        return report

    async def _fail(self, decision: SubstrateGraphDecisionV1, reason: str, report: ProjectionReportV1) -> None:
        report.failed[decision.decision_id] = reason
        last = await self._journal.last_materialization(decision.decision_id)
        if last is not None and last.outcome == "failed" and last.failure_reason == reason:
            return
        logger.warning("assertion_projection_refused decision_id=%s reason=%s", decision.decision_id, reason)
        await self._journal.append(
            SubstrateGraphMaterializationV1(
                proposal_id=decision.proposal_id,
                proposal_kind=decision.proposal_kind,
                target_id=decision.target_id,
                actor=PROJECTOR_ACTOR,
                materialization_id=_materialization_id(
                    decision.decision_id, "failed", 1 + await self._journal.failed_attempts(decision.decision_id)),
                decision_id=decision.decision_id,
                revision=decision.resulting_revision,
                outcome="failed",
                failure_reason=reason,
            )
        )

    def _identity_taken(self, record: SubstrateGraphRecordV1) -> bool:
        for node in record.nodes:
            held = self._store.get_node_id_by_identity(self._identity.canonical_node_key(node) or "")
            if held is not None and held != node.node_id:
                return True
        for edge in record.edges:
            held = self._store.get_edge_id_by_identity(self._identity.canonical_edge_key(edge))
            if held is not None and held != edge.edge_id:
                return True
        return False

    def _missing_nodes(self, proposal: SubstrateGraphProposalV1) -> str:
        wanted = [
            (proposal.subject_node_id, proposal.subject_kind),
            (proposal.object_node_id, proposal.object_kind),
            *[(evidence_id, "evidence") for evidence_id in proposal.supporting_evidence_ids],
        ]
        for node_id, kind in wanted:
            node = self._store.get_node_by_id(node_id)
            if node is None or node.node_kind != kind:
                return node_id
        return ""

    def _record(
        self, proposal: SubstrateGraphProposalV1, decision: SubstrateGraphDecisionV1
    ) -> tuple[SubstrateGraphRecordV1, list[str], list[str]]:
        decided_at = decision.recorded_at
        provenance = SubstrateProvenanceV1(
            authority=decision.authority,
            source_kind="substrate_graph_journal",
            source_channel=decision.policy,
            producer=proposal.actor,
            evidence_refs=sorted({*proposal.supporting_evidence_ids, decision.decision_id}),
        )
        assertion = AssertionNodeV1(
            node_id=proposal.target_id,
            anchor_scope=proposal.anchor_scope,
            subject_ref=proposal.subject_ref,
            promotion_state=decision.resulting_state,
            temporal=SubstrateTemporalWindowV1(observed_at=decided_at, valid_from=proposal.recorded_at),
            signals=SubstrateSignalBundleV1(confidence=0.5, salience=0.0),
            provenance=provenance,
            predicate=proposal.predicate,
            statement_key=proposal.statement_key,
            statement_text=proposal.statement_text,
            revision=decision.resulting_revision,
            decision_ref=decision.decision_id,
        )
        a_ref = NodeRefV1(node_id=assertion.node_id, node_kind="assertion")
        subject = NodeRefV1(node_id=proposal.subject_node_id, node_kind=proposal.subject_kind)
        obj = NodeRefV1(node_id=proposal.object_node_id, node_kind=proposal.object_kind)
        window = SubstrateTemporalWindowV1(observed_at=decided_at, valid_from=proposal.recorded_at)

        def edge(edge_id: str, source: NodeRefV1, target: NodeRefV1, predicate: str, **extra: Any) -> SubstrateEdgeV1:
            return SubstrateEdgeV1(
                edge_id=edge_id, source=source, target=target, predicate=predicate,
                temporal=extra.pop("temporal", window), provenance=provenance, **extra,
            )

        edges = [
            edge(_edge_id(assertion.node_id, "assertion_subject"), a_ref, subject, "assertion_subject",
                 edge_role="assertion_structure"),
            edge(_edge_id(assertion.node_id, "assertion_object"), a_ref, obj, "assertion_object",
                 edge_role="assertion_structure"),
            *[
                edge(_edge_id(evidence_id, "supports", assertion.node_id),
                     NodeRefV1(node_id=evidence_id, node_kind="evidence"), a_ref, "supports",
                     edge_role="assertion_structure")
                for evidence_id in sorted(set(proposal.supporting_evidence_ids))
            ],
        ]
        projection_id = _edge_id(assertion.node_id, "semantic_projection")
        accepted = decision.resulting_state in ACCEPTED_ASSERTION_STATES
        projection = edge(
            projection_id, subject, obj, proposal.predicate,
            edge_role="semantic_projection", assertion_id=assertion.node_id,
            assertion_revision=decision.resulting_revision,
            temporal=SubstrateTemporalWindowV1(
                observed_at=decided_at,
                valid_from=proposal.recorded_at,
                valid_to=None if accepted else max(decided_at, proposal.recorded_at),
            ),
        )
        projection_exists = self._store.get_edge_id_by_identity(
            self._identity.canonical_edge_key(projection)
        ) is not None
        if accepted or projection_exists:
            edges.append(projection)
        record = SubstrateGraphRecordV1(
            anchor_scope=proposal.anchor_scope, subject_ref=proposal.subject_ref,
            nodes=[assertion], edges=edges,
        )
        return record, [assertion.node_id], [e.edge_id for e in edges]
