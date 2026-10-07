"""``source_cooccurrence_v1``: the one relationship memory may assert (spec 2.2). Pure.

Two referents (neither Juniper nor Orion) are claimed to co-occur when ONE verified
prompt quote of the memory (Juniper's own words) names both, through a live alias of
each. It records that they were mentioned together, never why: #2497 forbids
narrating ``co_occurs_with`` as causation.

The claim is ACCEPTED (a decision to ``provisional``) only when all hold:
- the policy is on (``MEMORY_COOCCURRENCE_AUTO_ACCEPT``);
- both endpoints are provisional/canonical nodes;
- the memory yields no more than ``MAX_ACCEPTED_PER_MEMORY`` such claims (a long list
  is a topic dump, not a statement that these things belong together);
- the assertion has no earlier decision (a second memory adds a proposal, not a revision).
Otherwise only the proposal is journalled, for review.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from orion.core.schemas.substrate_graph_journal import SubstrateGraphDecisionV1, SubstrateGraphProposalV1
from orion.substrate.assertion_projector import assertion_node_id

from .aliases import alias_in_text, slug_text
from .resolve import LIVE_STATES, REFERENT_NAMESPACE, REFERENT_PRODUCER

POLICY = "source_cooccurrence_v1"
MAX_ACCEPTED_PER_MEMORY = 6


def evidence_node_id(memory_id: str) -> str:
    return f"ev-{uuid.uuid5(REFERENT_NAMESPACE, f'episode_memory:{memory_id}')}"


@dataclass(frozen=True)
class EndpointV1:
    key: str
    node_id: str
    node_kind: str        # entity | concept
    state: str
    names: tuple[str, ...]  # live alias norms


@dataclass(frozen=True)
class CooccurrenceClaim:
    proposal: SubstrateGraphProposalV1
    decision: Optional[SubstrateGraphDecisionV1]


def cooccurrence_claims(
    *,
    memory_id: str,
    endpoints: list[EndpointV1],
    prompt_quotes: list[str],
    decided_targets: set[str],
    accept: bool,
    recorded_at: datetime,
    exclude_node_ids: set[str] | frozenset[str] = frozenset(),
) -> list[CooccurrenceClaim]:
    """``exclude_node_ids``: Juniper's and Orion's nodes, by node id (not key)."""
    named: dict[str, EndpointV1] = {e.node_id: e for e in endpoints if e.node_id not in exclude_node_ids}
    pairs: set[tuple[str, str]] = set()
    for quote in prompt_quotes:
        present = sorted(nid for nid, e in named.items() if any(alias_in_text(n, quote) for n in e.names))
        pairs |= {(a, b) for i, a in enumerate(present) for b in present[i + 1:]}
    within_cap = len(pairs) <= MAX_ACCEPTED_PER_MEMORY
    claims: list[CooccurrenceClaim] = []
    for a, b in sorted(pairs):
        sub, obj = named[a], named[b]
        statement_key = f"{a}|co_occurs_with|{b}|"
        target = assertion_node_id(statement_key)
        proposal_id = str(uuid.uuid5(REFERENT_NAMESPACE, f"{memory_id}|{statement_key}"))
        proposal = SubstrateGraphProposalV1(
            proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target,
            actor=REFERENT_PRODUCER, subject_node_id=a, subject_kind=sub.node_kind, object_node_id=b,
            object_kind=obj.node_kind, predicate="co_occurs_with", statement_key=statement_key,
            statement_text=f"Juniper mentioned {slug_text(sub.key)} and {slug_text(obj.key)} together",
            anchor_scope="juniper", authority="user_asserted",
            supporting_evidence_ids=[evidence_node_id(memory_id)], recorded_at=recorded_at,
        )
        decision = None
        if (accept and within_cap and sub.state in LIVE_STATES and obj.state in LIVE_STATES
                and target not in decided_targets):
            decision = SubstrateGraphDecisionV1(
                proposal_id=proposal_id, proposal_kind="relationship_assertion", target_id=target,
                actor=REFERENT_PRODUCER, decision_id=str(uuid.uuid5(REFERENT_NAMESPACE, f"{proposal_id}|accept")),
                expected_prior_revision=0, resulting_state="provisional", policy=POLICY,
                authority="local_inferred", rationale="named together in one verified prompt quote",
                evidence_refs=[f"episode_memory:{memory_id}"], recorded_at=recorded_at,
            )
            decided_targets.add(target)
        claims.append(CooccurrenceClaim(proposal=proposal, decision=decision))
    return claims
