from __future__ import annotations

from orion.policy.policy import SubstratePolicyV1, ProposalKindRuleV1
from orion.schemas.proposal_frame import ProposalCandidateV1


def stable_policy_decision_id(*, proposal_id: str, policy_id: str) -> str:
    return f"policy.decision:{proposal_id}:{policy_id}"


def execution_intent_blob(candidate: ProposalCandidateV1) -> str:
    parts: list[str] = []
    for key, value in candidate.execution_intent.items():
        parts.append(str(key))
        parts.append(str(value))
    return " ".join(parts).lower()


def hard_block_hits(candidate: ProposalCandidateV1, policy: SubstratePolicyV1) -> list[str]:
    blob = execution_intent_blob(candidate)
    return [block for block in policy.hard_blocks if block.lower() in blob]


def is_read_only_candidate(candidate: ProposalCandidateV1, policy: SubstratePolicyV1) -> bool:
    if candidate.proposal_kind in ("observe", "inspect", "summarize"):
        return True
    return candidate.proposed_effect in policy.read_only_effects


def kind_rule(policy: SubstratePolicyV1, proposal_kind: str) -> ProposalKindRuleV1 | None:
    return policy.proposal_kind_rules.get(proposal_kind)
