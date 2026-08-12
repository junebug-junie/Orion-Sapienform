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
    """Does this candidate leave the world unchanged?

    2026-08-12: a mutating kind is never read-only, whatever its effect.

    This used to fall straight through to `proposed_effect in
    read_only_effects`, which is a category error: `proposed_effect` says what
    an action is FOR, not what it DOES. The build-cache prune declares
    `preserve_stability` -- an honest goal for a housekeeping action, and a
    member of `read_only_effects` -- so a verb that deletes ~142 GB of host
    files was classified read-only, approved through the `read_only_low_risk`
    branch, and persisted a decision record saying so.

    Checking the kind's own declared `mutating` flag first fixes the whole
    class rather than special-casing `maintain`, so the next mutating kind
    does not re-arm the same trap by inheriting an effect that happens to be
    on the list.
    """
    rule = kind_rule(policy, candidate.proposal_kind)
    if rule is not None and rule.mutating:
        return False
    if candidate.proposal_kind in ("observe", "inspect", "summarize"):
        return True
    return candidate.proposed_effect in policy.read_only_effects


def kind_rule(policy: SubstratePolicyV1, proposal_kind: str) -> ProposalKindRuleV1 | None:
    return policy.proposal_kind_rules.get(proposal_kind)
