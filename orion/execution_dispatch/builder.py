from __future__ import annotations

from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1, ExecutionDispatchPolicyV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


def stable_execution_dispatch_frame_id(*, policy_frame_id: str, policy_id: str) -> str:
    return f"execution.dispatch.frame:{policy_frame_id}:{policy_id}"


def stable_dispatch_id(*, proposal_id: str, policy_id: str) -> str:
    return f"dispatch:{proposal_id}:{policy_id}"


def _proposal_by_id(proposal_frame: ProposalFrameV1) -> dict[str, ProposalCandidateV1]:
    return {c.proposal_id: c for c in proposal_frame.candidates}


def _is_hard_blocked(candidate: ProposalCandidateV1, policy: ExecutionDispatchPolicyV1) -> list[str]:
    hits: list[str] = []
    if candidate.proposal_kind in policy.hard_blocks:
        hits.append(f"proposal_kind:{candidate.proposal_kind}")
    blob = " ".join(
        [
            candidate.proposal_kind,
            candidate.proposed_effect,
            candidate.required_policy_gate,
            *candidate.execution_intent.values(),
        ]
    ).lower()
    for block in policy.hard_blocks:
        if block.lower() in blob:
            hits.append(block)
    if candidate.required_policy_gate in ("execution_policy", "autonomy_policy"):
        hits.append(f"policy_gate:{candidate.required_policy_gate}")
    return hits


def _resolve_dispatch_mode(
    *,
    policy: ExecutionDispatchPolicyV1,
    override_dispatch_mode: str | None,
) -> str:
    if override_dispatch_mode:
        return override_dispatch_mode
    return policy.mode.default_dispatch_mode


def _candidate_status_for_mode(dispatch_mode: str) -> tuple[str, str]:
    if dispatch_mode == "prepare_only":
        return "prepared", "prepare_only"
    if dispatch_mode == "dispatch_read_only":
        return "prepared", "dispatch_read_only"
    return "dry_run", "dry_run"


def build_execution_dispatch_frame(
    *,
    policy_frame: PolicyDecisionFrameV1,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: ExecutionDispatchPolicyV1,
    now: datetime | None = None,
    override_dispatch_mode: str | None = None,
) -> ExecutionDispatchFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    dispatch_mode = _resolve_dispatch_mode(policy=policy, override_dispatch_mode=override_dispatch_mode)
    proposals = _proposal_by_id(proposal_frame)

    candidates: list[ExecutionDispatchCandidateV1] = []
    blocked: list[ExecutionDispatchCandidateV1] = []
    dispatched: list[ExecutionDispatchCandidateV1] = []
    warnings: list[str] = list(policy_frame.warnings)

    if policy.mode.allow_mutating_dispatch:
        warnings.append("mutating_dispatch_disabled_in_v1")

    dispatch_status_default, dispatch_mode_candidate = _candidate_status_for_mode(dispatch_mode)
    max_candidates = policy.limits.max_dispatch_candidates

    def make_blocked(
        decision: PolicyDecisionV1,
        candidate: ProposalCandidateV1 | None,
        *,
        reasons: list[str],
        blocked_by: list[str],
    ) -> ExecutionDispatchCandidateV1:
        return ExecutionDispatchCandidateV1(
            dispatch_id=stable_dispatch_id(proposal_id=decision.proposal_id, policy_id=policy.policy_id),
            source_decision_id=decision.decision_id,
            source_proposal_id=decision.proposal_id,
            dispatch_status="blocked",
            dispatch_mode=dispatch_mode_candidate,
            dispatch_kind="noop",
            target_id=candidate.target_id if candidate else "unknown",
            target_kind=candidate.target_kind if candidate else "system",
            reasons=reasons,
            blocked_by=blocked_by,
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
        )

    for decision in policy_frame.decisions:
        candidate = proposals.get(decision.proposal_id)
        if candidate is None:
            blocked.append(
                make_blocked(
                    decision,
                    None,
                    reasons=["missing_proposal_candidate"],
                    blocked_by=["proposal_not_found"],
                )
            )
            continue

        hard_hits = _is_hard_blocked(candidate, policy)
        if hard_hits:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["hard_block"],
                    blocked_by=hard_hits,
                )
            )
            continue

        if decision.decision in policy.blocked_policy_decisions:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=[f"policy_decision:{decision.decision}"],
                    blocked_by=[decision.decision],
                )
            )
            continue

        if decision.decision not in policy.allowed_policy_decisions:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["decision_not_allowed_for_dispatch_v1"],
                    blocked_by=[decision.decision],
                )
            )
            continue

        route: CortexRouteTemplateV1 | None = policy.proposal_kind_to_cortex.get(candidate.proposal_kind)
        if route is None:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["no_cortex_route_for_proposal_kind"],
                    blocked_by=[candidate.proposal_kind],
                )
            )
            continue

        if route.allowed_scope not in ("inspect_only", "summarize_only"):
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["route_scope_not_read_only"],
                    blocked_by=[route.allowed_scope],
                )
            )
            continue

        if len(candidates) >= max_candidates:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["max_dispatch_candidates_exceeded"],
                    blocked_by=["limit"],
                )
            )
            continue

        dry_run = dispatch_mode != "dispatch_read_only"
        envelope = build_cortex_request_envelope(
            candidate=candidate,
            decision=decision,
            route=route,
            self_state=self_state,
            dry_run=dry_run,
        )
        dispatch_status = dispatch_status_default
        if dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only:
            dispatch_status = "dispatched"
        elif dispatch_mode == "dispatch_read_only":
            dispatch_status = "dry_run"
            warnings.append("dispatch_read_only_disabled_by_policy")

        item = ExecutionDispatchCandidateV1(
            dispatch_id=stable_dispatch_id(proposal_id=decision.proposal_id, policy_id=policy.policy_id),
            source_decision_id=decision.decision_id,
            source_proposal_id=decision.proposal_id,
            dispatch_status=dispatch_status,
            dispatch_mode=dispatch_mode_candidate,
            dispatch_kind=route.dispatch_kind,
            target_id=candidate.target_id,
            target_kind=candidate.target_kind,
            cortex_verb=route.cortex_verb,
            cortex_mode=route.cortex_mode,
            request_envelope=envelope,
            constraints={k: str(v) for k, v in envelope.get("constraints", {}).items()},
            reasons=["approved_read_only_dispatch_v1"],
            evidence_refs=list(decision.evidence_refs),
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
        )
        if dispatch_status == "dispatched":
            if len(dispatched) < policy.limits.max_dispatches_per_tick:
                dispatched.append(item)
            else:
                blocked.append(
                    make_blocked(
                        decision,
                        candidate,
                        reasons=["max_dispatches_per_tick_exceeded"],
                        blocked_by=["limit"],
                    )
                )
        else:
            candidates.append(item)

    dispatch_attempted = dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only
    return ExecutionDispatchFrameV1(
        frame_id=stable_execution_dispatch_frame_id(
            policy_frame_id=policy_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=proposal_frame.frame_id,
        source_self_state_id=self_state.self_state_id,
        execution_dispatch_policy_id=policy.policy_id,
        dispatch_mode=dispatch_mode,
        candidates=candidates,
        blocked_candidates=blocked,
        dispatched_candidates=dispatched,
        dispatch_attempted=dispatch_attempted,
        dispatch_count=len(dispatched),
        blocked_count=len(blocked),
        warnings=warnings,
    )


def build_unevaluable_execution_dispatch_frame(
    *,
    policy_frame: PolicyDecisionFrameV1,
    policy_id: str,
    reason: str,
    now: datetime | None = None,
) -> ExecutionDispatchFrameV1:
    """A policy frame whose proposal or self-state could not be loaded still
    needs an execution_dispatch_frame -- otherwise it's the oldest
    undispatched policy frame forever, permanently blocking every policy
    frame queued behind it in the FIFO
    `load_latest_policy_frame_without_dispatch` query. Mirrors
    orion.policy.builder.build_unevaluable_policy_decision_frame's reasoning
    exactly: record honestly (dispatch_attempted=False, zero candidates,
    a warning), don't silently drop or silently dispatch. Uses the same
    stable_execution_dispatch_frame_id as a real build would, so this is
    naturally superseded (not duplicated) if the dependency later loads.
    """
    return ExecutionDispatchFrameV1(
        frame_id=stable_execution_dispatch_frame_id(
            policy_frame_id=policy_frame.frame_id,
            policy_id=policy_id,
        ),
        generated_at=now or datetime.now(timezone.utc),
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=policy_frame.source_proposal_frame_id,
        source_self_state_id=policy_frame.source_self_state_id,
        execution_dispatch_policy_id=policy_id,
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=0,
        warnings=[*policy_frame.warnings, reason],
    )
