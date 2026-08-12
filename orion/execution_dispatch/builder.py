from __future__ import annotations

from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import (
    MAINTENANCE_SCOPE,
    CortexRouteTemplateV1,
    ExecutionDispatchPolicyV1,
)
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1


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
    field_tick_id: str,
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

    # 2026-08-12: this flag used to append a warning and do nothing else --
    # set it and mutating dispatch stayed off, silently. That is the inverse of
    # CLAUDE.md 0A's "kill means kill": a switch that reports success while
    # changing nothing. It now actually gates the `maintenance_bounded` scope
    # below, and its OFF state is announced rather than its ON state, because
    # off is the state that silently blocks something an operator asked for.
    if not policy.mode.allow_mutating_dispatch:
        warnings.append("mutating_dispatch_disabled_by_policy")

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

        # `maintenance_bounded` is the only non-read-only scope, and it passes
        # here ONLY when the operator has explicitly opened
        # mode.allow_mutating_dispatch. Default false, so this branch preserves
        # exactly the previous behaviour until someone chooses otherwise.
        scope_allowed = route.allowed_scope in ("inspect_only", "summarize_only") or (
            route.allowed_scope == MAINTENANCE_SCOPE and policy.mode.allow_mutating_dispatch
        )
        if not scope_allowed:
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
            field_tick_id=field_tick_id,
            dry_run=dry_run,
        )
        dispatch_status = dispatch_status_default
        if dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only:
            dispatch_status = "prepared_for_dispatch"
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
        # Unreachable from this builder as of the prepared_for_dispatch fix above --
        # nothing here sets dispatch_status="dispatched" anymore, so max_dispatches_per_tick
        # is not enforced until a future patch's sender promotes candidates to a real,
        # evidenced "dispatched" after actually attempting a send.
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
        source_field_tick_id=field_tick_id,
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
    """A policy frame whose proposal could not be loaded still needs an
    execution_dispatch_frame -- otherwise it's the oldest undispatched policy
    frame forever, permanently blocking every policy frame queued behind it
    in the FIFO `load_oldest_policy_frames_without_dispatch` query. Record
    honestly (dispatch_attempted=False, zero candidates, a warning), don't
    silently drop or silently dispatch. Uses the same
    stable_execution_dispatch_frame_id as a real build would, so this is
    naturally superseded (not duplicated) if the dependency later loads.

    2026-07-22 (SelfStateV1 burn): this used to also cover a missing
    self-state load; that branch is gone from the worker now that
    build_execution_dispatch_frame takes field_tick_id directly off the
    already-loaded policy_frame (no separate load to fail).
    """
    return ExecutionDispatchFrameV1(
        frame_id=stable_execution_dispatch_frame_id(
            policy_frame_id=policy_frame.frame_id,
            policy_id=policy_id,
        ),
        generated_at=now or datetime.now(timezone.utc),
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=policy_frame.source_proposal_frame_id,
        source_field_tick_id=policy_frame.source_field_tick_id,
        execution_dispatch_policy_id=policy_id,
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=0,
        warnings=[*policy_frame.warnings, reason],
    )


def _decision_template_key(proposal_id: str) -> str:
    # stable_proposal_id() (orion/proposals/builder.py) is always
    # "proposal:{template_key}:{field_tick_id}:{attention_frame_id}" --
    # template_key is a plain config dict key (never contains ":"), so
    # index [1] is exact, same convention scripts/analysis/measure_proposal_
    # feedback_correlation.py's extract_template_key() already relies on.
    parts = proposal_id.split(":")
    return parts[1] if len(parts) > 1 else proposal_id


def build_stale_discard_execution_dispatch_frame(
    *,
    policy_frame: PolicyDecisionFrameV1,
    policy_id: str,
    age_sec: float,
    staleness_threshold_sec: float,
    now: datetime | None = None,
) -> ExecutionDispatchFrameV1:
    """A policy frame past its staleness window gets discarded rather than
    dispatched -- real cortex-exec dispatches are a synchronous, ~7-11s RPC
    (measured live 2026-07-30) each, so a single-threaded FIFO consumer
    cannot keep pace with real production of new policy_decision_frames
    (measured live: ~16/min produced vs. ~6.8/min consumed). Continuing to
    process a growing backlog strictly oldest-first means eventually
    dispatching real cortex actions that describe field pressure as "current"
    when it is actually hours old -- a "no empty-shell cognition" violation
    in its own right, not just a throughput problem. See docs/superpowers/
    specs/2026-07-30-execution-dispatch-staleness-discard-design.md.

    Materializes what was discarded rather than silently dropping it: one
    `stale_discard:{template_key}:{decision}` warning entry per candidate
    that was in the discarded policy frame, so real discard content stays
    queryable in `substrate_execution_dispatch_frames.warnings` (no new
    table -- this repo's existing "no keyword cathedral" bias against a new
    schema/surface unless it earns its keep). The worker separately tracks
    an EWMA of discard counts (`ExecutionDispatchFrameV1.staleness_discard_
    count_ewma`) as the aggregate backlog-pressure signal; this per-frame
    warning list is the per-decision forensic detail underneath it.

    Same "record honestly, never silently drop" contract as
    `build_unevaluable_execution_dispatch_frame` above, and the same stable
    frame_id so a later real build (should this exact policy_frame somehow
    still be the FIFO head, which it won't be once discarded) naturally
    supersedes rather than duplicates.
    """
    summary = [
        f"stale_discard:{_decision_template_key(d.proposal_id)}:{d.decision}"
        for d in policy_frame.decisions
    ]
    reason = (
        f"stale_backlog_discarded age_sec={age_sec:.1f} "
        f"threshold_sec={staleness_threshold_sec:.1f} candidates={len(policy_frame.decisions)}"
    )
    return ExecutionDispatchFrameV1(
        frame_id=stable_execution_dispatch_frame_id(
            policy_frame_id=policy_frame.frame_id,
            policy_id=policy_id,
        ),
        generated_at=now or datetime.now(timezone.utc),
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=policy_frame.source_proposal_frame_id,
        source_field_tick_id=policy_frame.source_field_tick_id,
        execution_dispatch_policy_id=policy_id,
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=len(policy_frame.decisions),
        warnings=[*policy_frame.warnings, reason, *summary],
    )
