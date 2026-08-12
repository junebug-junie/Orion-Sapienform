from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import (
    MAINTENANCE_SCOPE,
    CortexRouteTemplateV1,
    DispatchLimitsV1,
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


def proposal_template_key(proposal_id: str) -> str | None:
    """The template that produced a proposal, recovered from its id.

    `orion/proposals/builder.py::stable_proposal_id` builds
    `proposal:{template_key}:{field_tick_id}:{attention_frame_id}`. Splitting
    from the LEFT is what makes index 1 safe -- attention_frame_id itself
    contains colons ("attention.frame:tick_...") so counting from the right
    would not work. Returns None on anything that does not match, and every
    caller must have a real fallback: an id-format change must degrade this
    to a coarser key, never crash dispatch.
    """
    parts = proposal_id.split(":")
    if len(parts) >= 3 and parts[0] == "proposal" and parts[1]:
        return parts[1]
    return None


def starvation_key(candidate: ProposalCandidateV1) -> str:
    """Cross-tick identity for a repeatedly-proposed candidate.

    NOT proposal_id: that embeds the field_tick_id, so it is unique every
    tick and cannot answer "has this same thing been losing for an hour?" --
    which is the only question starvation counters exist to answer.

    The template key is load-bearing, and leaving it out was a live no-op bug
    (found 2026-08-12, minutes after deploy, in the frames this patch itself
    added). The first version keyed on `{proposal_kind}:{target_id}` alone --
    but distinct templates genuinely share a kind and target:

        inspect_execution_pressure  inspect  capability:orchestration
        inspect_attended_target     inspect  capability:orchestration

    `inspect_attended_target` averages rank 3.0 and is admitted nearly every
    tick; `inspect_execution_pressure` averages 7.5 and starves. Sharing one
    key meant the winner's reset popped the loser's counter every tick, so
    the candidate that most needed aging could never accrue any -- an aging
    mechanism that reported success while changing nothing, for precisely
    the population it existed to serve.

    target_id stays in the key on purpose: `inspect_attended_target` binds to
    whatever attention is on (node:substrate.chat, node:substrate.bus_synaptic,
    ...), and inspecting a different target really is a different piece of
    work, not a continuation of the same starved one.
    """
    template = proposal_template_key(candidate.proposal_id)
    if template is None:
        # Coarser, collision-prone, and better than crashing: a proposal_id
        # format change degrades aging, it does not stop dispatch.
        return f"kind:{candidate.proposal_kind}:{candidate.target_id}"
    return f"{template}:{candidate.target_id}"


def effective_priority(
    *,
    priority_score: float,
    starved_ticks: int,
    limits: DispatchLimitsV1,
) -> float:
    """Admission-ordering score: real priority plus a bounded aging bonus.

    The bonus is applied HERE and nowhere else -- the candidate's stored
    priority_score is never rewritten, so the proposal arena's own scoring
    stays auditable and this correction stays attributable to this layer.
    """
    bonus = min(
        limits.starvation_aging_bonus_cap,
        max(0, starved_ticks) * limits.starvation_aging_bonus_per_tick,
    )
    return priority_score + bonus


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


# How many consecutive losses stop being noise and start being a fault worth
# saying out loud in the frame. Uncalibrated starting value, disclosed as
# such: at the live ~90 active frames/hour measured 2026-08-12 this is a bit
# over 6 minutes of continuous losing.
STARVATION_WARN_TICKS = 10


def _starved_kind_counts(starved: list["_Eligible"]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in starved:
        counts[item.route.dispatch_kind] = counts.get(item.route.dispatch_kind, 0) + 1
    return counts


@dataclass(frozen=True)
class _Eligible:
    """A decision that cleared every gate and is now only competing for a slot."""

    decision: PolicyDecisionV1
    candidate: ProposalCandidateV1
    route: CortexRouteTemplateV1
    starved_ticks: int
    effective_priority: float


def _admit_candidates(
    eligible: list[_Eligible],
    limits: DispatchLimitsV1,
) -> tuple[list[_Eligible], list[_Eligible]]:
    """Split cleared candidates into (admitted, starved), highest first.

    Two passes, and the order matters:

      1. RESERVED -- each scope with a reservation takes up to that many
         slots from its own highest-priority candidates.
      2. GENERAL  -- everything still unadmitted competes for the remaining
         capacity, INCLUDING any reserved slot nobody claimed in pass 1.

    Pass 2 is what makes a reservation free: holding a slot for a scope that
    proposes nothing this tick would otherwise cost real throughput on every
    idle tick, which is how "reserve a lane" usually degrades into "dispatch
    fewer things." Nothing is held back here.
    """
    capacity = max(0, limits.max_dispatch_candidates)
    # Ties broken on proposal_id so the same inputs always produce the same
    # frame -- this builder is re-run on retries and the frame_id is stable.
    ranked = sorted(eligible, key=lambda e: (-e.effective_priority, e.decision.proposal_id))

    admitted_ids: set[str] = set()
    for scope, reserved in sorted(limits.reserved_slots_by_scope.items()):
        taken = 0
        for item in ranked:
            if taken >= reserved or len(admitted_ids) >= capacity:
                break
            if item.route.allowed_scope == scope and item.decision.decision_id not in admitted_ids:
                admitted_ids.add(item.decision.decision_id)
                taken += 1

    for item in ranked:
        if len(admitted_ids) >= capacity:
            break
        admitted_ids.add(item.decision.decision_id)

    admitted = [e for e in ranked if e.decision.decision_id in admitted_ids]
    starved = [e for e in ranked if e.decision.decision_id not in admitted_ids]
    return admitted, starved


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
    prev_starvation_counts: dict[str, int] | None = None,
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
    prev_counts = dict(prev_starvation_counts or {})
    eligible: list[_Eligible] = []

    def make_blocked(
        decision: PolicyDecisionV1,
        candidate: ProposalCandidateV1 | None,
        *,
        reasons: list[str],
        blocked_by: list[str],
        dispatch_kind: str = "noop",
        starvation_ticks: int = 0,
    ) -> ExecutionDispatchCandidateV1:
        return ExecutionDispatchCandidateV1(
            dispatch_id=stable_dispatch_id(proposal_id=decision.proposal_id, policy_id=policy.policy_id),
            source_decision_id=decision.decision_id,
            source_proposal_id=decision.proposal_id,
            dispatch_status="blocked",
            dispatch_mode=dispatch_mode_candidate,
            # 2026-08-12: was the hardcoded literal "noop" on EVERY blocked
            # record. That is what made starvation invisible: the stored row
            # for a starved MUTATING action was byte-identical in kind to a
            # starved inspect, so no query could ever surface "a maintenance
            # action has been losing for hours." Found by having to
            # reconstruct the kind from the proposal_id string to answer that
            # exact question. Only genuinely kind-less blocks (a decision
            # whose proposal or route could not be resolved at all) keep
            # "noop", and now they mean it.
            dispatch_kind=dispatch_kind,  # type: ignore[arg-type]
            target_id=candidate.target_id if candidate else "unknown",
            target_kind=candidate.target_kind if candidate else "system",
            reasons=reasons,
            blocked_by=blocked_by,
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
            starvation_ticks=starvation_ticks,
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
        # here ONLY when mode.allow_mutating_dispatch is open (ON since
        # 2026-08-12, by explicit operator decision). This is THE gate: the
        # decision record's own allowed_scope is never consulted here, so this
        # route-table check is what actually decides whether a mutating verb
        # can be dispatched at all.
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

        # Everything above is a real gate: pass it and the only thing left
        # between this candidate and a send is capacity. Capacity is decided
        # once, after the whole frame is known, in _admit_candidates -- not
        # first-come-first-served inside this loop, which is what let five
        # read-only templates hold all five slots indefinitely.
        starved_ticks = prev_counts.get(starvation_key(candidate), 0)
        eligible.append(
            _Eligible(
                decision=decision,
                candidate=candidate,
                route=route,
                starved_ticks=starved_ticks,
                effective_priority=effective_priority(
                    priority_score=candidate.priority_score,
                    starved_ticks=starved_ticks,
                    limits=policy.limits,
                ),
            )
        )

    admitted, starved = _admit_candidates(eligible, policy.limits)

    # Only non-zero counts are carried: a key absent from the map reads as 0
    # via .get(), so this stays proportional to what is actually starving
    # rather than to everything ever proposed.
    starvation_counts: dict[str, int] = {}
    for item in starved:
        starvation_counts[starvation_key(item.candidate)] = item.starved_ticks + 1
    # Applied second on purpose: if two eligible candidates collapse to the
    # same starvation key and one of them got in, the pair is not starving.
    for item in admitted:
        starvation_counts.pop(starvation_key(item.candidate), None)

    for item in starved:
        next_ticks = item.starved_ticks + 1
        blocked.append(
            make_blocked(
                item.decision,
                item.candidate,
                # reasons[0] is unchanged so existing queries and dashboards
                # built on this string keep working; the detail rides behind it.
                reasons=[
                    "max_dispatch_candidates_exceeded",
                    f"starved_consecutive_ticks:{next_ticks}",
                    f"effective_priority:{item.effective_priority:.4f}",
                ],
                blocked_by=[f"max_dispatch_candidates:{policy.limits.max_dispatch_candidates}"],
                dispatch_kind=item.route.dispatch_kind,
                starvation_ticks=next_ticks,
            )
        )

    # A frame-level, greppable record of what lost. CLAUDE.md 0A: a cap that
    # silently truncates reads as "covered everything" -- this arena did that
    # for its entire life and the omission was only findable by querying for
    # an absence.
    for kind, count in sorted(_starved_kind_counts(starved).items()):
        warnings.append(f"dispatch_starved kind={kind} candidates={count}")
    worst = max((e.starved_ticks + 1 for e in starved), default=0)
    if worst >= STARVATION_WARN_TICKS:
        warnings.append(f"dispatch_starvation_persistent max_consecutive_ticks={worst}")

    for eligible_item in admitted:
        decision = eligible_item.decision
        candidate = eligible_item.candidate
        route = eligible_item.route
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
            # 2026-08-12: was the hardcoded literal "approved_read_only_dispatch_v1"
            # on EVERY dispatched candidate. Once a mutating route exists that
            # string is simply false for it, and this is the reason recorded on
            # the stored candidate -- the same class of false-audit-record
            # defect as the decision literal one layer up.
            reasons=[
                "approved_maintenance_dispatch_v1"
                if route.allowed_scope == MAINTENANCE_SCOPE
                else "approved_read_only_dispatch_v1"
            ],
            evidence_refs=list(decision.evidence_refs),
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
            # Kept on the ADMITTED record too, not just the blocked one: "this
            # finally got in after losing 200 ticks" is exactly the evidence
            # that says whether aging is working, and it is unrecoverable
            # after the fact if only the blocked side carries it.
            starvation_ticks=eligible_item.starved_ticks,
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
        starvation_counts=starvation_counts,
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
