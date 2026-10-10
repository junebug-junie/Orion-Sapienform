from __future__ import annotations

from datetime import datetime, timezone

from orion.reverie.baseline import validate_eligibility
from orion.schemas.reverie_visual import VisualBaselineEligibilityV1

from orion.field.action_warrant import action_warrant
from orion.field.pressure import field_pressures as compute_field_pressures
from dataclasses import dataclass, field as dc_field
from typing import Any, Mapping

from orion.autonomy.self_shed import bind_workspace_winner
from orion.proposals.policy import WORKSPACE_WINNER_BINDING, ProposalPolicyV1, ProposalTemplateV1
from orion.proposals.scoring import (
    clamp01,
    proposal_confidence,
    proposal_priority,
    proposal_risk,
    proposal_urgency,
    template_match_score,
)
from orion.proposals.templates import (
    cast_policy_gate,
    cast_proposal_kind,
    cast_proposed_effect,
    cast_target_kind,
    template_title_description,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1


# The only recognized attention-binding literal in v1 -- no general
# binding-expression DSL, matched exactly against ProposalTemplateV1.target_binding.
# Renamed 2026-07-22 (SelfStateV1 burn) from "self_state.dominant_attention_targets[0]"
# -- attention targets were always FieldAttentionFrameV1.dominant_targets underneath;
# self_state was a pass-through hop, not the real source. config/proposals/
# proposal_policy.v1.yaml's target_binding literal was updated to match in the
# same changeset.
ATTENTION_FIRST_TARGET_BINDING = "attention.dominant_targets[0]"

# Intersection of FieldAttentionTargetV1.target_kind's Literal
# ("node", "capability", "channel", "edge", "field", "system") and
# ProposalCandidateV1.target_kind's Literal
# ("node", "capability", "field", "self_state", "service", "system").
# A resolved attention target outside this set fails closed to the
# template's literal target_id/target_kind rather than raising.
_ATTENTION_BOUND_TARGET_KINDS = frozenset({"node", "capability", "field", "system"})


def _is_internal_target(target: FieldAttentionTargetV1) -> bool:
    """False for a world-first external target. Frames built before
    world-first (no marker) only ever held internal targets."""
    from orion.attention.field_attention.selectors import field_target_source_kind

    return field_target_source_kind(target) != "external"


def stable_proposal_frame_id(*, field_tick_id: str, attention_frame_id: str, policy_id: str) -> str:
    return f"proposal.frame:{field_tick_id}:{attention_frame_id}:{policy_id}"


def stable_proposal_id(*, template_key: str, field_tick_id: str, attention_frame_id: str) -> str:
    return f"proposal:{template_key}:{field_tick_id}:{attention_frame_id}"


def _resolve_binding_target(
    *,
    template: ProposalTemplateV1,
    attention: FieldAttentionFrameV1 | None,
) -> tuple[str, str, str | None]:
    """Resolve a template's target_id/target_kind from live attention if the
    template declares a recognized binding and resolution succeeds; otherwise
    fail closed to the template's literal target_id/target_kind. Never raises.

    Returns (target_id, target_kind, binding_resolved_from).
    """
    if template.target_binding != ATTENTION_FIRST_TARGET_BINDING:
        return template.target_id, template.target_kind, None
    if attention is None:
        return template.target_id, template.target_kind, None
    # World-first attention (spec 2026-10-07 self-calibration, section A):
    # the frame's winner may be the WORLD (chat, camera surprise). A
    # self-modification proposal must only ever bind to an internal target,
    # so external winners are skipped and the first internal one is used.
    # No internal target (a world-only or no-winner frame) fails closed to
    # the template's literal target, same as an empty frame always did.
    resolved: FieldAttentionTargetV1 | None = next(
        (t for t in attention.dominant_targets if _is_internal_target(t)), None
    )
    if resolved is None:
        return template.target_id, template.target_kind, None
    if resolved.target_kind not in _ATTENTION_BOUND_TARGET_KINDS:
        return template.target_id, template.target_kind, None
    return resolved.target_id, resolved.target_kind, template.target_binding


def _build_candidate(
    *,
    template_key: str,
    template: ProposalTemplateV1,
    field: FieldStateV1,
    field_tick_id: str,
    attention: FieldAttentionFrameV1 | None,
    pressures: dict[str, float],
    policy: ProposalPolicyV1,
) -> ProposalCandidateV1:
    match_score, motivating_dimensions = template_match_score(
        field_pressures=pressures,
        template=template,
        policy=policy,
    )
    urgency = proposal_urgency(field_pressures=pressures, template=template)
    confidence = proposal_confidence(field=field, field_pressures=pressures, template=template)
    priority = proposal_priority(
        base_priority=template.base_priority,
        match_score=match_score,
        urgency=urgency,
        confidence=confidence,
    )
    risk = proposal_risk(
        base_risk=template.base_risk,
        field_pressures=pressures,
        template=template,
    )
    resolved_target_id, resolved_target_kind, binding_resolved_from = _resolve_binding_target(
        template=template,
        attention=attention,
    )
    title, description, reasons = template_title_description(
        template_key,
        target_id=resolved_target_id,
    )
    attention_frame_id = attention.frame_id if attention is not None else "none"
    evidence_refs = [
        f"field:{field_tick_id}",
        f"attention:{attention_frame_id}",
    ]
    motivating_targets = (
        [t.target_id for t in attention.dominant_targets[:5]] if attention is not None else []
    )
    execution_intent: dict[str, str] = {
        "mode": "descriptive_only",
        "template": template_key,
        "policy_gate": template.required_policy_gate,
    }
    if template.kind == "request_policy_review":
        execution_intent["note"] = "policy_review_not_execution"
    return ProposalCandidateV1(
        proposal_id=stable_proposal_id(
            template_key=template_key,
            field_tick_id=field_tick_id,
            attention_frame_id=attention_frame_id,
        ),
        proposal_kind=cast_proposal_kind(template.kind),
        title=title,
        description=description,
        target_id=resolved_target_id,
        target_kind=cast_target_kind(resolved_target_kind),
        priority_score=priority,
        urgency_score=urgency,
        confidence_score=confidence,
        risk_score=risk,
        reversibility_score=clamp01(template.reversibility),
        # 2026-08-21: carried straight through from the template, not
        # re-derived. The claim belongs to the template author; every layer
        # below is transport for it.
        expected_signal=template.expected_signal,
        expected_direction=template.expected_direction,  # type: ignore[arg-type]
        motivating_dimensions=motivating_dimensions,
        motivating_targets=motivating_targets,
        evidence_refs=sorted(set(evidence_refs)),
        reasons=reasons,
        proposed_effect=cast_proposed_effect(template.proposed_effect),
        required_policy_gate=cast_policy_gate(template.required_policy_gate),
        execution_intent=execution_intent,
        binding_resolved_from=binding_resolved_from,
    )


@dataclass(frozen=True)
class WorkspaceWinnerContext:
    """Inputs for ``workspace.winner`` templates (attend-to-act loop D1), read by the runtime.

    ``projection``: the substrate_attention_broadcast_projection row's JSON; ``broadcast_log_id``: the
    substrate_attention_broadcast_log row for the same tick; ``eligibility``: template key ->
    eligibility snapshot (orion.autonomy.self_shed.evaluate_shed_eligibility). A template with no
    snapshot is never emitted (fail closed)."""

    projection: Mapping[str, Any] | None
    broadcast_log_id: str | None
    eligibility: Mapping[str, Mapping[str, Any]] = dc_field(default_factory=dict)


def _build_workspace_candidates(
    *,
    policy: ProposalPolicyV1,
    workspace: WorkspaceWinnerContext | None,
    field_tick_id: str,
    now: datetime,
    warnings: list[str],
) -> list[ProposalCandidateV1]:
    """World actions bound to the workspace winner. Each is emitted only when the winner binds AND
    the world is in the state the action is for; otherwise the frame records why, never silently.

    These bypass the tick-level ``action_warrant`` gate on purpose, like the visual baseline: that gate
    asks whether Orion's INTERNAL state is busier than a median day, while a world action carries its
    own stricter, physical trigger (elevated AND rising, reflex idle, background work present), all
    recorded on ``world_eligibility``. They still pass policy, the allocator floor and the pool's caps."""
    out: list[ProposalCandidateV1] = []
    for key, template in policy.proposal_templates.items():
        if template.target_binding != WORKSPACE_WINNER_BINDING:
            continue
        if workspace is None:
            warnings.append(f"winner_unbindable:no_workspace_input:{key}")
            continue
        winner, why = bind_workspace_winner(
            workspace.projection, broadcast_log_id=workspace.broadcast_log_id,
            binds_to_nodes=template.binds_to_nodes, now=now)
        if winner is None:
            warnings.append(f"{why}:{key}")
            continue
        snapshot = workspace.eligibility.get(key)
        if not snapshot or not snapshot.get("eligible"):
            refusals = ",".join((snapshot or {}).get("refusals") or ["no_eligibility_snapshot"])
            warnings.append(f"world_action_ineligible:{key}:{refusals}")
            continue
        urgency = clamp01(float(((snapshot.get("cabinet") or {}).get("warming_error")) or 0.0))
        title, description, reasons = template_title_description(key, target_id=template.target_id)
        out.append(ProposalCandidateV1(
            proposal_id=stable_proposal_id(template_key=key, field_tick_id=field_tick_id,
                                           attention_frame_id=winner.broadcast_log_id),
            proposal_kind=cast_proposal_kind(template.kind),
            title=title,
            description=description,
            target_id=template.target_id,
            target_kind=cast_target_kind(template.target_kind),
            priority_score=clamp01(max(template.base_priority, policy.thresholds.min_priority)),
            urgency_score=urgency,
            # The trigger is a fresh physical reading checked against fixed rules, not a field
            # estimate: confidence is the reading's, and an unfresh reading never gets here.
            confidence_score=1.0,
            risk_score=clamp01(template.base_risk),
            reversibility_score=clamp01(template.reversibility),
            expected_signal=template.expected_signal,
            expected_direction=template.expected_direction,  # type: ignore[arg-type]
            motivating_dimensions={},
            motivating_targets=[winner.node_id],
            evidence_refs=sorted({f"field:{field_tick_id}", f"broadcast:{winner.broadcast_log_id}",
                                  f"open_loop:{winner.open_loop_id}"}),
            reasons=[*reasons, f"workspace_winner:{winner.node_id}"],
            proposed_effect=cast_proposed_effect(template.proposed_effect),
            required_policy_gate=cast_policy_gate(template.required_policy_gate),
            execution_intent={"mode": "world_action", "template": key,
                              "policy_gate": template.required_policy_gate,
                              "open_loop_id": winner.open_loop_id},
            binding_resolved_from=WORKSPACE_WINNER_BINDING,
            attention_winner=winner,
            world_eligibility={**dict(snapshot), "holdback_fraction": template.holdback_fraction},
        ))
    return out


def _overall_action_pressure(candidates: list[ProposalCandidateV1]) -> float:
    if not candidates:
        return 0.0
    return clamp01(max(c.priority_score for c in candidates))


def _overall_risk(candidates: list[ProposalCandidateV1]) -> float:
    if not candidates:
        return 0.0
    return clamp01(max(c.risk_score for c in candidates))


def _policy_required(
    *,
    candidates: list[ProposalCandidateV1],
    overall_risk: float,
    policy: ProposalPolicyV1,
) -> bool:
    if overall_risk >= policy.thresholds.policy_required_above_risk:
        return True
    return any(c.required_policy_gate not in ("none", "read_only") for c in candidates)


def _dominant_motivations(candidates: list[ProposalCandidateV1]) -> list[str]:
    counts: dict[str, float] = {}
    for candidate in candidates:
        for dim_id, weight in candidate.motivating_dimensions.items():
            counts[dim_id] = counts.get(dim_id, 0.0) + float(weight)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [dim_id for dim_id, _ in ranked[:5]]


def build_proposal_frame(
    *,
    field: FieldStateV1,
    attention: FieldAttentionFrameV1 | None,
    policy: ProposalPolicyV1,
    previous_frame: ProposalFrameV1 | None = None,
    now: datetime | None = None,
    external_candidates: list[ProposalCandidateV1] | None = None,
    baseline_eligibility: VisualBaselineEligibilityV1 | None = None,
    workspace: WorkspaceWinnerContext | None = None,
) -> ProposalFrameV1:
    """Build a ProposalFrameV1 directly from FieldStateV1 + FieldAttentionFrameV1.

    2026-07-22 (SelfStateV1 burn): previously took a SelfStateV1 as its
    primary input and `field` was received but discarded ("reserved for
    continuity in later revisions") -- self_state was always a lossy,
    hand-tuned-weight pass-through of field/attention data for this
    consumer's purposes. `field` is load-bearing now, not reserved.
    """
    del previous_frame  # reserved for continuity in later revisions
    generated_at = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    if attention is not None and attention.source_field_tick_id != field.tick_id:
        warnings.append(
            f"attention_source_tick_mismatch:{attention.source_field_tick_id}!={field.tick_id}"
        )

    pressures = compute_field_pressures(field)
    attention_frame_id = attention.frame_id if attention is not None else "none"

    built: list[ProposalCandidateV1] = []
    for template_key, template in policy.proposal_templates.items():
        if template.target_binding == WORKSPACE_WINNER_BINDING:
            continue  # built by _build_workspace_candidates, never from the field ranking
        built.append(
            _build_candidate(
                template_key=template_key,
                template=template,
                field=field,
                field_tick_id=field.tick_id,
                attention=attention,
                pressures=pressures,
                policy=policy,
            )
        )

    # Flag-gated NON-DETERMINISTIC producers competing as first-class citizens.
    # Renamed from `reverie_candidates` 2026-07-31: reverie is no longer the only
    # one. Each carries its own `source` (`reverie_thought`, `cognitive_hop`) and
    # an operator_review gate -- they compete and are policy-gated exactly like
    # builder-native candidates, and none can auto-dispatch. The arena is the
    # arbitration mechanism for all of them; a producer never gets a private
    # scheduler (stream-of-consciousness hop-chain design, "not a merge, a sibling
    # producer under the same contract").
    for candidate in external_candidates or []:
        if candidate.visual_baseline is not None:
            warnings.append(f"visual_baseline_untrusted:{candidate.proposal_id}")
            continue
        built.append(candidate)

    baseline = []
    if baseline_eligibility is not None and "render_scene" not in policy.proposal_templates:
        warnings.append("visual_baseline_template_unavailable")
    if baseline_eligibility is not None and "render_scene" in policy.proposal_templates:
        for candidate in reversed(built):
            if (candidate.execution_intent.get("template") != "render_scene"
                    or candidate.proposal_kind != "express"
                    or candidate.target_id != policy.proposal_templates["render_scene"].target_id):
                continue
            denial = validate_eligibility(baseline_eligibility, now=generated_at,
                target_id=candidate.target_id, template="render_scene", proposal_kind=candidate.proposal_kind)
            if denial:
                warnings.append(denial)
                break
            baseline = [candidate.model_copy(update={"visual_baseline": baseline_eligibility,
                "expected_signal": None, "expected_direction": None,
                "reasons": [*candidate.reasons, "visual_baseline_due"]})]
            built = [other for other in built if not (
                other.execution_intent.get("template") == "render_scene"
                and other.proposal_kind == "express" and other.target_id == candidate.target_id)]
            break

    built.sort(key=lambda c: (-c.priority_score, c.proposal_id))

    # THE TICK-LEVEL GATE (2026-08-12).
    #
    # Work Outcome O1 asks that Orion's action budget rise and fall with real
    # internal pressure. It never did: the arena emitted a full slate every
    # tick, forever, because `min_priority` was a cut on an absolute pressure
    # whose floor sat above it. 100.00% of ticks cleared it.
    #
    # `action_warrant` separates the two questions the old design conflated.
    # WHETHER this tick's state warrants acting is a property of the tick, not
    # of any template, and is answered here on a scale whose rest point is
    # defined. WHICH candidates then win is still the per-template scoring
    # below, unchanged.
    #
    # Applied AFTER external_candidates are merged in, deliberately. Reverie
    # and cognitive-hop producers write `priority_score` straight from their
    # own salience, bypassing `proposal_priority()` entirely, so a
    # per-candidate threshold could never gate them consistently -- their
    # numbers are on a different scale. A tick-level gate is producer-agnostic
    # by construction: if the state does not warrant acting, nothing acts,
    # whoever proposed it.
    warrant = action_warrant(field)
    if warrant.score is None:
        gate = "no_live_dimensions"
    elif warrant.score < policy.thresholds.action_warrant_min:
        gate = "below_threshold"
    else:
        gate = "warranted"

    active: list[ProposalCandidateV1] = []
    suppressed: list[ProposalCandidateV1] = []
    if gate != "warranted":
        # Quiet, not broken. Everything is recorded as suppressed rather than
        # dropped, and `action_warrant_gate` on the frame says which of the two
        # reasons applied -- a frame with no candidates and no explanation is
        # indistinguishable from a dead pipeline.
        suppressed.extend(built)
        if gate == "no_live_dimensions":
            warnings.append(
                "action_warrant_unavailable:"
                + ",".join(f"{d}={r}" for d, r in sorted(warrant.excluded.items()))
            )
    else:
        for candidate in built:
            if candidate.priority_score < policy.thresholds.suppress_below:
                suppressed.append(candidate)
            elif candidate.priority_score < policy.thresholds.min_priority:
                suppressed.append(candidate)
            else:
                active.append(candidate)

    world = _build_workspace_candidates(
        policy=policy, workspace=workspace, field_tick_id=field.tick_id, now=generated_at, warnings=warnings)
    # World candidates ride OUTSIDE max_candidates: proposing one must never push an existing field
    # candidate out of the frame (the proposal flag is meant to be record-only on its own).
    active = (baseline + active)[: max(0, policy.limits.max_candidates)] + world
    suppressed = suppressed[: policy.limits.max_suppressed]

    overall_risk = _overall_risk(active)
    return ProposalFrameV1(
        frame_id=stable_proposal_frame_id(
            field_tick_id=field.tick_id,
            attention_frame_id=attention_frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_field_tick_id=field.tick_id,
        source_field_generated_at=field.generated_at,
        source_attention_frame_id=attention_frame_id,
        proposal_policy_id=policy.policy_id,
        overall_action_pressure=_overall_action_pressure(active),
        overall_risk=overall_risk,
        policy_required=_policy_required(
            candidates=active,
            overall_risk=overall_risk,
            policy=policy,
        ),
        candidates=active,
        suppressed_candidates=suppressed,
        dominant_motivations=_dominant_motivations(active),
        warnings=warnings,
        action_warrant=warrant.score,
        action_warrant_dimensions=list(warrant.contributing),
        action_warrant_gate=gate,
    )
