"""Gating tests for the 2nd/3rd mutating routes and the tension-driven
read-only route (2026-08-16, docs/superpowers/specs/2026-08-16-tension-
driven-mutating-dispatch-design.md).

Same three-gate shape as test_maintenance_dispatch_gating.py's prune_build_
cache tests: mode.allow_mutating_dispatch, the maintenance_bounded scope
check, and each skill's own measured gate (tested separately in
services/orion-cortex-exec/tests/). This file proves the first two for the
two NEW mutating templates, and proves the read-only route is unaffected by
the mutating flag entirely -- the same three-gate shape this repo already
uses, applied to a second and third action rather than re-litigated.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.execution_dispatch.builder import build_execution_dispatch_frame, resolve_cortex_route
from orion.execution_dispatch.policy import (
    MAINTENANCE_SCOPE,
    CortexRouteTemplateV1,
    ExecutionDispatchPolicyV1,
    load_execution_dispatch_policy,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

_MUTATING_TEMPLATES = ("prune_dangling_images", "prune_stopped_containers")


def _candidate(template: str, *, target_id: str = "host:docker") -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id=f"proposal:{template}:tick:none",
        proposal_kind="maintain" if template in _MUTATING_TEMPLATES else "inspect",
        title="t",
        description="d",
        target_id=target_id,
        target_kind="system",
        priority_score=0.9,
        urgency_score=0.9,
        confidence_score=0.9,
        risk_score=0.12,
        reversibility_score=1.0,
        proposed_effect="preserve_stability",
        required_policy_gate="read_only",
        execution_intent={"mode": "descriptive_only", "template": template, "policy_gate": "read_only"},
    )


def _decision(candidate: ProposalCandidateV1) -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id="decision:1",
        proposal_id=candidate.proposal_id,
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=candidate.risk_score,
        reversibility_score=candidate.reversibility_score,
        confidence_score=candidate.confidence_score,
    )


def _frames(candidate: ProposalCandidateV1):
    decision = _decision(candidate)
    return (
        PolicyDecisionFrameV1(
            frame_id="policy.frame:1",
            generated_at=NOW,
            source_proposal_frame_id="proposal.frame:1",
            overall_risk=candidate.risk_score,
            decisions=[decision],
        ),
        ProposalFrameV1(
            frame_id="proposal.frame:1",
            generated_at=NOW,
            source_field_tick_id="tick",
            source_field_generated_at=NOW,
            source_attention_frame_id="none",
            overall_action_pressure=0.9,
            overall_risk=candidate.risk_score,
            candidates=[candidate],
        ),
    )


def _policy(*, allow_mutating: bool) -> ExecutionDispatchPolicyV1:
    return ExecutionDispatchPolicyV1.model_validate(
        {
            "mode": {
                "default_dispatch_mode": "dispatch_read_only",
                "allow_dispatch_read_only": True,
                "allow_mutating_dispatch": allow_mutating,
            },
            "allowed_policy_decisions": ["approved_read_only"],
            "proposal_kind_to_cortex": {
                "maintain": {
                    "dispatch_kind": "maintain",
                    "cortex_verb": "skills.runtime.builder_prune.v1",
                    "allowed_scope": MAINTENANCE_SCOPE,
                },
                "inspect": {
                    "dispatch_kind": "inspect",
                    "cortex_verb": "substrate.inspect",
                    "allowed_scope": "inspect_only",
                },
            },
            "template_to_cortex": {
                "prune_dangling_images": {
                    "dispatch_kind": "maintain",
                    "cortex_verb": "skills.runtime.image_prune.v1",
                    "allowed_scope": MAINTENANCE_SCOPE,
                },
                "prune_stopped_containers": {
                    "dispatch_kind": "maintain",
                    "cortex_verb": "skills.runtime.docker_prune_stopped_containers.v1",
                    "allowed_scope": MAINTENANCE_SCOPE,
                },
                "observe_tension_via_camera": {
                    "dispatch_kind": "inspect",
                    "cortex_verb": "skills.perception.look_at_camera.v1",
                    "allowed_scope": "inspect_only",
                },
            },
        }
    )


@pytest.mark.parametrize("template", _MUTATING_TEMPLATES)
def test_mutating_route_is_blocked_when_the_flag_is_off(template):
    candidate = _candidate(template)
    policy_frame, proposal_frame = _frames(candidate)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal_frame,
        field_tick_id="tick",
        policy=_policy(allow_mutating=False),
        now=NOW,
    )
    assert frame.dispatched_candidates == []
    assert frame.candidates == []
    assert frame.blocked_candidates and "route_scope_not_read_only" in frame.blocked_candidates[0].reasons


@pytest.mark.parametrize("template", _MUTATING_TEMPLATES)
def test_mutating_route_passes_only_when_the_flag_is_on(template):
    candidate = _candidate(template)
    policy_frame, proposal_frame = _frames(candidate)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal_frame,
        field_tick_id="tick",
        policy=_policy(allow_mutating=True),
        now=NOW,
    )
    assert frame.candidates, f"{template} must dispatch with the flag on"


def test_camera_route_is_unaffected_by_the_mutating_flag():
    """inspect_only is always allowed -- the read-only route must dispatch
    identically in both flag positions, same as watch_reliability etc."""
    candidate = _candidate("observe_tension_via_camera", target_id="host:camera_perception")
    policy_frame, proposal_frame = _frames(candidate)
    for allow in (False, True):
        frame = build_execution_dispatch_frame(
            policy_frame=policy_frame,
            proposal_frame=proposal_frame,
            field_tick_id="tick",
            policy=_policy(allow_mutating=allow),
            now=NOW,
        )
        assert frame.candidates, f"camera route must dispatch with allow_mutating={allow}"


@pytest.mark.parametrize(
    "template,verb",
    [
        ("prune_dangling_images", "skills.runtime.image_prune.v1"),
        ("prune_stopped_containers", "skills.runtime.docker_prune_stopped_containers.v1"),
        ("observe_tension_via_camera", "skills.perception.look_at_camera.v1"),
    ],
)
def test_each_template_resolves_to_its_own_verb_not_the_kind_fallback(template, verb):
    policy = _policy(allow_mutating=True)
    route = resolve_cortex_route(_candidate(template), policy)
    assert route.cortex_verb == verb


# --- shipped config -----------------------------------------------------


def test_live_config_second_and_third_mutating_routes_stay_maintenance_bounded():
    shipped = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )
    for template in _MUTATING_TEMPLATES:
        assert shipped.template_to_cortex[template].allowed_scope == MAINTENANCE_SCOPE

    assert shipped.template_to_cortex["observe_tension_via_camera"].allowed_scope == "inspect_only"
