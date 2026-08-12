"""Gating tests for the first MUTATING dispatch route (2026-08-12).

Every proposal kind before `maintain` was read-only by construction. This one
can change the host, so the tests that matter are the ones proving each gate
independently blocks — not the one proving it works.

Three gates stand between a proposal and a real prune:

  1. `mode.allow_mutating_dispatch`  (default FALSE)
  2. the `maintenance_bounded` scope passing builder.py's allowlist
  3. the skill's own measured gate (disk >= 75% AND reclaimable >= 40GB),
     tested separately in services/orion-cortex-exec/tests/

Any one saying no means nothing happens.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import (
    MAINTENANCE_SCOPE,
    CortexRouteTemplateV1,
    ExecutionDispatchPolicyV1,
)
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)

MAINT_ROUTE = CortexRouteTemplateV1(
    dispatch_kind="maintain",
    cortex_verb="skills.runtime.builder_prune.v1",
    cortex_mode="brain",
    allowed_scope=MAINTENANCE_SCOPE,
)
INSPECT_ROUTE = CortexRouteTemplateV1(
    dispatch_kind="inspect",
    cortex_verb="substrate.inspect",
    cortex_mode="brain",
    allowed_scope="inspect_only",
)


def _candidate(kind: str = "maintain") -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id="proposal:prune_build_cache:tick:none",
        proposal_kind=kind,
        title="t",
        description="d",
        target_id="host:docker_build_cache",
        target_kind="system",
        priority_score=0.9,
        urgency_score=0.9,
        confidence_score=0.9,
        risk_score=0.35,
        reversibility_score=1.0,
        proposed_effect="preserve_stability",
        required_policy_gate="operator_review",
    )


def _decision(candidate: ProposalCandidateV1) -> PolicyDecisionV1:
    # Fields read from the model rather than guessed -- an earlier version of
    # this helper invented `rationale` and omitted `policy_gate`/`risk_score`,
    # and every gate test failed on validation instead of on behaviour.
    return PolicyDecisionV1(
        decision_id="decision:1",
        proposal_id=candidate.proposal_id,
        decision="approved_read_only",
        policy_gate="operator_review",
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
            overall_risk=0.35,
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
                "maintain": MAINT_ROUTE.model_dump(),
                "inspect": INSPECT_ROUTE.model_dump(),
            },
        }
    )


# --- GATE 1: the policy flag, which used to be dead --------------------------


def test_maintenance_route_is_blocked_when_the_flag_is_off():
    """Default state. Before 2026-08-12 `allow_mutating_dispatch` appended a
    warning and did nothing else -- a switch that reported success while
    changing nothing. It now actually gates."""
    candidate = _candidate()
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
    blocked = frame.blocked_candidates
    assert blocked and "route_scope_not_read_only" in blocked[0].reasons
    assert "mutating_dispatch_disabled_by_policy" in frame.warnings


def test_maintenance_route_passes_only_when_the_flag_is_on():
    candidate = _candidate()
    policy_frame, proposal_frame = _frames(candidate)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal_frame,
        field_tick_id="tick",
        policy=_policy(allow_mutating=True),
        now=NOW,
    )
    assert frame.candidates, "flag on must let the maintenance route through"
    assert "mutating_dispatch_disabled_by_policy" not in frame.warnings


def test_read_only_routes_are_unaffected_by_the_flag():
    """The change must not widen anything else. An inspect route behaves
    identically in both flag positions."""
    candidate = _candidate(kind="inspect")
    policy_frame, proposal_frame = _frames(candidate)
    for allow in (False, True):
        frame = build_execution_dispatch_frame(
            policy_frame=policy_frame,
            proposal_frame=proposal_frame,
            field_tick_id="tick",
            policy=_policy(allow_mutating=allow),
            now=NOW,
        )
        assert frame.candidates, f"inspect must dispatch with allow_mutating={allow}"


# --- GATE 2: the envelope must tell the truth about what it is ---------------


def test_maintenance_envelope_is_not_marked_read_only():
    """`read_only` was hardcoded True. Telling the executor a mutating dispatch
    is read-only would be a lie it might act on."""
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(_candidate()),
        route=MAINT_ROUTE,
        field_tick_id="tick",
        dry_run=False,
    )
    assert env["constraints"]["read_only"] is False
    assert env["verb"] == "skills.runtime.builder_prune.v1"


def test_read_only_envelopes_keep_every_constraint():
    env = build_cortex_request_envelope(
        candidate=_candidate(kind="inspect"),
        decision=_decision(_candidate(kind="inspect")),
        route=INSPECT_ROUTE,
        field_tick_id="tick",
        dry_run=False,
    )
    assert env["constraints"]["read_only"] is True
    assert "skill_args" not in env["context"]


# --- GATE 3 handoff: dry_run must reach the skill as its own run mode --------


@pytest.mark.parametrize("dry_run,expected", [(True, "preview"), (False, "execute")])
def test_dispatch_dry_run_becomes_the_skills_run_mode(dry_run, expected):
    """The skill defaults to preview on its own. This carries the dispatch
    runtime's mode through explicitly so a dry-run dispatch cannot become a
    real prune by omission."""
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(_candidate()),
        route=MAINT_ROUTE,
        field_tick_id="tick",
        dry_run=dry_run,
    )
    assert env["context"]["skill_args"]["mode"] == expected


# --- shipped config must be the safe position --------------------------------


def test_shipped_policy_has_mutating_dispatch_off():
    raw = yaml.safe_load(
        (REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml").read_text()
    )
    assert raw["mode"]["allow_mutating_dispatch"] is False, (
        "this must ship OFF -- turning it on is Juniper's decision, not a merge's"
    )
    assert raw["proposal_kind_to_cortex"]["maintain"]["allowed_scope"] == MAINTENANCE_SCOPE


def test_maintenance_is_the_only_non_read_only_scope():
    raw = yaml.safe_load(
        (REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml").read_text()
    )
    scopes = {r["allowed_scope"] for r in raw["proposal_kind_to_cortex"].values()}
    assert scopes - {"inspect_only", "summarize_only"} == {MAINTENANCE_SCOPE}


def test_prune_template_risk_stays_dispatchable():
    """`risk` is a number the substrate policy ACTS on, not a label.

    The first draft used base_risk 0.35 to signal "mutating and serious". That
    exceeds substrate_policy.v1.yaml's `require_review_above_risk: 0.20`, so the
    evaluator returns `requires_operator_review` -- a BLOCKED decision. The
    number chosen to convey seriousness would have made the action permanently
    undispatchable, silently. This test is why that cannot come back.
    """
    from orion.policy.policy import load_substrate_policy
    from orion.proposals.policy import load_proposal_policy

    proposals = load_proposal_policy(REPO / "config" / "proposals" / "proposal_policy.v1.yaml")
    tpl = proposals.proposal_templates["prune_build_cache"]
    assert tpl.kind == "maintain"
    assert tpl.reversibility == 1.0

    substrate = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    assert tpl.base_risk < substrate.thresholds.require_review_above_risk
    assert tpl.base_risk <= substrate.thresholds.approve_read_only_max_risk


def test_maintain_kind_rule_exists():
    """Without a rule, `kind_rule()` returns None and the evaluator defaults to
    `deferred`, so the route is unreachable. Found by a test, not by reading."""
    raw = yaml.safe_load((REPO / "config" / "policy" / "substrate_policy.v1.yaml").read_text())
    rule = raw["proposal_kind_rules"]["maintain"]
    assert rule["allowed_scope"] == MAINTENANCE_SCOPE
    assert rule["default_decision"] == "approved_read_only"
