"""The retired maintenance dispatch route (2026-08-12 .. 2026-11-06), and what stays.

2026-08-12 opened Orion's first MUTATING dispatch route: `maintain` -> skills.runtime.builder_prune.v1
under the `maintenance_bounded` scope, behind `mode.allow_mutating_dispatch`. 2026-08-16 added
prune_dangling_images and prune_stopped_containers on the same scope. This file used to prove each of
those gates blocked on its own.

Ladder retirement PR 4 (2026-11-06) removed all three routes, the `maintain` fallback route, the
reserved maintenance slot, `allow_mutating_dispatch` and `approved_maintenance`. orion-hardware-watch's
disk rule asks for the image prune now (services/orion-hardware-watch/app/image_prune.py); build cache
and stopped containers are hand-run. What this file pins now:

- kill means kill: a route that still names the maintenance scope is blocked by scope, in any mode;
  no envelope carries a derived execute mode; the shipped config has no maintenance route at all;
- the policy runtime still evaluates a stray `maintain` candidate without wedging (its rule is inert
  until PR 6), and dispatch refuses that decision;
- the general dispatch/policy guarantees this file also held (dry-run default, env template, fault
  isolation, the confidence gate on read-only kinds).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import (
    CortexRouteTemplateV1,
    ExecutionDispatchPolicyV1,
    load_execution_dispatch_policy,
)
from orion.policy.builder import build_policy_decision_frame
from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import load_substrate_policy
from orion.policy.rules import is_read_only_candidate
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1

REPO = Path(__file__).resolve().parents[1]
DISPATCH_POLICY = REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
SUBSTRATE_POLICY = REPO / "config" / "policy" / "substrate_policy.v1.yaml"
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)
OLD_SCOPE = "maintenance_bounded"

MAINT_ROUTE = CortexRouteTemplateV1(
    dispatch_kind="maintain",
    cortex_verb="skills.runtime.builder_prune.v1",
    cortex_mode="brain",
    allowed_scope=OLD_SCOPE,
)
INSPECT_ROUTE = CortexRouteTemplateV1(
    dispatch_kind="inspect",
    cortex_verb="substrate.inspect",
    cortex_mode="brain",
    allowed_scope="inspect_only",
)


def _candidate(kind: str = "maintain", **update) -> ProposalCandidateV1:
    c = ProposalCandidateV1(
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
    return c.model_copy(update=update) if update else c


def _old_template_candidate() -> ProposalCandidateV1:
    """A `maintain` candidate with the retired prune_build_cache template's own values (base_risk
    0.12, reversibility 1.0, read_only gate, preserve_stability): what a stray one would look like."""
    return _candidate(risk_score=0.12, reversibility_score=1.0, required_policy_gate="read_only",
                      proposed_effect="preserve_stability")


def _decision(candidate: ProposalCandidateV1, decision: str = "approved_read_only") -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id="decision:1",
        proposal_id=candidate.proposal_id,
        decision=decision,
        policy_gate="operator_review",
        risk_score=candidate.risk_score,
        reversibility_score=candidate.reversibility_score,
        confidence_score=candidate.confidence_score,
    )


def _frames(candidate: ProposalCandidateV1, decision: str = "approved_read_only"):
    return (
        PolicyDecisionFrameV1(
            frame_id="policy.frame:1",
            generated_at=NOW,
            source_proposal_frame_id="proposal.frame:1",
            overall_risk=candidate.risk_score,
            decisions=[_decision(candidate, decision)],
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


def _stale_policy(**mode) -> ExecutionDispatchPolicyV1:
    """A policy file written before PR 4: it still names the flag (ON) and the maintain route."""
    return ExecutionDispatchPolicyV1.model_validate({
        "mode": {"default_dispatch_mode": "dispatch_read_only", "allow_dispatch_read_only": True,
                 "allow_mutating_dispatch": True, **mode},
        "allowed_policy_decisions": ["approved_read_only", "approved_maintenance"],
        "proposal_kind_to_cortex": {"maintain": MAINT_ROUTE.model_dump(), "inspect": INSPECT_ROUTE.model_dump()},
        "limits": {"max_dispatch_candidates": 5, "max_dispatches_per_tick": 5,
                   "reserved_slots_by_scope": {OLD_SCOPE: 1}},
    })


def _dispatch(candidate, policy, decision="approved_read_only"):
    policy_frame, proposal_frame = _frames(candidate, decision)
    return build_execution_dispatch_frame(policy_frame=policy_frame, proposal_frame=proposal_frame,
                                          field_tick_id="tick", policy=policy, now=NOW,
                                          override_dispatch_mode="dispatch_read_only")


# --- kill means kill ---------------------------------------------------------


def test_a_stale_maintenance_route_is_blocked_by_scope_even_with_the_old_flag_on():
    for decision in ("approved_read_only", "approved_maintenance"):
        frame = _dispatch(_candidate(), _stale_policy(), decision)
        assert frame.candidates == [] and frame.dispatched_candidates == []
        blocked = frame.blocked_candidates
        assert blocked and blocked[0].reasons == ["route_scope_not_read_only"]
        assert blocked[0].blocked_by == [OLD_SCOPE]
        assert "mutating_dispatch_disabled_by_policy" not in frame.warnings


def test_read_only_routes_still_dispatch_beside_it():
    frame = _dispatch(_candidate(kind="inspect"), _stale_policy())
    assert frame.candidates, "inspect must still dispatch"


def test_no_envelope_carries_a_derived_execute_mode():
    """Dispatch used to turn `dry_run=False` into skill_args.mode=execute for maintenance routes.
    That derivation is gone, so a skill reached through dispatch always runs in its default preview."""
    for dry_run in (False, True):
        env = build_cortex_request_envelope(candidate=_candidate(), decision=_decision(_candidate()),
                                            route=MAINT_ROUTE, field_tick_id="tick", dry_run=dry_run)
        assert "skill_args" not in env["context"]
        assert env["constraints"]["read_only"] is True
        assert env["constraints"]["no_file_writes"] is True


def test_the_shipped_config_has_no_maintenance_route_and_a_maintain_proposal_has_no_route():
    raw = yaml.safe_load(DISPATCH_POLICY.read_text())
    routes = list(raw["proposal_kind_to_cortex"].values()) + list(raw["template_to_cortex"].values())
    assert not [r for r in routes if r["allowed_scope"] == OLD_SCOPE]
    assert "allow_mutating_dispatch" not in raw["mode"]
    assert "approved_maintenance" not in raw["allowed_policy_decisions"]
    shipped = load_execution_dispatch_policy(DISPATCH_POLICY)
    frame = _dispatch(_old_template_candidate(), shipped, "approved_maintenance")
    assert frame.candidates == []
    assert frame.blocked_candidates, "a stray maintain proposal must be recorded as blocked, not dropped"


def test_the_policy_runtime_still_evaluates_a_stray_maintain_candidate_and_dispatch_refuses_it():
    """The `maintain` kind rule stays (inert) until PR 6, so a stray candidate cannot wedge the
    policy runtime (the 2026-08-12 Literal blocker). Its decision is one dispatch no longer accepts."""
    policy = load_substrate_policy(SUBSTRATE_POLICY)
    candidate = _old_template_candidate()
    _, proposal_frame = _frames(candidate)
    decision = evaluate_proposal_candidate(candidate=candidate, proposal_frame=proposal_frame, policy=policy)
    assert decision.decision == "approved_maintenance"
    assert "read_only_low_risk" not in decision.reasons
    frame = build_policy_decision_frame(proposal_frame=proposal_frame, policy=policy)
    assert len(frame.decisions) == 1 and not frame.warnings
    assert decision.decision not in load_execution_dispatch_policy(DISPATCH_POLICY).allowed_policy_decisions


# --- general dispatch / policy guarantees this file also held -----------------


def test_dispatch_mode_falls_back_to_dry_run_when_nothing_sets_it():
    """The env-unset safe default comes from the runtime's settings Literal (dry_run), not the policy
    file: `_resolve_dispatch_mode` returns the override whenever it is truthy."""
    from orion.execution_dispatch.builder import _resolve_dispatch_mode

    shipped = load_execution_dispatch_policy(DISPATCH_POLICY)
    assert _resolve_dispatch_mode(policy=shipped, override_dispatch_mode=None) == "dry_run"
    assert _resolve_dispatch_mode(policy=shipped, override_dispatch_mode="") == "dry_run"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_eds_settings", REPO / "services" / "orion-execution-dispatch-runtime" / "app" / "settings.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.Settings.model_fields["execution_dispatch_mode"].default == "dry_run"
    compose = (REPO / "services" / "orion-execution-dispatch-runtime" / "docker-compose.yml").read_text()
    assert "${EXECUTION_DISPATCH_MODE:-dry_run}" in compose


def test_env_template_states_a_real_dispatch_mode_not_a_stale_one():
    example = (REPO / "services" / "orion-execution-dispatch-runtime" / ".env_example").read_text()
    modes = [line.split("=", 1)[1].strip() for line in example.splitlines()
             if line.startswith("EXECUTION_DISPATCH_MODE=")]
    assert len(modes) == 1, "exactly one uncommented value, or the contract is ambiguous"
    assert modes[0] in ("dry_run", "prepare_only", "dispatch_read_only")


def test_read_only_kinds_are_classified_read_only():
    policy = load_substrate_policy(SUBSTRATE_POLICY)
    for kind in ("observe", "inspect", "summarize"):
        assert is_read_only_candidate(_candidate(kind=kind), policy) is True
    assert is_read_only_candidate(_old_template_candidate(), policy) is False


def test_one_unevaluable_candidate_cannot_wedge_the_whole_frame():
    """Fault isolation: a candidate the evaluator cannot handle degrades to a surfaced warning, never
    to an exception that leaves the frame unbuildable (a permanent FIFO stall)."""
    import orion.policy.builder as pb

    policy = load_substrate_policy(SUBSTRATE_POLICY)
    good = _candidate(kind="inspect", proposal_id="p-good", risk_score=0.05, required_policy_gate="read_only",
                      proposed_effect="increase_observability")
    bad = _candidate(kind="inspect", proposal_id="p-bad")
    _, proposal_frame = _frames(good)
    proposal_frame = proposal_frame.model_copy(update={"candidates": [bad, good]})
    real = pb.evaluate_proposal_candidate

    def _boom(*, candidate, **kw):
        if candidate.proposal_id == "p-bad":
            raise ValueError("simulated kind-rule/schema mismatch")
        return real(candidate=candidate, **kw)

    pb.evaluate_proposal_candidate = _boom
    try:
        frame = build_policy_decision_frame(proposal_frame=proposal_frame, policy=policy)
    finally:
        pb.evaluate_proposal_candidate = real
    assert [d.proposal_id for d in frame.decisions] == ["p-good"]
    assert any("candidate_unevaluable:p-bad" in w for w in frame.warnings)


def test_read_only_kinds_keep_their_confidence_gate():
    policy = load_substrate_policy(SUBSTRATE_POLICY)
    for kind in ("inspect", "summarize", "observe"):
        low_conf = _candidate(kind, confidence_score=0.42, risk_score=0.05, required_policy_gate="read_only")
        _, proposal_frame = _frames(low_conf)
        d = evaluate_proposal_candidate(candidate=low_conf, proposal_frame=proposal_frame, policy=policy)
        assert d.decision == "requires_operator_review", f"{kind} lost its confidence gate"
        assert "confidence_below_threshold" in d.reasons
