"""Gating tests for the first MUTATING dispatch route (2026-08-12).

Every proposal kind before `maintain` was read-only by construction. This one
can change the host, so the tests that matter are the ones proving each gate
independently blocks — not the one proving it works.

Three gates stand between a proposal and a real prune:

  1. `mode.allow_mutating_dispatch`  (ON since 2026-08-12, by explicit
     operator decision -- see test_mutating_dispatch_is_on_by_explicit_
     decision_and_stays_narrow for what keeps it narrow)
  2. the `maintenance_bounded` scope passing builder.py's allowlist
  3. the skill's own measured gate (disk >= 75% AND reclaimable >= 40GB),
     tested separately in services/orion-cortex-exec/tests/

Any one saying no means nothing happens.

A note on what this file got WRONG the first time, because it is the reason
the last section exists. Every gate test here hand-builds `PolicyDecisionV1`,
so none of them ever ran a `maintain` candidate through the real evaluator --
and the real evaluator could not build one at all (`maintenance_bounded` was
missing from two closed Literals), which would have permanently stalled the
policy runtime. The suite was green throughout. Tests that construct the
thing under test by hand do not prove the producer can construct it.
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
    load_execution_dispatch_policy,
)
from orion.policy.builder import build_policy_decision_frame
from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import load_substrate_policy
from orion.policy.rules import is_read_only_candidate
from orion.proposals.policy import load_proposal_policy
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


# --- shipped config: the gate that is open, and the ones that are not --------


def test_mutating_dispatch_is_on_by_explicit_decision_and_stays_narrow():
    """This shipped OFF and was flipped ON 2026-08-12 by Juniper's explicit
    decision, after a live preview on this host (`would_act`, disk 77.594%,
    142.5 GB reclaimable). The earlier version of this test asserted the OFF
    position; it is deliberately inverted rather than deleted, so the flip is
    a visible edit in history instead of a vanished guard.

    What this now guards is that the open gate stays a narrow one: exactly one
    route may mutate, and it is the build-cache prune.
    """
    raw = yaml.safe_load(
        (REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml").read_text()
    )
    assert raw["mode"]["allow_mutating_dispatch"] is True

    mutating = {
        name: r
        for name, r in raw["proposal_kind_to_cortex"].items()
        if r["allowed_scope"] == MAINTENANCE_SCOPE
    }
    assert set(mutating) == {"maintain"}, (
        "a second mutating route must be its own reviewed decision, not something "
        "that inherits an already-open flag"
    )
    assert mutating["maintain"]["cortex_verb"] == "skills.runtime.builder_prune.v1"


def test_dispatch_mode_falls_back_to_dry_run_when_nothing_sets_it():
    """Where the env-unset safe default ACTUALLY comes from.

    An earlier version of this test asserted
    `policy.mode.default_dispatch_mode == "dry_run"` and claimed in its
    docstring that this protects a deployment which never sets
    EXECUTION_DISPATCH_MODE. Review traced it: that value is dead code on this
    path. `_resolve_dispatch_mode` returns `override_dispatch_mode` whenever it
    is truthy, `worker.py` always passes `settings.execution_dispatch_mode`,
    and that setting is a non-optional Literal defaulting to `dry_run` -- so
    the override is never falsy and the policy value is never consulted. The
    test pinned a number with no runtime effect while its docstring described
    a mechanism that does not run.

    So this now asserts the real thing: the resolver's fallback, and the
    settings default that feeds it. If someone later makes
    `execution_dispatch_mode` optional, the first assertion starts doing real
    work instead of being decorative.
    """
    from orion.execution_dispatch.builder import _resolve_dispatch_mode

    shipped = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )
    assert _resolve_dispatch_mode(policy=shipped, override_dispatch_mode=None) == "dry_run"
    assert _resolve_dispatch_mode(policy=shipped, override_dispatch_mode="") == "dry_run"

    # Read the model's real default rather than grepping the source: a text
    # match here was brittle enough to fail on the actual (correct) code.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_eds_settings",
        REPO / "services" / "orion-execution-dispatch-runtime" / "app" / "settings.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    field = mod.Settings.model_fields["execution_dispatch_mode"]
    assert field.default == "dry_run", (
        "this default, not the policy's, is what actually protects an "
        "env-unset deployment"
    )

    compose = (
        REPO / "services" / "orion-execution-dispatch-runtime" / "docker-compose.yml"
    ).read_text()
    assert "${EXECUTION_DISPATCH_MODE:-dry_run}" in compose


def test_env_template_states_a_real_dispatch_mode_not_a_stale_one():
    """`.env_example` is the operator contract, not a placeholder.

    It drifted to `dry_run` while the live `.env` ran `dispatch_read_only`, so
    `sync_local_env_from_example.py --force` would have quietly stood dispatch
    down.

    Deliberately NOT `assert modes == ["dispatch_read_only"]`, which is what
    this first shipped as: that pins a literal on the emergency-stop path, so
    an operator standing dispatch down via the template would fail CI. A gate
    that fires when you use the emergency stop is a worse gate. What is
    actually checkable here is that the key is present exactly once and names
    a mode the runtime really accepts.
    """
    example = (
        REPO / "services" / "orion-execution-dispatch-runtime" / ".env_example"
    ).read_text()
    modes = [
        line.split("=", 1)[1].strip()
        for line in example.splitlines()
        if line.startswith("EXECUTION_DISPATCH_MODE=")
    ]
    assert len(modes) == 1, "exactly one uncommented value, or the contract is ambiguous"
    assert modes[0] in ("dry_run", "prepare_only", "dispatch_read_only")


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
    assert rule["default_decision"] == "approved_maintenance"
    assert rule["mutating"] is True


# --- the test whose absence let a blocker ship green -------------------------


def _template_faithful_candidate() -> ProposalCandidateV1:
    """A `maintain` candidate carrying the REAL template's values.

    The rest of this module's fixtures deliberately use off-template values to
    exercise gates. That is exactly how the blocker below got through: nothing
    ran a candidate that looks like what the builder actually emits.
    """
    tpl = load_proposal_policy(
        REPO / "config" / "proposals" / "proposal_policy.v1.yaml"
    ).proposal_templates["prune_build_cache"]
    return _candidate().model_copy(
        update={
            "risk_score": tpl.base_risk,
            "reversibility_score": tpl.reversibility,
            "required_policy_gate": tpl.required_policy_gate,
            "proposed_effect": tpl.proposed_effect,
        }
    )


def test_maintain_candidate_survives_policy_evaluation():
    """THE regression test for the 2026-08-12 blocker.

    `PolicyDecisionV1.allowed_scope`/`autonomy_tier` were closed Literals that
    omitted `maintenance_bounded`, while substrate_policy.v1.yaml's `maintain`
    rule set both to exactly that. So evaluating a real prune candidate raised
    ValidationError -- and because that happens during decision CONSTRUCTION,
    the policy runtime's only escape hatch (which covers a proposal frame that
    fails to LOAD) did not apply. The frame would stay the FIFO head forever
    and every downstream layer would go dark.

    Every gating test in this file passed the whole time, because they all
    hand-build PolicyDecisionV1 instead of going through the evaluator. Hence
    this test goes through the real evaluator and the real frame builder with
    the real configs -- no hand-built decision anywhere.
    """
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    candidate = _template_faithful_candidate()
    _, proposal_frame = _frames(candidate)

    decision = evaluate_proposal_candidate(
        candidate=candidate, proposal_frame=proposal_frame, policy=policy
    )
    assert decision.allowed_scope == MAINTENANCE_SCOPE
    assert decision.autonomy_tier == MAINTENANCE_SCOPE

    frame = build_policy_decision_frame(proposal_frame=proposal_frame, policy=policy)
    assert len(frame.decisions) == 1
    assert not frame.warnings, "an unevaluable candidate must not appear here"

    dispatch_policy = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )
    assert decision.decision in dispatch_policy.allowed_policy_decisions, (
        "the decision the evaluator really produces must be one the dispatch "
        "policy really accepts -- the two config files are edited separately"
    )


def test_a_mutating_action_is_never_approved_as_read_only():
    """The audit trail must not state something false as its grounds.

    `is_read_only_candidate` classified by `proposed_effect`, and the prune
    template's `preserve_stability` is in `read_only_effects` -- so a verb that
    deletes ~142 GB was approved through the `read_only_low_risk` branch and
    persisted a decision saying it reads nothing. That reason string was the
    stated justification for Orion's first destructive-capable action.
    """
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    candidate = _template_faithful_candidate()
    _, proposal_frame = _frames(candidate)

    assert is_read_only_candidate(candidate, policy) is False

    decision = evaluate_proposal_candidate(
        candidate=candidate, proposal_frame=proposal_frame, policy=policy
    )
    assert decision.decision == "approved_maintenance"
    assert "read_only_low_risk" not in decision.reasons
    assert decision.policy_gate != "read_only"


def test_read_only_kinds_are_unchanged_by_the_mutating_flag():
    """The `mutating` check must not reclassify anything else."""
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    for kind in ("observe", "inspect", "summarize"):
        assert is_read_only_candidate(_candidate(kind=kind), policy) is True


def test_one_unevaluable_candidate_cannot_wedge_the_whole_frame():
    """Fault isolation, the deterministic gate.

    The Literal fix cures this instance; this cures the class. A candidate the
    evaluator cannot handle must degrade to a surfaced warning, never to an
    exception that leaves the frame unbuildable -- because an unbuildable frame
    is a permanent FIFO stall, not a dropped tick.
    """
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    good = _template_faithful_candidate()
    bad = _candidate(kind="inspect").model_copy(update={"proposal_id": "p-bad"})
    _, proposal_frame = _frames(good)
    proposal_frame = proposal_frame.model_copy(update={"candidates": [bad, good]})

    import orion.policy.builder as pb

    real = pb.evaluate_proposal_candidate

    def _boom(*, candidate, proposal_frame, policy):
        if candidate.proposal_id == "p-bad":
            raise ValueError("simulated kind-rule/schema mismatch")
        return real(candidate=candidate, proposal_frame=proposal_frame, policy=policy)

    pb.evaluate_proposal_candidate = _boom
    try:
        frame = build_policy_decision_frame(proposal_frame=proposal_frame, policy=policy)
    finally:
        pb.evaluate_proposal_candidate = real

    assert len(frame.decisions) == 1, "the good candidate still gets a decision"
    assert frame.decisions[0].proposal_id == good.proposal_id
    assert any("candidate_unevaluable:p-bad" in w for w in frame.warnings), (
        "the failure must be visible on the frame, not only in logs"
    )


# --- the timeout ladder ------------------------------------------------------


def test_the_three_timeout_budgets_are_ordered_innermost_first():
    """A real prune is minutes of work and THREE nested budgets must clear it.

    The binding constraint was never the RPC timeout this arc first flagged.
    The prune's docker subprocess was inheriting SKILLS_MESH_OPS_TIMEOUT_SEC,
    which is 12 SECONDS live -- so a 142 GB prune would have been SIGKILLed
    ~12s in, mid-delete, on every attempt.

    They must stay strictly increasing outward, or a real overrun surfaces as
    an opaque outer timeout instead of being attributed to the docker command
    that actually ran long.
    """
    import importlib.util

    verb_yaml = yaml.safe_load(
        (
            REPO / "orion" / "cognition" / "verbs" / "skills.runtime.builder_prune.v1.yaml"
        ).read_text()
    )
    step_budget = float(verb_yaml["timeout_ms"]) / 1000.0

    spec = importlib.util.spec_from_file_location(
        "_cx_settings", REPO / "services" / "orion-cortex-exec" / "app" / "settings.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    subprocess_budget = float(mod.Settings.model_fields["builder_prune_timeout_sec"].default)
    mesh_ops_budget = float(mod.Settings.model_fields["skills_mesh_ops_timeout_sec"].default)

    route = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    ).proposal_kind_to_cortex["maintain"]
    rpc_budget = route.rpc_timeout_sec
    assert rpc_budget is not None, "the maintenance route needs its own RPC budget"

    assert subprocess_budget < step_budget < rpc_budget, (
        f"budgets must increase outward: subprocess={subprocess_budget} "
        f"step={step_budget} rpc={rpc_budget}"
    )
    assert subprocess_budget >= 600, "a real prune needs minutes, not seconds"
    assert subprocess_budget > mesh_ops_budget, (
        "the prune must NOT share the seconds-scale mesh-ops budget it used to inherit"
    )


def test_read_only_routes_do_not_get_a_long_rpc_budget():
    """The whole point of a PER-ROUTE timeout.

    This consumer is single-threaded, so a global ceiling big enough for the
    prune would let one hung inspect stall all dispatch for 12 minutes. Only
    the route that needs a long budget may carry one.
    """
    policy = load_execution_dispatch_policy(
        REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
    )
    with_budget = {
        name for name, r in policy.proposal_kind_to_cortex.items() if r.rpc_timeout_sec is not None
    }
    assert with_budget == {"maintain"}


def test_low_confidence_no_longer_bounces_a_maintain_candidate_to_a_human():
    """2026-08-12, Juniper's direction. confidence_score on this kind swings
    0.416-0.999 tick to tick on the same host state (25 real prune proposals
    measured over 3h), so a 0.50 review threshold flipped the same action
    between auto-approved and human-review at random.

    Goes through the REAL evaluator, not a hand-built decision -- the whole
    point of the preceding PR's blocker.
    """
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")
    candidate = _template_faithful_candidate().model_copy(update={"confidence_score": 0.42})
    _, proposal_frame = _frames(candidate)
    decision = evaluate_proposal_candidate(
        candidate=candidate, proposal_frame=proposal_frame, policy=policy
    )
    assert decision.decision == "approved_maintenance"
    assert "confidence_below_threshold" not in decision.reasons


def test_the_confidence_opt_out_is_narrow_and_does_not_leak_to_other_kinds():
    """Every other review trigger must still fire for `maintain`, and no
    other kind may inherit the opt-out."""
    policy = load_substrate_policy(REPO / "config" / "policy" / "substrate_policy.v1.yaml")

    def _decide(candidate):
        _, proposal_frame = _frames(candidate)
        return evaluate_proposal_candidate(
            candidate=candidate, proposal_frame=proposal_frame, policy=policy
        )

    for kind in ("inspect", "summarize", "observe"):
        # required_policy_gate is forced to read_only here because _candidate()
        # defaults it to operator_review, which short-circuits an EARLIER
        # branch -- the test would then pass without the confidence gate
        # existing at all.
        low_conf = _candidate(kind).model_copy(
            update={
                "confidence_score": 0.42,
                "risk_score": 0.05,
                "required_policy_gate": "read_only",
            }
        )
        d = _decide(low_conf)
        assert d.decision == "requires_operator_review", f"{kind} lost its confidence gate"
        assert "confidence_below_threshold" in d.reasons

    base = _template_faithful_candidate()
    risky = base.model_copy(update={"confidence_score": 0.42, "risk_score": 0.30})
    assert _decide(risky).decision == "requires_operator_review"
    irreversible = base.model_copy(
        update={"confidence_score": 0.42, "reversibility_score": 0.10}
    )
    assert _decide(irreversible).decision == "requires_operator_review"
