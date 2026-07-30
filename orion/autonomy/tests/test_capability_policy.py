from orion.autonomy.capability_policy import (
    CapabilityEvaluationContext,
    _domain_surprise_gate,
    evaluate_capability,
)
from orion.autonomy.models import CapabilityPolicyRuleV1
from orion.core.schemas.drives import GoalProposalV1


def _goal(**kwargs) -> GoalProposalV1:
    base = dict(
        artifact_id="goal-test",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        goal_statement="Reduce predictive uncertainty for hardware_compute_gpu.",
        proposal_signature="sig",
        drive_origin="predictive",
        proposal_status="proposed",
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    base.update(kwargs)
    return GoalProposalV1.model_validate(base)


def test_capability_policy_allows_readonly_when_goal_proposed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True


def test_capability_policy_denies_without_goal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=None,
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "missing_goal"


def test_capability_policy_denies_when_auto_readonly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "false")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "policy_auto_disabled"


def test_capability_policy_allows_episode_journal_at_proposed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("journal.compose.episode", ctx)
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True


def test_capability_policy_allows_recall_when_goal_proposed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("recall.query.readonly", ctx)
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True


def test_capability_policy_denies_recall_without_goal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=None,
        budget_used={},
    )
    decision = evaluate_capability("recall.query.readonly", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "missing_goal"


def test_capability_policy_denies_recall_when_auto_readonly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "false")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("recall.query.readonly", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "policy_auto_disabled"


def test_capability_policy_denies_episode_journal_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "false")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("journal.compose.episode", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "episode_journal_disabled"


def _rule(**kwargs) -> CapabilityPolicyRuleV1:
    base = dict(capability_id="test.capability", side_effect_class="readonly")
    base.update(kwargs)
    return CapabilityPolicyRuleV1.model_validate(base)


def test_domain_surprise_gate_passes_when_no_threshold_set() -> None:
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0, signal_kinds=[], goal=None
    )
    ok, reason = _domain_surprise_gate(ctx, _rule())
    assert ok is True
    assert reason == "no_surprise_gate"


def test_domain_surprise_gate_denies_honest_absence_when_threshold_set() -> None:
    """A rule that DOES require a threshold but got no real signal must deny, not
    silently pass -- see the gate's own docstring."""
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=None,
        domain_surprise_score=None,
    )
    ok, reason = _domain_surprise_gate(ctx, _rule(required_domain_surprise_below=0.5))
    assert ok is False
    assert reason == "domain_surprise_unavailable"


def test_domain_surprise_gate_denies_when_score_above_threshold() -> None:
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=None,
        domain_surprise_score=0.8,
    )
    ok, reason = _domain_surprise_gate(ctx, _rule(required_domain_surprise_below=0.5))
    assert ok is False
    assert reason == "domain_surprise_insufficient"


def test_domain_surprise_gate_satisfied_when_score_below_threshold() -> None:
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=None,
        domain_surprise_score=0.1,
    )
    ok, reason = _domain_surprise_gate(ctx, _rule(required_domain_surprise_below=0.5))
    assert ok is True
    assert reason == "domain_surprise_satisfied"


def test_evaluate_capability_surfaces_domain_surprise_in_notes_advisory_only(monkeypatch) -> None:
    """No shipped rule sets required_domain_surprise_below yet (advisory-first, per
    Acceptance check #5) -- the real value must still surface in notes for observability
    even though nothing gates on it."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
        domain_surprise_score=0.1675,
        domain_surprise_source="bus_synaptic",
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "allowed"
    assert any("0.1675" in note and "bus_synaptic" in note for note in decision.notes)


def test_evaluate_capability_no_notes_when_domain_surprise_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "allowed"
    assert decision.notes == []
