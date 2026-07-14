from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
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
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
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
        predictive_pressure=0.7,
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
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
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
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
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
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
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
        predictive_pressure=0.7,
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
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
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
        predictive_pressure=0.7,
        curiosity_strength=0.0,
        signal_kinds=[],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("journal.compose.episode", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "episode_journal_disabled"
