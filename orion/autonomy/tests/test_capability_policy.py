from orion.autonomy.capability_policy import (
    CapabilityEvaluationContext,
    evaluate_capability,
)
from orion.autonomy.models import CapabilityPolicyRuleV1
from orion.schemas.field_goal import FieldGoalProvenanceV1


def _goal(**kwargs) -> FieldGoalProvenanceV1:
    base = dict(
        artifact_id="goal-test",
        subject="attention",
        model_layer="field_attention",
        # A world goal: the only kind fetch/recall accept (requires_world_goal).
        entity_id="world:chat",
        kind="memory.field_goals.proposed.v1",
        field_target_id="world:chat",
        target_kind="channel",
        salience_score=0.8,
        source_field_tick_id="tick-1",
        source_attention_frame_id="frame-1",
        priority=0.8,
        proposal_status="proposed",
        provenance={"intake_channel": "internal.attention_runtime"},
    )
    base.update(kwargs)
    return FieldGoalProvenanceV1.model_validate(base)


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


def test_retired_domain_surprise_policy_field_is_rejected() -> None:
    """Retired 2026-11-10 (prediction-error suite rebuild, retire step 1): no rule ever
    set required_domain_surprise_below. The rule model is extra="forbid", so a policy
    file that tries to bring it back fails loudly instead of being silently ignored."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _rule(required_domain_surprise_below=0.5)


def test_retired_domain_surprise_context_fields_are_gone() -> None:
    """The ambient bus_synaptic score no longer reaches the capability gate at all."""
    import dataclasses

    import pytest

    names = {f.name for f in dataclasses.fields(CapabilityEvaluationContext)}
    assert "domain_surprise_score" not in names
    assert "domain_surprise_source" not in names
    with pytest.raises(TypeError):
        CapabilityEvaluationContext(  # type: ignore[call-arg]
            curiosity_strength=0.0, signal_kinds=[], goal=None, domain_surprise_score=0.1
        )


def test_evaluate_capability_attaches_no_notes(monkeypatch) -> None:
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


# -- Content gate (Decision 1, prediction-error suite rebuild, 2026-11-08) ----


def _body_goal() -> FieldGoalProvenanceV1:
    return _goal(
        entity_id="node:substrate.bus_synaptic",
        field_target_id="node:substrate.bus_synaptic",
        target_kind="node",
    )


def _ctx(goal: FieldGoalProvenanceV1 | None) -> CapabilityEvaluationContext:
    return CapabilityEvaluationContext(
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=goal,
        budget_used={},
    )


def test_body_goal_no_longer_unlocks_fetch_or_recall(monkeypatch) -> None:
    """The bug: a bus prediction-error goal was a permission slip for web fetch
    and recall. A body goal names nothing Orion could fetch or recall about."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    for cap in ("web.fetch.readonly", "recall.query.readonly"):
        decision = evaluate_capability(cap, _ctx(_body_goal()))
        assert decision.outcome == "denied", cap
        assert decision.reason_code == "goal_not_about_world", cap


def test_world_goals_unlock_fetch_and_recall(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    for target, kind in (
        ("world:chat", "channel"),
        ("world:reading", "node"),
        ("node:substrate.perception", "node"),
    ):
        goal = _goal(entity_id=target, field_target_id=target, target_kind=kind)
        for cap in ("web.fetch.readonly", "recall.query.readonly"):
            decision = evaluate_capability(cap, _ctx(goal))
            assert decision.outcome == "allowed", (cap, target)


def test_episode_journal_is_not_content_gated(monkeypatch) -> None:
    """Spec step 2 gates the read-only capabilities on goal content only."""
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    decision = evaluate_capability("journal.compose.episode", _ctx(_body_goal()))
    assert decision.outcome == "allowed"


def test_shipped_policy_content_gates_exactly_fetch_and_recall() -> None:
    from orion.autonomy.capability_policy import load_capability_policy

    gated = {r.capability_id for r in load_capability_policy().rules if r.requires_world_goal}
    assert gated == {"web.fetch.readonly", "recall.query.readonly"}
