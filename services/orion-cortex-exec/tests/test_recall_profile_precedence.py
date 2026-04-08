from __future__ import annotations

from app.recall_utils import resolve_profile
from orion.schemas.cortex.schemas import ExecutionStep


def _step(recall_profile: str | None = None) -> ExecutionStep:
    return ExecutionStep(
        verb_name="chat_general",
        step_name="recall",
        description="",
        order=0,
        services=["RecallService"],
        recall_profile=recall_profile,
        requires_memory=True,
    )


def test_precedence_explicit_override_wins_when_flagged() -> None:
    profile, source = resolve_profile(
        {"profile": "reflect.v1", "profile_explicit": True},
        verb_profile="chat.general.v1",
        step=_step("deep.graph.v1"),
        is_recall_step=True,
    )
    assert profile == "reflect.v1"
    assert source == "explicit"


def test_precedence_step_used_when_no_explicit_override() -> None:
    profile, source = resolve_profile(
        {"profile": "reflect.v1"},
        verb_profile="chat.general.v1",
        step=_step("deep.graph.v1"),
        is_recall_step=True,
    )
    assert profile == "deep.graph.v1"
    assert source == "step"


def test_precedence_verb_wins_over_inherited_profile() -> None:
    profile, source = resolve_profile(
        {"profile": "reflect.v1"},
        verb_profile="chat.general.v1",
        is_recall_step=False,
    )
    assert profile == "chat.general.v1"
    assert source == "verb"


def test_precedence_mode_used_when_no_verb_or_profile() -> None:
    profile, source = resolve_profile(
        {"mode": "graph"},
        verb_profile=None,
        is_recall_step=False,
    )
    assert profile == "graphtri.v1"
    assert source == "mode"


def test_precedence_fallback_when_nothing_else() -> None:
    profile, source = resolve_profile({}, verb_profile=None, is_recall_step=False)
    assert profile == "reflect.v1"
    assert source == "fallback"


def test_agent_runtime_ignores_inherited_reflect_profile_by_default() -> None:
    profile, source = resolve_profile(
        {"profile": "reflect.v1", "mode": "hybrid"},
        verb_profile=None,
        is_recall_step=False,
        runtime_mode="agent",
    )
    assert profile == "chat.general.v1"
    assert source == "runtime_mode"
