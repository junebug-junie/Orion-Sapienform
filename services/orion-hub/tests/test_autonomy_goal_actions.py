from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


@pytest.fixture
def hub_client(monkeypatch):
    _ensure_hub_scripts_import_path()
    monkeypatch.setenv("SUBSTRATE_MUTATION_OPERATOR_TOKEN", "secret-token")
    import importlib

    import scripts.main as hub_main

    importlib.reload(hub_main)
    with TestClient(hub_main.app) as client:
        yield client


def test_promote_goal_requires_operator_token(hub_client) -> None:
    resp = hub_client.post("/api/autonomy/goals/goal-abc/promote", json={})
    assert resp.status_code in {403, 503}


def test_promote_goal_rejects_when_execution_disabled(hub_client) -> None:
    resp = hub_client.post(
        "/api/autonomy/goals/goal-abc/promote",
        json={},
        headers={"X-Orion-Operator-Token": "secret-token"},
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "autonomy_goal_execution_disabled"


def test_promote_goal_success_when_enabled(hub_client, monkeypatch) -> None:
    from orion.autonomy.goal_actions import GoalActionResult

    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")

    def _fake_promote(**kwargs):
        return GoalActionResult(
            artifact_id=kwargs["artifact_id"],
            action="promote",
            proposal_status="planned",
            reasoning_outcome="escalated_hitl",
            reasoning_claim_id=f"goal-reasoning-{kwargs['artifact_id']}",
            hitl_satisfied=True,
        )

    monkeypatch.setattr("scripts.api_routes.promote_goal", _fake_promote)

    resp = hub_client.post(
        "/api/autonomy/goals/goal-abc/promote",
        json={"operator": "operator-1"},
        headers={"X-Orion-Operator-Token": "secret-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["artifact_id"] == "goal-abc"
    assert body["proposal_status"] == "planned"
    assert body["reasoning"]["hitl_satisfied"] is True


def test_dismiss_goal_success_when_enabled(hub_client, monkeypatch) -> None:
    from orion.autonomy.goal_actions import GoalActionResult

    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(
        "scripts.api_routes.dismiss_goal",
        lambda **kwargs: GoalActionResult(
            artifact_id=kwargs["artifact_id"],
            action="dismiss",
            proposal_status="archived",
        ),
    )

    resp = hub_client.post(
        "/api/autonomy/goals/goal-abc/dismiss",
        json={},
        headers={"X-Orion-Operator-Token": "secret-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["proposal_status"] == "archived"


def test_complete_goal_success_when_enabled(hub_client, monkeypatch) -> None:
    from orion.autonomy.goal_actions import GoalActionResult

    completed_at = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc).isoformat()
    monkeypatch.setenv("AUTONOMY_GOAL_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(
        "scripts.api_routes.complete_goal",
        lambda **kwargs: GoalActionResult(
            artifact_id=kwargs["artifact_id"],
            action="complete",
            proposal_status="completed",
            completed_at=completed_at,
        ),
    )

    resp = hub_client.post(
        "/api/autonomy/goals/goal-abc/complete",
        json={},
        headers={"X-Orion-Operator-Token": "secret-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["proposal_status"] == "completed"
    assert body["completed_at"] == completed_at


def test_operator_goal_reasoning_promotion_applies_hitl_gate() -> None:
    from orion.autonomy.goal_actions import apply_operator_goal_reasoning_promotion
    from orion.autonomy.models import AutonomyGoalHeadlineV1
    from orion.reasoning.repository import InMemoryReasoningRepository

    goal = AutonomyGoalHeadlineV1(
        artifact_id="goal-abc",
        goal_statement="Stabilize coherence without auto execution.",
        drive_origin="coherence",
        priority=0.72,
        proposal_signature="deadbeef01",
        proposal_status="active",
    )
    repo = InMemoryReasoningRepository()
    result = apply_operator_goal_reasoning_promotion(goal=goal, subject="orion", operator="operator-1", reasoning_repo=repo)
    assert result.items[0].outcome in {"promoted", "escalated_hitl"}
    claim = repo.get_by_id("goal-reasoning-goal-abc")
    assert claim is not None
    assert claim.status == "canonical"
