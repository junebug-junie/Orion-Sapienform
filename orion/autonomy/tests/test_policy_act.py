from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.autonomy.models import ActionOutcomeRefV1, SubstrateEpisodeIntentV1
from orion.autonomy.policy_act import (
    build_readonly_fetch_query,
    maybe_compose_autonomy_episode_after_fetch,
    maybe_execute_readonly_fetch_after_goal,
    maybe_execute_substrate_act_after_metabolism,
    resolve_episode_intent,
)
from orion.core.schemas.drives import DriveStateV1, GoalProposalV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def _gap_signal() -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="world coverage gap: hardware_compute_gpu had zero digest items",
        confidence=0.65,
    )


def _goal() -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": "goal-gap-gpu",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Reduce predictive uncertainty for hardware_compute_gpu.",
            "proposal_signature": "sig",
            "drive_origin": "predictive",
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _drive_state(predictive: float = 0.7) -> DriveStateV1:
    return DriveStateV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.state.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
            "pressures": {
                "coherence": 0.5,
                "continuity": 0.5,
                "capability": 0.5,
                "relational": 0.5,
                "predictive": predictive,
                "autonomy": 0.5,
            },
            "activations": {
                "coherence": False,
                "continuity": False,
                "capability": False,
                "relational": False,
                "predictive": True,
                "autonomy": False,
            },
        }
    )


def test_build_readonly_fetch_query_from_gap_section() -> None:
    query = build_readonly_fetch_query([_gap_signal()])
    assert "hardware compute gpu" in query


@pytest.mark.asyncio
async def test_policy_act_executes_fetch_when_allowed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))

    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    decision, outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=backend,
    )
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True
    assert outcome is not None
    assert outcome.success is True
    backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_policy_act_resolves_fetch_backend_when_omitted(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    monkeypatch.setattr(
        "orion.autonomy.policy_act.resolve_fetch_backend",
        lambda: backend,
    )
    decision, outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
    )
    assert decision.outcome == "allowed"
    assert outcome is not None
    assert outcome.success is True
    backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_policy_act_denied_when_pressure_low(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    decision, outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=_goal(),
        drive_state=_drive_state(predictive=0.2),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=AsyncMock(),
    )
    assert decision.outcome == "denied"
    assert decision.reason_code == "predictive_pressure_insufficient"
    assert outcome is None


@pytest.mark.asyncio
async def test_policy_act_dispatches_episode_journal_after_fetch(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    fetch_outcome = ActionOutcomeRefV1(
        action_id="fetch-test",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime.now(timezone.utc),
    )
    journal_dispatch = AsyncMock(return_value={"write": {"entry_id": "entry-1"}})
    decision, result = await maybe_compose_autonomy_episode_after_fetch(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_outcome=fetch_outcome,
        journal_dispatch=journal_dispatch,
    )
    assert decision.outcome == "allowed"
    assert result is not None
    journal_dispatch.assert_awaited_once()
    assert journal_dispatch.await_args.kwargs["narrative_seed"].startswith("fetch outcome:")


@pytest.mark.asyncio
async def test_policy_act_skips_episode_journal_when_fetch_missing(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    decision, result = await maybe_compose_autonomy_episode_after_fetch(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_outcome=None,
        journal_dispatch=AsyncMock(),
    )
    assert decision.reason_code == "fetch_outcome_missing"
    assert result is None


@pytest.mark.asyncio
async def test_policy_act_composes_episode_journal_on_fetch_failure(monkeypatch) -> None:
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    fetch_outcome = ActionOutcomeRefV1(
        action_id="fetch-test",
        kind="web.fetch.readonly",
        summary="fetch failed: timeout",
        success=False,
        surprise=1.0,
        observed_at=datetime.now(timezone.utc),
    )
    journal_dispatch = AsyncMock(return_value={"write": {"entry_id": "entry-1"}})
    decision, result = await maybe_compose_autonomy_episode_after_fetch(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_outcome=fetch_outcome,
        journal_dispatch=journal_dispatch,
    )
    assert decision.outcome == "allowed"
    assert result is not None
    assert "fetch failed" in journal_dispatch.await_args.kwargs["narrative_seed"]


def _intent() -> SubstrateEpisodeIntentV1:
    return SubstrateEpisodeIntentV1(
        goal_artifact_id="episode-wp-run-gap-gpu",
        drive_origin="predictive",
        spawned_correlation_id="wp-run-gap-gpu",
        subject="orion",
    )


class _FakeStore:
    def __init__(self, slot: dict | None = None) -> None:
        self._slot = slot or {}

    def load_goal_slot(self, subject: str, drive_origin: str) -> dict:
        return dict(self._slot)


def test_resolve_episode_intent_uses_predictive_slot() -> None:
    store = _FakeStore({"artifact_id": "goal-predictive-slot", "signature": "sig"})
    intent = resolve_episode_intent(store=store, subject="orion", run_id="wp-run-1")
    assert intent.goal_artifact_id == "goal-predictive-slot"
    assert intent.drive_origin == "predictive"


def test_resolve_episode_intent_synthetic_when_slot_empty() -> None:
    intent = resolve_episode_intent(store=_FakeStore(), subject="orion", run_id="wp-run-1")
    assert intent.goal_artifact_id == "episode-wp-run-1"
    assert intent.spawned_correlation_id == "wp-run-1"


@pytest.mark.asyncio
async def test_substrate_act_runs_when_goal_suppressed(monkeypatch, tmp_path) -> None:
    """Spec acceptance 4: proposal=None path still executes fetch."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        fetch_backend=backend,
    )
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.action_id == result.fetch_outcome_id
    assert result.fetch_outcome.kind == "web.fetch.readonly"
    backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_substrate_act_denied_without_gap_signal(monkeypatch) -> None:
    """Spec acceptance 5."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[],
        fetch_backend=AsyncMock(),
    )
    assert result.fetch_attempted is False


@pytest.mark.asyncio
async def test_substrate_act_preserves_fetch_when_journal_dispatch_fails(monkeypatch, tmp_path) -> None:
    """Regression: a journal-compose RPC failure (e.g. cortex-exec timeout) must NOT
    discard an already-successful fetch outcome, so the caller can still persist it."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    journal_dispatch = AsyncMock(side_effect=TimeoutError("cortex journal rpc timed out"))
    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        fetch_backend=backend,
        journal_dispatch=journal_dispatch,
        episode_journal_enabled=True,
    )
    # Fetch still succeeded and is returned despite the journal failure.
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.success is True
    # Journal was attempted but failed, so no journal entry recorded.
    assert result.journal_attempted is False
    journal_dispatch.assert_awaited_once()
