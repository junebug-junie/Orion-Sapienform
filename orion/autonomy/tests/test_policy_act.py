from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.autonomy.models import ActionOutcomeRefV1, SubstrateEpisodeIntentV1
from orion.autonomy.policy_act import (
    build_readonly_fetch_query,
    maybe_compose_autonomy_episode_after_fetch,
    maybe_execute_readonly_fetch_after_goal,
    maybe_execute_readonly_recall_after_goal,
    maybe_execute_substrate_act_after_metabolism,
    resolve_episode_intent,
)
from orion.core.contracts.recall import MemoryBundleV1, MemoryItemV1, RecallReplyV1
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
    from orion.autonomy.models import FetchedArticleRefV1

    fetch_outcome = ActionOutcomeRefV1(
        action_id="fetch-test",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime.now(timezone.utc),
        query="hardware compute gpu recent news coverage",
        articles=[
            FetchedArticleRefV1(url="https://example.com/a", title="GPU news", salience=0.67)
        ],
        salience=0.67,
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
    seed = journal_dispatch.await_args.kwargs["narrative_seed"]
    assert "hardware compute gpu" in seed
    assert "GPU news" in seed


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


def test_build_episode_narrative_seed_grounds_on_articles() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        query="hardware compute gpu recent news coverage",
        articles=[
            FetchedArticleRefV1(
                url="https://example.com/a",
                title="New GPU cluster",
                description="A big hardware compute launch.",
                salience=0.67,
            ),
            FetchedArticleRefV1(url="https://example.com/b", title="Side note", salience=0.0),
        ],
        salience=0.67,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)

    assert "hardware compute gpu" in seed          # the "why" (gap section)
    assert "New GPU cluster" in seed               # a real article title (the "what")
    assert "salience 0.67" in seed                 # salience marker
    assert "https://example.com/a" in seed         # real source url
    assert "Do not invent sources" in seed         # anti-confabulation ask
    assert "closes the gap" in seed                # satiation-assessment ask


def test_build_episode_narrative_seed_marks_unscored_when_no_gap_terms() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    # No section-derived gap terms and no query to fall back on -> there is nothing
    # to score against, so the marker is honestly "unscored".
    no_section = _gap_signal().model_copy(update={"focal_node_refs": ["node:not_a_section"]})
    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query=None,
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", salience=0.0)],
        salience=0.0,
    )
    seed = build_episode_narrative_seed(_goal(), [no_section], outcome)
    assert "unscored" in seed


def test_build_episode_narrative_seed_shows_zero_salience_when_scored_but_irrelevant() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    # Gap terms exist (section) but the article overlaps none -> honestly
    # "salience 0.00", NOT the misleading "unscored".
    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="hardware compute gpu recent news coverage",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="Cooking recipes", salience=0.0)],
        salience=0.0,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert "salience 0.00" in seed
    assert "unscored" not in seed


def test_build_episode_narrative_seed_failure_branch_unchanged() -> None:
    from orion.autonomy.policy_act import build_episode_narrative_seed

    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetch failed: timeout",
        success=False,
        surprise=1.0,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert seed == "fetch failed: fetch failed: timeout"


def test_build_episode_narrative_seed_truncates_long_description() -> None:
    from orion.autonomy.models import FetchedArticleRefV1
    from orion.autonomy.policy_act import build_episode_narrative_seed

    long_desc = "x" * 500
    outcome = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 1 article(s)",
        success=True,
        query="gpu news",
        articles=[FetchedArticleRefV1(url="https://example.com/a", title="A", description=long_desc, salience=0.5)],
        salience=0.5,
    )
    seed = build_episode_narrative_seed(_goal(), [_gap_signal()], outcome)
    assert "…" in seed
    assert "x" * 500 not in seed


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


def _fake_recall_reply(*, n: int = 1) -> RecallReplyV1:
    items = [
        MemoryItemV1(id=f"mem-{i}", source="journal", snippet=f"snippet {i}", score=0.8)
        for i in range(n)
    ]
    return RecallReplyV1(bundle=MemoryBundleV1(items=items, rendered="digest"))


def _fake_recall_bus(reply: RecallReplyV1 | None = None, *, exc: Exception | None = None) -> MagicMock:
    """Mirrors ``orion/cognition/tests/test_recall_prefetch.py``'s bus double."""
    bus = MagicMock()
    if exc is not None:

        async def _rpc(*_a: Any, **_k: Any) -> Any:
            raise exc

        bus.rpc_request = AsyncMock(side_effect=_rpc)
        return bus

    reply_payload = reply.model_dump(mode="json") if reply is not None else {"error": "fail"}

    class _Decoded:
        ok = True
        error = None

        class _Env:
            payload = reply_payload

        envelope = _Env()

    async def _rpc(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"data": b"x"}

    bus.codec.decode.return_value = _Decoded()
    bus.rpc_request = AsyncMock(side_effect=_rpc)
    return bus


@pytest.mark.asyncio
async def test_recall_capability_denied_without_gap_signal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    decision, outcome = await maybe_execute_readonly_recall_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[],
        spawned_correlation_id="wp-run-gap-gpu",
        bus=_fake_recall_bus(_fake_recall_reply()),
    )
    assert decision.outcome == "denied"
    assert decision.reason_code == "missing_signal_kinds"
    assert outcome is None


@pytest.mark.asyncio
async def test_recall_capability_finds_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    bus = _fake_recall_bus(_fake_recall_reply(n=2))
    decision, outcome = await maybe_execute_readonly_recall_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        bus=bus,
    )
    assert decision.outcome == "allowed"
    assert outcome is not None
    assert outcome.success is True
    assert outcome.kind == "recall.query.readonly"
    bus.rpc_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_recall_capability_empty_reply_degrades(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    bus = _fake_recall_bus(_fake_recall_reply(n=0))
    decision, outcome = await maybe_execute_readonly_recall_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        bus=bus,
    )
    assert decision.outcome == "allowed"
    assert outcome is not None
    assert outcome.success is False


@pytest.mark.asyncio
async def test_recall_capability_rpc_timeout_never_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    bus = _fake_recall_bus(exc=TimeoutError("rpc timeout"))
    decision, outcome = await maybe_execute_readonly_recall_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        bus=bus,
    )
    assert decision.outcome == "allowed"
    assert outcome is not None
    assert outcome.success is False
    assert "recall rpc failed" in outcome.summary


@pytest.mark.asyncio
async def test_recall_capability_no_bus_degrades_to_none(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    decision, outcome = await maybe_execute_readonly_recall_after_goal(
        goal=_goal(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        bus=None,
    )
    assert decision.outcome == "allowed"
    assert outcome is None


@pytest.mark.asyncio
async def test_substrate_act_recall_hit_skips_fetch_budget(monkeypatch, tmp_path) -> None:
    """Recall-first: when recall finds real content, the fetch capability is never
    invoked and its budget is not consumed that cycle."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fetch_backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    bus = _fake_recall_bus(_fake_recall_reply(n=1))
    budget: dict[str, int] = {}

    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=fetch_backend,
        budget_used=budget,
        recall_bus=bus,
    )

    fetch_backend.assert_not_awaited()
    bus.rpc_request.assert_awaited_once()
    assert budget.get("web.fetch.readonly", 0) == 0
    assert budget.get("recall.query.readonly", 0) == 1
    assert result.fetch_attempted is False
    assert result.fetch_outcome is None
    # A recall success must be visible on the result object the same way a
    # fetch success is (result.fetch_attempted/fetch_outcome) -- without this,
    # a successful recall-first check was invisible to the bus_worker.py
    # caller and never reached the durable action.outcome.emit.v1 bus path,
    # only the local append_action_outcome file-store fallback.
    assert result.recall_attempted is True
    assert result.recall_outcome is not None
    assert result.recall_outcome.success is True


@pytest.mark.asyncio
async def test_substrate_act_recall_miss_falls_through_to_fetch(monkeypatch, tmp_path) -> None:
    """Recall coming up empty must not block the tick: the fetch path still runs."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fetch_backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    bus = _fake_recall_bus(_fake_recall_reply(n=0))

    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=fetch_backend,
        recall_bus=bus,
    )

    fetch_backend.assert_awaited_once()
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.success is True
    # Recall was still attempted (an RPC really happened) even though it
    # found nothing -- "attempted" tracks the attempt, not the outcome,
    # mirroring fetch_attempted's own semantics.
    assert result.recall_attempted is True
    assert result.recall_outcome is not None
    assert result.recall_outcome.success is False


@pytest.mark.asyncio
async def test_substrate_act_recall_rpc_failure_falls_through_to_fetch(monkeypatch, tmp_path) -> None:
    """Recall RPC failure/timeout never raises and never blocks the tick."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fetch_backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    bus = _fake_recall_bus(exc=TimeoutError("rpc timeout"))

    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=fetch_backend,
        recall_bus=bus,
    )

    fetch_backend.assert_awaited_once()
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.success is True


@pytest.mark.asyncio
async def test_substrate_act_no_recall_bus_falls_through_to_fetch(monkeypatch, tmp_path) -> None:
    """Default behavior (no recall_bus wired) is unchanged: fetch runs as before."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fetch_backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})

    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        spawned_correlation_id="wp-run-gap-gpu",
        fetch_backend=fetch_backend,
    )

    fetch_backend.assert_awaited_once()
    assert result.fetch_attempted is True
