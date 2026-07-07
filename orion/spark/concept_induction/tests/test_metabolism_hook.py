from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings


def _world_pulse_envelope() -> BaseEnvelope:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    return BaseEnvelope(
        id=uuid4(),
        kind="world.pulse.run.result.v1",
        correlation_id=uuid4(),
        created_at=now,
        source=ServiceRef(name="orion-world-pulse", version="0.1.0", node="athena"),
        payload={
            "run": {
                "run_id": "wp-run-hook",
                "date": "2026-07-06",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "status": "completed",
                "dry_run": False,
            },
            "digest": {
                "run_id": "wp-run-hook",
                "date": "2026-07-06",
                "generated_at": now.isoformat(),
                "title": "t",
                "executive_summary": "e",
                "sections": {},
                "items": [],
                "orion_analysis_layer": "deterministic",
                "coverage_status": "sparse",
                "section_rollups": [
                    {
                        "section": "hardware_compute_gpu",
                        "status": "missing",
                        "article_count": 0,
                        "digest_item_count": 0,
                        "confidence": 0.35,
                    }
                ],
                "created_at": now.isoformat(),
            },
        },
    )


@pytest.mark.asyncio
async def test_metabolism_disabled_does_not_add_tensions(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({}, {}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    call_kwargs = worker.drive_engine.update.call_args.kwargs
    assert call_kwargs["tensions"] == []


@pytest.mark.asyncio
async def test_metabolism_enabled_merges_gap_tensions(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.2}, {"predictive": False}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    tensions = worker.drive_engine.update.call_args.kwargs["tensions"]
    assert any(getattr(t, "kind", "") == "substrate.world_coverage_gap" for t in tensions)


@pytest.mark.asyncio
async def test_world_pulse_run_id_lineage_when_metabolism_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.2}, {"predictive": False}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    assert worker.goal_engine.propose.call_args.kwargs["spawned_correlation_id"] == "wp-run-hook"


@pytest.mark.asyncio
async def test_metabolism_enriches_goal_window_summary(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.2}, {"predictive": False}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    summary = worker.goal_engine.propose.call_args.kwargs["window_summary"]
    assert summary is not None
    assert "hardware_compute_gpu" in summary


@pytest.mark.asyncio
async def test_policy_fetch_runs_after_goal_publish(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    cfg.autonomy_episode_journal_enabled = True
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {
        "pressures": {"predictive": 0.7, "coherence": 0.5, "continuity": 0.5, "capability": 0.5, "relational": 0.5, "autonomy": 0.5},
        "activations": {"predictive": True},
    }
    worker.drive_engine.update = MagicMock(
        return_value=(
            {"predictive": 0.7, "coherence": 0.5, "continuity": 0.5, "capability": 0.5, "relational": 0.5, "autonomy": 0.5},
            {"predictive": True},
        )
    )
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)

    proposal = MagicMock()
    proposal.artifact_id = "goal-gap-gpu"
    proposal.subject = "orion"
    proposal.drive_origin = "predictive"
    proposal.proposal_status = "proposed"
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=proposal, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    call_kwargs = substrate_act_mock.await_args.kwargs
    assert call_kwargs["spawned_correlation_id"] == "wp-run-hook"
    assert call_kwargs["episode_journal_enabled"] is True


@pytest.mark.asyncio
async def test_substrate_act_runs_when_goal_suppressed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.2}, {"predictive": False}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker.goal_engine.propose = MagicMock(
        return_value=MagicMock(proposal=None, suppressed_signature="sig-cooldown")
    )

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    call_kwargs = substrate_act_mock.await_args.kwargs
    assert call_kwargs["spawned_correlation_id"] == "wp-run-hook"
