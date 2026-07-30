from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.autonomy.models import ActionOutcomeRefV1, SubstrateActResultV1
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


def _mock_worker_store(worker: ConceptWorker) -> MagicMock:
    """Common store mock: drive-pressure/goal-generation machinery was
    deleted 2026-07-30, so the store no longer has load_drive_state/
    save_drive_state/goal-cooldown methods to configure here -- only the
    surviving non-drive methods (load_goal_slot for policy_act.py's episode-
    intent fallback, and the episode-idempotency pair) matter to this path.
    """
    worker.store = MagicMock()
    worker.store.load_goal_slot.return_value = {}
    return worker.store


@pytest.mark.asyncio
async def test_metabolism_disabled_collects_no_curiosity_signals(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    _mock_worker_store(worker)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=False, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    # Metabolism disabled -> no curiosity signals collected -> substrate-act
    # gate (metabolism_curiosity_signals truthy) never fires.
    substrate_act_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_metabolism_enabled_collects_gap_curiosity_signal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    _mock_worker_store(worker)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    curiosity_signals = substrate_act_mock.await_args.kwargs["curiosity_signals"]
    assert any("hardware_compute_gpu" in ref for sig in curiosity_signals for ref in sig.focal_node_refs)


@pytest.mark.asyncio
async def test_policy_fetch_runs_on_world_pulse_gap(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    cfg.autonomy_episode_journal_enabled = True
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    _mock_worker_store(worker)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    call_kwargs = substrate_act_mock.await_args.kwargs
    assert call_kwargs["spawned_correlation_id"] == "wp-run-hook"
    assert call_kwargs["episode_journal_enabled"] is True
    # drive_state was previously passed as an honestly-empty DriveStateV1
    # stub (Wave 1, 2026-07-30) because policy_act.py still required the
    # parameter. Wave 2a (same day) dropped the parameter from
    # maybe_execute_substrate_act_after_metabolism entirely instead of
    # perpetuating the stub -- bus_worker.py no longer passes it at all.
    assert "drive_state" not in call_kwargs


@pytest.mark.asyncio
async def test_action_outcome_emitted_after_substrate_act(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    outcome = ActionOutcomeRefV1(
        action_id="fetch-wp-run-hook-abcd1234",
        kind="web.fetch.readonly",
        summary="fetched 2 article(s)",
        success=True,
        surprise=0.0,
        observed_at=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )
    substrate_act_mock = AsyncMock(
        return_value=SubstrateActResultV1(
            fetch_attempted=True,
            fetch_outcome_id=outcome.action_id,
            fetch_outcome=outcome,
        )
    )
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    _mock_worker_store(worker)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    worker._publish_action_outcome = AsyncMock(return_value=None)

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    worker._publish_action_outcome.assert_awaited_once()
    emitted = worker._publish_action_outcome.await_args.args[0]
    assert emitted.subject == "orion"
    assert emitted.action_id == outcome.action_id
    assert emitted.success is True
    assert emitted.kind == "web.fetch.readonly"


@pytest.mark.asyncio
async def test_episode_skipped_when_run_already_processed(monkeypatch) -> None:
    # Idempotency backstop: with the stream flag on, a run already marked processed must
    # NOT re-run the substrate act (no duplicate Firecrawl fetch / journal RPC).
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("WP_RUN_RESULT_STREAM_ENABLED", "true")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    cfg.autonomy_episode_journal_enabled = True
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    store = _mock_worker_store(worker)
    store.is_episode_run_processed.return_value = True
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_not_awaited()
    store.mark_episode_run_processed.assert_not_called()


@pytest.mark.asyncio
async def test_episode_marks_run_processed_when_stream_enabled(monkeypatch) -> None:
    # Positive path: first delivery runs the act, then marks the run so a redelivery is
    # deduped.
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("WP_RUN_RESULT_STREAM_ENABLED", "true")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    cfg.autonomy_episode_journal_enabled = True
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    store = _mock_worker_store(worker)
    store.is_episode_run_processed.return_value = False
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    store.mark_episode_run_processed.assert_called_once()
    assert store.mark_episode_run_processed.call_args.args[0] == "wp-run-hook"


@pytest.mark.asyncio
async def test_episode_not_marked_when_stream_disabled(monkeypatch) -> None:
    # Flag-off path must be byte-identical: no dedup read, no mark.
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("WP_RUN_RESULT_STREAM_ENABLED", "false")
    substrate_act_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, fetch_outcome=None, recall_outcome=None))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_act_mock,
    )
    cfg = ConceptSettings()
    cfg.autonomy_episode_journal_enabled = True
    worker = ConceptWorker(cfg, fetch_backend=AsyncMock())
    store = _mock_worker_store(worker)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_act_mock.assert_awaited_once()
    store.is_episode_run_processed.assert_not_called()
    store.mark_episode_run_processed.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_uses_journal_timeout_not_cortex_timeout(monkeypatch) -> None:
    # Guard against regressing bus_worker back to cfg.cortex_timeout_sec: the
    # journal compose (~16s) must get the generous dedicated budget, not the tight
    # generic cortex timeout, or the episode journal silently times out.
    monkeypatch.setenv("CORTEX_TIMEOUT_SEC", "12")
    monkeypatch.setenv("ORION_AUTONOMY_EPISODE_JOURNAL_TIMEOUT_SEC", "120")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)

    dispatch_mock = AsyncMock(return_value={"write": {"entry_id": "e1"}})
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.dispatch_autonomy_episode_journal",
        dispatch_mock,
    )

    await worker._dispatch_autonomy_episode_journal(
        _world_pulse_envelope(),
        goal_artifact_id="goal-x",
        spawned_correlation_id="wp-run-hook",
        narrative_seed="seed",
    )

    dispatch_mock.assert_awaited_once()
    passed = dispatch_mock.await_args.kwargs["timeout_sec"]
    assert passed == cfg.autonomy_episode_journal_timeout_sec == 120.0
    assert passed != cfg.cortex_timeout_sec
