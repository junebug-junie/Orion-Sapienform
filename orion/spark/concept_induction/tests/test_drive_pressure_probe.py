"""Unit tests for the Phase 4 drive-pressure measurement probe (2026-07-12).

Measurement-only: logs `DriveEngine`'s pressure vector right after every
`save_drive_state` call so it can be compared offline (by grepping logs and
correlating on subject + nearby timestamp) against `AutonomyStateV2`'s
independently-computed pressures, logged from
`services/orion-cortex-exec/app/chat_stance.py`'s `_run_autonomy_reducer`.
No schema change, no behavior change -- see
docs/superpowers/plans/2026-07-12-self-state-mesh-substrate-redesign.md Phase 4.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.models import OrganClass, OrionSignalV1
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings

LOGGER_NAME = "orion.spark.concept.worker"


def _signal_env(kind: str, dims: dict, *, channel_kind: str = "signal.biometrics.biometrics_state") -> BaseEnvelope:
    now = datetime.now(timezone.utc)
    sig = OrionSignalV1(
        signal_id=str(uuid4()), organ_id="biometrics", organ_class=OrganClass.exogenous,
        signal_kind=kind, dimensions=dims, source_event_id="src-1",
        observed_at=now, emitted_at=now,
    )
    return BaseEnvelope(
        id=uuid4(), kind=channel_kind, correlation_id=uuid4(), created_at=now,
        source=ServiceRef(name="orion-signal-gateway", version="0.1.0", node="athena"),
        payload=sig.model_dump(mode="json"),
    )


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
                "run_id": "wp-run-probe",
                "date": "2026-07-06",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "status": "completed",
                "dry_run": False,
            },
            "digest": {
                "run_id": "wp-run-probe",
                "date": "2026-07-06",
                "generated_at": now.isoformat(),
                "title": "t",
                "executive_summary": "e",
                "sections": {},
                "items": [],
                "orion_analysis_layer": "deterministic",
                "coverage_status": "sparse",
                "section_rollups": [],
                "created_at": now.isoformat(),
            },
        },
    )


def _worker() -> ConceptWorker:
    worker = ConceptWorker(ConceptSettings())
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.store.save_drive_state = MagicMock()
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    return worker


def _probe_records(caplog) -> list:
    return [r for r in caplog.records if r.message.startswith("drive_engine_pressure_probe")]


def test_log_drive_pressure_probe_logs_subject_and_pressures(caplog) -> None:
    worker = _worker()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        worker._log_drive_pressure_probe("orion", {"coherence": 0.50001, "continuity": 0.2})

    records = _probe_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "orion" in msg
    assert "coherence" in msg
    assert "0.5" in msg  # rounded to 4dp, confirms the value round-trips through the log


def test_log_drive_pressure_probe_never_raises_on_logging_failure(caplog) -> None:
    worker = _worker()
    import orion.spark.concept_induction.bus_worker as bus_worker_mod

    class _ExplodingLogger:
        def info(self, *_a, **_kw):
            raise RuntimeError("boom")

        def warning(self, *args, **kwargs):
            logging.getLogger(LOGGER_NAME).warning(*args, **kwargs)

    original_logger = bus_worker_mod.logger
    bus_worker_mod.logger = _ExplodingLogger()
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            # Must not raise.
            worker._log_drive_pressure_probe("orion", {"coherence": 0.5})
    finally:
        bus_worker_mod.logger = original_logger

    assert any("drive_engine_pressure_probe_failed" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_signal_drive_tick_call_site_logs_probe(caplog) -> None:
    """Call site 1 (~line 717, _handle_signal_drive_tick via handle_envelope)."""
    worker = _worker()

    # Warm the deviation baseline with steady hrv, then a real drop -- same
    # warm-up pattern as test_signal_drive_consumer.py's homeostatic tests.
    for _ in range(30):
        await worker.handle_envelope(
            _signal_env("biometrics_state", {"hrv_level": 0.8, "confidence": 0.9}),
            "orion:signals:biometrics",
        )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await worker.handle_envelope(
            _signal_env("biometrics_state", {"hrv_level": 0.4, "confidence": 0.9}),
            "orion:signals:biometrics",
        )

    worker.store.save_drive_state.assert_called()
    records = _probe_records(caplog)
    assert len(records) == 1
    assert "orion" in records[0].getMessage()


@pytest.mark.asyncio
async def test_handle_envelope_call_site_logs_probe(caplog) -> None:
    """Call site 2 (~line 847, handle_envelope's non-homeostatic drive-update rail)."""
    worker = _worker()
    worker.drive_engine.update = MagicMock(
        return_value=({"predictive": 0.6001, "coherence": 0.5}, {"predictive": True})
    )
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    worker.store.save_drive_state.assert_called_once()
    records = _probe_records(caplog)
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "orion" in msg
    assert "predictive" in msg
    assert "0.6" in msg


@pytest.mark.asyncio
async def test_handle_envelope_call_site_survives_probe_failure(caplog) -> None:
    """A probe logging failure at call site 2 must not break save_drive_state or
    propagate out of handle_envelope -- unlike call site 1 (which sits inside the
    broader homeostatic try/except), call site 2 has no outer guard, so the
    helper's own internal try/except is the only thing protecting this path."""
    worker = _worker()
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.6}, {"predictive": True}))
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    import orion.spark.concept_induction.bus_worker as bus_worker_mod

    # Raise ONLY on the probe's own info call, not on every other logger.info
    # call downstream in handle_envelope -- otherwise this test would prove
    # nothing about the probe specifically.
    original_info = bus_worker_mod.logger.info

    def _raise_on_probe(msg, *args, **kwargs):
        if isinstance(msg, str) and msg.startswith("drive_engine_pressure_probe"):
            raise RuntimeError("boom")
        return original_info(msg, *args, **kwargs)

    bus_worker_mod.logger.info = _raise_on_probe
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            # Must not raise.
            await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")
    finally:
        bus_worker_mod.logger.info = original_info

    worker.store.save_drive_state.assert_called_once()
    assert any("drive_engine_pressure_probe_failed" in r.message for r in caplog.records)
