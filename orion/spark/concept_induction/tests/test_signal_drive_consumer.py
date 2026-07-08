"""Task 6 live: homeostatic signal channels ride a drive-only rail.

Asserts the worker (a) subscribes to the SPECIFIC organ/failure channels but not
the scene_state wildcard, (b) classifies homeostatic sources, (c) updates + publishes
drives from a real biometric drop WITHOUT triggering concept induction, and
(d) does nothing for a steady/unmapped signal.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.signals.models import OrganClass, OrionSignalV1
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings


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


def _worker() -> ConceptWorker:
    worker = ConceptWorker(ConceptSettings())
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.store.save_drive_state = MagicMock()
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    return worker


def test_pubsub_patterns_include_specific_channels_not_wildcard() -> None:
    worker = _worker()
    patterns = worker._pubsub_patterns()
    assert "orion:signals:biometrics" in patterns
    assert "orion:signals:spark" in patterns
    assert "orion:system:error" in patterns
    # The flood wildcard is NEVER subscribed.
    assert "orion:signals:*" not in patterns
    assert "orion:signals:vision" not in patterns


def test_homeostatic_source_classification() -> None:
    worker = _worker()
    assert worker._homeostatic_source("orion:signals:biometrics") == "signal"
    assert worker._homeostatic_source("orion:system:error") == "failure"
    assert worker._homeostatic_source("orion:chat:history:log") is None


@pytest.mark.asyncio
async def test_real_biometric_drop_updates_drives_no_induction() -> None:
    worker = _worker()
    worker.inducer = MagicMock()
    worker.inducer.run = AsyncMock()  # must NOT be called
    worker._extract_text = MagicMock(side_effect=AssertionError("concept path must not run"))

    # Warm the deviation baseline with steady hrv, then a real drop.
    for _ in range(30):
        await worker.handle_envelope(
            _signal_env("biometrics_state", {"hrv_level": 0.8, "confidence": 0.9}),
            "orion:signals:biometrics",
        )
    worker._publish_drive_state.reset_mock()
    worker._publish_tension_event.reset_mock()
    await worker.handle_envelope(
        _signal_env("biometrics_state", {"hrv_level": 0.4, "confidence": 0.9}),
        "orion:signals:biometrics",
    )

    assert worker._publish_tension_event.await_count >= 1
    assert worker._publish_drive_state.await_count == 1
    worker.store.save_drive_state.assert_called()
    worker.inducer.run.assert_not_awaited()  # concept induction never triggered


@pytest.mark.asyncio
async def test_steady_signal_is_noop() -> None:
    worker = _worker()
    for _ in range(20):
        await worker.handle_envelope(
            _signal_env("biometrics_state", {"hrv_level": 0.8, "confidence": 0.9}),
            "orion:signals:biometrics",
        )
    # Steady input past warm-up mints no tension and publishes no drive state.
    assert worker._publish_drive_state.await_count == 0


@pytest.mark.asyncio
async def test_failure_channel_mints_tension() -> None:
    worker = _worker()
    err_env = BaseEnvelope(
        id=uuid4(), kind="system.error.v1", correlation_id=uuid4(),
        created_at=datetime.now(timezone.utc),
        source=ServiceRef(name="orion-x", version="0.1.0", node="athena"),
        payload={"error": "boom"},
    )
    await worker.handle_envelope(err_env, "orion:system:error")
    assert worker._publish_tension_event.await_count >= 1
    assert worker._publish_drive_state.await_count == 1


@pytest.mark.asyncio
async def test_never_raises_when_drive_update_breaks() -> None:
    """A bad prior state / store fault in the drive-update section must degrade to
    a no-op, not tear down the bus loop (which re-raises)."""
    worker = _worker()
    # tz-naive updated_at is the concrete raise path the reviewer flagged.
    worker.store.load_drive_state.return_value = {
        "pressures": {}, "activations": {}, "updated_at": "2026-07-08T00:00:00",  # naive
    }
    worker.store.save_drive_state = MagicMock(side_effect=RuntimeError("store down"))
    # Must not raise.
    await worker.handle_envelope(
        BaseEnvelope(
            id=uuid4(), kind="system.error.v1", correlation_id=uuid4(),
            created_at=datetime.now(timezone.utc),
            source=ServiceRef(name="orion-x", version="0.1.0", node="athena"),
            payload={"error": "boom"},
        ),
        "orion:system:error",
    )


@pytest.mark.asyncio
async def test_disabled_flag_falls_through_to_concept_path(monkeypatch) -> None:
    monkeypatch.setenv("ORION_HOMEOSTATIC_DRIVES_ENABLED", "false")
    worker = ConceptWorker(ConceptSettings())
    assert worker._homeostatic_source is not None
    # With the flag off, the signal channel is not added to patterns.
    assert "orion:signals:biometrics" not in worker._pubsub_patterns()
