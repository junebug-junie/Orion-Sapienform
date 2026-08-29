"""Unit tests for the cabinet ambient spike bus listener."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app import worker as worker_module
from app.worker import BiometricsSubstrateWorker

from orion.schemas.telemetry.cabinet_ambient_spike import CabinetAmbientSpikeV1


def _make_worker(monkeypatch) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus = MagicMock()
    worker._catalog = MagicMock()
    worker._catalog.resolve.return_value = SimpleNamespace(node_id="athena")
    worker._store = MagicMock()
    worker._store.load_node_biometrics.return_value = None
    return worker


def _spike_payload() -> dict:
    return CabinetAmbientSpikeV1(
        spike_id="spike-1",
        node="athena",
        timestamp=datetime(2026, 8, 29, 4, 46, 54, tzinfo=timezone.utc),
        activity=0.34,
        rms=6500.0,
        peak=12000.0,
        activity_threshold=0.30,
        consecutive_ticks=2,
        source_service="orion-biometrics",
        source_node="athena",
    ).model_dump(mode="json")


@pytest.mark.asyncio
async def test_handle_cabinet_ambient_spike_publishes_grammar_and_saves_receipt(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(
        ok=True, envelope=SimpleNamespace(payload=_spike_payload())
    )
    worker._bus = bus

    publish_mock = AsyncMock()
    monkeypatch.setattr(worker_module, "publish_grammar_event", publish_mock)

    await worker._handle_cabinet_ambient_spike_message({"data": b"x"})

    assert publish_mock.await_count == 3
    worker._store.save_node_biometrics.assert_called_once()
    worker._store.save_receipt.assert_called_once()


@pytest.mark.asyncio
async def test_handle_cabinet_ambient_spike_fails_open_on_bad_decode(monkeypatch):
    worker = _make_worker(monkeypatch)
    bus = MagicMock()
    bus.codec.decode.return_value = MagicMock(ok=False, error="boom")
    worker._bus = bus

    publish_mock = AsyncMock()
    monkeypatch.setattr(worker_module, "publish_grammar_event", publish_mock)

    await worker._handle_cabinet_ambient_spike_message({"data": b"garbage"})

    publish_mock.assert_not_called()
    worker._store.save_receipt.assert_not_called()
