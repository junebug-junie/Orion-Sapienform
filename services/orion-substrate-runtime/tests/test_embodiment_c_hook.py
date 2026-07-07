"""Unit tests for the Orion embodiment C producer hook.

Verifies the substrate worker maps its cached ``DriveStateV1`` to exactly one
involuntary ``EmbodimentIntentV1`` per dynamics tick when enabled, publishes
nothing when disabled, and fails open when no drive state has been observed.
Constructs the worker via ``__new__`` (mirrors ``test_worker_dynamics_tick``).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker

from orion.core.schemas.drives import ArtifactProvenance, DriveStateV1


def _make_worker(monkeypatch, *, c_tick_enabled: bool) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("EMBODIMENT_C_TICK_ENABLED", "true" if c_tick_enabled else "false")
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._bus = MagicMock()
    worker._latest_drive_state = None
    return worker


def _drive(pressures: dict[str, float]) -> DriveStateV1:
    return DriveStateV1(
        subject="orion",
        model_layer="drive",
        entity_id="orion",
        kind="memory.drives.state.v1",
        provenance=ArtifactProvenance(intake_channel="test"),
        pressures=pressures,
    )


def test_flag_on_publishes_one_involuntary_intent(monkeypatch):
    worker = _make_worker(monkeypatch, c_tick_enabled=True)
    worker._latest_drive_state = _drive({"social": 0.6, "predictive": 0.1})

    pub = AsyncMock()
    with patch("orion.core.bus.resilience.publish_with_reconnect", pub):
        asyncio.run(worker._emit_c_intent())

    assert pub.await_count == 1
    _, _, env = pub.await_args.args[:3]
    assert env.kind == "embodiment.intent.v1"
    assert env.payload["source"] == "involuntary"
    assert env.payload["reason"].strip()


def test_flag_off_publishes_nothing(monkeypatch):
    worker = _make_worker(monkeypatch, c_tick_enabled=False)
    worker._latest_drive_state = _drive({"social": 0.6})

    pub = AsyncMock()
    with patch("orion.core.bus.resilience.publish_with_reconnect", pub):
        asyncio.run(worker._emit_c_intent())

    pub.assert_not_awaited()


def test_no_drive_state_publishes_nothing(monkeypatch):
    worker = _make_worker(monkeypatch, c_tick_enabled=True)
    worker._latest_drive_state = None

    pub = AsyncMock()
    with patch("orion.core.bus.resilience.publish_with_reconnect", pub):
        asyncio.run(worker._emit_c_intent())

    pub.assert_not_awaited()
