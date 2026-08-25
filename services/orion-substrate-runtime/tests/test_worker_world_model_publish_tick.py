"""Unit tests for the world-model publish tick -- the first real producer for
``orion:exec:request:WorldModelService`` (services/orion-world-model, PR
#1775/#1861).

Default-off, async (not a to_thread'd sync tick -- the only real work is an
in-memory feature assembly plus a bus publish), fail-open.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from orion.schemas.world_model import WorldModelTaskRequestPayload

from app.worker import BiometricsSubstrateWorker


def _make_worker(monkeypatch, *, enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv(
        "SUBSTRATE_WORLD_MODEL_PUBLISH_TICK_ENABLED", "true" if enabled else "false"
    )
    import app.settings as settings_mod

    settings_mod._settings = None

    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._process_started_at = datetime.now(timezone.utc)
    worker._last_execution_prediction_error = None
    worker._last_chat_prediction_error = None
    worker._last_route_prediction_error = None
    worker._last_bus_synaptic_prediction_error = None
    worker._last_perception_embedding_vector = None
    worker._bus = None
    return worker


def test_world_model_publish_tick_disabled_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=False)
    worker._bus = MagicMock()
    worker._bus.publish = AsyncMock()
    asyncio.run(worker._world_model_publish_tick())
    worker._bus.publish.assert_not_called()


def test_world_model_publish_tick_no_bus_is_noop(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus = None
    asyncio.run(worker._world_model_publish_tick())  # must not raise


def test_world_model_publish_tick_publishes_valid_payload_to_configured_channel(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus = MagicMock()
    worker._bus.publish = AsyncMock()
    worker._last_execution_prediction_error = 0.3
    worker._last_chat_prediction_error = 0.0
    worker._last_route_prediction_error = None
    worker._last_bus_synaptic_prediction_error = 0.1
    worker._last_perception_embedding_vector = [0.5] * worker._settings.world_model_dim_vision_embedding

    asyncio.run(worker._world_model_publish_tick())

    worker._bus.publish.assert_called_once()
    channel, envelope = worker._bus.publish.call_args.args
    assert channel == worker._settings.world_model_request_channel == "orion:exec:request:WorldModelService"
    assert envelope.kind == "world_model.task.request"
    assert envelope.reply_to is None

    payload = WorldModelTaskRequestPayload.model_validate(envelope.payload)
    assert payload.task_type == "predict_next_state"
    assert len(payload.trajectory) == 1  # single-step, no rolling window
    step = payload.trajectory[0]
    assert step.execution_context.vector[0] == 0.3  # execution
    assert step.execution_context.vector[1] == 0.0  # chat (real 0.0)
    assert step.execution_context.vector[2] == 0.0  # route (None -> zero slot)
    assert step.execution_context.vector[3] == 0.1  # bus_synaptic
    assert step.vision_embedding.vector == [0.5] * worker._settings.world_model_dim_vision_embedding
    assert payload.meta is not None
    assert payload.meta["vision_source"] == "real"
    assert set(payload.meta["zero_filled_groups"]) == {"biometrics", "affect", "memory_pointers"}
    assert payload.meta["real_execution_context_domains"] == ["execution", "chat", "bus_synaptic"]


def test_world_model_publish_tick_zero_fills_everything_when_no_state_cached(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus = MagicMock()
    worker._bus.publish = AsyncMock()

    asyncio.run(worker._world_model_publish_tick())

    worker._bus.publish.assert_called_once()
    _, envelope = worker._bus.publish.call_args.args
    payload = WorldModelTaskRequestPayload.model_validate(envelope.payload)
    step = payload.trajectory[0]
    assert step.execution_context.vector == [0.0] * worker._settings.world_model_dim_execution_context
    assert step.vision_embedding.vector == [0.0] * worker._settings.world_model_dim_vision_embedding
    assert payload.meta["vision_source"] == "unavailable"
    assert payload.meta["real_execution_context_domains"] == []


def test_world_model_publish_tick_fails_open_on_bus_publish_error(monkeypatch):
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus = MagicMock()
    worker._bus.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    asyncio.run(worker._world_model_publish_tick())  # must not raise


def test_world_model_publish_tick_vision_dim_mismatch_zero_fills_and_does_not_raise(monkeypatch):
    """The defensive path from a real embedding of the wrong length flowing
    all the way through the actual worker tick, not just the pure helper."""
    worker = _make_worker(monkeypatch, enabled=True)
    worker._bus = MagicMock()
    worker._bus.publish = AsyncMock()
    worker._last_perception_embedding_vector = [0.1] * 999  # deliberately wrong length

    asyncio.run(worker._world_model_publish_tick())  # must not raise

    worker._bus.publish.assert_called_once()
    _, envelope = worker._bus.publish.call_args.args
    payload = WorldModelTaskRequestPayload.model_validate(envelope.payload)
    step = payload.trajectory[0]
    assert step.vision_embedding.vector == [0.0] * worker._settings.world_model_dim_vision_embedding
    assert payload.meta["vision_source"] == "dim_mismatch"
    assert payload.meta["vision_dim_observed"] == 999
    assert payload.meta["vision_dim_configured"] == worker._settings.world_model_dim_vision_embedding
    assert "vision_embedding" in payload.meta["zero_filled_groups"]
