from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SUBSTRATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE_ROOT))

from app.worker import BiometricsSubstrateWorker


def _worker(monkeypatch, *, enabled: bool = True) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_ATTENTION_BROADCAST_ENABLED", "true")
    monkeypatch.setenv("SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED", "false")
    monkeypatch.setenv(
        "SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED",
        "true" if enabled else "false",
    )
    monkeypatch.setenv("SUBSTRATE_SYSTEM_ONE_BASE_URL", "http://kev:8009")

    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._substrate_graph_store = None
    worker._pending_system_one_grammar_events = []
    return worker


def _graph_node(node_id: str, label: str, pressure: float) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=node_id,
        label=label,
        metadata={"dynamic_pressure": pressure},
        signals=SimpleNamespace(confidence=0.8),
    )


def test_system_one_failure_does_not_break_attention_broadcast(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={
            "node:hot": _graph_node(
                "node:hot", "unresolved contradiction", 0.9
            )
        }
    )
    worker._system_one_appraisal_tick = MagicMock(
        side_effect=RuntimeError("kev unavailable")
    )

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()  # must not raise

    worker._store.save_attention_broadcast.assert_called_once()
    worker._system_one_appraisal_tick.assert_called_once()


def test_disabled_system_one_is_not_called(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=False)
    fake_store = MagicMock()
    fake_store.snapshot.return_value = SimpleNamespace(
        nodes={
            "node:hot": _graph_node(
                "node:hot", "unresolved contradiction", 0.9
            )
        }
    )
    worker._system_one_appraisal_tick = MagicMock()

    with patch(
        "orion.substrate.graphdb_store.build_substrate_store_from_env",
        return_value=fake_store,
    ):
        worker._attention_broadcast_tick()

    worker._store.save_attention_broadcast.assert_called_once()
    worker._system_one_appraisal_tick.assert_not_called()


def test_system_one_tick_omits_stale_field_frame(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    stale = MagicMock()
    stale.generated_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    worker._store.get_latest_field_attention_frame.return_value = stale
    worker._store.save_system_one_appraisal = MagicMock()

    frame = MagicMock()
    frame.model_id = "kev-latest"
    frame.frame_id = "frame-1"
    frame.latency_ms = 5
    frame.answers = {}

    with patch(
        "orion.substrate.system_one_appraisal.run_system_one_appraisal",
        return_value=frame,
    ) as run, patch(
        "orion.substrate.system_one_appraisal.build_system_one_grammar_events",
        return_value=[],
    ):
        worker._system_one_appraisal_tick(broadcast=MagicMock())

    assert run.call_args.kwargs["field_frame"] is None
    worker._store.save_system_one_appraisal.assert_called_once()


def test_system_one_tick_does_not_persist_when_inference_fails(monkeypatch) -> None:
    worker = _worker(monkeypatch, enabled=True)
    worker._store.get_latest_field_attention_frame.return_value = None

    with patch(
        "orion.substrate.system_one_appraisal.run_system_one_appraisal",
        side_effect=ValueError("incomplete provider response"),
    ):
        try:
            worker._system_one_appraisal_tick(broadcast=MagicMock())
        except ValueError:
            pass
        else:
            raise AssertionError("expected inference failure")

    worker._store.save_system_one_appraisal.assert_not_called()


# --- 2026-09-29: typed frame never reached the bus ------------------------
# Live container logged "1 validation error for BaseEnvelope correlation_id
# Input should be a valid UUID" on every attempt: the publisher passed the
# non-UUID frame_id (``system-one-appraisal-<24 hex>``) straight in as the
# envelope correlation_id, and the exception was swallowed by the publish
# guard, so orion:system_one:appraisal never carried a single frame.


def _real_frame():
    """Build a frame through the real producer so frame_id has its live shape."""
    from orion.schemas.attention_frame import (
        AttentionBroadcastProjectionV1,
        AttentionFrameV1,
        OpenLoopV1,
    )
    from orion.substrate.system_one_appraisal import run_system_one_appraisal

    now = datetime(2026, 9, 29, 5, 0, tzinfo=timezone.utc)
    broadcast = AttentionBroadcastProjectionV1(
        generated_at=now,
        frame=AttentionFrameV1(
            generated_at=now,
            open_loops=[
                OpenLoopV1(
                    id="loop-1",
                    target_type="concept",
                    description="unresolved graph contradiction",
                    salience=0.8,
                    combined_salience=0.8,
                    confidence=0.7,
                    source_refs=["node:substrate.a"],
                )
            ],
        ),
        selected_action_type="reflect",
        selected_open_loop_id="loop-1",
        selected_description="unresolved graph contradiction",
    )
    answers = {
        key: {
            "type": "score",
            "score": 1.2,
            "confidence": 0.7,
            "legend": {"0": "low", "1": "middle", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
        }
        for key in (
            "reverie_fit",
            "curiosity_pull",
            "deliberation_need",
            "attention_interrupt",
        )
    }

    class _Resp:
        headers = {"x-typesafe-request-id": "kev-request-1"}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "model": "kev-latest",
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "latency_ms": 42,
            }

    return run_system_one_appraisal(
        broadcast=broadcast,
        field_frame=None,
        base_url="http://kev:8009",
        model="kev-latest",
        post=lambda url, *, json, headers, timeout: _Resp(),
        now=now,
    )


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel, msg) -> None:
        self.published.append((channel, msg))

    async def reconnect(self) -> None:  # pragma: no cover - not exercised
        return None


def test_pending_system_one_frame_publishes_valid_envelope(monkeypatch) -> None:
    import asyncio
    from uuid import UUID

    from orion.core.bus.bus_schemas import BaseEnvelope
    from orion.schemas.system_one_appraisal import (
        SYSTEM_ONE_APPRAISAL_CHANNEL,
        SYSTEM_ONE_APPRAISAL_KIND,
        SystemOneAppraisalFrameV1,
    )

    worker = _worker(monkeypatch, enabled=True)
    bus = _FakeBus()
    worker._bus = bus
    frame = _real_frame()
    # Live shape: not a UUID. This is the precondition of the bug.
    assert frame.frame_id.startswith("system-one-appraisal-")
    with pytest.raises(ValueError):
        UUID(frame.frame_id)

    worker._pending_system_one_appraisal_frame = frame
    asyncio.run(worker._publish_pending_system_one_appraisal())

    assert len(bus.published) == 1, "typed frame was not published"
    channel, env = bus.published[0]
    assert channel == SYSTEM_ONE_APPRAISAL_CHANNEL
    assert isinstance(env, BaseEnvelope)
    assert env.kind == SYSTEM_ONE_APPRAISAL_KIND
    assert isinstance(env.correlation_id, UUID)
    # frame_id survives in the payload; the payload round-trips as the schema.
    assert env.payload["frame_id"] == frame.frame_id
    assert SystemOneAppraisalFrameV1.model_validate(env.payload).frame_id == frame.frame_id
    assert worker._pending_system_one_appraisal_frame is None

    # Deterministic: a retry of the same frame correlates to the same id.
    worker._pending_system_one_appraisal_frame = frame
    asyncio.run(worker._publish_pending_system_one_appraisal())
    assert bus.published[1][1].correlation_id == env.correlation_id
