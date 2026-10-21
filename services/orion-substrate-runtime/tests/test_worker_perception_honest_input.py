"""Camera surprise, honest input (2026-10-21).

Two live failures on node:substrate.perception / perception_walkway:

1. Peak-hold (publish the window's max) was proposed and REJECTED by the
   metric gate: the score saturates at 1.0, so the window max sat at 1.0 in 10%
   of windows (exact-zero ticks 61% -> 4.5%, curiosity crossings 7x). The tick
   publishes the latest frame not yet published; no new frame = missing.
2. Missing input (49 h with no embeddings, 2026-10-18..20) was written as 0.0
   every tick and recorded in substrate_node_prediction_error_history, which
   pulled the 7-day p90 from 0.227 to 0.022. Fix: the placeholder is marked
   ``prediction_error_missing_reason`` and never sampled into the history.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import app.worker as worker_mod
from app.worker import (
    PERCEPTION_MISSING_NO_FRAME,
    PERCEPTION_MISSING_STALE,
    PERCEPTION_MISSING_WARMING,
    BiometricsSubstrateWorker,
)

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.substrate.prediction_error import PerceptionEmbeddingBaseline, _DomainEwmaBaseline
from orion.substrate.prediction_error_freshness import (
    PREDICTION_ERROR_MISSING_REASON_KEY,
    STATUS_FRESH,
    STATUS_MISSING_INPUT,
    classify_reading,
    missing_input_reason,
)

ROOM = "node:substrate.perception"
WALKWAY = "node:substrate.perception_walkway"
_SOURCE = ServiceRef(name="vision-retina", version="0.1.0", node="athena")


def _message(stream: str) -> dict:
    payload = {
        "artifact_id": "a1",
        "correlation_id": "c1",
        "task_type": "retina_fast",
        "device": "cuda:0",
        "inputs": {"camera_id": stream, "stream_id": stream},
        "outputs": {"embedding": {"ref": "e", "path": "/tmp/e.npy", "dim": 2, "vector": [0.0, 1.0]}},
        "timing": {"latency_s": 0.1},
        "model_fingerprints": {},
    }
    envelope = BaseEnvelope(kind="vision.artifact.v1", source=_SOURCE, payload=payload)
    return {"type": "message", "data": OrionCodec().encode(envelope)}


def _worker(monkeypatch, scores: list[float | None]):
    """Worker whose scorer returns ``scores`` in order (one per frame)."""
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=10, surprise=_DomainEwmaBaseline(ewma=0.01, variance=1e-4, n=10)
    )
    queue = list(scores)

    def fake_score(_vec, _baseline):
        return SimpleNamespace(score=queue.pop(0), baseline=baseline, raw_surprise=0.1)

    monkeypatch.setattr(worker_mod, "perception_prediction_error", fake_score)
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    worker._store = MagicMock()
    worker._store.get_latest_perception_embedding_baseline.return_value = baseline
    worker._settings = MagicMock()
    worker._settings.perception_baseline_retention_days = 30.0
    worker._settings.enable_perception_prediction_error_tick = True
    worker._settings.perception_walkway_stream_ids = "walkway"
    worker._settings.perception_log_only_stream_ids = "doorbell"
    worker._write_prediction_error_node = MagicMock()
    worker._last_perception_prediction_error = None
    worker._last_perception_embedding_at = None
    worker._last_perception_stream_id = None
    worker._last_perception_surprise_n = 0
    worker._last_perception_embedding_vector = None
    worker._process_started_at = datetime.now(timezone.utc)
    return worker


def _write(worker, node_id: str) -> dict:
    calls = [c.kwargs for c in worker._write_prediction_error_node.call_args_list if c.kwargs["node_id"] == node_id]
    return calls[-1]


def test_tick_publishes_the_latest_frame_not_the_window_peak(monkeypatch) -> None:
    # Pins the metric-gate decision: the window max saturates (see docstring).
    worker = _worker(monkeypatch, [0.2, 1.0, 0.1])
    for _ in range(3):
        worker._handle_perception_prediction_error_message(_message("cam0"))
    worker._perception_prediction_error_tick()
    room = _write(worker, ROOM)
    assert room["error"] == 0.1
    assert room["missing_reason"] is None


def test_walkway_tick_publishes_its_own_latest_frame(monkeypatch) -> None:
    worker = _worker(monkeypatch, [0.3, 0.9, 0.4])
    for _ in range(3):
        worker._handle_perception_prediction_error_message(_message("walkway"))
    worker._perception_prediction_error_tick()
    assert _write(worker, WALKWAY)["error"] == 0.4
    assert _write(worker, WALKWAY)["missing_reason"] is None


def test_published_frame_is_never_republished_and_an_empty_window_is_missing(monkeypatch) -> None:
    # Regression: the old tick re-published the last frame's score while the
    # input was < 60 s old, a copy recorded as a new reading.
    worker = _worker(monkeypatch, [0.8])
    worker._handle_perception_prediction_error_message(_message("cam0"))
    worker._perception_prediction_error_tick()
    assert _write(worker, ROOM)["error"] == 0.8
    # Next tick: embeddings still fresh, but no scored frame since the publish.
    worker._perception_prediction_error_tick()
    room = _write(worker, ROOM)
    assert room["error"] == 0.0
    assert room["missing_reason"] == PERCEPTION_MISSING_NO_FRAME


def test_a_calm_window_is_a_real_zero_not_missing(monkeypatch) -> None:
    # Rest point: every frame at or below the usual reads 0.0 AND is a reading.
    worker = _worker(monkeypatch, [0.0, 0.0])
    for _ in range(2):
        worker._handle_perception_prediction_error_message(_message("cam0"))
    worker._perception_prediction_error_tick()
    room = _write(worker, ROOM)
    assert room["error"] == 0.0
    assert room["missing_reason"] is None


def test_stale_input_is_marked_missing_not_calm(monkeypatch) -> None:
    worker = _worker(monkeypatch, [])
    worker._last_perception_prediction_error = 0.7  # held from before the camera died
    worker._last_perception_embedding_at = datetime.now(timezone.utc) - timedelta(hours=2)
    worker._last_perception_surprise_n = 50
    worker._perception_prediction_error_tick()
    room = _write(worker, ROOM)
    assert room["error"] == 0.0
    assert room["missing_reason"] == PERCEPTION_MISSING_STALE
    assert room["extra_channels"]["embedding_staleness"] == 1.0


def test_warming_input_is_marked_missing(monkeypatch) -> None:
    worker = _worker(monkeypatch, [])
    walkway = worker._perception_walkway_state()
    walkway.embedding_at = datetime.now(timezone.utc)
    walkway.surprise_n = 1
    walkway.score = 0.9  # must not leak through while warming
    worker._perception_prediction_error_tick()
    assert _write(worker, WALKWAY)["error"] == 0.0
    assert _write(worker, WALKWAY)["missing_reason"] == PERCEPTION_MISSING_WARMING


def _node(node_id: str, value: float, observed_at: datetime, missing: str | None = None):
    metadata = {"prediction_error": value}
    if missing:
        metadata[PREDICTION_ERROR_MISSING_REASON_KEY] = missing
    return SimpleNamespace(node_id=node_id, metadata=metadata, temporal=SimpleNamespace(observed_at=observed_at))


def _history_worker(monkeypatch) -> BiometricsSubstrateWorker:
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    import app.settings as settings_mod

    settings_mod._settings = None
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._settings = settings_mod.get_settings()
    worker._store = MagicMock()
    worker._store.fetch_prediction_error_history.return_value = []
    return worker


def test_missing_placeholder_never_reaches_the_history(monkeypatch) -> None:
    worker = _history_worker(monkeypatch)
    now = datetime.now(timezone.utc)
    nodes = [
        _node(ROOM, 0.0, now - timedelta(seconds=5), missing=PERCEPTION_MISSING_STALE),
        _node(WALKWAY, 0.42, now - timedelta(seconds=5)),
    ]
    magnitudes = worker._prediction_error_magnitudes(nodes=nodes, now=now)
    saved = worker._store.save_prediction_error_history_samples.call_args.args[0]
    assert [row[0] for row in saved] == [WALKWAY]
    # The placeholder has no magnitude this tick either: absent, not quiet.
    assert ROOM not in magnitudes
    assert WALKWAY in magnitudes


def test_reading_after_a_missing_stretch_is_recorded(monkeypatch) -> None:
    worker = _history_worker(monkeypatch)
    now = datetime.now(timezone.utc)
    worker._prediction_error_magnitudes(
        nodes=[_node(ROOM, 0.0, now - timedelta(seconds=60), missing=PERCEPTION_MISSING_STALE)], now=now
    )
    worker._store.save_prediction_error_history_samples.assert_not_called()
    worker._prediction_error_magnitudes(nodes=[_node(ROOM, 0.31, now)], now=now)
    saved = worker._store.save_prediction_error_history_samples.call_args.args[0]
    assert saved == [(ROOM, now, 0.31)]


def test_freshness_classifier_omits_missing_input() -> None:
    now = datetime.now(timezone.utc)
    assert classify_reading(_node(ROOM, 0.0, now, missing="stale"), now=now)[0] == STATUS_MISSING_INPUT
    assert classify_reading(_node(ROOM, 0.0, now), now=now)[0] == STATUS_FRESH
    assert missing_input_reason(_node(ROOM, 0.0, now, missing="  ")) is None
