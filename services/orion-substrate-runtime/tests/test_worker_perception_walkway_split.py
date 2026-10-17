"""The walkway camera's frame-embedding surprise feeds its own node,
node:substrate.perception_walkway, and never the room node
node:substrate.perception (2026-10-15).

Before this split one process-wide cache held whichever stream scored last, so
a busy street at 2 frames/s would have overwritten the room cameras' reading
that endogenous curiosity and attention read.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSTRATE_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, SUBSTRATE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.worker import BiometricsSubstrateWorker, _PerceptionGroupState, _parse_stream_ids

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.substrate.prediction_error import _DomainEwmaBaseline, PerceptionEmbeddingBaseline

ROOM = "node:substrate.perception"
WALKWAY = "node:substrate.perception_walkway"
_SOURCE = ServiceRef(name="vision-retina", version="0.1.0", node="athena")


def _message(stream: str, vector: list[float]) -> dict:
    payload = {
        "artifact_id": "a1",
        "correlation_id": "c1",
        "task_type": "retina_fast",
        "device": "cuda:0",
        "inputs": {"camera_id": stream, "stream_id": stream},
        "outputs": {"embedding": {"ref": "emb:x", "path": "/tmp/e.npy", "dim": len(vector), "vector": vector}},
        "timing": {"latency_s": 0.1},
        "model_fingerprints": {},
    }
    envelope = BaseEnvelope(kind="vision.artifact.v1", source=_SOURCE, payload=payload)
    return {"type": "message", "data": OrionCodec().encode(envelope)}


def _warm_baseline() -> PerceptionEmbeddingBaseline:
    """A baseline that already has an embedding EWMA and a warm surprise EWMA,
    so the next real frame produces a real score (not the warm-up None)."""
    return PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0),
        n=10,
        surprise=_DomainEwmaBaseline(ewma=0.01, variance=0.0001, n=10),
    )


def _worker(*, walkway_ids: str | None = "walkway", started_ago_sec: float = 0.0):
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    worker._store = MagicMock()
    worker._store.get_latest_perception_embedding_baseline.side_effect = lambda _sid: _warm_baseline()
    worker._settings = MagicMock()
    worker._settings.perception_baseline_retention_days = 30.0
    worker._settings.enable_perception_prediction_error_tick = True
    if walkway_ids is not None:
        worker._settings.perception_walkway_stream_ids = walkway_ids
    worker._write_prediction_error_node = MagicMock()
    worker._last_perception_prediction_error = None
    worker._last_perception_embedding_at = None
    worker._last_perception_stream_id = None
    worker._last_perception_surprise_n = 0
    worker._last_perception_embedding_vector = None
    worker._process_started_at = datetime.now(timezone.utc) - timedelta(seconds=started_ago_sec)
    return worker


def _writes(worker) -> dict[str, dict]:
    return {
        call.kwargs["node_id"]: call.kwargs
        for call in worker._write_prediction_error_node.call_args_list
    }


def _receipt_targets(worker) -> list[str]:
    return [
        call.args[0].state_deltas[0].target_id for call in worker._store.save_receipt.call_args_list
    ]


def test_walkway_embedding_never_changes_the_room_node() -> None:
    worker = _worker()
    worker._last_perception_prediction_error = 0.11
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_stream_id = "cam0"
    worker._last_perception_surprise_n = 5
    worker._last_perception_embedding_vector = [0.5, 0.5]

    worker._handle_perception_prediction_error_message(_message("walkway", [0.0, 1.0]))

    # Room cache untouched, including the world-model vector cache.
    assert worker._last_perception_prediction_error == 0.11
    assert worker._last_perception_stream_id == "cam0"
    assert worker._last_perception_surprise_n == 5
    assert worker._last_perception_embedding_vector == [0.5, 0.5]
    # Walkway got a real score of its own.
    walkway = worker._perception_walkway_state()
    assert walkway.stream_id == "walkway"
    assert walkway.score is not None and walkway.score > 0.0
    assert walkway.surprise_n >= 2

    worker._perception_prediction_error_tick()
    writes = _writes(worker)
    assert writes[ROOM]["error"] == 0.11
    assert writes[WALKWAY]["error"] == walkway.score


def test_room_embedding_never_changes_the_walkway_node() -> None:
    worker = _worker()
    walkway = worker._perception_walkway_state()
    walkway.score = 0.33
    walkway.embedding_at = datetime.now(timezone.utc)
    walkway.stream_id = "walkway"
    walkway.surprise_n = 4

    worker._handle_perception_prediction_error_message(_message("cam0", [0.0, 1.0]))

    assert walkway == _PerceptionGroupState(
        score=0.33, surprise_n=4, embedding_at=walkway.embedding_at, stream_id="walkway"
    )
    assert worker._last_perception_stream_id == "cam0"
    assert worker._last_perception_embedding_vector == [0.0, 1.0]
    assert worker._last_perception_prediction_error is not None

    worker._perception_prediction_error_tick()
    writes = _writes(worker)
    assert writes[WALKWAY]["error"] == 0.33
    assert writes[ROOM]["error"] == worker._last_perception_prediction_error


def test_both_nodes_written_every_tick_with_receipts() -> None:
    worker = _worker()
    now = datetime.now(timezone.utc)
    worker._last_perception_prediction_error = 0.2
    worker._last_perception_embedding_at = now
    worker._last_perception_surprise_n = 3
    walkway = worker._perception_walkway_state()
    walkway.score, walkway.embedding_at, walkway.surprise_n = 0.6, now, 3

    for _ in range(3):
        worker._perception_prediction_error_tick()

    calls = [c.kwargs["node_id"] for c in worker._write_prediction_error_node.call_args_list]
    assert calls.count(ROOM) == 3 and calls.count(WALKWAY) == 3
    targets = _receipt_targets(worker)
    assert targets.count(ROOM) == 3 and targets.count(WALKWAY) == 3
    walkway_receipt = next(
        c.args[0] for c in worker._store.save_receipt.call_args_list
        if c.args[0].state_deltas[0].target_id == WALKWAY
    )
    assert walkway_receipt.state_deltas[0].reducer_id == "substrate.perception_walkway"
    assert _writes(worker)[WALKWAY]["reducer_key"] == "perception_walkway"
    assert "embedding_staleness" in _writes(worker)[WALKWAY]["extra_channels"]


def test_walkway_goes_stale_to_zero_while_room_stays_live() -> None:
    worker = _worker()
    now = datetime.now(timezone.utc)
    worker._last_perception_prediction_error = 0.4
    worker._last_perception_embedding_at = now  # room camera delivering
    worker._last_perception_surprise_n = 3
    walkway = worker._perception_walkway_state()
    walkway.score = 0.9  # last real walkway score, must not outlive its evidence
    walkway.embedding_at = now - timedelta(seconds=300)
    walkway.surprise_n = 3

    worker._perception_prediction_error_tick()
    writes = _writes(worker)
    assert writes[WALKWAY]["error"] == 0.0
    assert writes[WALKWAY]["extra_channels"]["embedding_staleness"] == 1.0
    assert writes[ROOM]["error"] == 0.4
    assert writes[ROOM]["extra_channels"]["embedding_staleness"] == 0.0


def test_walkway_never_seen_reads_stale_after_startup_grace_not_skipped() -> None:
    """The walkway camera has not delivered yet: once past the startup grace the
    node is still written, as 0.0 with staleness rising, not left unwritten for
    the field digester's decay to fake a calm."""
    worker = _worker(started_ago_sec=600.0)
    worker._last_perception_prediction_error = 0.4
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_surprise_n = 3

    worker._perception_prediction_error_tick()
    writes = _writes(worker)
    assert writes[WALKWAY]["error"] == 0.0
    assert writes[WALKWAY]["extra_channels"]["embedding_staleness"] == 1.0


def test_walkway_still_warming_writes_zero() -> None:
    worker = _worker()
    walkway = worker._perception_walkway_state()
    walkway.score = 0.8
    walkway.embedding_at = datetime.now(timezone.utc)
    walkway.surprise_n = 1
    worker._perception_prediction_error_tick()
    assert _writes(worker)[WALKWAY]["error"] == 0.0


def test_walkway_write_failure_does_not_skip_the_room_node() -> None:
    worker = _worker()
    worker._last_perception_prediction_error = 0.2
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_surprise_n = 3
    walkway = worker._perception_walkway_state()
    walkway.embedding_at = datetime.now(timezone.utc)

    def _boom(**kwargs):
        if kwargs["node_id"] == WALKWAY:
            raise RuntimeError("boom")

    worker._write_prediction_error_node.side_effect = _boom
    worker._perception_prediction_error_tick()  # must not raise
    assert ROOM in [c.kwargs["node_id"] for c in worker._write_prediction_error_node.call_args_list]


def test_stream_group_parsing() -> None:
    assert _parse_stream_ids("walkway", "x") == frozenset({"walkway"})
    assert _parse_stream_ids(" walkway , street ,", "x") == frozenset({"walkway", "street"})
    assert _parse_stream_ids(None, "walkway") == frozenset({"walkway"})
    assert _parse_stream_ids(MagicMock(), "walkway") == frozenset({"walkway"})
    assert _parse_stream_ids("", "walkway") == frozenset()


def test_stream_not_in_group_feeds_room_node() -> None:
    """With the walkway group pointed elsewhere, a walkway frame is a room frame
    again -- the setting, not the stream name, decides the group."""
    worker = _worker(walkway_ids="none")
    worker._handle_perception_prediction_error_message(_message("walkway", [0.0, 1.0]))
    assert worker._last_perception_stream_id == "walkway"
    assert worker._perception_walkway_state().stream_id is None


def test_settings_default_routes_walkway_stream() -> None:
    from app.settings import Settings

    settings = Settings(POSTGRES_URI="postgresql://unused/unused")
    assert settings.perception_walkway_stream_ids == "walkway"


def test_doorbell_is_scored_and_baselined_but_feeds_no_node() -> None:
    """2026-10-17: Ring doorbell bursts must not overwrite the walkway node
    (score, warm-up count, staleness clock) nor the room node."""
    worker = _worker()
    worker._last_perception_prediction_error = 0.11
    worker._last_perception_stream_id = "cam0"
    walkway = worker._perception_walkway_state()
    walkway.score, walkway.surprise_n, walkway.stream_id = 0.2, 7, "walkway"

    worker._handle_perception_prediction_error_message(_message("doorbell", [0.0, 1.0]))

    assert (walkway.score, walkway.surprise_n, walkway.stream_id, walkway.embedding_at) == (0.2, 7, "walkway", None)
    assert worker._last_perception_prediction_error == 0.11 and worker._last_perception_stream_id == "cam0"
    # Its own per-stream baseline still learns.
    saved = [c.args[0] for c in worker._store.save_perception_embedding_baseline.call_args_list]
    assert saved == ["doorbell"]


def test_settings_default_log_only_stream_is_doorbell() -> None:
    from app.settings import Settings

    assert Settings(POSTGRES_URI="postgresql://unused/unused").perception_log_only_stream_ids == "doorbell"
