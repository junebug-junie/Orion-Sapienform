"""The camera tick under the rank_v2 scorer (PE retire Patch 3, design rev 4 item 3).

Each frame's raw 1 - cos feeds stage 1 for its node (room or walkway; never the log-only
doorbell). Every 30 s tick runs stage 2 and writes the node through ``_write_signal_node``:
``warming`` until both stages can score, then the value in nats with ``signal_scale``
(``v2_partial`` until stage 2 holds 24 h, then ``v2``). Stage 2 is restored from the DB on the
first tick and saved whenever it gains a value; a failed load never lets an empty window
overwrite the saved one. No field receipt, no prediction_error write, in any state.
"""
from __future__ import annotations

import random
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

import app.worker as worker_mod  # noqa: E402
from app.worker import BiometricsSubstrateWorker  # noqa: E402

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.substrate.prediction_error import PerceptionEmbeddingBaseline, _DomainEwmaBaseline  # noqa: E402
from orion.substrate.scene_change import STAGE2_MIN_VALUES, SceneChangeScorer  # noqa: E402

ROOM = "node:substrate.perception"
WALKWAY = "node:substrate.perception_walkway"
_SOURCE = ServiceRef(name="vision-retina", version="0.1.0", node="athena")


def _worker(*, scorer: str = "rank_v2", saved=None, load_error: Exception | None = None):
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    worker._store = MagicMock()
    if load_error is not None:
        worker._store.load_scene_change_stage2.side_effect = load_error
    else:
        worker._store.load_scene_change_stage2.return_value = saved
    worker._settings = MagicMock()
    worker._settings.enable_perception_prediction_error_tick = True
    worker._settings.perception_walkway_stream_ids = "walkway"
    worker._settings.perception_log_only_stream_ids = "doorbell"
    worker._settings.perception_baseline_retention_days = 30.0
    worker._settings.perception_scene_change_scorer = scorer
    worker._write_signal_node = MagicMock(return_value=True)
    worker._write_prediction_error_node = MagicMock()
    worker._last_perception_prediction_error = None
    worker._last_perception_embedding_at = None
    worker._last_perception_stream_id = None
    worker._last_perception_surprise_n = 0
    worker._last_perception_embedding_vector = None
    worker._process_started_at = datetime.now(timezone.utc) - timedelta(hours=1)
    return worker


def _write(worker, node_id: str) -> dict:
    return [c.kwargs for c in worker._write_signal_node.call_args_list if c.kwargs["node_id"] == node_id][-1]


def _warm(scorer: SceneChangeScorer, now: datetime, *, hours_of_stage2: float, rng) -> None:
    """Stage 1 ready (300 frames over 2 h) and a stage-2 history spanning ``hours_of_stage2``."""
    for i in range(300):
        scorer.observe("cam0", now - timedelta(hours=2) + timedelta(seconds=24 * i), rng.random())
    n = int(hours_of_stage2 * 12)
    scorer.stage2.extend((now - timedelta(minutes=5 * (n - i)), 0.01 + rng.random() * 0.04) for i in range(n))
    scorer.last_stage2_at = now - timedelta(minutes=6)
    scorer.restored = True


def _fresh(worker, now):
    worker._last_perception_embedding_at = now
    worker._last_perception_stream_id = "cam0"
    walkway = worker._perception_walkway_state()
    walkway.embedding_at, walkway.stream_id = now, "walkway"


def test_a_cold_scorer_writes_the_warming_placeholder_never_a_calm_zero() -> None:
    worker = _worker()
    now = datetime.now(timezone.utc)
    _fresh(worker, now)
    worker._perception_prediction_error_tick()
    for node_id in (ROOM, WALKWAY):
        w = _write(worker, node_id)
        assert w["value"] == 0.0 and w["missing_reason"] == "warming"
        assert w["signal_scale"] is None
    worker._store.save_receipt.assert_not_called()
    worker._write_prediction_error_node.assert_not_called()


def test_a_warm_scorer_writes_nats_with_a_partial_scale_and_saves_stage2() -> None:
    worker = _worker()
    now = datetime.now(timezone.utc)
    _fresh(worker, now)
    rng = random.Random(11)
    with worker._scene_change_lock():
        scorer = worker._scene_change_scorer_for(ROOM)
    _warm(scorer, now, hours_of_stage2=6, rng=rng)
    # a frame above everything in the last 2 h, in the trailing 5 min
    worker._scene_change_observe(ROOM, "cam0", now - timedelta(seconds=20), 9.0)
    worker._perception_prediction_error_tick()
    w = _write(worker, ROOM)
    assert w["missing_reason"] is None
    assert w["signal_scale"] == "v2_partial"
    assert w["value"] > 2.0  # rarer than every past 5-min minimum: crosses the curiosity bar
    assert w["extra_channels"]["embedding_staleness"] == 0.0
    # it sampled a stage-2 value this tick, so the history was saved under the node's own key
    (node_id, payload), _ = worker._store.save_scene_change_stage2.call_args
    assert node_id == ROOM and payload["version"] == 1 and len(payload["values"]) == 6 * 12 + 1
    worker._store.save_receipt.assert_not_called()


def test_restored_stage2_spanning_24h_reads_scale_v2() -> None:
    now = datetime.now(timezone.utc)
    saved = {
        "version": 1,
        "values": [[(now - timedelta(minutes=5 * (288 - i))).isoformat(), 0.02] for i in range(288)],
        "last_sample_at": (now - timedelta(minutes=5)).isoformat(),
    }
    worker = _worker(saved=saved)
    _fresh(worker, now)
    with worker._scene_change_lock():
        scorer = worker._scene_change_scorer_for(ROOM)
    rng = random.Random(2)
    for i in range(300):
        scorer.observe("cam0", now - timedelta(hours=2) + timedelta(seconds=24 * i), rng.random())
    worker._perception_prediction_error_tick()
    worker._store.load_scene_change_stage2.assert_any_call(ROOM)
    w = _write(worker, ROOM)
    assert w["missing_reason"] is None and w["signal_scale"] == "v2"
    assert 0.0 <= w["value"] <= 4.67


def test_a_failed_load_never_overwrites_the_saved_window_and_is_retried() -> None:
    now = datetime.now(timezone.utc)
    worker = _worker(load_error=RuntimeError("db down"))
    _fresh(worker, now)
    with worker._scene_change_lock():
        scorer = worker._scene_change_scorer_for(ROOM)
    for i in range(300):
        scorer.observe("cam0", now - timedelta(hours=2) + timedelta(seconds=24 * i), 0.1)
    worker._perception_prediction_error_tick()
    worker._store.save_scene_change_stage2.assert_not_called()
    assert not scorer.restored
    worker._store.load_scene_change_stage2.side_effect = None
    worker._store.load_scene_change_stage2.return_value = None
    worker._perception_prediction_error_tick()
    assert scorer.restored


def test_a_stale_camera_is_missing_even_when_the_scorer_has_a_value() -> None:
    worker = _worker()
    now = datetime.now(timezone.utc)
    _fresh(worker, now)
    worker._last_perception_embedding_at = now - timedelta(minutes=10)  # no frame for 10 min
    with worker._scene_change_lock():
        scorer = worker._scene_change_scorer_for(ROOM)
    _warm(scorer, now, hours_of_stage2=6, rng=random.Random(4))
    worker._perception_prediction_error_tick()
    w = _write(worker, ROOM)
    assert w["value"] == 0.0 and w["missing_reason"] == "stale" and w["signal_scale"] is None
    assert w["extra_channels"]["embedding_staleness"] == 1.0


def _message(stream: str) -> dict:
    payload = {
        "artifact_id": "a1", "correlation_id": "c1", "task_type": "retina_fast", "device": "cuda:0",
        "inputs": {"camera_id": stream, "stream_id": stream},
        "outputs": {"embedding": {"ref": "e", "path": "/tmp/e.npy", "dim": 2, "vector": [0.0, 1.0]}},
        "timing": {"latency_s": 0.1}, "model_fingerprints": {},
    }
    envelope = BaseEnvelope(kind="vision.artifact.v1", source=_SOURCE, payload=payload)
    return {"type": "message", "data": OrionCodec().encode(envelope)}


def test_each_frame_feeds_stage1_of_its_own_node_and_never_the_doorbell(monkeypatch) -> None:
    baseline = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=10, surprise=_DomainEwmaBaseline(ewma=0.01, variance=1e-4, n=10)
    )
    monkeypatch.setattr(
        worker_mod, "perception_prediction_error",
        lambda _v, _b: SimpleNamespace(score=0.3, baseline=baseline, raw_surprise=0.123),
    )
    worker = _worker()
    worker._store.get_latest_perception_embedding_baseline.return_value = baseline
    for stream in ("cam0", "walkway", "walkway", "doorbell"):
        worker._handle_perception_prediction_error_message(_message(stream))
    scorers = worker._scene_change_scorers
    assert len(scorers[ROOM].streams["cam0"]) == 1
    assert len(scorers[WALKWAY].streams["walkway"]) == 2
    assert all("doorbell" not in s.streams for s in scorers.values())


def test_legacy_scorer_writes_the_old_score_as_scale_v1() -> None:
    worker = _worker(scorer="legacy_z")
    now = datetime.now(timezone.utc)
    _fresh(worker, now)
    worker._last_perception_surprise_n = 5
    worker._last_perception_prediction_error = 0.42
    worker._perception_prediction_error_tick()
    w = _write(worker, ROOM)
    assert w["value"] == 0.42 and w["signal_scale"] == "v1" and w["missing_reason"] is None
    worker._store.load_scene_change_stage2.assert_not_called()


def test_an_unknown_scorer_value_runs_rank_v2() -> None:
    worker = _worker(scorer="rank_v3")
    assert worker._scene_change_scorer_mode() == "rank_v2"


def test_setting_ships_rank_v2_everywhere() -> None:
    import os

    from app.settings import Settings

    os.environ.setdefault("POSTGRES_URI", "postgresql://unused/unused")
    assert Settings().perception_scene_change_scorer == "rank_v2"
    assert "\nSUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER=rank_v2\n" in (SUBSTRATE_ROOT / ".env_example").read_text()
    assert "SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER=${SUBSTRATE_PERCEPTION_SCENE_CHANGE_SCORER:-rank_v2}" in (
        SUBSTRATE_ROOT / "docker-compose.yml"
    ).read_text()
    assert STAGE2_MIN_VALUES == 48
