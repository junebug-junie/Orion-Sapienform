"""Unit tests for P2's perceptual prediction-error producer in `app/worker.py`
(docs/superpowers/specs/2026-08-12-perception-frontier-design.md):
`_handle_perception_prediction_error_message` (event-driven, scores a real
embedding-bearing vision artifact) and `_perception_prediction_error_tick`
(clock-driven companion, writes `node:substrate.perception` on a fixed
interval so silence still refreshes the node -- same shape as
`_vision_channel_tick`/`_bus_synaptic_tick`).

Uses the real OrionCodec to encode/decode a real raw-dict vision-artifact
payload (matching `_handle_vision_artifact_message`'s own test-free but
directly-inspected shape), mirroring
test_worker_codebase_delta_consumer.py's pattern -- store and
`_write_prediction_error_node` mocked.
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

from app.worker import BiometricsSubstrateWorker

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.codec import OrionCodec
from orion.substrate.prediction_error import _DomainEwmaBaseline, PerceptionEmbeddingBaseline

_SOURCE = ServiceRef(name="vision-host", version="0.1.0", node="athena")


def _encode_artifact(payload: dict) -> dict:
    codec = OrionCodec()
    envelope = BaseEnvelope(kind="vision.artifact.v1", source=_SOURCE, payload=payload)
    return {"type": "message", "data": codec.encode(envelope)}


def _artifact_payload(*, camera_id: str = "cam0", vector=None) -> dict:
    outputs = {"objects": [{"label": "cup", "score": 0.9, "box_xyxy": [0, 0, 1, 1]}]}
    if vector is not None:
        outputs["embedding"] = {"ref": "emb:x", "path": "/tmp/e.npy", "dim": len(vector), "vector": vector}
    return {
        "artifact_id": "a1",
        "correlation_id": "c1",
        "task_type": "retina_fast",
        "device": "cuda:0",
        "inputs": {"camera_id": camera_id, "stream_id": "stream-1"},
        "outputs": outputs,
        "timing": {"latency_s": 0.1},
        "model_fingerprints": {},
    }


def _make_worker(*, baseline: PerceptionEmbeddingBaseline | None = None) -> tuple[BiometricsSubstrateWorker, MagicMock]:
    worker = BiometricsSubstrateWorker.__new__(BiometricsSubstrateWorker)
    worker._bus = MagicMock()
    worker._bus.codec = OrionCodec()
    fake_store = MagicMock()
    fake_store.get_latest_perception_embedding_baseline.return_value = (
        baseline or PerceptionEmbeddingBaseline()
    )
    worker._store = fake_store
    worker._settings = MagicMock()
    worker._settings.perception_baseline_retention_days = 30.0
    worker._settings.enable_perception_prediction_error_tick = True
    worker._write_prediction_error_node = MagicMock()
    worker._last_perception_prediction_error = None
    worker._last_perception_embedding_at = None
    worker._last_perception_stream_id = None
    worker._last_perception_surprise_n = 0
    worker._process_started_at = datetime.now(timezone.utc)
    return worker, fake_store


# --- _handle_perception_prediction_error_message ----------------------------


def test_message_without_embedding_is_a_noop() -> None:
    worker, fake_store = _make_worker()
    worker._handle_perception_prediction_error_message(_encode_artifact(_artifact_payload(camera_id="cam0")))
    fake_store.get_latest_perception_embedding_baseline.assert_not_called()
    fake_store.save_perception_embedding_baseline.assert_not_called()
    assert worker._last_perception_embedding_at is None


def test_cold_start_message_seeds_baseline_and_caches_no_score() -> None:
    worker, fake_store = _make_worker()
    worker._handle_perception_prediction_error_message(
        _encode_artifact(_artifact_payload(camera_id="cam0", vector=[1.0, 0.0]))
    )
    fake_store.save_perception_embedding_baseline.assert_called_once()
    args, kwargs = fake_store.save_perception_embedding_baseline.call_args
    assert args[0] == "cam0"
    assert args[1].embedding_ewma == (1.0, 0.0)
    assert args[1].n == 1
    assert kwargs["retention_days"] == 30.0
    # No score yet on cold start -- must not fabricate one.
    assert worker._last_perception_prediction_error is None
    assert worker._last_perception_embedding_at is not None
    assert worker._last_perception_stream_id == "cam0"


def test_orthogonal_second_observation_seeds_scalar_baseline_but_caches_no_score() -> None:
    """Hand-computed raw magnitude: cos([1,0],[0,1]) = 0 -> raw surprise =
    1.0 (same arithmetic as orion/substrate/tests/test_perception_
    prediction_error.py's own orthogonal-vector test). This stream's first
    real cosine comparison, but its scalar surprise EWMA (2026-08-19
    z-score migration -- see orion/substrate/prediction_error.py's own
    comment) is still cold, so no score is reported or cached yet -- same
    "no empty-shell cognition" convention as the cold-start case above, one
    layer up."""
    worker, fake_store = _make_worker(
        baseline=PerceptionEmbeddingBaseline(embedding_ewma=(1.0, 0.0), n=1)
    )
    worker._handle_perception_prediction_error_message(
        _encode_artifact(_artifact_payload(camera_id="cam0", vector=[0.0, 1.0]))
    )
    assert worker._last_perception_prediction_error is None
    saved_args, _ = fake_store.save_perception_embedding_baseline.call_args
    assert saved_args[1].embedding_ewma == (0.8, 0.2)
    assert saved_args[1].n == 2
    assert saved_args[1].surprise.ewma == 1.0
    assert saved_args[1].surprise.n == 1


def test_warmed_scalar_baseline_scores_and_caches_zscored_result() -> None:
    """Hand-computed (same numbers as orion/substrate/tests/
    test_perception_prediction_error.py::
    test_mid_range_zscore_hand_computed_against_seeded_baseline):
    baseline=(1,0) unit, embedding chosen so cos=0.825 exactly -> raw
    surprise=0.175. Scalar surprise baseline pre-seeded ewma=0.1,
    variance=0.0025 (std=0.05), n=5.
    zscore = (0.175-0.1)/0.05 = 1.5 -> score = min(1.0, 1.5/3.0) = 0.5.
    """
    warmed = PerceptionEmbeddingBaseline(
        embedding_ewma=(1.0, 0.0), n=6, surprise=_DomainEwmaBaseline(ewma=0.1, variance=0.0025, n=5)
    )
    worker, fake_store = _make_worker(baseline=warmed)
    worker._handle_perception_prediction_error_message(
        _encode_artifact(_artifact_payload(camera_id="cam0", vector=[0.825, 0.5651327277728658]))
    )
    assert worker._last_perception_prediction_error == 0.5
    saved_args, _ = fake_store.save_perception_embedding_baseline.call_args
    assert saved_args[1].surprise.n == 6


def test_falls_back_to_stream_id_when_camera_id_absent() -> None:
    worker, fake_store = _make_worker()
    payload = _artifact_payload(camera_id="cam0", vector=[1.0, 0.0])
    del payload["inputs"]["camera_id"]
    worker._handle_perception_prediction_error_message(_encode_artifact(payload))
    args, _ = fake_store.save_perception_embedding_baseline.call_args
    assert args[0] == "stream-1"


def test_missing_camera_and_stream_id_falls_back_to_unknown() -> None:
    worker, fake_store = _make_worker()
    payload = _artifact_payload(camera_id="cam0", vector=[1.0, 0.0])
    payload["inputs"] = {}
    worker._handle_perception_prediction_error_message(_encode_artifact(payload))
    args, _ = fake_store.save_perception_embedding_baseline.call_args
    assert args[0] == "unknown"


# --- _perception_prediction_error_tick --------------------------------------


def test_tick_disabled_is_noop() -> None:
    worker, fake_store = _make_worker()
    worker._settings.enable_perception_prediction_error_tick = False
    worker._perception_prediction_error_tick()
    worker._write_prediction_error_node.assert_not_called()
    fake_store.save_receipt.assert_not_called()


def test_tick_before_any_embedding_and_within_startup_grace_skips_write() -> None:
    worker, _ = _make_worker()
    worker._process_started_at = datetime.now(timezone.utc)  # fresh start, inside grace
    worker._perception_prediction_error_tick()
    worker._write_prediction_error_node.assert_not_called()


def test_tick_writes_cached_score_and_zero_staleness_when_fresh() -> None:
    worker, fake_store = _make_worker()
    worker._last_perception_prediction_error = 0.42
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_stream_id = "cam0"
    worker._last_perception_surprise_n = 3  # confirmed score, not still warming
    worker._perception_prediction_error_tick()

    worker._write_prediction_error_node.assert_called_once()
    _, kwargs = worker._write_prediction_error_node.call_args
    assert kwargs["node_id"] == "node:substrate.perception"
    assert kwargs["reducer_key"] == "perception"
    assert kwargs["error"] == 0.42
    assert kwargs["extra_channels"]["embedding_staleness"] == 0.0
    fake_store.save_receipt.assert_called_once()


def test_tick_still_warming_writes_zero_not_a_stale_cached_score() -> None:
    """Review finding, 2026-08-19: real embeddings have arrived (not stale)
    but the scalar surprise EWMA (z-score migration) hasn't seen a second
    real magnitude yet -- surprise_n=0. Must not fall through to whatever
    self._last_perception_prediction_error happens to hold (here, a stale
    leftover None -> or 0.0 would coincidentally also be 0.0, so this test
    is really about exercising the dedicated still-warming branch, not
    about a different numeric outcome -- see the tick's own comment for why
    the published value is unavoidably 0.0 either way)."""
    worker, fake_store = _make_worker()
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_stream_id = "cam0"
    worker._last_perception_surprise_n = 0  # still warming
    worker._perception_prediction_error_tick()

    worker._write_prediction_error_node.assert_called_once()
    _, kwargs = worker._write_prediction_error_node.call_args
    assert kwargs["error"] == 0.0
    assert kwargs["extra_channels"]["embedding_staleness"] == 0.0
    fake_store.save_receipt.assert_called_once()


def test_tick_still_warming_does_not_leak_a_stale_nonzero_cached_score() -> None:
    """Stronger version of the test above: even if a stale/unrelated
    self._last_perception_prediction_error value happens to be set (e.g.
    left over from before a dimension-mismatch reseed cleared the scalar
    baseline but not this cache), the still-warming branch must publish
    0.0, not that leftover value -- proves the branch is taken on its own
    terms, not coincidentally producing the same number the old `or 0.0`
    fallback would have."""
    worker, fake_store = _make_worker()
    worker._last_perception_prediction_error = 0.9  # stale, must not leak through
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._last_perception_stream_id = "cam0"
    worker._last_perception_surprise_n = 0  # still warming
    worker._perception_prediction_error_tick()

    _, kwargs = worker._write_prediction_error_node.call_args
    assert kwargs["error"] == 0.0


def test_tick_zeroes_score_once_staleness_faults() -> None:
    """A stale reading must not outlive its own evidence -- same rule
    _vision_channel_tick applies to perceptual_yield, applied here to
    prediction_error itself. Grace=15s, saturation=60s
    (vision_channel_staleness_pressure's own constants)."""
    worker, fake_store = _make_worker()
    worker._last_perception_prediction_error = 0.9
    worker._last_perception_embedding_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    worker._perception_prediction_error_tick()

    _, kwargs = worker._write_prediction_error_node.call_args
    assert kwargs["error"] == 0.0
    assert kwargs["extra_channels"]["embedding_staleness"] == 1.0
    # Receipt is unconditional every tick (review finding, fixed): the
    # corrective 0.0 must reach orion-field-digester's field-node-vector
    # perturbation path too, or that vector gets stuck at the last nonzero
    # score forever -- gating this the way _bus_synaptic_tick's receipt is
    # gated would reproduce the exact stuck-field-vector bug
    # _vision_channel_tick's own receipt already documents fixing.
    fake_store.save_receipt.assert_called_once()
    receipt = fake_store.save_receipt.call_args.args[0]
    assert receipt.state_deltas[0].after["pressure_hints"]["prediction_error"] == 0.0


def test_tick_fails_open_on_unexpected_error() -> None:
    worker, _ = _make_worker()
    worker._last_perception_prediction_error = 0.5
    worker._last_perception_embedding_at = datetime.now(timezone.utc)
    worker._write_prediction_error_node.side_effect = RuntimeError("boom")
    worker._perception_prediction_error_tick()  # must not raise


# --- task registration -------------------------------------------------------


def test_perception_tick_loop_is_always_registered_regardless_of_flag() -> None:
    """Mirrors _vision_channel_tick_loop/_bus_synaptic_tick_loop: the clock-
    driven tick loop is unconditional in start()'s base task list -- its own
    body no-ops when the flag is off, same shape asserted by
    test_tick_disabled_is_noop above."""
    import inspect

    source = inspect.getsource(BiometricsSubstrateWorker.start)
    assert "_perception_prediction_error_tick_loop" in source
    assert "_perception_prediction_error_listener_loop" in source
    assert "enable_perception_prediction_error_tick" in source
