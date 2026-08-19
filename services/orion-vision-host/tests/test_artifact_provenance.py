from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.artifacts import build_artifact_payload, merge_result_inputs
from app.models import VisionResult


def test_merge_result_inputs_combines_request_and_meta() -> None:
    request = {
        "image_path": "/tmp/test.jpg",
        "want_caption": True,
        "want_embeddings": True,
    }
    meta = {
        "camera_id": "mock-cam-01",
        "stream_id": "mock-stream",
        "frame_ts": 123.4,
        "source_frame_envelope_id": "frame-env-1",
        "source_frame_correlation_id": "frame-corr-1",
        "router_policy": "defaults",
    }
    merged = merge_result_inputs(request, meta)
    assert merged["image_path"] == "/tmp/test.jpg"
    assert merged["want_caption"] is True
    assert merged["want_embeddings"] is True
    assert merged["camera_id"] == "mock-cam-01"
    assert merged["stream_id"] == "mock-stream"
    assert merged["frame_ts"] == 123.4
    assert merged["source_frame_envelope_id"] == "frame-env-1"
    assert merged["source_frame_correlation_id"] == "frame-corr-1"
    assert merged["router_policy"] == "defaults"


def test_build_artifact_payload_preserves_request_and_meta_provenance() -> None:
    request = {
        "image_path": "/tmp/test.jpg",
        "want_caption": True,
        "want_embeddings": True,
    }
    meta = {
        "camera_id": "mock-cam-01",
        "stream_id": "mock-stream",
        "frame_ts": 123.4,
        "source_frame_envelope_id": "frame-env-1",
        "source_frame_correlation_id": "frame-corr-1",
        "router_policy": "defaults",
    }
    res = VisionResult(
        corr_id="c1",
        ok=True,
        task_type="retina_fast",
        device="cuda:0",
        artifacts={
            "objects": [{"label": "cup", "score": 0.9, "box_xyxy": [0, 0, 1, 1]}],
            "caption": {"text": "A terminal window is visible.", "confidence": 0.8},
            "embedding": {"ref": "emb:x", "path": "/tmp/e.npy", "dim": 16},
            "model_id": "caption-model",
            "_fingerprints": {"retina_detect_open_vocab": "gdino", "vlm_caption": "blip2"},
        },
        warnings=[],
        meta={"latency_s": 0.5},
        inputs=merge_result_inputs(request, meta),
    )
    art = build_artifact_payload(res)
    assert art is not None
    assert art.inputs["image_path"] == "/tmp/test.jpg"
    assert art.inputs["camera_id"] == "mock-cam-01"
    assert art.inputs["stream_id"] == "mock-stream"
    assert art.inputs["frame_ts"] == 123.4
    assert art.inputs["source_frame_envelope_id"] == "frame-env-1"
    assert art.inputs["source_frame_correlation_id"] == "frame-corr-1"
    assert art.inputs["router_policy"] == "defaults"
    assert art.inputs["want_caption"] is True
    assert art.inputs["want_embeddings"] is True
    assert art.outputs.caption.text == "A terminal window is visible."
    assert art.outputs.embedding.ref == "emb:x"
    assert art.outputs.objects[0].label == "cup"
    assert art.model_fingerprints == {"retina_detect_open_vocab": "gdino", "vlm_caption": "blip2"}


def test_build_artifact_payload_passes_embedding_vector_through() -> None:
    """2026-08-19 (P2 wire-contract patch): the inline vector field is new and
    additive -- must round-trip when the runner supplies one, and must stay
    None (not raise) when it doesn't (see
    test_build_artifact_payload_preserves_request_and_meta_provenance above,
    whose embedding dict has no "vector" key at all)."""
    res = VisionResult(
        corr_id="c2",
        ok=True,
        task_type="retina_fast",
        artifacts={
            "embedding": {"ref": "emb:y", "path": "/tmp/e2.npy", "dim": 4, "vector": [0.1, 0.2, 0.3, 0.4]},
        },
    )
    art = build_artifact_payload(res)
    assert art is not None
    assert art.outputs.embedding.vector == [0.1, 0.2, 0.3, 0.4]


def test_build_artifact_payload_none_when_empty() -> None:
    res = VisionResult(corr_id="c", ok=True, task_type="retina_fast", artifacts={})
    assert build_artifact_payload(res) is None


def test_model_fingerprints_fallback_single_profile() -> None:
    res = VisionResult(
        corr_id="c",
        ok=True,
        task_type="retina_fast",
        artifacts={"caption": {"text": "hi"}, "model_id": "m1"},
    )
    art = build_artifact_payload(res)
    assert art is not None
    assert art.model_fingerprints == {"retina_fast": "m1"}
