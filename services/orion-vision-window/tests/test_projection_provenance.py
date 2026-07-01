from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionCaption,
)

from app.projection import (
    artifact_uris_from_artifact,
    camera_id_from_artifact,
    stream_key_from_artifact,
    summarize_items,
)


def _artifact(**kwargs) -> VisionArtifactPayload:
    base = dict(
        artifact_id="a1",
        correlation_id="c1",
        task_type="detect",
        device="cam-1",
        inputs={},
        outputs=VisionArtifactOutputs(objects=[]),
        timing={},
        model_fingerprints={},
    )
    base.update(kwargs)
    return VisionArtifactPayload(**base)


def test_stream_key_prefers_stream_id_over_device():
    art = _artifact(
        inputs={
            "stream_id": "mock-stream",
            "camera_id": "mock-cam-01",
            "image_path": "/tmp/test.jpg",
        },
        device="cuda:0",
    )
    assert stream_key_from_artifact(art) == "mock-stream"


def test_camera_id_from_inputs():
    art = _artifact(
        inputs={
            "stream_id": "mock-stream",
            "camera_id": "mock-cam-01",
            "image_path": "/tmp/test.jpg",
        },
        device="cuda:0",
    )
    assert camera_id_from_artifact(art) == "mock-cam-01"


def test_artifact_uris_includes_image_path():
    art = _artifact(
        inputs={
            "stream_id": "mock-stream",
            "camera_id": "mock-cam-01",
            "image_path": "/tmp/test.jpg",
        },
        device="cuda:0",
    )
    assert "/tmp/test.jpg" in artifact_uris_from_artifact(art)


def test_summarize_items_includes_caption():
    art = _artifact(
        outputs=VisionArtifactOutputs(
            caption=VisionCaption(text="A terminal window is visible."),
        ),
    )
    result = summarize_items([(art, 1719800000.0)])
    assert "A terminal window is visible." in result["captions"]
