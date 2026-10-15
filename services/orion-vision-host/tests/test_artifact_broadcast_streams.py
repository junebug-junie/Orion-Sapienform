"""The qwen lane's artifact broadcast is off except for named camera streams
(2026-10-15, walkway captions). Dream captions carry no stream_id and must
never reach orion-vision-window's camera projection."""
from __future__ import annotations

from app.main import artifact_broadcast_allowed
from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCaption


def _art(inputs):
    return VisionArtifactPayload(
        artifact_id="a1",
        correlation_id="c1",
        task_type="caption_frame",
        device="cuda:0",
        inputs=inputs,
        outputs=VisionArtifactOutputs(caption=VisionCaption(text="A person walks a dog.", confidence=1.0)),
        timing={},
        model_fingerprints={},
    )


def test_enabled_broadcasts_everything_as_before():
    assert artifact_broadcast_allowed(_art({}), enabled=True, streams="")
    assert artifact_broadcast_allowed(_art({"stream_id": "cam0"}), enabled=True, streams="walkway")


def test_disabled_broadcasts_only_listed_streams():
    assert artifact_broadcast_allowed(_art({"stream_id": "walkway"}), enabled=False, streams="walkway")
    assert artifact_broadcast_allowed(_art({"stream_id": "walkway"}), enabled=False, streams=" cam9 , walkway ")
    assert not artifact_broadcast_allowed(_art({"stream_id": "cam0"}), enabled=False, streams="walkway")


def test_dream_caption_without_stream_never_broadcasts():
    assert not artifact_broadcast_allowed(_art({"percept_sha256": "ab" * 32}), enabled=False, streams="walkway")
    assert not artifact_broadcast_allowed(_art({"stream_id": "walkway"}), enabled=False, streams="")
    assert not artifact_broadcast_allowed(None, enabled=False, streams="walkway")
