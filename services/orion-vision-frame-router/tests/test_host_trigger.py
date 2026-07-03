from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionObject,
    VisionTaskResultPayload,
)

from app.host_trigger import extract_host_trigger_labels, stream_id_from_host_result


def _result(*, stream_id: str = "cam0", labels: list[tuple[str, float]]) -> VisionTaskResultPayload:
    objects = [
        VisionObject(label=label, score=score, box_xyxy=[0, 0, 1, 1])
        for label, score in labels
    ]
    artifact = VisionArtifactPayload(
        artifact_id="a1",
        correlation_id="c1",
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": stream_id, "camera_id": "rtsp://cam"},
        outputs=VisionArtifactOutputs(objects=objects),
        timing={},
        model_fingerprints={},
    )
    return VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=artifact)


def test_extract_host_trigger_labels_person_above_threshold() -> None:
    result = _result(labels=[("person", 0.9), ("door", 0.8)])
    labels = extract_host_trigger_labels(result, allowed={"person"}, score_threshold=0.25)
    assert labels == ["person"]


def test_extract_host_trigger_labels_ignores_low_score() -> None:
    result = _result(labels=[("person", 0.1)])
    labels = extract_host_trigger_labels(result, allowed={"person"}, score_threshold=0.25)
    assert labels == []


def test_extract_host_trigger_labels_empty_when_no_artifact() -> None:
    result = VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=None)
    assert extract_host_trigger_labels(result, allowed={"person"}) == []


def test_stream_id_from_host_result_prefers_artifact_inputs() -> None:
    result = _result(stream_id="cam0", labels=[("person", 0.9)])
    assert stream_id_from_host_result(result, fallback_stream_id="cam99") == "cam0"


def test_stream_id_from_host_result_uses_fallback() -> None:
    result = VisionTaskResultPayload(ok=True, task_type="retina_fast", artifact=None)
    assert stream_id_from_host_result(result, fallback_stream_id="cam99") == "cam99"
