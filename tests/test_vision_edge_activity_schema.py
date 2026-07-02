from orion.schemas.registry import SCHEMA_REGISTRY
from orion.schemas.vision import VisionEdgeActivityPayload


def test_vision_edge_activity_payload_roundtrip() -> None:
    payload = VisionEdgeActivityPayload(
        stream_id="cam0",
        camera_id="rtsp://192.168.1.21/Preview_01_sub",
        labels=["person", "motion"],
        max_score=0.82,
        frame_ts=1783025641.853,
        image_path="/mnt/telemetry/vision/frames/frame_x.jpg",
        artifact_id="art-1",
    )
    data = payload.model_dump(mode="json")
    restored = VisionEdgeActivityPayload.model_validate(data)
    assert restored.labels == ["person", "motion"]
    assert restored.stream_id == "cam0"


def test_registry_has_edge_activity_kind() -> None:
    assert "VisionEdgeActivityPayload" in SCHEMA_REGISTRY
    assert SCHEMA_REGISTRY["VisionEdgeActivityPayload"].kind == "vision.edge.activity.v1"
