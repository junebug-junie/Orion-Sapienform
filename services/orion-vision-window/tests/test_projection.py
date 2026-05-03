import time
import uuid

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionObject

from app.projection import (
    artifact_uris_from_artifact,
    build_window_payload,
    envelope_to_http_dict,
    stream_key_from_artifact,
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


def test_stream_key_from_inputs():
    a = _artifact(inputs={"stream_id": "s42"})
    assert stream_key_from_artifact(a) == "s42"


def test_stream_key_fallback_device():
    a = _artifact(inputs={}, device="edge-9")
    assert stream_key_from_artifact(a) == "edge-9"


def test_artifact_uris_caps():
    a = _artifact(
        inputs={
            "image_uri": "https://example.com/x.jpg",
            "thumb": "https://example.com/t.jpg",
        }
    )
    uris = artifact_uris_from_artifact(a)
    assert "https://example.com/x.jpg" in uris


def test_build_window_payload_cursor_and_schema():
    art = _artifact(
        inputs={"camera_id": "cam-a"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="cup", score=0.9, box_xyxy=[0, 0, 1, 1])]
        ),
    )
    corr = uuid.uuid4()
    env = BaseEnvelope(
        kind="vision.artifact",
        source=ServiceRef(name="src"),
        correlation_id=corr,
    )
    now = time.time()
    p = build_window_payload(
        stream_id="cam-a",
        items=[(art, now)],
        envs=[env],
        window_start=now - 1,
        window_end=now,
        cursor="vw:000000000001:abc",
        stale_after_ms=5000,
    )
    assert p.schema_version == "vision_window_snapshot.v1"
    assert p.cursor == "vw:000000000001:abc"
    assert str(corr) in p.upstream_event_ids
    assert p.camera_id == "cam-a"


def test_envelope_to_http_dict_stale():
    art = _artifact()
    env = BaseEnvelope(
        kind="vision.artifact",
        source=ServiceRef(name="src"),
        correlation_id=uuid.uuid4(),
    )
    old = time.time() - 999.0
    p = build_window_payload(
        stream_id="default",
        items=[(art, old)],
        envs=[env],
        window_start=old,
        window_end=old,
        cursor="vw:000000000002:def",
        stale_after_ms=1000,
    )
    body = envelope_to_http_dict(p, source="live_state")
    assert body["status"] == "stale"
    assert body["source"] == "live_state"
