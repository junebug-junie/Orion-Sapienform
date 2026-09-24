"""Host artifact -> VisionCropObservationV1 (walkway individuals, idea 1)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload, VisionCropObservationV1, VisionObject

from app import main as app_main
from app.crops import build_crop_observation
from app.main import WindowService

VEC = [0.6, 0.8]


def _artifact(objects, *, inputs=None, extra=None) -> VisionArtifactPayload:
    outputs = VisionArtifactOutputs(objects=objects, **(extra or {}))
    return VisionArtifactPayload(
        artifact_id="art-1",
        correlation_id="00000000-0000-0000-0000-000000000001",
        task_type="retina_fast",
        device="cuda:0",
        inputs=inputs if inputs is not None else {"stream_id": "walkway", "camera_id": "walkway", "frame_ts": 1_700_000_000.0},
        outputs=outputs,
        timing={},
        model_fingerprints={},
    )


def _objs():
    return [
        VisionObject(label="person", score=0.9, box_xyxy=[1, 2, 3, 4], zone="walkway", embedding_ref="crop:x", embedding=VEC,
                     thumb_ref="thumb:" + "ab" * 32),
        VisionObject(label="person", score=0.8, box_xyxy=[5, 6, 7, 8], zone="patio"),
        VisionObject(label="chair", score=0.9, box_xyxy=[1, 1, 2, 2]),
    ]


def test_tracked_boxes_become_crops_patio_without_vector() -> None:
    obs = build_crop_observation(_artifact(_objs(), extra={"frame_width": 640, "frame_height": 360}))
    assert isinstance(obs, VisionCropObservationV1)
    assert obs.stream_id == "walkway" and obs.artifact_id == "art-1"
    assert obs.observation_id == "cropobs:art-1"
    assert (obs.frame_width, obs.frame_height) == (640, 360)
    assert obs.observed_at.timestamp() == 1_700_000_000.0
    zones = {c.zone: c for c in obs.crops}
    assert set(zones) == {"walkway", "patio"}  # chair not tracked
    assert zones["patio"].embedding is None
    assert zones["walkway"].embedding == VEC
    assert zones["walkway"].thumb_ref == "thumb:" + "ab" * 32  # the ask card's picture rides along
    assert zones["patio"].thumb_ref is None
    # round-trips through the registered contract
    VisionCropObservationV1.model_validate(obs.model_dump(mode="json"))


def test_untracked_artifact_publishes_nothing() -> None:
    art = _artifact([VisionObject(label="person", score=0.9, box_xyxy=[1, 2, 3, 4])], inputs={"stream_id": "cam0"})
    assert build_crop_observation(art) is None


def test_frame_size_falls_back_to_inputs() -> None:
    obs = build_crop_observation(_artifact(_objs(), inputs={"stream_id": "walkway", "width": 1280, "height": 720}))
    assert (obs.frame_width, obs.frame_height) == (1280, 720)


def test_rtsp_camera_id_never_forwarded() -> None:
    inputs = {"stream_id": "walkway", "camera_id": "rtsp://admin:secret@10.0.0.2:554/x"}
    obs = build_crop_observation(_artifact(_objs(), inputs=inputs))
    dumped = obs.model_dump_json()
    assert "secret" not in dumped and "rtsp://" not in dumped
    assert obs.stream_id == "walkway" and obs.camera_id is None


def test_ingest_publishes_on_crops_channel(monkeypatch) -> None:
    svc = WindowService()
    svc.bus = AsyncMock()
    env = BaseEnvelope(kind="vision.artifact", source=ServiceRef(name="t"), payload={})
    asyncio.run(svc._publish_crop_observation(_artifact(_objs()), env))
    svc.bus.publish.assert_awaited_once()
    channel, sent = svc.bus.publish.await_args.args
    assert channel == "orion:vision:crops:sql-write"
    assert sent.kind == "vision.crop.observation.v1"
    VisionCropObservationV1.model_validate(sent.payload)
    assert svc._m_crop_obs_published == 1

    monkeypatch.setattr(app_main.settings, "WINDOW_CROP_OBSERVATIONS_ENABLED", False)
    svc.bus.publish.reset_mock()
    asyncio.run(svc._publish_crop_observation(_artifact(_objs()), env))
    svc.bus.publish.assert_not_awaited()


def test_projection_never_uses_rtsp_camera_id_as_camera_or_stream_key() -> None:
    """vision_scene_inventory.camera_id carried the RTSP URL (password) in
    ~316k rows before 2026-09-24; the window's camera/stream helpers drop it."""
    from app.projection import camera_id_from_artifact, stream_key_from_artifact

    art = _artifact(_objs(), inputs={"camera_id": "rtsp://admin:pw@10.0.0.2/x"})
    assert camera_id_from_artifact(art) is None
    assert "rtsp://" not in stream_key_from_artifact(art)
    art2 = _artifact(_objs(), inputs={"camera_id": "cam0", "stream_id": "cam0"})
    assert camera_id_from_artifact(art2) == "cam0"
