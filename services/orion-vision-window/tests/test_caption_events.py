"""Accepted captions reach vision_events (2026-10-16).

Live 2026-10-16 22:53 UTC the gate accepted its first caption ever ("A person
is pushing a stroller on a sidewalk ...") and nothing downstream saw it: the
council's transition gate skipped the window as stable_scene, so no
vision_events row, so no street line. These tests pin the window's own
publish of each accepted caption as a labelled camera_caption row."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionCaption,
    VisionEventPayload,
    VisionObject,
)

from app.caption_events import CAMERA_CAPTION_EVENT_TYPE, build_caption_event, caption_event_id
from app.main import WindowService, settings

STROLLER = ("A person is pushing a stroller on a sidewalk in a suburban neighborhood, "
            "with a car parked on the street and a mountain in the background.")
WALKER = [400.0, 900.0, 520.0, 1400.0]
PRAM = [520.0, 1100.0, 640.0, 1400.0]


def _detect(objects) -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id="det", correlation_id="c", task_type="retina_fast", device="cuda:0",
        inputs={"stream_id": "walkway"},
        outputs=VisionArtifactOutputs(objects=[VisionObject(label=l, score=s, box_xyxy=b) for l, s, b in objects]),
        timing={}, model_fingerprints={},
    )


def _caption(text, frame_ts, *, aid="cap-1") -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id=aid, correlation_id="c293d785", task_type="caption_frame", device="cuda:0",
        inputs={
            "stream_id": "walkway",
            "frame_ts": frame_ts,
            "image_path": "/mnt/telemetry/vision/frames/walkway_1.jpg",
            "caption_crop_boxes": [WALKER, PRAM],
            "source_objects": [
                {"label": "person", "score": 0.6, "box_xyxy": WALKER},
                {"label": "stroller", "score": 0.5, "box_xyxy": PRAM},
                {"label": "cat", "score": 0.9, "box_xyxy": WALKER},  # not a gate label
            ],
        },
        outputs=VisionArtifactOutputs(caption=VisionCaption(text=text, confidence=1.0)),
        timing={}, model_fingerprints={"vlm_caption": "Qwen/Qwen2-VL-2B-Instruct"},
    )


def _flush(svc, buffered):
    return svc._flush_and_publish(stream_id="walkway", buffered=buffered, correlation_id=None, causality_chain=[])


@pytest.mark.asyncio
async def test_accepted_caption_is_published_as_a_labelled_camera_caption_row():
    svc = WindowService()
    now = time.time()
    buffered = [
        {"artifact": _detect([("person", 0.6, WALKER), ("stroller", 0.5, PRAM)]), "ts": now - 3, "env": None},
        {"artifact": _caption(STROLLER, now - 3), "ts": now, "env": None},
    ]
    with patch.object(svc.bus, "publish", new_callable=AsyncMock) as pub:
        await _flush(svc, buffered)
    sent = [(c.args[0], c.args[1]) for c in pub.call_args_list if c.args[0] == settings.CHANNEL_CAPTION_EVENTS_PUB]
    assert len(sent) == 1
    channel, env = sent[0]
    assert channel == "orion:vision:events"
    assert env.kind == "vision.event.bundle"
    (evt,) = VisionEventPayload.model_validate(env.payload).events
    assert evt.event_type == CAMERA_CAPTION_EVENT_TYPE
    assert evt.narrative == STROLLER
    assert evt.stream_id == "walkway"
    assert evt.entities == []
    assert evt.tags == ["camera_caption", "motion_unverified", "detector:person", "detector:stroller"]
    assert "artifact:cap-1" in evt.evidence_refs
    assert "frame:/mnt/telemetry/vision/frames/walkway_1.jpg" in evt.evidence_refs
    assert "box:person:400,900,520,1400" in evt.evidence_refs
    assert "crop:520,1100,640,1400" in evt.evidence_refs
    assert "model:Qwen/Qwen2-VL-2B-Instruct" in evt.evidence_refs
    assert any(r.startswith("frame_ts:") for r in evt.evidence_refs)
    assert evt.event_id == caption_event_id("cap-1")
    assert svc._m_caption_events_published == 1


@pytest.mark.asyncio
async def test_rejected_caption_publishes_nothing():
    svc = WindowService()
    now = time.time()
    art = _caption(STROLLER, now - 3)
    art.inputs["source_objects"] = []  # detector saw no gate object
    with patch.object(svc.bus, "publish", new_callable=AsyncMock) as pub:
        await _flush(svc, [{"artifact": art, "ts": now, "env": None}])
    assert not [c for c in pub.call_args_list if c.args[0] == settings.CHANNEL_CAPTION_EVENTS_PUB]
    assert svc._m_caption_events_published == 0


@pytest.mark.asyncio
async def test_disabled_flag_publishes_nothing_but_still_logs(monkeypatch):
    monkeypatch.setattr(settings, "WINDOW_CAPTION_EVENTS_ENABLED", False)
    svc = WindowService()
    now = time.time()
    logged = []
    from app import main as main_mod

    monkeypatch.setattr(main_mod.logger, "info", lambda msg, *a, **k: logged.append(str(msg)))
    with patch.object(svc.bus, "publish", new_callable=AsyncMock) as pub:
        await _flush(svc, [{"artifact": _caption(STROLLER, now - 3), "ts": now, "env": None}])
    assert not [c for c in pub.call_args_list if c.args[0] == settings.CHANNEL_CAPTION_EVENTS_PUB]
    (line,) = [m for m in logged if "caption_accepted" in m]
    assert "labels=person,stroller" in line and "A person is pushing a stroller" in line


@pytest.mark.asyncio
async def test_publish_failure_is_counted_and_never_fails_the_flush():
    svc = WindowService()
    now = time.time()

    async def _pub(channel, env):
        if channel == settings.CHANNEL_CAPTION_EVENTS_PUB:
            raise RuntimeError("bus down")

    with patch.object(svc.bus, "publish", side_effect=_pub):
        await _flush(svc, [{"artifact": _caption(STROLLER, now - 3), "ts": now, "env": None}])
    assert svc._m_caption_events_failed == 1
    assert svc._live_by_stream["walkway"].summary["captions"] == [STROLLER]


def test_blank_caption_builds_no_event():
    assert build_caption_event({"artifact_id": "a", "text": "   "}, stream_id="walkway") is None


def test_stationary_box_is_not_named_as_the_reason_it_passed():
    from app.caption_gate import CaptionGate

    gate = CaptionGate(streams="walkway", labels="person,stroller", min_score=0.25, max_age_sec=20.0,
                       stationary_min_sec=1800.0, stationary_iou=0.5, stationary_max_gap_sec=300.0)
    skeleton = [2060.0, 877.0, 2274.0, 1227.0]
    t0 = time.time() - 2400
    for k in range(105):
        gate.apply("walkway", [(_detect([("person", 0.37, skeleton)]), t0 + 20.0 * k)])
    now = t0 + 2100
    art = _caption(STROLLER, now)
    art.inputs["source_objects"] = [
        {"label": "person", "score": 0.37, "box_xyxy": skeleton},
        {"label": "stroller", "score": 0.5, "box_xyxy": PRAM},
    ]
    accepted: list = []
    gate.apply("walkway", [(art, now + 1)], accepted_out=accepted)
    (rec,) = accepted
    assert rec["labels"] == ["stroller"]
