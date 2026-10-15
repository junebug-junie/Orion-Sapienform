"""Walkway captions come from the Qwen2-VL lane, not the athena host's BLIP
(2026-10-15). Driven by the REAL config/vision_frame_router.yaml so a config
edit that turns BLIP back on, or drops caption_dispatch, fails here."""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import (
    VisionArtifactOutputs,
    VisionArtifactPayload,
    VisionCaption,
    VisionFramePointerPayload,
    VisionObject,
    VisionTaskResultPayload,
)

from app.dispatcher import FrameDispatcher
from app.metrics import RouterMetrics
from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState

REAL_POLICY = Path(__file__).resolve().parents[3] / "config" / "vision_frame_router.yaml"
CAPTION_CHANNEL = "orion:exec:request:VisionHostService:circe-vl"
HOST_CHANNEL = "orion:exec:request:VisionHostService"
_FRAMES = Path(tempfile.mkdtemp(prefix="walkway_caption_test_"))
atexit.register(shutil.rmtree, _FRAMES, ignore_errors=True)


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, BaseEnvelope]] = []

    async def publish(self, channel: str, envelope: BaseEnvelope) -> None:
        self.published.append((channel, envelope))


def _make(*, dry_run: bool = False) -> tuple[FrameDispatcher, FakeBus]:
    settings = Settings(ROUTER_POLICY_PATH=str(REAL_POLICY), REQUIRE_IMAGE_PATH_EXISTS=False, DRY_RUN=dry_run)
    bus = FakeBus()
    d = FrameDispatcher(
        settings=settings,
        policy=FrameDispatchPolicy.load(settings),
        state=RouterState(),
        metrics=RouterMetrics(),
        bus=bus,
    )
    return d, bus


def _ready(d: FrameDispatcher, stream: str) -> None:
    """Make the next frame of `stream` pass sampling and the per-camera clock
    (both tiers inherit every_n_frames from baseline)."""
    cam = d.state.camera(stream)
    cam.frames_seen = -1  # mark_seen -> 0, and 0 % every_n == 0
    cam.last_dispatch_ts = None


def _frame_env(stream: str = "walkway") -> BaseEnvelope:
    # The real policy has global.require_image_path_exists: true.
    path = _FRAMES / f"{stream}_{uuid4().hex[:6]}.jpg"
    path.write_bytes(b"jpg")
    payload = VisionFramePointerPayload(
        image_path=str(path),
        camera_id=stream,
        stream_id=stream,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.1.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def _reply(corr: str, *, stream: str = "walkway", objects=None, caption=None, ok=True, error_code=None,
           task_type="retina_fast") -> BaseEnvelope:
    art = None
    if ok:
        art = VisionArtifactPayload(
            artifact_id=uuid4().hex,
            correlation_id=corr,
            task_type=task_type,
            device="cuda:0",
            inputs={"stream_id": stream, "camera_id": stream},
            outputs=VisionArtifactOutputs(objects=objects, caption=caption),
            timing={},
            model_fingerprints={},
        )
    res = VisionTaskResultPayload(ok=ok, task_type=task_type, artifact=art, error_code=error_code)
    return BaseEnvelope(
        kind="vision.task.result",
        source=ServiceRef(name="vision-host", version="0.1.0"),
        correlation_id=corr,
        payload=res.model_dump(mode="json"),
    )


async def _triggered_primary(d: FrameDispatcher, bus: FakeBus, stream: str = "walkway") -> tuple[str, BaseEnvelope]:
    d.state.record_activity(stream, ["person"], now=time.time())
    _ready(d, stream)
    await d.handle_frame_envelope(_frame_env(stream))
    channel, env = bus.published[-1]
    assert channel == HOST_CHANNEL
    return str(env.correlation_id), env


PERSON = VisionObject(label="person", score=0.62, box_xyxy=[100, 200, 180, 420])


def test_real_config_turns_blip_off_for_walkway_only():
    policy = FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(REAL_POLICY)))
    walkway = policy.resolve_stream_policy("walkway", "walkway")[0]
    assert walkway["triggered"]["request"]["want_caption"] is False
    assert walkway["baseline"]["request"]["want_caption"] is False
    cfg = policy.caption_dispatch_for("walkway", "walkway")
    assert cfg["enabled"] is True
    assert "nothing notable" in cfg["prompt"] and cfg["null_answers"] == ["nothing notable"]
    assert set(cfg["labels"]) >= {"person", "dog", "bicycle", "stroller", "vehicle", "package", "mail truck"}
    # cam0/carbon unchanged: BLIP caption, no VLM lane.
    for stream in ("cam0", "carbon"):
        merged = policy.resolve_stream_policy(stream, stream)[0]
        assert merged["triggered"]["request"]["want_caption"] is True
        assert policy.caption_dispatch_for(stream, stream) == {}


@pytest.mark.asyncio
async def test_walkway_primary_asks_no_caption_from_host():
    d, bus = _make()
    _, env = await _triggered_primary(d, bus)
    assert env.payload["request"]["want_caption"] is False


@pytest.mark.asyncio
async def test_detection_in_reply_sends_caption_task_to_qwen_lane():
    d, bus = _make()
    corr, _ = await _triggered_primary(d, bus)
    await d.handle_reply_envelope(_reply(corr, objects=[PERSON, VisionObject(label="chair", score=0.9, box_xyxy=[0, 0, 1, 1])]))

    channel, env = bus.published[-1]
    assert channel == CAPTION_CHANNEL
    task = env.payload
    assert task["task_type"] == "caption_frame"
    assert Path(task["request"]["image_path"]).name.startswith("walkway_")
    assert "nothing notable" in task["request"]["caption_prompt"]
    assert task["request"]["caption_null_answers"] == ["nothing notable"]
    assert task["meta"]["stream_id"] == "walkway"
    assert task["meta"]["source_objects"] == [{"label": "person", "score": 0.62, "box_xyxy": [100.0, 200.0, 180.0, 420.0]}]
    assert str(env.correlation_id) != corr and env.reply_to.endswith(str(env.correlation_id))
    assert d.metrics.caption_dispatched_total == 1
    # Own lane: does not consume an athena host slot.
    assert d.state.inflight_total() == 0
    assert d.state.inflight_lane("caption") == 1


@pytest.mark.asyncio
async def test_empty_dark_frame_never_asks_the_vlm():
    d, bus = _make()
    corr, _ = await _triggered_primary(d, bus)
    n = len(bus.published)
    await d.handle_reply_envelope(_reply(corr, objects=[]))
    assert len(bus.published) == n
    assert d.metrics.caption_skip_counts == {"no_detection": 1}


@pytest.mark.asyncio
async def test_low_score_or_non_gate_label_does_not_count():
    d, bus = _make()
    corr, _ = await _triggered_primary(d, bus)
    n = len(bus.published)
    await d.handle_reply_envelope(_reply(corr, objects=[
        VisionObject(label="person", score=0.1, box_xyxy=[0, 0, 1, 1]),
        VisionObject(label="chair", score=0.9, box_xyxy=[0, 0, 1, 1]),
    ]))
    assert len(bus.published) == n


@pytest.mark.asyncio
async def test_baseline_tier_reply_does_not_caption():
    d, bus = _make()
    _ready(d, "walkway")
    await d.handle_frame_envelope(_frame_env())  # no activity -> baseline
    corr = str(bus.published[-1][1].correlation_id)
    await d.handle_reply_envelope(_reply(corr, objects=[PERSON]))
    assert all(ch == HOST_CHANNEL for ch, _ in bus.published)


@pytest.mark.asyncio
async def test_cam0_never_uses_the_qwen_lane():
    d, bus = _make()
    corr, env = await _triggered_primary(d, bus, stream="cam0")
    assert env.payload["request"]["want_caption"] is True
    await d.handle_reply_envelope(_reply(corr, stream="cam0", objects=[PERSON]))
    assert all(ch != CAPTION_CHANNEL for ch, _ in bus.published)


@pytest.mark.asyncio
async def test_rate_limit_and_lane_cap():
    d, bus = _make()
    corr1, _ = await _triggered_primary(d, bus)
    await d.handle_reply_envelope(_reply(corr1, objects=[PERSON]))
    corr2, _ = await _triggered_primary(d, bus)
    await d.handle_reply_envelope(_reply(corr2, objects=[PERSON]))
    assert d.metrics.caption_dispatched_total == 1
    assert d.metrics.caption_skip_counts == {"rate_limited": 1}

    # Clock past the rate limit, first caption still in flight -> lane cap.
    d.state.camera("walkway").last_caption_dispatch_ts = time.time() - 60
    corr3, _ = await _triggered_primary(d, bus)
    await d.handle_reply_envelope(_reply(corr3, objects=[PERSON]))
    assert d.metrics.caption_skip_counts == {"rate_limited": 1, "lane_inflight_limit": 1}


@pytest.mark.asyncio
async def test_caption_lane_outcomes_are_counted_by_reason():
    d, bus = _make()

    async def one_caption() -> str:
        d.state.camera("walkway").last_caption_dispatch_ts = None
        corr, _ = await _triggered_primary(d, bus)
        await d.handle_reply_envelope(_reply(corr, objects=[PERSON]))
        return str(bus.published[-1][1].correlation_id)

    c = await one_caption()
    await d.handle_reply_envelope(_reply(c, task_type="caption_frame",
                                         caption=VisionCaption(text="A man walks a dog along the sidewalk.", confidence=1.0)))
    c = await one_caption()
    await d.handle_reply_envelope(_reply(c, task_type="caption_frame",
                                         caption=VisionCaption(text="", confidence=0.0, rejected_reason="null_answer")))
    c = await one_caption()
    await d.handle_reply_envelope(_reply(c, task_type="caption_frame", ok=False, error_code="queue_full"))

    assert d.metrics.caption_captioned_total == 1
    assert d.metrics.caption_no_caption_counts == {"null_answer": 1}
    assert d.metrics.caption_failure_counts == {"queue_full": 1}
    assert d.state.inflight_lane("caption") == 0
    # A caption-lane failure is not an athena host failure.
    assert d.metrics.host_errors_total == 0


@pytest.mark.asyncio
async def test_caption_lane_timeout_uses_its_own_timeout_and_counter():
    d, bus = _make()
    corr, _ = await _triggered_primary(d, bus)
    await d.handle_reply_envelope(_reply(corr, objects=[PERSON]))
    pending = next(t for t in d.state.pending.values() if t.lane == "caption")
    assert pending.timeout_s == 15
    cleared = await d.sweep_timeouts(now=pending.dispatched_at + 16)
    assert cleared == 1
    assert d.metrics.caption_failure_counts == {"timeout": 1}
    assert d.metrics.host_timeouts_total == 0


@pytest.mark.asyncio
async def test_dry_run_books_but_does_not_publish_caption():
    d, bus = _make(dry_run=True)
    d.state.record_activity("walkway", ["person"], now=time.time())
    _ready(d, "walkway")
    await d.handle_frame_envelope(_frame_env())
    corr = next(iter(d.state.pending))
    await d.handle_reply_envelope(_reply(corr, objects=[PERSON]))
    assert bus.published == []
    assert d.metrics.caption_dispatched_total == 1
