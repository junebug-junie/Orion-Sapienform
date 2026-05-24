from __future__ import annotations

import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload, VisionTaskResultPayload

from app.dispatcher import FrameDispatcher
from app.metrics import RouterMetrics
from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState


class FakeBus:
    def __init__(self, *, fail_publish: bool = False) -> None:
        self.published: list[tuple[str, object]] = []
        self.fail_publish = fail_publish

    async def publish(self, channel: str, envelope: object) -> None:
        if self.fail_publish:
            raise RuntimeError("publish failed")
        self.published.append((channel, envelope))


@pytest.fixture
def policy_path(tmp_path: Path) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 1
  request: {}
global:
  max_inflight_total: 2
  drop_when_busy: true
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    return p


def _make_dispatcher(
    policy_path: Path,
    *,
    dry_run: bool = False,
    every_n_frames: int = 1,
    fail_publish: bool = False,
) -> tuple[FrameDispatcher, FakeBus, Settings]:
    p = policy_path
    if every_n_frames != 1:
        text = p.read_text(encoding="utf-8").replace("every_n_frames: 1", f"every_n_frames: {every_n_frames}")
        p.write_text(text, encoding="utf-8")
    settings = Settings(
        ROUTER_POLICY_PATH=str(p),
        REQUIRE_IMAGE_PATH_EXISTS=False,
        DRY_RUN=dry_run,
    )
    bus = FakeBus(fail_publish=fail_publish)
    policy = FrameDispatchPolicy.load(settings)
    dispatcher = FrameDispatcher(
        settings=settings,
        policy=policy,
        state=RouterState(),
        metrics=RouterMetrics(),
        bus=bus,
    )
    return dispatcher, bus, settings


def _frame_env(
    *,
    camera_id: str = "cam1",
    image_path: str = "/tmp/f.jpg",
    correlation_id: object | None = None,
) -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path=image_path,
        camera_id=camera_id,
        frame_ts=time.time(),
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.1.0"),
        correlation_id=correlation_id or uuid4(),
        payload=payload.model_dump(mode="json"),
    )


@pytest.mark.asyncio
async def test_valid_frame_publishes_host_task(policy_path: Path) -> None:
    dispatcher, bus, settings = _make_dispatcher(policy_path)
    env = _frame_env()
    await dispatcher.handle_frame_envelope(env)
    assert len(bus.published) == 1
    channel, envelope = bus.published[0]
    assert channel == settings.CHANNEL_HOST_INTAKE
    assert envelope.kind == "vision.task.request"
    assert envelope.reply_to == f"{settings.CHANNEL_REPLY_PREFIX}:{env.correlation_id}"
    assert envelope.correlation_id == env.correlation_id
    assert dispatcher.metrics.frames_dispatched_total == 1


@pytest.mark.asyncio
async def test_sampled_out_frame_publishes_nothing(policy_path: Path) -> None:
    dispatcher, bus, _ = _make_dispatcher(policy_path, every_n_frames=2)
    await dispatcher.handle_frame_envelope(_frame_env())
    assert bus.published == []
    assert dispatcher.metrics.frames_skipped_total == 1
    assert dispatcher.metrics.skip_reason_counts.get("frame_sampled_out") == 1


@pytest.mark.asyncio
async def test_dry_run_records_without_publish(policy_path: Path) -> None:
    dispatcher, bus, _ = _make_dispatcher(policy_path, dry_run=True)
    await dispatcher.handle_frame_envelope(_frame_env())
    assert bus.published == []
    assert dispatcher.metrics.frames_dispatched_total == 1
    assert dispatcher.state.inflight_total() == 1


@pytest.mark.asyncio
async def test_invalid_payload_increments_error(policy_path: Path) -> None:
    dispatcher, bus, _ = _make_dispatcher(policy_path)
    env = BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-retina", version="0.1.0"),
        correlation_id=uuid4(),
        payload={"not_a_valid_frame": True},
    )
    await dispatcher.handle_frame_envelope(env)
    assert bus.published == []
    assert dispatcher.metrics.last_error is not None
    assert "invalid_frame_payload" in dispatcher.metrics.last_error


@pytest.mark.asyncio
async def test_reply_clears_pending(policy_path: Path) -> None:
    dispatcher, bus, _ = _make_dispatcher(policy_path)
    corr = uuid4()
    await dispatcher.handle_frame_envelope(_frame_env(correlation_id=corr))
    assert dispatcher.state.inflight_total() == 1
    reply = BaseEnvelope(
        kind="vision.task.result",
        source=ServiceRef(name="vision-host", version="0.1.0"),
        correlation_id=corr,
        payload=VisionTaskResultPayload(ok=True, task_type="retina_fast").model_dump(mode="json"),
    )
    await dispatcher.handle_reply_envelope(reply)
    assert dispatcher.state.inflight_total() == 0
    assert dispatcher.metrics.host_replies_total == 1


@pytest.mark.asyncio
async def test_timeout_clears_pending(policy_path: Path) -> None:
    dispatcher, bus, settings = _make_dispatcher(policy_path)
    corr = uuid4()
    await dispatcher.handle_frame_envelope(_frame_env(correlation_id=corr))
    assert dispatcher.state.inflight_total() == 1
    cleared = await dispatcher.sweep_timeouts(now=time.time() + settings.TASK_TIMEOUT_SECONDS + 1)
    assert cleared == 1
    assert dispatcher.state.inflight_total() == 0
    assert dispatcher.metrics.host_timeouts_total == 1


@pytest.mark.asyncio
async def test_reply_before_timeout_does_not_count_timeout(policy_path: Path) -> None:
    dispatcher, _, settings = _make_dispatcher(policy_path)
    corr = uuid4()
    await dispatcher.handle_frame_envelope(_frame_env(correlation_id=corr))
    reply = BaseEnvelope(
        kind="vision.task.result",
        source=ServiceRef(name="vision-host", version="0.1.0"),
        correlation_id=corr,
        payload=VisionTaskResultPayload(ok=True, task_type="retina_fast").model_dump(mode="json"),
    )
    await dispatcher.handle_reply_envelope(reply)
    cleared = await dispatcher.sweep_timeouts(now=time.time() + settings.TASK_TIMEOUT_SECONDS + 1)
    assert cleared == 0
    assert dispatcher.metrics.host_timeouts_total == 0


@pytest.mark.asyncio
async def test_publish_failure_does_not_mark_inflight(policy_path: Path) -> None:
    dispatcher, bus, _ = _make_dispatcher(policy_path, fail_publish=True)
    await dispatcher.handle_frame_envelope(_frame_env())
    assert bus.published == []
    assert dispatcher.state.inflight_total() == 0
    assert dispatcher.metrics.frames_dispatched_total == 0
    assert dispatcher.metrics.last_error is not None
    assert "frame_handler_error" in dispatcher.metrics.last_error
