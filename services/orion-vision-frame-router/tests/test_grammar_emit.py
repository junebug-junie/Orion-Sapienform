"""The router's vision organ self-report (app/grammar_emit.py)."""

from __future__ import annotations

import sys
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
from orion.schemas.vision_organ_projection import (
    ROLE_STREAM_WINDOW,
    ROLE_WINDOW_COMPLETED,
    VISION_ORGAN_SOURCE_SERVICE,
    VISION_ORGAN_TRACE_PREFIX,
)

from app import grammar_emit
from app.dispatcher import FrameDispatcher
from app.metrics import RouterMetrics
from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState


class _Clock:
    def __init__(self, t: float = 1_790_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _summary_kv(summary: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in summary.split() if "=" in part)


def test_source_service_literal_matches_contract() -> None:
    assert grammar_emit.SOURCE_SERVICE == VISION_ORGAN_SOURCE_SERVICE


def test_configured_silent_stream_is_reported_absent_not_calm() -> None:
    """carbon is in the policy file and has sent nothing: it must still appear,
    with no frame age (absent), not be omitted or reported as age 0."""
    clock = _Clock()
    rec = grammar_emit.OrganWindowRecorder(configured_streams=["cam0", "carbon"], clock=clock)
    clock.t += 1.0
    rec.record_frame("cam0")
    clock.t += 59.0
    events = grammar_emit.build_window_events(router="vision-frame-router", snapshot=rec.drain())

    roles = [e.atom.semantic_role for e in events]
    assert roles == [ROLE_STREAM_WINDOW, ROLE_STREAM_WINDOW, ROLE_WINDOW_COMPLETED]
    by_stream = {e.atom.text_value: _summary_kv(e.atom.summary) for e in events[:-1]}
    assert by_stream["cam0"]["frames"] == "1"
    assert by_stream["cam0"]["last_frame_age_sec"] == "59.0"
    assert by_stream["carbon"]["frames"] == "0"
    assert by_stream["carbon"]["last_frame_age_sec"] == "none"
    assert by_stream["carbon"]["configured"] == "1"
    assert by_stream["carbon"]["uptime_sec"] == "60.0"
    assert all(e.trace_id.startswith(VISION_ORGAN_TRACE_PREFIX) for e in events)
    assert all(e.provenance.source_service == VISION_ORGAN_SOURCE_SERVICE for e in events)


def test_closing_atom_is_sent_when_nothing_happened() -> None:
    rec = grammar_emit.OrganWindowRecorder(configured_streams=[], clock=_Clock())
    events = grammar_emit.build_window_events(router="r", snapshot=rec.drain())
    assert [e.atom.semantic_role for e in events] == [ROLE_WINDOW_COMPLETED]
    assert "streams=0" in events[0].atom.summary


def test_frame_age_spans_windows_and_counts_reset() -> None:
    clock = _Clock()
    rec = grammar_emit.OrganWindowRecorder(configured_streams=["cam0"], clock=clock)
    rec.record_frame("cam0")
    clock.t += 60.0
    rec.drain()
    clock.t += 60.0
    events = grammar_emit.build_window_events(router="r", snapshot=rec.drain())
    kv = _summary_kv(events[0].atom.summary)
    assert kv["frames"] == "0"
    assert kv["last_frame_age_sec"] == "120.0"


def test_failure_classes_and_yield_are_bounded_tokens() -> None:
    rec = grammar_emit.OrganWindowRecorder(configured_streams=["cam0"], clock=_Clock())
    rec.record_dispatch("cam0")
    rec.record_dispatch("cam0", identity=True)
    rec.record_failure("cam0", "timeout")
    rec.record_failure("cam0", "Model Load Failed; drop table")
    rec.record_reply_ok("cam0", primary=True, objects=4, caption_requested=True, caption_present=False)
    rec.record_reply_ok("cam0", primary=False, objects=None, caption_requested=False, caption_present=False)
    rec.record_skip("cam0", "frame_sampled_out")
    events = grammar_emit.build_window_events(router="r", snapshot=rec.drain())
    kv = _summary_kv(events[0].atom.summary)
    assert kv["dispatched"] == "1"
    assert kv["identity_dispatched"] == "1"
    assert kv["failed"] == "2"
    assert kv["replies_ok"] == "2"
    assert kv["failure_classes"] == "modelloadfaileddroptable:1|timeout:1"
    assert kv["detect_replies"] == "1"
    assert kv["objects"] == "4"
    assert kv["caption_requested"] == "1"
    assert kv["captions"] == "0"
    assert kv["skips"] == "frame_sampled_out:1"


def test_embed_only_reply_is_not_an_empty_detection() -> None:
    rec = grammar_emit.OrganWindowRecorder(configured_streams=["cam0"], clock=_Clock())
    rec.record_reply_ok("cam0", primary=True, objects=None, caption_requested=False, caption_present=False)
    kv = _summary_kv(grammar_emit.build_window_events(router="r", snapshot=rec.drain())[0].atom.summary)
    assert kv["detect_replies"] == "0"
    assert kv["replies_ok"] == "1"


def test_stream_count_is_bounded() -> None:
    rec = grammar_emit.OrganWindowRecorder(configured_streams=[], clock=_Clock())
    for i in range(40):
        rec.record_frame(f"cam{i}")
    snap = rec.drain()
    assert len(set(snap.buckets) | set(snap.last_frame_at)) <= 17  # 16 + "other"


# ---- dispatcher integration -------------------------------------------------


@pytest.fixture
def organ():
    rec = grammar_emit.install_recorder(["cam1", "carbon"])
    yield rec
    grammar_emit.reset_recorder_for_tests()


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def publish(self, channel: str, envelope: object) -> None:
        self.published.append((channel, envelope))


def _dispatcher(tmp_path: Path) -> FrameDispatcher:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
version: 1
defaults:
  enabled: true
  task_type: retina_fast
  every_n_frames: 1
  min_seconds_between_tasks_per_camera: 0
  max_inflight_per_camera: 5
  request: {want_caption: true}
global:
  max_inflight_total: 10
  drop_when_busy: true
  require_image_path_exists: false
cameras: {}
""",
        encoding="utf-8",
    )
    settings = Settings(ROUTER_POLICY_PATH=str(p), REQUIRE_IMAGE_PATH_EXISTS=False, TASK_TIMEOUT_SECONDS=5.0)
    return FrameDispatcher(
        settings=settings,
        policy=FrameDispatchPolicy.load(settings),
        state=RouterState(),
        metrics=RouterMetrics(),
        bus=_FakeBus(),
    )


def _frame(corr) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.1.0"),
        correlation_id=corr,
        payload=VisionFramePointerPayload(
            image_path="/tmp/f.jpg", camera_id="cam1", stream_id="cam1", frame_ts=time.time()
        ).model_dump(mode="json"),
    )


def _reply(corr, result: VisionTaskResultPayload) -> BaseEnvelope:
    return BaseEnvelope(
        kind="vision.task.result",
        source=ServiceRef(name="vision-host", version="0.1.0"),
        correlation_id=corr,
        payload=result.model_dump(mode="json"),
    )


def _artifact(corr, *, objects: int, caption: str | None) -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id="a",
        correlation_id=str(corr),
        task_type="retina_fast",
        device="cuda:0",
        inputs={"stream_id": "cam1"},
        outputs=VisionArtifactOutputs(
            objects=[VisionObject(label="chair", score=0.9, box_xyxy=[0, 0, 1, 1]) for _ in range(objects)],
            caption=VisionCaption(text=caption) if caption is not None else None,
        ),
        timing={},
        model_fingerprints={},
    )


@pytest.mark.asyncio
async def test_dispatcher_feeds_the_recorder(tmp_path: Path, organ) -> None:
    d = _dispatcher(tmp_path)
    ok_corr, fail_corr, lost_corr = uuid4(), uuid4(), uuid4()
    for corr in (ok_corr, fail_corr, lost_corr):
        await d.handle_frame_envelope(_frame(corr))
    await d.handle_reply_envelope(
        _reply(ok_corr, VisionTaskResultPayload(ok=True, task_type="retina_fast",
                                                artifact=_artifact(ok_corr, objects=3, caption="a room")))
    )
    await d.handle_reply_envelope(
        _reply(fail_corr, VisionTaskResultPayload(ok=False, task_type="retina_fast", error_code="model_oom"))
    )
    await d.sweep_timeouts(now=time.time() + 60.0)

    events = grammar_emit.build_window_events(router="r", snapshot=organ.drain())
    by_stream = {e.atom.text_value: _summary_kv(e.atom.summary) for e in events[:-1]}
    cam1 = by_stream["cam1"]
    assert cam1["frames"] == "3"
    assert cam1["dispatched"] == "3"
    assert cam1["replies_ok"] == "1"
    assert cam1["failed"] == "2"
    assert cam1["failure_classes"] == "model_oom:1|timeout:1"
    assert cam1["objects"] == "3"
    assert cam1["caption_requested"] == "1"
    assert cam1["captions"] == "1"
    # configured, silent
    assert by_stream["carbon"]["last_frame_age_sec"] == "none"


@pytest.mark.asyncio
async def test_flag_off_records_nothing(tmp_path: Path) -> None:
    grammar_emit.reset_recorder_for_tests()
    d = _dispatcher(tmp_path)
    await d.handle_frame_envelope(_frame(uuid4()))
    assert grammar_emit.get_recorder() is None
    assert d.metrics.frames_dispatched_total == 1


def test_window_length_is_bounded_below_the_digester_expiry() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        Settings(VISION_ORGAN_WINDOW_SEC=600.0)
    assert Settings(VISION_ORGAN_WINDOW_SEC=60.0).VISION_ORGAN_WINDOW_SEC == 60.0


@pytest.mark.asyncio
async def test_dry_run_counts_frames_but_not_tasks(tmp_path: Path, organ) -> None:
    d = _dispatcher(tmp_path)
    d.settings.DRY_RUN = True
    await d.handle_frame_envelope(_frame(uuid4()))
    await d.sweep_timeouts(now=time.time() + 60.0)
    kv = {e.atom.text_value: _summary_kv(e.atom.summary)
          for e in grammar_emit.build_window_events(router="r", snapshot=organ.drain())[:-1]}["cam1"]
    assert kv["frames"] == "1"
    assert kv["dispatched"] == "0"
    assert kv["failed"] == "0"
