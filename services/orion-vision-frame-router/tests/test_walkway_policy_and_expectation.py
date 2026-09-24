"""Walkway stream policy + expectation steering (walkway spec Patch 0, ideas 1, 8)."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.vision import VisionFramePointerPayload

from app.expectation import ExpectationCache, expect_key
from app.policy import FrameDispatchPolicy
from app.settings import Settings
from app.state import RouterState

CFG = Path(__file__).resolve().parents[3] / "config" / "vision_frame_router.yaml"
PROMPTS = ["person", "dog", "bicycle", "stroller", "vehicle", "package", "mail truck"]


def _policy(expectation: ExpectationCache | None = None) -> FrameDispatchPolicy:
    return FrameDispatchPolicy.load(Settings(ROUTER_POLICY_PATH=str(CFG)), expectation=expectation)


def _env(stream_id: str = "walkway") -> BaseEnvelope:
    payload = VisionFramePointerPayload(
        image_path="/tmp/f.jpg", camera_id=stream_id, stream_id=stream_id, frame_ts=time.time()
    )
    return BaseEnvelope(
        kind="vision.frame.pointer",
        source=ServiceRef(name="vision-edge", version="0.2.0"),
        correlation_id=uuid4(),
        payload=payload.model_dump(mode="json"),
    )


def _decide(policy: FrameDispatchPolicy, state: RouterState, now: float = 100.0):
    # every_n_frames=5: mark 4 frames seen so the 5th is not sampled out.
    for _ in range(4):
        state.mark_seen("walkway")
    return policy.decide(_env(), state, now=now, image_path_exists=True)


class _FakePipe:
    def __init__(self, keys: set[str], fail: bool = False) -> None:
        self.keys, self.fail, self.asked = keys, fail, []

    def exists(self, key: str) -> None:
        self.asked.append(key)

    async def execute(self):
        if self.fail:
            raise ConnectionError("redis down")
        return [1 if k in self.keys else 0 for k in self.asked]


class _FakeRedis:
    def __init__(self, keys: set[str], fail: bool = False) -> None:
        self.keys, self.fail = keys, fail

    def pipeline(self) -> _FakePipe:
        return _FakePipe(self.keys, self.fail)


class _HangingPipe(_FakePipe):
    async def execute(self):
        await asyncio.sleep(3600)


class _HangingRedis(_FakeRedis):
    def pipeline(self) -> _FakePipe:
        return _HangingPipe(self.keys)


def test_walkway_baseline_requests_crop_embeddings_and_prompts() -> None:
    d = _decide(_policy(), RouterState())
    assert d.should_dispatch and d.policy_name == "walkway" and d.dispatch_tier == "baseline"
    req = d.request_overrides
    assert req["want_crop_embeddings"] is True
    assert req["prompts"] == PROMPTS
    assert set(req["crop_embedding_labels"]) == {"person", "dog", "bicycle", "stroller", "vehicle"}
    assert req["want_caption"] is False
    assert not d.identity_dispatch_cfg  # no faces on this camera


def test_walkway_dog_triggers_captioned_tier() -> None:
    policy = _policy()
    assert "dog" in policy.trigger_labels_for("walkway", "walkway")
    state = RouterState()
    state.record_activity("walkway", ["dog"], now=99.0)
    d = _decide(policy, state)
    assert d.dispatch_tier == "triggered" and d.triggered_by == "labels"
    assert d.request_overrides["want_caption"] is True
    assert d.request_overrides["want_crop_embeddings"] is True


def test_cam0_trigger_labels_unchanged() -> None:
    assert _policy().trigger_labels_for("cam0", "cam0") == ["person"]


def test_open_expectation_selects_triggered_tier_and_is_traced() -> None:
    cache = ExpectationCache(refresh_sec=5)
    policy = _policy(cache)
    state = RouterState()
    assert _decide(policy, state).dispatch_tier == "baseline"  # notes the stream

    asyncio.run(cache.refresh_once(_FakeRedis({expect_key("walkway")})))
    assert cache.is_open("walkway")
    state = RouterState()
    d = _decide(policy, state)
    assert d.dispatch_tier == "triggered" and d.triggered_by == "expectation"
    frame = VisionFramePointerPayload.model_validate(_env().payload)
    task = policy.build_task_request(frame, _env(), d)
    assert task.meta["triggered_by"] == "expectation"


def test_expectation_key_absent_or_redis_down_means_baseline() -> None:
    cache = ExpectationCache(refresh_sec=5)
    cache.note_stream("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis({expect_key("walkway")})))
    assert cache.is_open("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis(set(), fail=True)))
    assert not cache.is_open("walkway") and cache.refresh_failures == 1
    d = _decide(_policy(cache), RouterState())
    assert d.dispatch_tier == "baseline"


def test_expectation_ignored_for_stream_with_empty_trigger_labels() -> None:
    cache = ExpectationCache(refresh_sec=5)
    asyncio.run(cache.refresh_once(_FakeRedis({expect_key("porch_eye")}), streams=["porch_eye"]))
    policy = _policy(cache)
    merged, _ = policy.resolve_stream_policy("porch_eye", "porch_eye")
    tier, _cfg, why = policy._tier_config(merged, RouterState(), "porch_eye", now=1.0)
    assert tier == "baseline" and why is None


def test_hung_redis_clears_open_expectations() -> None:
    cache = ExpectationCache(refresh_sec=0.5)
    cache.note_stream("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis({expect_key("walkway")})))
    assert cache.is_open("walkway")
    asyncio.run(cache.refresh_once(_HangingRedis({expect_key("walkway")})))
    assert not cache.is_open("walkway") and cache.refresh_failures == 1
