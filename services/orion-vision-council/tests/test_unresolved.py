import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.registry import resolve
from orion.schemas.vision import VisionSceneInterpretationV1, VisionUnresolvedV1, VisionWindowPayload

from app import main as council_main
from app.main import CouncilService
from app.unresolved import UnresolvedRateLimiter, build_unresolved, unresolved_id_for


def _window(*, detections=0, hard=(), counts=None, stream="walkway", window_id="w1") -> VisionWindowPayload:
    return VisionWindowPayload(
        window_id=window_id,
        start_ts=1.0,
        end_ts=2.0,
        stream_id=stream,
        summary={
            "object_counts": counts or {},
            "detection_count": detections,
            "captions": ["a blurry shape near a fence"],
            "evidence": {"hard_labels": list(hard)},
        },
        artifact_ids=["art-1", "art-2"],
        artifact_uris=["/mnt/telemetry/vision/frames/a.jpg"],
    )


def _interp(uncertainties=()) -> VisionSceneInterpretationV1:
    return VisionSceneInterpretationV1(
        window_id="w1",
        scene_summary="walkway",
        uncertainties=[{"uncertainty": u, "reason": "low light", "evidence_refs": ["art-9"]} for u in uncertainties],
    )


def test_nothing_unresolved_returns_none():
    assert build_unresolved(_window(detections=3, hard=["person"]), _interp(), council_model="m", council_route="r") is None
    assert build_unresolved(_window(detections=0), None, council_model="m", council_route="r") is None


def test_no_label_when_detections_but_no_hard_labels():
    item = build_unresolved(
        _window(detections=4, counts={"dog": 1, "cat": 1}), None, council_model="m", council_route="r"
    )
    assert item is not None
    assert item.reason == "no_label"
    assert "could not name" in item.description
    assert "cat, dog" in item.description
    assert item.evidence_refs == ["art-1", "art-2"]
    assert item.image_ref == "/mnt/telemetry/vision/frames/a.jpg"
    assert item.unresolved_id == unresolved_id_for("w1", "no_label")
    assert any("detector" in t for t in item.what_was_tried)
    # Council did not run: it must not be listed as tried.
    assert not any("scene interpretation" in t for t in item.what_was_tried)


def test_council_uncertainty_wins_and_carries_its_evidence():
    item = build_unresolved(
        _window(detections=4), _interp(["is that a dog or a fox"]), council_model="qwen", council_route="metacog_background"
    )
    assert item.reason == "council_uncertainty"
    assert "is that a dog or a fox (low light)" in item.description
    assert "art-9" in item.evidence_refs
    assert any("qwen" in t for t in item.what_was_tried)


def test_payload_validates_against_registered_schema():
    item = build_unresolved(_window(detections=1), None, council_model="m", council_route="r")
    assert resolve("VisionUnresolvedV1") is VisionUnresolvedV1
    VisionUnresolvedV1.model_validate(item.model_dump(mode="json"))


def test_rate_limiter_per_stream():
    lim = UnresolvedRateLimiter(600)
    assert lim.allow("walkway", 1000.0)
    lim.mark("walkway", 1000.0)
    assert not lim.allow("walkway", 1500.0)
    assert lim.allow("cam0", 1500.0)
    assert lim.allow("walkway", 1600.0)


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, channel, env):
        self.published.append((channel, env))


def _env() -> BaseEnvelope:
    return BaseEnvelope(kind="vision.window", source=ServiceRef(name="t"), payload={})


def test_service_publishes_once_then_rate_limits(monkeypatch):
    svc = CouncilService()
    bus = _FakeBus()
    svc.bus = bus
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", True)
    window = _window(detections=2)

    async def run():
        a = await svc._maybe_publish_unresolved(window, None, _env(), now=100.0)
        b = await svc._maybe_publish_unresolved(_window(detections=2, window_id="w2"), None, _env(), now=200.0)
        c = await svc._maybe_publish_unresolved(_window(detections=2, window_id="w3"), None, _env(), now=800.0)
        return a, b, c

    assert asyncio.run(run()) == (True, False, True)
    assert len(bus.published) == 2
    channel, env = bus.published[0]
    assert channel == "orion:vision:unresolved:sql-write"
    assert env.kind == "vision.unresolved.v1"
    assert VisionUnresolvedV1.model_validate(env.payload).reason == "no_label"


def test_service_disabled_publishes_nothing(monkeypatch):
    svc = CouncilService()
    bus = _FakeBus()
    svc.bus = bus
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", False)
    assert asyncio.run(svc._maybe_publish_unresolved(_window(detections=2), None, _env(), now=1.0)) is False
    assert bus.published == []


def test_publish_failure_does_not_raise_or_consume_budget(monkeypatch):
    svc = CouncilService()

    class _Boom:
        async def publish(self, *_a, **_k):
            raise RuntimeError("bus down")

    svc.bus = _Boom()
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", True)
    assert asyncio.run(svc._maybe_publish_unresolved(_window(detections=2), None, _env(), now=1.0)) is False
    assert svc._unresolved_limiter.allow("walkway", 2.0)
