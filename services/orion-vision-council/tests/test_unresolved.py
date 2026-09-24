import asyncio
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.registry import resolve
from orion.schemas.vision import VisionSceneInterpretationV1, VisionUnresolvedV1, VisionWindowPayload

from app import main as council_main
from app.main import CouncilService
from app.unresolved import UnresolvedRateLimiter, build_unresolved, unresolved_id_for


def _window(*, detections=0, hard=(), counts=None, stream="cam0", window_id="w1") -> VisionWindowPayload:
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
    assert build_unresolved(
        _window(detections=3, hard=["person"], counts={"person": 1}), _interp(), council_model="m", council_route="r"
    ) is None
    assert build_unresolved(_window(detections=0), None, council_model="m", council_route="r") is None


def test_no_label_when_a_box_has_no_name():
    item = build_unresolved(
        _window(detections=4, counts={"dog": 1, "": 2, "object": 1}), None, council_model="m", council_route="r"
    )
    assert item is not None
    assert item.reason == "no_label"
    assert "could not name" in item.description
    assert "3 thing(s)" in item.description
    assert "dog" in item.description
    assert item.evidence_refs == ["art-1", "art-2"]
    assert item.image_ref == "/mnt/telemetry/vision/frames/a.jpg"
    assert item.unresolved_id == unresolved_id_for("w1", "no_label")
    assert any("detector" in t for t in item.what_was_tried)
    # Council did not run: it must not be listed as tried.
    assert not any("scene interpretation" in t for t in item.what_was_tried)


def test_council_uncertainty_wins_and_carries_its_evidence():
    item = build_unresolved(
        _window(detections=4, counts={"dog": 1}), _interp(["is that a dog or a fox"]), council_model="qwen", council_route="metacog_background"
    )
    assert item.reason == "council_uncertainty"
    assert "is that a dog or a fox (low light)" in item.description
    assert "art-9" in item.evidence_refs
    assert any("qwen" in t for t in item.what_was_tried)


def test_payload_validates_against_registered_schema():
    item = build_unresolved(_window(detections=1, counts={"": 1}), None, council_model="m", council_route="r")
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
    window = _window(detections=2, counts={"": 1})

    async def run():
        a = await svc._maybe_publish_unresolved(window, None, _env(), now=100.0)
        b = await svc._maybe_publish_unresolved(_window(detections=2, counts={"": 1}, window_id="w2"), None, _env(), now=200.0)
        c = await svc._maybe_publish_unresolved(_window(detections=2, counts={"": 1}, window_id="w3"), None, _env(), now=800.0)
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
    assert asyncio.run(svc._maybe_publish_unresolved(_window(detections=2, counts={"": 1}), None, _env(), now=1.0)) is False
    assert bus.published == []


def test_publish_failure_does_not_raise_or_consume_budget(monkeypatch):
    svc = CouncilService()

    class _Boom:
        async def publish(self, *_a, **_k):
            raise RuntimeError("bus down")

    svc.bus = _Boom()
    monkeypatch.setattr(council_main.settings, "COUNCIL_UNRESOLVED_ENABLED", True)
    assert asyncio.run(svc._maybe_publish_unresolved(_window(detections=2, counts={"": 1}), None, _env(), now=1.0)) is False
    assert svc._unresolved_limiter.allow("walkway", 2.0)


def _load_window_projection():
    """services/orion-vision-window/app/projection.py, loaded by path (its
    package is also named ``app``). The window service owns the summary shape,
    so this test builds summaries the way production does, not by hand."""
    path = Path(__file__).resolve().parents[2] / "orion-vision-window" / "app" / "projection.py"
    spec = importlib.util.spec_from_file_location("_vision_window_projection_for_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact(objects, artifact_id="art-real"):
    from orion.schemas.vision import VisionArtifactPayload

    def _obj(o):
        label, score = o[0], o[1]
        box = o[2] if len(o) > 2 else [0, 0, 1, 1]
        return {"label": label, "score": score, "box_xyxy": box}

    return VisionArtifactPayload(
        artifact_id=artifact_id,
        correlation_id="c",
        task_type="detect_open_vocab",
        device="cuda:0",
        inputs={},
        outputs={"objects": [_obj(o) for o in objects], "frame_width": 1000, "frame_height": 1000,
                 "caption": {"text": "two people sitting on the patio"}},
        timing={},
        model_fingerprints={},
    )


def _real_window(objects, stream="cam0") -> VisionWindowPayload:
    proj = _load_window_projection()
    summary = proj.summarize_items([(_artifact(objects), 1.0)])
    return VisionWindowPayload(window_id="wr", start_ts=1.0, end_ts=2.0, stream_id=stream, summary=summary,
                               artifact_ids=["art-real"], artifact_uris=["/mnt/telemetry/vision/frames/walkway.jpg"])


def test_real_summary_named_boxes_do_not_fire():
    # Host already drops < 0.25, so everything that arrives is >= 0.25.
    window = _real_window([("person", 0.3), ("dog", 0.26)])
    assert build_unresolved(window, None, council_model="m", council_route="r") is None


def test_real_summary_unnamed_box_fires_no_label():
    window = _real_window([("person", 0.9), ("", 0.4)])
    item = build_unresolved(window, None, council_model="m", council_route="r")
    assert item is not None and item.reason == "no_label"
    assert "1 thing(s)" in item.description


# --- a camera with a no-embed zone (the walkway's patio) ---------------------
# Shipped config/vision_zones.yaml: patio = x 0..0.35, y 0.7..1.0 (embed:
# false); walkway = y 0.35..1.0 (embed: true). Boxes are in a 1000x1000 frame.
PATIO_BOX = [100.0, 800.0, 200.0, 950.0]      # bottom-center (150, 950): patio
WALKWAY_BOX = [600.0, 600.0, 700.0, 900.0]    # bottom-center (650, 900): walkway
SKY_BOX = [400.0, 50.0, 500.0, 200.0]         # outside every zone


def test_walkway_unnamed_box_on_the_patio_records_nothing():
    window = _real_window([("", 0.5, PATIO_BOX), ("person", 0.9, PATIO_BOX)], stream="walkway")
    assert build_unresolved(window, _interp(["who is that on the patio"]), council_model="m", council_route="r") is None


def test_walkway_council_uncertainty_is_never_recorded_as_free_text():
    window = _real_window([("person", 0.9, WALKWAY_BOX)], stream="walkway")
    assert build_unresolved(window, _interp(["someone on the patio holding something"]),
                            council_model="m", council_route="r") is None


def test_walkway_unnamed_box_outside_the_patio_is_no_label_without_frame_or_captions():
    window = _real_window([("", 0.5, WALKWAY_BOX), ("", 0.5, PATIO_BOX), ("object", 0.4, SKY_BOX),
                           ("person", 0.9, PATIO_BOX)], stream="walkway")
    item = build_unresolved(window, _interp(["two people on the patio"]), council_model="m", council_route="r")
    assert item is not None and item.reason == "no_label"
    assert "1 thing(s)" in item.description  # the walkway box only
    assert "walkway camera" in item.description
    assert item.image_ref is None
    text = " ".join([item.description, *item.what_was_tried]).lower()
    assert "patio" not in text and "caption" not in text and "person" not in text
    assert item.evidence_refs == ["art-real"]


def test_second_no_embed_zone_guards_its_stream_too():
    from orion.vision.zones import Zone

    zones = {"driveway": [
        Zone(name="garden", polygon=((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)), embed=False),
        Zone(name="street", polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.0, 0.5)), embed=True),
    ]}
    window = _real_window([("", 0.5, [100.0, 700.0, 200.0, 900.0])], stream="driveway")  # in the garden
    assert build_unresolved(window, _interp(["x"]), council_model="m", council_route="r",
                            zones_by_stream=zones) is None
    window = _real_window([("", 0.5, [100.0, 100.0, 200.0, 300.0])], stream="driveway")  # on the street
    item = build_unresolved(window, None, council_model="m", council_route="r", zones_by_stream=zones)
    assert item is not None and item.image_ref is None and "driveway camera" in item.description


def test_unknown_zones_fail_closed_for_every_stream():
    window = _window(detections=4, counts={"": 2})
    assert build_unresolved(window, _interp(["x"]), council_model="m", council_route="r",
                            zones_by_stream=None) is None


def test_other_cameras_keep_full_behaviour():
    item = build_unresolved(_window(detections=3, hard=["person"], counts={"person": 1}),
                            _interp(["is that a coat or a person"]), council_model="m", council_route="r")
    assert item is not None and item.reason == "council_uncertainty"
    assert item.image_ref == "/mnt/telemetry/vision/frames/a.jpg"
