"""Home-camera face matches become IdentitySightingV1 (2026-10-09).

Live: cam0 matched Juniper 6x "probable" / 14x "possible" on 10-09 01:56-02:00, but the match only
lived in the presence row while she was in view. One sighting per sitting now reaches the
situation graph.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion.schemas.vision import VisionArtifactOutputs, VisionArtifactPayload  # noqa: E402
from orion.schemas.vision_sighting import IDENTITY_SIGHTING_KIND, IdentitySightingV1  # noqa: E402

from app import main as app_main  # noqa: E402
from app.main import WindowService  # noqa: E402


def _artifact(stream: str) -> VisionArtifactPayload:
    return VisionArtifactPayload(
        artifact_id="art-id", correlation_id="00000000-0000-0000-0000-0000000000aa", task_type="identity_face",
        device="cuda:0", inputs={"stream_id": stream, "camera_id": stream}, outputs=VisionArtifactOutputs(),
        timing={}, model_fingerprints={})


PROBABLE = {"subject": "juniper", "state": "probable", "similarity": 0.701}


def _svc() -> WindowService:
    svc = WindowService()
    svc.bus = AsyncMock()
    return svc


def test_one_probable_frame_is_not_enough_two_within_ten_minutes_are():
    """Review finding: a single borderline frame (live 0.56 at detect 0.71) must not publish."""
    svc = _svc()
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE)) is False
    svc.bus.publish.assert_not_awaited()
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE)) is True


def test_probable_match_on_a_home_camera_publishes_one_sighting():
    svc = _svc()
    asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE))
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE)) is True
    channel, env = svc.bus.publish.await_args.args
    assert channel == app_main.settings.CHANNEL_IDENTITY_SIGHTING_PUB and env.kind == IDENTITY_SIGHTING_KIND
    s = IdentitySightingV1.model_validate(env.payload)
    assert (s.subject, s.stream_id, s.place, s.similarity) == ("juniper", "cam0", "home", 0.701)


def test_once_per_sitting():
    svc = _svc()
    for _ in range(2):
        asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE))
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE)) is False
    assert svc.bus.publish.await_count == 1


def test_laptop_webcam_never_counts_it_travels_with_her():
    svc = _svc()
    for _ in range(3):
        assert asyncio.run(svc._maybe_publish_sighting(_artifact("carbon"), PROBABLE)) is False
    svc.bus.publish.assert_not_awaited()


def test_possible_or_no_match_never_counts():
    svc = _svc()
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), {**PROBABLE, "state": "possible"})) is False
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), None)) is False
    svc.bus.publish.assert_not_awaited()


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(app_main.settings, "WINDOW_SIGHTING_ENABLED", False)
    svc = _svc()
    assert asyncio.run(svc._maybe_publish_sighting(_artifact("cam0"), PROBABLE)) is False



def test_seen_at_is_the_frame_time_when_known():
    svc = _svc()
    art = _artifact("cam0")
    art.inputs["frame_ts"] = 1_791_500_000.0
    asyncio.run(svc._maybe_publish_sighting(art, PROBABLE))
    asyncio.run(svc._maybe_publish_sighting(art, PROBABLE))
    s = IdentitySightingV1.model_validate(svc.bus.publish.await_args.args[1].payload)
    assert s.seen_at.timestamp() == 1_791_500_000.0
