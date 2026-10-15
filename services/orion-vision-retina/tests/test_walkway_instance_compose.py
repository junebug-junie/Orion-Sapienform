"""The walkway camera runs on its own retina instance, not on the edge."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _walkway_env() -> dict:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    svc = compose["services"]["orion-vision-retina-walkway"]
    assert svc["profiles"] == ["walkway"]  # a plain `up` never starts it unconfigured
    return dict(e.split("=", 1) for e in svc["environment"])


def test_walkway_identity_matches_router_and_zones():
    env = _walkway_env()
    assert env["RETINA_SOURCE_TYPE"] == "rtsp"
    assert env["RETINA_STREAM_ID"] == "walkway" and env["RETINA_CAMERA_ID"] == "walkway"
    # Own heartbeat identity, distinct from the room/carbon retina.
    assert env["SERVICE_NAME"] == "vision-retina-walkway"
    assert env["RETINA_CLIP_ENABLED"] == "false"
    assert env["RETINA_SOURCE"] == "${WALKWAY_RTSP_URL:-}"


def test_walkway_stream_has_router_policy_and_zones():
    repo = ROOT.parents[1]
    router = yaml.safe_load((repo / "config" / "vision_frame_router.yaml").read_text())
    zones = yaml.safe_load((repo / "config" / "vision_zones.yaml").read_text())
    assert "walkway" in router["streams"]
    assert "walkway" in zones["streams"]


def test_edge_no_longer_captures_the_walkway():
    edge = yaml.safe_load((ROOT.parent / "orion-vision-edge" / "docker-compose.yml").read_text())
    assert "vision-edge-walkway" not in edge["services"]
