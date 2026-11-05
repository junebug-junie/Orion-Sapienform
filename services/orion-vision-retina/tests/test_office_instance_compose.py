"""The office camera (cam0) runs on its own retina instance, not on the edge
(2026-11-04, Juniper: "make sure this routes in through the vision retina and
not the vision edge")."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def _office() -> dict:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return compose["services"]["orion-vision-retina-office"]


def _office_env() -> dict:
    return dict(e.split("=", 1) for e in _office()["environment"])


def test_office_instance_reads_the_main_stream_as_cam0():
    env = _office_env()
    assert env["RETINA_SOURCE_TYPE"] == "rtsp"
    assert env["RETINA_SOURCE"] == "${OFFICE_RTSP_URL:-}"
    # The camera keeps its name: presence, history and config key on cam0.
    assert env["RETINA_STREAM_ID"] == "cam0" and env["RETINA_CAMERA_ID"] == "cam0"
    assert env["SERVICE_NAME"] == "vision-retina-office"
    assert env["RETINA_CLIP_ENABLED"] == "false"
    assert env["RETINA_FRAME_MODE"] == "local"


def test_office_instance_never_starts_unconfigured_on_a_plain_up():
    # carbon runs a plain `up` of this file, and an empty URL is refused at boot.
    assert _office()["profiles"] == ["office"]
    assert _office()["ports"] == ["127.0.0.1:${OFFICE_RETINA_HTTP_PORT:-8030}:8000"]


def test_office_env_template_has_every_room_key_and_no_credentials():
    example = (ROOT / ".env_example").read_text()
    for key in ("OFFICE_RTSP_URL=", "OFFICE_RETINA_FPS=", "OFFICE_RETINA_JPEG_QUALITY=", "OFFICE_RETINA_HTTP_PORT="):
        assert key in example
    assert "OFFICE_RTSP_URL=\n" in example  # empty: the real URL carries the camera password


def test_office_stream_has_router_policy_zone_and_window_parity():
    router = yaml.safe_load((REPO / "config" / "vision_frame_router.yaml").read_text())
    zones = yaml.safe_load((REPO / "config" / "vision_zones.yaml").read_text())
    room = router["streams"]["cam0"]
    assert room["baseline"]["request"]["want_crop_embeddings"] is True
    assert room["triggered"]["request"]["want_crop_faces"] is True
    assert "still_url" not in room["triggered"]["identity_dispatch"]
    # Fail-closed zone rule: no embed zone on file, no crops ever.
    assert any(z.get("embed") for z in zones["streams"]["cam0"]["zones"])


def test_vision_edge_is_kept_but_never_started_by_the_rebuild():
    """The edge stays in the repo but must not run: it would publish a second
    cam0 frame stream (its 640x480 substream) next to this retina's."""
    assert (REPO / "services" / "orion-vision-edge").is_dir()
    lines = [l.strip() for l in (REPO / "mesh-utilities" / "common" / "exclude_services.txt").read_text().splitlines()]
    assert "orion-vision-edge" in lines
