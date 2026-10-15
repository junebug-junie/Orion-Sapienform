"""Walkway (live) in the Vision panel (2026-10-15).

Covers scripts/vision_live_frame.py (path confinement, safe read,
downscale, metadata) and the generic /api/vision/frames/latest routes in
api_routes.py, called directly (same convention as
test_vision_affect_capture_api.py) against a real temp frames directory.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

os.environ.setdefault("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript")
os.environ.setdefault("CHANNEL_VOICE_LLM", "orion:voice:llm")
os.environ.setdefault("CHANNEL_VOICE_TTS", "orion:voice:tts")
os.environ.setdefault("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake")
os.environ.setdefault("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage")

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from PIL import Image  # noqa: E402

from scripts import api_routes  # noqa: E402
from scripts import vision_frame_cache  # noqa: E402
from scripts import vision_live_frame as vlf  # noqa: E402


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 80, 40)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fresh_memo():
    vlf.memo = vlf.RenderedFrameMemo()
    yield
    vlf.memo = vlf.RenderedFrameMemo()


@pytest.fixture
def frames_dir(tmp_path: Path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    return d


# --- path confinement -------------------------------------------------------


def test_resolve_accepts_a_jpeg_directly_inside_the_frames_dir(frames_dir: Path):
    p = frames_dir / "walkway_walkway_1_ab.jpg"
    assert vlf.resolve_local_frame(str(p), str(frames_dir)) == str(p.resolve())


@pytest.mark.parametrize(
    "rel",
    [
        "../secret.jpg",
        "sub/walkway.jpg",
        "walkway.png",
        ".hidden.jpg",
        "",
    ],
)
def test_resolve_rejects_anything_not_a_direct_jpeg_child(frames_dir: Path, rel: str):
    assert vlf.resolve_local_frame(str(frames_dir / rel) if rel else "", str(frames_dir)) is None


def test_resolve_rejects_relative_and_nul_paths(frames_dir: Path):
    assert vlf.resolve_local_frame("walkway.jpg", str(frames_dir)) is None
    assert vlf.resolve_local_frame(str(frames_dir / "a\x00.jpg"), str(frames_dir)) is None


def test_resolve_rejects_traversal_that_normalises_outside(frames_dir: Path, tmp_path: Path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg(8, 8))
    sneaky = f"{frames_dir}/../outside.jpg"
    assert vlf.resolve_local_frame(sneaky, str(frames_dir)) is None


def test_resolve_rejects_a_symlinked_subdir_pointing_back_in(frames_dir: Path, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    link_dir = frames_dir / "linkdir"
    link_dir.symlink_to(other)
    assert vlf.resolve_local_frame(str(link_dir / "a.jpg"), str(frames_dir)) is None


# --- safe read --------------------------------------------------------------


def test_read_refuses_a_symlinked_file(frames_dir: Path, tmp_path: Path):
    target = tmp_path / "elsewhere.jpg"
    target.write_bytes(_jpeg(8, 8))
    link = frames_dir / "walkway_link.jpg"
    link.symlink_to(target)
    assert vlf.load_local_frame(str(link), str(frames_dir)) is None


def test_read_refuses_a_fifo_without_blocking(frames_dir: Path):
    fifo = frames_dir / "walkway_fifo.jpg"
    os.mkfifo(fifo)
    assert vlf.load_local_frame(str(fifo), str(frames_dir)) is None


def test_read_refuses_an_oversize_file(frames_dir: Path):
    big = frames_dir / "walkway_big.jpg"
    big.write_bytes(b"\xff" * 2048)
    assert vlf.read_frame_file(str(big), max_bytes=1024) is None


def test_read_returns_none_for_a_pruned_file(frames_dir: Path):
    assert vlf.load_local_frame(str(frames_dir / "gone.jpg"), str(frames_dir)) is None


def test_read_returns_the_bytes_of_a_real_frame(frames_dir: Path):
    data = _jpeg(16, 12)
    p = frames_dir / "walkway_ok.jpg"
    p.write_bytes(data)
    assert vlf.load_local_frame(str(p), str(frames_dir)) == data


# --- downscale --------------------------------------------------------------


def test_downscale_shrinks_a_walkway_sized_frame_to_max_width():
    out = vlf.downscale_jpeg(_jpeg(2560, 1920), 1280)
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert im.size == (1280, 960)


def test_downscale_leaves_a_small_jpeg_untouched():
    data = _jpeg(640, 480)
    assert vlf.downscale_jpeg(data, 1280) is data


def test_downscale_zero_means_full_size():
    data = _jpeg(2560, 1920)
    assert vlf.downscale_jpeg(data, 0) is data


def test_downscale_raises_on_garbage():
    with pytest.raises(Exception):
        vlf.downscale_jpeg(b"not a jpeg", 1280)


# --- metadata ---------------------------------------------------------------


def test_public_meta_never_exposes_the_host_path_and_marks_staleness():
    latest = {
        "sha256": None,
        "image_path": "/mnt/telemetry/vision/frames/walkway_walkway_9_ff.jpg",
        "frame_ts": 9.0,
        "width": 2560,
        "height": 1920,
        "cached_at": 100.0,
    }
    fresh = vlf.public_frame_meta("walkway", latest, now=101.0, stale_after_sec=10.0)
    assert "image_path" not in fresh
    assert "/mnt/" not in repr(fresh)
    assert fresh["frame_id"] == "walkway_walkway_9_ff.jpg"
    assert fresh["stale"] is False
    assert fresh["source"] == "local"

    stale = vlf.public_frame_meta("walkway", latest, now=125.0, stale_after_sec=10.0)
    assert stale["stale"] is True
    assert stale["age_sec"] == 25.0


# --- routes -----------------------------------------------------------------


class _FakeFrameCache:
    def __init__(self, latest=None, stream_ids=("walkway",)):
        self._latest = latest
        self.stream_ids = set(stream_ids)
        self.calls = 0

    async def get_latest(self, stream_id):
        self.calls += 1
        return self._latest


def _route_patches(cache, frames_dir: Path, max_width: int = 1280):
    return (
        patch.object(vision_frame_cache, "cache", cache),
        patch.object(api_routes.settings, "HUB_VISION_FRAMES_DIR", str(frames_dir), create=True),
        patch.object(api_routes.settings, "HUB_VISION_LIVE_MAX_WIDTH", max_width, create=True),
        patch.object(api_routes.settings, "HUB_VISION_LIVE_STALE_SEC", 10.0, create=True),
    )


@pytest.mark.asyncio
async def test_walkway_image_route_serves_a_downscaled_local_frame(frames_dir: Path):
    p = frames_dir / "walkway_walkway_1_ab.jpg"
    p.write_bytes(_jpeg(2560, 1920))
    latest = {"sha256": None, "image_path": str(p), "cached_at": time.time()}
    cache = _FakeFrameCache(latest)
    a, b, c, d = _route_patches(cache, frames_dir)
    with a, b, c, d:
        resp = await api_routes.api_vision_frames_latest_image(stream_id="walkway")
    assert resp.media_type == "image/jpeg"
    assert resp.headers["cache-control"] == "no-store"
    with Image.open(io.BytesIO(resp.body)) as im:
        assert im.size == (1280, 960)


@pytest.mark.asyncio
async def test_image_route_memoises_the_same_frame(frames_dir: Path):
    p = frames_dir / "walkway_walkway_1_ab.jpg"
    p.write_bytes(_jpeg(2560, 1920))
    latest = {"sha256": None, "image_path": str(p), "cached_at": time.time()}
    a, b, c, d = _route_patches(_FakeFrameCache(latest), frames_dir)
    with a, b, c, d:
        first = await api_routes.api_vision_frames_latest_image(stream_id="walkway")
        p.unlink()  # pruned -- a second tab polling the same frame still gets it
        second = await api_routes.api_vision_frames_latest_image(stream_id="walkway")
    assert first.body == second.body


@pytest.mark.asyncio
async def test_image_route_404s_for_a_path_outside_the_frames_dir(frames_dir: Path, tmp_path: Path):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg(64, 48))
    latest = {"sha256": None, "image_path": str(outside), "cached_at": time.time()}
    a, b, c, d = _route_patches(_FakeFrameCache(latest), frames_dir)
    with a, b, c, d:
        with pytest.raises(HTTPException) as exc_info:
            await api_routes.api_vision_frames_latest_image(stream_id="walkway")
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "frame_not_readable"


@pytest.mark.asyncio
async def test_image_route_404s_for_a_stream_not_on_the_allowlist(frames_dir: Path):
    cache = _FakeFrameCache({"image_path": "/x.jpg"}, stream_ids=("carbon",))
    a, b, c, d = _route_patches(cache, frames_dir)
    with a, b, c, d:
        with pytest.raises(HTTPException) as exc_info:
            await api_routes.api_vision_frames_latest_image(stream_id="cam0")
    assert exc_info.value.status_code == 404
    assert cache.calls == 0


@pytest.mark.asyncio
async def test_image_route_404s_when_the_file_does_not_decode(frames_dir: Path):
    p = frames_dir / "walkway_bad.jpg"
    p.write_bytes(b"\xff\xd8 truncated")
    latest = {"sha256": None, "image_path": str(p), "cached_at": time.time()}
    a, b, c, d = _route_patches(_FakeFrameCache(latest), frames_dir)
    with a, b, c, d:
        with pytest.raises(HTTPException) as exc_info:
            await api_routes.api_vision_frames_latest_image(stream_id="walkway")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_meta_route_reports_stale_when_no_new_pointer_for_a_while(frames_dir: Path):
    latest = {
        "sha256": None,
        "image_path": str(frames_dir / "walkway_walkway_1_ab.jpg"),
        "cached_at": time.time() - 60,
    }
    a, b, c, d = _route_patches(_FakeFrameCache(latest), frames_dir)
    with a, b, c, d:
        meta = await api_routes.api_vision_frames_latest(stream_id="walkway")
    assert meta["available"] is True
    assert meta["stale"] is True
    assert "image_path" not in meta


@pytest.mark.asyncio
async def test_meta_route_reports_no_frame_yet(frames_dir: Path):
    a, b, c, d = _route_patches(_FakeFrameCache(None), frames_dir)
    with a, b, c, d:
        meta = await api_routes.api_vision_frames_latest(stream_id="walkway")
    assert meta == {"available": False, "stream_id": "walkway", "reason": "no_frame_seen_yet"}


# --- page wiring (CLAUDE.md section 9) ----------------------------------------


def test_walkway_option_and_live_frame_module_are_wired_into_the_page():
    template = (HUB_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert '<option value="walkway-live">Walkway (live)</option>' in template
    module_tag = template.index("/static/js/vision-live-frame.js")
    app_tag = template.index("/static/js/app.js")
    assert module_tag < app_tag, "vision-live-frame.js must load before app.js"
    assert (HUB_ROOT / "static" / "js" / "vision-live-frame.js").is_file()

    app_js = (HUB_ROOT / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert "OrionVisionLiveFrame" in app_js
    assert "/api/vision/frames/latest?stream_id=" in app_js


def test_compose_mounts_the_frames_dir_read_only_and_caches_walkway():
    compose = (HUB_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert (
        "${HUB_VISION_FRAMES_DIR:-/mnt/telemetry/vision/frames}:"
        "${HUB_VISION_FRAMES_DIR:-/mnt/telemetry/vision/frames}:ro"
    ) in compose
    assert "VISION_FRAME_CACHE_STREAM_IDS=${VISION_FRAME_CACHE_STREAM_IDS:-carbon,walkway}" in compose
