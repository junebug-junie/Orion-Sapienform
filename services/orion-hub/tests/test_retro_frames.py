"""orion/retro/frames.py against the engine's own packer (services/orion-retro/app/media.py).

Each test builds a packet the way the engine does (``MediaProbe.publish`` then ``compress``) and reads it
back. Not in tests/retro/: it needs numpy and Pillow, which the native-boundary CI job does not install."""
from __future__ import annotations

import importlib.util
import io
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orion.retro import frames as F

REPO = Path(__file__).resolve().parents[3]


def engine_media():
    """services/orion-retro/app/media.py, loaded by path (``app`` is Hub's package name here)."""
    name = "orion_retro_engine_media"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "services/orion-retro/app/media.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Pipe:
    def __init__(self): self.packets = []
    def put_nowait(self, packet): self.packets.append(packet)


def pack(rgb: np.ndarray, fmt: int, frame: int = 2, pcm: bytes = b"\x01\x00\x02\x00" * 64) -> bytes:
    """One engine packet of ``rgb`` in pixel format ``fmt``, through the engine's own publish + compress."""
    media = engine_media()
    h, w = rgb.shape[:2]
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    if fmt == F.FORMAT_XRGB8888:
        video = ((r << 16) | (g << 8) | b).astype("<u4").tobytes()
    elif fmt == F.FORMAT_RGB565:
        video = (((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)).astype("<u2").tobytes()
    else:
        video = (((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)).astype("<u2").tobytes()
    probe = SimpleNamespace(video=video, pcm=bytearray(pcm), geometry=(w, h), pixel_format=fmt,
                            av=SimpleNamespace(timing=SimpleNamespace(sample_rate=32040.0)), audio_samples=999)
    pipe = Pipe()
    media.MediaProbe.publish(probe, pipe, frame)
    assert pipe.packets, "the engine publishes even frames only"
    return media.compress(pipe.packets[0])


def picture(h=224, w=256, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("fmt,tolerance", [(F.FORMAT_XRGB8888, 0), (F.FORMAT_RGB565, 8), (F.FORMAT_0RGB1555, 8)])
def test_every_pixel_format_reads_back_from_the_engines_own_packet(fmt, tolerance):
    rgb = picture()
    frame = F.decode_packet(pack(rgb, fmt))
    assert (frame.width, frame.height, frame.format, frame.frame) == (256, 224, fmt, 2)
    assert frame.rgb.shape == (224, 256, 3) and frame.rgb.dtype == np.uint8
    assert int(np.abs(frame.rgb.astype(int) - rgb.astype(int)).max()) <= tolerance
    if fmt != F.FORMAT_XRGB8888:                    # 5/6-bit white reads back as full white, black as black
        edge = np.zeros((224, 256, 3), np.uint8); edge[0, 0] = 255
        got = F.decode_packet(pack(edge, fmt)).rgb
        assert tuple(got[0, 0]) == (255, 255, 255) and tuple(got[1, 1]) == (0, 0, 0)


def test_the_grid_is_32_by_28_and_the_change_is_a_mean_absolute_difference():
    still = np.full((224, 256, 3), 40, np.uint8)
    a = F.luma_grid(F.decode_packet(pack(still, F.FORMAT_XRGB8888)).rgb)
    assert a.shape == (F.GRID_H, F.GRID_W) and np.allclose(a, 40 / 255.0)
    assert F.screen_change(a, a) == 0.0                                  # rest is reachable: exactly 0
    moved = still.copy(); moved[:8, :8] = 255                            # one 8x8 sprite cell turns white
    b = F.luma_grid(F.decode_packet(pack(moved, F.FORMAT_XRGB8888)).rgb)
    assert F.screen_change(a, b) == pytest.approx((255 - 40) / 255.0 / (32 * 28))
    black, white = np.zeros_like(still), np.full_like(still, 255)
    assert F.screen_change(F.luma_grid(black), F.luma_grid(white)) == pytest.approx(1.0)


def test_an_uneven_picture_still_gives_the_grid():
    rgb = picture(h=240, w=320, seed=3)                                  # N64 after the engine's decimation
    g = F.luma_grid(F.decode_packet(pack(rgb, F.FORMAT_RGB565)).rgb)
    assert g.shape == (28, 32) and 0.0 <= g.min() and g.max() <= 1.0


def test_the_png_is_enlarged_with_hard_edges():
    Image = pytest.importorskip("PIL.Image")

    rgb = picture(h=6, w=5, seed=7)
    data = F.png(rgb, 2)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    assert img.shape == (12, 10, 3)
    assert (img[::2, ::2] == rgb).all() and (img[1::2, 1::2] == rgb).all()   # each pixel is a 2x2 block
    with pytest.raises(F.FrameError):
        F.png(rgb, 5)


@pytest.mark.parametrize("bad", [
    b"not zlib at all",
    __import__("zlib").compress(b"\x00\x00"),
    __import__("zlib").compress(struct.pack("!I", 5) + b"{oops"),
    __import__("zlib").compress(struct.pack("!I", 2) + b"{}"),
])
def test_a_packet_that_is_not_a_frame_is_refused(bad):
    with pytest.raises(F.FrameError):
        F.decode_packet(bad)


def test_a_header_that_lies_about_its_video_is_refused():
    import json
    import zlib
    header = json.dumps({"version": 1, "frame": 2, "width": 256, "height": 224, "format": 1, "video_bytes": 10}).encode()
    with pytest.raises(F.FrameError, match="video_bytes"):
        F.decode_packet(zlib.compress(struct.pack("!I", len(header)) + header + b"x" * 10))
    header = json.dumps({"version": 1, "width": 2, "height": 2, "format": 1, "video_bytes": 16}).encode()
    with pytest.raises(F.FrameError, match="shorter than its video"):
        F.decode_packet(zlib.compress(struct.pack("!I", len(header)) + header + b"x" * 8))


def test_a_packet_that_inflates_past_the_engines_bound_is_refused():
    import zlib
    with pytest.raises(F.FrameError, match="bound"):
        F.decode_packet(zlib.compress(b"\x00" * (F.MAX_INFLATED + 10)))
