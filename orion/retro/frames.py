"""One game screen, read by Hub: decode an engine video packet, reduce it, compare it, picture it.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4,
section 2 ("The screen is a camera") and the metric quality gate record "Screen change".

Pure: no I/O, no clock. numpy only (Hub has it); the PNG is written with zlib, so no image library is
needed. Hub's Retro gateway (services/orion-hub/scripts/retro_watch.py) calls it on packets it already
relays to the browser. Nothing here opens a connection to the engine.

The wire, as the engine writes it (services/orion-retro/app/media.py ``MediaProbe.publish`` and
``compress``): ``zlib(u32 BE header length + UTF-8 JSON header + packed video + s16le PCM)``. The video
keeps the libretro pixel format (0 = 0RGB1555, 1 = XRGB8888, 2 = RGB565, native little-endian) and has no
row padding (``width * depth`` bytes per row). The sound is never read here.

Three numbers come out of a frame:

* ``luma_grid``: a 32 x 28 brightness grid, each cell the mean luma (Rec. 601 weights) of its block, 0..1.
  A SNES or NES frame (256 x 224) divides exactly into 8 x 8 blocks, the same grid step 0 measured.
* ``screen_change``: the mean absolute difference of two grids, 0..1. Step 0: a still screen reads
  0.0002-0.0004, walking in a room 0.05-0.11, a room change up to 0.43 (A Link to the Past). It does not
  separate still from walking on a fixed-screen game (The Legend of Zelda, NES): that is why a floor is
  per game and calibrated from live play (orion/retro/watch.py).
* ``png``: the frame enlarged by a whole factor with hard pixel edges (default 2: step 0 found 3x cost
  twice the tokens and gave no better notes).
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

GRID_W, GRID_H = 32, 28
FORMAT_0RGB1555, FORMAT_XRGB8888, FORMAT_RGB565 = 0, 1, 2
DEPTH = {FORMAT_0RGB1555: 2, FORMAT_XRGB8888: 4, FORMAT_RGB565: 2}
MAX_SIDE = 512                     # the engine decimates anything larger (media.py)
# One packet inflated: a 512 x 512 XRGB8888 picture, the engine's audio bound and a header.
MAX_INFLATED = 512 * 512 * 4 + 65536 + 4096
LUMA = (0.299, 0.587, 0.114)
ENLARGE_DEFAULT = 2
ENLARGE_MAX = 4


class FrameError(ValueError):
    """A packet that is not one decodable video frame."""


@dataclass(frozen=True)
class Frame:
    width: int
    height: int
    format: int
    frame: int | None           # the engine's frame counter, when the header names it
    rgb: np.ndarray             # (height, width, 3) uint8


def inflate(packet: bytes) -> bytes:
    """The packet's bytes, bounded: a packet that inflates past the engine's own bound is refused."""
    d = zlib.decompressobj()
    try:
        raw = d.decompress(bytes(packet), MAX_INFLATED + 1)
    except zlib.error as exc:
        raise FrameError(f"not zlib: {exc}") from exc
    if len(raw) > MAX_INFLATED or d.unconsumed_tail:
        raise FrameError("packet inflates past the engine's bound")
    return raw


def parse(raw: bytes) -> tuple[dict, bytes]:
    """(header, video bytes) of an inflated packet."""
    if len(raw) < 4:
        raise FrameError("packet shorter than its header length")
    (hlen,) = struct.unpack("!I", raw[:4])
    if hlen <= 0 or 4 + hlen > len(raw):
        raise FrameError("header length out of range")
    try:
        header = json.loads(raw[4:4 + hlen].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FrameError("header is not JSON") from exc
    if not isinstance(header, dict) or header.get("version") != 1:
        raise FrameError("unknown packet version")
    try:
        w, h, fmt, n = int(header["width"]), int(header["height"]), int(header["format"]), int(header["video_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrameError("header lacks width/height/format/video_bytes") from exc
    if fmt not in DEPTH:
        raise FrameError(f"unknown pixel format {fmt}")
    if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
        raise FrameError(f"geometry {w}x{h} out of range")
    if n != w * h * DEPTH[fmt]:
        raise FrameError(f"video_bytes {n} != {w}x{h}x{DEPTH[fmt]}")
    start = 4 + hlen
    if start + n > len(raw):
        raise FrameError("packet shorter than its video")
    return header, raw[start:start + n]


def _scale(channel: np.ndarray, bits: int) -> np.ndarray:
    top = (1 << bits) - 1
    return ((channel.astype(np.uint32) * 255 + top // 2) // top).astype(np.uint8)


def to_rgb(video: bytes, width: int, height: int, fmt: int) -> np.ndarray:
    """(height, width, 3) uint8 from one packed libretro picture."""
    if fmt == FORMAT_XRGB8888:
        a = np.frombuffer(video, dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(a[..., [2, 1, 0]])       # little-endian XRGB: bytes B, G, R, X
    v = np.frombuffer(video, dtype="<u2").reshape(height, width)
    if fmt == FORMAT_RGB565:
        r, g, b = _scale((v >> 11) & 31, 5), _scale((v >> 5) & 63, 6), _scale(v & 31, 5)
    elif fmt == FORMAT_0RGB1555:
        r, g, b = _scale((v >> 10) & 31, 5), _scale((v >> 5) & 31, 5), _scale(v & 31, 5)
    else:
        raise FrameError(f"unknown pixel format {fmt}")
    return np.dstack([r, g, b])


def decode_packet(packet: bytes) -> Frame:
    """One engine media packet (as relayed, still compressed) -> one Frame. Raises FrameError."""
    header, video = parse(inflate(packet))
    w, h, fmt = int(header["width"]), int(header["height"]), int(header["format"])
    frame = header.get("frame")
    return Frame(width=w, height=h, format=fmt, frame=int(frame) if isinstance(frame, int) else None,
                 rgb=to_rgb(video, w, h, fmt))


def _bin_means(a: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = a.shape
    if h % rows == 0 and w % cols == 0:
        return a.reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))
    # Uneven: each cell is the mean of the pixels whose index falls in it (np.linspace edges).
    re = np.linspace(0, h, rows + 1).round().astype(int)
    ce = np.linspace(0, w, cols + 1).round().astype(int)
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        band = a[re[i]:max(re[i + 1], re[i] + 1)]
        for j in range(cols):
            out[i, j] = band[:, ce[j]:max(ce[j + 1], ce[j] + 1)].mean()
    return out


def luma_grid(rgb: np.ndarray, cols: int = GRID_W, rows: int = GRID_H) -> np.ndarray:
    """The brightness grid: (rows, cols) float64, each cell 0..1."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise FrameError("not an RGB picture")
    f = rgb.astype(np.float64)
    lum = LUMA[0] * f[..., 0] + LUMA[1] * f[..., 1] + LUMA[2] * f[..., 2]
    return _bin_means(lum, rows, cols) / 255.0


def screen_change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two grids, 0..1."""
    if a.shape != b.shape:
        raise FrameError(f"grid shapes differ: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png(rgb: np.ndarray, scale: int = ENLARGE_DEFAULT) -> bytes:
    """An 8-bit RGB PNG of ``rgb`` enlarged ``scale`` times by repeating pixels (hard edges)."""
    if not (isinstance(scale, int) and 1 <= scale <= ENLARGE_MAX):
        raise FrameError(f"scale must be an integer 1..{ENLARGE_MAX}")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise FrameError("not an RGB picture")
    big = np.repeat(np.repeat(np.ascontiguousarray(rgb, dtype=np.uint8), scale, axis=0), scale, axis=1)
    h, w = big.shape[:2]
    rows = np.zeros((h, 1 + w * 3), dtype=np.uint8)     # filter byte 0 (None) per row
    rows[:, 1:] = big.reshape(h, w * 3)
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + _chunk(b"IEND", b""))


__all__ = [
    "DEPTH", "ENLARGE_DEFAULT", "FORMAT_0RGB1555", "FORMAT_RGB565", "FORMAT_XRGB8888", "Frame", "FrameError",
    "GRID_H", "GRID_W", "decode_packet", "inflate", "luma_grid", "parse", "png", "screen_change", "to_rgb",
]
