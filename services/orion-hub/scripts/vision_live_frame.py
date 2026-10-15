"""Serve the latest frame of a camera stream to the Vision panel (2026-10-15).

Juniper's ask: watch the walkway camera from Hub's Vision dropdown the way
she already watches carbon. The walkway retina runs on athena and writes
2560x1920 JPEGs (~750KB, ~2/s, pruned after a minute) into the shared frames
directory; its bus pointer carries only that local ``image_path`` -- no
percept-store sha256. Hub mounts that directory read-only and reads the
file itself.

Everything here is blocking file/CPU work, called via ``asyncio.to_thread``
from the routes in api_routes.py so it never stalls the event loop
(``scripts/check_async_routes_not_blocking.py``).

Safety of the local read: the path comes off the bus, not from the browser,
but it is still treated as untrusted. It is served only when it is a direct
child of the configured frames directory, named like a JPEG, a regular file,
not a symlink (``O_NOFOLLOW``), not a FIFO (``O_NONBLOCK``), and under a size
cap. ``O_NOFOLLOW`` does not stop a hard link inside the directory; making
one needs write access to the host dir, which Hub (read-only mount) lacks.

Every frame -- carbon's percept-store bytes included -- goes through
``downscale_jpeg``: wider than HUB_VISION_LIVE_MAX_WIDTH is re-encoded at
quality 80, and bytes that will not decode are a 404, not served raw.
"""
from __future__ import annotations

import io
import os
import re
import stat
from typing import Any, Dict, Optional

DEFAULT_FRAMES_DIR = "/mnt/telemetry/vision/frames"
# Walkway frames are ~750KB; anything this large is not one of ours.
MAX_FRAME_BYTES = 16 * 1024 * 1024
_FRAME_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}\.(?:jpg|jpeg)")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def safe_header_frame_id(frame_id: Optional[str]) -> str:
    """frame_id if it is a plain JPEG name or a sha256, else ''."""
    fid = frame_id or ""
    if _FRAME_NAME_RE.fullmatch(fid) or _SHA256_RE.fullmatch(fid):
        return fid
    return ""


def resolve_local_frame(image_path: Optional[str], frames_dir: str) -> Optional[str]:
    """Return the safe absolute path for ``image_path``, or None if it is not
    a JPEG sitting directly inside ``frames_dir`` (after normalising ``..`` and
    resolving any symlinked parent directories)."""
    if not image_path or not frames_dir or "\x00" in image_path:
        return None
    if not os.path.isabs(image_path):
        return None
    norm = os.path.normpath(image_path)
    name = os.path.basename(norm)
    if not _FRAME_NAME_RE.fullmatch(name):
        return None
    root = os.path.realpath(frames_dir)
    if os.path.realpath(os.path.dirname(norm)) != root:
        return None
    return os.path.join(root, name)


def read_frame_file(path: str, max_bytes: int = MAX_FRAME_BYTES) -> Optional[bytes]:
    """Regular files only, final component never followed if it is a symlink,
    never blocks on a FIFO, size-capped. Any OS error (including the retina
    having pruned the file already) is None, never an exception."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        chunks = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(fd, min(262144, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        return None if not data or len(data) > max_bytes else data
    except OSError:
        return None
    finally:
        os.close(fd)


def load_local_frame(image_path: Optional[str], frames_dir: str) -> Optional[bytes]:
    safe = resolve_local_frame(image_path, frames_dir)
    if safe is None:
        return None
    return read_frame_file(safe)


def downscale_jpeg(data: bytes, max_width: int, quality: int = 80) -> bytes:
    """Shrink to at most ``max_width`` pixels wide, re-encoded as JPEG.

    Uses Pillow's JPEG draft mode, so a 2560px frame is decoded at a reduced
    DCT scale rather than full size first. Images already narrow enough are
    returned untouched. Raises on undecodable input (caller maps to 404).
    """
    if max_width <= 0:
        return data
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        if width <= max_width and im.format == "JPEG":
            return data
        target = (max_width, max(1, round(height * max_width / width)))
        im.draft("RGB", target)
        out_im = im.convert("RGB")
        if out_im.width > max_width:
            out_im = out_im.resize(
                (max_width, max(1, round(out_im.height * max_width / out_im.width))),
                Image.BILINEAR,
            )
        buf = io.BytesIO()
        out_im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def frame_id_of(latest: Dict[str, Any]) -> Optional[str]:
    """Stable id for 'is this a new frame?' -- the sha256 when there is one,
    else the file name. Never the full host path."""
    if latest.get("sha256"):
        return str(latest["sha256"])
    path = latest.get("image_path")
    return os.path.basename(str(path)) if path else None


def public_frame_meta(
    stream_id: str, latest: Dict[str, Any], *, now: float, stale_after_sec: float
) -> Dict[str, Any]:
    """Browser-facing metadata. Age is measured from when Hub saw the
    pointer (Hub's own clock), so a skewed camera clock can't make a dead
    stream look live."""
    cached_at = float(latest.get("cached_at") or 0.0)
    age = max(0.0, now - cached_at) if cached_at else None
    return {
        "available": True,
        "stream_id": stream_id,
        "frame_id": frame_id_of(latest),
        "sha256": latest.get("sha256"),
        "frame_ts": latest.get("frame_ts"),
        "width": latest.get("width"),
        "height": latest.get("height"),
        "cached_at": latest.get("cached_at"),
        "age_sec": None if age is None else round(age, 2),
        "stale_after_sec": stale_after_sec,
        "stale": age is None or age > stale_after_sec,
        "source": "local" if latest.get("image_path") else "percept_store",
    }


class RenderedFrameMemo:
    """Last downscaled frame per stream, so several open tabs polling the
    same frame decode it once. Event-loop only (no locking needed)."""

    def __init__(self) -> None:
        self._by_stream: Dict[str, tuple] = {}

    def get(self, stream_id: str, frame_id: str, max_width: int) -> Optional[bytes]:
        hit = self._by_stream.get(stream_id)
        if hit and hit[0] == frame_id and hit[1] == max_width:
            return hit[2]
        return None

    def put(self, stream_id: str, frame_id: str, max_width: int, data: bytes) -> None:
        self._by_stream[stream_id] = (frame_id, max_width, data)


memo = RenderedFrameMemo()
