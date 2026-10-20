from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid

import cv2
import numpy as np
import urllib.error
import urllib.request
from loguru import logger
from pydantic import BaseModel


class SavedFrame(BaseModel):
    """Where the frame ended up.

    `image_path` is empty in percept-store mode: the frame never touched this
    node's disk, which is the point. `sha256` is empty in local mode.
    """

    image_path: str = ""
    sha256: str | None = None
    width: int
    height: int
    format: str = "jpg"
    bytes_written: int | None = None


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def ensure_frame_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_frame(
    frame: np.ndarray,
    *,
    directory: str,
    camera_id: str,
    stream_id: str,
    ts: float,
    quality: int,
) -> SavedFrame:
    ensure_frame_dir(directory)
    cam = _sanitize_id(camera_id)
    stream = _sanitize_id(stream_id)
    epoch_ms = int(ts * 1000)
    short = uuid.uuid4().hex[:8]
    filename = f"{cam}_{stream}_{epoch_ms}_{short}.jpg"
    image_path = os.path.join(directory, filename)
    ok = cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise OSError(f"cv2.imwrite failed for {image_path}")
    h, w = frame.shape[:2]
    size = os.path.getsize(image_path)
    return SavedFrame(
        image_path=image_path,
        width=w,
        height=h,
        format="jpg",
        bytes_written=size,
    )


def owned_frame_pattern(camera_id: str, stream_id: str) -> re.Pattern[str]:
    """File names save_frame() writes for this camera/stream, and nothing else."""
    prefix = f"{_sanitize_id(camera_id)}_{_sanitize_id(stream_id)}_"
    return re.compile(rf"^{re.escape(prefix)}\d+_[0-9a-f]{{8}}\.jpg$")


def cleanup_old_frames(directory: str, max_age_seconds: int, *, camera_id: str, stream_id: str) -> int:
    """Delete this camera/stream's own frames older than max_age_seconds.
    Never raises.

    Ownership: only names save_frame() wrote for this camera_id/stream_id.
    The directory is shared, and each writer prunes only its own files at its
    own retention (2026-10-17: the edge pruned every file at 60 s, including
    the walkway's, which are meant to live FRAME_RETENTION_SECONDS).

    The frames directory is shared: orion-vision-edge and every local-mode
    retina prune it, and orion-sql-writer reads it. Any file can vanish
    between listing it and looking at it. 2026-10-17: a file deleted between
    the old isfile() and getmtime() raised FileNotFoundError, which killed
    the walkway's capture task and the eye went dark for 3 h with no log.
    So: one stat per file, inside a try, and skip anything that is gone.
    """
    owned = owned_frame_pattern(camera_id, stream_id)
    now = time.time()
    removed = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not owned.match(entry.name):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if now - entry.stat(follow_symlinks=False).st_mtime <= max_age_seconds:
                        continue
                    os.remove(entry.path)
                    removed += 1
                except FileNotFoundError:
                    continue  # another pruner got it first
                except OSError as exc:
                    logger.debug(f"[RETINA] cleanup skip {entry.path}: {exc}")
    except FileNotFoundError:
        return removed  # no frames dir yet (or it was removed): nothing to prune
    except OSError as exc:
        logger.warning(f"[RETINA] cleanup cannot list {directory}: {exc}")
    return removed


class PerceptUploadError(RuntimeError):
    """The frame could not be handed to the percept store."""


def upload_bytes(
    data: bytes,
    *,
    base_url: str,
    token: str | None = None,
    timeout_sec: float = 10.0,
) -> str:
    """POST arbitrary bytes (a clip, not necessarily a JPEG frame) to
    orion-percept-store and return the content hash.

    Added 2026-08-22 for clip_capture.py's audio/video output -- upload_frame
    above stays JPEG-frame-specific and untouched (it is the proven,
    already-live path; this is a new sibling for a new content shape, not a
    refactor of working code). Same "never touches disk beyond what the
    caller already gave us" and hash-verification-on-response discipline.
    """
    local_sha = hashlib.sha256(data).hexdigest()
    url = f"{str(base_url).rstrip('/')}"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PerceptUploadError(f"percept upload to {url} failed: {exc}") from exc

    sha256 = str(body.get("sha256") or "")
    if sha256 != local_sha:
        raise PerceptUploadError(
            f"percept store returned {sha256[:12]!r} for content hashing to {local_sha[:12]!r}"
        )
    return sha256


def upload_frame(
    frame: np.ndarray,
    *,
    base_url: str,
    quality: int,
    token: str | None = None,
    timeout_sec: float = 10.0,
) -> SavedFrame:
    """Encode in memory and POST to orion-percept-store. Never touches disk.

    This is how a node with no shared filesystem feeds the pipeline. Everything
    else in this module writes a file and publishes a path, which is correct
    only when the consumer can read that path -- true for athena, false for
    every other machine.

    **Nothing is written locally, deliberately.** A capture agent on a personal
    laptop must not accumulate a spool of webcam frames: if the store is
    unreachable the frame is dropped and the next one is attempted. A backlog of
    images of someone's face on their own machine is a worse failure than a gap
    in the record.

    Uses urllib rather than adding httpx/requests to this service (AGENTS.md
    section 10 -- no dependency for a stdlib task). Callers run it off the event
    loop via asyncio.to_thread.
    """
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise PerceptUploadError("cv2.imencode failed")
    data = buf.tobytes()
    local_sha = hashlib.sha256(data).hexdigest()

    url = f"{str(base_url).rstrip('/')}"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/octet-stream")
    if token:
        req.add_header("X-Orion-Percept-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise PerceptUploadError(f"percept upload to {url} failed: {exc}") from exc

    sha256 = str(body.get("sha256") or "")
    if sha256 != local_sha:
        # The store is content-addressed; if it disagrees with us about what we
        # just sent, something re-encoded or truncated the body and the address
        # we would publish would not resolve to this frame.
        raise PerceptUploadError(
            f"percept store returned {sha256[:12]!r} for content hashing to {local_sha[:12]!r}"
        )

    h, w = frame.shape[:2]
    return SavedFrame(
        image_path="",
        sha256=sha256,
        width=w,
        height=h,
        format="jpg",
        bytes_written=len(data),
    )
