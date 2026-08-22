"""Content-addressed store for reverie-visual-chain generated images.

Patch 1 of docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md
(design doc §6). Follows `services/orion-hub/scripts/chat_attachments.py`'s
discipline exactly, for the same reason stated there: per-event blobs into
Postgres already took the host down once (TOAST OOM crash loop, 2026-07-23).
Image bytes belong on disk; only the pointer row
(`orion.schemas.reverie_visual.ReverieVisualArtifactV1`) belongs in Postgres.

Load-bearing design notes:

* **Content-addressed by sha256.** The hash of the bytes is the filename.
  Storing a duplicate image is a no-op that returns the existing path.
* **MIME sniffed from magic bytes, never trusted from the caller.** A
  diffusion host's claimed content type is exactly as untrustworthy as a chat
  upload's declared `Content-Type` — decided by the same sniffing, not by
  what the generator says it produced.
* **Write-then-atomic-rename.** No torn files under a concurrent reader —
  `tmp.replace(target)` is atomic on the same filesystem.
* **Path convention**: `{base_dir}/{sha256}.{ext}` under
  `/mnt/storage-lukewarm/orion/reverie-visual/` in production (design doc
  §6, confirmed-live mount as of 2026-08-20). Tests must pass an explicit
  `base_dir` pointing at a temp directory — never the real mount.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default production path (design doc §6). Callers writing real artifacts
# should not need to know this constant exists -- pass an explicit
# `base_dir` in tests, let this default carry production callers.
DEFAULT_BASE_DIR = "/mnt/storage-lukewarm/orion/reverie-visual"

# Sniffed, not declared -- mirrors chat_attachments.SUPPORTED_MIMES.
SUPPORTED_MIMES = ("image/png", "image/jpeg", "image/webp", "image/gif")

_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


class UnsupportedImageError(ValueError):
    """Raised when the bytes don't sniff as one of SUPPORTED_MIMES."""


# ───────────────────────────────────────────────────────────────
# Format sniffing (no image library — same probes as chat_attachments.py)
# ───────────────────────────────────────────────────────────────


def _sniff_png(data: bytes) -> Optional[tuple[int, int]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _sniff_gif(data: bytes) -> Optional[tuple[int, int]]:
    if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")):
        return None
    if len(data) < 10:
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _sniff_webp(data: bytes) -> Optional[tuple[int, int]]:
    if len(data) < 30 or not data.startswith(b"RIFF") or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    try:
        if chunk == b"VP8 ":
            width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return width, height
        if chunk == b"VP8L":
            bits = struct.unpack("<I", data[21:25])[0]
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
    except (struct.error, IndexError):
        return None
    return 0, 0


def _sniff_jpeg(data: bytes) -> Optional[tuple[int, int]]:
    if not data.startswith(b"\xff\xd8"):
        return None
    i = 2
    end = len(data)
    while i + 9 < end:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > end:
            break
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 <= end:
                height, width = struct.unpack(">HH", data[i + 5 : i + 9])
                return width, height
            break
        if seg_len < 2:
            break
        i += 2 + seg_len
    return 0, 0


def sniff_image(data: bytes) -> Optional[tuple[str, int, int]]:
    """Return (mime, width, height) from magic bytes, or None if unsupported.

    Width/height of 0 means "recognised format, dimensions not readable" --
    the artifact is still usable; only the dimension hint is lost.
    """
    for mime, probe in (
        ("image/png", _sniff_png),
        ("image/jpeg", _sniff_jpeg),
        ("image/gif", _sniff_gif),
        ("image/webp", _sniff_webp),
    ):
        dims = probe(data)
        if dims is not None:
            return mime, dims[0], dims[1]
    return None


# ───────────────────────────────────────────────────────────────
# Store
# ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StoredVisualArtifact:
    """What `store_visual_artifact` returns -- enough to build a
    `ReverieVisualArtifactV1` row without re-deriving anything."""

    sha256: str
    mime: str
    bytes: int
    width: Optional[int]
    height: Optional[int]
    path: str


def store_visual_artifact(
    data: bytes,
    *,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> StoredVisualArtifact:
    """Validate, sniff, persist content-addressed, and describe.

    Idempotent on content -- storing the same bytes twice returns the same
    path without rewriting the file. Raises `UnsupportedImageError` if the
    bytes don't sniff as one of SUPPORTED_MIMES (never trusts a caller-passed
    mime hint, because there isn't one to pass -- the diffusion host's claimed
    output format is not authoritative any more than a chat upload's is).
    """
    if not data:
        raise ValueError("empty image bytes")

    sniffed = sniff_image(data)
    if sniffed is None:
        raise UnsupportedImageError(
            f"unsupported image bytes; accepted: {', '.join(SUPPORTED_MIMES)}"
        )
    mime, width, height = sniffed
    ext = _MIME_TO_EXT[mime]

    sha256 = hashlib.sha256(data).hexdigest()
    out_dir = Path(base_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{sha256}.{ext}"

    if not target.exists():
        # Write-then-rename so a concurrent reader never sees a partial file.
        # The temp name carries the pid so two workers storing the same
        # content don't collide on the same scratch path -- content-
        # addressing means concurrent writers of one image collide on the
        # sha, which is exactly when this matters.
        tmp = out_dir / f".{sha256}.{os.getpid()}.part"
        tmp.write_bytes(data)
        tmp.replace(target)

    return StoredVisualArtifact(
        sha256=sha256,
        mime=mime,
        bytes=len(data),
        width=width or None,
        height=height or None,
        path=str(target),
    )


def load_visual_artifact(sha256: str, *, base_dir: str | Path = DEFAULT_BASE_DIR) -> bytes:
    """Read back a previously stored artifact's bytes by hash.

    Tries every known extension since the caller does not carry the mime.
    Raises FileNotFoundError if no matching file exists under `base_dir`.
    """
    out_dir = Path(base_dir).expanduser()
    for ext in _MIME_TO_EXT.values():
        candidate = out_dir / f"{sha256}.{ext}"
        if candidate.exists():
            return candidate.read_bytes()
    raise FileNotFoundError(f"no visual artifact found for sha256={sha256} under {out_dir}")
