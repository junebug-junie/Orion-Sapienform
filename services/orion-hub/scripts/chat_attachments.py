"""Content-addressed store for images Juniper attaches to a chat turn.

Design notes that are load-bearing, not decoration:

* **Refs on the wire, bytes on disk.** Upload returns an ``AttachmentRefV1``
  carrying a sha256 and a URL. The bytes never enter the chat payload, the bus,
  ``chat_history``, or Postgres. A 2MB image is ~2.7MB of base64 and would be
  replicated into every store that mirrors a turn; per-event blobs into Postgres
  already took this host down once (TOAST OOM crash loop, 2026-07-23).

* **Content-addressed, so re-sending the same image is free.** The sha256 is the
  filename. Uploading a duplicate is a no-op that returns the existing ref.

* **The declared MIME is not trusted.** Type is decided by sniffing magic bytes,
  because the declared ``Content-Type`` is attacker-controlled and it is what
  the gateway will later hand to a model. A file claiming ``image/png`` that is
  actually something else is rejected, not stored under its claim.

* **No Pillow.** Dimensions come from parsing the container headers of the four
  formats we accept. The client downscales before upload, so there is no
  server-side resize to do, and this keeps the hub's dependency set unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import struct
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Response, UploadFile, File

from orion.core.bus.bus_schemas import AttachmentRefV1
from scripts.settings import settings

logger = logging.getLogger("orion-hub.chat-attachments")

router = APIRouter(prefix="/api/chat/attachments", tags=["chat-attachments"])

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Sniffed, not declared. Keyed by the magic-byte probe below.
SUPPORTED_MIMES = ("image/png", "image/jpeg", "image/webp", "image/gif")


# ───────────────────────────────────────────────────────────────
# Format sniffing + dimensions (no image library)
# ───────────────────────────────────────────────────────────────

def _sniff_png(data: bytes) -> Optional[tuple[int, int]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    # IHDR is required to be the first chunk: 8 sig + 4 len + 4 tag, then w,h.
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
            # Lossy: 3-byte frame tag, 3-byte sync code, then 14-bit w/h.
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
    # Recognised as WEBP, dimensions unknown -- still a valid attachment.
    return 0, 0


def _sniff_jpeg(data: bytes) -> Optional[tuple[int, int]]:
    if not data.startswith(b"\xff\xd8"):
        return None
    # Walk the segment chain to the first SOF marker carrying dimensions.
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
        seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
        # SOF0-SOF15 except the non-frame DHT/JPG/DAC markers.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 <= end:
                height, width = struct.unpack(">HH", data[i + 5:i + 9])
                return width, height
            break
        if seg_len < 2:
            break
        i += 2 + seg_len
    # Valid JPEG signature but no readable SOF -- accept without dimensions.
    return 0, 0


def sniff_image(data: bytes) -> Optional[tuple[str, int, int]]:
    """Return (mime, width, height) from magic bytes, or None if unsupported.

    Width/height of 0 means "recognised format, dimensions not readable" -- the
    attachment is still usable; only the UI's aspect-ratio hint is lost.
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

def storage_dir() -> Path:
    path = Path(settings.HUB_CHAT_ATTACHMENT_DIR).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path_for(sha256: str) -> Path:
    if not _SHA256_RE.match(sha256):
        # Guards path traversal at the only place a caller-supplied string
        # reaches the filesystem.
        raise HTTPException(status_code=400, detail="invalid attachment id")
    return storage_dir() / f"{sha256}.bin"


def _mime_path_for(sha256: str) -> Path:
    return storage_dir() / f"{sha256}.mime"


def store_bytes(data: bytes, filename: Optional[str] = None) -> AttachmentRefV1:
    """Validate, sniff, persist, and describe. Idempotent on content."""
    max_bytes = int(settings.HUB_CHAT_ATTACHMENT_MAX_BYTES)
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"attachment is {len(data)} bytes, limit is {max_bytes}",
        )

    sniffed = sniff_image(data)
    if sniffed is None:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported attachment type; accepted: {', '.join(SUPPORTED_MIMES)}",
        )
    mime, width, height = sniffed

    allowed = {m.strip() for m in str(settings.HUB_CHAT_ATTACHMENT_ALLOWED_MIMES).split(",") if m.strip()}
    if mime not in allowed:
        raise HTTPException(status_code=415, detail=f"attachment type {mime} is not enabled")

    sha256 = hashlib.sha256(data).hexdigest()
    target = _path_for(sha256)
    if not target.exists():
        # Write-then-rename so a concurrent GET never sees a partial file. The
        # temp name carries the pid so two workers storing the same content do
        # not write the same scratch path at once -- content-addressing means
        # concurrent uploads of one image collide on the sha, which is exactly
        # when this matters.
        tmp = target.with_suffix(f".{os.getpid()}.part")
        tmp.write_bytes(data)
        tmp.replace(target)
        _mime_path_for(sha256).write_text(mime, encoding="utf-8")
        logger.info("stored chat attachment sha=%s mime=%s bytes=%s", sha256[:12], mime, len(data))

    return AttachmentRefV1(
        kind="image",
        sha256=sha256,
        mime=mime,
        bytes=len(data),
        width=width or None,
        height=height or None,
        source_url=f"{str(settings.HUB_CHAT_ATTACHMENT_PUBLIC_BASE).rstrip('/')}/api/chat/attachments/{sha256}",
        filename=(filename or None),
    )


def load_bytes(sha256: str) -> tuple[bytes, str]:
    target = _path_for(sha256)
    if not target.exists():
        raise HTTPException(status_code=404, detail="attachment not found")
    mime_path = _mime_path_for(sha256)
    mime = mime_path.read_text(encoding="utf-8").strip() if mime_path.exists() else ""
    data = target.read_bytes()
    if mime not in SUPPORTED_MIMES:
        # Re-sniff rather than trust a missing or edited sidecar.
        sniffed = sniff_image(data)
        mime = sniffed[0] if sniffed else "application/octet-stream"
    return data, mime


# ───────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────

@router.post("")
async def upload_attachment(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept one image and return its AttachmentRefV1.

    Reads with a hard cap so an oversized or lying Content-Length cannot make
    the hub buffer an unbounded body.
    """
    max_bytes = int(settings.HUB_CHAT_ATTACHMENT_MAX_BYTES)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"attachment exceeds the {max_bytes} byte limit",
        )
    ref = store_bytes(data, filename=file.filename)
    return ref.model_dump(mode="json")


@router.get("/{sha256}")
async def get_attachment(sha256: str) -> Response:
    data, mime = load_bytes(sha256)
    return Response(
        content=data,
        media_type=mime,
        headers={
            # Content-addressed: the bytes at this URL can never change.
            "Cache-Control": "public, max-age=31536000, immutable",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Content-Type-Options": "nosniff",
        },
    )
