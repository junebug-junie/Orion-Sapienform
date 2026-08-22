"""Content-addressed percept blob storage.

Percepts are camera frames, and -- since 2026-08-22 (AffectGPT multimodal
affect capture) -- short audio/video clips. They exist so a model can look
at (or listen to) one, once, and they are the shortest-lived artifact in the
system -- the *interpretation* is what gets kept, not the picture or clip.

Three properties, in order of how much they matter:

1. **Content-addressed.** The only caller-supplied component of a read path is
   a 64-char hex digest, validated by regex. There is no directory to traverse
   and no name to guess: you cannot enumerate this store, you can only ask for
   a hash you already hold.
2. **Verified on read.** Bytes are re-hashed and refused if they do not match
   the address requested. A store that served the wrong file would be worse
   than one that served nothing.
3. **Expiring by default.** Retention is a foreground property, not a cleanup
   afterthought, because the failure mode of forgetting is an unbounded archive
   of a private home. `orion/schemas/...` retention work elsewhere in this repo
   exists because that is exactly what happens when nobody decides.

Deliberately separate from Hub's chat attachment store
(`HUB_CHAT_ATTACHMENT_DIR`): different content, different lifetime, and Hub is
the process holding the docker socket.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

from loguru import logger

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# (mime, magic prefix). Order matters only for readability; prefixes are
# disjoint. WEBP needs the RIFF container check plus the WEBP fourcc.
_MAGIC: tuple[tuple[str, bytes], ...] = (
    ("image/jpeg", b"\xff\xd8\xff"),
    ("image/png", b"\x89PNG\r\n\x1a\n"),
)


def sniff_mime(data: bytes) -> Optional[str]:
    """Identify percept bytes by magic number (images, and since 2026-08-22
    the audio/video clips AffectGPT capture needs).

    The declared content-type is not trusted: it is caller-supplied and the
    value gets handed to a model downstream.
    """
    for mime, prefix in _MAGIC:
        if data.startswith(prefix):
            return mime
    if len(data) >= 12 and data[0:4] == b"RIFF":
        fourcc = data[8:12]
        if fourcc == b"WEBP":
            return "image/webp"
        # WAV: same RIFF container family as WEBP, different fourcc. ffmpeg's
        # default PCM WAV muxer writes this at a fixed offset. Also requires
        # a "fmt " subchunk marker somewhere in the header region (review
        # finding, 2026-08-22: the bare 12-byte RIFF/WAVE fourcc alone,
        # header with no real audio data, used to sniff as valid audio/wav
        # and would have been handed straight to orion-affectgpt-worker's
        # audio decode -- checking for "fmt " is still a cheap sniff, not a
        # full parser, but it rules out that specific degenerate case).
        if fourcc == b"WAVE" and b"fmt " in data[12:64]:
            return "audio/wav"
    # MP4 (ISO base media file format): the first box is [4-byte size][4-byte
    # type], so "ftyp" sits at offset 4, not offset 0 -- not a startswith()
    # prefix like the formats above. ffmpeg's default (non-fragmented) mp4
    # muxer writes ftyp as the first box; this is NOT universally true of
    # every possible MP4 producer (a fragmented/faststart stream can lead
    # with a different box name) -- correct for what this service's actual
    # producer (clip_capture.py) writes, not a general MP4 validator.
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "video/mp4"
    return None


class PerceptStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _blob(self, sha256: str) -> Path:
        return self.root / f"{sha256}.bin"

    def _mime(self, sha256: str) -> Path:
        return self.root / f"{sha256}.mime"

    def put(self, data: bytes, *, mime: str) -> str:
        """Store bytes, return their sha256. Idempotent by construction."""
        sha256 = hashlib.sha256(data).hexdigest()
        target = self._blob(sha256)
        if not target.exists():
            # Write-then-rename so a concurrent GET never observes a partial
            # file. The pid in the temp name matters because content-addressing
            # means two uploads of the SAME frame collide on the same target --
            # which is the common case here, not the rare one.
            tmp = target.with_suffix(f".{os.getpid()}.part")
            tmp.write_bytes(data)
            tmp.replace(target)
            self._mime(sha256).write_text(mime, encoding="utf-8")
        else:
            # Refresh mtime so a frame that is still being requested does not
            # get swept out from under an in-flight interpretation.
            os.utime(target, None)
        return sha256

    def get(self, sha256: str) -> Tuple[bytes, str]:
        """Return (bytes, mime). Raises KeyError if absent, ValueError if corrupt."""
        if not SHA256_RE.match(sha256):
            raise ValueError("not a 64-char hex sha256")
        target = self._blob(sha256)
        if not target.exists():
            raise KeyError(sha256)
        data = target.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256:
            # Content-addressed storage that returns non-matching content is
            # worse than storage that returns nothing.
            raise ValueError(f"content hash mismatch: asked {sha256[:12]}, got {actual[:12]}")
        mime_path = self._mime(sha256)
        mime = mime_path.read_text(encoding="utf-8").strip() if mime_path.exists() else ""
        return data, (mime or sniff_mime(data) or "application/octet-stream")

    def sweep(self, *, max_age_seconds: int, now: Optional[float] = None) -> int:
        """Delete blobs older than max_age_seconds. Returns how many went.

        Age is mtime, and `put` refreshes mtime on a re-store, so "old" means
        "not seen recently" rather than "first seen long ago".
        """
        ts = float(now if now is not None else time.time())
        cutoff = ts - float(max_age_seconds)
        removed = 0
        for blob in self._iter_blobs():
            try:
                if blob.stat().st_mtime >= cutoff:
                    continue
                sha = blob.stem
                blob.unlink(missing_ok=True)
                self._mime(sha).unlink(missing_ok=True)
                removed += 1
            except OSError as exc:
                logger.warning("percept sweep skipped {}: {}", blob.name, exc)
        return removed

    def _iter_blobs(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for p in self.root.glob("*.bin"):
            if SHA256_RE.match(p.stem):
                yield p

    def stats(self) -> dict:
        n = 0
        total = 0
        oldest: Optional[float] = None
        for blob in self._iter_blobs():
            try:
                st = blob.stat()
            except OSError:
                continue
            n += 1
            total += st.st_size
            oldest = st.st_mtime if oldest is None else min(oldest, st.st_mtime)
        return {
            "count": n,
            "bytes": total,
            "oldest_age_seconds": round(time.time() - oldest, 1) if oldest else None,
        }
