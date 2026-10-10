"""Keep the frames where a face check found a face (2026-10-10).

The camera frame buffer (/mnt/telemetry/vision/frames) rolls over in about 65 seconds, so the
frame behind a face match -- or a miss -- is gone before anyone can look at it. Juniper's wave at
cam0 on 10-10 04:02 scored 0.21/0.27 against her enrolled face with the detector 99.7-100% sure
it saw a face; re-enrolling from cam0's own angle needs those frames, and auditing a sighting
(or a stranger matched as her) needs them too.

Every identity_face check that detects at least one face writes the frame and a JSON sidecar:

    <root>/<stream_id>/<UTC date>/<HHMMSS_micro>_<best_state>_<best_similarity>.jpg / .json

At most one per stream per ``min_interval_sec``; files older than ``retention_days`` are pruned
on a daemon thread (never on the detection path). The writer owns its retention.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from PIL import Image

from .crop_embeddings import ThumbRateLimiter

FRAME_MAX_SIDE = 1280
FRAME_QUALITY = 90
_STATE_RANK = {"probable": 3, "possible": 2, "unsure": 1}


def best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Highest band first, then highest similarity."""
    scored = [c for c in candidates if isinstance(c, dict)]
    if not scored:
        return None
    return max(scored, key=lambda c: (_STATE_RANK.get(str(c.get("state")), 0), float(c.get("similarity") or -1.0)))


def _safe(part: str) -> str:
    return "".join(ch for ch in str(part) if ch.isalnum() or ch in ("-", "_")) or "unknown"


class FaceFrameStore:
    def __init__(
        self,
        root: str | Path,
        *,
        retention_days: float = 14.0,
        min_interval_sec: float = 5.0,
        prune_interval_sec: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.retention_sec = max(0.0, float(retention_days)) * 86400.0
        self.prune_interval_sec = float(prune_interval_sec)
        self._clock = clock
        self._limiter = ThumbRateLimiter(min_interval_sec)
        self._pruner: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def save(self, stream_id: str, img: Image.Image, meta: Dict[str, Any]) -> Optional[str]:
        """Write the frame + sidecar; returns the path relative to root, or None when rate-limited."""
        stream = _safe(stream_id)
        if not self._limiter.take(stream):
            return None
        now = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        best = best_candidate(meta.get("candidates") or []) or {}
        sim = best.get("similarity")
        stem = f"{now:%H%M%S_%f}_{_safe(best.get('state') or 'none')}_{float(sim):.3f}" if isinstance(sim, (int, float)) \
            else f"{now:%H%M%S_%f}_{_safe(best.get('state') or 'none')}"
        folder = self.root / stream / f"{now:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        frame = img.convert("RGB")
        frame.thumbnail((FRAME_MAX_SIDE, FRAME_MAX_SIDE))
        tmp = folder / f".{stem}.{os.getpid()}.tmp"
        frame.save(tmp, format="JPEG", quality=FRAME_QUALITY)
        os.replace(tmp, folder / f"{stem}.jpg")
        (folder / f"{stem}.json").write_text(json.dumps({"stream_id": stream_id, "saved_at": now.isoformat(), **meta},
                                                        default=str, indent=1))
        return str((folder / f"{stem}.jpg").relative_to(self.root))

    def prune(self, now: Optional[float] = None) -> int:
        now = self._clock() if now is None else now
        if not self.root.is_dir():
            return 0
        removed = 0
        for p in self.root.rglob("*"):
            if not p.is_file() or p.suffix not in (".jpg", ".json", ".tmp"):
                continue
            try:
                if now - p.stat().st_mtime > self.retention_sec:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        for d in sorted((d for d in self.root.rglob("*") if d.is_dir()), reverse=True):
            try:
                d.rmdir()       # only removes empty day folders
            except OSError:
                pass
        if removed:
            logger.info(f"[FACE_FRAMES] pruned {removed} file(s) older than {self.retention_sec / 86400:.0f}d")
        return removed

    def start_pruner(self) -> None:
        if self._pruner is not None and self._pruner.is_alive():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.prune()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[FACE_FRAMES] prune failed: {exc}")
                self._stop.wait(self.prune_interval_sec)

        self._pruner = threading.Thread(target=_loop, name="face-frames-prune", daemon=True)
        self._pruner.start()
