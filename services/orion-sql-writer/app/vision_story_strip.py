"""Pictures of the moment for walkway motion stories (2026-10-17).

Juniper (2026-10-17, on the Hub's "Walkway stories" card): "I can't tell a
person walked, it just looks like it's dark." The card drew a story's boxes
over the CURRENT walkway view, and the frames of the moment were long gone:
the shared frames directory keeps a walkway frame for only ~60 s (the edge's
capture worker prunes it at its own FRAME_RETENTION_SECONDS), and a story is
written at least settle + close-gap (30 + 11 s) after its LAST box.

So the pictures are saved in two steps, both in this service's motion loop:

1. ``capture_pending`` (every tick, before the frames are pruned): for each
   new walker-kind crop on the stream, find its frame by name (the retina
   names a frame ``<camera>_<stream>_<epoch_ms>_<hex>.jpg`` with
   ``epoch_ms = int(frame_ts * 1000)``, and vision-window sets the crop's
   ``observed_at`` from that same ``frame_ts``), then save a padded crop
   (``CROP_H`` px tall) and a downscaled full frame (``FRAME_W`` px wide) into
   ``<strip_dir>/_pending``. A crop whose frame is already gone gets a small
   ``.miss`` marker instead: the strip then says "frame gone", it never fakes it.
2. ``assemble_strip`` (for every published story without a strip): up to
   ``MAX_CROPS`` crops spread over the passage, plus one full frame with all of
   the passage's boxes drawn, into ``<strip_dir>/<story_id>/``.

Darkness. ``frame_luminance`` is the mean luma of the undrawn full frame;
``scene_luminance`` / ``scene_contrast`` are the median, over the strip's
crops, of each crop's median luma / RMS contrast (std of luma, 0..1 scale;
Peli 1990: visibility of detail follows contrast, not mean level). ``dark``
uses the contrast, because brightness alone cannot separate the cases
(measured live 2026-10-17, README): at night the porch light holds the
whole-frame mean at 0.25, and a lit-lawn crop with nothing visible in it has
median 0.11-0.125 -- the same as a dark-clothed person in daylight (0.11).
Contrast does separate them: night crops 0.012-0.075, day/dusk crops >= 0.134.

Pure apart from file I/O; the loop owns the database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("sql-writer.vision_story_strip")

PENDING = "_pending"
MAX_CROPS = 4
CROP_H = 256
FRAME_W = 640
PAD = 0.25            # padding around a crop box, as a share of its width / height
JPEG_Q = 85
MS_TOLERANCE = 2      # frame_ts float -> ms truncation vs observed_at microseconds
STORY_ID_RE = re.compile(r"[0-9a-f]{32}")
_FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<ms>\d{10,})_[0-9a-f]+\.jpg$")

# Box colours match the Hub card: first green, last red, the rest blue.
FIRST, LAST, MID = (34, 197, 94), (239, 68, 68), (96, 165, 250)


@dataclass(frozen=True)
class StripConfig:
    frames_dir: str = "/mnt/telemetry/vision/frames"
    strip_dir: str = "/mnt/telemetry/vision/story_strips"
    dark_contrast: float = 0.10
    pending_sec: float = 3600.0
    retention_days: float = 0.0  # 0 = strips are kept forever
    frame_w: float = 2560.0
    frame_h: float = 1920.0


def sanitize_id(value: str) -> str:
    """Same rule as orion-vision-retina frame_store._sanitize_id (the frame's name)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("_") or "unknown"


def _h(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:24]


def crop_key(crop_id: str) -> str:
    return "c_" + _h(crop_id)


def frame_key(stream_id: str, t: datetime) -> str:
    return "f_" + _h(f"{stream_id}|{epoch_ms(t)}")


def epoch_ms(t: datetime) -> int:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return int(t.timestamp() * 1000)


class FrameIndex:
    """One directory listing per tick: frame prefix -> sorted [(ms, path)]."""

    def __init__(self, frames_dir: str) -> None:
        self.by_prefix: Dict[str, List[Tuple[int, str]]] = {}
        try:
            with os.scandir(frames_dir) as it:
                for e in it:
                    m = _FRAME_RE.match(e.name)
                    if m:
                        self.by_prefix.setdefault(m["prefix"], []).append((int(m["ms"]), e.path))
        except OSError as exc:
            logger.warning("story_strip_frames_dir_unreadable dir=%s error=%s", frames_dir, exc)
        for v in self.by_prefix.values():
            v.sort()

    def find(self, camera_id: Optional[str], stream_id: str, t: datetime) -> Optional[str]:
        prefix = f"{sanitize_id(camera_id or stream_id)}_{sanitize_id(stream_id)}"
        want = epoch_ms(t)
        best: Optional[Tuple[int, str]] = None
        for ms, path in self.by_prefix.get(prefix, ()):
            d = abs(ms - want)
            if d <= MS_TOLERANCE and (best is None or d < best[0]):
                best = (d, path)
        return best[1] if best else None


def _atomic_save(img, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, "JPEG", quality=JPEG_Q)
    os.replace(tmp, path)


def padded_box(box: Sequence[float], w: float, h: float, pad: float = PAD) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in box)
    pw, ph = (x1 - x0) * pad, (y1 - y0) * pad
    return (int(max(0.0, x0 - pw)), int(max(0.0, y0 - ph)), int(min(w, x1 + pw)), int(min(h, y1 + ph)))


@dataclass(frozen=True)
class PendingCrop:
    crop_id: str
    t: datetime
    label: str
    box: Tuple[float, float, float, float]
    camera_id: Optional[str]


def capture_pending(crops: Iterable[PendingCrop], *, stream_id: str, cfg: StripConfig,
                    index: Optional[FrameIndex] = None) -> Dict[str, int]:
    """Save a padded crop and a small full frame for each crop while its frame exists."""
    from PIL import Image

    pending = Path(cfg.strip_dir) / PENDING
    pending.mkdir(parents=True, exist_ok=True)
    index = index or FrameIndex(cfg.frames_dir)
    counts = {"saved": 0, "frame_gone": 0, "already": 0, "failed": 0}
    by_frame: Dict[str, List[PendingCrop]] = {}
    for c in crops:
        if (pending / f"{crop_key(c.crop_id)}.jpg").exists() or (pending / f"{crop_key(c.crop_id)}.miss").exists():
            counts["already"] += 1
            continue
        path = index.find(c.camera_id, stream_id, c.t)
        if path is None:
            _write_miss(pending, c, "frame_gone")
            counts["frame_gone"] += 1
            continue
        by_frame.setdefault(path, []).append(c)
    for path, group in by_frame.items():
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
        except (OSError, ValueError) as exc:     # pruned between listing and opening, or a torn write
            for c in group:
                _write_miss(pending, c, "frame_gone")
            counts["frame_gone"] += len(group)
            logger.info("story_strip_frame_unreadable path=%s error=%s", os.path.basename(path), exc)
            continue
        try:
            W, H = im.size
            fpath = pending / f"{frame_key(stream_id, group[0].t)}.jpg"
            if not fpath.exists():
                small = im.resize((FRAME_W, max(1, round(H * FRAME_W / W))))
                _atomic_save(small, fpath)
            sx, sy = W / cfg.frame_w, H / cfg.frame_h
            for c in group:
                b = (c.box[0] * sx, c.box[1] * sy, c.box[2] * sx, c.box[3] * sy)
                x0, y0, x1, y1 = padded_box(b, W, H)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    _write_miss(pending, c, "empty_box")
                    counts["failed"] += 1
                    continue
                piece = im.crop((x0, y0, x1, y1))
                piece = piece.resize((max(1, round(piece.width * CROP_H / piece.height)), CROP_H))
                _atomic_save(piece, pending / f"{crop_key(c.crop_id)}.jpg")
                counts["saved"] += 1
        except OSError as exc:
            counts["failed"] += len(group)
            logger.warning("story_strip_capture_failed path=%s error=%s", os.path.basename(path), exc)
    return counts


def mark_skipped(c: PendingCrop, reason: str, cfg: StripConfig) -> None:
    """Record why a crop was not captured (once), so its strip entry can say so."""
    pending = Path(cfg.strip_dir) / PENDING
    key = crop_key(c.crop_id)
    if (pending / f"{key}.jpg").exists() or (pending / f"{key}.miss").exists():
        return
    pending.mkdir(parents=True, exist_ok=True)
    _write_miss(pending, c, reason)


def _write_miss(pending: Path, c: PendingCrop, reason: str) -> None:
    try:
        (pending / f"{crop_key(c.crop_id)}.miss").write_text(json.dumps({"reason": reason, "t": c.t.isoformat()}))
    except OSError:
        pass


def _miss_reason(path: Path) -> str:
    """frame_gone | empty_box | prop_overlap (anything unreadable: frame_gone)."""
    try:
        reason = json.loads(path.read_text()).get("reason")
    except (OSError, ValueError, AttributeError):
        return "frame_gone"
    return reason if reason in ("frame_gone", "empty_box", "prop_overlap") else "frame_gone"


def spread(n: int, k: int = MAX_CROPS) -> List[int]:
    """Up to k indexes over range(n), first and last included, evenly spaced."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def luma_stats(img) -> Tuple[float, float, float]:
    """(mean, median, RMS contrast = std) of 8-bit luma, on a 0..1 scale."""
    hist = img.convert("L").histogram()
    total = sum(hist) or 1
    mean = sum(i * n for i, n in enumerate(hist)) / total
    var = sum(n * (i - mean) ** 2 for i, n in enumerate(hist)) / total
    acc, median = 0, 0
    for i, n in enumerate(hist):
        acc += n
        if acc * 2 >= total:
            median = i
            break
    return mean / 255.0, median / 255.0, (var ** 0.5) / 255.0


def _median(values: Sequence[float]) -> Optional[float]:
    v = sorted(values)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2.0


def _parse_t(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def assemble_strip(story_id: str, stream_id: str, evidence: Sequence[Dict[str, Any]], *,
                   cfg: StripConfig) -> Dict[str, Any]:
    """Write ``<strip_dir>/<story_id>/`` and return the ``evidence_strip`` record.

    Record: ``{"version": 1, "crops": [{crop_id, t, file} | {crop_id, t, missing}],
    "frame": {file, t, crop_id} | null, "frame_luminance", "scene_luminance", "scene_contrast",
    "scene_measured_on": "crops"|"frame"|null, "dark", "dark_threshold", "retention_days", "saved_at"}``.
    ``missing`` is ``prop_overlap`` / ``empty_box`` (skipped on purpose), ``frame_gone``
    (looked for while capturing; already pruned) or ``not_captured`` (the
    capture never ran for it, e.g. the writer was down)."""
    from PIL import Image, ImageDraw

    if not STORY_ID_RE.fullmatch(story_id):
        raise ValueError(f"bad story_id {story_id!r}")
    root = Path(cfg.strip_dir)
    pending = root / PENDING
    out = root / story_id
    out.mkdir(parents=True, exist_ok=True)
    ev = [e for e in evidence if isinstance(e, dict) and e.get("crop_id")]
    crops_out: List[Dict[str, Any]] = []
    crop_stats: List[Tuple[float, float, float]] = []
    for k, i in enumerate(spread(len(ev))):
        e = ev[i]
        key = crop_key(str(e["crop_id"]))
        entry: Dict[str, Any] = {"crop_id": e["crop_id"], "t": e.get("t")}
        src = pending / f"{key}.jpg"
        if src.exists():
            name = f"crop_{k}.jpg"
            tmp = out / (name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, out / name)
            entry["file"] = name
            with Image.open(out / name) as cim:
                crop_stats.append(luma_stats(cim))
        else:
            miss = pending / f"{key}.miss"
            entry["missing"] = _miss_reason(miss) if miss.exists() else "not_captured"
        crops_out.append(entry)

    # The full frame: the captured one nearest the middle of the passage.
    frame_rec: Optional[Dict[str, Any]] = None
    frame_lum = None
    region_stats: List[Tuple[float, float, float]] = []
    mid = (len(ev) - 1) / 2.0
    for i in sorted(range(len(ev)), key=lambda j: abs(j - mid)):
        t = _parse_t(ev[i].get("t"))
        if t is None:
            continue
        src = pending / f"{frame_key(stream_id, t)}.jpg"
        if not src.exists():
            continue
        with Image.open(src) as im:
            im = im.convert("RGB")
        W, H = im.size
        sx, sy = W / cfg.frame_w, H / cfg.frame_h
        frame_lum = luma_stats(im)[0]
        boxes = [[float(v) for v in e["box"]] for e in ev if isinstance(e.get("box"), list) and len(e["box"]) == 4]
        # No saved crop: measure each box's padded area on the (undrawn) small frame instead.
        for b in boxes:
            region = padded_box((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), W, H)
            if region[2] - region[0] >= 2 and region[3] - region[1] >= 2:
                region_stats.append(luma_stats(im.crop(region)))
        draw = ImageDraw.Draw(im)
        n = len(boxes)
        for j, b in enumerate(boxes):
            colour = FIRST if j == 0 else LAST if j == n - 1 else MID
            draw.rectangle((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), outline=colour, width=2)
            draw.text((b[0] * sx + 2, max(0.0, b[1] * sy - 11)), str(j + 1), fill=colour)
        _atomic_save(im, out / "frame.jpg")
        frame_rec = {"file": "frame.jpg", "t": ev[i].get("t"), "crop_id": ev[i]["crop_id"]}
        break
    stats, measured_on = (crop_stats, "crops") if crop_stats else (region_stats, "frame") if region_stats \
        else ([], None)
    scene_lum = _median([s[1] for s in stats])
    contrast = _median([s[2] for s in stats])
    dark = None if contrast is None else bool(contrast < cfg.dark_contrast)
    r4 = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {
        "version": 1,
        "crops": crops_out,
        "frame": frame_rec,
        "frame_luminance": r4(frame_lum),
        "scene_luminance": r4(scene_lum),
        "scene_contrast": r4(contrast),
        "scene_measured_on": measured_on,
        "dark": dark,
        "dark_threshold": cfg.dark_contrast,
        "retention_days": cfg.retention_days,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def failed_record(error: str, cfg: StripConfig) -> Dict[str, Any]:
    """Final record for a story whose strip could not be built inside the pending window."""
    return {"version": 1, "error": error[:300], "crops": [], "frame": None, "frame_luminance": None,
            "scene_luminance": None, "scene_contrast": None, "scene_measured_on": None, "dark": None,
            "dark_threshold": cfg.dark_contrast, "retention_days": cfg.retention_days,
            "saved_at": datetime.now(timezone.utc).isoformat()}


def prune_pending(cfg: StripConfig, now: Optional[float] = None) -> int:
    pending = Path(cfg.strip_dir) / PENDING
    cut = (now or time.time()) - cfg.pending_sec
    n = 0
    try:
        with os.scandir(pending) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False) and e.stat(follow_symlinks=False).st_mtime < cut:
                        os.unlink(e.path)
                        n += 1
                except OSError:
                    pass
    except OSError:
        pass
    return n


def remove_strip_dir(cfg: StripConfig, story_id: str) -> bool:
    """Delete ``<strip_dir>/<story_id>`` only (a 32-hex name, a real directory, not a link)."""
    if not STORY_ID_RE.fullmatch(story_id or ""):
        return False
    d = Path(cfg.strip_dir) / story_id
    if d.is_symlink() or not d.is_dir():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def strip_dirs(cfg: StripConfig) -> List[str]:
    try:
        return sorted(e.name for e in os.scandir(cfg.strip_dir)
                      if STORY_ID_RE.fullmatch(e.name) and e.is_dir(follow_symlinks=False))
    except OSError:
        return []
