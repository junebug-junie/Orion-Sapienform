"""Per-box crop embeddings for the walkway camera (individuals, idea 1).

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md.

When a detect request sets ``want_crop_embeddings``, every detected box whose
label is in the tracked set gets a zone (``orion.vision.zones``) and, unless
that zone forbids it, an L2-normalized SigLIP vector of its crop. The vector
rides on the object (``VisionObject.embedding``) so the individuals reducer in
orion-sql-writer can cluster it without reaching into this service's volume.

The patio rule is enforced here, before any pixels reach the embedder: a box
in a no-embed zone is never cropped for embedding at all. It also fails
closed: if the zones file is missing, nothing is embedded, because "no zones
known" must not silently mean "the patio is fair game".

Pure functions over plain dicts plus an injected ``embed_fn`` so the privacy
rule is testable without a GPU.

**Thumbnails (the ask card's picture).** For every crop it embeds -- and only
those; the thumbnail is made from the exact same crop list the embedder is
handed, so a no-embed-zone box structurally cannot get one -- the host writes
a small JPEG (``THUMB_MAX_SIDE`` px, quality ``THUMB_QUALITY``) named by the
sha256 of its bytes under ``VISION_CROP_THUMB_DIR`` and returns
``thumb_ref="thumb:<sha256>"`` on the object. At most one thumbnail per
stream per ``VISION_CROP_THUMB_MIN_INTERVAL_SEC`` (``ThumbRateLimiter``), and
only on the router's triggered tier when the request says which tier it is.
Retention: a background thread in ``ThumbStore`` (never the detect path)
deletes thumbnails not written for ``VISION_CROP_THUMB_RETENTION_DAYS``
(default 10, longer than an ask's 7-day life). The Hub serves them read-only
by hash (``/api/vision/crop-thumbs/<sha256>``).

A box whose bottom-center is outside a no-embed zone can still contain its
pixels, so any box that INTERSECTS a no-embed zone
(``orion.vision.zones.intersects_no_embed``) gets neither an embedding nor a
thumbnail. Since 2026-10-15 no configured zone is no-embed (Juniper lifted the
patio rule), so this is a generic mechanism nothing uses today.

**Faces (2026-10-15).** When the request also sets ``want_crop_faces`` (the
router's walkway policy; no other stream), each EMBEDDED crop whose label is
in ``crop_face_labels`` (default ``person``) is handed to ``face_fn`` -- the
identity_face profile's already-loaded MTCNN + InceptionResnetV1 -- which
returns the most confident face's 512-d vector, its probability and the face
crop, or None. The vector rides on the object (``face_embedding``) to the
individuals reducer; the face crop becomes ``face_thumb_ref`` (own rate
limiter, triggered tier only), which the ask card prefers. A stream with no
zones on file (cam0, carbon) never reaches the embed list, so it can never
get a face here. Juniper's enrolled gallery is not read or written by this
path.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from PIL import Image

from orion.vision.zones import (
    DEFAULT_ZONES_PATH,
    Zone,
    intersects_no_embed,
    load_zones,
    may_embed,
    zone_for_box,
)

DEFAULT_CROP_EMBEDDING_LABELS = ("person", "dog", "bicycle", "stroller", "vehicle")
DEFAULT_MAX_CROPS_PER_FRAME = 8
DEFAULT_MIN_SIDE_PX = 16

EmbedFn = Callable[[List[Image.Image]], np.ndarray]
# Crops in, per crop: (face vector, face detection probability, face crop) or None.
FaceHit = Tuple[Sequence[float], float, Image.Image]
FaceFn = Callable[[List[Image.Image]], List[Optional[FaceHit]]]
DEFAULT_CROP_FACE_LABELS = ("person",)
# A crop in, "thumb:<sha256>" (or None) out.
ThumbFn = Callable[[Image.Image], Optional[str]]

THUMB_REF_PREFIX = "thumb:"
THUMB_MAX_SIDE = 160
THUMB_QUALITY = 70
DEFAULT_THUMB_RETENTION_DAYS = 10.0
DEFAULT_THUMB_MIN_INTERVAL_SEC = 10.0


class ThumbRateLimiter:
    """At most one thumbnail per stream per ``min_interval_sec``. A parked car
    embedded every frame must not write ~30k files a day."""

    def __init__(self, min_interval_sec: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._clock = clock
        self._last: Dict[str, float] = {}
        self._lock = threading.Lock()

    def take(self, stream_id: str) -> bool:
        now = self._clock()
        with self._lock:
            last = self._last.get(stream_id)
            if last is not None and now - last < self.min_interval_sec:
                return False
            self._last[stream_id] = now
            return True


def encode_thumb(crop: Image.Image, *, max_side: int = THUMB_MAX_SIDE, quality: int = THUMB_QUALITY) -> bytes:
    img = crop.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class ThumbStore:
    """Content-addressed crop thumbnails on local disk, pruned by age.

    ``put`` writes ``<root>/<sha256>.jpg`` atomically (temp file + rename) and
    returns ``thumb:<sha256>``. A second put of identical bytes just refreshes
    the file's mtime, so a thumbnail still being produced is not pruned. The
    prune runs on a daemon thread every ``prune_interval_sec``
    (``start_pruner``), never on the detect path -- the writer owns its
    retention, no separate job to forget.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        retention_days: float = DEFAULT_THUMB_RETENTION_DAYS,
        prune_interval_sec: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.retention_sec = max(0.0, float(retention_days)) * 86400.0
        self.prune_interval_sec = float(prune_interval_sec)
        self._clock = clock
        self._pruner: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start_pruner(self) -> None:
        if self._pruner is not None and self._pruner.is_alive():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.prune()
                except Exception as exc:
                    logger.warning(f"[CROP] thumbnail prune failed: {exc}")
                self._stop.wait(self.prune_interval_sec)

        self._pruner = threading.Thread(target=_loop, name="crop-thumb-prune", daemon=True)
        self._pruner.start()

    def stop_pruner(self) -> None:
        self._stop.set()

    def put(self, crop: Image.Image) -> Optional[str]:
        data = encode_thumb(crop)
        digest = hashlib.sha256(data).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{digest}.jpg"
        now = self._clock()
        if path.exists():
            os.utime(path, (now, now))
        else:
            tmp = self.root / f".{digest}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.utime(tmp, (now, now))
            os.replace(tmp, path)
        return f"{THUMB_REF_PREFIX}{digest}"

    def prune(self, now: Optional[float] = None) -> int:
        now = self._clock() if now is None else now
        if not self.root.is_dir():
            return 0
        removed = 0
        for p in self.root.iterdir():
            if not (p.name.endswith(".jpg") or p.name.endswith(".tmp")):
                continue
            try:
                if now - p.stat().st_mtime > self.retention_sec:
                    p.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
        if removed:
            logger.info(f"[CROP] pruned {removed} crop thumbnail(s) older than {self.retention_sec / 86400:.0f}d")
        return removed

_ZONES_CACHE: Dict[str, Optional[Dict[str, List[Zone]]]] = {}


def zones_path() -> Path:
    return Path(os.getenv("VISION_ZONES_PATH") or DEFAULT_ZONES_PATH)


def load_zones_fail_closed(path: Optional[Path] = None) -> Optional[Dict[str, List[Zone]]]:
    """stream_id -> zones, or None when the zones file is absent/unreadable.

    None is a distinct state from {} on purpose: callers must treat it as
    "embedding forbidden everywhere", not "no zones, embed everything".
    Cached per path; the file is read once per process.
    """
    p = Path(path) if path is not None else zones_path()
    key = str(p)
    if key in _ZONES_CACHE:
        return _ZONES_CACHE[key]
    result: Optional[Dict[str, List[Zone]]]
    if not p.exists():
        logger.warning(f"[CROP] zones file missing path={p}; crop embeddings disabled (fail closed)")
        result = None
    else:
        try:
            result = load_zones(p)
        except Exception as exc:  # malformed yaml is a config bug, not a reason to embed the patio
            logger.warning(f"[CROP] zones file unreadable path={p} err={exc}; crop embeddings disabled")
            result = None
    _ZONES_CACHE[key] = result
    return result


def _norm_label(label: Any) -> str:
    return str(label or "").strip().lower()


def resolve_labels(request: Dict[str, Any], params: Dict[str, Any]) -> set[str]:
    raw = request.get("crop_embedding_labels")
    if not raw:
        raw = params.get("crop_embedding_labels") or list(DEFAULT_CROP_EMBEDDING_LABELS)
    if isinstance(raw, str):
        raw = [x for x in raw.split(",")]
    return {_norm_label(x) for x in raw if _norm_label(x)}


def _clamp_box(box: Sequence[float], width: int, height: int) -> List[float]:
    """Clip a box into the frame, keeping its bottom edge strictly inside.

    GroundingDINO boxes are not clipped, and the point-in-polygon test treats
    a point exactly on a polygon's bottom edge (y == 1.0) as outside every
    zone. A person standing close to the camera has y2 == height (or more),
    which used to escape the patio. Nudging the bottom-center just inside
    the frame puts it back in the zone it visibly stands in.
    """
    if len(box) != 4:
        return list(box)
    x1, y1, x2, y2 = (float(v) for v in box)
    eps_x, eps_y = width * 1e-6, height * 1e-6
    return [
        min(max(x1, 0.0), width - eps_x),
        min(max(y1, 0.0), height - eps_y),
        min(max(x2, 0.0), width - eps_x),
        min(max(y2, 0.0), height - eps_y),
    ]


def _crop_ref(frame_key: str, box: Sequence[float], model_id: str, embed_profile: str) -> str:
    seed = f"{frame_key}|{','.join(f'{float(v):.1f}' for v in box)}|{model_id}"
    return f"crop:{embed_profile}:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def attach_crop_embeddings(
    objects: List[Dict[str, Any]],
    image: Image.Image,
    *,
    request: Dict[str, Any],
    params: Dict[str, Any],
    embed_fn: EmbedFn,
    zones_by_stream: Optional[Dict[str, List[Zone]]],
    model_id: str,
    embed_profile: str,
    min_score: float,
    thumb_fn: Optional[ThumbFn] = None,
    thumb_limiter: Optional[ThumbRateLimiter] = None,
    face_fn: Optional[FaceFn] = None,
    face_thumb_limiter: Optional[ThumbRateLimiter] = None,
) -> Dict[str, int]:
    """Mutates ``objects`` in place: sets ``zone`` on every tracked-label box and
    ``embedding``/``embedding_ref`` (and ``thumb_ref``, when ``thumb_fn`` is
    given) on the ones allowed to be embedded.

    Returns counters for the detect artifact (``crop_embeddings`` block) so a
    live check can see how many boxes were tracked, embedded, and withheld.
    """
    stats = {"tracked": 0, "embedded": 0, "withheld_no_embed_zone": 0, "withheld_other": 0, "thumbs": 0,
             "faces": 0, "face_thumbs": 0}
    labels = resolve_labels(request, params)
    stream_id = str(request.get("stream_id") or "").strip()
    width, height = image.width, image.height
    # Fail closed unless this camera has zones on file: an empty or drifted
    # stream_id must not turn into "no patio, embed everything".
    stream_zones: List[Zone] = []
    zones_known = zones_by_stream is not None and bool(stream_id) and stream_id in zones_by_stream
    if zones_known:
        stream_zones = list(zones_by_stream.get(stream_id) or [])
    max_n = int(params.get("crop_embedding_max_per_frame", DEFAULT_MAX_CROPS_PER_FRAME))
    min_side = float(params.get("crop_min_side_px", DEFAULT_MIN_SIDE_PX))
    frame_key = str(request.get("percept_sha256") or request.get("image_path") or request.get("frame_path") or "")

    to_embed: List[int] = []
    for i, obj in enumerate(objects):
        if _norm_label(obj.get("label")) not in labels:
            continue
        if float(obj.get("score", 0.0)) < min_score:
            continue
        stats["tracked"] += 1
        box = obj.get("box_xyxy") or []
        zone = zone_for_box(stream_zones, _clamp_box(box, width, height), width, height) if stream_zones else None
        obj["zone"] = zone.name if zone is not None else None
        touches_no_embed = zones_known and intersects_no_embed(
            stream_zones, _clamp_box(box, width, height), width, height)
        if not zones_known or zone is None or not may_embed(zone) or touches_no_embed:
            # Patio, outside every zone, or zones unknown for this camera: no
            # crop, no vector. Presence only.
            obj["embedding"] = None
            obj["embedding_ref"] = None
            obj["thumb_ref"] = None
            if (zone is not None and not may_embed(zone)) or touches_no_embed:
                stats["withheld_no_embed_zone"] += 1
            else:
                stats["withheld_other"] += 1
            continue
        if len(box) != 4:
            stats["withheld_other"] += 1
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if (x2 - x1) < min_side or (y2 - y1) < min_side:
            stats["withheld_other"] += 1
            continue
        to_embed.append(i)

    # Highest-scoring boxes first when over the per-frame cap.
    if len(to_embed) > max_n:
        to_embed.sort(key=lambda i: float(objects[i].get("score", 0.0)), reverse=True)
        stats["withheld_other"] += len(to_embed) - max_n
        to_embed = to_embed[:max_n]
    if not to_embed:
        return stats

    crops: List[Image.Image] = []
    for i in to_embed:
        x1, y1, x2, y2 = (float(v) for v in objects[i]["box_xyxy"])
        left = max(0, int(np.floor(x1)))
        top = max(0, int(np.floor(y1)))
        right = min(width, int(np.ceil(x2)))
        bottom = min(height, int(np.ceil(y2)))
        crops.append(image.crop((left, top, right, bottom)))

    vecs = np.asarray(embed_fn(crops), dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] != len(to_embed):
        raise ValueError(f"crop embedder returned shape {vecs.shape} for {len(to_embed)} crops")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    # Thumbnails only on the router's triggered tier when the request names
    # its tier (a request with no tier -- e.g. a direct RPC -- is not gated).
    tier = str(request.get("dispatch_tier") or "").strip()
    thumb_allowed = not tier or tier == "triggered"

    for row, i in enumerate(to_embed):
        obj = objects[i]
        obj["embedding"] = [float(x) for x in vecs[row].tolist()]
        obj["embedding_ref"] = _crop_ref(frame_key, obj["box_xyxy"], model_id, embed_profile)
        stats["embedded"] += 1
        # Same crop the embedder saw -- the only pixels a thumbnail is ever
        # made from. A thumbnail failure costs the picture, not the vector.
        obj["thumb_ref"] = None
        if thumb_fn is not None and thumb_allowed and (
                thumb_limiter is None or thumb_limiter.take(stream_id)):
            try:
                obj["thumb_ref"] = thumb_fn(crops[row])
            except Exception as exc:
                logger.warning(f"[CROP] thumbnail write failed: {exc}")
            if obj["thumb_ref"]:
                stats["thumbs"] += 1

    if face_fn is not None and request.get("want_crop_faces") is True:
        _attach_faces(objects, to_embed, crops, request=request, params=params, face_fn=face_fn,
                      thumb_fn=thumb_fn if thumb_allowed else None, thumb_limiter=face_thumb_limiter,
                      stream_id=stream_id, stats=stats)
    return stats


def resolve_face_labels(request: Dict[str, Any], params: Dict[str, Any]) -> set[str]:
    raw = request.get("crop_face_labels") or params.get("crop_face_labels") or list(DEFAULT_CROP_FACE_LABELS)
    if isinstance(raw, str):
        raw = raw.split(",")
    return {_norm_label(x) for x in raw if _norm_label(x)}


def _attach_faces(
    objects: List[Dict[str, Any]],
    embedded: List[int],
    crops: List[Image.Image],
    *,
    request: Dict[str, Any],
    params: Dict[str, Any],
    face_fn: FaceFn,
    thumb_fn: Optional[ThumbFn],
    thumb_limiter: Optional[ThumbRateLimiter],
    stream_id: str,
    stats: Dict[str, int],
) -> None:
    """Face vector + face thumbnail on embedded crops of a face label. A face
    failure costs the face, never the crop vector or the detection."""
    labels = resolve_face_labels(request, params)
    rows = [row for row, i in enumerate(embedded) if _norm_label(objects[i].get("label")) in labels]
    for i in embedded:
        objects[i]["face_embedding"] = None
        objects[i]["face_score"] = None
        objects[i]["face_thumb_ref"] = None
    if not rows:
        return
    try:
        hits = face_fn([crops[row] for row in rows])
    except Exception as exc:
        logger.warning(f"[CROP] face embedding failed, crop vectors kept: {exc}")
        stats["face_error"] = 1
        return
    if len(hits) != len(rows):
        logger.warning(f"[CROP] face_fn returned {len(hits)} results for {len(rows)} crops; faces dropped")
        stats["face_error"] = 1
        return
    for row, hit in zip(rows, hits):
        if hit is None:
            continue
        vec, prob, face_img = hit
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        if v.size == 0 or n <= 0 or not np.isfinite(n):
            continue
        obj = objects[embedded[row]]
        obj["face_embedding"] = [float(x) for x in (v / n).tolist()]
        obj["face_score"] = round(float(prob), 4)
        stats["faces"] += 1
        if thumb_fn is not None and face_img is not None and (
                thumb_limiter is None or thumb_limiter.take(stream_id)):
            try:
                obj["face_thumb_ref"] = thumb_fn(face_img)
            except Exception as exc:
                logger.warning(f"[CROP] face thumbnail write failed: {exc}")
            if obj["face_thumb_ref"]:
                stats["face_thumbs"] += 1


def _face_crop_box(box: Sequence[float], width: int, height: int, pad: float = 0.25) -> tuple[int, int, int, int]:
    """The face box grown by ``pad`` on each side (hair, chin), clipped."""
    x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
            min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy))))


def crop_faces(
    crops: List[Image.Image],
    *,
    mtcnn: Any,
    model: Any,
    device: str,
    min_prob: float,
    min_side_px: float = 40.0,
) -> List[Optional[FaceHit]]:
    """Per crop, the most confident face (MTCNN) at or above ``min_prob`` and
    at least ``min_side_px`` on its short side, embedded by InceptionResnetV1
    in one batched forward pass. Same detect + extract split as
    runner._run_identity_face. Returns (vector, probability, face crop) or
    None per crop."""
    import torch

    picked: List[Optional[tuple]] = []
    tensors = []
    for crop in crops:
        img = crop.convert("RGB")
        boxes, probs = mtcnn.detect(img)
        if boxes is None or probs is None or len(boxes) == 0:
            picked.append(None)
            continue
        best = max(range(len(probs)), key=lambda j: (probs[j] if probs[j] is not None else -1.0))
        prob = probs[best]
        box = [float(v) for v in boxes[best]]
        if prob is None or float(prob) < min_prob or min(box[2] - box[0], box[3] - box[1]) < min_side_px:
            picked.append(None)
            continue
        face = mtcnn.extract(img, np.asarray([box], dtype=np.float32), None)
        if face is None:
            picked.append(None)
            continue
        if face.dim() == 3:
            face = face.unsqueeze(0)
        tensors.append(face[:1])
        picked.append((float(prob), img.crop(_face_crop_box(box, img.width, img.height))))
    if not tensors:
        return [None for _ in crops]
    batch = torch.cat(tensors, dim=0)
    if device.startswith("cuda"):
        batch = batch.to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        vecs = model(batch).detach().float().cpu().numpy()
    out: List[Optional[FaceHit]] = []
    k = 0
    for item in picked:
        if item is None:
            out.append(None)
            continue
        out.append((vecs[k].tolist(), item[0], item[1]))
        k += 1
    return out
