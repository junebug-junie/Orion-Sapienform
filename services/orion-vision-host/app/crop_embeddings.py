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
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from loguru import logger
from PIL import Image

from orion.vision.zones import DEFAULT_ZONES_PATH, Zone, load_zones, may_embed, zone_for_box

DEFAULT_CROP_EMBEDDING_LABELS = ("person", "dog", "bicycle", "stroller", "vehicle")
DEFAULT_MAX_CROPS_PER_FRAME = 8
DEFAULT_MIN_SIDE_PX = 16

EmbedFn = Callable[[List[Image.Image]], np.ndarray]

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
) -> Dict[str, int]:
    """Mutates ``objects`` in place: sets ``zone`` on every tracked-label box and
    ``embedding``/``embedding_ref`` on the ones allowed to be embedded.

    Returns counters for the detect artifact (``crop_embeddings`` block) so a
    live check can see how many boxes were tracked, embedded, and withheld.
    """
    stats = {"tracked": 0, "embedded": 0, "withheld_no_embed_zone": 0, "withheld_other": 0}
    labels = resolve_labels(request, params)
    stream_id = str(request.get("stream_id") or "").strip()
    width, height = image.width, image.height
    stream_zones: List[Zone] = []
    if zones_by_stream is not None and stream_id:
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
        zone = zone_for_box(stream_zones, box, width, height) if stream_zones else None
        obj["zone"] = zone.name if zone is not None else None
        if zones_by_stream is None or not may_embed(zone):
            # Patio (or zones unknown): no crop, no vector. Presence only.
            obj["embedding"] = None
            obj["embedding_ref"] = None
            if zones_by_stream is not None:
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

    for row, i in enumerate(to_embed):
        obj = objects[i]
        obj["embedding"] = [float(x) for x in vecs[row].tolist()]
        obj["embedding_ref"] = _crop_ref(frame_key, obj["box_xyxy"], model_id, embed_profile)
        stats["embedded"] += 1
    return stats
