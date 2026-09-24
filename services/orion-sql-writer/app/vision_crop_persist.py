"""Persist ``vision.crop.observation.v1`` -- one bus event, one row per crop.

The generic ``MODEL_MAP`` path writes one row per envelope. A crop
observation fans out to N rows in ``vision_crop_observation`` (crop_id =
``<observation_id>:<index>``), so it gets its own handler, the same way
``chat.history.spark_meta.patch.v1`` does.

**Patio rule, defense in depth.** The producer (orion-vision-host / window)
is supposed to skip embeddings for boxes in a no-embed zone. This module
does not trust that: it recomputes each box's zone from
``config/vision_zones.yaml`` and drops ``embedding``/``embedding_ref`` if
EITHER the producer-declared zone or the recomputed zone is a no-embed zone.
The DB CHECK constraint on the table is the third layer.

``build_crop_rows`` is pure (zones passed in); ``persist_crop_observation``
is the blocking DB write, called from a worker thread.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from orion.schemas.vision import VisionCropObservationV1
from orion.vision.zones import Zone, load_zones, zone_for_box

logger = logging.getLogger("sql-writer.vision_crop_persist")


_ZONES_CACHE: Optional[Dict[str, List[Zone]]] = None


def _zones_by_stream() -> Optional[Dict[str, List[Zone]]]:
    """Zones per stream, cached only on SUCCESS. None = could not load (fail closed)."""
    global _ZONES_CACHE
    if _ZONES_CACHE is not None:
        return _ZONES_CACHE
    try:
        zones = load_zones()
    except Exception as exc:
        logger.error("vision_zones_load_failed error=%s -- embeddings will be stripped until it loads", exc)
        return None
    if not zones:
        logger.error("vision_zones_empty (config/vision_zones.yaml missing?) -- embeddings will be stripped")
        return None
    _ZONES_CACHE = zones
    return zones


def redact_crop_payload(payload: Any) -> Dict[str, Any]:
    """What may go to bus_fallback_log for this kind: no vectors, no boxes, just counts."""
    p = payload if isinstance(payload, dict) else {}
    crops = p.get("crops") if isinstance(p.get("crops"), list) else []
    return {
        "observation_id": p.get("observation_id"),
        "stream_id": p.get("stream_id"),
        "observed_at": str(p.get("observed_at")) if p.get("observed_at") is not None else None,
        "crop_count": len(crops),
        "redacted": "crop embeddings and boxes are never written to the fallback log",
    }


def no_embed_zone_names(zones: Sequence[Zone]) -> set[str]:
    return {z.name for z in zones if not z.embed}


def build_crop_rows(
    obs: VisionCropObservationV1, zones: Optional[Sequence[Zone]]
) -> List[Dict[str, Any]]:
    """One DB row per crop. Embeddings stripped for any no-embed zone.

    ``zones=None`` means the zone config could not be loaded: fail closed and
    strip every embedding (a patio box cannot be told apart without zones).
    Malformed boxes (not 4 numbers) are skipped, not fatal to the observation.
    """
    fail_closed = zones is None
    zones = zones or []
    forbidden = no_embed_zone_names(zones)
    rows: List[Dict[str, Any]] = []
    for idx, crop in enumerate(obs.crops):
        if len(crop.box_xyxy) != 4:
            logger.warning("vision_crop_bad_box observation_id=%s index=%s len=%s -- skipped",
                           obs.observation_id, idx, len(crop.box_xyxy))
            continue
        computed: Optional[Zone] = None
        if obs.frame_width and obs.frame_height:
            computed = zone_for_box(zones, crop.box_xyxy, obs.frame_width, obs.frame_height)
        declared = crop.zone
        # A no-embed zone wins over everything: if either source says patio,
        # the row is a patio row.
        if declared in forbidden:
            zone = declared
        elif computed is not None and computed.name in forbidden:
            zone = computed.name
        else:
            zone = computed.name if computed is not None else declared
        embedding = crop.embedding
        embedding_ref = crop.embedding_ref
        if zone in forbidden or fail_closed:
            if embedding is not None or embedding_ref is not None:
                logger.warning(
                    "vision_crop_embedding_dropped observation_id=%s index=%s zone=%s "
                    "(producer sent an embedding for a no-embed zone)",
                    obs.observation_id, idx, zone,
                )
            embedding = None
            embedding_ref = None
        rows.append({
            "crop_id": f"{obs.observation_id}:{idx}",
            "observation_id": obs.observation_id,
            "stream_id": obs.stream_id,
            "camera_id": obs.camera_id,
            "artifact_id": obs.artifact_id,
            "observed_at": obs.observed_at,
            "label": crop.label,
            "score": float(crop.score),
            "box_xyxy": [float(v) for v in crop.box_xyxy],
            "zone": zone,
            "embedding_ref": embedding_ref,
            "embedding": [float(v) for v in embedding] if embedding is not None else None,
        })
    return rows


_INSERT_SQL = """
    INSERT INTO vision_crop_observation
        (crop_id, observation_id, stream_id, camera_id, artifact_id, observed_at,
         label, score, box_xyxy, zone, embedding_ref, embedding)
    VALUES
        (:crop_id, :observation_id, :stream_id, :camera_id, :artifact_id, :observed_at,
         :label, :score, :box_xyxy, :zone, :embedding_ref, :embedding)
    ON CONFLICT (crop_id) DO NOTHING
"""


def persist_crop_observation(payload: Dict[str, Any]) -> int:
    """Validate, fan out, insert. Returns rows attempted. Raises on DB error."""
    from sqlalchemy import text

    from app.db import get_session, remove_session

    obs = VisionCropObservationV1.model_validate(payload)
    zones = _zones_by_stream()
    rows = build_crop_rows(obs, None if zones is None else zones.get(obs.stream_id, []))
    if not rows:
        return 0
    sess = get_session()
    try:
        sess.execute(text(_INSERT_SQL), rows)
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        try:
            sess.close()
        finally:
            remove_session()
    return len(rows)
