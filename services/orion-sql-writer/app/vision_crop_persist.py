"""Persist ``vision.crop.observation.v1`` -- one bus event, one row per crop.

The generic ``MODEL_MAP`` path writes one row per envelope. A crop
observation fans out to N rows in ``vision_crop_observation`` (crop_id =
``<observation_id>:<index>``), so it gets its own handler, the same way
``chat.history.spark_meta.patch.v1`` does.

**No-embed zones, defense in depth.** The producer (orion-vision-host /
window) is supposed to skip embeddings for boxes in a no-embed zone. This
module does not trust that: it recomputes each box's zone from
``config/vision_zones.yaml`` and drops ``embedding``/``embedding_ref``,
``thumb_ref``, the face fields and the OSNet vector if EITHER the producer-declared zone or the
recomputed zone is a no-embed zone, and it sets ``zone_no_embed`` from the
config -- never from the zone's name. No configured zone is no-embed since
2026-10-15 (Juniper lifted the patio rule; manual_migration_walkway_privacy_v2.sql
dropped the DB CHECKs), so this is a generic mechanism nothing uses today.

``build_crop_rows`` is pure (zones passed in); ``persist_crop_observation``
is the blocking DB write, called from a worker thread.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from orion.schemas.vision import VisionCropObservationV1
from orion.vision.zones import Zone, intersects_no_embed, load_zones, zone_for_box

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
        thumb_ref = crop.thumb_ref
        face_embedding = crop.face_embedding
        face_thumb_ref = crop.face_thumb_ref
        face_score = crop.face_score
        osnet_embedding = crop.osnet_embedding
        zone_no_embed = zone in forbidden
        # A box placed outside the patio by its bottom-center can still hold
        # patio pixels: any overlap with a no-embed zone strips the vector and
        # thumbnail too (without frame size it cannot be checked -> strip).
        # zone_no_embed stays "placed in the zone", so presence counts do not
        # pick up passers-by at the patio's edge.
        touches = bool(forbidden) and intersects_no_embed(
            zones, crop.box_xyxy, obs.frame_width or 0, obs.frame_height or 0)
        if zone_no_embed or fail_closed or touches:
            if any(v is not None for v in (embedding, embedding_ref, thumb_ref, face_embedding, face_thumb_ref,
                                           osnet_embedding)):
                logger.warning(
                    "vision_crop_embedding_dropped observation_id=%s index=%s zone=%s "
                    "(producer sent an embedding for a no-embed zone)",
                    obs.observation_id, idx, zone,
                )
            embedding = None
            embedding_ref = None
            thumb_ref = None
            face_embedding = None
            face_thumb_ref = None
            face_score = None
            osnet_embedding = None
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
            "thumb_ref": thumb_ref,
            "zone_no_embed": zone_no_embed,
            "face_embedding": [float(v) for v in face_embedding] if face_embedding is not None else None,
            "face_score": float(face_score) if face_score is not None else None,
            "face_thumb_ref": face_thumb_ref,
            # The body model that made ``embedding`` (NULL with no embedding),
            # and the shadow OSNet vector (manual_migration_vision_osnet_shadow_v1.sql).
            # Nothing in the individuals reducer reads either.
            "embedding_model_id": crop.embedding_model_id if embedding is not None else None,
            "osnet_embedding": [float(v) for v in osnet_embedding] if osnet_embedding is not None else None,
            "osnet_model_id": crop.osnet_model_id if osnet_embedding is not None else None,
            # The frame the detector saw (manual_migration_vision_motion_story_edges_v1.sql):
            # the motion stories measure each box against its own camera's frame.
            "frame_width": int(obs.frame_width) if obs.frame_width else None,
            "frame_height": int(obs.frame_height) if obs.frame_height else None,
        })
    return rows


_INSERT_TMPL = """
    INSERT INTO vision_crop_observation
        (crop_id, observation_id, stream_id, camera_id, artifact_id, observed_at,
         label, score, box_xyxy, zone, embedding_ref, embedding, thumb_ref, zone_no_embed,
         face_embedding, face_score, face_thumb_ref,
         embedding_model_id, osnet_embedding, osnet_model_id{cols})
    VALUES
        (:crop_id, :observation_id, :stream_id, :camera_id, :artifact_id, :observed_at,
         :label, :score, :box_xyxy, :zone, :embedding_ref, :embedding, :thumb_ref, :zone_no_embed,
         :face_embedding, :face_score, :face_thumb_ref,
         :embedding_model_id, :osnet_embedding, :osnet_model_id{vals})
    ON CONFLICT (crop_id) DO NOTHING
"""
_INSERT_SQL = _INSERT_TMPL.format(cols=", frame_width, frame_height", vals=", :frame_width, :frame_height")
# Before manual_migration_vision_motion_story_edges_v1.sql: the frame size is dropped.
_INSERT_SQL_NO_FRAME = _INSERT_TMPL.format(cols="", vals="")

# Whether vision_crop_observation has frame_width/frame_height. Present is
# cached for the process; absent is re-checked every FRAME_COLUMNS_RECHECK_SEC,
# so applying the migration takes effect without a restart.
FRAME_COLUMNS_RECHECK_SEC = 300.0
_frame_columns: Dict[str, Any] = {"present": None, "checked_at": 0.0}


def _frame_columns_present(sess) -> bool:
    from sqlalchemy import text

    st = _frame_columns
    if st["present"] is True:
        return True
    now = time.monotonic()
    if st["present"] is False and now - st["checked_at"] < FRAME_COLUMNS_RECHECK_SEC:
        return False
    n = sess.execute(text(
        "SELECT count(*) FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = 'vision_crop_observation' AND column_name IN ('frame_width', 'frame_height')")).scalar()
    st["present"], st["checked_at"] = (int(n or 0) == 2), now
    if not st["present"]:
        logger.error("vision_crop_observation has no frame_width/frame_height -- apply services/orion-sql-db/"
                     "manual_migration_vision_motion_story_edges_v1.sql; crops are written without frame size")
    return bool(st["present"])


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
        sql = _INSERT_SQL if _frame_columns_present(sess) else _INSERT_SQL_NO_FRAME
        sess.execute(text(sql), rows)
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
