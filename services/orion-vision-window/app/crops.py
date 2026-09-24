"""Host artifact -> VisionCropObservationV1 for the individuals reducer.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 1.
orion-vision-host marks tracked-label boxes (``want_crop_embeddings``) with a
``zone`` and, where the zone allows, an inline ``embedding``. Every such box is
forwarded, including patio boxes with no vector: they drive patio presence
counts in the reducer without Orion keeping a picture or a vector of anyone
there. Boxes the host did not mark (untracked labels, streams that did not ask)
are not forwarded, so this lane stays silent for cam0/carbon.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.vision import VisionArtifactPayload, VisionCropObservationV1, VisionCropV1
from orion.vision.stream_ids import is_url_like, safe_camera_name

from .projection import camera_id_from_artifact, stream_key_from_artifact


def _is_tracked(obj: Any) -> bool:
    return bool(getattr(obj, "embedding", None)) or bool(getattr(obj, "zone", None))


def _as_int(value: Any) -> Optional[int]:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def build_crop_observation(art: VisionArtifactPayload) -> Optional[VisionCropObservationV1]:
    objects = list(art.outputs.objects or []) if art.outputs else []
    crops = [
        VisionCropV1(
            label=o.label,
            score=float(o.score),
            box_xyxy=[float(x) for x in o.box_xyxy],
            zone=o.zone,
            # The host never sets a vector on a no-embed-zone box; the DB
            # CHECK constraint enforces the same. Nothing to strip here, and
            # stripping silently would hide a host bug instead of failing it.
            embedding_ref=o.embedding_ref,
            embedding=o.embedding,
        )
        for o in objects
        if _is_tracked(o)
    ]
    if not crops:
        return None

    inputs = art.inputs or {}
    raw_camera_id = camera_id_from_artifact(art)
    outputs_extra = art.outputs.model_extra or {}
    width = _as_int(outputs_extra.get("frame_width")) or _as_int(inputs.get("width"))
    height = _as_int(outputs_extra.get("frame_height")) or _as_int(inputs.get("height"))

    observed_at = datetime.now(timezone.utc)
    frame_ts = inputs.get("frame_ts")
    if isinstance(frame_ts, (int, float)) and frame_ts > 0:
        observed_at = datetime.fromtimestamp(float(frame_ts), tz=timezone.utc)

    return VisionCropObservationV1(
        # One observation per host artifact; the artifact id makes a redelivery
        # idempotent at the reducer.
        observation_id=f"cropobs:{art.artifact_id}",
        # Names only: an edge that still puts its RTSP source in camera_id
        # must not leak the camera password into the individuals tables.
        stream_id=safe_camera_name(inputs.get("stream_id"), stream_key_from_artifact(art)),
        camera_id=None if is_url_like(raw_camera_id) else raw_camera_id,
        artifact_id=art.artifact_id,
        observed_at=observed_at,
        frame_width=width,
        frame_height=height,
        crops=crops,
    )
