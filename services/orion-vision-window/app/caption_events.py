"""Accepted camera captions -> `vision_events` rows (2026-10-16, walkway).

Why this exists: live 2026-10-16 the caption gate accepted its first Qwen
caption ("A person is pushing a stroller on a sidewalk ...", 22:53 UTC) and it
went nowhere. The gate puts an accepted caption into `summary.captions`, and
the only reader of that field is orion-vision-council -- whose transition gate
skips every window whose believed labels did not change (`stable_scene`). The
caption window was skipped, so no `vision_events` row was written, so the
street line (`orion.situational.perception_reader.fetch_street_summary`),
reverie, dreams and curiosity -- which all read `vision_events` -- never saw
it.

So the window writes each accepted caption itself, as one `vision_events` row
on the existing scribe path (`orion:vision:events` -> orion-vision-scribe ->
`orion:vision:events:sql-write` -> orion-sql-writer):

* ``event_type = "camera_caption"`` -- readers label it as a caption, never as
  a checked fact (`orion.situational.street_material`). Qwen2-VL reads one
  frame and still invents motion ("is pushing", "walks"), PR #2737.
* ``narrative`` = the caption text, unedited apart from whitespace/length.
* ``tags`` = ``camera_caption``, ``motion_unverified``, and one
  ``detector:<label>`` per moving gate-label box that let it through.
* ``evidence_refs`` = artifact id, frame path, frame time, each moving box,
  the crop boxes the VLM read, and the captioning model.
* ``entities`` stays empty: a walkway caption names no one.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.schemas.vision import VisionEventBundleItem

CAMERA_CAPTION_EVENT_TYPE = "camera_caption"
CAPTION_TAG_MOTION_UNVERIFIED = "motion_unverified"
NARRATIVE_MAX_CHARS = 400
# A caption is a model's description of one frame, not a detection: same cap
# the council puts on caption-inferred activity (evidence_grounding.py).
CAPTION_EVENT_CONFIDENCE = 0.4
CAPTION_EVENT_SALIENCE = 0.3


def caption_event_id(artifact_id: str) -> str:
    """Stable per artifact: a re-flushed caption cannot write a second row."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:camera_caption:{artifact_id}"))


def _iso(ts: Any) -> Optional[str]:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _box(box: List[Any]) -> str:
    return ",".join(str(int(round(float(v)))) for v in box)


def build_caption_event(record: Dict[str, Any], *, stream_id: str) -> Optional[VisionEventBundleItem]:
    """One accepted-caption record (`CaptionGate._accepted_record`) -> one
    `vision_events` row. None when there is no text (never an empty row)."""
    text = " ".join(str(record.get("text") or "").split())
    if not text:
        return None
    if len(text) > NARRATIVE_MAX_CHARS:
        text = text[: NARRATIVE_MAX_CHARS - 1].rstrip() + "…"
    labels = [str(l) for l in record.get("labels") or [] if str(l).strip()]
    refs: List[str] = [f"artifact:{record.get('artifact_id')}"]
    if record.get("image_path"):
        refs.append(f"frame:{record['image_path']}")
    frame_iso = _iso(record.get("frame_ts"))
    if frame_iso:
        refs.append(f"frame_ts:{frame_iso}")
    for label, box in record.get("boxes") or []:
        refs.append(f"box:{label}:{_box(box)}")
    for box in record.get("crop_boxes") or []:
        if isinstance(box, (list, tuple)) and len(box) == 4:
            refs.append(f"crop:{_box(box)}")
    if record.get("model"):
        refs.append(f"model:{record['model']}")
    return VisionEventBundleItem(
        event_id=caption_event_id(str(record.get("artifact_id") or "")),
        event_type=CAMERA_CAPTION_EVENT_TYPE,
        narrative=text,
        entities=[],
        tags=[CAMERA_CAPTION_EVENT_TYPE, CAPTION_TAG_MOTION_UNVERIFIED, *(f"detector:{l}" for l in labels)],
        confidence=CAPTION_EVENT_CONFIDENCE,
        salience=CAPTION_EVENT_SALIENCE,
        evidence_refs=refs,
        stream_id=stream_id,
    )
