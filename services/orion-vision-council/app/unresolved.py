"""Things the council looked at and could not name (VisionUnresolvedV1).

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 4.
Study material for curiosity, not an alert. Two deterministic triggers:

- ``no_label``: the detector drew a box but gave it no name -- an empty label
  (GroundingDINO returns "" when no prompt token clears ``text_threshold``) or
  the host runner's ``"object"`` fallback (services/orion-vision-host/app/
  runner.py). Read from ``summary.object_counts`` (per-frame max).

  Not "no box cleared the score threshold": the host already drops boxes
  below 0.25 (config/vision_profiles.yaml ``score_threshold``) before the
  window service applies its own 0.25 ``HARD_SCORE_THRESHOLD``, so every box
  that arrives is a hard label and that condition can never be true.
- ``council_uncertainty``: the council's own interpretation returned a
  non-empty ``uncertainties`` list.

A per-stream minimum interval keeps a noisy scene from flooding the table.
Pure module: no bus I/O, so it is testable without a running service.

**Cameras with a no-embed zone (the walkway's patio).** The council's
uncertainties and the captions describe the WHOLE frame, patio included, and
``artifact_uris`` are whole frames. For any stream that has an ``embed:
false`` zone in ``config/vision_zones.yaml`` (``orion.vision.zones``) this
module therefore never emits ``council_uncertainty``, never quotes a caption,
never names other detected labels, and never sets ``image_ref``. It emits
``no_label`` only, counted from the unnamed boxes whose zone is a real,
embeddable zone (``summary.unnamed_boxes``, placed by the window service);
a box in a no-embed zone, outside every zone, or with no frame size is left
out entirely. If the zones file cannot be read, every stream is treated this
way (fail closed). Other cameras keep the full behaviour.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, Iterable, List, Optional

from orion.schemas.vision import (
    VisionSceneInterpretationV1,
    VisionUnresolvedV1,
    VisionWindowPayload,
)

MAX_EVIDENCE_REFS = 20
MAX_GUESS_LABELS = 5
MAX_UNCERTAINTIES_IN_TEXT = 3

logger = logging.getLogger(__name__)

_ZONES: Optional[Dict[str, list]] = None
_ZONES_LOADED = False
_FROM_CONFIG = object()


def _load_zones_once() -> Optional[Dict[str, list]]:
    """stream -> zones, or None when the zones file is missing/unreadable
    (fail closed: every stream is then treated as having a no-embed zone)."""
    global _ZONES, _ZONES_LOADED
    if _ZONES_LOADED:
        return _ZONES
    try:
        from orion.vision.zones import DEFAULT_ZONES_PATH, load_zones
        import os
        from pathlib import Path

        path = Path(os.getenv("VISION_ZONES_PATH") or DEFAULT_ZONES_PATH)
        if not path.exists():
            logger.warning("vision zones file missing path=%s; unresolved percepts fail closed", path)
            _ZONES = None
        else:
            _ZONES = load_zones(path)
    except Exception as exc:  # malformed yaml must not open the patio up
        logger.warning("vision zones unreadable err=%s; unresolved percepts fail closed", exc)
        _ZONES = None
    _ZONES_LOADED = True
    return _ZONES


def _guarded_zones(window: VisionWindowPayload, zones_by_stream: Optional[Dict[str, list]]) -> Optional[list]:
    """The stream's zones if it has a no-embed zone (or zones are unknown ->
    []), else None meaning "not guarded, full behaviour"."""
    if zones_by_stream is None:
        return []
    zones = list(zones_by_stream.get(str(window.stream_id or "")) or [])
    if any(not z.embed for z in zones):
        return zones
    return None


def _embeddable_unnamed_count(window: VisionWindowPayload, zones: list) -> int:
    """Per-frame max of unnamed boxes that sit in a real, embeddable zone."""
    from orion.vision.zones import zone_for_box

    raw = (window.summary or {}).get("unnamed_boxes") or []
    per_frame: Dict[str, int] = {}
    for box in raw if isinstance(raw, list) else []:
        if not isinstance(box, dict):
            continue
        try:
            xyxy = [float(v) for v in box.get("box_xyxy") or []]
            w = float(box.get("frame_width") or 0)
            h = float(box.get("frame_height") or 0)
        except (TypeError, ValueError):
            continue
        zone = zone_for_box(zones, xyxy, w, h) if zones else None
        if zone is None or not zone.embed:
            continue  # no-embed zone, outside every zone, or unplaceable
        key = str(box.get("frame") or "")
        per_frame[key] = per_frame.get(key, 0) + 1
    return max(per_frame.values()) if per_frame else 0


def _evidence(window: VisionWindowPayload) -> dict:
    return (window.summary or {}).get("evidence") or {}


def _detection_count(window: VisionWindowPayload) -> int:
    summary = window.summary or {}
    try:
        return int(summary.get("detection_count") or 0)
    except (TypeError, ValueError):
        return 0


UNNAMED_LABELS = frozenset({"", "object"})


def _object_counts(window: VisionWindowPayload) -> dict:
    counts = (window.summary or {}).get("object_counts") or {}
    return counts if isinstance(counts, dict) else {}


def _unnamed_count(window: VisionWindowPayload) -> int:
    total = 0
    for label, n in _object_counts(window).items():
        if str(label).strip().lower() in UNNAMED_LABELS:
            try:
                total += int(n)
            except (TypeError, ValueError):
                continue
    return total


def _named_labels(window: VisionWindowPayload) -> list[str]:
    names = [str(k) for k in _object_counts(window) if str(k).strip().lower() not in UNNAMED_LABELS]
    return sorted(names)[:MAX_GUESS_LABELS]


def _where(window: VisionWindowPayload) -> str:
    return window.stream_id or window.camera_id or "the camera"


def _dedupe(items: Iterable[str], cap: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def unresolved_id_for(window_id: str, reason: str) -> str:
    """Deterministic, so a redelivered window cannot mint a second row."""
    digest = hashlib.sha256(f"{window_id}|{reason}".encode("utf-8")).hexdigest()[:24]
    return f"vu-{digest}"


def build_unresolved(
    window: VisionWindowPayload,
    interpretation: Optional[VisionSceneInterpretationV1],
    *,
    council_model: str,
    council_route: str,
    zones_by_stream: Optional[Dict[str, list]] | object = _FROM_CONFIG,
) -> Optional[VisionUnresolvedV1]:
    """Return a VisionUnresolvedV1 for this window, or None if nothing went unnamed.

    ``interpretation`` is None when the council did not run (stable-scene gate
    or LLM failure); only the ``no_label`` trigger can fire then.

    ``zones_by_stream`` defaults to ``config/vision_zones.yaml``; ``None``
    means the zones are unknown, which fails closed (every stream guarded).
    """
    if zones_by_stream is _FROM_CONFIG:
        zones_by_stream = _load_zones_once()
    guarded = _guarded_zones(window, zones_by_stream)
    if guarded is not None:
        return _build_guarded(window, guarded)

    uncertainties = list(interpretation.uncertainties) if interpretation is not None else []
    detections = _detection_count(window)
    unnamed = _unnamed_count(window)
    no_label = unnamed > 0
    if not uncertainties and not no_label:
        return None

    where = _where(window)
    named = _named_labels(window)
    captions = [str(c) for c in ((window.summary or {}).get("captions") or []) if str(c).strip()]

    tried: list[str] = []
    if detections > 0:
        tried.append(
            f"object detector on the vision host: {detections} detection(s)"
            + (f", named: {', '.join(named)}" if named else "")
            + (f", {unnamed} box(es) it could not put a name to" if no_label else "")
        )
    if captions:
        tried.append(f"caption model said: {captions[0][:200]!r}")
    if interpretation is not None:
        tried.append(f"scene interpretation by {council_model} (route {council_route})")

    evidence_refs = list(window.artifact_ids or [])
    if uncertainties:
        reason = "council_uncertainty"
        parts = []
        for u in uncertainties[:MAX_UNCERTAINTIES_IN_TEXT]:
            text = u.uncertainty.strip()
            if u.reason:
                text = f"{text} ({u.reason.strip()})"
            parts.append(text)
            evidence_refs.extend(u.evidence_refs or [])
        description = f"Looking at {where}, I was not sure about: " + "; ".join(parts) + "."
    else:
        reason = "no_label"
        description = (
            f"Something showed up on {where} that I could not name. "
            f"The detector drew a box around {unnamed} thing(s) but could not say what they were"
            + (f" (things I could name in the same view: {', '.join(named)})" if named else "")
            + "."
        )

    image_ref = (window.artifact_uris or [None])[0]
    return VisionUnresolvedV1(
        unresolved_id=unresolved_id_for(window.window_id, reason),
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        window_id=window.window_id,
        reason=reason,
        description=description,
        what_was_tried=tried,
        evidence_refs=_dedupe(evidence_refs, MAX_EVIDENCE_REFS),
        image_ref=image_ref,
    )


def _build_guarded(window: VisionWindowPayload, zones: list) -> Optional[VisionUnresolvedV1]:
    """A camera that sees a no-embed zone: say only what is known about the
    unnamed boxes outside it. No free text from the council, no captions, no
    other labels, no frame."""
    count = _embeddable_unnamed_count(window, zones)
    if count <= 0:
        return None
    where = _where(window)
    return VisionUnresolvedV1(
        unresolved_id=unresolved_id_for(window.window_id, "no_label"),
        stream_id=window.stream_id,
        camera_id=window.camera_id,
        window_id=window.window_id,
        reason="no_label",
        description=(
            f"Something showed up on the {where} camera that I could not name. "
            f"The detector drew a box around {count} thing(s) there but could not say what they were."
        ),
        what_was_tried=[f"object detector on the vision host: {count} box(es) it could not put a name to"],
        evidence_refs=_dedupe(window.artifact_ids or [], MAX_EVIDENCE_REFS),
        image_ref=None,
    )


class UnresolvedRateLimiter:
    """At most one unresolved percept per stream per ``min_interval_sec``.

    In-memory on purpose: a restart can let one extra row through per stream,
    which is harmless for study material (unlike an outreach cap).
    """

    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last: dict[str, float] = {}

    def allow(self, stream_key: str, now: float) -> bool:
        last = self._last.get(stream_key)
        return last is None or (now - last) >= self.min_interval_sec

    def mark(self, stream_key: str, now: float) -> None:
        self._last[stream_key] = now
