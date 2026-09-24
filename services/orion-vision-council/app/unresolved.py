"""Things the council looked at and could not name (VisionUnresolvedV1).

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 4.
Study material for curiosity, not an alert. Two deterministic triggers:

- ``no_label``: the host detector fired on the window (``detection_count > 0``)
  but no box cleared the window service's naming threshold, so
  ``summary.evidence.hard_labels`` is empty. Something was there; nothing
  could be named.
- ``council_uncertainty``: the council's own interpretation returned a
  non-empty ``uncertainties`` list.

A per-stream minimum interval keeps a noisy scene from flooding the table.
Pure module: no bus I/O, so it is testable without a running service.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

from orion.schemas.vision import (
    VisionSceneInterpretationV1,
    VisionUnresolvedV1,
    VisionWindowPayload,
)

MAX_EVIDENCE_REFS = 20
MAX_GUESS_LABELS = 5
MAX_UNCERTAINTIES_IN_TEXT = 3


def _evidence(window: VisionWindowPayload) -> dict:
    return (window.summary or {}).get("evidence") or {}


def _detection_count(window: VisionWindowPayload) -> int:
    summary = window.summary or {}
    try:
        return int(summary.get("detection_count") or 0)
    except (TypeError, ValueError):
        return 0


def _hard_labels(window: VisionWindowPayload) -> list[str]:
    return [str(x) for x in (_evidence(window).get("hard_labels") or []) if str(x).strip()]


def _guess_labels(window: VisionWindowPayload) -> list[str]:
    counts = (window.summary or {}).get("object_counts") or {}
    if not isinstance(counts, dict):
        return []
    return sorted(str(k) for k in counts.keys())[:MAX_GUESS_LABELS]


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
) -> Optional[VisionUnresolvedV1]:
    """Return a VisionUnresolvedV1 for this window, or None if nothing went unnamed.

    ``interpretation`` is None when the council did not run (stable-scene gate
    or LLM failure); only the ``no_label`` trigger can fire then.
    """
    uncertainties = list(interpretation.uncertainties) if interpretation is not None else []
    detections = _detection_count(window)
    no_label = detections > 0 and not _hard_labels(window)
    if not uncertainties and not no_label:
        return None

    where = _where(window)
    guesses = _guess_labels(window)
    captions = [str(c) for c in ((window.summary or {}).get("captions") or []) if str(c).strip()]

    tried: list[str] = []
    if detections > 0:
        tried.append(
            f"object detector on the vision host: {detections} detection(s)"
            + (f", best guesses {', '.join(guesses)}" if guesses else "")
            + (", none confident enough to name" if no_label else "")
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
            f"The detector fired {detections} time(s) but nothing was confident enough to call it anything"
            + (f" (closest guesses: {', '.join(guesses)})" if guesses else "")
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
