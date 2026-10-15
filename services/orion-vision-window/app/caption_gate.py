"""Detector-agreement gate for camera captions (2026-10-15, walkway).

A caption on a gated stream becomes part of the window -- and so of the
council's narrative, ``vision_events``, the chat situation brief's street
line, reveries -- only if the detector saw at least one non-stationary
object of a gate label (person, dog, bicycle, stroller, vehicle, package,
mail truck) in the frame the caption describes.

Why: live 2026-10-15 the walkway's BLIP captioner described a dark, empty
frame as "A camera is placed on a wooden bench in a garden", and that line
reached Orion's situation brief as what was on the walkway. Captions now come
from Qwen2-VL (vision-frame-router caption_dispatch), but no captioner gets
to narrate a frame the detector found nobody in.

Evidence used: ``inputs.source_objects`` -- the detections of the caption's
own frame, attached by the frame router when it asked for the caption. A
caption without them (e.g. the athena host's BLIP, if ``want_caption`` were
turned back on for the stream) is rejected. Stationary = the box overlaps a
run held for ``VISION_STATIONARY_MIN_SEC`` (``app/fixtures.py``), tracked
here per gate label; the walkway's Halloween skeleton is the motivating
prop.

A rejected caption is kept on the artifact as rejected -- text "",
confidence 0.0, ``rejected_reason="gate:<reason>"`` -- never as an empty
caption at full confidence, and each rejection is counted and logged.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from orion.schemas.vision import VisionArtifactPayload, VisionCaption

from .fixtures import FixtureTracker

Item = Tuple[VisionArtifactPayload, float]


def _csv(value: Any) -> List[str]:
    raw = str(value or "").split(",") if isinstance(value, str) or value is None else list(value)
    return [str(x).strip().lower() for x in raw if str(x).strip()]


class CaptionGate:
    def __init__(
        self,
        *,
        streams: Iterable[str] | str,
        labels: Iterable[str] | str,
        min_score: float,
        max_age_sec: float,
        stationary_min_sec: float,
        stationary_iou: float,
        stationary_max_gap_sec: float,
    ) -> None:
        self.streams = set(_csv(streams))
        self.labels = list(dict.fromkeys(_csv(labels)))
        self.min_score = float(min_score)
        self.max_age_sec = float(max_age_sec)
        self._fixtures: Dict[str, FixtureTracker] = {
            label: FixtureTracker(
                streams=self.streams,
                min_sec=stationary_min_sec,
                iou=stationary_iou,
                max_gap_sec=stationary_max_gap_sec,
                label=label,
            )
            for label in self.labels
        }

    def enabled_for(self, stream_id: str) -> bool:
        return bool(self.labels) and str(stream_id or "").strip().lower() in self.streams

    def _observe(self, stream_id: str, items: Sequence[Item]) -> None:
        """Feed this window's detections to the per-label prop trackers."""
        per_label: Dict[str, List[Tuple[float, List[List[float]]]]] = {l: [] for l in self.labels}
        for art, ts in items:
            objects = art.outputs.objects if art.outputs else None
            if objects is None:  # caption-only / embed-only: not a detection pass
                continue
            boxes: Dict[str, List[List[float]]] = {l: [] for l in self.labels}
            for o in objects:
                label = str(o.label or "").strip().lower()
                if label in boxes and float(o.score) >= self.min_score and len(o.box_xyxy or []) == 4:
                    boxes[label].append([float(v) for v in o.box_xyxy])
            for label in self.labels:
                per_label[label].append((float(ts), boxes[label]))
        for label, frames in per_label.items():
            if frames:
                self._fixtures[label].observe_window(stream_id, frames)

    def _verdict(self, stream_id: str, art: VisionArtifactPayload, ts: float) -> Optional[str]:
        """None = accept; otherwise the rejection reason."""
        inputs = art.inputs or {}
        source = inputs.get("source_objects")
        if not isinstance(source, list):
            return "no_source_detections"
        frame_ts = inputs.get("frame_ts")
        try:
            if frame_ts is not None and (float(ts) - float(frame_ts)) > self.max_age_sec:
                return "stale"
        except (TypeError, ValueError):
            pass
        candidates = []
        for o in source:
            if not isinstance(o, dict):
                continue
            label = str(o.get("label") or "").strip().lower()
            box = o.get("box_xyxy") or []
            try:
                score = float(o.get("score") or 0.0)
            except (TypeError, ValueError):
                continue
            if label in self._fixtures and score >= self.min_score and len(box) == 4:
                candidates.append((label, box))
        if not candidates:
            return "no_gate_detection"
        if all(self._fixtures[label].is_fixture_box(stream_id, box) for label, box in candidates):
            return "stationary_only"
        return None

    def apply(
        self, stream_id: str, items: List[Item]
    ) -> Tuple[List[Item], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Returns (items with failing captions marked rejected, this window's
        gate counts or None when the stream is not gated / had no caption,
        the individual rejections for the log).

        The counts go into the window summary; the rejected TEXT never does
        -- the summary is what the council narrates from."""
        if not self.enabled_for(stream_id):
            return items, None, []
        self._observe(stream_id, items)
        accepted = 0
        rejected: Counter = Counter()
        rejections: List[Dict[str, Any]] = []
        out: List[Item] = []
        for art, ts in items:
            cap = art.outputs.caption if art.outputs else None
            if cap is None:
                out.append((art, ts))
                continue
            text = str(cap.text or "").strip()
            if not text:
                # Already no caption (host-side null_answer / sanitizer reason).
                rejected[f"host:{cap.rejected_reason or 'empty'}"] += 1
                out.append((art, ts))
                continue
            reason = self._verdict(stream_id, art, ts)
            if reason is None:
                accepted += 1
                out.append((art, ts))
                continue
            rejected[reason] += 1
            rejections.append({"artifact_id": art.artifact_id, "reason": reason, "text": text[:160]})
            dead = VisionCaption(text="", confidence=0.0, rejected_reason=f"gate:{reason}")
            out.append((art.model_copy(update={"outputs": art.outputs.model_copy(update={"caption": dead})}), ts))
        if not accepted and not rejected:
            return out, None, rejections
        return out, {"accepted": accepted, "rejected": dict(rejected)}, rejections
