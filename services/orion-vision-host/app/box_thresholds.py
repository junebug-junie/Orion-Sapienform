"""Per-camera, per-label box cut-offs for the open-vocabulary detector.

Why (2026-10-29): one ``score_threshold`` (0.35) served every camera and every
label. On 2026-10-22..23 the walkway's parked-car scores sat at a median of
~0.36 (they were ~0.40 on 2026-10-28), so most frames fell under 0.35 and the
walkway went half-blind while cam0 was fine. A camera-wide or label-wide
value cannot be tuned for one without moving the other.

Profile params (``config/vision_profiles.yaml``, ``retina_detect_open_vocab``)::

    score_threshold: 0.35          # default for every stream and label
    box_threshold_by_stream:
      walkway:
        default: 0.35              # optional: this stream's default
        vehicle: 0.35              # optional: one label on this stream

The request's ``stream_id`` (set by the frame router) picks the entry. The
detector runs once at the lowest cut-off that applies to the request, then
each box is kept only if its score reaches the cut-off for its own label.
With every value equal, the result is identical to the old single threshold.

Labels match the detector's returned phrase after strip + lower. A phrase that
matches no entry (GroundingDINO can return merged phrases) gets the stream's
default, i.e. today's behaviour.

This module only selects and applies cut-offs. Duplicate suppression (NMS,
tile merge) stays in the runner and is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

from loguru import logger

_FALLBACK_DEFAULT = 0.25  # the runner's historical default when the profile sets nothing


def _norm(label: Any) -> str:
    return str(label or "").strip().lower()


def _valid(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if 0.0 < v < 1.0 else None


@dataclass(frozen=True)
class BoxThresholds:
    default: float
    by_label: Dict[str, float] = field(default_factory=dict)

    @property
    def floor(self) -> float:
        """The cut-off the detector runs at: the lowest one that applies."""
        return min([self.default, *self.by_label.values()])

    def for_label(self, label: Any) -> float:
        return self.by_label.get(_norm(label), self.default)


def resolve_box_thresholds(params: Mapping[str, Any], stream_id: Optional[str]) -> BoxThresholds:
    """Cut-offs for one request. Invalid values are logged and ignored (the
    profile default stands), so a typo can never lower a cut-off to 0."""
    raw_base = params.get("box_threshold", params.get("score_threshold", _FALLBACK_DEFAULT))
    base = _valid(raw_base)
    if base is None:
        logger.warning(f"[DETECT] invalid profile box threshold {raw_base!r}; using {_FALLBACK_DEFAULT}")
    default = base if base is not None else _FALLBACK_DEFAULT
    table = params.get("box_threshold_by_stream") or {}
    entry = table.get(str(stream_id)) if (stream_id and isinstance(table, Mapping)) else None
    if not isinstance(entry, Mapping):
        return BoxThresholds(default)
    by_label: Dict[str, float] = {}
    for key, raw in entry.items():
        v = _valid(raw)
        if v is None:
            logger.warning(f"[DETECT] ignoring invalid box threshold {stream_id}.{key}={raw!r}")
            continue
        if _norm(key) == "default":
            default = v
        else:
            by_label[_norm(key)] = v
    return BoxThresholds(default, by_label)


def apply_box_thresholds(objects: Iterable[Dict[str, Any]], th: BoxThresholds) -> List[Dict[str, Any]]:
    """Keep each box whose score reaches its own label's cut-off."""
    return [o for o in objects if float(o.get("score") or 0.0) >= th.for_label(o.get("label"))]
