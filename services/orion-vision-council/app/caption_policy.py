"""Which cameras may show caption text to the council.

A small caption model describes one still frame and guesses motion from it
("a car is driving" about cars that never moved). The window service already
writes each caption as its own labelled ``camera_caption`` row. When the same
text also reaches the council prompt, the council sometimes copies it into an
unlabelled ``visual_observation`` row, and readers show that row as fact.

So captions reach the council only for streams in ``COUNCIL_CAPTION_STREAMS``
(default ``cam0``, the room camera). For every other stream the council gets a
copy of the window with the caption material removed: no caption text, no
caption-derived ``soft_labels``, no ``caption_count``. Kill, not fence: there is
no soft-hint fallback. Spec: docs/superpowers/specs/
2026-10-20-walkway-followups-context-spec.md, item 3, patch A.

The stream key is the same one the evidence-transition gate uses
(``window.stream_id``, then ``window.camera_id``). orion-vision-window sets
``stream_id`` from the frame's ``inputs.stream_id`` / ``camera_id`` /
``clip_id`` / ``device``; a window with no stream key is not on the list and
gets no captions.
"""

from __future__ import annotations

from typing import Iterable

from orion.schemas.vision import VisionWindowPayload

from .evidence_transition import stream_key_from_window

DEFAULT_CAPTION_STREAMS: frozenset[str] = frozenset({"cam0"})

# Evidence keys built from caption text by orion-vision-window
# (projection._build_evidence): caption stoplist tokens and a caption count.
CAPTION_EVIDENCE_KEYS: tuple[str, ...] = ("soft_labels", "caption_count")

# Non-text marker set on a stripped window. Grounding reads it to keep the
# activity-claim confidence cap (a still frame cannot show activity), and
# intake reads it to skip windows that had nothing but a caption.
CAPTIONS_WITHHELD_KEY = "captions_withheld"


def parse_caption_streams(raw: str | Iterable[str] | None) -> frozenset[str]:
    """Comma list (or iterable) to a set of stream ids. Empty means no stream."""
    if raw is None:
        return frozenset()
    parts = raw.split(",") if isinstance(raw, str) else list(raw)
    return frozenset(str(p).strip().lower() for p in parts if str(p).strip())


def captions_admitted(window: VisionWindowPayload, caption_streams: frozenset[str]) -> bool:
    return stream_key_from_window(window).lower() in caption_streams


def window_for_council(
    window: VisionWindowPayload, caption_streams: frozenset[str]
) -> VisionWindowPayload:
    """The window the council may reason over.

    Unchanged for an admitted stream. For any other stream, a copy with
    ``summary.captions`` emptied and caption-derived evidence keys dropped.
    The input window is never mutated.
    """
    if captions_admitted(window, caption_streams):
        return window
    summary = dict(window.summary or {})
    if not summary:
        return window
    summary["captions"] = []
    summary[CAPTIONS_WITHHELD_KEY] = True
    evidence = summary.get("evidence")
    if isinstance(evidence, dict):
        summary["evidence"] = {
            k: v for k, v in evidence.items() if k not in CAPTION_EVIDENCE_KEYS
        }
    return window.model_copy(update={"summary": summary})


def captions_withheld(window: VisionWindowPayload) -> bool:
    return bool((window.summary or {}).get(CAPTIONS_WITHHELD_KEY))


def is_caption_only_after_strip(window: VisionWindowPayload) -> bool:
    """True when captions were withheld and the window had no detection pass.

    orion-vision-window publishes windows holding only a late caption artifact
    (``item_count`` 0; it counts detection passes only). With the caption gone
    there is nothing left to interpret; an LLM call would invent content.
    """
    if not captions_withheld(window):
        return False
    summary = window.summary or {}
    if summary.get("item_count") is None:
        # Not a window-service summary shape; do not guess it is empty.
        return False
    try:
        return int(summary.get("item_count") or 0) == 0
    except (TypeError, ValueError):
        return False
