"""Prime's ``tending`` lines (personality spec rev 4, move 4; rev 3 "Read-back").

What Orion tends to choose, read back from the personality graph (durable-runs
GET /personality/frame), at most 2 lines ``tending: <label> [disposition:<id>]``. The same lines go
into the Prime block (orion/prime/block.py, every unified turn) and into Prime's own thought frame
(orion/prime/thought.py), so Prime picks with its tendencies in view and the next pick tests them.

Shown only: rev 4 choice-built nodes (origin "choice"), active, with at least ``MIN_WEIGHT``
weighted choices (the node's ``weight``: the faded sum of its choices' evidence weights, 0.25 for
following the market's order, 1.0 for departing). No shadow (Juniper, 2026-11-13): the line lights
at 10 weighted choices; her "no" in the Hub tab retires a tendency (the frame already leaves out
rejected nodes). A frame older than ``MAX_AGE_SEC`` is not read as current.

Pure: ``tendencies_from_frame`` takes the raw frame dict the caller fetched. Never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from orion.schemas.personality import PersonalityFrameV1

FRAME_PATH = "/personality/frame"
MIN_WEIGHT = 10.0
MAX_LINES = 2
# The reducer ticks every 600 s; an hour without a committed tick is stale (same as orion/personality/cue.py).
MAX_AGE_SEC = 3600.0
LABEL_CHARS = 120
REF_KIND = "disposition"


@dataclass(frozen=True)
class TendingV1:
    disposition_id: str
    label: str
    weight: float

    @property
    def ref(self) -> str:
        return f"{REF_KIND}:{self.disposition_id}"

    def line(self) -> str:
        return f"tending: {self.label} [{self.ref}]"


def _label(text: str) -> str:
    # No ref-shaped text from the label: a "[thought:..]" inside it is not a ref the block showed.
    text = " ".join((text or "").split()).replace("[", "(").replace("]", ")")
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 1].rstrip() + "…"


def tendencies_from_frame(raw: Any, *, now: Optional[datetime] = None, min_weight: float = MIN_WEIGHT,
                          max_lines: int = MAX_LINES,
                          max_age_sec: float = MAX_AGE_SEC) -> tuple[list[TendingV1], Optional[str]]:
    """(the tendencies to show, why none). Why is ``invalid_frame``, ``stale`` or ``below_threshold``."""
    try:
        frame = raw if isinstance(raw, PersonalityFrameV1) else PersonalityFrameV1.model_validate(raw)
    except Exception:  # noqa: BLE001
        return [], "invalid_frame"
    now = now or datetime.now(timezone.utc)
    if (now - frame.generated_at).total_seconds() > max_age_sec:
        return [], "stale"
    picked = sorted((d for d in frame.dispositions
                     if d.origin == "choice" and d.status == "active" and d.weight >= min_weight),
                    key=lambda d: (-d.weight, d.disposition_id))[:max_lines]
    if not picked:
        return [], "below_threshold"
    return [TendingV1(d.disposition_id, _label(d.label), d.weight) for d in picked], None


__all__ = ["FRAME_PATH", "MAX_LINES", "MIN_WEIGHT", "REF_KIND", "TendingV1", "tendencies_from_frame"]
