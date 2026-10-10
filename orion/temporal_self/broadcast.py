"""Broadcast log rows -> ``BroadcastTickView``: the attention-arc driver's only input.

Subject rule, measured live 2026-10-10 over the 7 days held in
``substrate_attention_broadcast_log`` (14,872 rows): 6,544 ticks (44%) selected a loop, and
every selected loop had a non-empty ``source_refs`` (6,544 of 6,544), so the subject is
``source_refs[0]`` of the selected loop -- the loop-id fallback the spec allowed is not
needed. The other 56% are honest no-winner ticks (#2528's world-first seam can return
``winner=null``); they carry ``ref=None`` and are never invented into a subject.

``selected_open_loop_id`` sits at the projection's top level and the loops sit in
``frame.open_loops``. ``dwell_ticks`` and ``attended_node_ids`` are deliberately ignored:
dwell resets on every restart and is recomputed from the row sequence instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from orion.temporal_self.day import as_utc


@dataclass(frozen=True)
class BroadcastTickView:
    log_id: str
    generated_at: datetime  # UTC
    ref: str | None  # selected loop's source_refs[0]; None = no winner this tick
    label: str = ""  # the selected loop's own description (system-written, not chat text)


def tick_from_log_row(row: Mapping[str, Any]) -> BroadcastTickView:
    projection = row.get("projection_json")
    if isinstance(projection, (str, bytes)):
        projection = json.loads(projection)
    projection = projection or {}
    selected = projection.get("selected_open_loop_id")
    ref: str | None = None
    label = ""
    if selected:
        for loop in ((projection.get("frame") or {}).get("open_loops") or []):
            if isinstance(loop, dict) and loop.get("id") == selected:
                refs = loop.get("source_refs") or []
                ref = str(refs[0]) if refs else None
                label = str(loop.get("description") or "")[:300]
                break
    at = as_utc(row.get("generated_at"))
    assert at is not None, "broadcast row without generated_at"
    return BroadcastTickView(log_id=str(row["log_id"]), generated_at=at, ref=ref, label=label)
