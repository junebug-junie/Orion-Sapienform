"""Patch-4 fixtures: real temporal-self rows compacted by patch4_rows/build_fixtures.py (see its docstring)."""
from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.temporal_self import TemporalSelfDayV1

DATA = Path(__file__).parent / "patch4_rows"
# The real frame was fetched at this moment; the busy frame is the same day rebuilt at 17:20Z.
TALKING_NOW = datetime(2026, 10, 11, 23, 58, 34, tzinfo=timezone.utc)
BUSY_NOW = datetime(2026, 10, 11, 17, 20, tzinfo=timezone.utc)


def frame(name: str) -> dict:
    return json.loads((DATA / f"frame_2026-10-11_{name}.json").read_text())


def rest_frame() -> dict:
    """Just after local midnight: a fresh day with no arcs yet. A real rest state."""
    f = copy.deepcopy(frame("talking"))
    f.update(as_of="2026-10-11T06:04:00Z", day_phase="pre_dawn", active_arc=None, previous_arc=None,
             active_by_kind={}, arcs_today=[], arcs_today_total=0, open_threads=[], expectations_pending=[],
             expectations_resolved_today=[], expectations_resolved_total=0, constraint_event_ids=[],
             sleep_arc_ids=[], warnings=[])
    return f


def day_1009() -> TemporalSelfDayV1:
    return TemporalSelfDayV1.model_validate(json.loads(gzip.decompress((DATA / "day_2026-10-09.json.gz").read_bytes())))
