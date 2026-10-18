"""The walkway's newest day rollup as curiosity study material (2026-10-18).

Pinned: the newest day row of the configured street streams is read; it sits
alongside the 6 h captions (they stay); a running day is marked as running; a
failing read (table not migrated) costs only this card; no row = no section.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from orion.curiosity.kickoff_prompt import _material_section
from orion.curiosity.study_material import STREET_DAY_ROLLUP_SQL, StudyMaterial, build_street_day_card
from scripts.curiosity_investigation import CuriosityInvestigation

NOW = datetime(2026, 10, 18, 3, 0, tzinfo=timezone.utc)
LINE = "Walkway Sat 17 Oct 2026 (so far, through 21:00): 76 passings (35 people, 31 vehicles) by 70 distinct."


class _Conn:
    def __init__(self, day_row):
        self.day_row = day_row
        self.day_args = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if "FROM vision_events" in sql:
            return [{"event_id": "e1", "stream_id": "walkway", "event_type": "visual_observation",
                     "narrative": "A cyclist rides past.", "created_at": NOW}]
        if "GROUP BY m.kind" in sql:
            return [{"kind": "semantic", "n": 1, "manual_n": 1}]
        if "FROM memory_crystallizations" in sql and "random()" in sql:
            return [{"crystallization_id": "c1", "kind": "semantic", "subject": "s",
                     "summary": "s", "salience": 0.5, "created_at": NOW}]
        return []

    async def fetchrow(self, sql, *args):
        assert sql == STREET_DAY_ROLLUP_SQL
        self.day_args = args
        if isinstance(self.day_row, Exception):
            raise self.day_row
        return self.day_row

    async def fetchval(self, sql, *args):
        return 0


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return self._conn


def _loop(conn, streams=("walkway",)) -> CuriosityInvestigation:
    return CuriosityInvestigation(
        enabled=True, tick_interval_sec=60.0, min_cooldown_sec=14400.0, daily_cap=3,
        timeout_sec=1500.0, session_id="orion_curiosity", crystallization_sample=12,
        relation_sample=6, street_stream_ids=streams, pool_provider=lambda: _Pool(conn),
        source_ref="test",
    )


def test_newest_day_rollup_joins_the_material_next_to_the_captions() -> None:
    conn = _Conn({"stream_id": "walkway", "label": "2026-10-17", "complete": False, "line": LINE,
                  "computed_at": NOW})
    material = asyncio.run(_loop(conn)._read_study_material(NOW))
    assert conn.day_args == (["walkway"],)
    assert material.street_day is not None and material.street_day.line == LINE
    assert [c.event_id for c in material.street_captions] == ["e1"]  # the 6 h captions stay
    text = "\n".join(_material_section(material))
    assert "THE STREET OVER A WHOLE DAY (walkway camera, 2026-10-17; counted by code" in text
    assert f"  - {LINE}\n" in text + "\n"  # the line itself says "(so far, through 21:00)"
    assert text.index("WHAT THE STREET CAMERA SAID LATELY") < text.index("THE STREET OVER A WHOLE DAY")


def test_missing_table_or_no_row_costs_only_this_card() -> None:
    for row in (RuntimeError('relation "vision_rollup" does not exist'), None):
        material = asyncio.run(_loop(_Conn(row))._read_study_material(NOW))
        assert material.street_day is None and material.has_material
        assert "THE STREET OVER A WHOLE DAY" not in "\n".join(_material_section(material))


def test_no_street_streams_means_no_query() -> None:
    conn = _Conn(RuntimeError("must not be read"))
    material = asyncio.run(_loop(conn, streams=())._read_study_material(NOW))
    assert material.street_day is None and conn.day_args is None


def test_card_builder_refuses_blank_lines_and_marks_complete_days_plainly() -> None:
    assert build_street_day_card({"stream_id": "walkway", "label": "d", "complete": True, "line": "  "}) is None
    card = build_street_day_card({"stream_id": "walkway", "label": "2026-10-16", "complete": True, "line": "x."})
    assert card.preview() == "x."
    assert StudyMaterial(generated_at=NOW).street_day is None
