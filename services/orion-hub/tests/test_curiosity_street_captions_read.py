"""Recent walkway (street) captions as curiosity study material (2026-10-15).

Pinned: only the configured street streams are queried; newest distinct fresh
captions, capped; reducer rows skipped; a failing read costs only this section;
empty = no prompt section; the section is labelled as the street.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.curiosity.kickoff_prompt import _material_section
from orion.curiosity.study_material import StudyMaterial, build_street_caption_cards
from scripts.curiosity_investigation import CuriosityInvestigation

NOW = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc)


def _row(eid, text, minutes_ago, event_type="visual_observation"):
    return {"event_id": eid, "stream_id": "walkway", "event_type": event_type, "narrative": text,
            "created_at": NOW - timedelta(minutes=minutes_ago)}


class _Conn:
    def __init__(self, captions):
        self.captions = captions
        self.caption_args = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def fetch(self, sql, *args):
        if "FROM vision_events" in sql:
            self.caption_args = args
            if isinstance(self.captions, Exception):
                raise self.captions
            return self.captions
        if "GROUP BY m.kind" in sql:
            return [{"kind": "semantic", "n": 1, "manual_n": 1}]
        if "FROM memory_crystallizations" in sql and "random()" in sql:
            return [{"crystallization_id": "c1", "kind": "semantic", "subject": "s",
                     "summary": "s", "salience": 0.5, "created_at": NOW}]
        return []

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


def test_captions_are_read_from_street_streams_deduped_and_capped() -> None:
    conn = _Conn([
        _row("e1", "A van is parked by the curb.", 5),
        _row("e2", "a van is parked by the curb", 6),
        _row("e3", "The black dog came.", 1, event_type="arrived_as_expected"),
        _row("e4", "A cyclist rides past.", 30),
        _row("e5", "Two people talk at the gate.", 60),
        _row("e6", "A car turns the corner.", 90),
        _row("e7", "Very old.", 60 * 10),
    ])
    material = asyncio.run(_loop(conn)._read_study_material(NOW))
    assert conn.caption_args[0] == ["walkway"]
    assert [c.event_id for c in material.street_captions] == ["e1", "e4", "e5"]
    assert material.has_material and not material.is_unavailable


def test_failing_read_costs_only_the_street_section() -> None:
    material = asyncio.run(_loop(_Conn(RuntimeError("boom")))._read_study_material(NOW))
    assert material.street_captions == [] and material.has_material


def test_empty_stream_list_never_queries() -> None:
    conn = _Conn([_row("e1", "A van.", 1)])
    material = asyncio.run(_loop(conn, streams=())._read_study_material(NOW))
    assert material.street_captions == [] and conn.caption_args is None


def test_prompt_section_is_labelled_as_the_street_and_absent_when_empty(monkeypatch) -> None:
    monkeypatch.delenv("ORION_CAMERA_DESCRIPTIONS", raising=False)
    empty = StudyMaterial(generated_at=NOW)
    assert not any("STREET CAMERA" in line for line in _material_section(empty))
    full = StudyMaterial(generated_at=NOW, local_timezone="America/Denver",
                         street_captions=build_street_caption_cards([_row("e1", "A van is parked.", 5)], now=NOW))
    text = "\n".join(_material_section(full))
    # Names Orion's own outdoor camera and what it looks over (2026-10-15).
    assert (
        "WHAT THE STREET CAMERA SAID LATELY (your outdoor camera (walkway), looking over the community"
        " walkway, the street and our patio; your own eye outside the house, public, not your room"
    ) in text
    assert "At 11:55 on Thu 15 Oct, the walkway camera said: A van is parked." in text


def test_contradicted_camera_caption_card_says_the_car_did_not_move() -> None:
    """The motion cross-check tag (orion-sql-writer app/vision_caption_motion.py) reaches curiosity."""
    row = _row("c1", "A car is driving down the street.", 2, event_type="camera_caption")
    row["tags"] = ["camera_caption", "motion_unverified", "detector:person",
                   "motion_contradicted", "motion_check:vehicle:driving:parked"]
    (card,) = build_street_caption_cards([row], now=NOW)
    assert "caption said driving; the vehicle did not move" in card.narrative
    assert "motion not verified" not in card.narrative
