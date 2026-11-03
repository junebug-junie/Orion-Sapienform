"""Street material shown in a real kickoff prompt is recorded with the run (2026-11-03).

Before this, `curiosity_offer_decisions.material_ids` held only crystallization
and relation ids, so "the walkway captions / day card reached a real
investigation" could not be checked. Pinned here:

- a real tick whose prompt carries street captions and the day card records
  `street_caption:<event_id>` and `street_day:<stream>:<day>` for exactly those;
- captions fetched but dropped (dedupe, cap, too old) are NOT recorded;
- with no street material the ids are exactly the old crystallization+relation list.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt, kickoff_material_ids
from orion.vision.walkway_rollup import LIVING_RANGE_RULES_VERSION
from orion.curiosity.study_material import (
    STREET_CAPTIONS_RECENT_SQL,
    STREET_DAY_ROLLUP_SQL,
    CrystallizationCard,
    RelationCard,
    StreetCaptionCard,
    StreetDayCard,
    StudyMaterial,
)
from test_curiosity_spend_log import _SpendConn, _graph_loop, _moves_during_turn, _reader, _state
from test_curiosity_investigation import _FakeBus

DAY_LINE = ("Walkway Mon 02 Nov 2026: people: at least 3, at most 35 (the camera splits "
            "one person into several); 31 vehicles (9 distinct) passing.")


def _caption_row(eid, text, minutes_ago, now):
    return {"event_id": eid, "stream_id": "walkway", "event_type": "visual_observation",
            "narrative": text, "created_at": now - timedelta(minutes=minutes_ago)}


class _StreetConn(_SpendConn):
    def __init__(self, *, captions, day_row, **kw):
        super().__init__(**kw)
        self.captions = captions
        self.day_row = day_row

    async def fetch(self, sql, *args):
        if sql == STREET_CAPTIONS_RECENT_SQL:
            return self.captions
        return await super().fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        if sql == STREET_DAY_ROLLUP_SQL:
            return self.day_row
        return await super().fetchrow(sql, *args)


def _tick(conn):
    bus = _FakeBus()
    reader = _reader([_state("p1", 0.5, 0)])
    loop = _graph_loop(bus, reader=reader, conn=conn)
    _moves_during_turn(loop, reader, lambda run_id: [_state("p1", 0.5, 0)])
    assert asyncio.run(loop.tick()) is None
    assert len(conn.decisions) == 1
    return loop, next(iter(conn.decisions.values()))


def test_a_real_tick_records_the_street_material_its_prompt_carried() -> None:
    now = datetime.now(timezone.utc)
    conn = _StreetConn(
        captions=[
            _caption_row("ev-van", "A van is parked by the curb.", 5, now),
            _caption_row("ev-van-dup", "a van is parked by the curb", 6, now),  # deduped away
            _caption_row("ev-bike", "A cyclist rides past.", 30, now),
            _caption_row("ev-gate", "Two people talk at the gate.", 60, now),
            _caption_row("ev-car", "A car turns the corner.", 90, now),  # past the cap of 3
            _caption_row("ev-old", "Very old caption.", 60 * 30, now),  # too old
        ],
        day_row={"stream_id": "walkway", "label": "2026-11-02", "complete": True, "line": DAY_LINE,
                 "rollup": {"rules_version": LIVING_RANGE_RULES_VERSION}, "computed_at": now},
    )
    loop, decision = _tick(conn)
    ids = decision["material_ids"]

    street_captions = sorted(i for i in ids if i.startswith("street_caption:"))
    assert street_captions == ["street_caption:ev-bike", "street_caption:ev-gate", "street_caption:ev-van"]
    assert f"street_day:walkway:2026-11-02@{now.isoformat()}" in ids
    # Every recorded street id is backed by text in the prompt the turn actually got.
    # (Captions with a count get "(count from one frame)" appended, so match the stem.)
    for text in ("A van is parked by the curb", "A cyclist rides past", "Two people talk at the gate", DAY_LINE):
        assert text in loop.seen_prompt
    for dropped in ("A car turns the corner.", "Very old caption."):
        assert dropped not in loop.seen_prompt
    # The existing ids are still there, first.
    assert ids[0].startswith("crystallization:")


def test_without_street_material_the_ids_are_unchanged() -> None:
    conn = _StreetConn(captions=[], day_row=None)
    _, decision = _tick(conn)
    # Exactly the pre-2026-11-03 list: the fixture's crystallizations, then its relation.
    assert decision["material_ids"] == [
        "crystallization:c0", "crystallization:c1", "crystallization:c2", "crystallization:c3", "relation:d1",
    ]


def _material(**street) -> StudyMaterial:
    material = StudyMaterial(generated_at=datetime(2026, 11, 3, tzinfo=timezone.utc), **street)
    material.crystallizations = [CrystallizationCard("c1", "semantic", "s", "summary", 0.5, None)]
    material.relations = [RelationCard("r1", "supports", 0.8, "a", "b", None)]
    return material


def test_ids_follow_the_rendered_sections() -> None:
    """Id-less captions are rendered but cannot be recorded (pinned on purpose)."""
    at = datetime(2026, 11, 3, 1, 0, tzinfo=timezone.utc)
    material = _material(
        street_captions=[StreetCaptionCard("ev1", at, "walkway", "A dog walks by."),
                         StreetCaptionCard("", at, "walkway", "No id on this one.")],
        street_day=StreetDayCard(stream_id="walkway", day="2026-11-02", line=DAY_LINE, complete=False),
    )
    assert kickoff_material_ids(material) == [
        "crystallization:c1", "relation:r1", "street_caption:ev1", "street_day:walkway:2026-11-02",
    ]
    # Every recorded caption id's own text is in the rendered prompt.
    for card in material.street_captions:
        if f"street_caption:{card.event_id}" in kickoff_material_ids(material):
            assert card.narrative in build_kickoff_prompt(material, graph_enabled=False)
    prompt = build_kickoff_prompt(material, graph_enabled=False)
    assert "A dog walks by." in prompt and DAY_LINE in prompt

    plain = _material()
    assert kickoff_material_ids(plain) == ["crystallization:c1", "relation:r1"]
