"""Game company patch 4: a finished sitting's event want reaches Prime's EVENTS WORTH EFFORT menu with its own label."""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parents[1])]

from app.event_wants_menu import (  # noqa: E402
    LATEST_EVENT_DECISIONS_SQL, SITTING_LABELS_SQL, SITTING_TABLE_SQL, read_menu, sitting_label,
)
from tests.test_event_wants_menu import FAMS, NOW, _Pool, _row  # noqa: E402

REF = "retro_sitting:S1"
ROW = {"sitting_id": "S1", "game_title": "The Legend of Zelda: A Link to the Past", "game_id": "a-link-to-the-past",
       "player_stated": "juniper", "started_at": datetime(2026, 11, 4, 9, 0, tzinfo=timezone.utc),
       "last_input_at": datetime(2026, 11, 4, 11, 22, tzinfo=timezone.utc)}


def test_the_label_is_counts_only_and_names_juniper_only_when_the_page_was_told():
    assert sitting_label(ROW) == ("a game sitting: Juniper played The Legend of Zelda: A Link to the Past for 142 min, "
                                  "ended 2026-11-04 11:22Z")
    assert sitting_label({**ROW, "player_stated": None}).startswith("a game sitting: Someone played")
    assert sitting_label({**ROW, "player_stated": "guest"}).startswith("a game sitting: Someone played")
    assert sitting_label({"sitting_id": "x"}) == "a game sitting: Someone played a game"


def test_a_funded_sitting_want_is_on_the_menu_with_its_label_inside_the_family_window():
    rows = [_row("s", REF, family="hub.retro_sitting", decided=NOW - timedelta(hours=1))]
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows, SITTING_TABLE_SQL: [{"present": True}], SITTING_LABELS_SQL: [ROW]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    [w] = menu.wants
    assert w.family == "hub.retro_sitting" and w.label.startswith("a game sitting: Juniper played")
    assert w.expires_at == rows[0]["decided_at"] + timedelta(hours=6)
    # Past the family's 6 h window it is gone.
    old = [_row("s", REF, family="hub.retro_sitting", decided=NOW - timedelta(hours=7))]
    assert asyncio.run(read_menu(_Pool({LATEST_EVENT_DECISIONS_SQL: old}), FAMS, now=NOW, hours=24, limit=8)).wants == []


def test_before_the_table_exists_the_want_stays_unlabelled_and_no_label_read_runs():
    rows = [_row("s", REF, family="hub.retro_sitting")]
    pool = _Pool({LATEST_EVENT_DECISIONS_SQL: rows, SITTING_TABLE_SQL: [{"present": False}]})
    menu = asyncio.run(read_menu(pool, FAMS, now=NOW, hours=24, limit=8))
    assert [w.label for w in menu.wants] == [""]
    assert not any(sql == SITTING_LABELS_SQL for sql, _ in pool.seen)
