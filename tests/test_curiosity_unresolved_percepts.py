"""Walkway percepts Orion could not name, offered as curiosity material.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 4.
The properties under test: at most three, newest first, phrased as something
Orion saw; nothing rendered when there are none; and nothing about how the
topic is chosen changes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.curiosity.kickoff_prompt import build_kickoff_prompt
from orion.curiosity.study_material import (
    UNRESOLVED_RECENT_SQL,
    StudyMaterial,
    build_unresolved_cards,
)

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _row(uid: str, minutes_ago: int, **over):
    row = {
        "unresolved_id": uid,
        "stream_id": "walkway",
        "camera_id": "walkway",
        "observed_at": NOW - timedelta(minutes=minutes_ago),
        "reason": "no_label",
        "description": "a low shape moving along the fence line",
        "what_was_tried": '["yolo", "council"]',
        "evidence_refs": "[]",
        "image_ref": None,
    }
    row.update(over)
    return row


def test_cards_are_capped_at_three_and_newest_first() -> None:
    rows = [_row(f"u{i}", minutes_ago=i * 10) for i in (4, 1, 3, 2, 5)]
    cards = build_unresolved_cards(rows)
    assert [c.unresolved_id for c in cards] == ["u1", "u2", "u3"]


def test_rows_without_id_or_time_are_skipped_not_rendered_empty() -> None:
    cards = build_unresolved_cards([_row("", 1), _row("u2", 2, observed_at=None), _row("u3", 3)])
    assert [c.unresolved_id for c in cards] == ["u3"]


def test_preview_is_first_person_local_time_and_cites_its_id() -> None:
    from zoneinfo import ZoneInfo

    # 09:12 UTC is 03:12 MDT.
    card = build_unresolved_cards([_row("u1", 0, observed_at=datetime(2026, 9, 24, 9, 12, tzinfo=timezone.utc))])[0]
    text = card.preview(ZoneInfo("America/Denver"))
    assert text.startswith("At 03:12 on Thu 24 Sep on the walkway I saw something I could not name")
    assert "a low shape moving along the fence line" in text
    assert "tried: yolo, council" in text
    assert "unresolved_id: u1" in text


def test_sql_is_bounded_by_time_and_count_and_ordered_newest_first() -> None:
    assert "FROM vision_unresolved" in UNRESOLVED_RECENT_SQL
    assert "ORDER BY observed_at DESC" in UNRESOLVED_RECENT_SQL
    assert "LIMIT $2" in UNRESOLVED_RECENT_SQL
    # Newest-first over a window, never random or ranked by some salience.
    assert "random()" not in UNRESOLVED_RECENT_SQL


def test_prompt_has_no_walkway_section_when_there_is_nothing() -> None:
    prompt = build_kickoff_prompt(StudyMaterial(generated_at=NOW), graph_enabled=False)
    assert "COULD NOT NAME" not in prompt
    assert "could not name" not in prompt


def test_prompt_offers_them_without_choosing() -> None:
    material = StudyMaterial(generated_at=NOW)
    material.unresolved = build_unresolved_cards([_row("u1", 5), _row("u2", 30)])
    prompt = build_kickoff_prompt(material, graph_enabled=False)
    assert "THINGS YOUR CAMERAS SAW AND COULD NOT NAME" in prompt
    assert "unresolved_id: u1" in prompt and "unresolved_id: u2" in prompt
    assert ":Prior" in prompt and "Juniper" in prompt
    assert "not because you should pick one" in prompt
    assert prompt.index("unresolved_id: u1") < prompt.index("unresolved_id: u2")


def test_unresolved_does_not_change_the_run_gate() -> None:
    """`has_material` gates whether a run happens at all; the street is extra
    material, not a new reason to wake the loop."""
    material = StudyMaterial(generated_at=NOW)
    material.unresolved = build_unresolved_cards([_row("u1", 5)])
    assert material.has_material is False
    assert "u1" in material.shown_ids()


def test_image_ref_never_reaches_the_prompt() -> None:
    card = build_unresolved_cards([_row("u1", 0, image_ref="/crops/patio/1.jpg")])[0]
    assert "patio" not in card.preview()
