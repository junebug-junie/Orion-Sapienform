"""A journal entry always has a title (#3386).

The compose prompt allows ``"title": null`` and the model usually takes it: 13 of 15 walkway
entries from 2026-11-10 to 11-14 were stored with no title, because build_write_payload wrote
the title as the model gave it.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.journaler.schemas import JournalEntryDraftV1, JournalTriggerV1
from orion.journaler.worker import build_write_payload, code_built_title

WRITTEN = datetime(2026, 11, 15, 5, 31, tzinfo=timezone.utc)


def _walkway(kind: str, for_date: str) -> JournalTriggerV1:
    return JournalTriggerV1(trigger_kind=kind, source_kind="scheduler",
                            source_ref=f"{kind}:walkway:{for_date}", summary="s")


def _write(trigger: JournalTriggerV1, title):
    draft = JournalEntryDraftV1(mode="digest", title=title, body="The street was quiet.")
    return build_write_payload(draft, trigger=trigger, correlation_id="c", author="orion", created_at=WRITTEN)


@pytest.mark.parametrize("title", [None, "", "   "])
def test_no_model_title_gets_one_built_from_kind_and_date(title) -> None:
    assert _write(_walkway("walkway_day", "2026-11-14"), title).title == "Walkway day · 2026-11-14"


def test_the_date_is_the_day_the_entry_is_about_not_the_day_it_was_written() -> None:
    # Tonight's forecast is about tomorrow; a night that finished after midnight is about yesterday.
    assert _write(_walkway("walkway_forecast", "2026-11-15"), None).title == "Walkway forecast · 2026-11-15"
    assert _write(_walkway("walkway_grade", "2026-11-13"), None).title == "Walkway grade · 2026-11-13"


def test_a_model_title_is_kept_as_given() -> None:
    assert _write(_walkway("walkway_day", "2026-11-14"), "  A slow Saturday ").title == "A slow Saturday"


def test_a_trigger_with_no_date_in_its_ref_uses_the_day_it_was_written() -> None:
    trigger = JournalTriggerV1(trigger_kind="manual", source_kind="manual", source_ref="note-7", summary="s")
    assert code_built_title(trigger, created_at=WRITTEN) == "Manual · 2026-11-15"
    trigger = JournalTriggerV1(trigger_kind="collapse_response", source_kind="collapse_mirror", summary="s")
    assert _write(trigger, None).title == "Collapse response · 2026-11-15"
