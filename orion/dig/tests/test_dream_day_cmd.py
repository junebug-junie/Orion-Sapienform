"""`dig dream-day <date>` (day-dream spec Patch 2): the organ roundup and the day's digs on one page, plus the read
receipt the dream.day run checks. What must hold: the receipt says exactly which parts were read; a part that
could not be read is UNREAD in both the page and the receipt (never 0); the command is routed and in the help."""
from __future__ import annotations

import json

from orion.dig import __main__ as dig
from orion.dig.tests.test_digs import CORR, DAY, HELP_PROMPT, _event, _res, _Runner, _use, _write_transcript
from orion.schemas.dream_day import DreamDayReadV1, DreamDayTotalsV1

LOOKUP = ("correlation_id,workflow,traced,chat_id,session_id\n"
          f"{CORR},pr_field_note,t,,\n")
EMPTY_COCKPIT = "ts,correlation_id,id,command,has_result,is_error,result,chat_id,session_id\n"


def test_dream_day_reads_the_day_and_leaves_a_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "dighome"))
    home = tmp_path / "claude"
    _write_transcript(home, "sess-a", [
        _event("user", "2026-11-16T09:00:00Z", HELP_PROMPT),
        _event("assistant", "2026-11-16T09:00:04Z", [_use("t2", "python -P -m orion.dig body 2>&1 | tail -40")]),
        _event("user", "2026-11-16T09:00:09Z", [_res("t2", "body atlas as of now\nmore")]),
    ])
    runner = _Runner({"FROM cockpit_turn_sighting": EMPTY_COCKPIT, "unnest(ARRAY": LOOKUP})
    text = dig.cmd_dream_day(DAY, dig.Out("t"), runner=runner, claude_home=home)
    assert "python -P -m orion.dig body" in text and "1 dig command(s): 0 in chat, 1 in background runs" in text
    assert "1 transcript(s) read" in text
    path = dig.dream_day_receipt_path(DAY)
    assert str(path) in text and path == tmp_path / "dighome" / "self" / "dream_day" / "2026-11-16.json"
    receipt = DreamDayReadV1.model_validate_json(path.read_text())
    assert receipt.day == DAY and receipt.digs_total == 1 and receipt.digs_background == 1
    assert "background runs (claude transcripts)" not in receipt.unread
    # The totals beside it: the citable dream_day_read:<day>, stamped with THIS receipt's read_at.
    totals = DreamDayTotalsV1.model_validate_json(path.with_name("2026-11-16.totals.json").read_text())
    assert totals.day == DAY and totals.read_at == receipt.read_at
    assert totals.text.splitlines()[-1] == "digs that day: 1 dig command(s), 0 in chat, 1 in background runs"
    assert "cite [dream_day_read:2026-11-16]" in text
    # read-only on Orion's records: every query is a SELECT/WITH
    assert all(s.lstrip().upper().startswith(("SELECT", "WITH")) for s in runner.sql)


def test_dream_day_says_unread_in_page_and_receipt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "dighome"))
    runner = _Runner({"FROM cockpit_turn_sighting": EMPTY_COCKPIT})
    text = dig.cmd_dream_day(DAY, dig.Out("t"), runner=runner, claude_home=tmp_path / "nowhere")
    assert "UNREAD background runs (claude transcripts)" in text and "part(s) UNREAD" in text
    receipt = json.loads(dig.dream_day_receipt_path(DAY).read_text())
    assert "background runs (claude transcripts)" in receipt["unread"]
    # A partial digs read is never a citable count: the totals say UNREAD, not "0 in background runs".
    totals = json.loads(dig.dream_day_receipt_path(DAY).with_name("2026-11-16.totals.json").read_text())
    assert "UNREAD background runs (claude transcripts)" in totals["text"]
    assert "digs that day:" not in totals["text"]


def test_main_routes_dream_day(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(dig, "cmd_dream_day", lambda day, out: seen.setdefault("day", day) and "ok")
    assert dig.main(["dream-day", "2026-11-16"]) == 0 and seen["day"] == DAY
    assert dig.main(["dream-day"]) == 2
    assert "dream-day <date>" in dig.HELP
