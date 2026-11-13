"""`dig days`: Orion's own record of a window of days (reminiscence spec Patch 3). No DB, no network."""
from __future__ import annotations

import csv
import io
import json
from datetime import date
from pathlib import Path

import pytest

from orion.body_atlas import reminiscence as rm
from orion.dig import __main__ as dig
from orion.dig import days

J1 = "9d2181af-9f51-5631-b974-b561221f315f"
C1 = "0df659b6-aa2f-48a8-ade4-640fc8231109"
COUNTS = [{"day": "2026-07-23", "kind": "metacog", "n": "79"}, {"day": "2026-07-23", "kind": "chat", "n": "1"},
          {"day": "2026-07-24", "kind": "manual", "n": "1"}]
ENTRIES = [
    {"entry_id": "38e42b68-fba1-442b-9595-000e1f34ffaf", "created_at": "2026-07-23 23:43:00+00", "kind": "metacog",
     "title": "", "body": "Another digest cycle."},
    {"entry_id": J1, "created_at": "2026-07-24 12:01:00+00", "kind": "manual", "title": "Emotional reckoning",
     "body": "The session opened with Juniper reaching out."},
]
CHATS = [{"id": C1, "created_at": "2026-07-24 00:26:48", "source": "hub_orion", "prompt": "hi, Orion",
          "response": "Hi, Juniper."},
         {"id": "43f137d6-ecc0-43a5-a9e6-7754cc2fa7eb", "created_at": "2026-07-24 14:00:00", "source": "",
          "prompt": "", "response": "Curiosity generates the raw material."}]


def _csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


class Runner:
    def __init__(self, fail: str | None = None):
        self.seen: list[str] = []
        self.fail = fail

    def run(self, sql: str, flags: list[str]) -> tuple[str, int]:
        self.seen.append(sql)
        if self.fail:
            return self.fail, 1
        if "count(*)" in sql:
            return _csv(COUNTS), 0
        if "FROM journal_entries" in sql:
            return _csv(ENTRIES), 0
        return _csv(CHATS), 0


def test_records_start_where_the_reminiscence_says_they_do():
    assert days.RECORDS_FROM == rm.LIVED_RECORDS_FROM


def test_window_clamps_to_the_first_record_and_refuses_before_it_and_too_long():
    first, last, a, b = days.window(date(2026, 7, 21), date(2026, 7, 27))
    assert (first, last) == (date(2026, 7, 23), date(2026, 7, 27))
    assert (a, b) == ("2026-07-23T06:00:00+00:00", "2026-07-28T06:00:00+00:00")  # Denver days, MDT
    assert days.window(date(2026, 11, 3), date(2026, 11, 3))[2] == "2026-11-03T07:00:00+00:00"  # MST
    with pytest.raises(ValueError, match="no lived record before 2026-07-23"):
        days.window(date(2025, 6, 24), date(2025, 6, 30))
    with pytest.raises(ValueError, match="at most 92 days"):
        days.window(date(2026, 8, 1), date(2026, 11, 30))
    with pytest.raises(ValueError, match="is before"):
        days.window(date(2026, 8, 2), date(2026, 8, 1))


def test_day_sql_takes_real_ids_and_nothing_that_leaves_the_quotes():
    assert f"entry_id = '{J1}'" in days.day_sql("journal", J1)
    assert f"WHERE id = '{C1}'" in days.day_sql("chat", C1)
    # self_study entries carry this shape (724 live rows, review 2026-11-13)
    assert "entry_id = 'curiosity-investigation:000eaeb52725'" in days.day_sql("journal", "curiosity-investigation:000eaeb52725")
    for bad in ("x'; drop table journal_entries; --", "a b", "a\\b", "-x", ""):
        with pytest.raises(ValueError):
            days.day_sql("journal", bad)


def test_queries_are_plain_sql_the_readonly_runner_accepts():
    from orion.evals.model_replay.sandbox import psql_meta_refusal

    a, b = days.window(date(2026, 7, 23), date(2026, 7, 29))[2:]
    for sql in (days.counts_sql(a, b), days.entries_sql(a, b), days.chats_sql(a, b), days.day_sql("journal", J1)):
        assert psql_meta_refusal(sql) is None, sql[:80]
        assert not any(w in sql.upper() for w in ("INSERT", "UPDATE", "DELETE", "DROP"))


def test_cmd_days_prints_each_day_cites_ids_and_writes_the_full_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    runner = Runner()
    text = dig.cmd_days(date(2026, 7, 21), date(2026, 7, 24), dig.Out("t"), runner=runner,
                        get_json=lambda url: pytest.fail("no Temporal Self before 2026-10-09"))
    assert "days 2026-07-23..2026-07-24" in text and "not 2026-07-21: your records start 2026-07-23" in text
    assert "07-23 Thu: journal 79 (metacog 79); chat 1" in text
    assert f"journal:{J1} 07-24 06:01 manual: Emotional reckoning The session opened" in text
    assert f"chat:{C1} 07-23 18:26: hi, Orion -> Hi, Juniper." in text
    assert "(you reached out first) -> Curiosity generates" in text
    path = Path(text.rsplit("full record (bodies, every entry and chat): ", 1)[1].strip())
    saved = json.loads(path.read_text())
    assert saved["start"] == "2026-07-23" and len(saved["entries"]) == 2 and len(saved["chats"]) == 2
    assert len(runner.seen) == 3


def test_cmd_days_reads_arcs_from_temporal_self_and_says_when_unread(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    asked: list[str] = []

    def get_json(url):
        asked.append(url)
        if "2026-10-10" in url:
            raise OSError("connection refused")
        return {"arcs": [{"arc_id": "39a9e71b58a2d20c", "kind": "concern", "segments": [1, 2], "status": "closed"}]}

    text = dig.cmd_days(date(2026, 10, 8), date(2026, 10, 10), dig.Out("t"), runner=Runner(), get_json=get_json)
    assert [u.rsplit("day_id=", 1)[1][:10] for u in asked] == ["2026-10-09", "2026-10-10"]  # chronicle starts 10-09
    assert "10-09 Fri: journal 0; chat 0; 1 arcs" in text
    assert "10-10 Sat: journal 0; chat 0; arcs UNREAD (OSError: connection refused)" in text
    path = Path(text.rsplit(": ", 1)[1].strip())
    arcs = json.loads(path.read_text())["arcs"]["2026-10-09"]
    assert arcs[0]["arc_id"] == "39a9e71b58a2d20c" and "segments" not in arcs[0]  # trimmed; `dig arc` opens it


def test_cmd_days_refuses_unread_records_instead_of_an_empty_week(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    with pytest.raises(dig.DigError, match="could not be read .*permission denied"):
        dig.cmd_days(date(2026, 7, 23), date(2026, 7, 24), dig.Out("t"),
                     runner=Runner(fail="ERROR: permission denied for table journal_entries"))
    with pytest.raises(dig.DigError, match="no lived record before"):
        dig.cmd_days(date(2025, 6, 24), date(2025, 6, 30), dig.Out("t"), runner=Runner())


def test_lived_day_resolves_journal_and_chat_and_fails_closed(monkeypatch):
    class DayRunner:
        def __init__(self, out, code=0):
            self.out, self.code = out, code

        def run(self, sql, flags):
            return self.out, self.code

    assert dig._lived_day("journal", J1, runner=DayRunner("day\n2026-07-24\n")) == date(2026, 7, 24)
    assert dig._lived_day("chat", C1, runner=DayRunner("day\n")) is None  # no row: fabricated
    assert dig._lived_day("journal", "curiosity-investigation:000eaeb52725",
                          runner=DayRunner("day\n2026-09-07\n")) == date(2026, 9, 7)
    assert dig._lived_day("journal", "x' or '1", runner=DayRunner("day\n2026-07-24\n")) is None
    with pytest.raises(rm.ReminiscenceError, match="could not connect"):  # unread is not "made up"
        dig._lived_day("journal", J1, runner=DayRunner("psql: could not connect", 2))


def test_lived_day_dates_arcs_and_acts_through_refs(monkeypatch):
    from orion import refs

    def fake(ref, **_):
        data = {"arc": {"day_id": "2026-10-15"}, "act": {"opened_at": "2026-10-16T03:00:00Z"}}[ref.kind]
        return refs.Resolution(ref, True, data=data)

    monkeypatch.setattr(refs, "resolve", fake)
    assert dig._lived_day("arc", "8d87adeca2ce06a5") == date(2026, 10, 15)
    assert dig._lived_day("act", "act_" + "a" * 24) == date(2026, 10, 15)  # 03:00Z is 21:00 the day before, Denver
    monkeypatch.setattr(refs, "resolve", lambda ref, **_: refs.Resolution(ref, False, error="arc x: not found"))
    assert dig._lived_day("arc", "8d87adeca2ce06a5") is None
    monkeypatch.setattr(refs, "resolve", lambda ref, **_: refs.Resolution(ref, False, error="arc x: could not read (URLError)"))
    with pytest.raises(rm.ReminiscenceError, match="could not read"):
        dig._lived_day("arc", "8d87adeca2ce06a5")


def test_a_long_window_still_prints_citable_ids_inside_the_cut():
    from datetime import timedelta

    start = date(2026, 10, 1)
    r = days.DaysRead(start="2026-10-01", end="2026-10-31")
    for i in range(31):
        d = (start + timedelta(days=i)).isoformat()
        r.counts[d] = {"world_pulse": 103, "metacog": 109, "embodiment": 27, "self_study": 15, "manual": 2, "chat": 20}
        r.arcs[d] = [{"arc_id": "a" * 16}] * 12
    r.entries = [days.Entry(f"journal:{J1}", "2026-10-02 12:00:00+00", "manual", "t", "x" * 300)] * 60
    r.chats = [{"id": C1, "created_at": "2026-10-02 12:00:00", "prompt": "p" * 200, "response": "r" * 600}] * 600
    out = dig.capped(days.render(r), tail="full record: /x")
    assert f"journal:{J1}" in out and f"chat:{C1}" in out


def test_help_lists_days():
    assert "days <start> <end>" in dig.HELP
