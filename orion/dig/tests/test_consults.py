"""`dig consult [<id>]`: open a consult by the id on its Hub card (2026-11-25)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

import orion.dig.__main__ as dig
from orion.dig import consults

CONSULT = {"consult_id": "f1842fa5ab10", "source_kind": "curiosity_finding", "source_ref": "run_abc",
           "status": "brief_sent", "ending": "agreed_proposal", "turns": "4",
           "question": "Should the  heartbeat\nseats be retired?", "answer": "",
           "brief_text": "@Juniper -- Orion and Claude have a proposal for you.", "decision": "no_because",
           "decision_reason": "not yet", "decided_at": "2026-11-24 10:00:00+00",
           "brief_sent_at": "2026-11-24 09:00:00+00", "created_at": "2026-11-24 08:00:00+00"}
TURNS = [{"turn_no": "1", "speaker": "orion", "ending": "", "text": "x" * 2000, "created_at": "t1"},
         {"turn_no": "2", "speaker": "claude", "ending": "agreed_proposal", "text": "Agreed.", "created_at": "t2"}]


def _csv(rows, fields):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


class Runner:
    def __init__(self, consult=(CONSULT,), turns=TURNS, fail=None):
        self.consult, self.turns, self.fail, self.seen = list(consult), turns, fail, []

    def run(self, sql, flags):
        self.seen.append(sql)
        if self.fail:
            return self.fail, 1
        if "orion_consult_turn" in sql:
            return _csv(self.turns, list(TURNS[0])), 0
        if "LIMIT" in sql:
            fields = ["consult_id", "status", "ending", "decision", "created_at", "question"]
            return _csv([{k: c[k] for k in fields} for c in self.consult], fields), 0
        return _csv(self.consult, list(CONSULT)), 0


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))


def test_one_consult_prints_question_decision_turns_and_writes_full_file():
    runner = Runner()
    text = dig.cmd_consult("F1842FA5AB10", dig.Out("t"), runner=runner)
    assert "consult f1842fa5ab10: brief_sent, ending agreed_proposal" in text
    assert "question: Should the heartbeat seats be retired?" in text
    assert "Juniper said no, because: not yet" in text
    assert "turn 2 claude (agreed_proposal): Agreed." in text
    assert "x" * 2000 not in text  # trimmed in the summary ...
    path = text.rsplit(": ", 1)[1].strip()
    assert json.loads(Path(path).read_text())["turns"][0]["text"] == "x" * 2000  # ... whole in the file
    assert all("f1842fa5ab10" in sql for sql in runner.seen)


def test_bad_id_is_refused_before_any_sql():
    runner = Runner()
    with pytest.raises(dig.DigError, match="12 hex"):
        dig.cmd_consult("x'; drop table orion_consult; --", dig.Out("t"), runner=runner)
    assert runner.seen == []


def test_unknown_id_says_not_found():
    with pytest.raises(dig.DigError, match="not found"):
        dig.cmd_consult("000000000000", dig.Out("t"), runner=Runner(consult=()))


def test_missing_grant_names_the_grant_not_an_empty_answer():
    runner = Runner(fail="ERROR:  permission denied for table orion_consult")
    with pytest.raises(dig.DigError, match="GRANT SELECT ON orion_consult"):
        dig.cmd_consult("f1842fa5ab10", dig.Out("t"), runner=runner)


def test_no_id_lists_the_newest():
    text = dig.cmd_consult(None, dig.Out("t"), runner=Runner())
    assert "f1842fa5ab10" in text and "no_because" in text


def test_undecided_and_unsent_lines():
    assert consults.decision_line({"brief_sent_at": "t"}) == "Juniper has not answered yet."
    assert consults.decision_line({}) == "No brief was sent to Juniper."


def test_help_lists_consult():
    assert "consult [<id>]" in dig.HELP


def test_main_routes_consult(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(dig, "cmd_consult", lambda cid, out: seen.setdefault("id", cid) or "ok")
    assert dig.main(["consult", "f1842fa5ab10"]) == 0
    assert seen["id"] == "f1842fa5ab10"
