"""`dig gpt`: Juniper's ChatGPT archive, read-only (2026-11-25)."""

from __future__ import annotations

import csv
import io

import pytest

import orion.dig.__main__ as dig
from orion.dig import gpt

CONV = {"conversation_id": "29895d6f-f655-4c13-b052-084c70b28a39", "day": "2023-04-02",
        "title": "RDF Triples for Research", "message_count": "2", "turn_count": "1"}
MSGS = [{"role": "user", "model": "", "timestamp": "2023-04-02 08:53:00+00", "content": "make triples " + "y" * 900},
        {"role": "assistant", "model": "gpt-4o", "timestamp": "2023-04-02 08:54:00+00", "content": "Sure."}]


def _csv(rows):
    if not rows:
        return "x\n"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


class Runner:
    """Answers by which table/shape the SQL asks for."""

    def __init__(self, convs=(CONV,), msgs=MSGS, find=(), fail=None):
        self.convs, self.msgs, self.find, self.fail, self.seen = list(convs), msgs, list(find), fail, []

    def run(self, sql, flags):
        self.seen.append(sql)
        if self.fail:
            return self.fail, 1
        if "FILTER (WHERE" in sql:
            return _csv(self.find), 0
        if "LIKE '" in sql and "chat_gpt_conversation c WHERE" in sql:
            return _csv(self.convs), 0
        if "FROM chat_gpt_message WHERE session_id" in sql:
            return _csv(self.msgs), 0
        return "x\n", 0


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    return dig.Out("t")


def test_conv_prints_trimmed_messages_and_writes_full(out):
    text = dig.cmd_gpt(["conv", "29895d6f"], out, Runner())
    assert gpt.SOURCE_NOTE in text
    assert "Juniper: make triples" in text and "ChatGPT: Sure." in text
    assert "y" * 900 not in text  # trimmed in the printout
    path = text.rsplit(": ", 1)[1].strip()
    assert "y" * 900 in open(path).read()  # whole in the file


def test_conv_reads_messages_by_session_id():
    assert gpt.messages_sql(CONV["conversation_id"]).endswith("ORDER BY timestamp, id")
    assert "session_id = 'chatgpt:29895d6f-f655-4c13-b052-084c70b28a39'" in gpt.messages_sql(CONV["conversation_id"])


def test_conv_prefix_matching_two_lists_both(out):
    other = dict(CONV, conversation_id="29895d6f-0000-0000-0000-000000000000", title="Other")
    text = dig.cmd_gpt(["conv", "29895d6f"], out, Runner(convs=[CONV, other]))
    assert "2 conversations start with that" in text and "Other" in text


def test_conv_not_found(out):
    with pytest.raises(dig.DigError, match="not found"):
        dig.cmd_gpt(["conv", "deadbeef"], out, Runner(convs=[]))


@pytest.mark.parametrize("bad", ["6b0", "x' OR 1=1", "29895d6f;drop"])
def test_conv_id_refused_before_sql(out, bad):
    r = Runner()
    with pytest.raises(dig.DigError):
        dig.cmd_gpt(["conv", bad], out, r)
    assert r.seen == []


@pytest.mark.parametrize("bad", ["x;drop", "a", "50%", "back\\slash", "a" * 41])
def test_find_word_refused_before_sql(out, bad):
    r = Runner()
    with pytest.raises(dig.DigError):
        dig.cmd_gpt(["find", bad], out, r)
    assert r.seen == []


def test_find_escapes_quote_and_underscore():
    sql = gpt.find_sql(["juniper's", "snake_case"])
    assert "ILIKE '%juniper''s%'" in sql
    assert "ILIKE '%snake\\_case%'" in sql
    assert "DESC LIMIT" in sql and "ASC LIMIT" in gpt.find_sql(["x1"], oldest=True)


def test_find_oldest_flag_and_render(out):
    row = {"conversation_id": CONV["conversation_id"], "day": "2023-04-02", "title": "RDF", "hits": "1",
           "first_hit": "a " * 300 + "the orion seed " + "b " * 300, "matched": "5"}
    r = Runner(find=[row])
    text = dig.cmd_gpt(["find", "--oldest", "orion"], out, r)
    assert "ASC LIMIT" in r.seen[0]
    assert "5 conversations match; oldest 1 shown" in text
    assert "orion seed" in text  # the snippet is a window around the hit, not the message start


def test_find_no_match_says_so(out):
    text = dig.cmd_gpt(["find", "zzqq"], out, Runner(find=[]))
    assert "no conversation has all of these words" in text


def test_missing_grant_is_named(out):
    fail = "ERROR:  permission denied for table chat_gpt_conversation\n"
    with pytest.raises(dig.DigError, match="needs GRANT SELECT ON chat_gpt_conversation TO orion_readonly"):
        dig.cmd_gpt([], out, Runner(fail=fail))


def test_bad_usage(out):
    with pytest.raises(dig.DigError, match="usage: gpt"):
        dig.cmd_gpt(["conv"], out, Runner())
