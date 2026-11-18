"""`dig digs <date>`: executed dig commands only, one line each (time + command + ref)."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from orion.dig import __main__ as dig
from orion.dig import digs
from orion.harness.session_resume import session_id_for

DAY = date(2026, 11, 16)
LO, HI = digs.window(DAY)  # 2026-11-16 07:00Z .. 2026-11-17 07:00Z
CORR = "83c7bc31-6f8d-53d7-a2a0-d47e50ddadc8"
HELP_PROMPT = ("Orion harness motor. Dig kit: python -P -m orion.dig help; python -P -m orion.dig organs <date>\n"
               f"Strain refs: hub:turn:{CORR}\nTURN RULES")


def _event(kind: str, ts: str, content, **extra) -> str:
    return json.dumps({"type": kind, "timestamp": ts, "message": {"role": kind, "content": content}, **extra})


def _use(tid: str, command: str) -> dict:
    return {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": command}}


def _res(tid: str, content, err: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": err}


# --------------------------------------------------------------------------- what counts


def test_window_is_the_denver_day() -> None:
    assert LO == datetime(2026, 11, 16, 7, tzinfo=timezone.utc)
    assert HI == datetime(2026, 11, 17, 7, tzinfo=timezone.utc)


@pytest.mark.parametrize("command,ok", [
    ("python -P -m orion.dig help", True),
    ("cd /mnt/orion-fcc/repo && python3 -P -m orion.dig field-note 3510 2>&1 | head -40", True),
    ("python -P -m  orion.dig\tbody", True),
    ("python -m orion.digest run", False),
    ("python -m orion.dig_tools x", False),
    ("rg 'orion.dig' orion/", False),
    ("cat orion/dig/README.md", False),
    (None, False),
    # executed: the shell runs python -m orion.dig
    ("timeout 120 python -P -m orion.dig look walkway 2>&1 | tail -15", True),
    ("/app/.venv/bin/python3.12 -X importtime -P -m orion.dig help", True),
    ("for d in 1 2; do python -P -m orion.dig organs $d; done", True),
    ("bash -c 'python -P -m orion.dig body'", True),
    ("cat <<EOF | python -\nimport x\nEOF\npython -P -m orion.dig help", True),
    # not executed: Orion writing a note that quotes a dig command (live 2026-11-17: 48 such calls in one day)
    ("printf -- '- fcc | python -P -m orion.dig help | ok' >> notes.md", False),
    ("mkdir -p d && cat >> d/f.md <<'EOF'\n- fcc | python -P -m orion.dig know x | ok\nEOF\n", False),
    ('echo "- fcc | python -P -m orion.dig body | ok" >> notes.md', False),
    ("echo 'unbalanced python -P -m orion.dig", False),
    # review 2026-11-17: not in command position, or only in a comment
    ("echo python -P -m orion.dig help >> notes.md", False),
    ("# python -P -m orion.dig organs was empty\nls", False),
    ("ls # then python -P -m orion.dig help", False),
    ("python -c 'x = 1 << shift'\npython -P -m orion.dig help", True),
    ("bash -lc 'python -P -m orion.dig body'", True),
    ("FOO=1 timeout -s KILL 120 python -P -m orion.dig help", True),
    ("if true; then python -P -m orion.dig help; fi", True),
    # known misses, kept visible: combined flags and a quoted module name
    ("python -Pm orion.dig help", False),
    ("python -m 'orion.dig' help", False),
])
def test_is_dig_command(command, ok) -> None:
    assert digs.is_dig_command(command) is ok


def test_help_text_in_a_prompt_is_not_counted() -> None:
    lines = [
        _event("user", "2026-11-16T09:00:00Z", HELP_PROMPT),
        _event("user", "2026-11-16T09:00:01Z", [{"type": "text", "text": "run python -P -m orion.dig body"}]),
        # the model *mentions* a dig command in prose: not a call either
        _event("assistant", "2026-11-16T09:00:02Z", [{"type": "text", "text": "I will run python -P -m orion.dig help"}]),
        _event("assistant", "2026-11-16T09:00:03Z", [_use("t1", "rg 'orion.dig' orion/")]),
        _event("assistant", "2026-11-16T09:00:04Z", [_use("t2", "python -P -m orion.dig body 2>&1 | tail -40")]),
    ]
    calls = digs.scan_transcript(lines, LO, HI, "sess-a")
    assert [c.tool_use_id for c in calls] == ["t2"]
    assert calls[0].correlation_id == CORR  # its own motor prompt's Strain refs line


def test_non_array_content_shapes_are_safe() -> None:
    lines = [
        _event("assistant", "2026-11-16T09:00:00Z", "plain string content"),
        _event("assistant", "2026-11-16T09:00:01Z", {"type": "tool_use"}),
        _event("assistant", "2026-11-16T09:00:02Z", None),
        json.dumps({"type": "assistant", "timestamp": "2026-11-16T09:00:03Z", "message": "not a dict"}),
        "not json at all tool_use",
        _event("assistant", "2026-11-16T09:00:04Z", ["tool_use", 3, None, _use("ok", "python -P -m orion.dig help")]),
        _event("user", "2026-11-16T09:00:05Z", [_res("ok", [{"type": "text", "text": "\n\ndig: read-only tools"}])]),
    ]
    calls = digs.scan_transcript(lines, LO, HI, "s")
    assert [(c.tool_use_id, digs.first_line(c.result)) for c in calls] == [("ok", "dig: read-only tools")]
    # cockpit rows: the SQL guards jsonb_typeof; a row whose command is not a dig call is dropped
    rows = [{"ts": "2026-11-16 09:00:00", "id": "x", "command": "", "has_result": "f"},
            {"ts": "garbage", "id": "y", "command": "python -P -m orion.dig help", "has_result": "f"}]
    assert digs.parse_cockpit_rows(rows) == []
    assert "jsonb_typeof(s.raw->'step'->'raw'->'message'->'content')" in digs.cockpit_sql(DAY)


def test_a_dig_call_with_no_result_says_so() -> None:
    lines = [_event("assistant", "2026-11-16T09:00:00Z", [_use("t1", "python -P -m orion.dig organs 2026-11-15")])]
    (call,) = digs.scan_transcript(lines, LO, HI, "s")
    assert call.result is None
    line = digs.line_of(call)
    assert "-> (no result recorded)" in line
    digs.check_lines([line])


def test_failed_result_and_window_edges() -> None:
    lines = [
        _event("assistant", "2026-11-16T06:59:59Z", [_use("before", "python -P -m orion.dig help")]),
        _event("assistant", "2026-11-17T07:00:00Z", [_use("after", "python -P -m orion.dig help")]),
        _event("assistant", "2026-11-17T06:59:59Z", [_use("last", "python -P -m orion.dig field-note 1 write x")]),
        _event("user", "2026-11-17T07:00:02Z", [_res("last", "Exit code 1\nrefused", err=True)]),
        # a sub-agent's sidechain is not the run's own call
        _event("assistant", "2026-11-16T10:00:00Z", [_use("side", "python -P -m orion.dig help")], isSidechain=True),
    ]
    calls = digs.scan_transcript(lines, LO, HI, "s")
    assert [c.tool_use_id for c in calls] == ["last"]
    assert '-> failed: "Exit code 1"' in digs.line_of(calls[0])


def test_resumed_history_is_not_counted_twice() -> None:
    use = _event("assistant", "2026-11-16T09:00:00Z", [_use("t1", "python -P -m orion.dig help")])
    assert len(digs.scan_transcript([use, use], LO, HI, "s")) == 1


def test_warm_pool_chat_session_is_labelled_chat_and_never_inherits_a_turn_id() -> None:
    clear = _event("user", "2026-11-16T10:00:00Z", "<command-name>/clear</command-name>\n<command-message>clear</command-message>")
    lines = [
        _event("user", "2026-11-16T09:00:00Z", HELP_PROMPT),  # an earlier turn with an id
        clear,
        _event("user", "2026-11-16T10:00:01Z", "Orion harness motor. User message: what do you know about chicago"),
        _event("assistant", "2026-11-16T10:00:05Z", [_use("t1", 'python -P -m orion.dig know chicago 2>&1 | head -60')]),
    ]
    (call,) = digs.scan_transcript(lines, LO, HI, "pool-1")
    assert call.chat_pool and call.correlation_id == ""
    assert (call.where, call.ref) == ("chat (no turn id)", "[transcript:pool-1]")
    digs.check_lines([digs.line_of(call)])


# --------------------------------------------------------------------------- join and where


def test_merge_prefers_the_cockpit_row_and_lookup_names_the_workflow() -> None:
    cockpit = digs.parse_cockpit_rows([{
        "ts": "2026-11-17 06:45:00.448316", "correlation_id": "5b82e69b-aad5-4323-9272-c1dc31c933d9",
        "id": "call_1", "command": "python -P -m orion.dig help 2>&1 | head -60", "has_result": "t",
        "is_error": "false", "result": "dig: read-only tools\nmore", "chat_id": "5b82e69b-aad5-4323-9272-c1dc31c933d9",
        "session_id": "9c91d646-cbd0-4977-8cdd-d2c82045c7f9"}])
    lines = [_event("user", "2026-11-16T09:00:00Z", HELP_PROMPT),
             _event("assistant", "2026-11-17T06:45:00Z", [_use("call_1", "python -P -m orion.dig help")]),
             _event("assistant", "2026-11-16T09:01:00Z", [_use("t2", "python -P -m orion.dig field-note 3511")])]
    calls = digs.merge(cockpit, digs.scan_transcript(lines, LO, HI, "sess"))
    assert [(c.tool_use_id, c.source) for c in calls] == [("t2", "transcript"), ("call_1", "cockpit")]
    sql = digs.lookup_sql(c.correlation_id for c in calls if c.source == "transcript")
    assert sql and CORR in sql
    digs.attach(calls, [{"correlation_id": CORR, "workflow": "pr_field_note", "traced": "t", "chat_id": "", "session_id": ""}])
    lines_out = digs.render(calls)
    assert lines_out[0].startswith("02:01:00 pr_field_note $ python -P -m orion.dig field-note 3511 -> ")
    assert lines_out[0].endswith(f"[harness_turn_trace:{CORR}]")
    assert lines_out[1].startswith("23:45:00 chat 9c91d646 $ ") and '"dig: read-only tools"' in lines_out[1]
    assert lines_out[1].endswith("[chat:5b82e69b-aad5-4323-9272-c1dc31c933d9]")


def test_lookup_sql_refuses_non_uuid_ids() -> None:
    assert digs.lookup_sql(["x'; drop table chat_history_log; --", ""]) is None


def test_durable_step_linked_through_the_resume_registry(tmp_path) -> None:
    sid = session_id_for("pr-field-note-3510:pr_field_note", 0)
    sid2 = session_id_for("20261116T191407Z-9b68ae:curiosity_self_sense_eval", 2)
    carry = session_id_for("intention-0123456789abcdef:intention_carry:0123456789abcdef", 1)  # a 3-part key
    calls = [digs.Call(at="2026-11-16T07:49:27+00:00", tool_use_id="a", command="python -P -m orion.dig field-note 3510",
                       source="transcript", transcript=sid),
             digs.Call(at="2026-11-16T08:00:00+00:00", tool_use_id="b", command="python -P -m orion.dig help",
                       source="transcript", transcript=sid2),
             digs.Call(at="2026-11-16T08:00:30+00:00", tool_use_id="e", command="python -P -m orion.dig help",
                       source="transcript", transcript=carry),
             digs.Call(at="2026-11-16T08:01:00+00:00", tool_use_id="c", command="python -P -m orion.dig help",
                       source="transcript", transcript="random-session")]
    reg = tmp_path / "orion-resume"
    reg.mkdir()
    for i, (key, gen) in enumerate([("pr-field-note-3510:pr_field_note", 0),
                                    ("20261116T191407Z-9b68ae:curiosity_self_sense_eval", 2),
                                    ("intention-0123456789abcdef:intention_carry:0123456789abcdef", 1)]):
        (reg / f"{i:024x}.json").write_text(json.dumps({"key": key, "generation": gen, "state": "completed"}))
    (reg / "junk.json").write_text("not json")
    (reg / "x.lock").write_text("")
    records = digs.read_resume_registry(tmp_path)
    assert len(records) == 3
    assert digs.link_by_resume_key(calls, records) == 3
    assert [(c.where, c.ref) for c in calls] == [
        ("pr_field_note", "[durable_run:pr-field-note-3510]"),
        ("curiosity_self_sense_eval", "[durable_run:20261116T191407Z-9b68ae]"),
        ("intention_carry", "[durable_run:intention-0123456789abcdef]"),
        ("unlinked run", "[transcript:random-session]")]
    digs.check_lines(digs.render(calls))
    # a generation past the registry's last is not guessed
    late = digs.Call(at="2026-11-16T09:00:00+00:00", tool_use_id="f", command="python -P -m orion.dig help",
                     source="transcript", transcript=session_id_for("pr-field-note-3510:pr_field_note", 1))
    assert digs.link_by_resume_key([late], records) == 0
    # a traced turn whose trace row has no workflow (rows before the column was filled) keeps its trace ref
    # and gets its source tag as the workflow
    traced = digs.Call(at="2026-11-16T09:00:00+00:00", tool_use_id="d", command="python -P -m orion.dig help",
                       source="transcript", transcript=sid, correlation_id=CORR, traced=True)
    assert traced.needs_link and digs.link_by_resume_key([traced], records) == 1
    assert (traced.where, traced.ref) == ("pr_field_note", f"[harness_turn_trace:{CORR}]")


def test_merge_dedupes_a_duplicated_cockpit_hop() -> None:
    row = {"ts": "2026-11-16 09:00:00", "correlation_id": CORR, "id": "call_1", "command": "python -P -m orion.dig help",
           "has_result": "f", "chat_id": CORR, "session_id": "s"}
    assert len(digs.merge(digs.parse_cockpit_rows([row, dict(row)]), [])) == 1


def test_scan_stops_past_the_day_and_skips_calls_before_it() -> None:
    def gen():
        yield _event("assistant", "2026-11-16T06:00:00Z", [_use("early", "python -P -m orion.dig help")])
        yield _event("assistant", "2026-11-16T09:00:00Z", [_use("in", "python -P -m orion.dig help")])
        yield _event("user", "2026-11-17T10:00:00Z", [_res("in", "late but within grace? no: past 09:00Z")])
        raise AssertionError("read past the day plus grace")
    (call,) = digs.scan_transcript(gen(), LO, HI, "s")
    assert call.tool_use_id == "in" and call.result is None


# --------------------------------------------------------------------------- the line rule


@pytest.mark.parametrize("line", [
    "",
    "118 dig command(s) ran on 2026-11-16",                                     # a summary: no time, no ref
    "Orion dug a lot today, mostly field notes.",                                # prose
    "09:00:00 pr_field_note $ python -P -m orion.dig help -> \"x\"",             # no ref
    "pr_field_note $ python -P -m orion.dig help -> \"x\" [chat:abc]",           # no time
    "09:00:00 pr_field_note $ ls -la -> \"x\" [chat:abc]",                       # not a dig command
    "09:00:00 pr_field_note $ python -P -m orion.dig help -> \"x\" [experience:abc]",  # not this command's ref kind
    "09:00:00 pr_field_note $ python -P -m orion.dig help -> \"x\" [chat:abc] and more",
])
def test_line_rule_refuses(line) -> None:
    with pytest.raises(ValueError):
        digs.check_lines([line])


def test_line_rule_accepts_every_ref_kind() -> None:
    digs.check_lines([f"09:00:00 w $ python -P -m orion.dig help -> \"x\" [{k}:abc-1]" for k in digs.REF_KINDS])


def test_printed_command_is_the_one_that_ran_not_a_quoted_note() -> None:
    cmd = "printf '%s' 'notes about python -P -m orion.dig help and lots more' >> n; python -P -m orion.dig body 2>&1"
    c = digs.Call(at="2026-11-16T09:00:00+00:00", tool_use_id="a", command=cmd, result="ok", source="transcript",
                  transcript="s")
    assert " $ python -P -m orion.dig body -> " in digs.line_of(c)


def test_long_command_keeps_the_dig_part_and_cuts() -> None:
    cmd = "cd /mnt/orion-fcc/repo && " + "export X=1 && " * 20 + "python -P -m orion.dig field-note 3511 write \"" + "y" * 400 + "\""
    c = digs.Call(at="2026-11-16T09:00:00+00:00", tool_use_id="a", command=cmd, result="ok", source="transcript",
                  transcript="s")
    line = digs.line_of(c)
    shown = line.split(" $ ", 1)[1].split(" -> ")[0]
    assert shown.startswith("python -P -m orion.dig field-note 3511 write") and shown.endswith("…")
    assert len(shown) <= digs.COMMAND_CHARS
    digs.check_lines([line])


# --------------------------------------------------------------------------- the CLI


class _Runner:
    role = "orion_readonly"

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers, self.sql = answers, []

    def run(self, sql: str, flags):
        self.sql.append(sql)
        for key, out in self.answers.items():
            if key in sql:
                return out, (1 if out.startswith("ERROR") else 0)
        return "", 0


def _write_transcript(home: Path, name: str, lines: list[str]) -> Path:
    p = home / "projects" / "-mnt-orion-fcc-repo" / f"{name}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    t = datetime(2026, 11, 16, 12, tzinfo=timezone.utc).timestamp()
    os.utime(p, (t, t))
    return p


def test_cli_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "dighome"))
    home = tmp_path / "claude"
    _write_transcript(home, "sess-a", [
        _event("user", "2026-11-16T09:00:00Z", HELP_PROMPT),
        _event("assistant", "2026-11-16T09:00:04Z", [_use("t2", "python -P -m orion.dig body 2>&1 | tail -40")]),
        _event("user", "2026-11-16T09:00:09Z", [_res("t2", "body atlas as of now\nmore")]),
    ])
    # began after the day: never opened
    _write_transcript(home, "sess-late", [
        _event("assistant", "2026-11-17T08:00:00Z", [_use("t9", "python -P -m orion.dig help")])])
    lookup = ("correlation_id,workflow,traced,chat_id,session_id\n"
              f"{CORR},pr_field_note,t,,\n")
    runner = _Runner({"FROM cockpit_turn_sighting": "ts,correlation_id,id,command,has_result,is_error,result,chat_id,session_id\n",
                      "unnest(ARRAY": lookup})
    text = dig.cmd_digs(DAY, dig.Out("t"), runner=runner, claude_home=home)
    lines = text.splitlines()
    assert lines[0] == (f'02:00:04 pr_field_note $ python -P -m orion.dig body -> "body atlas as of now" '
                        f"[harness_turn_trace:{CORR}]")
    assert "1 dig command(s) ran on 2026-11-16 (times Denver): 0 in chat, 1 in background runs" in text
    assert "UNREAD" not in text
    # read-only: every query is a SELECT/WITH; no durable_run lookup was needed (all linked)
    assert all(s.lstrip().upper().startswith(("SELECT", "WITH")) for s in runner.sql)
    assert not any("durable_run_workflow" in s for s in runner.sql)


def test_cli_says_unread_never_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path / "dighome"))
    runner = _Runner({"FROM cockpit_turn_sighting": "ERROR: permission denied for table cockpit_turn_sighting"})
    text = dig.cmd_digs(DAY, dig.Out("t"), runner=runner, claude_home=tmp_path / "nowhere")
    assert "UNREAD chat turns (cockpit_turn_sighting)" in text and "needs GRANT SELECT" in text
    assert "UNREAD background runs (claude transcripts)" in text


def test_main_routes_digs(monkeypatch, capsys) -> None:
    seen = {}
    monkeypatch.setattr(dig, "cmd_digs", lambda day, out: seen.setdefault("day", day) and "ok")
    assert dig.main(["digs", "2026-11-16"]) == 0 and seen["day"] == DAY
    assert dig.main(["digs"]) == 2
    assert "digs <date>" in dig.HELP
