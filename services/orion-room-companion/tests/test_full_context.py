"""Patch 1a: the whole conversation for Claude (companion half).

Design: docs/superpowers/specs/2026-11-12-claude-hub-tools-orion-consent-design.md,
acceptance checks 2 (stdin) and 3 (no duplicates on resume), plus the daily
rotation (open question 6).
"""

from __future__ import annotations

import json
import stat
import sys
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.claude_session import ClaudeTurnResult, build_argv, run_room_turn
from app.main import room_key, run_turn
from app.room_prompt import build_turn_prompt, lines_after_mark, newest_created_at, render_line
from app.session_store import (
    _read_all,
    _write_all,
    peek_or_mint_session,
    remember_session,
    room_day,
)
from app.settings import Settings
from orion.schemas.room_claude import RoomClaudeRequestV1, RoomTranscriptEntryV1

KW = dict(
    claude_bin="claude",
    model="claude-sonnet-5",
    effort="medium",
    setting_sources_env_key="ROOM_COMPANION_SETTING_SOURCES",
)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        ROOM_COMPANION_SESSION_STATE_PATH=str(tmp_path / "sessions.json"),
        ROOM_COMPANION_CLAUDE_CONFIG_DIR="/root/.claude",
        ROOM_COMPANION_WORKSPACE=str(tmp_path / "workspace"),
    )


def _line(text, at, *, who="juniper", ref="aaaa1111", **kw) -> RoomTranscriptEntryV1:
    name = {"juniper": "Juniper", "orion": "Oríon", "claude": "Claude"}[who]
    return RoomTranscriptEntryV1(
        speaker_id=who, speaker_name=name, text=text, created_at=at, turn_ref=ref, **kw
    )


def _ok(session="sess-1") -> ClaudeTurnResult:
    return ClaudeTurnResult(
        ok=True, text="hi", claude_session_id=session, model="claude-sonnet-5",
        cost_usd=0.01, duration_ms=10, exit_code=0,
    )


# -- stdin (check 2) ---------------------------------------------------------


def test_prompt_is_never_in_argv():
    argv = build_argv(session_id="s", resume=False, **KW)
    assert argv[:2] == ["claude", "-p"]
    assert argv[2] == "--output-format", "nothing may sit between -p and the flags: the prompt goes over stdin"


def test_run_room_turn_sends_the_prompt_on_stdin():
    seen = {}

    def _fake_run(argv, **kw):
        seen["argv"], seen["input"] = argv, kw.get("input")
        import subprocess

        return subprocess.CompletedProcess(argv, 0, json.dumps({"result": "ok", "session_id": "s"}), "")

    with patch("app.claude_session.subprocess.run", side_effect=_fake_run):
        run_room_turn("the prompt", session_id="s", resume=False, timeout_sec=5, **KW)
    assert seen["input"] == "the prompt"
    assert "the prompt" not in seen["argv"]


def test_a_200k_prompt_reaches_a_real_subprocess_whole(tmp_path):
    """A real child process, not a mock: a fake `claude` that reads stdin and
    reports what it got. 200,000 characters is over Linux's 131,072-byte
    single-argument limit, so this would fail with E2BIG if it went in argv."""
    fake = tmp_path / "claude"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "data = sys.stdin.read()\n"
        "print(json.dumps({'result': f'len={len(data)} tail={data[-12:]}', 'session_id': 's',"
        " 'argv_len': sum(len(a) for a in sys.argv)}))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    prompt = ("x" * 199_988) + "MARKER-END-1"
    assert len(prompt) == 200_000
    result = run_room_turn(
        prompt, session_id="s", resume=False, timeout_sec=20,
        claude_bin=str(fake), model="m", effort="", setting_sources_env_key="NOPE",
    )
    assert result.ok, result.error
    assert result.text == "len=200000 tail=MARKER-END-1"


# -- daily rotation (Q6) -------------------------------------------------------


def test_room_day_turns_over_at_four_am_denver():
    # 2026-11-12 is MST (UTC-7): 04:00 local = 11:00 UTC.
    before = datetime(2026, 11, 12, 10, 59, tzinfo=timezone.utc)
    after = datetime(2026, 11, 12, 11, 0, tzinfo=timezone.utc)
    assert room_day(before) == "2026-11-11"
    assert room_day(after) == "2026-11-12"


def test_session_rotates_at_the_first_invite_of_a_new_room_day(tmp_path):
    path = tmp_path / "s.json"
    day1 = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)
    day2 = datetime(2026, 11, 13, 11, 5, tzinfo=timezone.utc)
    remember_session(path, "hub:hub-direct", "sess-a", day=room_day(day1), sent_up_to="2026-11-12T14:00:00+00:00")

    same = peek_or_mint_session(path, "hub:hub-direct", now=day1)
    assert same.resume and same.session_id == "sess-a"
    assert same.sent_up_to == "2026-11-12T14:00:00+00:00"

    nxt = peek_or_mint_session(path, "hub:hub-direct", now=day2)
    assert not nxt.resume and nxt.session_id != "sess-a"
    assert nxt.minted_reason == "rotated" and nxt.sent_up_to is None


def test_legacy_bare_string_entry_rotates_once(tmp_path):
    """Before 2026-11-12 an entry was a bare session id with no day. The first
    invite after deploy reseeds with the whole conversation."""
    path = tmp_path / "s.json"
    _write_all(path, {"hub:hub-direct": "legacy-sess"})
    slot = peek_or_mint_session(path, "hub:hub-direct")
    assert not slot.resume and slot.minted_reason == "rotated"


def test_consult_rooms_never_rotate(tmp_path):
    path = tmp_path / "s.json"
    remember_session(path, "hub:consult:abc", "c-sess", day="2026-11-01")
    slot = peek_or_mint_session(path, "hub:consult:abc", rotate_daily=False)
    assert slot.resume and slot.session_id == "c-sess"
    _write_all(path, {"hub:consult:old": "legacy"})
    assert peek_or_mint_session(path, "hub:consult:old", rotate_daily=False).resume


def test_mark_only_moves_forward(tmp_path):
    path = tmp_path / "s.json"
    remember_session(path, "k", "s1", day="d", sent_up_to="2026-11-12T10:00:00+00:00")
    remember_session(path, "k", "s1", day="d", sent_up_to="2026-11-12T09:00:00+00:00")
    remember_session(path, "k", "s1", day="d", sent_up_to=None)
    assert _read_all(path)["k"]["sent_up_to"] == "2026-11-12T10:00:00+00:00"


# -- no duplicates on resume (check 3) -----------------------------------------


def test_resumed_turn_sends_only_lines_newer_than_the_mark(settings):
    """The lifecycle: turn 1 seeds two lines; turn 2's request carries the
    same two plus one new one; only the new one is sent."""
    a = _line("first", "2026-11-12T15:00:00+00:00", ref="t1")
    b = _line("orion reply", "2026-11-12T15:00:00+00:00", who="orion", ref="t1")
    c = _line("second", "2026-11-12T15:30:00+00:00", ref="t2")
    prompts = []
    now = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)
    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: prompts.append((p, kw)) or _ok()), \
            patch("app.session_store.datetime") as dt:
        dt.now.return_value = now
        dt.fromisoformat = datetime.fromisoformat
        req = dict(room_id="hub-direct", invited_by="Juniper", prompt="hey")
        run_turn(settings, RoomClaudeRequestV1(**req, transcript=[a, b]))
        run_turn(settings, RoomClaudeRequestV1(**req, transcript=[a, b, c]))
        run_turn(settings, RoomClaudeRequestV1(**req, transcript=[a, b, c]))

    assert prompts[0][1]["resume"] is False and "first" in prompts[0][0]
    assert prompts[1][1]["resume"] is True
    assert "second" in prompts[1][0] and "first" not in prompts[1][0] and "orion reply" not in prompts[1][0]
    assert prompts[2][0] == "Juniper: hey", "nothing new since the mark -> just the new message"


def test_resume_skips_side_exchange_and_claudes_own_lines():
    lines = [
        _line("@claude q", "2026-11-12T15:40:00+00:00", side_exchange=True),
        _line("my answer", "2026-11-12T15:40:00+00:00", who="claude", side_exchange=True),
        _line("new juniper", "2026-11-12T15:41:00+00:00"),
    ]
    out = lines_after_mark(lines, sent_up_to="2026-11-12T15:00:00+00:00", claude_participant_id="claude")
    assert [e.text for e in out] == ["new juniper"]


def test_unreadable_time_is_kept_not_dropped():
    out = lines_after_mark(
        [_line("x", "not a time")], sent_up_to="2026-11-12T15:00:00+00:00", claude_participant_id="claude"
    )
    assert len(out) == 1


def test_mark_is_stored_after_success_only(settings):
    a = _line("first", "2026-11-12T15:00:00+00:00")
    fail = ClaudeTurnResult(ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
                            duration_ms=1, exit_code=1, error="boom")
    req = RoomClaudeRequestV1(room_id="hub-direct", invited_by="Juniper", prompt="hey", transcript=[a])
    with patch("app.main.run_room_turn", return_value=fail):
        run_turn(settings, req)
    assert _read_all(settings.ROOM_COMPANION_SESSION_STATE_PATH) == {}
    with patch("app.main.run_room_turn", return_value=_ok()):
        run_turn(settings, req)
    entry = _read_all(settings.ROOM_COMPANION_SESSION_STATE_PATH)[room_key(req)]
    assert entry["sent_up_to"] == "2026-11-12T15:00:00+00:00"
    assert entry["day"] == room_day()


def test_newest_created_at_picks_the_latest_readable():
    lines = [_line("a", "2026-11-12T15:00:00+00:00"), _line("b", "bad"), _line("c", "2026-11-12T16:00:00Z")]
    assert newest_created_at(lines) == "2026-11-12T16:00:00Z"


# -- rendering ---------------------------------------------------------------------


def test_line_carries_label_turn_ref_and_local_time():
    e = _line("hello", "2026-11-12T16:16:00+00:00", who="orion", ref="1a2b3c4d", unprompted=True)
    assert render_line(e) == "Oríon (unprompted) [turn:1a2b3c4d] (Thu Nov 12 09:16): hello"


def test_line_without_turn_ref_shows_no_time():
    """Old Hub stamped every line with the invite time; showing it would lie."""
    e = RoomTranscriptEntryV1(speaker_id="juniper", speaker_name="Juniper", text="hi")
    assert render_line(e) == "Juniper: hi"


def test_first_turn_has_hub_index_and_earlier_note_resumed_turn_does_not():
    lines = [_line("hello", "2026-11-12T16:16:00+00:00")]
    first = build_turn_prompt(
        prompt="q", invited_by_name="Juniper", transcript=lines, social_memory_summary={},
        first_turn=True, hub_index="- Open asks: 5", transcript_note="[Earlier: 3 turns not shown.]",
    )
    assert "What is open in Hub now (names and counts only):\n- Open asks: 5" in first
    assert "Conversation so far, oldest first:\n[Earlier: 3 turns not shown.]\nJuniper [turn:" in first
    later = build_turn_prompt(
        prompt="q", invited_by_name="Juniper", transcript=lines, social_memory_summary={},
        first_turn=False, hub_index="- Open asks: 5", transcript_note="[Earlier: 3 turns not shown.]",
    )
    assert "Open asks" not in later and "Earlier" not in later


def test_old_hub_payload_without_new_fields_still_parses():
    """Consumer first: the companion accepts what today's Hub sends."""
    req = RoomClaudeRequestV1.model_validate({
        "room_id": "hub-direct", "invited_by": "Juniper", "prompt": "hi",
        "transcript": [{"speaker_id": "juniper", "speaker_name": "Juniper", "text": "x"}],
    })
    assert req.hub_index is None and req.transcript[0].turn_ref is None


# -- review findings (2026-11-12) ---------------------------------------------------


def _run_two(settings, first, second):
    prompts = []
    now = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)
    with patch("app.main.run_room_turn", side_effect=lambda p, **kw: prompts.append(p) or _ok()), \
            patch("app.session_store.datetime") as dt:
        dt.now.return_value = now
        dt.fromisoformat = datetime.fromisoformat
        for transcript, sid in (first, second):
            run_turn(settings, RoomClaudeRequestV1(
                room_id="hub-direct", invited_by="Juniper", prompt="hey", session_id=sid, transcript=transcript,
            ))
    return prompts


def test_orions_reply_merged_into_a_half_written_row_is_still_sent(settings):
    """Hub writes Juniper's half when the turn starts and merges Orion's reply
    into the same row later, with the same created_at. Juniper invited Claude
    in between. A time mark would drop Orion's reply forever."""
    at = "2026-11-12T15:00:00+00:00"
    juniper = _line("what about X?", at, ref="t9")
    orion = _line("X is fine.", at, who="orion", ref="t9")
    prompts = _run_two(settings, ([juniper], "s1"), ([juniper, orion], "s1"))
    assert "X is fine." in prompts[1]
    assert "what about X?" not in prompts[1]


def test_a_second_hub_session_does_not_hide_its_older_lines(settings):
    """All Hub sessions share one Claude room session. Session B's lines are
    older than session A's newest one, and must still be sent."""
    a = _line("from A", "2026-11-12T15:05:00+00:00", ref="a1")
    b = _line("from B", "2026-11-12T14:00:00+00:00", ref="b1")
    prompts = _run_two(settings, ([a], "hub-A"), ([b], "hub-B"))
    assert "from B" in prompts[1]


def test_sent_keys_are_stored_and_bounded(tmp_path):
    from app.session_store import MAX_SENT_KEYS

    path = tmp_path / "s.json"
    remember_session(path, "k", "s1", day="d", sent_keys=[f"r{i}|j" for i in range(MAX_SENT_KEYS + 10)])
    remember_session(path, "k", "s1", day="d", sent_keys=["new|j"])
    keys = _read_all(path)["k"]["sent_keys"]
    assert len(keys) == MAX_SENT_KEYS and keys[-1] == "new|j"
    remember_session(path, "k", "s2", day="d", sent_keys=["x|j"])
    assert _read_all(path)["k"]["sent_keys"] == ["x|j"], "a new session starts its keys over"


def test_unset_new_fields_are_left_out_of_the_payload():
    """An old companion (extra=forbid) must accept what a new producer sends
    when it does not use the new fields -- that is what makes
    HUB_ROOM_CLAUDE_CONTEXT_MODE=recent a real rollback."""
    req = RoomClaudeRequestV1(
        room_id="hub-direct", invited_by="Juniper", prompt="hi",
        transcript=[RoomTranscriptEntryV1(speaker_id="juniper", speaker_name="Juniper", text="x")],
    )
    data = req.model_dump(mode="json")
    assert "hub_index" not in data and "transcript_note" not in data
    assert set(data["transcript"][0]) == {"speaker_id", "speaker_name", "speaker_kind", "text", "created_at"}
    full = RoomClaudeRequestV1(
        room_id="hub-direct", invited_by="Juniper", prompt="hi", hub_index="i",
        transcript=[_line("x", "2026-11-12T15:00:00+00:00", side_exchange=True)],
    )
    assert RoomClaudeRequestV1.model_validate(full.model_dump(mode="json")) == full
