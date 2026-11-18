"""`dig watch`: Orion's own switch for watching a person's game (game company patch 2). No database, no bus."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.dig import __main__ as dig
from orion.dig import look as lk
from orion.dig import watch as wt

NOW = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)
SITTING = {"sitting_id": "s1", "game_id": "a-link-to-the-past", "game_title": "The Legend of Zelda: A Link to the Past",
           "started_at": "2026-11-18 19:00:00+00", "last_input_at": "2026-11-18 19:59:50+00", "player_stated": ""}
GAMES = [{"game_id": "a-link-to-the-past", "game_title": "The Legend of Zelda: A Link to the Past", "sittings": "9"},
         {"game_id": "the-legend-of-zelda-prg1", "game_title": "The Legend of Zelda", "sittings": "13"},
         {"game_id": "tetris", "game_title": "Tetris", "sittings": "3"}]


@pytest.fixture(autouse=True)
def published(monkeypatch) -> list:
    sent: list = []
    monkeypatch.setattr(lk, "_publish_bus", lambda events: sent.extend(events))
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("ORION_TURN_CORRELATION_ID", "turn-77")
    return sent


def rows_for(open_sitting=True, extra=None):
    def rows(sql: str) -> list[dict]:
        if sql == wt.OPEN_SITTING_SQL:
            return [dict(SITTING)] if open_sitting else []
        if sql == wt.GAMES_SQL:
            return [dict(g) for g in GAMES]
        for key, value in (extra or {}).items():
            if key in sql:
                if isinstance(value, Exception):
                    raise value
                return value
        return []
    return rows


def test_parse():
    assert wt.parse([]) == {"action": "status"} and wt.parse(["status"]) == {"action": "status"}
    assert wt.parse(["off", "--why", "I", "want", "to", "--just", "keep", "company"]) == {
        "action": "set", "mode": "off", "game": None, "why": "I want to --just keep company"}
    assert wt.parse(["default", "zelda", "nes", "off", "--why", "not", "this"])["game"] == "zelda nes"
    for bad in (["loud"], ["default", "off"], ["status", "--why", "x"]):
        with pytest.raises(wt.WatchError):
            wt.parse(bad)


def test_off_for_this_sitting_is_one_act_with_orions_words(published):
    lines, entry = wt.set_choice("off", "I would rather just keep you company", rows=rows_for(), now=NOW)
    opened, closed = published
    assert opened.event_kind == "act_opened" and closed.event_kind == "act_closed"
    dumped = closed.model_dump(mode="json")
    text = str(dumped)
    assert "orion.watch_choice" in text and "'regulate'" in text and "retro_sitting:s1" in text
    assert "watch:off" in text and "watch off: I would rather just keep you company" in text
    assert closed.session_id == "turn-77" and entry["act_id"] and entry["object_ref"] == "retro_sitting:s1"
    assert lines[0] == "watch off for this sitting of The Legend of Zelda: A Link to the Past: I would rather just keep you company"
    assert any("lower of your setting and Juniper" in ln for ln in lines)
    assert not closed.model_dump(mode="json").get("correlation_id")      # a choice inside a turn: charged to no lease


def test_off_and_quiet_need_words_and_on_does_not(published):
    with pytest.raises(wt.WatchError, match="say why"):
        wt.set_choice("off", None, rows=rows_for(), now=NOW)
    with pytest.raises(wt.WatchError, match="3-200"):
        wt.set_choice("quiet", "x" * 201, rows=rows_for(), now=NOW)
    lines, entry = wt.set_choice("on", None, rows=rows_for(), now=NOW)
    assert entry["why"] is None and len(published) == 2


def test_no_sitting_open_points_to_the_game_default(published):
    with pytest.raises(wt.WatchError, match="nobody is playing now"):
        wt.set_choice("off", "rest", rows=rows_for(open_sitting=False), now=NOW)
    assert not published


def test_a_game_default_names_the_game_by_id_or_by_one_word_of_its_title(published):
    lines, entry = wt.set_choice("off", "not this one", rows=rows_for(open_sitting=False), game="tetris", now=NOW)
    assert entry["object_ref"] == "retro_game:tetris" and lines[0].startswith("watch off for every sitting of Tetris")
    _, entry = wt.set_choice("off", "not this one", rows=rows_for(), game="Link to the Past", now=NOW)
    assert entry["object_ref"] == "retro_game:a-link-to-the-past"
    with pytest.raises(wt.WatchError, match="names 2 games"):
        wt.set_choice("off", "not this one", rows=rows_for(), game="zelda", now=NOW)
    _, entry = wt.set_choice("quiet", "ahead of time", rows=rows_for(), game="super-metroid", now=NOW)
    assert entry["object_ref"] == "retro_game:super-metroid"          # a game nobody played yet
    with pytest.raises(wt.WatchError, match="no game"):
        wt.set_choice("off", "words", rows=rows_for(), game="Ms. Pac Man!", now=NOW)


def test_status_says_who_set_the_mode_and_shows_the_counts_and_looks():
    extra = {
        "FROM retro_watch_preference": [
            {"scope": "game", "scope_id": "a-link-to-the-past", "set_by": "orion", "mode": "off", "why": "standing", "set_at": "1"},
            {"scope": "all", "scope_id": "", "set_by": "juniper", "mode": "on", "why": "", "set_at": "2"}],
        "FROM retro_watch_floor": [{"floor": "0.0112", "status": "set", "checks_input": "900", "checks_no_input": "950"}],
        "FROM retro_watch_look": [{"looked_at": "2026-11-18 19:58", "outcome": "stored", "verdict": "unchecked",
                                   "failure": "", "note": "Link walks north."}],
        "FROM retro_sitting WHERE sitting_id": [{c: "3" for c in __import__("orion.retro.watch", fromlist=["x"]).COUNTERS}],
        "FROM want_ledger": [{"decided_at": "2026-11-18 19:50", "outcome": "deferred", "reason": "price", "lane": "chat",
                              "net": "-0.1", "binding_constraint": "lane:chat"}],
    }
    lines, payload = wt.status(rows=rows_for(extra=extra), now=NOW)
    text = "\n".join(lines)
    assert "mode in force: off, set by you: standing" in text
    assert "floor for a-link-to-the-past: 0.0112 (set; 900 played / 950 still checks)" in text
    assert "counts: ring checks 3, checks over floor 3" in text and "look 2026-11-18 19:58 stored (unchecked): Link walks north." in text
    assert "last refused look bid: 2026-11-18 19:50 deferred (price; lane chat" in text


def test_status_names_a_table_it_cannot_read():
    extra = {"FROM want_ledger": RuntimeError("needs GRANT SELECT ON want_ledger TO orion_readonly; permission denied")}
    lines, payload = wt.status(rows=rows_for(extra=extra), now=NOW)
    assert any(ln.startswith("last_refusal: unread (needs GRANT SELECT ON want_ledger") for ln in lines)
    assert "last_refusal_error" in payload


def test_the_cli_routes_watch(monkeypatch, tmp_path, published):
    monkeypatch.setenv("ORION_DIG_HOME", str(tmp_path))
    calls = []

    def fake_cmd(args, out, runner=None, publish=None):
        calls.append(args)
        return "ok"
    monkeypatch.setattr(dig, "cmd_watch", fake_cmd)
    assert dig.main(["watch", "off", "--why", "rest"]) == 0
    assert dig.main(["watch"]) == 0
    assert calls == [["off", "--why", "rest"], []]
