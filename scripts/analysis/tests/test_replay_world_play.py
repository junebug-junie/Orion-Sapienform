"""The world:play replay (scripts/analysis/replay_world_play.py) on a small synthetic week: it reproduces every
stored winner, counts reading's losses, honours the chat yield, and the note rule holds out a sitting with no look."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(REPO))

import replay_world_play as rp  # noqa: E402

T = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)


def _cand(sid, *, eligible, rank=0.5):
    return {"source_id": sid, "eligible": eligible, "score": rank, "rank_score": rank, "reason": "x"}


def _frames():
    out = []
    for k in range(12):                         # one tick every 5 min, 20:00..20:55, inside the live sitting
        t = T + timedelta(minutes=5 * k)
        cands = [_cand("world:reading", eligible=True, rank=0.91), _cand("node:substrate.execution", eligible=False)]
        if k >= 10:
            cands.append(_cand("world:chat", eligible=True, rank=0.95))
        winner = "world:chat" if k >= 10 else "world:reading"
        out.append((t, {"winner": winner, "candidates": cands}))
    return out


def _sittings():
    rows = [{"sitting_id": f"b{d}", "started_at": T - timedelta(days=d, hours=1), "last_input_at": T - timedelta(days=d),
             "ended_at": T - timedelta(days=d), "backfilled": True, "game_title": "Crystalis", "game_id": "crystalis"}
            for d in range(1, 7)]
    rows.append({"sitting_id": "S1", "started_at": T - timedelta(minutes=40), "last_input_at": T + timedelta(hours=1),
                 "ended_at": None, "backfilled": False, "game_title": "Crystalis", "game_id": "crystalis"})
    return rows


def test_the_mode_rule_takes_reading_ticks_but_never_chats():
    r = rp.replay(_frames(), _sittings(), [], [], "mode")
    assert r["stored_winner_reproduced"] == 12 and r["ticks_with_open_sitting"] == 12
    assert r["reading_lost_to_play"] == r["play_won"] == 10 and r["yielded_to_chat"] == 2
    assert r["during_sittings"]["all"]["reading_share_before"] == round(10 / 12, 3)
    assert r["during_sittings"]["all"]["reading_share_after"] == 0.0


def test_the_note_rule_needs_a_fresh_look_note():
    looks = [{"sitting_id": "S1", "looked_at": (T + timedelta(minutes=14)).isoformat(), "outcome": "stored",
              "note": "a hero walks"}]
    r = rp.replay(_frames(), _sittings(), looks, [], "note")
    # Only the ticks within 3 min after the note (20:15) are watched.
    assert r["play_won"] == 1 and r["reading_lost_to_play"] == 1
    none = rp.replay(_frames(), _sittings(), [], [], "note")
    assert none["play_won"] == 0 and none["play_eligible_ticks"] == 0


def test_orion_off_holds_play_out_of_every_rule():
    prefs = [{"scope": "game", "scope_id": "crystalis", "set_by": "orion", "mode": "off", "why": "no",
              "set_at": (T - timedelta(days=1)).isoformat()}]
    for rule in rp.RULES:
        assert rp.replay(_frames(), _sittings(), [], prefs, rule)["play_won"] == 0


def test_main_reads_the_exports(tmp_path, capsys):
    f = tmp_path / "frames.jsonl"
    f.write_text("".join(json.dumps({"t": t.isoformat(), "wf": wf}) + "\n" for t, wf in _frames()))
    s = tmp_path / "sittings.jsonl"
    s.write_text("".join(json.dumps({k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r.items()}) + "\n"
                         for r in _sittings()))
    (tmp_path / "looks.jsonl").write_text("")
    (tmp_path / "prefs.jsonl").write_text("")
    sys.argv = ["x", str(f), str(s), str(tmp_path / "looks.jsonl"), str(tmp_path / "prefs.jsonl"), "--json"]
    assert rp.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["rule"] for r in out] == list(rp.RULES) + [f"hunger:{rp.PLAY_HUNGER_TAU_SEC:g}"]


def test_the_hunger_rule_resets_on_a_win_and_counts_wins_per_sitting():
    # A look note every 2 minutes keeps Orion watching; reading sits at 0.91 every tick (ticks 5 min apart).
    looks = [{"sitting_id": "S1", "looked_at": (T + timedelta(minutes=2 * k)).isoformat(), "outcome": "stored",
              "note": "a hero walks"} for k in range(30)]
    r = rp.replay(_frames(), _sittings(), looks, [], "hunger:90")
    # tau 90 s: 300 s after a win the pull is 1 - e^(-300/90) = 0.964 > 0.91, so play wins all 10 non-chat
    # ticks; the 2 chat ticks yield.
    assert (r["play_won"], r["yielded_to_chat"], r["reading_lost_to_play"]) == (10, 2, 10)
    assert r["by_length"]["medium"] == {"sittings": 1, "watched_sittings": 1, "watched_won_once": 1,
                                        "watched_ticks": 12, "wins": 10, "wins_per_watched_sitting": 10.0}
    # tau 600 s: the first tick wins on the pull built since the sitting began (40 min). The win resets it, and
    # it passes 0.91 again only 25 min later (1 - e^(-2.5) = 0.92): 2 wins of 10, reading keeps the other 8.
    slow = rp.replay(_frames(), _sittings(), looks, [], "hunger:600")
    assert slow["play_won"] == 2 and slow["reading_lost_to_play"] == 2
