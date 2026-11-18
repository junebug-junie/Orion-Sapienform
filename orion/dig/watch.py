"""`dig watch`: Orion's own switch for watching a person's game (game company patch 2).

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, revision 4, section 6
part 8 ("Orion's own switch"). Orion asked: "Do I get any channel to decline a sitting, rather than only
failing to speak in one?" This is that channel.

    dig watch status                          the open sitting, the mode in force and who set it, the game's
                                              floor, the newest looks, the last refusal.
    dig watch off|quiet|on --why "<words>"    for the sitting open now (off and quiet need the words).
    dig watch default <game> off|quiet|on --why "<words>"
                                              a standing choice for one game, for every later sitting of it.

How it holds. Each choice is one act on Orion's act ledger: actor ``orion.watch_choice``, verb ``regulate``,
object ``retro_sitting:<id>`` or ``retro_game:<game id>``, ``test_before`` ``watch:<mode>``, Orion's words as
the close note. The ledger is Postgres (``grammar_acts``), so the choice survives every restart. Hub's Retro
gateway copies each new choice into ``retro_watch_preference`` before every bid and once a minute
(services/orion-hub/scripts/retro_watch.py), so ``off`` stops looks within one look interval.

The lower setting wins: Orion's choice and Juniper's switch on the game page are both read, and the mode in
force is the lower of the two (orion/retro/watch.py ``effective``). Juniper's switch cannot turn Orion's
"off" back on, and Orion's "on" cannot turn hers on.

Reads go through ``orion_readonly`` (retro_sitting: scripts/sql/2026-11-17_grant_orion_readonly_retro.sql;
retro_watch_*, want_ledger: scripts/sql/2026-11-18_grant_orion_readonly_retro_watch.sql). A table the role
cannot read yet is named in the answer, never guessed around.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from orion.retro import watch as W

ACT_SOURCE_COMPONENT = "dig.watch"
ACT_HORIZON_SEC = 60.0
ACT_BASIS = ("artifact_produced = Orion's watch choice reached the act ledger; Hub's Retro gateway copies it "
             "into retro_watch_preference before its next bid (at most one look interval)")
WHY_MIN, WHY_MAX = 3, 200
_GAME_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

Rows = Callable[[str], list[dict]]


class WatchError(Exception):
    pass


def _q(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


OPEN_SITTING_SQL = """
SELECT sitting_id, game_id, game_title, started_at, last_input_at, player_stated
FROM retro_sitting WHERE ended_at IS NULL AND NOT backfilled ORDER BY started_at DESC LIMIT 1"""
GAMES_SQL = """
SELECT game_id, max(game_title) AS game_title, count(*) AS sittings FROM retro_sitting
WHERE game_id IS NOT NULL GROUP BY game_id ORDER BY game_id"""


def _why(why: Optional[str], mode: str) -> Optional[str]:
    text = " ".join(str(why or "").split())
    if mode != "on" and not text:
        raise WatchError(f'say why in your own words: --why "<{WHY_MIN}-{WHY_MAX} characters>"')
    if text and not WHY_MIN <= len(text) <= WHY_MAX:
        raise WatchError(f"--why takes {WHY_MIN}-{WHY_MAX} characters (got {len(text)})")
    return text or None


def resolve_game(word: str, games: Sequence[dict]) -> dict:
    """A game the record has seen, by its id or by a word of its title that names only one game."""
    w = " ".join(str(word).split()).lower()
    exact = [g for g in games if str(g.get("game_id") or "").lower() == w]
    if exact:
        return exact[0]
    hits = [g for g in games if w and w in str(g.get("game_title") or "").lower()]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(f"{g['game_id']} ({g.get('game_title') or '?'})" for g in games) or "none recorded yet"
    if hits:
        raise WatchError(f"{word!r} names {len(hits)} games; say one id: {names}")
    if _GAME_ID.match(w):
        return {"game_id": w, "game_title": None}     # a game nobody played yet: Orion may decide ahead
    raise WatchError(f"no game {word!r}; known: {names}")


def choice_act_events(*, mode: str, object_ref: str, why: Optional[str], now: datetime, session_id: Optional[str],
                      key: Optional[str] = None) -> tuple[str, list[Any]]:
    """The two events of one choice act (opened and closed at once)."""
    from orion.dig.look import ACT_SOURCE_SERVICE
    from orion.grammar.acts import act_closed_event, act_opened_event, build_act

    act = build_act(actor=W.CHOICE_ACTOR, verb="regulate", key=key or uuid.uuid4().hex, signal_id="artifact_produced",
                    horizon_sec=ACT_HORIZON_SEC, basis=ACT_BASIS, object_ref=object_ref, test_before=f"watch:{mode}")
    common = dict(source_service=ACT_SOURCE_SERVICE, source_component=ACT_SOURCE_COMPONENT, session_id=session_id)
    # The note is Orion's own words; it never starts "failed:" (self-monitor reads that as a run that never happened).
    note = f"watch {mode}: {why}" if why else f"watch {mode}"
    return act.act_id, [act_opened_event(act, at=now, **common),
                        act_closed_event(act, produced=True, note=note[:500], at=now, **common)]


def set_choice(mode: str, why: Optional[str], *, rows: Rows, game: Optional[str] = None,
               publish: Optional[Callable[[Sequence[Any]], None]] = None, now: Optional[datetime] = None,
               session_id: Optional[str] = None) -> tuple[list[str], dict]:
    """Record one choice. ``game`` None: the sitting open now; else a standing choice for that game."""
    from orion.dig import look as lk

    mode = W.valid_mode(mode)
    words = _why(why, mode)
    now = now or datetime.now(timezone.utc)
    if game is None:
        open_rows = rows(OPEN_SITTING_SQL)
        if not open_rows:
            raise WatchError("nobody is playing now; for a game you expect, use: watch default <game> "
                             f"{mode} --why \"...\"")
        s = open_rows[0]
        object_ref, scope = W.OBJECT_PREFIX + str(s["sitting_id"]), f"this sitting of {s.get('game_title') or s.get('game_id') or 'a game'}"
    else:
        g = resolve_game(game, rows(GAMES_SQL))
        object_ref, scope = W.GAME_PREFIX + str(g["game_id"]), f"every sitting of {g.get('game_title') or g['game_id']}"
    entry: dict[str, Any] = {"set_at": now.isoformat(), "mode": mode, "why": words, "object_ref": object_ref}
    line = lk._publish_into(entry, lambda: choice_act_events(mode=mode, object_ref=object_ref, why=words, now=now,
                                                             session_id=session_id if session_id is not None else lk.turn_id()),
                            publish or lk._publish_bus)
    lines = [f"watch {mode} for {scope}" + (f": {words}" if words else ""),
             "The lower of your setting and Juniper's switch on the game page is in force. Hub reads this before "
             "its next look (at most one minute).", line]
    return lines, entry


def status(*, rows: Rows, now: Optional[datetime] = None) -> tuple[list[str], dict]:
    """The open sitting, the mode in force and who set it, the floor, the newest looks, the last refusal."""
    now = now or datetime.now(timezone.utc)
    payload: dict[str, Any] = {}
    lines: list[str] = []
    open_rows = rows(OPEN_SITTING_SQL)
    s = open_rows[0] if open_rows else None
    payload["sitting"] = s
    if s is None:
        lines.append("sitting: nobody is playing now")
    else:
        who = "Juniper" if s.get("player_stated") == "juniper" else "someone"
        lines.append(f"sitting: {who} plays {s.get('game_title') or s.get('game_id') or 'a game'} since "
                     f"{s.get('started_at')} (last press {s.get('last_input_at')}), id {s['sitting_id']}")

    def read(name: str, sql: str) -> Optional[list[dict]]:
        try:
            got = rows(sql)
            payload[name] = got
            return got
        except RuntimeError as exc:
            payload[name + "_error"] = str(exc)[:300]
            lines.append(f"{name}: unread ({str(exc)[:200]})")
            return None

    sid = str(s["sitting_id"]) if s else ""
    game = str(s.get("game_id") or "") if s else ""
    prefs = read("preferences", f"""
SELECT scope, scope_id, set_by, mode, why, set_at FROM retro_watch_preference
WHERE (set_by = 'orion' AND ((scope = 'sitting' AND scope_id = {_q(sid)}) OR scope = 'game'))
   OR (set_by = 'juniper' AND scope = 'all')
ORDER BY set_at""")
    if prefs is not None:
        mine = W.orion_choice([p for p in prefs if p.get("set_by") == "orion"], sid or None, game or None)
        jun = [p for p in prefs if p.get("set_by") == "juniper"]
        jun_last = jun[-1] if jun else None
        m = W.effective(mine.get("mode") if mine else None, jun_last.get("mode") if jun_last else None,
                        orion_why=mine.get("why") if mine else None)
        who = {"orion": "you", "juniper": "Juniper's switch", None: "nobody (both on)"}[m.set_by]
        lines.append(f"mode in force: {m.mode}, set by {who}" + (f": {m.why}" if m.why else ""))
        lines.append(f"  yours: {(mine or {}).get('mode', 'on (no choice)')}; Juniper's switch: "
                     f"{(jun_last or {}).get('mode', 'on (never set)')}")
        defaults = {}
        for p in prefs:
            if p.get("set_by") == "orion" and p.get("scope") == "game":
                defaults[p["scope_id"]] = p
        for gid, p in sorted(defaults.items()):
            lines.append(f"  your default for {gid}: {p['mode']}" + (f" ({p['why']})" if p.get("why") else ""))
    if game:
        floor = read("floor", f"""
SELECT floor, status, checks_input, checks_no_input, seconds, noinput_above, input_above, computed_at
FROM retro_watch_floor WHERE game_id = {_q(game)}""")
        if floor is not None:
            if not floor:
                lines.append(f"floor for {game}: none yet (no screen measured)")
            else:
                f = floor[0]
                lines.append(f"floor for {game}: {f.get('floor') or 'none'} ({f.get('status')}; "
                             f"{f.get('checks_input')} played / {f.get('checks_no_input')} still checks)")
    if sid:
        looks = read("looks", f"""
SELECT looked_at, outcome, verdict, failure, left(coalesce(note, ''), 300) AS note
FROM retro_watch_look WHERE sitting_id = {_q(sid)} ORDER BY looked_at DESC LIMIT 3""")
        counts = read("counts", f"SELECT {', '.join(W.COUNTERS)} FROM retro_sitting WHERE sitting_id = {_q(sid)}")
        if counts:
            c = counts[0]
            lines.append("counts: " + ", ".join(f"{k.replace('_', ' ')} {c.get(k)}" for k in W.COUNTERS))
        for lk_row in looks or []:
            lines.append(f"  look {lk_row['looked_at']} {lk_row['outcome']} ({lk_row['verdict']}): "
                         f"{lk_row.get('note') or lk_row.get('failure') or ''}")
    refusal = read("last_refusal", f"""
SELECT decided_at, outcome, reason, lane, net, binding_constraint FROM want_ledger
WHERE family = {_q(W.FAMILY)} AND outcome <> 'funded' ORDER BY decided_at DESC LIMIT 1""")
    if refusal:
        r = refusal[0]
        lines.append(f"last refused look bid: {r['decided_at']} {r['outcome']} ({r.get('reason')}; lane {r.get('lane')}, "
                     f"net {r.get('net')}, binds {r.get('binding_constraint')})")
    elif refusal is not None:
        lines.append("last refused look bid: none on record")
    lines.append('Change it: watch off|quiet|on --why "<your words>"  |  watch default <game> off --why "<words>"')
    return lines, payload


def parse(args: Sequence[str]) -> dict:
    """``watch`` arguments -> {action, mode, game, why}. Raises WatchError on anything else."""
    a = list(args)
    why = None
    if "--why" in a:
        i = a.index("--why")
        why = " ".join(a[i + 1:])
        a = a[:i]
    if not a or a == ["status"]:
        if why:
            raise WatchError("status takes no --why")
        return {"action": "status"}
    if len(a) == 1 and a[0] in W.MODES:
        return {"action": "set", "mode": a[0], "game": None, "why": why}
    if len(a) >= 3 and a[0] == "default" and a[-1] in W.MODES:
        return {"action": "set", "mode": a[-1], "game": " ".join(a[1:-1]), "why": why}
    raise WatchError('usage: watch [status] | watch off|quiet|on --why "<words>" | '
                     'watch default <game> off|quiet|on --why "<words>"')


__all__ = ["WatchError", "choice_act_events", "parse", "resolve_game", "set_choice", "status"]
