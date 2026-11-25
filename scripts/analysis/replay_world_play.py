#!/usr/bin/env python3
"""Replay the stored attention broadcast ticks with ``world:play`` added. Read-only analysis.

Design: docs/superpowers/specs/2026-11-16-game-company-watch-and-commentary-design.md, section 7 ("What
reading loses") and "Acceptance checks: Patch 4": the replay runs first, and the eligibility rule of
``world:play`` is chosen from its numbers, not assumed (Orion's point 10 on PR #3632: about 89% of
half-hour windows read zero play, so "top tenth of its own week" passes for almost any open sitting).

For every stored world-first broadcast tick (``substrate_attention_broadcast_log``, the contest whose winner
becomes the open loop that curiosity and the act ledger read), the production ``play_candidate``
(orion/attention/world_first.py) is built AS OF that tick from the sittings, look notes and watch choices
that existed then, judged with the real ``judge_candidate``, yields to Juniper's chat with the real rule, and
the winner is re-picked with the key ``rank_candidates`` uses. With no play candidate the script must
reproduce the stored winner; it reports how often it does.

Rules compared (each replayed alone):

  mode     -- the brief's rule: the existing top-tenth band rule, while a sitting is open and the watch in
              force is not off (``PLAY_RULE_MODE``).
  note     -- the spec's "watching": ``mode`` and a look note stored in the last 3 minutes
              (``PLAY_RULE_NOTE``).
  unusual  -- ``mode`` with the stricter "unusual" band only (top 1% of its own week): the obvious way to
              make "top tenth" filter harder. Replay-only; not in production unless chosen.
  hunger:<tau> -- the 2026-11-25 rule (production since then): while watching (rule ``note``), play is
              eligible and ranks by its pull, ``1 - exp(-t / tau)``, t = seconds since play last won in this
              sitting (or since the sitting began). A win resets t. One run per tau (``--taus``).

Per rule it also reports wins per sitting by sitting length (short < 15 min, medium 15 min - 2 h, long >= 2 h)
and how many watched sittings won at least once, plus each sitting of the last day (``--sittings-since``).

What it reports per rule: ticks play was eligible, ticks it won, ticks ``world:reading`` would have lost to
it, ticks a CURIOSITY-SEEDING winner would have lost (a winner whose loop carries a ``node:`` ref:
``world:reading`` and every body/camera node; orion/substrate/endogenous_curiosity.py
``_attention_loop_candidates`` skips a loop with none, so a ``world:play`` loop seeds no curiosity), and the
share of ticks during sittings won by reading and by curiosity-seeding winners, before and after, for all
sittings and for sittings of 2 hours or more. A sitting is "open" at a tick from its first input to 5 minutes
after its last (the live close rule); backfilled sittings count as if they had been live (the question is
what would have happened), but they have no look notes.

Inputs (SELECT-only exports, one JSON object per line):

    P='docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc'
    $P "select json_build_object('t',generated_at,'wf',projection_json->'frame'->'debug'->'world_first') \\
        from substrate_attention_broadcast_log where generated_at >= now()-interval '7 days' \\
        and projection_json->'frame'->'debug' ? 'world_first' order by generated_at" > frames.jsonl
    $P "select json_build_object('sitting_id',sitting_id,'game_id',game_id,'game_title',game_title, \\
        'started_at',started_at,'last_input_at',last_input_at,'ended_at',ended_at,'backfilled',backfilled, \\
        'player_stated',player_stated) from retro_sitting order by started_at" > sittings.jsonl
    $P "select json_build_object('sitting_id',sitting_id,'looked_at',looked_at,'outcome',outcome,'note',note) \\
        from retro_watch_look order by looked_at" > looks.jsonl
    $P "select json_build_object('scope',scope,'scope_id',scope_id,'set_by',set_by,'mode',mode,'why',why, \\
        'set_at',set_at) from retro_watch_preference" > prefs.jsonl
    # optional, the field contest during sittings (3 days kept):
    $P "select json_build_object('t',f.generated_at,'dom',(select json_agg(json_build_object('id', \\
        t->>'target_id','s',(t->>'salience_score')::float)) from jsonb_array_elements(f.frame_json-> \\
        'dominant_targets') t)) from substrate_attention_frames f where exists (select 1 from retro_sitting s \\
        where f.generated_at between s.started_at and s.last_input_at + interval '5 minutes') \\
        order by f.generated_at" > field_frames.jsonl
    python scripts/analysis/replay_world_play.py frames.jsonl sittings.jsonl looks.jsonl prefs.jsonl \\
        --rules note --taus 30,60,90,120,180,300,600 --field-frames field_frames.jsonl

Needs pydantic (the repo venv). No writes anywhere.
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.attention.world_first import (  # noqa: E402
    PLAY_HUNGER_TAU_SEC,
    PLAY_RULE_MODE,
    PLAY_RULE_NOTE,
    PlaySitting,
    judge_candidate,
    play_candidate,
    play_watch_from_rows,
    open_play_sitting,
)
from orion.retro.sitting import GAP_SEC_DEFAULT  # noqa: E402

RULES = ("mode", "note", "unusual")
LONG_SITTING = timedelta(hours=2)
SHORT_SITTING = timedelta(minutes=15)


def length_class(r: dict) -> str:
    d = r["last_input_at"] - r["started_at"]
    return "short" if d < SHORT_SITTING else ("long" if d >= LONG_SITTING else "medium")
GAP = timedelta(seconds=GAP_SEC_DEFAULT)


def _ts(s):
    if s is None:
        return None
    return datetime.fromisoformat(str(s))


def _jsonl(path: str) -> list[dict]:
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_frames(path: str) -> list[tuple[datetime, dict]]:
    out = []
    for row in _jsonl(path):
        wf = row.get("wf")
        if isinstance(wf, dict) and wf.get("candidates") is not None:
            out.append((_ts(row["t"]), wf))
    out.sort(key=lambda r: r[0])
    return out


def load_sittings(path: str) -> list[dict]:
    rows = _jsonl(path)
    for r in rows:
        for k in ("started_at", "last_input_at", "ended_at"):
            r[k] = _ts(r.get(k))
    rows.sort(key=lambda r: r["started_at"])
    return rows


def as_of(rows: list[dict], t: datetime) -> list[PlaySitting]:
    """The sittings as they stood at ``t``: started by then, input capped at ``t``, open until 5 min after
    their last input (or their recorded end). Every row is treated as live (``backfilled`` False)."""
    out = []
    for r in rows:
        if r["started_at"] > t:
            continue
        last = min(r["last_input_at"], t)
        end = r["ended_at"] or (r["last_input_at"] + GAP)
        out.append(PlaySitting(
            sitting_id=r["sitting_id"], started_at=r["started_at"], last_input_at=last,
            ended_at=None if t < end and t - last <= GAP else end,
            game_title=r.get("game_title"), game_id=r.get("game_id"), player_stated=r.get("player_stated"),
            backfilled=False))
    return out


def key(c: dict) -> float:
    return c["rank_score"] if c.get("rank_score") is not None else (c.get("score") or 0.0)


def pick(cands: list[dict]) -> str | None:
    elig = [c for c in cands if c.get("eligible")]
    if not elig:
        return None
    return sorted(elig, key=lambda c: (-key(c), c["source_id"]))[0]["source_id"]


def seeds_curiosity(source_id: str | None) -> bool:
    """A loop that endogenous curiosity can seed from: it carries a ``node:`` ref (every node winner) or
    concept endpoints (world:reading)."""
    return bool(source_id) and (source_id.startswith("node:") or source_id == "world:reading")


def replay(frames, sittings, looks, prefs, rule: str, recent_since: datetime | None = None) -> dict:
    tau = float(rule.split(":", 1)[1]) if rule.startswith("hunger:") else None
    last_won: dict[str, datetime] = {}
    per_sitting: dict[str, Counter] = {}
    look_times: dict[str, list[tuple[datetime, str]]] = {}
    for lk in looks:
        if lk.get("outcome") == "stored":
            look_times.setdefault(lk["sitting_id"], []).append((_ts(lk["looked_at"]), str(lk.get("note") or "")))
    for v in look_times.values():
        v.sort()
    prefs = [{**p, "set_at": _ts(p["set_at"])} for p in prefs]
    first_look = min((v[0][0] for v in look_times.values()), default=None)
    long_ids = {r["sitting_id"] for r in sittings
                if r["last_input_at"] - r["started_at"] >= LONG_SITTING}
    backfilled_ids = {r["sitting_id"] for r in sittings if r.get("backfilled")}

    n = stored_match = 0
    open_ticks = open_ticks_since_looks = open_ticks_backfilled = 0
    band_when_open: Counter = Counter()
    eligible = won = yielded = 0
    lost_reading = lost_seed = 0
    displaced: Counter = Counter()
    during = {k: Counter() for k in ("all", "long")}
    since_looks = Counter()
    for t, wf in frames:
        n += 1
        # The stored play row (live since 2026-11-24) is replaced by the replayed one, never doubled.
        cands = [c for c in wf["candidates"] if c.get("source_id") != "world:play"]
        base = pick(cands)
        stored_match += base == wf.get("winner")
        rows = as_of(sittings, t)
        sitting = open_play_sitting(rows, now=t)
        watch = None
        if sitting is not None:
            notes = look_times.get(sitting.sitting_id, [])
            i = bisect.bisect_right(notes, (t, "￿"))
            newest = notes[i - 1] if i else None
            watch = play_watch_from_rows(sitting, [p for p in prefs if p["set_at"] <= t], newest)
        if tau is not None and sitting is None:
            cand = None  # no open sitting: never eligible under any rule (skip the slow 7-day grid)
        else:
            cand = play_candidate(rows, now=t, watch=watch,
                                  rule=PLAY_RULE_NOTE if rule in ("note",) or tau is not None else PLAY_RULE_MODE,
                                  hunger=tau is not None,
                                  last_won_at=last_won.get(sitting.sitting_id) if sitting is not None else None,
                                  hunger_tau_sec=tau if tau is not None else PLAY_HUNGER_TAU_SEC)
        v = judge_candidate(cand) if cand is not None else None
        if v is None:
            from types import SimpleNamespace
            v = SimpleNamespace(eligible=False, score=None, rank_score=None, reason="no sitting open", band="quiet")
        if sitting is not None:
            open_ticks += 1
            open_ticks_backfilled += sitting.sitting_id in backfilled_ids
            band_when_open[v.band] += 1
            if first_look is not None and t >= first_look:
                open_ticks_since_looks += 1
        vd = {"source_id": "world:play", "eligible": v.eligible, "score": v.score, "rank_score": v.rank_score,
              "reason": v.reason}
        if rule == "unusual" and vd["eligible"] and v.band != "unusual":
            vd = {**vd, "eligible": False, "reason": "replay rule: unusual band only"}
        if vd["eligible"] and any(c.get("source_id") == "world:chat" and c.get("eligible") for c in cands):
            vd = {**vd, "eligible": False, "reason": "yields to Juniper (world:chat eligible)"}
            yielded += 1
        eligible += bool(vd["eligible"])
        new = pick(cands + [vd])
        if sitting is not None:
            ps = per_sitting.setdefault(sitting.sitting_id, Counter())
            ps["open_ticks"] += 1
            # Watched under this rule's gate (an open sitting always has >= 1 play minute in the window).
            ps["watched_ticks"] += bool(cand is not None and cand.not_eligible_reason is None)
            ps["wins"] += new == "world:play"
            ps["yielded"] += vd.get("reason") == "yields to Juniper (world:chat eligible)"
        if new == "world:play" and sitting is not None:
            last_won[sitting.sitting_id] = t
        if new == "world:play":
            won += 1
            displaced[base or "(no winner)"] += 1
            lost_reading += base == "world:reading"
            lost_seed += seeds_curiosity(base)
        if sitting is not None:
            buckets = ["all"] + (["long"] if sitting.sitting_id in long_ids else [])
            for b in buckets:
                c = during[b]
                c["ticks"] += 1
                c["reading_before"] += base == "world:reading"
                c["reading_after"] += new == "world:reading"
                c["seed_before"] += seeds_curiosity(base)
                c["seed_after"] += seeds_curiosity(new)
                c["play_after"] += new == "world:play"
            if first_look is not None and t >= first_look:
                since_looks["ticks"] += 1
                since_looks["play_after"] += new == "world:play"
                since_looks["reading_lost"] += base == "world:reading" and new == "world:play"

    def share(c: Counter, k: str):
        return round(c[k] / c["ticks"], 3) if c["ticks"] else None

    return {
        "rule": rule,
        "ticks": n,
        "span": [frames[0][0].isoformat(timespec="minutes"), frames[-1][0].isoformat(timespec="minutes")],
        "stored_winner_reproduced": stored_match,
        "ticks_with_open_sitting": open_ticks,
        # Of those, ticks whose open sitting is a backfilled row (counted as if live; it has no look notes).
        "open_ticks_backfilled": open_ticks_backfilled,
        "play_band_when_open": dict(band_when_open),
        "play_eligible_ticks": eligible,
        "yielded_to_chat": yielded,
        "play_won": won,
        "reading_lost_to_play": lost_reading,
        "curiosity_seed_lost_to_play": lost_seed,
        "displaced": dict(displaced.most_common()),
        "during_sittings": {b: {"ticks": c["ticks"],
                                "reading_share_before": share(c, "reading_before"),
                                "reading_share_after": share(c, "reading_after"),
                                "curiosity_seed_share_before": share(c, "seed_before"),
                                "curiosity_seed_share_after": share(c, "seed_after"),
                                "play_share_after": share(c, "play_after")} for b, c in during.items()},
        "looks_began": first_look.isoformat(timespec="minutes") if first_look else None,
        "since_looks_began": {"open_ticks": since_looks["ticks"], "play_won": since_looks["play_after"],
                              "reading_lost": since_looks["reading_lost"]},
        "by_length": _by_length(per_sitting, sittings),
        "recent_sittings": {
            sid: {**dict(c), "minutes": round((r["last_input_at"] - r["started_at"]).total_seconds() / 60, 1)}
            for r in sittings if recent_since is not None and r["started_at"] >= recent_since
            for sid, c in [(r["sitting_id"], per_sitting.get(r["sitting_id"], Counter()))]
        },
        "long_sittings_in_span": sorted(
            r["sitting_id"] for r in sittings if r["sitting_id"] in long_ids
            and r["last_input_at"] >= frames[0][0] and r["started_at"] <= frames[-1][0]),
    }


def replay_field(field_frames: list[dict], sittings, looks, prefs, tau: float) -> dict:
    """APPROXIMATE replay of the field contest (orion-attention-runtime, ~2 s ticks) under hunger.

    ``substrate_attention_frames`` keeps 3 days and stores each attended target's ``salience_score`` (the
    strict-below percentile), not its ranking key (the mid-rank), so rivals are compared on salience: a
    slightly LOW rival bar, which favours play. Watching uses the production rule (``play_watch_reason``,
    rule ``note``) on the sittings and look notes as of each frame; the pull is the production
    ``play_hunger``. Play yields when ``world:chat`` was attended. Reports the share of watched frames play
    wins, and how many of those it took from a curiosity-seeding winner (a ``node:`` target or reading)."""
    from orion.attention.world_first import play_hunger, play_watch_reason

    look_times: dict[str, list[tuple[datetime, str]]] = {}
    for lk in looks:
        if lk.get("outcome") == "stored":
            look_times.setdefault(lk["sitting_id"], []).append((_ts(lk["looked_at"]), str(lk.get("note") or "")))
    for v in look_times.values():
        v.sort()
    prefs = [{**p, "set_at": _ts(p["set_at"])} for p in prefs]
    last_won: dict[str, datetime] = {}
    c: Counter = Counter()
    for row in field_frames:
        t = _ts(row["t"])
        rows = as_of(sittings, t)
        sitting = open_play_sitting(rows, now=t)
        if sitting is None:
            continue
        notes = look_times.get(sitting.sitting_id, [])
        i = bisect.bisect_right(notes, (t, "\uffff"))
        watch = play_watch_from_rows(sitting, [p for p in prefs if p["set_at"] <= t], notes[i - 1] if i else None)
        if play_watch_reason(sitting, watch, now=t) is not None:
            continue
        dom = [d for d in (row.get("dom") or []) if d.get("id") != "world:play"]
        base = dom[0]["id"] if dom else None
        c["watched_frames"] += 1
        if any(d.get("id") == "world:chat" for d in dom):
            c["yielded_to_chat"] += 1
            continue
        pull, _ = play_hunger(sitting, last_won.get(sitting.sitting_id), now=t, tau_sec=tau)
        best = max((float(d.get("s") or 0.0) for d in dom), default=None)
        if best is None or pull > best:
            c["play_won"] += 1
            c["won_from_no_winner"] += base is None
            c["won_from_rival"] += base is not None
            c["reading_lost"] += base == "world:reading"
            c["seed_lost"] += seeds_curiosity(base)
            last_won[sitting.sitting_id] = t
        c["seed_before"] += seeds_curiosity(base)
    w = c["watched_frames"]
    return {"tau": tau, **dict(c),
            "play_share_of_watched": round(c["play_won"] / w, 3) if w else None,
            "seed_share_before": round(c["seed_before"] / w, 3) if w else None,
            "seed_share_after": round((c["seed_before"] - c["seed_lost"]) / w, 3) if w else None}


def _by_length(per_sitting: dict[str, Counter], sittings: list[dict]) -> dict:
    """Per length class: sittings with an open tick, sittings watched at least one tick, watched sittings
    that won at least once, wins, and wins per watched sitting."""
    rows = {r["sitting_id"]: r for r in sittings}
    out: dict[str, Counter] = {k: Counter() for k in ("short", "medium", "long")}
    for sid, c in per_sitting.items():
        k = length_class(rows[sid])
        out[k]["sittings"] += 1
        if c["watched_ticks"]:
            out[k]["watched_sittings"] += 1
            out[k]["watched_won_once"] += c["wins"] > 0
            out[k]["watched_ticks"] += c["watched_ticks"]
        out[k]["wins"] += c["wins"]
    res = {}
    for k, c in out.items():
        res[k] = {**dict(c), "wins_per_watched_sitting": round(c["wins"] / c["watched_sittings"], 2)
                  if c["watched_sittings"] else None}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("sittings")
    ap.add_argument("looks")
    ap.add_argument("prefs")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rules", default=",".join(RULES), help="comma list of mode,note,unusual")
    ap.add_argument("--taus", default=str(int(PLAY_HUNGER_TAU_SEC)),
                    help="comma list of hunger time constants (s); empty = no hunger runs")
    ap.add_argument("--sittings-since", default=None, help="ISO time: list each sitting started after it")
    ap.add_argument("--field-frames", default=None,
                    help="optional field-contest export (one {t, dom:[{id, s}]} per line): approximate hunger replay")
    args = ap.parse_args()
    frames = load_frames(args.frames)
    if not frames:
        print("no frames", file=sys.stderr)
        return 1
    sittings = load_sittings(args.sittings)
    looks = _jsonl(args.looks)
    prefs = _jsonl(args.prefs)
    rules = [r for r in args.rules.split(",") if r]
    rules += [f"hunger:{float(x):g}" for x in args.taus.split(",") if x.strip()]
    since = _ts(args.sittings_since) if args.sittings_since else frames[-1][0] - timedelta(days=1)
    results = [replay(frames, sittings, looks, prefs, r, since) for r in rules]
    if args.field_frames:
        ff = _jsonl(args.field_frames)
        results += [{"rule": f"field_hunger:{float(x):g}", **replay_field(ff, sittings, looks, prefs, float(x))}
                    for x in args.taus.split(",") if x.strip()]
    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return 0
    print(f"ticks {frames[0][0]} .. {frames[-1][0]}  n={len(frames)}  sittings={len(sittings)} "
          f"(backfilled {sum(1 for s in sittings if s.get('backfilled'))})  stored looks="
          f"{sum(1 for x in looks if x.get('outcome') == 'stored')}")
    for r in results:
        print(f"\n== rule {r['rule']}")
        for k, val in r.items():
            if k not in ("rule",):
                print(f"  {k}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
