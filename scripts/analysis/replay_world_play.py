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
    python scripts/analysis/replay_world_play.py frames.jsonl sittings.jsonl looks.jsonl prefs.jsonl

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


def replay(frames, sittings, looks, prefs, rule: str) -> dict:
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

    n = stored_match = 0
    open_ticks = open_ticks_since_looks = 0
    band_when_open: Counter = Counter()
    eligible = won = yielded = 0
    lost_reading = lost_seed = 0
    displaced: Counter = Counter()
    during = {k: Counter() for k in ("all", "long")}
    since_looks = Counter()
    for t, wf in frames:
        n += 1
        cands = list(wf["candidates"])
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
        cand = play_candidate(rows, now=t, watch=watch,
                              rule=PLAY_RULE_NOTE if rule == "note" else PLAY_RULE_MODE)
        v = judge_candidate(cand)
        if sitting is not None:
            open_ticks += 1
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
        "long_sittings_in_span": sorted(
            r["sitting_id"] for r in sittings if r["sitting_id"] in long_ids
            and r["last_input_at"] >= frames[0][0] and r["started_at"] <= frames[-1][0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("sittings")
    ap.add_argument("looks")
    ap.add_argument("prefs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    frames = load_frames(args.frames)
    if not frames:
        print("no frames", file=sys.stderr)
        return 1
    sittings = load_sittings(args.sittings)
    looks = _jsonl(args.looks)
    prefs = _jsonl(args.prefs)
    results = [replay(frames, sittings, looks, prefs, r) for r in RULES]
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
