#!/usr/bin/env python3
"""Reverie redesign, Patch 0: read-only go/no-go numbers (design PR #3736).

Measures, from live data, whether each part of the "reveries that ask" design has material:

  quiet      stretches where arousal is idle AND the attention broadcast had no winner for >= N min
  absence    walkway rhythm expectations scored `missed` per day
  lost_bet   self-bet calibration gaps (template x probability bin) and single confident misses
  rehearse   can an hour-of-week model of Juniper's chat turns call "likely within 2 h" better than chance?
  body       commits per day in Orion's own sandbox that touch orion/ or services/
  self_def   self-definitions written per day (contradiction seed material)
  joint      per day, how many seed kinds had material (does rotation have a real choice?)
  far_memory similarity spread in the lived-memory index: is there a middle band below the nearest?

Reads only: Postgres (docker exec psql, SELECT only), Chroma HTTP (/get), git log. Writes nothing.
Stdlib only (the host has no numpy). Prints aggregate numbers; with --show-texts it also prints a few
memory excerpts for a by-eye check of the band (keep those out of tracked files).

    python3 scripts/analysis/reverie_patch0_measure.py [--days 7] [--calm-min 10] [--show-texts]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE  # noqa: E402  (pure constant, stdlib-only module)

PSQL = ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-AtF", "\t", "-c"]
CHROMA = "http://localhost:8500/api/v1/collections"
LIVED = "orion_lived_memory"
SANDBOX = "/mnt/orion-fcc/repo"


def sql(query: str) -> list[list[str]]:
    out = subprocess.run(PSQL + [query], check=True, capture_output=True, text=True).stdout
    return [line.split("\t") for line in out.splitlines() if line]


def ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace(" ", "T").replace("Z", "+00:00"))


# ---------------------------------------------------------------- quiet gate

def quiet(days: int, calm_min: float) -> dict:
    ticks = sql(
        "select generated_at, (jsonb_array_length(coalesce(projection_json->'attended_node_ids','[]'::jsonb)) > 0 "
        "or coalesce(projection_json->>'selected_open_loop_id','') <> '') "
        f"from substrate_attention_broadcast_log where generated_at > now() - interval '{days} days' "
        "order by generated_at"
    )
    trans = sql(
        "select occurred_at, label from temporal_self_event where source_kind = 'arousal_transition' "
        f"and occurred_at > now() - interval '{days + 2} days' order by occurred_at"
    )
    levels = [(ts(t), lvl) for t, lvl in trans]

    def level_at(t: datetime) -> str | None:
        cur = None
        for at, lvl in levels:
            if at > t:
                break
            cur = lvl
        return cur

    # A stretch = consecutive ticks with no winner AND arousal idle; a gap > 3 min between ticks breaks it.
    stretches: list[tuple[datetime, float]] = []
    start = prev = None
    for raw_t, won in ticks:
        t = ts(raw_t)
        ok = won == "f" and level_at(t) == "idle"
        if ok and start is not None and prev is not None and (t - prev) <= timedelta(minutes=3):
            prev = t
            continue
        if start is not None and prev is not None:
            stretches.append((start, (prev - start).total_seconds() / 60))
        start, prev = (t, t) if ok else (None, None)
    if start is not None and prev is not None:
        stretches.append((start, (prev - start).total_seconds() / 60))
    long = [s for s in stretches if s[1] >= calm_min]
    per_day = Counter(s[0].date().isoformat() for s in long)
    return {
        "ticks": len(ticks),
        "stretches_ge_min": len(long),
        "per_day": dict(sorted(per_day.items())),
        "median_len_min": round(sorted(s[1] for s in long)[len(long) // 2], 1) if long else 0,
        "hours_quiet_ge_min": round(sum(s[1] for s in long) / 60, 1),
    }


# ---------------------------------------------------------------- seeds

# A `missed` for a named individual means no sighting was ATTRIBUTED to that individual in the window;
# the individuals reducer (re-identification) decides attribution, not the raw detector. So split misses:
# clean = the raw census saw no person at all in the window; ambiguous = a person was detected but not
# attributed (they may have come and gone unrecognised). Orion's question 1 on PR #3736, checked 11-18.
_NAMED_PERSON_MISSES = (
    "select e.window_end::date, "
    "(select count(*) from vision_scene_inventory v where v.stream_id = e.stream_id "
    " and v.observed_at between e.window_start and e.window_end "
    " and coalesce((v.counts->>'person')::int, 0) > 0) "
    "from vision_percept_expectation e join vision_individual i on e.subject_key = 'individual:' || i.individual_id "
    "where e.status = 'missed' and i.kind = 'person' and i.label is not null "
    "and e.window_end > now() - interval '{days} days'"
)


def absence(days: int) -> dict:
    per_day: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for d, raw_person in sql(_NAMED_PERSON_MISSES.format(days=days)):
        per_day[d][0 if int(raw_person) == 0 else 1] += 1
    return {"named_person_misses_per_day(clean, ambiguous)": dict(sorted((k, tuple(v)) for k, v in per_day.items()))}


def lost_bet(days: int) -> dict:
    rows = sql(
        "select faculty, stated_p, (status = 'happened')::int from prediction_turn_self_bet "
        f"where status in ('happened','not_happened') and stated_p is not null "
        f"and settled_at > now() - interval '{days} days'"
    )
    bins: dict[tuple[str, int], list[int]] = defaultdict(list)
    ps: dict[tuple[str, int], list[float]] = defaultdict(list)
    confident_miss = 0
    for kind, p_raw, hit_raw in rows:
        p, hit = float(p_raw), int(hit_raw)
        b = min(9, int(p * 10))
        bins[(kind, b)].append(hit)
        ps[(kind, b)].append(p)
        if (p >= 0.8 and not hit) or (p <= 0.2 and hit):
            confident_miss += 1
    gaps = []
    for key, hits in bins.items():
        n = len(hits)
        if n < 20:
            continue
        rate, p = sum(hits) / n, sum(ps[key]) / n
        # Gap must beat 2 binomial standard errors, so a bin is not flagged for noise alone.
        se = math.sqrt(max(p * (1 - p), 1e-6) / n)
        if abs(rate - p) >= 0.15 and abs(rate - p) >= 2 * se:
            gaps.append((key[0], f"{key[1] / 10:.1f}-{(key[1] + 1) / 10:.1f}", n, round(p, 2), round(rate, 2)))
    return {"settled": len(rows), "confident_misses": confident_miss, "calibration_gaps": sorted(gaps)}


def rehearse() -> dict:
    """Train hour-of-week on 4 weeks of Juniper's own turns (chat_history_log + the canonical
    JUNIPER_TURN_PREDICATE, so outreach and Orion's internal turns never count), test each later hour: flag 'turn within 2 h' when the
    trained rate >= 0.5. Report precision vs the base rate, over the 2 most recent weeks (rolling)."""
    rows = sql(
        "select created_at at time zone 'UTC' from chat_history_log "
        f"where {JUNIPER_TURN_PREDICATE} and created_at > now() - interval '42 days' order by created_at"
    )
    turns = [ts(r[0]).replace(tzinfo=timezone.utc) for r in rows]
    if not turns:
        return {"turns": 0}
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    def any_turn(start: datetime, hours: int) -> bool:
        end = start + timedelta(hours=hours)
        return any(start <= t < end for t in turns)

    results = []
    for test_week in (1, 0):
        test_end = now - timedelta(weeks=test_week)
        test_start = test_end - timedelta(weeks=1)
        train_start = test_start - timedelta(weeks=4)
        how_counts: dict[int, list[int]] = defaultdict(list)
        h = train_start
        while h < test_start - timedelta(hours=2):
            how_counts[h.weekday() * 24 + h.hour].append(int(any_turn(h, 2)))
            h += timedelta(hours=1)
        flagged = flagged_hit = total = total_hit = 0
        h = test_start
        while h < test_end - timedelta(hours=2):
            got = any_turn(h, 2)
            total += 1
            total_hit += got
            hist = how_counts.get(h.weekday() * 24 + h.hour, [])
            if hist and sum(hist) / len(hist) >= 0.5:
                flagged += 1
                flagged_hit += got
            h += timedelta(hours=1)
        results.append({
            "test_week_ending": test_end.date().isoformat(),
            "base_rate": round(total_hit / total, 2) if total else None,
            "flagged_hours": flagged,
            "precision": round(flagged_hit / flagged, 2) if flagged else None,
            "recall": round(flagged_hit / total_hit, 2) if total_hit else None,
        })
    return {"juniper_turns_42d": len(turns), "weeks": results}


def body(days: int) -> dict:
    out = subprocess.run(
        ["git", "-C", SANDBOX, "log", "--no-merges", f"--since={days} days ago", "--format=%ad",
         "--date=short", "--", "orion", "services"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return {"commits_per_day(author date)": dict(sorted(Counter(out).items()))}


def joint(days: int) -> dict:
    """Per day: how many seed kinds had material, so rotation has a real choice (Orion's question 5).

    A kind counts on day D when: absence -- a named person miss (clean or ambiguous) ended on D;
    lost_bet -- a calibration gap in the 7 days ending D; far_memory -- always (597-doc index, anchor = now);
    rehearse -- a Juniper turn in the 72 h before D ends; body -- a sandbox commit on D touching orion/ or
    services/; contradiction -- a self-definition in the 7 days ending D. Upper bound: the 14-day
    near-duplicate refusal will spend a standing lost-bet gap or contradiction after its first question.
    """
    today = datetime.now(timezone.utc).date()
    day_list = [(today - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    has: dict[str, set[str]] = {d: {"far_memory"} for d in day_list}
    for d, _ in sql(_NAMED_PERSON_MISSES.format(days=days)):
        if d in has:
            has[d].add("absence")
    for d in day_list:
        gaps = lost_bet_gaps_ending(d)
        if gaps:
            has[d].add("lost_bet")
    juniper = [ts(r[0]).replace(tzinfo=timezone.utc) for r in sql(
        "select created_at at time zone 'UTC' from chat_history_log "
        f"where {JUNIPER_TURN_PREDICATE} and created_at > now() - interval '{days + 3} days'")]
    for d in day_list:
        end = datetime.fromisoformat(d).replace(tzinfo=timezone.utc) + timedelta(days=1)
        if any(end - timedelta(hours=72) <= t < end for t in juniper):
            has[d].add("rehearse")
    commits = set(subprocess.run(
        ["git", "-C", SANDBOX, "log", "--no-merges", f"--since={days} days ago", "--format=%ad", "--date=short",
         "--", "orion", "services"], check=True, capture_output=True, text=True).stdout.split())
    defs = [r[0] for r in sql(
        "select created_at::date from self_concept_history where concept_id = 'self:definition' "
        f"and created_at > now() - interval '{days + 7} days'")]
    for d in day_list:
        if d in commits:
            has[d].add("body")
        lo = (datetime.fromisoformat(d) - timedelta(days=6)).date().isoformat()
        if any(lo <= x <= d for x in defs):
            has[d].add("contradiction")
    counts = {d: len(k) for d, k in has.items()}
    return {
        "kinds_per_day": counts,
        "days_with_ge3_kinds": sum(1 for n in counts.values() if n >= 3),
        "days": len(counts),
        "missing_by_kind": dict(Counter(k for d in day_list for k in
                                        ("absence", "lost_bet", "rehearse", "body", "contradiction")
                                        if k not in has[d])),
    }


def lost_bet_gaps_ending(day: str) -> int:
    rows = sql(
        "select faculty, width_bucket(stated_p, 0, 1, 10), count(*), avg(stated_p), avg((status = 'happened')::int) "
        "from prediction_turn_self_bet where status in ('happened','not_happened') and stated_p is not null "
        f"and settled_at >= date '{day}' - 6 and settled_at < date '{day}' + 1 group by 1, 2 having count(*) >= 20"
    )
    n = 0
    for _, _, cnt, p_raw, rate_raw in rows:
        p, rate, k = float(p_raw), float(rate_raw), int(cnt)
        if abs(rate - p) >= 0.15 and abs(rate - p) >= 2 * math.sqrt(max(p * (1 - p), 1e-6) / k):
            n += 1
    return n


def self_def(days: int) -> dict:
    rows = sql(
        "select created_at::date, count(*) from self_concept_history where concept_id = 'self:definition' "
        f"and created_at > now() - interval '{days} days' group by 1 order by 1"
    )
    return {"self_definitions_per_day": {d: int(n) for d, n in rows}}


# ---------------------------------------------------------------- far memory band

def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def far_memory(anchors: int, show_texts: bool) -> dict:
    with urllib.request.urlopen(f"{CHROMA}/{LIVED}", timeout=30) as resp:
        coll_id = json.loads(resp.read())["id"]
    ids, vecs, docs = [], [], []
    offset = 0
    while True:
        page = _post(f"{CHROMA}/{coll_id}/get",
                     {"limit": 200, "offset": offset, "include": ["embeddings", "documents"]})
        ids += page["ids"]
        vecs += page["embeddings"]
        docs += page["documents"]
        if len(page["ids"]) < 200:
            break
        offset += 200
    raw = [_unit(v) for v in vecs]
    dim = len(raw[0])
    mean = [sum(v[i] for v in raw) / len(raw) for i in range(dim)]
    centered = [_unit([v[i] - mean[i] for i in range(dim)]) for v in raw]
    rng = random.Random(3736)
    picks = rng.sample(range(len(raw)), min(anchors, len(raw)))
    ranks = (1, 3, 5, 10, 20, 50, 100)
    out: dict = {"docs": len(raw)}
    for name, space in (("raw_cosine", raw), ("centered_cosine", centered)):
        per_rank: dict[int, list[float]] = defaultdict(list)
        medians = []
        for a in picks:
            sims = sorted((sum(x * y for x, y in zip(space[a], space[j])), j) for j in range(len(space)) if j != a)
            sims.reverse()
            for r in ranks:
                if r <= len(sims):
                    per_rank[r].append(sims[r - 1][0])
            medians.append(sims[len(sims) // 2][0])
            if show_texts and name == "centered_cosine" and a in picks[:3]:
                print(f"\n[anchor] {docs[a][:160]}")
                for r in (1, 8, 25):
                    s, j = sims[r - 1]
                    print(f"  rank {r:>2} sim {s:.2f}: {docs[j][:160]}")
        out[name] = {f"rank{r}": round(sum(v) / len(v), 3) for r, v in per_rank.items()}
        out[name]["median_doc"] = round(sum(medians) / len(medians), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--calm-min", type=float, default=10.0)
    ap.add_argument("--anchors", type=int, default=20)
    ap.add_argument("--show-texts", action="store_true")
    args = ap.parse_args()
    report = {
        "quiet": quiet(args.days, args.calm_min),
        "absence": absence(args.days),
        "lost_bet": lost_bet(args.days),
        "rehearse": rehearse(),
        "body": body(14),
        "self_def": self_def(30),
        "joint": joint(14),
        "far_memory": far_memory(args.anchors, args.show_texts),
    }
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
