#!/usr/bin/env python3
"""Replay eval: the walkway body-match line, old rule vs the deployed rule.

docs/superpowers/specs/2026-10-16-vision-body-match-proposal.md, "Replay
results", made permanent. READ ONLY: one READ ONLY transaction on the live DB
(it cannot write), nothing copied anywhere, no pictures printed.

Replays every walkway person crop with a body vector through the deployed
clusterer (``vision_individuals.apply_batch``) in 60 s landing-time batches,
from empty, the way ``_one_stream`` feeds it (sightings overlapping the batch
+- the merge gap), once per rule:

- ``baseline``: one body line at 0.80, no continuation lane (the rule before
  2026-10-16; also what the rollback switches restore).
- ``deployed``: the rule in ``IndividualsConfig()`` (continuation 0.88 within
  10 s / 1.5 box heights, re-identification 0.90).

Scores (no person labels exist, so these use the camera's own geometry):

- **track pair**: a crop and the crop in the next frame (<= 3 s later) whose
  box moved <= 1 box height and kept its size (0.75-1.33), mutual nearest,
  no second candidate within 2x. Proxy for "same person".
- **tracklet**: a chain of track pairs. Two tracklets in one frame are two
  different people.
- **mixed** (lower is better): share of those different-people tracklet
  pairs that share an individual anywhere.
- **split** (lower is better): share of track pairs that end in different
  individuals.
- **re-seen**: share of crops in an individual seen on two visits > 10 min
  apart (a ceiling, not a goal: blobs score high).
- **individuals**, individuals with >= 5 crops, the three largest.

Pinned window: every crop that had LANDED by 2026-10-16 03:11 UTC (the
proposal's read-only export: 4,988 crops, 2026-10-15 06:21 -> 2026-10-16
01:53 landing). The proposal's numbers on it:

    baseline  40 individuals, top 1166/758/738, mixed 13.6%, split 17.6%
    deployed  493 individuals,                  mixed  6.3%, split 15.6%

Exit 0 = reproduced (individual counts exact for the baseline, +-5% for the
deployed rule; mixed/split +-0.5 points), 1 = a number moved, 2 = cannot
replay (crops pruned after VISION_CROP_RETENTION_DAYS=7, i.e. from about
2026-10-22). ``--until``/``--since`` replay another window (e.g. the 24 h
after deploy) and print the table without asserting.

    python evals/run_vision_body_match_replay_eval.py --live postgresql://postgres:...@localhost:55432/conjourney
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402

PINNED_UNTIL = datetime(2026, 10, 16, 3, 11, tzinfo=timezone.utc)
PINNED_CROPS = 4988
EXPECTED = {
    "baseline": {"individuals": 40, "top": [1166, 758, 738], "mixed": 0.136, "split": 0.176},
    "deployed": {"individuals": 493, "mixed": 0.063, "split": 0.156},
}
RATE_TOL = 0.005
COUNT_TOL = 0.05

BASELINE = vi.IndividualsConfig(match_threshold=0.80, continuation_enabled=False)


def rules() -> Dict[str, vi.IndividualsConfig]:
    return {"baseline": BASELINE, "deployed": vi.IndividualsConfig()}


# --- replay -----------------------------------------------------------------------------------


def replay(crops: Sequence[vi.CropRow], landed: Sequence[datetime], cfg: vi.IndividualsConfig,
           tz: ZoneInfo, cannot_link: Optional[Dict[str, set]] = None) -> Tuple[Dict[str, str], Dict[str, vi.Individual]]:
    """crop_id -> individual id, final individuals. ``apply_batch`` per 60 s
    landing bucket, fed the sightings that overlap the batch +- the merge gap
    (what ``_one_stream`` loads)."""
    batches: Dict[int, List[vi.CropRow]] = defaultdict(list)
    for c, t in zip(crops, landed):
        batches[int(t.timestamp() // 60)].append(c)
    inds: Dict[str, vi.Individual] = {}
    sightings: Dict[str, vi.Sighting] = {}
    member: Dict[str, str] = {}
    by_crop = {c.crop_id: c for c in crops}
    gap = timedelta(seconds=cfg.merge_gap_sec)
    cgap = timedelta(seconds=min(cfg.continuation_gap_sec, cfg.merge_gap_sec))
    counter = itertools.count()
    for key in sorted(batches):
        batch = batches[key]
        lo = min(c.observed_at for c in batch) - gap
        hi = max(c.observed_at for c in batch) + gap
        nearby: Dict[str, List[vi.Sighting]] = defaultdict(list)
        for s in sightings.values():
            if s.ended_at >= lo and s.started_at <= hi:
                nearby[s.individual_id].append(s)
        # Stored membership around the batch, as _one_stream loads it.
        olo = min(c.observed_at for c in batch) - cgap
        ohi = max(c.observed_at for c in batch) + cgap
        prior = [(iid, by_crop[cid].observed_at, by_crop[cid].box_xyxy) for cid, iid in member.items()
                 if olo <= by_crop[cid].observed_at <= ohi] if cfg.continuation_enabled else []
        res = vi.apply_batch(
            batch, individuals=inds, latest_sightings={}, nearby_sightings=dict(nearby),
            no_embed_zones=(), match_threshold=cfg.match_threshold, merge_gap_sec=cfg.merge_gap_sec, tz=tz,
            new_id=lambda: f"r{next(counter):06d}",
            face_match_threshold=cfg.face_match_threshold if cfg.face_match_threshold > 0 else None,
            stationary_min_sec=cfg.stationary_min_sec, stationary_iou=cfg.stationary_iou,
            stationary_max_gap_sec=cfg.stationary_max_gap_sec,
            cannot_link=cannot_link, cannot_link_margin=cfg.cannot_link_margin,
            continuation_threshold=cfg.continuation_threshold if cfg.continuation_enabled else None,
            continuation_gap_sec=min(cfg.continuation_gap_sec, cfg.merge_gap_sec),
            continuation_max_disp=cfg.continuation_max_disp,
            prior_observations=prior,
        )
        inds = res.individuals
        for sid, s in res.sightings.items():
            sightings[sid] = replace(s, zone_counts=dict(s.zone_counts))
        member.update(res.crop_individual)
    return member, inds


# --- scores -----------------------------------------------------------------------------------


def _geom(c: vi.CropRow):
    x1, y1, x2, y2 = (float(v) for v in c.box_xyxy[:4])
    return (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1


def track_pairs(crops: Sequence[vi.CropRow], max_dt: float = 3.0, max_disp: float = 1.0) -> List[Tuple[int, int]]:
    frames: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(crops):
        frames[c.observation_id].append(i)
    order = sorted(frames, key=lambda o: crops[frames[o][0]].observed_at)
    g = [_geom(c) for c in crops]

    def d(i: int, j: int) -> float:
        (ax, ay, ah), (bx, by, bh) = g[i], g[j]
        if ah <= 0 or not (0.75 <= bh / ah <= 1.33):
            return 1e9
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / ((ah + bh) / 2)

    pairs = []
    for a, b in zip(order, order[1:]):
        A, B = frames[a], frames[b]
        if (crops[B[0]].observed_at - crops[A[0]].observed_at).total_seconds() > max_dt:
            continue
        M = np.array([[d(i, j) for j in B] for i in A])
        for ii, i in enumerate(A):
            jj = int(M[ii].argmin())
            v = M[ii, jj]
            if v > max_disp or int(M[:, jj].argmin()) != ii:
                continue
            row, col = np.sort(M[ii]), np.sort(M[:, jj])
            if (len(row) > 1 and row[1] < 2 * v) or (len(col) > 1 and col[1] < 2 * v):
                continue
            pairs.append((i, B[jj]))
    return pairs


def tracklets(n: int, pairs: Sequence[Tuple[int, int]]) -> List[int]:
    parent = list(range(n))

    def f(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[f(i)] = f(j)
    return [f(i) for i in range(n)]


def conflicts(crops: Sequence[vi.CropRow], tl: Sequence[int]) -> List[Tuple[int, int]]:
    fr: Dict[str, set] = defaultdict(set)
    for i, c in enumerate(crops):
        fr[c.observation_id].add(tl[i])
    return sorted({tuple(sorted(p)) for s in fr.values() for p in itertools.combinations(s, 2)})


def score(crops: Sequence[vi.CropRow], member: Dict[str, str], tp, tl, conf) -> dict:
    assign = [member.get(c.crop_id) for c in crops]
    by_ind: Dict[str, List[float]] = defaultdict(list)
    for c, a in zip(crops, assign):
        by_ind[a].append(c.observed_at.timestamp())
    sizes = sorted((len(v) for v in by_ind.values()), reverse=True)
    mem: Dict[int, set] = defaultdict(set)
    for i, a in enumerate(assign):
        mem[tl[i]].add(a)
    mixed = sum(1 for a, b in conf if mem[a] & mem[b])
    reseen = 0
    for ts in by_ind.values():
        ts = sorted(ts)
        if len(ts) > 1 and float(np.max(np.diff(ts))) > 600:
            reseen += len(ts)
    days = max(1e-9, (max(c.observed_at for c in crops) - min(c.observed_at for c in crops)).total_seconds() / 86400)
    return {
        "individuals": len(by_ind),
        "individuals_per_day": round(len(by_ind) / max(days, 1.0), 1),
        "with_5_or_more": sum(1 for s in sizes if s >= 5),
        "top": sizes[:3],
        "mixed": round(mixed / len(conf), 3) if conf else None,
        "split": round(sum(assign[i] != assign[j] for i, j in tp) / len(tp), 3) if tp else None,
        "reseen": round(reseen / len(crops), 3),
    }


def check(name: str, got: dict) -> List[str]:
    exp = EXPECTED.get(name)
    problems: List[str] = []
    if not exp:
        return problems
    n = exp["individuals"]
    if name == "baseline":
        if got["individuals"] != n:
            problems.append(f"{name}: individuals {got['individuals']} != {n}")
        if got["top"] != exp["top"]:
            problems.append(f"{name}: top sizes {got['top']} != {exp['top']}")
    elif abs(got["individuals"] - n) > COUNT_TOL * n:
        problems.append(f"{name}: individuals {got['individuals']} not within {COUNT_TOL:.0%} of {n}")
    for k in ("mixed", "split"):
        if got[k] is None or abs(got[k] - exp[k]) > RATE_TOL:
            problems.append(f"{name}: {k} {got[k]} not within {RATE_TOL} of {exp[k]}")
    return problems


# --- live -------------------------------------------------------------------------------------


def load(conn, stream: str, since: Optional[datetime], until: datetime):
    from sqlalchemy import text

    rows = conn.execute(text(
        "SELECT crop_id, observation_id, stream_id, observed_at, created_at, label, box_xyxy, zone, embedding, "
        "embedding_ref, artifact_id, zone_no_embed, thumb_ref, face_embedding, face_thumb_ref "
        "FROM vision_crop_observation WHERE stream_id=:s AND label='person' AND embedding IS NOT NULL "
        "AND NOT zone_no_embed AND created_at <= :until AND (CAST(:since AS timestamptz) IS NULL OR created_at > :since) "
        "ORDER BY created_at, crop_id"), {"s": stream, "since": since, "until": until}).fetchall()
    return [vi._row_to_crop(r) for r in rows], [r.created_at for r in rows]


def run(crops: Sequence[vi.CropRow], landed: Sequence[datetime], pinned: bool) -> dict:
    tz = ZoneInfo(vi.IndividualsConfig().local_tz)
    tp = track_pairs(crops)
    tl = tracklets(len(crops), tp)
    conf = conflicts(crops, tl)
    report = {"crops": len(crops), "faces": sum(1 for c in crops if c.face_embedding),
              "track_pairs": len(tp), "tracklets": len(set(tl)), "different_people_pairs": len(conf),
              "rules": {}, "problems": []}
    for name, cfg in rules().items():
        member, _ = replay(crops, landed, cfg, tz)
        got = score(crops, member, tp, tl, conf)
        got["config"] = {"match_threshold": cfg.match_threshold, "continuation": cfg.continuation_enabled,
                         "continuation_threshold": cfg.continuation_threshold,
                         "continuation_gap_sec": cfg.continuation_gap_sec,
                         "continuation_max_disp": cfg.continuation_max_disp}
        report["rules"][name] = got
        if pinned:
            report["problems"] += check(name, got)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True, help="live DSN (read in a READ ONLY transaction)")
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--since", help="ISO time: only crops that landed after this (default: all)")
    ap.add_argument("--until", help="ISO time: crops that landed up to this (default: the pinned window)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from sqlalchemy import create_engine, text

    since = datetime.fromisoformat(args.since) if args.since else None
    until = datetime.fromisoformat(args.until) if args.until else PINNED_UNTIL
    pinned = since is None and until == PINNED_UNTIL
    engine = create_engine(args.live)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        crops, landed = load(conn, args.stream, since, until)
        conn.rollback()
    engine.dispose()
    if pinned and len(crops) != PINNED_CROPS:
        print(f"cannot replay: {len(crops)} crops in the pinned window, expected {PINNED_CROPS} "
              "(pruned after 7 days?)", file=sys.stderr)
        return 2
    if not crops:
        print("no crops in the window", file=sys.stderr)
        return 2
    report = run(crops, landed, pinned)
    report["window"] = {"since": since.isoformat() if since else None, "until": until.isoformat(), "pinned": pinned}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"crops={report['crops']} faces={report['faces']} track_pairs={report['track_pairs']} "
              f"tracklets={report['tracklets']} different_people_pairs={report['different_people_pairs']}")
        print(f"{'rule':<10} {'inds':>5} {'/day':>6} {'>=5':>4} {'mixed':>6} {'split':>6} {'re-seen':>8}  top")
        for name, r in report["rules"].items():
            print(f"{name:<10} {r['individuals']:>5} {r['individuals_per_day']:>6} {r['with_5_or_more']:>4} "
                  f"{r['mixed']:>6} {r['split']:>6} {r['reseen']:>8}  {r['top']}")
        for p in report["problems"]:
            print("MOVED:", p)
        if pinned and not report["problems"]:
            print("REPRODUCED")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
