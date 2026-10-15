#!/usr/bin/env python3
"""Replay eval: which visual confirmation asks would open on deploy?

Reads the LIVE walkway tables read-only (a READ ONLY transaction: it cannot
write), copies them into a THROWAWAY database that has the three walkway
migrations applied, and runs the real opener (``_open_asks``) there -- the
same SQL and rules the deployed loop will run. Prints individual ids, kinds,
face counts, pair similarities and image counts. Never prints or copies
pictures (only their refs are copied, and only counted in the output).

Also prints the face-similarity distribution behind
VISION_SAME_ASK_MIN_FACE_COSINE: two faces in ONE frame (known different
people) versus the same face a few seconds apart in the same place (almost
always the same person), from vision_crop_observation.face_embedding.

    python evals/run_vision_confirm_asks_replay_eval.py \\
        --live postgresql://postgres:...@localhost:55432/conjourney \\
        --scratch postgresql://postgres:t@127.0.0.1:55499/postgres

The scratch database is TRUNCATEd. Never point --scratch at production.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
for p in (str(SERVICE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app import vision_individuals as vi  # noqa: E402

IND_COLS = ("individual_id, stream_id, kind, centroid, centroid_n, first_seen_at, last_seen_at, sighting_count, "
            "distinct_days, label, labeled_at, label_ask_id, face_centroid, face_n, stationary")
SIGHT_COLS = ("sighting_id, individual_id, stream_id, zone, started_at, ended_at, dwell_sec, observation_count, "
              "thumb_ref, face_thumb_ref, zone_no_embed, stationary, evidence_ref")
ASK_COLS = ("ask_id, asked_of, question, evidence_refs, image_ref, status, answer, answered_at, applied_at, "
            "created_at, expires_at, source_kind, source_ref")


def _copy(live, scratch, table: str, cols: str, where: str = "") -> int:
    from sqlalchemy import text

    rows = live.execute(text(f"SELECT {cols} FROM {table} {where}")).mappings().all()
    names = [c.strip() for c in cols.split(",")]
    jsonb = {"evidence_refs"}
    vals = ", ".join(f"CAST(:{n} AS jsonb)" if n in jsonb else f":{n}" for n in names)
    for r in rows:
        params = {n: (json.dumps(r[n]) if n in jsonb else r[n]) for n in names}
        scratch.execute(text(f"INSERT INTO {table} ({cols}) VALUES ({vals})"), params)
    return len(rows)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def face_distribution(live) -> dict:
    from sqlalchemy import text

    rows = live.execute(text(
        "SELECT observation_id, extract(epoch from observed_at), box_xyxy, face_embedding "
        "FROM vision_crop_observation WHERE face_embedding IS NOT NULL")).fetchall()
    crops = [(r[0], float(r[1]), list(r[2] or []), _unit(list(r[3]))) for r in rows]
    same_frame, same_track, all_pairs = [], [], []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            a, b = crops[i], crops[j]
            sim = sum(x * y for x, y in zip(a[3], b[3]))
            all_pairs.append(sim)
            if a[0] == b[0]:
                same_frame.append(sim)
            elif abs(a[1] - b[1]) <= 3.0 and len(a[2]) == 4 and len(b[2]) == 4 and _iou(a[2], b[2]) >= 0.3:
                same_track.append(sim)
    hist = Counter(min(19, max(0, int((s + 0.2) / 0.05))) for s in all_pairs)
    return {
        "face_crops": len(crops),
        "all_pairs": len(all_pairs),
        "histogram_all_pairs": {f"{-0.2 + 0.05 * k:+.2f}": hist[k] for k in range(20) if hist[k]},
        "same_frame_known_different": {"n": len(same_frame), **{f"p{int(q * 100)}": _pct(same_frame, q)
                                                               for q in (0.1, 0.25, 0.5, 0.75, 0.9)}},
        "same_track_likely_same": {"n": len(same_track), **{f"p{int(q * 100)}": _pct(same_track, q)
                                                           for q in (0.1, 0.25, 0.5, 0.75, 0.9)}},
    }


def run(live_dsn: str, scratch_dsn: str, now: datetime) -> dict:
    from sqlalchemy import create_engine, text

    live_e, scratch_e = create_engine(live_dsn), create_engine(scratch_dsn)
    cfg = vi.IndividualsConfig()
    tz = ZoneInfo(cfg.local_tz)
    with live_e.connect() as live, scratch_e.begin() as scratch:
        live.execute(text("SET TRANSACTION READ ONLY"))
        vi._check_tables(scratch)  # the scratch db must have all three migrations
        scratch.execute(text("TRUNCATE orion_ask, vision_individual_pair_verdict, vision_individual_sighting, "
                             "vision_individual CASCADE"))
        copied = {
            "vision_individual": _copy(live, scratch, "vision_individual", IND_COLS),
            "vision_individual_sighting": _copy(live, scratch, "vision_individual_sighting", SIGHT_COLS),
            "orion_ask": _copy(live, scratch, "orion_ask", ASK_COLS),
        }
        has_verdicts = live.execute(text("SELECT to_regclass('vision_individual_pair_verdict')")).scalar()
        if has_verdicts:
            copied["vision_individual_pair_verdict"] = _copy(
                live, scratch, "vision_individual_pair_verdict",
                "individual_a, individual_b, verdict, face_cosine, ask_id, decided_at")
        faces = face_distribution(live)
        opened = vi._open_asks(scratch, now, tz, cfg)
        face_n = dict(scratch.execute(text("SELECT individual_id, face_n FROM vision_individual")).fetchall())
        live.rollback()
    asks = []
    for a in opened:
        cos = next((r.split(":", 1)[1] for r in a["evidence_refs"] if r.startswith("face_cosine:")), None)
        ids = a["source_ref"].split(vi.PAIR_SEP)
        asks.append({"kind": a["source_kind"], "individuals": ids,
                     "face_n": [face_n.get(i) for i in ids], "face_cosine": cos,
                     "images": len(a["image_refs"]),
                     "rule": "face" if vi.FACE_RULE_REF in a["evidence_refs"] else
                             ("pair" if a["source_kind"] == vi.SOURCE_KIND_SAME else "sightings"),
                     "question": a["question"]})
    return {"now": now.isoformat(), "copied_rows": copied, "asks_that_would_open": asks,
            "count_by_kind": dict(Counter(a["kind"] for a in asks)),
            "config": {"ask_face_min_faces": cfg.ask_face_min_faces,
                       "same_ask_min_face_cosine": cfg.same_ask_min_face_cosine,
                       "confirm_ask_daily_cap": cfg.confirm_ask_daily_cap,
                       "face_match_threshold": cfg.face_match_threshold},
            "face_similarity": faces}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--live", required=True, help="live DSN (opened READ ONLY)")
    ap.add_argument("--scratch", required=True, help="throwaway DSN with the walkway migrations (TRUNCATEd)")
    args = ap.parse_args()
    if args.live == args.scratch:
        print("refusing: --scratch must not be the live database", file=sys.stderr)
        return 2
    report = run(args.live, args.scratch, datetime.now(timezone.utc))
    print(json.dumps(report, indent=2, default=str))
    # Pass/fail: every opened ask carries at least one picture, a pair carries two.
    bad = [a for a in report["asks_that_would_open"]
           if a["images"] < (2 if a["kind"] == vi.SOURCE_KIND_SAME else 1) and a["rule"] != "sightings"]
    if bad:
        print(f"FAIL: {len(bad)} visual ask(s) without their pictures", file=sys.stderr)
        return 1
    print(f"OK: {len(report['asks_that_would_open'])} ask(s) would open", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
