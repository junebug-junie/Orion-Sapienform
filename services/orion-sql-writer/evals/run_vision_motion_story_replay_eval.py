"""Read-only replay of walkway motion stories over live crops.

docs/superpowers/specs/2026-10-16-walkway-motion-stories-proposal.md, acceptance check 1.

Reads ``vision_crop_observation`` (+ labels / stationary / prop boxes) in a
READ ONLY session, replays the reducer the way the loop runs it (cursor steps
of ``--step-sec``), and reports passages, published stories, glimpses, props,
together pairs, the speed-class histogram and the detected-frame interval
distribution. Nothing is written.

Default window starts at VISION_MOTION_CLEAN_SINCE (2026-10-16T18:59Z, the
RTSP stream fix). Earlier rows are refused unless ``--include-corrupt``.

Fails (exit 1) when:
- a published story carries a name the strict rule does not allow;
- a published fast-class story (running / riding / driving) has < 3 usable detections;
- a published story's zone path names a zone no unclipped box stood in;
- the stepwise replay (what the loop does) and a single-window replay disagree.

Names are never printed: a named story shows ``<named:xxxxxxxx>`` (individual id prefix).

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_vision_motion_story_replay_eval.py \\
        [--since 2026-10-16T18:59:00Z] [--until ...] [--pg postgresql://...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from app.vision_motion import (  # noqa: E402
    MotionConfig, Passage, build_stories, is_edge_clipped, link_passages, name_for,
)
from app.vision_motion_loop import LoopConfig, close_window, parse_utc, read_context, read_crops  # noqa: E402

DEFAULT_CLEAN_SINCE = "2026-10-16T18:59:00Z"
DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
FAST = {"running", "riding", "driving"}
PROP_IDS = ("fefb24d7e4074e9f827ad711874f6edf",)


def redact(story) -> str:
    if story.name_label and story.individual_id:
        return story.sentence.replace(story.name_label, f"<named:{story.individual_id[:8]}>", 1)
    return story.sentence


def stepwise(crops, *, stream_id, since, until, cfg, zones, context, step_sec):
    """What the loop does: cursor steps of step_sec from ``since``, settle
    ignored (the replay window is in the past)."""
    out = {}
    pu = since
    gap = timedelta(seconds=cfg.motion.link_gap_sec)
    final = until + gap + timedelta(seconds=1)   # flush passages ending right at `until`
    while pu < until:
        horizon = min(final, pu + timedelta(seconds=step_sec) + gap)
        ws = pu - timedelta(seconds=cfg.window_sec)
        window = [c for c in crops if ws < c.t <= horizon]
        stories, new_cursor = close_window(window, stream_id=stream_id, processed_until=pu, horizon=horizon,
                                           window_start=ws, cfg=cfg, zones=zones, context=context)
        for s in stories:
            if s.started_at >= since:
                out.setdefault(s.story_id, s)
        pu = new_cursor
    return out


def check(stories, labels, cfg: MotionConfig):
    failures = []
    for s in stories:
        if not s.published:
            continue
        if s.name_label:
            p = Passage(s.kind, [])
            from app.vision_motion import MotionCrop

            for e in s.evidence:
                p.crops.append(MotionCrop(e["crop_id"], datetime.fromisoformat(e["t"]), e["label"],
                                          tuple(e["box"]), individual_id=e["individual_id"]))
            iid, _label, _why = name_for(p, labels, cfg)
            if iid != s.individual_id:
                failures.append(f"{s.story_id}: name not allowed by the strict rule")
        if s.speed_class in FAST and s.used_detections < 3:
            failures.append(f"{s.story_id}: {s.speed_class} on {s.used_detections} usable detections")
        stood = {e["zone"] or "outside the zones" for e in s.evidence if not e["clipped"]}
        bad = [z for z in s.zone_path if z not in stood]
        if bad:
            failures.append(f"{s.story_id}: zone claim {bad} not backed by an unclipped box")
        if any(is_edge_clipped(e["box"], cfg) != e["clipped"] for e in s.evidence):
            failures.append(f"{s.story_id}: evidence clip flag disagrees with the box")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", default=os.getenv("VISION_MOTION_CLEAN_SINCE", DEFAULT_CLEAN_SINCE))
    ap.add_argument("--until", default="")
    ap.add_argument("--stream", default="walkway")
    ap.add_argument("--pg", default=os.getenv("POSTGRES_URI_READONLY") or DEFAULT_PG)
    ap.add_argument("--include-corrupt", action="store_true")
    ap.add_argument("--step-sec", type=float, default=15.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=10, help="published clean stories to print")
    args = ap.parse_args(argv)

    clean_since = parse_utc(DEFAULT_CLEAN_SINCE)
    since = parse_utc(args.since)
    if since < clean_since and not args.include_corrupt:
        print(f"REFUSED: --since {args.since} is before the stream fix ({DEFAULT_CLEAN_SINCE}); "
              "pass --include-corrupt to replay corrupt frames (never counted toward accuracy)")
        return 2
    until = parse_utc(args.until) if args.until else datetime.now(timezone.utc) - timedelta(seconds=60)

    from orion.vision.zones import load_zones
    from sqlalchemy import create_engine

    zones = load_zones().get(args.stream, [])
    cfg = LoopConfig(clean_since=clean_since, prop_individual_ids=PROP_IDS)
    engine = create_engine(args.pg, connect_args={"options": "-c default_transaction_read_only=on"})
    try:
        with engine.connect() as conn:
            lead = timedelta(seconds=cfg.window_sec)
            crops = read_crops(conn, args.stream, since - lead, until + timedelta(seconds=cfg.motion.link_gap_sec))
            ids = sorted({c.individual_id for c in crops if c.individual_id})
            context = read_context(conn, args.stream, ids, PROP_IDS)
            conn.rollback()
    finally:
        engine.dispose()
    labels = context[0]

    step = stepwise(crops, stream_id=args.stream, since=since, until=until, cfg=cfg, zones=zones,
                    context=context, step_sec=args.step_sec)
    passages = link_passages([c for c in crops if c.t <= until + timedelta(seconds=cfg.motion.link_gap_sec)],
                             cfg.motion)
    whole_idx = {i for i, p in enumerate(passages) if since <= p.first.t and p.last.t <= until}
    whole = {s.story_id: s for s in build_stories(
        passages, stream_id=args.stream, zones=zones, cfg=cfg.motion, labels=labels,
        stationary_ids=context[1], prop_ids=set(PROP_IDS), prop_boxes=context[2], clean_since=clean_since,
        emit=whole_idx)}

    stories = sorted(step.values(), key=lambda s: s.started_at)
    failures = check(stories, labels, cfg.motion)
    pub_step = {k for k, s in step.items() if s.published and s.ended_at <= until}
    pub_whole = {k for k, s in whole.items() if s.published}
    if pub_step != pub_whole:
        failures.append(f"stepwise vs whole-window published ids differ: only stepwise "
                        f"{sorted(pub_step - pub_whole)[:5]}, only whole {sorted(pub_whole - pub_step)[:5]}")

    times = sorted({c.t for c in crops if since <= c.t <= until})
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    q = (lambda xs, f: round(sorted(xs)[min(len(xs) - 1, int(f * len(xs)))], 2) if xs else None)
    published = [s for s in stories if s.published]
    reasons = Counter(s.drop_reason or "published" for s in stories)
    withheld = Counter(s.name_withheld for s in stories if s.name_withheld)
    loose_only = sum(1 for s in stories if s.name_withheld)
    report = {
        "window": [since.isoformat(), until.isoformat()],
        "crops": sum(1 for c in crops if since <= c.t <= until),
        "passages": len(stories),
        "published": len(published),
        "outcomes": dict(reasons),
        "together_pairs": sum(len(s.together_story_ids) for s in published) // 2,
        "speed_classes_published": dict(Counter(s.speed_class for s in published)),
        "kinds_published": dict(Counter(s.kind for s in published)),
        "names_shown": sum(1 for s in published if s.name_label),
        "names_withheld_by_strict_rule": dict(withheld),
        "loose_rule_would_have_named": loose_only,
        "detected_frame_interval_sec": {"n": len(gaps), "p10": q(gaps, .1), "p50": q(gaps, .5),
                                        "p90": q(gaps, .9), "p99": q(gaps, .99)},
        "failures": failures,
    }
    print(json.dumps(report, indent=1))
    print(f"\nFirst {args.show} published stories (clean frames):")
    for s in [s for s in published if s.clean][:args.show]:
        print(f"- {s.started_at:%Y-%m-%d %H:%M:%S}Z {s.story_id[:10]} [{s.kind}/{s.speed_class}, "
              f"{s.detection_count} det, {s.net_travel_bh} BH] {redact(s)}")
    for t0, t1, tag in (("2026-10-16T19:33:20+00:00", "2026-10-16T19:33:37+00:00", "S9"),
                        ("2026-10-16T19:33:25+00:00", "2026-10-16T19:33:54+00:00", "S10")):
        a, b = datetime.fromisoformat(t0), datetime.fromisoformat(t1)
        hits = [s for s in stories if a <= s.started_at <= b]
        if since <= a and b <= until:
            print(f"\n{tag} window {t0[11:19]}-{t1[11:19]}:")
            for s in hits:
                print(f"  {s.kind}/{s.speed_class} n={s.detection_count} used={s.used_detections} "
                      f"published={s.published} reason={s.drop_reason} name_withheld={s.name_withheld} :: {redact(s)}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [dict(story_id=s.story_id, started_at=s.started_at.isoformat(), kind=s.kind, published=s.published,
                  drop_reason=s.drop_reason, speed_class=s.speed_class, sentence=redact(s),
                  individual_id=(s.individual_id or "")[:8], n=s.detection_count) for s in stories], indent=1))
    if failures:
        print("\nFAIL:\n" + "\n".join(failures))
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
