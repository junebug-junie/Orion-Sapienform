"""Read-only check of walkway story pictures and the darkness threshold (2026-10-17).

Two questions, both from live data, nothing written:

1. Coverage: of the published stories since ``--since``, how many got a strip,
   a full frame, and how many crops were saved vs ``frame_gone`` / ``not_captured``.
   A high ``frame_gone`` share means the capture step is losing the race with
   the ~60 s frame pruning.
2. Threshold: the strip's median crop contrast (``scene_contrast``, std of
   luma) split by Juniper's verdict. ``VISION_STORY_STRIP_DARK_CONTRAST`` (0.10)
   was set from tonight's frames and daytime crop thumbnails before any verdict
   existed. Once she has checked stories, "Can't tell" stories should sit
   below it and judged stories above it. The eval prints where each group
   falls and how many judged stories the threshold would have hidden.

Fails (exit 1) when a published story with a strip claims a frame file that is
not on disk (unless pruned), or a story with ``dark`` set has no frame.

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_story_strip_darkness_eval.py \\
        [--since 2026-10-17T00:00:00Z] [--pg postgresql://...] [--strip-dir /mnt/telemetry/vision/story_strips]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"


def _q(values):
    if not values:
        return "n=0"
    v = sorted(values)
    pick = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1)))]  # noqa: E731
    return f"n={len(v)} min={v[0]:.3f} p10={pick(0.1):.3f} p50={statistics.median(v):.3f} max={v[-1]:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pg", default=os.getenv("ORION_EVAL_PG_URI", DEFAULT_PG))
    ap.add_argument("--since", default="2026-10-17T00:00:00Z")
    ap.add_argument("--strip-dir", default=os.getenv("VISION_STORY_STRIP_DIR", "/mnt/telemetry/vision/story_strips"))
    ap.add_argument("--threshold", type=float, default=float(os.getenv("VISION_STORY_STRIP_DARK_CONTRAST", "0.10")))
    args = ap.parse_args(argv)

    from sqlalchemy import create_engine, text

    engine = create_engine(args.pg)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        rows = conn.execute(text(
            "SELECT s.story_id, s.evidence_strip, s.scene_contrast, s.frame_luminance, s.dark, v.verdict, "
            "v.cant_tell FROM vision_motion_story s LEFT JOIN vision_motion_story_verdict v USING (story_id) "
            "WHERE s.published AND s.started_at >= CAST(:since AS timestamptz) ORDER BY s.started_at"),
            {"since": args.since}).mappings().all()
    problems = []
    cov = Counter()
    crops = Counter()
    by_group = {"cant_tell": [], "judged": [], "unchecked": []}
    for r in rows:
        strip = r["evidence_strip"]
        if isinstance(strip, str):
            strip = json.loads(strip)
        if not strip:
            cov["no_strip"] += 1
            continue
        cov["strip"] += 1
        frame = strip.get("frame")
        if frame:
            cov["frame"] += 1
            if not strip.get("pruned_at") and not (Path(args.strip_dir) / r["story_id"] / frame["file"]).is_file():
                problems.append(f"{r['story_id']}: frame file missing on disk")
        elif r["dark"] is not None and not any("file" in c for c in strip.get("crops", [])):
            problems.append(f"{r['story_id']}: dark={r['dark']} without any picture")
        for c in strip.get("crops", []):
            crops["saved" if "file" in c else c.get("missing", "?")] += 1
        if r["scene_contrast"] is not None:
            g = "cant_tell" if r["cant_tell"] else "judged" if r["verdict"] else "unchecked"
            by_group[g].append(float(r["scene_contrast"]))

    print(f"published stories since {args.since}: {len(rows)}")
    print(f"  strips: {dict(cov)}")
    print(f"  crops:  {dict(crops)}")
    print(f"scene_contrast (median crop RMS contrast), threshold {args.threshold}:")
    for g, vals in by_group.items():
        below = sum(1 for v in vals if v < args.threshold)
        print(f"  {g:10s} {_q(vals)}  below threshold: {below}")
    hidden = sum(1 for v in by_group["judged"] if v < args.threshold)
    if by_group["judged"] or by_group["cant_tell"]:
        print(f"  judged stories the threshold would hide: {hidden}; "
              f"can't-tell stories it lets through: {sum(1 for v in by_group['cant_tell'] if v >= args.threshold)}")
    else:
        print("  no verdicts on stories with a picture yet: threshold UNVERIFIED against Juniper's judgement")
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
