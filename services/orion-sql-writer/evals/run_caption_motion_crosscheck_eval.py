"""Read-only replay of the caption motion cross-check over the live tables.

Runs `app/vision_caption_motion.check_caption` (the same reads and decision the
reducer uses) over every `camera_caption` row of the story streams in the last
``--hours`` (default 72) inside a READ ONLY transaction, and prints counts per
stream and verdict, plus the claim words behind each verdict. Nothing is
written: no tag is added here.

``--fixture PATH`` also writes the inputs of every decided caption whose text
says "driving" (caption text + refs, the frame's crops, and the story speed
class of each crop) as JSON, for the offline replay test
(`tests/fixtures/caption_motion_driving_captions.json`).

Usage:
    PYTHONPATH=. python services/orion-sql-writer/evals/run_caption_motion_crosscheck_eval.py \\
        [--hours 72] [--streams walkway,doorbell] [--pg postgresql://...] [--fixture out.json]
        [--since ISO --until ISO]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3]))

from app.vision_caption_motion import _refs, check_caption, parse_caption_refs, read_frame  # noqa: E402

DEFAULT_PG = "postgresql://postgres:postgres@localhost:55432/conjourney"
# Same as the reducer: LoopConfig.window_sec + the close gap.
SETTLE_SEC = 900.0 + 11.0
FIXTURE_SEC = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--since", default="")
    ap.add_argument("--until", default="")
    ap.add_argument("--streams", default="walkway,doorbell")
    ap.add_argument("--pg", default=DEFAULT_PG)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--fixture", default="")
    args = ap.parse_args()

    from sqlalchemy import create_engine, text

    now = datetime.now(timezone.utc)
    since = datetime.fromisoformat(args.since) if args.since else now - timedelta(hours=args.hours)
    until = datetime.fromisoformat(args.until) if args.until else now
    engine = create_engine(args.pg)
    fixture = []
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        for stream in [s.strip() for s in args.streams.split(",") if s.strip()]:
            cursor = conn.execute(text(
                "SELECT processed_until FROM vision_motion_story_cursor WHERE stream_id = :s"),
                {"s": stream}).scalar()
            rows = conn.execute(text(
                "SELECT event_id, created_at, narrative, evidence_refs FROM vision_events "
                "WHERE stream_id = :s AND event_type = 'camera_caption' AND created_at BETWEEN :a AND :b "
                "ORDER BY created_at"), {"s": stream, "a": since, "b": until}).fetchall()
            status = Counter()
            claim_words = Counter()
            for event_id, created_at, narrative, raw_refs in rows:
                refs = _refs(raw_refs)
                v = check_caption(conn, stream, narrative or "", refs, cursor=cursor,
                                  settle_sec=SETTLE_SEC, min_iou=args.min_iou)
                status[v.status] += 1
                for ch in v.checks:
                    claim_words[(ch.claim.word, ch.verdict, ch.speed_class or "-")] += 1
                if args.fixture and "driving" in (narrative or "").lower() and v.status in (
                        "contradicted", "supported", "undecided"):
                    frame_ts, _ = parse_caption_refs(refs)
                    crops, story_evidence = read_frame(conn, stream, frame_ts) if frame_ts else ([], {})
                    # Only the boxes near the frame (local_motion reads LOCAL_WINDOW_SEC of them).
                    near = {cid: [e for e in ev if isinstance(e, dict) and e.get("t") and abs(
                                (datetime.fromisoformat(str(e["t"])) - frame_ts).total_seconds()) <= FIXTURE_SEC]
                            for cid, ev in story_evidence.items()}
                    fixture.append({
                        "event_id": event_id, "created_at": created_at.isoformat(), "stream_id": stream,
                        "narrative": narrative, "evidence_refs": refs,
                        "crops": [{"crop_id": c.crop_id, "label": c.label, "box": list(c.box)} for c in crops],
                        "story_evidence": near, "live_status": v.status,
                    })
            total = sum(status.values())
            print(f"== {stream}: {total} camera_caption rows {since:%Y-%m-%d %H:%M} .. {until:%Y-%m-%d %H:%M} UTC "
                  f"(story cursor {cursor})")
            for k in ("contradicted", "supported", "undecided", "pending", "no_claim"):
                print(f"   {k:13s} {status.get(k, 0)}")
            for (word, verdict, speed), n in sorted(claim_words.items(), key=lambda kv: -kv[1]):
                print(f"   claim {word!r:12s} {verdict:13s} local={speed:9s} {n}")
    if args.fixture:
        Path(args.fixture).write_text(json.dumps(fixture, indent=1, sort_keys=True))
        print(f"wrote {len(fixture)} driving captions to {args.fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
