"""Live check: does the office camera (cam0) get what the walkway gets?

Run after deploying orion-vision-retina-office (2026-11-04). Read-only. Each
step is one row in the live database (or one file in the frames folder), so a
dead step names itself instead of hiding behind a later one.

    python services/orion-vision-retina/evals/run_office_parity_live_eval.py --hours 2

Exit code 0 only when every step moved. Needs `docker exec` access to
orion-athena-sql-db (the same way the repo's other live checks read it).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

FRAMES_DIR = Path("/mnt/telemetry/vision/frames")

# (step, SQL returning one count). :H is the lookback in hours.
STEPS = [
    ("person crops stored", "SELECT count(*) FROM vision_crop_observation WHERE stream_id='cam0' AND created_at > now() - interval ':H hours'"),
    ("crops with a face vector", "SELECT count(*) FROM vision_crop_observation WHERE stream_id='cam0' AND face_embedding IS NOT NULL AND created_at > now() - interval ':H hours'"),
    ("individual sightings", "SELECT count(*) FROM vision_individual_sighting WHERE stream_id='cam0' AND started_at > now() - interval ':H hours'"),
    ("activity captions (Qwen lane)", "SELECT count(*) FROM vision_events WHERE stream_id='cam0' AND event_type='camera_caption' AND created_at > now() - interval ':H hours'"),
]


def _count(sql: str) -> int | None:
    try:
        out = subprocess.run(
            ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-Atc", sql],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return int(out or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"  read failed: {exc}", file=sys.stderr)
        return None


def _fresh_frames(max_age_s: float = 60.0) -> int:
    now = time.time()
    return sum(1 for p in FRAMES_DIR.glob("cam0_cam0_*.jpg") if now - p.stat().st_mtime < max_age_s)


def _fresh_edge_frames(max_age_s: float = 60.0) -> int:
    """orion-vision-edge's own cam0 files (``frame_*``). Any fresh one means the
    edge is still running and cam0 is published twice."""
    now = time.time()
    return sum(1 for p in FRAMES_DIR.glob("frame_*.jpg") if now - p.stat().st_mtime < max_age_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=2)
    args = ap.parse_args()
    ok = True
    frames = _fresh_frames()
    print(f"{'retina frames in the last 60 s':34s} {frames}")
    ok &= frames > 0
    edge = _fresh_edge_frames()
    print(f"{'edge frames in the last 60 s (want 0)':34s} {edge}")
    ok &= edge == 0
    for name, sql in STEPS:
        n = _count(sql.replace(":H", str(int(args.hours))))
        print(f"{name:34s} {'UNREAD' if n is None else n}")
        ok &= bool(n)
    print("VERDICT:", "PASS" if ok else "FAIL (a zero row is the first dead step; someone must be in view)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
