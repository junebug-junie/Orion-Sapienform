"""Measure RTSP packet-loss smear in saved retina frames.

When RTP packets are lost, the H.264 decoder copies the last good row of
pixels down the rest of the slice: the picture below that point becomes
vertical streaks (walkway camera, 2026-10-15, UDP transport). Inside a smear,
each row equals the row above it, so the mean absolute difference between
neighbouring rows collapses to ~0. A real outdoor scene almost never has a
long run of such rows.

smear_fraction = longest run of near-identical consecutive rows / height.
Clean frames read ~0; a frame smeared from 27% down reads ~0.7.

Usage (host or inside the retina image):
  python evals/frame_smear.py /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20
Exit code 1 if any frame's smear_fraction exceeds --max (default 0.10).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def smear_fraction(gray: np.ndarray, *, row_diff_max: float = 1.0) -> float:
    g = gray.astype(np.int16)
    diffs = np.abs(np.diff(g, axis=0)).mean(axis=1)  # one value per row pair
    flat = diffs < row_diff_max
    best = run = 0
    for f in flat:
        run = run + 1 if f else 0
        best = max(best, run)
    return best / gray.shape[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--max", type=float, default=0.10)
    a = ap.parse_args(argv)
    files = sorted(Path(a.directory).glob(a.glob), key=lambda p: p.stat().st_mtime)[-a.last :]
    if not files:
        print("no frames matched")
        return 2
    worst = 0.0
    bad = 0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        s = smear_fraction(img)
        worst = max(worst, s)
        bad += s > a.max
    print(f"frames={len(files)} smeared={bad} worst_smear_fraction={worst:.2f} (max allowed {a.max})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
