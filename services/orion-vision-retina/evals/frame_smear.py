"""Measure RTSP packet-loss smear in saved retina frames.

When RTP packets are lost, the H.264 decoder copies the last good row of
pixels down the rest of the slice: the picture below that point becomes
vertical streaks (walkway camera, 2026-10-15, UDP transport).

smear_fraction = share of 16-row strips whose vertical change is under 0.3 of
their horizontal change. Live calibration 2026-10-15: 60 indoor cam0 frames
never went below 0.56 (smear_fraction 0.00); 44 UDP walkway frames went to
0.17 (smear_fraction up to 0.17, earlier frames ~0.7 by eye). No clean walkway
frame existed yet to calibrate against: the post-deploy run is that check.

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


def smear_fraction(
    gray: np.ndarray, *, band: int = 16, ratio_max: float = 0.3, dx_min: float = 2.0
) -> float:
    """Fraction of `band`-row strips that are vertical streaks.

    For each strip: mean |change going down| / mean |change going across|.
    A real scene is roughly isotropic (ratio ~0.5-1). A smear copies rows
    down, so going down barely changes while going across still does
    (ratio << 1). Strips with almost no horizontal texture (flat sky, wall,
    night) are skipped: they cannot be told apart and are not smear.
    """
    g = gray.astype(np.float32)
    dy = np.abs(np.diff(g, axis=0))[:, :-1]
    dx = np.abs(np.diff(g, axis=1))[:-1]
    strips = smeared = 0
    for y in range(0, dy.shape[0] - band + 1, band):
        across = float(dx[y : y + band].mean())
        strips += 1
        if across > dx_min and float(dy[y : y + band].mean()) / across < ratio_max:
            smeared += 1
    return smeared / strips if strips else 0.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--last", type=int, default=20)
    ap.add_argument("--max", type=float, default=0.05)
    a = ap.parse_args(argv)
    files = sorted(Path(a.directory).glob(a.glob), key=lambda p: p.stat().st_mtime)[-a.last :]
    if not files:
        print("no frames matched")
        return 2
    worst = 0.0
    bad = 0
    read = 0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        read += 1
        s = smear_fraction(img)
        worst = max(worst, s)
        bad += s > a.max
    print(f"frames={read} smeared={bad} worst_smear_fraction={worst:.2f} (max allowed {a.max})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
