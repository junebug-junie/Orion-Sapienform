"""Find H.264 decode corruption in saved retina frames: smear and decoder gray.

Two ways a broken H.264 stream shows up in a frame:

1. Smear. A slice loses data, so the decoder copies the last good row of
   pixels down the rest of the slice: the picture below that point becomes
   vertical streaks. `smear_fraction` = share of 16-row strips whose vertical
   change is under 0.3 of their horizontal change.
2. Decoder gray. A reference frame is lost, so FFmpeg's error concealment
   paints the missing picture with mid-gray (Y=Cb=Cr=128, which OpenCV turns
   into YCrCb ~(130, 128, 128)). `gray_fraction` = share of 16x16 macroblocks
   that are flat (luma std < 0.6), at that exact gray (|Y-130| < 4) and
   colourless (|Cr-128|, |Cb-128| < 4). Clear sky fails the colour test (blue);
   dark night fails the luma test. Overcast sky, fog and infrared (black and
   white) night mode can be neutral and mid-gray: they are only kept out by
   the tight flatness/luma window and the 0.4 frame limit.

A frame is `corrupt` when smear_fraction > --max or gray_fraction > --max-gray.

Smear is measured on a copy scaled down to 640 px wide. Per-pixel change
shrinks as resolution grows, so on the 2560x1920 walkway frames every strip
fell under the 2.0 texture floor and was skipped as "flat": streaked frames
read smear_fraction 0.00 (2026-10-15, after PR #2692).

Frames whose JPEG has no end marker are skipped, not scored: the retina
writes frames in place, so the newest file can be half-written, and libjpeg
fills the missing part with flat gray.

Calibration 2026-10-15 (numbers in the fix/retina-smear-eval-gray PR report).
Gray was NOT measured on real corrupted camera frames (none were on disk);
its positive side comes from FFmpeg's own concealment output.

Usage (host; needs opencv, e.g. the repo .venv):
  .venv/bin/python services/orion-vision-retina/evals/frame_smear.py \\
      /mnt/telemetry/vision/frames --glob 'walkway_*.jpg' --last 20
Exit 1 if the share of corrupt frames exceeds --max-corrupt-fraction
(default 0: any corrupt frame fails). Exit 2 if no frame could be read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# The math lives in app/frame_corruption.py so the live retina self-heal uses
# exactly what this eval measures. Make the service dir importable when this
# runs as a plain script from the repo root.
_SERVICE_DIR = str(Path(__file__).resolve().parents[1])
if _SERVICE_DIR not in sys.path:
    sys.path.insert(0, _SERVICE_DIR)

from app.frame_corruption import (  # noqa: E402
    DEFAULT_MAX_GRAY,
    DEFAULT_MAX_SMEAR,
    GRAY_LUMA,
    SMEAR_WIDTH,
    frame_report,
    gray_fraction,
    smear_fraction,
)

__all__ = ["DEFAULT_MAX_GRAY", "DEFAULT_MAX_SMEAR", "GRAY_LUMA", "SMEAR_WIDTH", "frame_report", "gray_fraction", "smear_fraction", "main"]


def _positive_int(v: str) -> int:
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _newest(directory: Path, pattern: str, last: int) -> list[Path]:
    """Newest `last` files by mtime; files deleted mid-listing are dropped."""
    found = []
    for p in directory.glob(pattern):
        try:
            found.append((p.stat().st_mtime, p))
        except FileNotFoundError:  # retina cleanup removed it
            continue
    found.sort(key=lambda t: t[0])
    return [p for _, p in found[-last:]]


def load_frame(p: Path) -> np.ndarray | None:
    """Decode a saved frame; None if gone, unreadable, or a JPEG with no end marker."""
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if p.suffix.lower() in (".jpg", ".jpeg") and data.rstrip(b"\x00")[-2:] != b"\xff\xd9":
        return None  # half-written: libjpeg would fill the rest with flat gray
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find H.264 smear and decoder-gray corruption in saved frames.")
    ap.add_argument("directory")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--last", type=_positive_int, default=20)
    ap.add_argument("--max", type=float, default=DEFAULT_MAX_SMEAR, help="smear_fraction limit per frame")
    ap.add_argument("--max-gray", type=float, default=DEFAULT_MAX_GRAY, help="gray_fraction limit per frame")
    ap.add_argument(
        "--max-corrupt-fraction",
        type=float,
        default=0.0,
        help="share of frames allowed to be corrupt before exit 1 (default 0: any corrupt frame fails)",
    )
    ap.add_argument("--quiet", action="store_true", help="print the summary line only")
    a = ap.parse_args(argv)
    files = _newest(Path(a.directory), a.glob, a.last)
    read = skipped = smeared = grayed = corrupt = 0
    worst_smear = worst_gray = 0.0
    for p in files:
        img = load_frame(p)
        if img is None:
            skipped += 1
            continue
        read += 1
        r = frame_report(img, max_smear=a.max, max_gray=a.max_gray)
        worst_smear = max(worst_smear, r["smear_fraction"])
        worst_gray = max(worst_gray, r["gray_fraction"])
        smeared += r["smear_fraction"] > a.max
        grayed += r["gray_fraction"] > a.max_gray
        corrupt += r["corrupt"]
        if not a.quiet:
            print(
                f"{p.name} smear_fraction={r['smear_fraction']:.2f} "
                f"gray_fraction={r['gray_fraction']:.2f} corrupt={r['corrupt']}"
            )
    if not read:
        print("no frames matched" if not files else f"no frame could be read (skipped={skipped})")
        return 2
    share = corrupt / read
    print(
        f"frames={read} skipped_unreadable={skipped} corrupt={corrupt} ({share:.2f}, max allowed {a.max_corrupt_fraction}) "
        f"smeared={smeared} grayed={grayed} "
        f"worst_smear_fraction={worst_smear:.2f} (max {a.max}) "
        f"worst_gray_fraction={worst_gray:.2f} (max {a.max_gray})"
    )
    return 1 if share > a.max_corrupt_fraction else 0


if __name__ == "__main__":
    sys.exit(main())
