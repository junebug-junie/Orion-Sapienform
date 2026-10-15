"""Find H.264 decode corruption in saved retina frames: smear and decoder gray.

Two ways a broken H.264 stream shows up in a frame:

1. Smear. A slice loses data, so the decoder copies the last good row of
   pixels down the rest of the slice: the picture below that point becomes
   vertical streaks. `smear_fraction` = share of 16-row strips whose vertical
   change is under 0.3 of their horizontal change.
2. Decoder gray. A reference frame is lost, so FFmpeg's error concealment
   paints the missing picture with mid-gray (Y=Cb=Cr=128, which OpenCV turns
   into YCrCb ~(130, 128, 128)). `gray_fraction` = share of 16x16 macroblocks
   that are flat (luma std < 1.5), near mid-gray luma (|Y-128| < 8) and
   colourless (|Cr-128|, |Cb-128| < 4). Clear sky fails the colour test (blue);
   night fails the luma test (dark).

A frame is `corrupt` when smear_fraction > --max or gray_fraction > --max-gray.

Smear is measured on a copy scaled down to 640 px wide. Per-pixel change
shrinks as resolution grows, so on the 2560x1920 walkway frames every strip
fell under the 2.0 texture floor and was skipped as "flat": streaked frames
read smear_fraction 0.00 (2026-10-15, after PR #2692).

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

SMEAR_WIDTH = 640
DEFAULT_MAX_SMEAR = 0.05
DEFAULT_MAX_GRAY = 0.25


def smear_fraction(
    gray: np.ndarray,
    *,
    band: int = 16,
    ratio_max: float = 0.3,
    dx_min: float = 2.0,
    width: int = SMEAR_WIDTH,
) -> float:
    """Fraction of `band`-row strips that are vertical streaks.

    For each strip: mean |change going down| / mean |change going across|.
    A real scene is roughly isotropic (ratio ~0.5-1). A smear copies rows
    down, so going down barely changes while going across still does
    (ratio << 1). Strips with almost no horizontal texture (flat sky, wall,
    night) are skipped: they cannot be told apart and are not smear.
    Frames wider than `width` are scaled down first so `dx_min` means the
    same thing at every resolution.
    """
    if gray.shape[1] > width:
        h = max(2, round(gray.shape[0] * width / gray.shape[1]))
        gray = cv2.resize(gray, (width, h), interpolation=cv2.INTER_AREA)
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


def gray_fraction(
    bgr: np.ndarray,
    *,
    block: int = 16,
    std_max: float = 1.5,
    luma_tol: float = 8.0,
    chroma_tol: float = 4.0,
) -> float:
    """Fraction of `block`x`block` macroblocks filled with decoder mid-gray."""
    h, w = bgr.shape[:2]
    h -= h % block
    w -= w % block
    if h == 0 or w == 0:
        return 0.0
    ycc = cv2.cvtColor(np.ascontiguousarray(bgr[:h, :w]), cv2.COLOR_BGR2YCrCb).astype(np.float32)
    blocks = ycc.reshape(h // block, block, w // block, block, 3)
    mean = blocks.mean(axis=(1, 3))
    luma_std = blocks[..., 0].std(axis=(1, 3))
    hit = (
        (luma_std < std_max)
        & (np.abs(mean[..., 0] - 128.0) < luma_tol)
        & (np.abs(mean[..., 1] - 128.0) < chroma_tol)
        & (np.abs(mean[..., 2] - 128.0) < chroma_tol)
    )
    return float(hit.mean())


def frame_report(
    bgr: np.ndarray, *, max_smear: float = DEFAULT_MAX_SMEAR, max_gray: float = DEFAULT_MAX_GRAY
) -> dict:
    """smear_fraction, gray_fraction and the combined corrupt verdict for one BGR frame."""
    s = smear_fraction(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    g = gray_fraction(bgr)
    return {"smear_fraction": s, "gray_fraction": g, "corrupt": s > max_smear or g > max_gray}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find H.264 smear and decoder-gray corruption in saved frames.")
    ap.add_argument("directory")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--last", type=int, default=20)
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
    files = sorted(Path(a.directory).glob(a.glob), key=lambda p: p.stat().st_mtime)[-a.last :]
    read = smeared = grayed = corrupt = 0
    worst_smear = worst_gray = 0.0
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
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
        print("no frames matched" if not files else "no frame could be read")
        return 2
    share = corrupt / read
    print(
        f"frames={read} corrupt={corrupt} ({share:.2f}, max allowed {a.max_corrupt_fraction}) "
        f"smeared={smeared} grayed={grayed} "
        f"worst_smear_fraction={worst_smear:.2f} (max {a.max}) "
        f"worst_gray_fraction={worst_gray:.2f} (max {a.max_gray})"
    )
    return 1 if share > a.max_corrupt_fraction else 0


if __name__ == "__main__":
    sys.exit(main())
