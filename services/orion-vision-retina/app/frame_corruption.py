"""H.264 decode-corruption scores for one frame: smear and decoder gray.

The single implementation behind both the offline eval
(``evals/frame_smear.py``, which documents the method and its calibration)
and the live self-heal in ``app/sources.py`` (RtspFrameSource). Lives in
``app/`` because only ``app/`` ships in the image.
"""

from __future__ import annotations

import cv2
import numpy as np

SMEAR_WIDTH = 640
DEFAULT_MAX_SMEAR = 0.05
DEFAULT_MAX_GRAY = 0.4
GRAY_LUMA = 130.0  # FFmpeg concealment Y=128 (limited range) after OpenCV BGR->YCrCb


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
    std_max: float = 0.6,
    luma: float = GRAY_LUMA,
    luma_tol: float = 4.0,
    chroma_tol: float = 4.0,
) -> float:
    """Fraction of `block`x`block` macroblocks filled with decoder mid-gray.

    Takes BGR (or a single-channel image, treated as black and white).
    """
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
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
        & (np.abs(mean[..., 0] - luma) < luma_tol)
        & (np.abs(mean[..., 1] - 128.0) < chroma_tol)
        & (np.abs(mean[..., 2] - 128.0) < chroma_tol)
    )
    return float(hit.mean())


def frame_report(
    bgr: np.ndarray, *, max_smear: float = DEFAULT_MAX_SMEAR, max_gray: float = DEFAULT_MAX_GRAY
) -> dict:
    """smear_fraction, gray_fraction and the combined corrupt verdict for one BGR frame."""
    luma = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr[..., :3], cv2.COLOR_BGR2GRAY)
    s = smear_fraction(luma)
    g = gray_fraction(bgr)
    return {"smear_fraction": s, "gray_fraction": g, "corrupt": s > max_smear or g > max_gray}
