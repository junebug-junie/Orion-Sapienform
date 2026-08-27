"""Pick the few frames a VL model actually sees, and measure how good they were.

Replaces the role `orion-affectgpt-worker/app/face_extract.py` played for the
retired AffectGPT backend, and deliberately does NOT import from it -- that is
a different service's internals (CLAUDE.md section 5). The Haar parameters and
the BGR/margin conventions are copied verbatim on purpose, so `detection_rate`
means the same thing across both backends and rows in
`juniper_multimodal_affect_log` stay comparable across the 2026-08-26 cutover.

**Two jobs, deliberately split.**

1. *Which frames to send.* A VL model gets a handful of stills, not a 231-frame
   tensor, so the choice of stills is load-bearing in a way it never was for
   AffectGPT. Frames where a face was actually detected are preferred; the even
   spread across the clip is what carries the temporal information (an
   expression settling vs tightening) that a single frame cannot.

2. *Whether to believe the answer.* `detection_rate` over ALL frames -- not just
   the sampled ones -- becomes the quality gate on the resulting read. This is
   the check the replaced pipeline never had: on 2026-08-26 a live capture
   scored `detection_rate=0.052` (170 of 231 frames were raw scene frames with
   no face in them at all) and the model still returned a confident
   "anger, frustration, or sadness". Measuring the input is the only way that
   failure is visible from the outside.

**Full frames are sent, not face crops.** Deliberate reversal of the AffectGPT
approach, which cropped because its checkpoint was trained on OpenFace crops. A
general VL model does materially better with the whole frame: the live read that
motivated this module cited posture, gaze direction and head tilt, none of which
survive a tight face crop. The Haar box is still computed -- it just gates
trust rather than defining the pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Copied from face_extract.py so detection_rate stays comparable. See docstring.
_MIN_FACE_PX = 60
_SCALE_FACTOR = 1.1
_MIN_NEIGHBORS = 5


@dataclass(frozen=True)
class SampledFrame:
    """One JPEG-encoded still, plus where it came from."""

    jpeg: bytes
    index: int
    width: int
    height: int
    face_detected: bool


@dataclass
class FrameSampleResult:
    frames: list[SampledFrame]
    frames_total: int
    frames_detected: int

    @property
    def detection_rate(self) -> float:
        if self.frames_total == 0:
            return 0.0
        return self.frames_detected / self.frames_total

    def as_meta(self) -> dict:
        """Same key names face_extract.py's as_meta() emits, minus the two
        crop-specific counters that have no meaning when nothing is cropped.
        A consumer reading `detection_rate` off either backend's event gets
        the same quantity measured the same way."""
        return {
            "frames_total": self.frames_total,
            "frames_detected": self.frames_detected,
            "detection_rate": round(self.detection_rate, 4),
            "frames_sampled": len(self.frames),
        }


class FrameSampleError(RuntimeError):
    """The clip could not be turned into model input."""


def _evenly_spaced(candidates: list[int], want: int) -> list[int]:
    """Pick `want` items spread across `candidates`, preserving order.

    Not `random.sample` and not "the first N": both would cluster or jitter.
    An even spread is what makes the set a *sequence* the model can read
    change out of, which is the entire reason more than one frame is sent.
    """
    if want <= 0 or not candidates:
        return []
    if len(candidates) <= want:
        return list(candidates)
    step = (len(candidates) - 1) / (want - 1) if want > 1 else 0.0
    picked: list[int] = []
    for i in range(want):
        idx = candidates[int(round(i * step))]
        if idx not in picked:
            picked.append(idx)
    return picked


def _scan(video_path: str, cascade) -> tuple[int, list[int]]:
    """Pass 1: Haar-scan every frame, retaining NONE of them.

    Returns (frames_total, indices_with_a_detection). Deliberately does not
    keep the decoded frames -- see sample_frames' docstring for the memory
    arithmetic that makes that the whole point of splitting the passes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameSampleError(f"could not open video: {video_path!r}")
    total = 0
    detected: list[int] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=_SCALE_FACTOR,
                minNeighbors=_MIN_NEIGHBORS,
                minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
            )
            if len(faces) > 0:
                detected.append(total)
            total += 1
    finally:
        cap.release()
    return total, detected


def _collect(video_path: str, wanted: list[int]) -> dict[int, "np.ndarray"]:
    """Pass 2: re-decode, keeping ONLY the frames at `wanted` indices.

    Sequential re-read rather than `cap.set(CAP_PROP_POS_FRAMES, ...)`: seeking
    a compressed mp4 by frame index lands on the nearest keyframe with several
    backends, which would silently return a frame OTHER than the one Haar
    judged. A second decode pass is cheap next to the Haar scan and is exact.
    """
    remaining = set(wanted)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameSampleError(f"could not re-open video: {video_path!r}")
    out: dict[int, "np.ndarray"] = {}
    idx = 0
    try:
        while remaining:
            ret, frame = cap.read()
            if not ret:
                break
            if idx in remaining:
                out[idx] = frame
                remaining.discard(idx)
            idx += 1
    finally:
        cap.release()
    return out


def sample_frames(
    video_path: str,
    *,
    max_frames: int,
    jpeg_quality: int = 85,
) -> FrameSampleResult:
    """Decode the clip twice, Haar-scan every frame, return `max_frames` stills.

    Raises FrameSampleError on an unreadable/empty clip -- a caller must treat
    that as a real failure (`ok=False`), never as an empty-but-fine read. The
    replaced pipeline's habit of reporting `ok=True` over meaningless input is
    the specific thing this module exists to stop.

    **Two passes, not one, because one pass is an unbounded memory hazard**
    (review finding, 2026-08-26). Retaining every decoded frame so the chosen
    indices can be picked at the end costs `frames_total x frame_bytes`, and
    nothing here bounds either term: `RETINA_CLIP_WIDTH`/`HEIGHT` default to
    None (the device's own default), so a 1080p webcam is ~6.2MB/frame, and the
    live capture this module was written against had 231 frames -- ~1.4GB of
    peak RSS, on a path that fires on every spoken chat turn. Only `max_frames`
    frames are ever encoded, so pass 1 keeps indices and pass 2 keeps pixels;
    peak retention is `max_frames` frames regardless of clip length or
    resolution.
    """
    if max_frames <= 0:
        raise FrameSampleError(f"max_frames must be >= 1, got {max_frames}")

    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        raise FrameSampleError(f"failed to load Haar cascade from {_CASCADE_PATH!r}")

    frames_total, detected_idx = _scan(video_path, cascade)
    if frames_total == 0:
        raise FrameSampleError(f"video had zero readable frames: {video_path!r}")

    # Prefer frames with a real detection; fall back to the whole clip when
    # Haar found nothing anywhere. The fallback still SENDS frames -- a low
    # detection_rate is reported to the caller as a trust signal rather than
    # silently substituting a refusal here, because Haar's frontal-face
    # cascade misses profile/tilted faces a VL model reads without trouble
    # (observed live: 100% detection and 5.2% detection on two captures of
    # the same person two minutes apart).
    pool = detected_idx if detected_idx else list(range(frames_total))
    chosen = _evenly_spaced(pool, max_frames)

    collected = _collect(video_path, chosen)
    detected_set = set(detected_idx)
    sampled: list[SampledFrame] = []
    for idx in chosen:
        frame = collected.get(idx)
        if frame is None:
            # Pass 2 read fewer frames than pass 1 -- a truncated/racing file.
            # A real failure, not something to quietly sample around.
            raise FrameSampleError(
                f"frame {idx} was readable during the scan but not on re-read: {video_path!r}"
            )
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise FrameSampleError(f"cv2.imencode failed for frame {idx}")
        height, width = frame.shape[:2]
        sampled.append(
            SampledFrame(
                jpeg=buf.tobytes(),
                index=idx,
                width=int(width),
                height=int(height),
                face_detected=idx in detected_set,
            )
        )

    return FrameSampleResult(
        frames=sampled,
        frames_total=frames_total,
        frames_detected=len(detected_idx),
    )
