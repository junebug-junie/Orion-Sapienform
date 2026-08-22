"""Haar-cascade face-crop extraction -- substitute for real OpenFace.

AffectGPT's only released checkpoint is trained on OpenFace-extracted,
similarity-aligned face crops (see `func_compress_openface_into_npy` in the
upstream repo's toolkit/utils/functions.py). Real OpenFace could not be stood
up on this hardware: the only reachable Docker image (algebr/openface, built
2018 against OpenCV 3.4) segfaults (SIGSEGV) during actual per-frame
detection even after its MTCNN->HOG-SVM fallback engages cleanly, and
upstream (TadasBaltrusaitis/OpenFace) ships no Dockerfile to build from
source (confirmed 404 on master, 2026-08-22).

This module is a deliberate, documented approximation: per-frame largest-face
Haar-cascade detection + fixed-margin crop + resize, carrying the last known
box forward across any frame where detection drops out (never fabricates
coordinates -- only ever carries forward a real prior detection). It is NOT
bit-identical to OpenFace's CE-CLM landmark-based similarity alignment (no
eye/rotation normalization, different landmark model) -- but it puts the
model's real, trained-in-face_or_frame mode in front of an actual face
region instead of a raw scene frame, which is what the released checkpoint
requires to produce meaningful output at all (there is no raw-frame
checkpoint available, see settings.py).

Output array is stored BGR (OpenCV's native decode order, no color
conversion) to match what an OpenFace-produced .npy would contain via
`cv2.imread` -- `load_face()` in AffectGPT's own video_processor.py does no
BGR->RGB swap, so this matches what the released checkpoint actually saw
during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceExtractionResult:
    faces: np.ndarray  # [T, 224, 224, 3] uint8, BGR
    frames_total: int
    frames_detected: int
    frames_carried_forward: int
    frames_no_face_fallback_full_frame: int

    @property
    def detection_rate(self) -> float:
        if self.frames_total == 0:
            return 0.0
        return self.frames_detected / self.frames_total

    def as_meta(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_detected": self.frames_detected,
            "frames_carried_forward": self.frames_carried_forward,
            "frames_no_face_fallback_full_frame": self.frames_no_face_fallback_full_frame,
            "detection_rate": round(self.detection_rate, 4),
        }


_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def extract_face_crops(
    video_path: str,
    *,
    margin: float = 0.25,
    min_size_px: int = 60,
    crop_size: int = 224,
) -> FaceExtractionResult:
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError(f"failed to load Haar cascade from {_CASCADE_PATH!r}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path!r}")

    crops: list[np.ndarray] = []
    last_box: tuple[int, int, int, int] | None = None
    n_detected = 0
    n_carried = 0
    n_no_face_ever = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size_px, min_size_px)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad_w, pad_h = int(w * margin), int(h * margin)
                x0 = max(0, x - pad_w)
                y0 = max(0, y - pad_h)
                x1 = min(frame.shape[1], x + w + pad_w)
                y1 = min(frame.shape[0], y + h + pad_h)
                last_box = (x0, y0, x1, y1)
                n_detected += 1
            elif last_box is not None:
                n_carried += 1
            else:
                n_no_face_ever += 1

            if last_box is not None:
                x0, y0, x1, y1 = last_box
                crop = frame[y0:y1, x0:x1]
            else:
                crop = frame  # no detection anywhere in the clip so far
            crop = cv2.resize(crop, (crop_size, crop_size))
            crops.append(crop)
    finally:
        cap.release()

    if not crops:
        raise RuntimeError(f"video had zero readable frames: {video_path!r}")

    arr = np.array(crops, dtype=np.uint8)
    return FaceExtractionResult(
        faces=arr,
        frames_total=len(crops),
        frames_detected=n_detected,
        frames_carried_forward=n_carried,
        frames_no_face_fallback_full_frame=n_no_face_ever,
    )
