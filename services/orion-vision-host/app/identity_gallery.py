"""One-subject face-identity gallery: load, match, and (enrollment-only) save.

docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-situated-
observation-design.md section 4's non-negotiables, each enforced here:

- **One enrolled subject. Gallery does not grow.** `save_gallery_embedding`
  exists only for `scripts/enroll_identity_face.py` (a human-run CLI tool)
  to call -- nothing in the request-handling path (`runner.py`'s
  `_run_identity_face`) ever calls it. There is no code path that lets a
  live vision task write a new gallery entry.
- **Non-matches are never stored.** This module never persists a query
  embedding, matched or not -- `match_embedding` takes an embedding already
  computed by the caller, compares it in memory, and returns a plain dict.
  The embedding itself is never written to disk or returned to the caller.
- **`unsure` must be common.** `classify_similarity` has three bands, not
  two -- a binary match/no-match would let a low-confidence guess dress up
  as a real answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _gallery_path(gallery_dir: str, subject: str) -> Path:
    safe_subject = "".join(c for c in subject.lower() if c.isalnum() or c in ("-", "_")) or "subject"
    return Path(gallery_dir) / f"{safe_subject}.json"


def load_gallery_embedding(gallery_dir: str, subject: str) -> Optional[np.ndarray]:
    """Returns the enrolled mean embedding for `subject`, or None if nothing
    has been enrolled yet. Never raises on a missing/corrupt file -- an
    unenrolled gallery is an expected, common state (this feature ships
    with zero real photos of anyone), not an error.
    """
    path = _gallery_path(gallery_dir, subject)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        vec = np.asarray(data["embedding"], dtype=np.float32)
    except Exception:
        return None
    if vec.ndim != 1 or vec.size == 0:
        return None
    return vec


def save_gallery_embedding(
    gallery_dir: str, subject: str, embedding: np.ndarray, *, sample_count: int
) -> Path:
    """Enrollment-only write -- see this module's own docstring. Only ever
    called from scripts/enroll_identity_face.py, a human-run CLI tool, not
    from any live request path.
    """
    path = _gallery_path(gallery_dir, subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": subject,
        "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
        "sample_count": int(sample_count),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def classify_similarity(
    similarity: float, *, match_threshold: float, probable_threshold: float
) -> str:
    """Three bands, not a binary match/no-match -- design doc: "`unsure`
    must be common. A ceiling camera yields bad angles, backlighting, and
    the back of someone's head. If `unsure` is rare in live data, the
    threshold is lying and the whole thing is miscalibrated."

    - similarity >= probable_threshold -> "probable"
    - match_threshold <= similarity < probable_threshold -> "possible"
    - similarity < match_threshold -> "unsure"
    """
    if similarity >= probable_threshold:
        return "probable"
    if similarity >= match_threshold:
        return "possible"
    return "unsure"


def match_embedding(
    embedding: np.ndarray,
    gallery_embedding: Optional[np.ndarray],
    *,
    subject: str,
    match_threshold: float,
    probable_threshold: float,
) -> Dict[str, Any]:
    """Compares one already-computed query embedding against the enrolled
    gallery entry and returns a hypothesis dict -- never a label. `subject`
    in the return is `"unknown"` for anything below `match_threshold`,
    matching the design doc's exact contract:
    ``{"subject": "juniper", "similarity": 0.61, "state": "probable"}``.
    """
    if gallery_embedding is None:
        return {"subject": "unknown", "similarity": None, "state": "unsure", "reason": "not_enrolled"}
    similarity = cosine_similarity(embedding, gallery_embedding)
    state = classify_similarity(
        similarity, match_threshold=match_threshold, probable_threshold=probable_threshold
    )
    matched_subject = subject if state != "unsure" else "unknown"
    return {"subject": matched_subject, "similarity": round(similarity, 4), "state": state}
