"""Who is usually in front of each camera: prior probabilities per stream.

``config/person_model/camera_priors.example.yaml`` (committed, redacted) and its
gitignored ``camera_priors.local.yaml`` (real names, read first; ``local_first``).
Read by orion-sql-writer's person model: on a camera with a prior, when no album
match is confident, the guess is the most likely person on that camera
(2026-11-04, the office camera: "99% of the time it is me in there").
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from orion.person_model.person import local_first

DEFAULT_PRIORS_PATH = Path(__file__).resolve().parents[2] / "config" / "person_model" / "camera_priors.example.yaml"


def load_camera_priors(path: Optional[str | Path] = None) -> Dict[str, Dict[str, float]]:
    """stream_id -> person name -> prior probability. Values need not sum to 1
    (the rest is "someone else") but may not exceed it. Non-positive values and
    empty names drop; a stream with nothing left has no prior. Missing file:
    no priors anywhere."""
    p = Path(path) if path else local_first(Path(os.getenv("VISION_CAMERA_PRIORS_PATH") or DEFAULT_PRIORS_PATH))
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: Dict[str, Dict[str, float]] = {}
    for stream_id, block in (raw.get("streams") or {}).items():
        prior = {str(k).strip(): float(v) for k, v in (block or {}).items() if str(k).strip() and float(v) > 0}
        folded = [k.casefold() for k in prior]
        if len(set(folded)) != len(folded):
            raise ValueError(f"camera prior for {stream_id} names one person twice (case-insensitive)")
        total = sum(prior.values())
        if total > 1.0 + 1e-9:
            raise ValueError(f"camera prior for {stream_id} sums to {total:.3f} > 1")
        if prior:
            out[str(stream_id)] = prior
    return out
