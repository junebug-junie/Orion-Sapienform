from __future__ import annotations

from typing import Optional


SUPPORTED_METRICS = {"euclidean", "cosine", "manhattan", "l1", "l2"}


def normalize_metric(value: Optional[str]) -> str:
    cleaned = (value or "euclidean").strip().lower()
    return cleaned or "euclidean"


def validate_metric(metric: str) -> None:
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric '{metric}'")
