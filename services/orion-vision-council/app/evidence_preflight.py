"""Deterministic pre-LLM gate: skip council inference when window evidence is unchanged."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from orion.schemas.vision import VisionWindowPayload


@dataclass(frozen=True)
class EvidenceSkipDecision:
    skip: bool
    reason: str = ""


def stream_key_from_window(window: VisionWindowPayload) -> str:
    for value in (window.stream_id, window.camera_id):
        if value is not None and str(value).strip():
            return str(value).strip()
    return "default"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evidence_material(window: VisionWindowPayload) -> dict[str, Any]:
    """Fields council grounding cares about; excludes volatile window ids and timestamps."""
    summary = window.summary or {}
    evidence = summary.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    object_counts = summary.get("object_counts") or {}
    if not isinstance(object_counts, dict):
        object_counts = {}
    return {
        "evidence": {
            "hard_labels": sorted(str(x) for x in (evidence.get("hard_labels") or [])),
            "soft_labels": sorted(str(x) for x in (evidence.get("soft_labels") or [])),
            "host_person_hits": _safe_int(evidence.get("host_person_hits")),
            "caption_count": _safe_int(evidence.get("caption_count")),
        },
        "object_counts": {str(k): _safe_int(v) for k, v in sorted(object_counts.items())},
        "detection_count": _safe_int(summary.get("detection_count")),
    }


def evidence_fingerprint(window: VisionWindowPayload) -> str:
    payload = json.dumps(evidence_material(window), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvidenceSkipTracker:
    def __init__(self) -> None:
        self._by_stream: dict[str, dict[str, Any]] = {}

    def evaluate(
        self,
        *,
        stream_key: str,
        fingerprint: str,
        now: float,
        max_skip_sec: float,
    ) -> EvidenceSkipDecision:
        row = self._by_stream.get(stream_key)
        if row is None:
            return EvidenceSkipDecision(skip=False, reason="first_window")
        if row.get("fingerprint") != fingerprint:
            return EvidenceSkipDecision(skip=False, reason="evidence_changed")
        last_llm_at = float(row.get("last_llm_at") or 0.0)
        if max_skip_sec > 0 and (now - last_llm_at) >= max_skip_sec:
            return EvidenceSkipDecision(skip=False, reason="max_skip_sec_elapsed")
        return EvidenceSkipDecision(skip=True, reason="evidence_unchanged")

    def record_llm(self, *, stream_key: str, fingerprint: str, now: float | None = None) -> None:
        ts = time.time() if now is None else now
        self._by_stream[stream_key] = {"fingerprint": fingerprint, "last_llm_at": ts}
