"""Deterministic pre-LLM gate: interpret only on host evidence transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from orion.schemas.vision import VisionWindowPayload


@dataclass(frozen=True)
class EvidenceSnapshot:
    hard_labels: frozenset[str]
    person_present: bool


@dataclass(frozen=True)
class EvidenceTransitionDecision:
    interpret: bool
    reason: str = ""


def stream_key_from_window(window: VisionWindowPayload) -> str:
    for value in (window.stream_id, window.camera_id):
        if value is not None and str(value).strip():
            return str(value).strip()
    return "default"


def _hard_labels_from_window(window: VisionWindowPayload) -> frozenset[str]:
    summary = window.summary or {}
    evidence = summary.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    raw = evidence.get("hard_labels") or []
    return frozenset(str(label).strip().lower() for label in raw if str(label).strip())


def snapshot_from_window(window: VisionWindowPayload) -> EvidenceSnapshot:
    hard_labels = _hard_labels_from_window(window)
    return EvidenceSnapshot(
        hard_labels=hard_labels,
        person_present="person" in hard_labels,
    )


def _labels_summary(labels: frozenset[str]) -> str:
    if not labels:
        return "-"
    return ",".join(sorted(labels))


class EvidenceTransitionTracker:
    def __init__(self) -> None:
        self._by_stream: dict[str, dict[str, Any]] = {}

    def evaluate(
        self,
        *,
        stream_key: str,
        snapshot: EvidenceSnapshot,
        now: float,
        max_refresh_sec: float,
    ) -> EvidenceTransitionDecision:
        row = self._by_stream.get(stream_key)
        if row is None:
            return EvidenceTransitionDecision(interpret=True, reason="first_window")

        prev_labels = frozenset(row.get("hard_labels") or [])
        prev_person = bool(row.get("person_present"))
        last_interpret_at = float(row.get("last_interpret_at") or 0.0)

        if snapshot.hard_labels != prev_labels:
            if snapshot.person_present and not prev_person:
                return EvidenceTransitionDecision(interpret=True, reason="person_entered")
            if prev_person and not snapshot.person_present:
                return EvidenceTransitionDecision(interpret=True, reason="person_exited")
            return EvidenceTransitionDecision(interpret=True, reason="labels_changed")

        if max_refresh_sec > 0 and (now - last_interpret_at) >= max_refresh_sec:
            return EvidenceTransitionDecision(interpret=True, reason="refresh_ttl")

        return EvidenceTransitionDecision(interpret=False, reason="stable_scene")

    def record_interpretation(
        self,
        *,
        stream_key: str,
        snapshot: EvidenceSnapshot,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else now
        self._by_stream[stream_key] = {
            "hard_labels": snapshot.hard_labels,
            "person_present": snapshot.person_present,
            "last_interpret_at": ts,
        }

    @staticmethod
    def labels_summary(snapshot: EvidenceSnapshot) -> str:
        return _labels_summary(snapshot.hard_labels)
