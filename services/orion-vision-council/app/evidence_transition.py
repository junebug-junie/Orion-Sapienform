"""Deterministic pre-LLM gate: interpret only on host evidence transitions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from orion.schemas.vision import VisionWindowPayload


@dataclass(frozen=True)
class EvidenceSnapshot:
    hard_labels: frozenset[str]
    person_present: bool
    # Labels whose every box, the last time the label was seen, was a
    # stationary fixture (a parked car, a prop). From orion-vision-window
    # (summary.evidence.fixture_only_labels). Not part of what "changed":
    # compare=False keeps two snapshots with the same labels equal.
    fixture_only_labels: frozenset[str] = field(default=frozenset(), compare=False)


@dataclass(frozen=True)
class EvidenceTransitionDecision:
    interpret: bool
    reason: str = ""


def stream_key_from_window(window: VisionWindowPayload) -> str:
    for value in (window.stream_id, window.camera_id):
        if value is not None and str(value).strip():
            return str(value).strip()
    return "default"


def _hard_labels_from_evidence(evidence: dict) -> frozenset[str]:
    raw = evidence.get("hard_labels") or []
    return frozenset(str(label).strip().lower() for label in raw if str(label).strip())


def _labels_for_gate(window: VisionWindowPayload) -> frozenset[str]:
    summary = window.summary or {}
    evidence = summary.get("evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    believed = evidence.get("believed_hard_labels")
    belief_meta = evidence.get("belief") or {}
    if (
        isinstance(believed, list)
        and believed
        and isinstance(belief_meta, dict)
        and belief_meta.get("schema") == "scene_belief.v1"
    ):
        return frozenset(str(label).strip().lower() for label in believed if str(label).strip())
    return _hard_labels_from_evidence(evidence)


def _fixture_only_labels(window: VisionWindowPayload) -> frozenset[str]:
    evidence = (window.summary or {}).get("evidence") or {}
    raw = evidence.get("fixture_only_labels") if isinstance(evidence, dict) else None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(label).strip().lower() for label in raw if str(label).strip())


def snapshot_from_window(window: VisionWindowPayload) -> EvidenceSnapshot:
    hard_labels = _labels_for_gate(window)
    return EvidenceSnapshot(
        hard_labels=hard_labels,
        person_present="person" in hard_labels,
        fixture_only_labels=_fixture_only_labels(window),
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
        if row is not None and row.get("interpret_in_flight"):
            return EvidenceTransitionDecision(interpret=False, reason="interpret_in_flight")

        if row is None:
            return EvidenceTransitionDecision(interpret=True, reason="first_window")

        prev_labels = frozenset(row.get("hard_labels") or [])
        prev_person = bool(row.get("person_present"))
        last_interpret_at = float(row.get("last_interpret_at") or 0.0)

        changed = snapshot.hard_labels ^ prev_labels
        # A change made only of stationary fixtures is the detector losing and
        # re-finding something that did not move (a far parked car at night),
        # not a change in the scene. It does not trigger; the refresh TTL below
        # still does. The remembered label set is left as it was, so a real
        # label arriving later is still compared against the last
        # interpretation (#3387).
        fixture_only_change = bool(changed) and changed <= snapshot.fixture_only_labels
        if changed and not fixture_only_change:
            if snapshot.person_present and not prev_person:
                return EvidenceTransitionDecision(interpret=True, reason="person_entered")
            if prev_person and not snapshot.person_present:
                return EvidenceTransitionDecision(interpret=True, reason="person_exited")
            return EvidenceTransitionDecision(interpret=True, reason="salient_labels_changed")

        if max_refresh_sec > 0 and (now - last_interpret_at) >= max_refresh_sec:
            return EvidenceTransitionDecision(interpret=True, reason="refresh_ttl")

        if fixture_only_change:
            return EvidenceTransitionDecision(interpret=False, reason="fixture_only_change")
        return EvidenceTransitionDecision(interpret=False, reason="stable_scene")

    def begin_interpretation(self, *, stream_key: str) -> None:
        row = dict(self._by_stream.get(stream_key) or {})
        row["interpret_in_flight"] = True
        self._by_stream[stream_key] = row

    def abort_interpretation(self, *, stream_key: str) -> None:
        row = self._by_stream.get(stream_key)
        if row is None:
            return
        row = dict(row)
        row.pop("interpret_in_flight", None)
        if row.get("hard_labels") is None and row.get("last_interpret_at") is None:
            self._by_stream.pop(stream_key, None)
        else:
            self._by_stream[stream_key] = row

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
