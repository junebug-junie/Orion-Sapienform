from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSpanV1:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    reason: str
    weight: float


@dataclass(frozen=True)
class SignalEvidenceBundleV1:
    spans: tuple[EvidenceSpanV1, ...]
    truncated: bool
    degraded: bool
    notes: tuple[str, ...]
