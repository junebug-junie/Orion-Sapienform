"""Purpose-conditioned recall (PCR) schema types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RetrievalIntentV1 = Literal[
    "none",
    "continuity",
    "relational",
    "semantic",
    "procedural",
    "open_loop",
    "contradiction",
]

RecallPhaseV1 = Literal["skip", "continuity", "purposeful"]


@dataclass
class PcrChatMemoryV1:
    """Cortex ctx shape for PCR chat memory surfaces."""

    phase: RecallPhaseV1
    retrieval_intent: RetrievalIntentV1 | None
    continuity_digest: str
    belief_digest: str
    memory_digest: str
    skip_reasons: list[str] = field(default_factory=list)
    recall_debug: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PcrChatMemoryV1",
    "RecallPhaseV1",
    "RetrievalIntentV1",
]
