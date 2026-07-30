from __future__ import annotations

from .base import AttentionSignalDetector
from .concept_induction import ConceptInductionSignalDetector
from .current_turn import CurrentTurnSignalDetector
from .situation import SituationSignalDetector


def default_attention_detectors() -> list[AttentionSignalDetector]:
    return [
        CurrentTurnSignalDetector(),
        ConceptInductionSignalDetector(),
        SituationSignalDetector(),
    ]


__all__ = [
    "AttentionSignalDetector",
    "ConceptInductionSignalDetector",
    "CurrentTurnSignalDetector",
    "SituationSignalDetector",
    "default_attention_detectors",
]
