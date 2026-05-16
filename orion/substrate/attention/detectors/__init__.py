from __future__ import annotations

from .autonomy import AutonomySignalDetector
from .base import AttentionSignalDetector
from .concept_induction import ConceptInductionSignalDetector
from .current_turn import CurrentTurnSignalDetector
from .situation import SituationSignalDetector


def default_attention_detectors() -> list[AttentionSignalDetector]:
    return [
        CurrentTurnSignalDetector(),
        AutonomySignalDetector(),
        ConceptInductionSignalDetector(),
        SituationSignalDetector(),
    ]


__all__ = [
    "AttentionSignalDetector",
    "AutonomySignalDetector",
    "ConceptInductionSignalDetector",
    "CurrentTurnSignalDetector",
    "SituationSignalDetector",
    "default_attention_detectors",
]
