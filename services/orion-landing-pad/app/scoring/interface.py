from __future__ import annotations

from dataclasses import dataclass

from orion.schemas.pad import PadEventV1


@dataclass
class ScoreResult:
    salience: float
    novelty: float
    confidence: float
    reason: str | None = None


class Scorer:
    def score(self, event: PadEventV1) -> ScoreResult:  # pragma: no cover - interface
        raise NotImplementedError
