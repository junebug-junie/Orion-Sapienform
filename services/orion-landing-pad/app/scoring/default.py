from __future__ import annotations

from orion.schemas.pad import PadEventV1

from .interface import ScoreResult, Scorer


class DefaultScorer(Scorer):
    TYPE_PRIORS = {
        "anomaly": 0.7,
        "decision": 0.6,
        "intent": 0.6,
        "reflection": 0.4,
        "snapshot": 0.35,
        "metric": 0.25,
        "observation": 0.2,
        "percept": 0.25,
        "memory": 0.3,
        "task_state_change": 0.4,
        "unknown": 0.1,
    }

    def score(self, event: PadEventV1) -> ScoreResult:
        base = self.TYPE_PRIORS.get(event.type, 0.1)
        salience = max(base, event.salience)
        novelty = min(max(event.novelty, 0.0), 1.0)
        confidence = min(max(event.confidence, 0.0), 1.0)

        # Incorporate novelty lightly
        boosted = min(1.0, salience + (0.2 * novelty))
        return ScoreResult(salience=boosted, novelty=novelty, confidence=confidence, reason="default")
