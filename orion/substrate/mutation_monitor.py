from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from orion.core.schemas.substrate_mutation import MutationAdoptionV1, MutationRollbackV1


@dataclass(frozen=True)
class PostAdoptionMonitor:
    regression_threshold: float = -0.05

    def should_rollback(self, *, delta_score: float) -> bool:
        return delta_score <= self.regression_threshold

    def build_rollback(self, *, adoption: MutationAdoptionV1, reason: str) -> MutationRollbackV1:
        return MutationRollbackV1(
            adoption_id=adoption.adoption_id,
            proposal_id=adoption.proposal_id,
            reason=reason,
            payload=dict(adoption.rollback_payload),
            created_at=datetime.now(timezone.utc),
        )
