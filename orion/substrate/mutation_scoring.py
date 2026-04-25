from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationClassV1, MutationTrialStatusV1
from orion.substrate.mutation_contracts import CONTRACTS, metric_passed


@dataclass(frozen=True)
class ClassSpecificScorer:
    def evaluate(self, *, mutation_class: MutationClassV1, metrics: dict[str, float]) -> tuple[MutationTrialStatusV1, list[str]]:
        contract = CONTRACTS[mutation_class]
        notes: list[str] = []
        passed = True
        for metric in contract.evaluation_metrics:
            if not metric_passed(metrics, metric):
                passed = False
                notes.append(f"metric_failed:{metric}")
        if passed:
            notes.append("class_specific_metrics_passed")
            return "passed", notes
        if any(key in metrics for key in contract.evaluation_metrics):
            return "failed", notes
        return "inconclusive", ["missing_class_metrics"]
