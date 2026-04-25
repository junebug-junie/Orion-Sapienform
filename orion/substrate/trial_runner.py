from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationProposalV1, MutationTrialV1
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner


@dataclass
class OrionSubstrateTrialRunner:
    """Deployable-facing wrapper for the V2.1 trial runner."""

    runner: SubstrateTrialRunner

    @classmethod
    def with_defaults(cls) -> "OrionSubstrateTrialRunner":
        registry = ReplayCorpusRegistry(
            corpus_by_class={},
            baseline_metric_ref_by_class={},
        )
        return cls(runner=SubstrateTrialRunner(scorer=ClassSpecificScorer(), corpus_registry=registry))

    def execute(self, *, proposal: MutationProposalV1, measured_metrics: dict[str, float]) -> MutationTrialV1:
        return self.runner.run_trial(proposal=proposal, measured_metrics=measured_metrics)
