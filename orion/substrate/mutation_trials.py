from __future__ import annotations

from dataclasses import dataclass

from orion.core.schemas.substrate_mutation import MutationProposalV1, MutationTrialV1
from orion.substrate.mutation_scoring import ClassSpecificScorer


@dataclass
class ReplayCorpusRegistry:
    corpus_by_class: dict[str, str]
    baseline_metric_ref_by_class: dict[str, str]

    def ready_for_class(self, mutation_class: str) -> bool:
        return mutation_class in self.corpus_by_class and mutation_class in self.baseline_metric_ref_by_class


@dataclass
class SubstrateTrialRunner:
    scorer: ClassSpecificScorer
    corpus_registry: ReplayCorpusRegistry

    def run_trial(self, *, proposal: MutationProposalV1, measured_metrics: dict[str, float]) -> MutationTrialV1:
        if not self.corpus_registry.ready_for_class(proposal.mutation_class):
            return MutationTrialV1(
                proposal_id=proposal.proposal_id,
                mutation_class=proposal.mutation_class,
                replay_corpus_id=self.corpus_registry.corpus_by_class.get(proposal.mutation_class, "missing"),
                baseline_metric_ref=self.corpus_registry.baseline_metric_ref_by_class.get(proposal.mutation_class, "missing"),
                status="inconclusive",
                metrics=measured_metrics,
                notes=["missing_replay_corpus_or_baseline_metrics"],
            )
        status, notes = self.scorer.evaluate(mutation_class=proposal.mutation_class, metrics=measured_metrics)
        return MutationTrialV1(
            proposal_id=proposal.proposal_id,
            mutation_class=proposal.mutation_class,
            replay_corpus_id=self.corpus_registry.corpus_by_class[proposal.mutation_class],
            baseline_metric_ref=self.corpus_registry.baseline_metric_ref_by_class[proposal.mutation_class],
            status=status,
            metrics=measured_metrics,
            notes=notes,
        )
