from __future__ import annotations

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.substrate.mutation_decision import DecisionEngine, unpromoted_goal_blocks_execution
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner


def _goal_execution_proposal(*, proposal_status: str) -> MutationProposalV1:
    # Carrier class history: "routing_threshold_patch" (retired 2026-09-05),
    # "graph_consolidation_param_patch" (retired 2026-10-16), then
    # "prediction_expectation_patch" (retired 2026-11-03). Retired classes are
    # rejected on mutation_class_retired before the goal-execution check this
    # test is about, so a non-retired class carries it now.
    return MutationProposalV1(
        lane="operational",
        mutation_class="field_topology_weight_patch",
        risk_tier="low",
        target_surface="field_topology",
        anchor_scope="orion",
        subject_ref="entity:orion",
        source_pressure_id="pressure-1",
        evidence_refs=["telemetry:field_topology:1"],
        source_signal_ids=["signal-1"],
        notes=[f"autonomy_goal_execute:goal-abc", f"autonomy_goal_proposal_status={proposal_status}"],
        patch=MutationPatchV1(
            mutation_class="field_topology_weight_patch",
            target_surface="field_topology",
            target_ref="field_topology",
            patch={"edge_weight_delta": 0.1},
            rollback_payload={"edge_weight_delta": 0.0},
        ),
    )


def test_unpromoted_goal_blocks_execution_helper() -> None:
    assert unpromoted_goal_blocks_execution(goal_proposal_status="proposed") is True
    assert unpromoted_goal_blocks_execution(goal_proposal_status="active") is True
    assert unpromoted_goal_blocks_execution(goal_proposal_status="planned") is False
    assert unpromoted_goal_blocks_execution(goal_proposal_status="executing") is False


def _run_trial(proposal: MutationProposalV1):
    return SubstrateTrialRunner(
        scorer=ClassSpecificScorer(),
        corpus_registry=ReplayCorpusRegistry(
            corpus_by_class={proposal.mutation_class: "corpus-v1"},
            baseline_metric_ref_by_class={proposal.mutation_class: "baseline-v1"},
        ),
    ).run_trial(proposal=proposal, measured_metrics={"divergence_delta": 0.1})


def test_mutation_decision_rejects_unpromoted_goal_execution() -> None:
    proposal = _goal_execution_proposal(proposal_status="proposed")
    trial = _run_trial(proposal)
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action == "reject"
    assert decision.reason == "unpromoted_goal_execution_blocked"


def test_mutation_decision_allows_promoted_goal_execution_to_continue() -> None:
    proposal = _goal_execution_proposal(proposal_status="planned")
    trial = _run_trial(proposal)
    decision = DecisionEngine().decide(
        proposal=proposal,
        trial=trial,
        has_replay_and_baseline=True,
        active_surface_exists=False,
    )
    assert decision.action != "reject"
    assert decision.reason != "unpromoted_goal_execution_blocked"
