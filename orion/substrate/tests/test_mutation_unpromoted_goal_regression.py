from __future__ import annotations

from orion.core.schemas.substrate_mutation import MutationPatchV1, MutationProposalV1
from orion.substrate.mutation_decision import DecisionEngine, unpromoted_goal_blocks_execution
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner


def _goal_execution_proposal(*, proposal_status: str) -> MutationProposalV1:
    return MutationProposalV1(
        lane="operational",
        mutation_class="routing_threshold_patch",
        risk_tier="low",
        target_surface="routing",
        anchor_scope="orion",
        subject_ref="entity:orion",
        source_pressure_id="pressure-1",
        evidence_refs=["telemetry:1"],
        source_signal_ids=["signal-1"],
        notes=[f"autonomy_goal_execute:goal-abc", f"autonomy_goal_proposal_status={proposal_status}"],
        patch=MutationPatchV1(
            mutation_class="routing_threshold_patch",
            target_surface="routing",
            target_ref="routing",
            patch={"autonomy_route_threshold": 0.5},
            rollback_payload={"autonomy_route_threshold": 0.4},
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
    ).run_trial(proposal=proposal, measured_metrics={"success_rate_delta": 0.1, "latency_ms_delta": 0.0})


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
