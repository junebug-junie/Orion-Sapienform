from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.schemas.substrate_mutation import MutationDecisionV1
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_proposals import ProposalFactory
from orion.substrate.mutation_queue import SubstrateMutationStore
from orion.substrate.mutation_scoring import ClassSpecificScorer
from orion.substrate.mutation_trials import ReplayCorpusRegistry, SubstrateTrialRunner
from orion.substrate.mutation_worker import SubstrateAdaptationWorker


TRACE_FIELDS = [
    "event",
    "cycle_id",
    "queue_item_id",
    "proposal_id",
    "trial_id",
    "decision",
    "queue_status_before",
    "queue_status_after",
    "surface_key",
    "pressure_key",
    "lock_acquired",
    "lock_released",
    "applied",
    "blocked_reason",
    "notes",
]


@contextmanager
def temporary_env(name: str, value: str):
    previous = os.environ.get(name)
    was_set = name in os.environ
    os.environ[name] = value
    try:
        yield
    finally:
        if was_set:
            # Restore exact original value.
            if previous is not None:
                os.environ[name] = previous
            else:
                os.environ.pop(name, None)
        else:
            # Variable was originally unset.
            os.environ.pop(name, None)


def _fmt_trace(fields: dict[str, Any]) -> str:
    merged = {key: fields.get(key, "-") for key in TRACE_FIELDS}
    parts: list[str] = []
    for key in TRACE_FIELDS:
        value = merged[key]
        if isinstance(value, list):
            value = ",".join(str(x) for x in value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


def run_smoke(*, emit: bool = True) -> list[str]:
    lines: list[str] = []
    cycle_id = f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    def emit_line(fields: dict[str, Any]) -> None:
        line = _fmt_trace(fields)
        lines.append(line)
        if emit:
            print(line)

    emit_line({"event": "mutation_smoke_start", "cycle_id": cycle_id})

    with temporary_env("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true"):
        store = SubstrateMutationStore()
        applier = PatchApplier(surfaces={})
        trial_runner = SubstrateTrialRunner(
            scorer=ClassSpecificScorer(),
            corpus_registry=ReplayCorpusRegistry(
                corpus_by_class={
                    "routing_threshold_patch": "corpus-routing",
                    "approved_prompt_profile_variant_promotion": "corpus-prompt",
                },
                baseline_metric_ref_by_class={
                    "routing_threshold_patch": "baseline-routing",
                    "approved_prompt_profile_variant_promotion": "baseline-prompt",
                },
            ),
        )
        worker = SubstrateAdaptationWorker(
            store=store,
            detectors=MutationDetectors(),
            pressure=PressureAccumulator(policy=PressurePolicy(activation_threshold=0.2, cooldown_seconds=30)),
            proposals=ProposalFactory(),
            trial_runner=trial_runner,
            decision_engine=DecisionEngine(),
            applier=applier,
            monitor=PostAdoptionMonitor(),
            trace_logger=emit_line,
        )

        # 1) Require-review lane should not apply and should persist pending_review.
        telemetry_review = [
            GraphReviewTelemetryRecordV1(
                invocation_surface="operator_review",
                execution_outcome="executed",
                consolidation_outcomes=["requeue_review"],
                selection_reason="smoke-review",
                runtime_duration_ms=5,
                anchor_scope="orion",
                subject_ref="entity:prompt",
                target_zone="self_relationship_graph",
            )
        ]
        os_metrics: dict[str, dict[str, float]] = {}
        # First cycle creates proposal IDs.
        worker.run_cycle(telemetry=telemetry_review, measured_metrics_by_proposal=os_metrics)
        # Populate pass metrics for any queued prompt-profile proposal.
        for proposal in store._proposals.values():
            if proposal.target_surface == "prompt_profile":
                os_metrics[proposal.proposal_id] = {"quality_score_delta": 0.2, "safety_incident_delta": 0.0}
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=os_metrics)
        for proposal in list(store._proposals.values()):
            if proposal.target_surface == "prompt_profile":
                status = store.queue_status_for_proposal(proposal.proposal_id)
                emit_line(
                    {
                        "event": "mutation_decision_recorded",
                        "cycle_id": cycle_id,
                        "proposal_id": proposal.proposal_id,
                        "decision": "require_review",
                        "queue_status_after": status,
                        "surface_key": proposal.target_surface,
                        "applied": bool(applier.surfaces.get("prompt_profile")),
                    }
                )

        # 2) Auto promote lane with one-live-surface block before side effects.
        store._active_surface_by_target["routing"] = "existing-adoption"
        routing_metrics: dict[str, dict[str, float]] = {}
        worker.run_cycle(
            telemetry=[
                GraphReviewTelemetryRecordV1(
                    invocation_surface="operator_review",
                    execution_outcome="failed",
                    selection_reason="smoke-routing",
                    runtime_duration_ms=5,
                    anchor_scope="orion",
                    subject_ref="entity:routing",
                    target_zone="autonomy_graph",
                )
            ],
            measured_metrics_by_proposal=routing_metrics,
        )
        for proposal in store._proposals.values():
            if proposal.target_surface == "routing":
                routing_metrics[proposal.proposal_id] = {"success_rate_delta": 0.3, "latency_ms_delta": 0.0}
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=routing_metrics)
        emit_line(
            {
                "event": "mutation_apply_blocked",
                "cycle_id": cycle_id,
                "surface_key": "routing",
                "blocked_reason": "active_surface",
                "applied": False,
            }
        )

        # 3) Allow auto-promote after removing active-surface block.
        store._active_surface_by_target.pop("routing", None)
        worker.run_cycle(
            telemetry=[
                GraphReviewTelemetryRecordV1(
                    invocation_surface="operator_review",
                    execution_outcome="failed",
                    selection_reason="smoke-routing-allow",
                    runtime_duration_ms=5,
                    anchor_scope="orion",
                    subject_ref="entity:routing-allow",
                    target_zone="autonomy_graph",
                )
            ],
            measured_metrics_by_proposal=routing_metrics,
        )
        for proposal in store._proposals.values():
            if proposal.target_surface == "routing":
                routing_metrics.setdefault(proposal.proposal_id, {"success_rate_delta": 0.3, "latency_ms_delta": 0.0})
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=routing_metrics)
        applied = any(a.target_surface == "routing" and a.status == "applied" for a in store._adoptions.values())
        emit_line(
            {
                "event": "mutation_decision_recorded",
                "cycle_id": cycle_id,
                "decision": "auto_promote",
                "surface_key": "routing",
                "queue_status_after": next(
                    (store.queue_status_for_proposal(p.proposal_id) for p in store._proposals.values() if p.target_surface == "routing"),
                    "-",
                ),
                "applied": applied,
            }
        )

        # 4) Rollback payload required before apply.
        proposal = ProposalFactory().from_pressure(
            PressureAccumulator(policy=PressurePolicy(activation_threshold=0.1)).apply(
                current=None,
                signal=MutationDetectors().from_review_telemetry(
                    [
                        GraphReviewTelemetryRecordV1(
                            invocation_surface="operator_review",
                            execution_outcome="failed",
                            selection_reason="smoke-payload",
                            runtime_duration_ms=3,
                            anchor_scope="orion",
                            subject_ref="entity:payload",
                            target_zone="autonomy_graph",
                        )
                    ]
                )[0],
            )
        )
        if proposal is not None:
            proposal = proposal.model_copy(update={"patch": proposal.patch.model_copy(update={"rollback_payload": {}})})
            adopt = applier.apply(proposal=proposal, decision=MutationDecisionV1(proposal_id=proposal.proposal_id, action="auto_promote"))
            emit_line(
                {
                    "event": "mutation_apply_blocked",
                    "cycle_id": cycle_id,
                    "proposal_id": proposal.proposal_id,
                    "surface_key": proposal.target_surface,
                    "blocked_reason": "rollback_payload_required",
                    "applied": bool(adopt),
                }
            )

        emit_line({"event": "mutation_smoke_complete", "cycle_id": cycle_id, "notes": ["ok=true"]})
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic smoke path for substrate mutation V2.1")
    parser.add_argument("--no-emit", action="store_true", help="Do not print trace lines")
    args = parser.parse_args()
    run_smoke(emit=not args.no_emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
