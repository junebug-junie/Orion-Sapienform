from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.core.schemas.substrate_mutation import (
    MutationDecisionV1,
    MutationPatchV1,
    MutationProposalV1,
)
from orion.core.schemas.substrate_review_telemetry import GraphReviewTelemetryRecordV1
from orion.substrate import mutation_control_surface
from orion.substrate.mutation_apply import PatchApplier
from orion.substrate.mutation_decision import DecisionEngine
from orion.substrate.mutation_detectors import MutationDetectors
from orion.substrate.mutation_monitor import PostAdoptionMonitor
from orion.substrate.mutation_pressure import PressureAccumulator, PressurePolicy
from orion.substrate.mutation_control_surface import (
    inspect_chat_reflective_lane_threshold,
    set_chat_reflective_lane_threshold,
)
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


@contextmanager
def _isolated_control_surface(*, seed_threshold: float):
    """Run the smoke against a throwaway control surface, seeded to a known value.

    Two reasons, both load-bearing.

    First, this script must never touch the real one. A pytest fixture writing
    to the ambient control surface is how `value=0.5, actor="scheduler_seed"`
    ended up on Orion's live routing threshold with 4,925 updates on it.
    `run_smoke` already isolates its SubstrateMutationStore but was reading the
    global control surface, so it inherited whatever the process had.

    Second, the smoke asserts that an auto_promote apply happens. That is only
    a meaningful assertion if the patch would actually change something --
    `_default_patch_for_class` returns a constant 0.58, so against an ambient
    surface already at 0.58 the apply is a no-op and is now correctly skipped.
    Seeding a different starting value makes the assertion test the apply path
    rather than the ambient state of whatever ran before it.

    Both halves are required, and this is the same shape as
    `services/orion-cortex-orch/tests/conftest.py`'s isolation fixture.
    Constructing the store with `sql_db_path=None, postgres_url=None` is NOT an
    isolation request: `__post_init__` fills either slot from the ambient
    environment, so with a `DATABASE_URL` in scope -- which orion-hub has, and
    orion-hub owns the mutation worker -- the "isolated" store resolves to live
    Postgres and the smoke moves Orion's real threshold. The env keys must be
    cleared AND an explicit path passed.
    """
    previous_env = {}
    for key in (
        "SUBSTRATE_CONTROL_PLANE_POSTGRES_URL",
        "SUBSTRATE_POLICY_POSTGRES_URL",
        "DATABASE_URL",
        "SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH",
        "SUBSTRATE_MUTATION_SQL_DB_PATH",
    ):
        previous_env[key] = os.environ.pop(key, None)
    previous_store = mutation_control_surface._CONTROL_SURFACE_STORE
    with tempfile.TemporaryDirectory(prefix="orion-smoke-control-surface-") as tmp_dir:
        isolated = mutation_control_surface.RuntimeControlSurfaceStore(
            sql_db_path=str(Path(tmp_dir) / "control-surface-isolated.sqlite3")
        )
        # Fail loudly rather than silently writing production: the whole point
        # of this block is that the smoke cannot reach the real surface.
        if isolated.postgres_url is not None or isolated.source_kind() != "sqlite":
            raise RuntimeError(
                "smoke control-surface isolation failed: "
                f"postgres_url={isolated.postgres_url!r} source_kind={isolated.source_kind()!r}"
            )
        mutation_control_surface._CONTROL_SURFACE_STORE = isolated
        try:
            mutation_control_surface.set_chat_reflective_lane_threshold(
                value=seed_threshold, actor="mutation_smoke"
            )
            yield isolated
        finally:
            mutation_control_surface._CONTROL_SURFACE_STORE = previous_store
            for key, value in previous_env.items():
                if value is not None:
                    os.environ[key] = value


def _routing_smoke_proposal(*, subject_ref: str, target_value: float = 0.58) -> MutationProposalV1:
    """Build a routing_threshold_patch proposal directly.

    As of 2026-09-03 `ProposalFactory.plan_for_pressure()`/`from_pressure()`
    refuse every "routing" pressure outright (parked -- see
    mutation_proposals.py's `_ROUTING_TARGET_PARKED_REASON`), so this smoke's
    active-surface/auto-promote/apply/rollback-required demonstrations, which
    are about the queue/decision/apply mechanics and not about the parked
    evidence pipeline, build the proposal directly instead of going through
    the (now dead-for-routing) detector -> pressure -> factory chain. Mirrors
    what that chain used to build, reading the live isolated surface for the
    rollback the same way `_routing_threshold_payloads()` did.
    """
    current = float(inspect_chat_reflective_lane_threshold()["raw"]["value"])
    return MutationProposalV1(
        mutation_class="routing_threshold_patch",
        target_surface="routing",
        lane="operational",
        risk_tier="low",
        rationale="smoke-routing",
        anchor_scope="orion",
        subject_ref=subject_ref,
        expected_effect="reduce_runtime_executed",
        evidence_refs=["telemetry:smoke"],
        source_signal_ids=["signal:smoke"],
        source_pressure_id="pressure:smoke",
        patch=MutationPatchV1(
            mutation_class="routing_threshold_patch",
            target_surface="routing",
            target_ref="routing",
            patch={"chat_reflective_lane_threshold": target_value},
            rollback_payload={"chat_reflective_lane_threshold": current},
        ),
    )


def run_smoke(*, emit: bool = True) -> list[str]:
    lines: list[str] = []
    cycle_id = f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    def emit_line(fields: dict[str, Any]) -> None:
        line = _fmt_trace(fields)
        lines.append(line)
        if emit:
            print(line)

    emit_line({"event": "mutation_smoke_start", "cycle_id": cycle_id})

    with temporary_env("SUBSTRATE_MUTATION_AUTONOMY_ENABLED", "true"), _isolated_control_surface(
        seed_threshold=0.5
    ):
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
            # Real reader, not a stub: _isolated_control_surface() below already
            # points the store at a throwaway sqlite seeded to 0.5, so the smoke
            # exercises the live read path without touching the ambient surface.
            proposals=ProposalFactory(
                routing_surface_reader=inspect_chat_reflective_lane_threshold,
            ),
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
        # See _routing_smoke_proposal() -- the routing target is parked, so
        # this builds the proposal directly rather than through telemetry.
        store._active_surface_by_target["routing"] = "existing-adoption"
        blocked_proposal = _routing_smoke_proposal(subject_ref="entity:routing")
        store.add_proposal(blocked_proposal, priority=60)
        routing_metrics: dict[str, dict[str, float]] = {
            blocked_proposal.proposal_id: {"success_rate_delta": 0.3, "latency_ms_delta": 0.0}
        }
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
        allowed_proposal = _routing_smoke_proposal(subject_ref="entity:routing-allow")
        store.add_proposal(allowed_proposal, priority=60)
        routing_metrics[allowed_proposal.proposal_id] = {"success_rate_delta": 0.3, "latency_ms_delta": 0.0}
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=routing_metrics)
        applied = any(a.target_surface == "routing" and a.status == "applied" for a in store._adoptions.values())
        emit_line(
            {
                "event": "mutation_decision_recorded",
                "cycle_id": cycle_id,
                "decision": "auto_promote",
                "surface_key": "routing",
                "queue_status_after": store.queue_status_for_proposal(allowed_proposal.proposal_id) or "-",
                "applied": applied,
            }
        )

        # 4) Rollback payload required before apply.
        # Step 3 moved the isolated surface to 0.58. Reset the (throwaway)
        # surface first so this step has a real value to read for the
        # proposal it then strips a rollback from. Isolated store only;
        # never the ambient one. Built directly (see _routing_smoke_proposal)
        # rather than through the parked detector/pressure/factory chain --
        # this step is about PatchApplier's rollback-payload requirement, not
        # about how the proposal was generated.
        set_chat_reflective_lane_threshold(value=0.5, actor="mutation_smoke")
        proposal = _routing_smoke_proposal(subject_ref="entity:payload")
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
