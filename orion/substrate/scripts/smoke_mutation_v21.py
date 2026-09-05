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
from orion.substrate.mutation_proposals import ProposalFactory, build_placeholder_routing_proposal
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
    """A routing_threshold_patch proposal, bypassing the retired ProposalFactory path.

    "routing" was parked 2026-09-03, then retired outright 2026-09-05 --
    `ProposalFactory.plan_for_pressure()`/`from_pressure()` no longer even
    recognize it as a target_surface, so this smoke's
    active-surface/auto-promote/apply/rollback-required demonstrations, which
    are about the queue/decision/apply mechanics and not about the (now-gone)
    live evidence pipeline, build the proposal directly instead of going
    through the (now dead-for-routing) detector -> pressure -> factory chain, via the
    shared `build_placeholder_routing_proposal()` (also used by
    test_mutation_v21.py and the orion-hub replay-inspection endpoint).

    Rollback is a fixed 0.50, matching `_isolated_control_surface()`'s seed,
    rather than a live read of the isolated surface -- no assertion in this
    script depends on the exact rollback value, and a fixed value removes any
    dependency on the surface having been written before this is called.
    """
    return build_placeholder_routing_proposal(
        target_value=target_value,
        rollback_value=0.50,
        subject_ref=subject_ref,
        source_pressure_id="pressure:smoke",
    )


def _graph_consolidation_smoke_proposal(*, subject_ref: str, target_value: int = 96) -> MutationProposalV1:
    """A graph_consolidation_param_patch proposal, appliable for real.

    Steps 2/3 below need a live end-to-end apply through the real worker
    cycle to demonstrate the active-surface lock actually releasing --
    `_routing_smoke_proposal()` can no longer do that (`routing_threshold_patch`
    is retired 2026-09-05, RETIRED_MUTATION_CLASSES; PatchApplier.apply()
    refuses it unconditionally now, active-surface-locked or not). This is
    the remaining auto-promotable class apply() does not special-case away.
    """
    return MutationProposalV1(
        lane="operational",
        mutation_class="graph_consolidation_param_patch",
        risk_tier="medium",
        target_surface="graph_consolidation",
        anchor_scope="orion",
        subject_ref=subject_ref,
        rationale=f"smoke:graph_consolidation_param_patch target={target_value}",
        expected_effect="reduce_runtime_failure",
        evidence_refs=["telemetry:smoke"],
        source_signal_ids=["signal:smoke"],
        source_pressure_id="pressure:smoke",
        patch=MutationPatchV1(
            mutation_class="graph_consolidation_param_patch",
            target_surface="graph_consolidation",
            target_ref="graph_consolidation",
            patch={"query_limit_nodes": target_value},
            rollback_payload={"query_limit_nodes": 64},
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
                    "graph_consolidation_param_patch": "corpus-graph-consolidation",
                    "approved_prompt_profile_variant_promotion": "corpus-prompt",
                },
                baseline_metric_ref_by_class={
                    "graph_consolidation_param_patch": "baseline-graph-consolidation",
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
        # Was "routing" -- retired 2026-09-05 (RETIRED_MUTATION_CLASSES),
        # PatchApplier.apply() now refuses it unconditionally, active-surface
        # block or not, which would make step 3 below (release the block,
        # confirm a real apply happens) demonstrate nothing. Switched to
        # "graph_consolidation", the remaining auto-promotable class apply()
        # does not special-case away. Built directly (see
        # _graph_consolidation_smoke_proposal) rather than through the
        # detector -> pressure -> factory chain -- this step is about the
        # active-surface lock, not about how the proposal was generated.
        store._active_surface_by_target["graph_consolidation"] = "existing-adoption"
        blocked_proposal = _graph_consolidation_smoke_proposal(subject_ref="entity:graph-consolidation")
        store.add_proposal(blocked_proposal, priority=60)
        graph_consolidation_metrics: dict[str, dict[str, float]] = {
            blocked_proposal.proposal_id: {"queue_resolution_delta": 0.3, "requeue_rate_delta": 0.0}
        }
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=graph_consolidation_metrics)
        emit_line(
            {
                "event": "mutation_apply_blocked",
                "cycle_id": cycle_id,
                "surface_key": "graph_consolidation",
                "blocked_reason": "active_surface",
                "applied": False,
            }
        )

        # 3) Allow auto-promote after removing active-surface block.
        store._active_surface_by_target.pop("graph_consolidation", None)
        allowed_proposal = _graph_consolidation_smoke_proposal(subject_ref="entity:graph-consolidation-allow")
        store.add_proposal(allowed_proposal, priority=60)
        graph_consolidation_metrics[allowed_proposal.proposal_id] = {"queue_resolution_delta": 0.3, "requeue_rate_delta": 0.0}
        worker.run_cycle(telemetry=[], measured_metrics_by_proposal=graph_consolidation_metrics)
        applied = any(a.target_surface == "graph_consolidation" and a.status == "applied" for a in store._adoptions.values())
        emit_line(
            {
                "event": "mutation_decision_recorded",
                "cycle_id": cycle_id,
                "decision": "auto_promote",
                "surface_key": "graph_consolidation",
                "queue_status_after": store.queue_status_for_proposal(allowed_proposal.proposal_id) or "-",
                "applied": applied,
            }
        )

        # 4) Rollback payload required before apply.
        # Also switched off "routing" (see step 2's note) -- this step is
        # about PatchApplier's rollback-payload requirement, which needs no
        # control-surface value at all on the generic apply path
        # "graph_consolidation" now takes, so the isolated-control-surface
        # setup this step used to depend on is no longer needed here.
        proposal = _graph_consolidation_smoke_proposal(subject_ref="entity:payload")
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
