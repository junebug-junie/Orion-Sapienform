from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.pipeline import process_execution_grammar_events

FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
TRACE = "cortex.exec:athena:corr-pipe"


def _event(role: str, eid: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=eid,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:{role}",
            trace_id=TRACE,
            atom_type="observation",
            semantic_role=role,
            layer="execution",
            summary=f"stub {role}",
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
        ),
        correlation_id="corr-pipe",
    )


def test_pipeline_groups_by_trace_and_persists_receipts() -> None:
    state = {
        "projection": ExecutionTrajectoryProjectionV1(
            projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
            generated_at=FIXED_TS,
            runs={},
        ),
        "receipts": [],
    }

    stats = process_execution_grammar_events(
        events=[
            _event("exec_plan_started", "gev_1"),
            _event("exec_result_emitted", "gev_2"),
        ],
        load_projection=lambda: state["projection"],
        save_projection=lambda p: state.update(projection=p),
        save_receipt=lambda r: state["receipts"].append(r),
        now=FIXED_TS,
    )

    assert stats["events"] == 2
    assert stats["receipts"] == 1
    assert TRACE in state["projection"].runs
    assert state["receipts"][0].state_deltas[0].target_kind == "execution_run"
