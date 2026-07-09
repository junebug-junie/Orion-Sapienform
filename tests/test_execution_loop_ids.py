from __future__ import annotations

from orion.substrate.execution_loop.ids import cortex_exec_trace_id, parse_execution_trace_id


def test_cortex_exec_trace_id_without_lane() -> None:
    assert cortex_exec_trace_id("athena", "corr-1") == "cortex.exec:athena:corr-1"


def test_cortex_exec_trace_id_with_lane_suffix() -> None:
    assert (
        cortex_exec_trace_id("athena", "corr-1", lane="harness_finalize_reflect")
        == "cortex.exec:athena:corr-1:harness_finalize_reflect"
    )


def test_parse_execution_trace_id_accepts_lane_suffix() -> None:
    parsed = parse_execution_trace_id("cortex.exec:athena:corr-1:harness_finalize_reflect")
    assert parsed == ("athena", "corr-1:harness_finalize_reflect")
