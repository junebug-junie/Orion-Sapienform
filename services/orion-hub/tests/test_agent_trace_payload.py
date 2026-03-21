from __future__ import annotations

import importlib.util
from pathlib import Path

from orion.schemas.cortex.contracts import AgentTraceSummaryV1, CortexClientResult


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trace_payloads.py"
SPEC = importlib.util.spec_from_file_location("hub_trace_payloads", MODULE_PATH)
assert SPEC and SPEC.loader
trace_payloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_payloads)


def test_extract_agent_trace_payload_returns_json_safe_summary() -> None:
    cortex_result = CortexClientResult(
        ok=True,
        mode="agent",
        verb="agent_runtime",
        status="success",
        final_text="Ship it.",
        memory_used=False,
        recall_debug={},
        steps=[],
        correlation_id="corr-agent-ui",
        agent_trace=AgentTraceSummaryV1(
            corr_id="corr-agent-ui",
            message_id="corr-agent-ui",
            mode="agent",
            status="success",
            duration_ms=120,
            step_count=2,
            tool_call_count=2,
            unique_tool_count=2,
            unique_tool_families=["planning", "communication"],
            action_counts={"decide": 1, "summarize": 1},
            effect_counts={"read_only": 2},
            summary_text="Agent made planning decisions and returned a final response without write or side-effect actions.",
            tools=[],
            steps=[],
            raw={"source": "test"},
        ),
        metadata={"agent_trace_available": True},
    )

    payload = trace_payloads.extract_agent_trace_payload(cortex_result)

    assert payload is not None
    assert payload["corr_id"] == "corr-agent-ui"
    assert payload["summary_text"].startswith("Agent made planning decisions")
