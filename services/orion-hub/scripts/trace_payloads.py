from __future__ import annotations

from typing import Any, Dict


def extract_agent_trace_payload(cortex_result: Any) -> Dict[str, Any] | None:
    if cortex_result is None:
        return None
    agent_trace = getattr(cortex_result, "agent_trace", None)
    if agent_trace is None:
        return None
    if hasattr(agent_trace, "model_dump"):
        return agent_trace.model_dump(mode="json")
    if isinstance(agent_trace, dict):
        return agent_trace
    return None
