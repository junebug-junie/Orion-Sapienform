from __future__ import annotations

from orion.schemas.registry import _REGISTRY


def test_agent_trace_models_are_registered() -> None:
    assert "AgentTraceToolStatV1" in _REGISTRY
    assert "AgentTraceStepV1" in _REGISTRY
    assert "AgentTraceSummaryV1" in _REGISTRY
