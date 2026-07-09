from __future__ import annotations

from app.store import EXECUTION_GRAMMAR_SOURCE_SERVICES, GRAMMAR_CURSOR_REGISTRY
from orion.substrate.execution_loop.constants import (
    EXECUTION_GRAMMAR_CURSOR_NAME,
    EXECUTION_SOURCE_SERVICES,
)


def test_execution_grammar_fetch_includes_harness_governor() -> None:
    source_services, trace_prefix = GRAMMAR_CURSOR_REGISTRY[EXECUTION_GRAMMAR_CURSOR_NAME]
    assert "orion-harness-governor" in source_services
    assert "orion-cortex-exec" in source_services
    assert source_services == EXECUTION_GRAMMAR_SOURCE_SERVICES
    assert set(EXECUTION_GRAMMAR_SOURCE_SERVICES) == set(EXECUTION_SOURCE_SERVICES)
    assert trace_prefix == "cortex.exec:"
