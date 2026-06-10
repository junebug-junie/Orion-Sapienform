from __future__ import annotations

import pytest

from app import trace_tools
from app.organ_runtime import OrganRuntime
from orion.schemas.context_exec import ContextExecRequestV1


@pytest.mark.asyncio
async def test_traces_search_does_not_default_to_current_run_id(monkeypatch):
    from app import settings as settings_mod

    trace_tools.clear_trace_store()
    monkeypatch.setattr(settings_mod.settings, "context_exec_real_trace_enabled", True)

    historical_corr = "550e8400-e29b-41d4-a716-446655440099"
    current_run = "ctxrun_current123"
    trace_tools.register_trace_hit(
        source="cortex_exec",
        kind="step.failed",
        corr_id=historical_corr,
        run_id="ctxrun_historical999",
        snippet="fail open on planner hop",
    )

    runtime = OrganRuntime(
        bus=None,
        request=ContextExecRequestV1(
            text=f"Why did corr {historical_corr} fail open?",
            mode="trace_autopsy",
            correlation_id=current_run,
        ),
        run_id=current_run,
    )
    hits = await runtime.traces_search(corr_id=historical_corr, limit=10)
    assert len(hits) == 1
    assert hits[0]["corr_id"] == historical_corr
    assert hits[0]["run_id"] == "ctxrun_historical999"


@pytest.mark.asyncio
async def test_traces_search_can_filter_explicit_current_run_id(monkeypatch):
    from app import settings as settings_mod

    trace_tools.clear_trace_store()
    monkeypatch.setattr(settings_mod.settings, "context_exec_real_trace_enabled", True)

    current_run = "ctxrun_only_me"
    trace_tools.register_trace_hit(
        source="context_exec",
        kind="context.exec.started.v1",
        corr_id="corr-a",
        run_id=current_run,
        snippet="started",
    )
    trace_tools.register_trace_hit(
        source="context_exec",
        kind="context.exec.started.v1",
        corr_id="corr-b",
        run_id="ctxrun_other",
        snippet="started other",
    )

    runtime = OrganRuntime(
        bus=None,
        request=ContextExecRequestV1(text="inspect current run", mode="trace_autopsy"),
        run_id=current_run,
    )
    hits = await runtime.traces_search(run_id=current_run, limit=10)
    assert len(hits) == 1
    assert hits[0]["run_id"] == current_run
