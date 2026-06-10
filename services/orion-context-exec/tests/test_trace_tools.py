from __future__ import annotations

from app import trace_tools


def test_trace_search_by_corr_id_and_read():
    trace_tools.clear_trace_store()
    hit = trace_tools.register_trace_hit(
        source="context_exec",
        kind="context.exec.started.v1",
        corr_id="corr-abc",
        run_id="ctxrun_test1",
        snippet="started belief provenance",
        payload={"mode": "belief_provenance"},
    )
    found = trace_tools.traces_search(corr_id="corr-abc", limit=10)
    assert len(found) == 1
    assert found[0].handle == hit.handle
    body = trace_tools.traces_read(hit.handle)
    assert body["corr_id"] == "corr-abc"
    assert body["payload"]["mode"] == "belief_provenance"


def test_trace_search_query_filters_snippet():
    trace_tools.clear_trace_store()
    trace_tools.register_trace_hit(
        source="cortex_exec",
        kind="step.failed",
        corr_id="corr-1",
        snippet="fail open on planner hop",
    )
    trace_tools.register_trace_hit(
        source="cortex_exec",
        kind="step.ok",
        corr_id="corr-2",
        snippet="unrelated ok step",
    )
    hits = trace_tools.traces_search(query="fail open", limit=10)
    assert len(hits) == 1
    assert "fail open" in hits[0].snippet
