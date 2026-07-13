from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.route_projection import (
    RouteArbitrationProjectionV1,
    RouteArbitrationRunStateV1,
)
from orion.substrate.route_loop.constants import ROUTE_ARBITRATION_PROJECTION_ID
from orion.substrate.route_loop.grammar_extract import (
    extract_route_state_from_events,
)
from orion.substrate.route_loop.merge import merge_route_run_state
from orion.substrate.route_loop.reducer import reduce_route_trace_events

FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
TRACE = "orch.route:athena:corr-abc"


def _route_atom(
    summary: str,
    *,
    event_id: str = "gev_x",
    role: str = "route_arbitration_decided",
) -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{TRACE}:{role}:{event_id}",
        trace_id=TRACE,
        atom_type="reasoning_step",
        semantic_role=role,
        layer="route",
        summary=summary,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-orch",
            source_component="orch_grammar_emit",
        ),
        correlation_id="corr-abc",
        session_id="sess-1",
        turn_id="turn-1",
    )


def _empty_projection() -> RouteArbitrationProjectionV1:
    return RouteArbitrationProjectionV1(
        projection_id=ROUTE_ARBITRATION_PROJECTION_ID,
        generated_at=FIXED_TS,
        runs={},
    )


def _event_for_trace(
    trace_id: str,
    summary: str,
    *,
    event_id: str,
    ts: datetime,
) -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{trace_id}:route_arbitration_decided",
        trace_id=trace_id,
        atom_type="reasoning_step",
        semantic_role="route_arbitration_decided",
        layer="route",
        summary=summary,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=trace_id,
        emitted_at=ts,
        observed_at=ts,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-orch",
            source_component="orch_grammar_emit",
        ),
        correlation_id=trace_id.split(":")[-1],
    )


# --- extraction ---


def test_extract_builds_run_state_from_route_atom() -> None:
    events = [
        _route_atom(
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
            event_id="gev_1",
        )
    ]
    run = extract_route_state_from_events(events, now=FIXED_TS)
    assert run.trace_id == TRACE
    assert run.node_id == "athena"
    assert run.correlation_id == "corr-abc"
    assert run.lane == "chat"
    assert run.lane_reason == "default"
    assert run.mind_requested is True
    assert run.mind_skip_reason is None
    assert run.output_mode == "stream"
    assert "gev_1" in run.evidence_event_ids


def test_extract_mind_skip_reason_populated_when_not_none() -> None:
    events = [
        _route_atom(
            "lane=plan,lane_reason=complex_task,mind_requested=false,mind_skip_reason=budget_exhausted,output_mode=batch",
            event_id="gev_2",
        )
    ]
    run = extract_route_state_from_events(events, now=FIXED_TS)
    assert run.lane == "plan"
    assert run.mind_requested is False
    assert run.mind_skip_reason == "budget_exhausted"
    assert run.output_mode == "batch"


def test_extract_ignores_non_orch_source_service() -> None:
    events = [
        GrammarEventV1(
            event_id="gev_other",
            event_kind="atom_emitted",
            trace_id=TRACE,
            emitted_at=FIXED_TS,
            observed_at=FIXED_TS,
            atom=GrammarAtomV1(
                atom_id=f"{TRACE}:trace_started",
                trace_id=TRACE,
                atom_type="reasoning_step",
                semantic_role="trace_started",
                layer="route",
                summary="lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
            ),
            provenance=GrammarProvenanceV1(source_service="orion-cortex-exec"),
        )
    ]
    run = extract_route_state_from_events(events, now=FIXED_TS)
    assert run.lane == "unknown"
    assert run.evidence_event_ids == []


def test_extract_raises_on_empty_events() -> None:
    import pytest

    with pytest.raises(ValueError):
        extract_route_state_from_events([], now=FIXED_TS)


# --- merge ---


def test_merge_last_write_wins_on_non_default_fields() -> None:
    base = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        lane="unknown",
        lane_reason="unknown",
        mind_requested=False,
        mind_skip_reason=None,
        output_mode="unknown",
        evidence_event_ids=["gev_1"],
        last_updated_at=FIXED_TS,
    )
    incoming = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        lane="chat",
        lane_reason="default",
        mind_requested=True,
        mind_skip_reason="none",
        output_mode="stream",
        evidence_event_ids=["gev_2"],
        last_updated_at=FIXED_TS + timedelta(seconds=1),
    )
    merged = merge_route_run_state(base, incoming)
    assert merged.lane == "chat"
    assert merged.lane_reason == "default"
    assert merged.mind_requested is True
    assert merged.output_mode == "stream"
    assert set(merged.evidence_event_ids) == {"gev_1", "gev_2"}


def test_merge_does_not_downgrade_mind_requested() -> None:
    base = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        mind_requested=True,
        last_updated_at=FIXED_TS,
    )
    incoming = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        mind_requested=False,
        last_updated_at=FIXED_TS + timedelta(seconds=1),
    )
    merged = merge_route_run_state(base, incoming)
    assert merged.mind_requested is True


def test_merge_caps_evidence_event_ids_at_200() -> None:
    base = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        evidence_event_ids=[f"gev_old_{i}" for i in range(250)],
        last_updated_at=FIXED_TS,
    )
    incoming = RouteArbitrationRunStateV1(
        trace_id=TRACE,
        correlation_id="corr-abc",
        node_id="athena",
        evidence_event_ids=[f"gev_new_{i}" for i in range(100)],
        last_updated_at=FIXED_TS + timedelta(seconds=1),
    )
    merged = merge_route_run_state(base, incoming)
    assert len(merged.evidence_event_ids) <= 200


# --- reducer ---


def test_reducer_emits_route_run_delta() -> None:
    events = [
        _route_atom(
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        )
    ]
    proj, receipt = reduce_route_trace_events(
        events=events,
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    assert len(receipt.state_deltas) == 1
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "route_arbitration_run"
    assert delta.target_id == TRACE
    assert delta.after["node_id"] == "athena"
    assert delta.after["lane"] == "chat"
    assert TRACE in proj.runs


def test_reducer_noops_non_orch_source() -> None:
    bio = GrammarEventV1(
        event_id="gev_bio",
        event_kind="atom_emitted",
        trace_id="biometrics.node:atlas:ts",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id="a1",
            trace_id="biometrics.node:atlas:ts",
            atom_type="signal",
            semantic_role="body_state",
            layer="biometrics",
            summary="body",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-biometrics", source_component="x"),
    )
    proj, receipt = reduce_route_trace_events(
        events=[bio], projection=_empty_projection(), now=FIXED_TS
    )
    assert proj.runs == {}
    assert receipt.noop_event_ids == ["gev_bio"]


def test_reducer_noops_when_trace_prefix_does_not_parse() -> None:
    bad = GrammarEventV1(
        event_id="gev_bad",
        event_kind="atom_emitted",
        trace_id="cortex.exec:athena:corr-abc",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id="a1",
            trace_id="cortex.exec:athena:corr-abc",
            atom_type="reasoning_step",
            semantic_role="route_arbitration_decided",
            layer="route",
            summary="lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-cortex-orch"),
    )
    proj, receipt = reduce_route_trace_events(
        events=[bad], projection=_empty_projection(), now=FIXED_TS
    )
    assert proj.runs == {}
    assert receipt.noop_event_ids == ["gev_bad"]


def test_reducer_empty_events_is_noop() -> None:
    proj = _empty_projection()
    result_proj, receipt = reduce_route_trace_events(
        events=[], projection=proj, now=FIXED_TS
    )
    assert result_proj is proj
    assert receipt.accepted_event_ids == []


def test_stable_delta_id_on_replay() -> None:
    events = [
        _route_atom(
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        )
    ]
    _, r1 = reduce_route_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    _, r2 = reduce_route_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    assert r1.receipt_id == r2.receipt_id
    assert r1.state_deltas[0].delta_id == r2.state_deltas[0].delta_id


# --- eviction / cap regression (mirrors execution_loop's 8daeecf7 fix) ---


def test_reducer_evicts_oldest_runs_when_max_runs_exceeded() -> None:
    proj = _empty_projection()
    trace_ids = [f"orch.route:athena:corr-{i}" for i in range(5)]
    for i, trace_id in enumerate(trace_ids):
        ts = FIXED_TS + timedelta(seconds=i)
        event = _event_for_trace(
            trace_id,
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
            event_id=f"gev_evict_{i}",
            ts=ts,
        )
        proj, _ = reduce_route_trace_events(
            events=[event],
            projection=proj,
            now=ts,
            max_runs=3,
        )
        assert len(proj.runs) <= 3

    assert set(proj.runs) == set(trace_ids[-3:])


def test_reducer_never_evicts_run_just_written_even_on_last_updated_at_tie() -> None:
    # Regression for the hard requirement: "the just-written run must never be
    # evicted by this pass in the same tick it was written." pipeline.py shares
    # a single clock across every trace_id in a batch, so two distinct runs can
    # legitimately end up with an identical last_updated_at.
    proj = _empty_projection()
    trace_a = "orch.route:athena:corr-a"
    trace_b = "orch.route:athena:corr-b"

    proj, _ = reduce_route_trace_events(
        events=[
            _event_for_trace(
                trace_a,
                "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
                event_id="gev_tie_a1",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=2,
    )
    assert set(proj.runs) == {trace_a}

    proj, _ = reduce_route_trace_events(
        events=[
            _event_for_trace(
                trace_b,
                "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
                event_id="gev_tie_b1",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=2,
    )
    assert set(proj.runs) == {trace_a, trace_b}

    proj, _ = reduce_route_trace_events(
        events=[
            _event_for_trace(
                trace_a,
                "lane=plan,lane_reason=complex_task,mind_requested=true,mind_skip_reason=none,output_mode=batch",
                event_id="gev_tie_a2",
                ts=FIXED_TS,
            )
        ],
        projection=proj,
        now=FIXED_TS,
        max_runs=1,
    )
    assert set(proj.runs) == {trace_a}, (
        "the just-written run (trace_a) must never be evicted in the same "
        "call it was written, even when tied on last_updated_at with another "
        "run from the same batch"
    )


def test_reducer_evicts_runs_older_than_max_age_sec() -> None:
    proj = _empty_projection()
    old_trace = "orch.route:athena:corr-old"
    new_trace = "orch.route:athena:corr-new"

    old_ts = FIXED_TS
    old_event = _event_for_trace(
        old_trace,
        "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        event_id="gev_age_old",
        ts=old_ts,
    )
    proj, _ = reduce_route_trace_events(
        events=[old_event], projection=proj, now=old_ts, max_runs=None, max_age_sec=None
    )
    assert old_trace in proj.runs

    new_ts = old_ts + timedelta(seconds=3600)
    new_event = _event_for_trace(
        new_trace,
        "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        event_id="gev_age_new",
        ts=new_ts,
    )
    proj, _ = reduce_route_trace_events(
        events=[new_event], projection=proj, now=new_ts, max_runs=None, max_age_sec=60
    )
    assert old_trace not in proj.runs
    assert new_trace in proj.runs


def test_reducer_load_test_stays_capped_at_route_arbitration_max_runs() -> None:
    """Load-test regression: >2000 synthetic trace_ids must not exceed the cap,
    and the just-written run in each batch is never evicted in that same batch.
    """
    from orion.substrate.route_loop.constants import ROUTE_ARBITRATION_MAX_RUNS

    proj = _empty_projection()
    total = ROUTE_ARBITRATION_MAX_RUNS + 500
    for i in range(total):
        ts = FIXED_TS + timedelta(seconds=i)
        trace_id = f"orch.route:athena:corr-load-{i}"
        event = _event_for_trace(
            trace_id,
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
            event_id=f"gev_load_{i}",
            ts=ts,
        )
        proj, _ = reduce_route_trace_events(
            events=[event],
            projection=proj,
            now=ts,
        )
        assert trace_id in proj.runs, "just-written run evicted in its own batch"
        assert len(proj.runs) <= ROUTE_ARBITRATION_MAX_RUNS

    assert len(proj.runs) == ROUTE_ARBITRATION_MAX_RUNS


def test_reducer_default_caps_do_not_evict_single_trace() -> None:
    events = [
        _route_atom(
            "lane=chat,lane_reason=default,mind_requested=true,mind_skip_reason=none,output_mode=stream",
        )
    ]
    proj, _ = reduce_route_trace_events(
        events=events, projection=_empty_projection(), now=FIXED_TS
    )
    assert TRACE in proj.runs
    assert len(proj.runs) == 1
