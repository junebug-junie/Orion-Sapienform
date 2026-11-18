"""llm_inference lane: gateway window events -> reducer -> StateDeltaV1.

The round-trip tests build their input with the gateway's real emitter
(services/orion-llm-gateway/app/grammar_emit.py, loaded by path so its service
``app`` package never collides with another service's), so a wire-format drift
between producer and reducer fails here rather than silently producing no deltas.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.llm_inference_projection import (
    LLM_INFERENCE_SOURCE_SERVICE,
    ROLE_NODE_WINDOW,
    LlmInferenceProjectionV1,
)
from orion.substrate.llm_inference_loop.constants import (
    LLM_INFERENCE_PROJECTION_ID,
    LLM_INFERENCE_TARGET_KIND,
)
from orion.substrate.llm_inference_loop.extract import (
    known_field_node,
    parse_llm_inference_trace_id,
)
from orion.substrate.llm_inference_loop.failure_window import (
    FAILURE_MIN_COUNT,
    FAILURE_MIN_DENOMINATOR,
    FAILURE_WINDOW_SEC,
    failure_reading,
)
from orion.schemas.llm_inference_projection import LlmInferenceWindowCountV1
from orion.substrate.llm_inference_loop.pipeline import (
    empty_llm_inference_projection,
    process_llm_inference_grammar_events,
)
from orion.substrate.llm_inference_loop.reducer import reduce_llm_inference_trace_events

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _load_emitter():
    path = REPO / "services" / "orion-llm-gateway" / "app" / "grammar_emit.py"
    spec = importlib.util.spec_from_file_location("llm_gateway_grammar_emit_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    spec.loader.exec_module(mod)
    return mod


EMIT = _load_emitter()

OK = {"text": "answer", "raw": {"usage": {"prompt_tokens": 50, "completion_tokens": 10}}}
TIMEOUT = {"text": "[Error: llamacpp timed out after waiting]", "raw": {}}
REFUSED = {"text": "", "raw": {"error": "gateway_overloaded"}}


T0 = 1_758_801_600.0


def _window(
    calls: list[tuple[dict, str | None]], *, gateway: str = "athena", start: float = T0
) -> list[GrammarEventV1]:
    rec = EMIT.InferenceWindowRecorder(clock=lambda: start)
    for result, served_by in calls:
        rec.record(result, served_by=served_by)
    start, _end, buckets = rec.drain()
    return EMIT.build_window_events(gateway_node=gateway, window_start=start, window_end=start + 60, buckets=buckets)


def _reduce(events, projection=None):
    return reduce_llm_inference_trace_events(
        events=events,
        projection=projection or empty_llm_inference_projection(now=NOW),
        now=NOW,
    )


def test_round_trip_failure_share_reaches_the_delta():
    events = _window([(OK, "circe-worker-2")] * 3 + [(TIMEOUT, "circe-worker-fast-1")])
    projection, receipt = _reduce(events)

    assert len(receipt.state_deltas) == 1
    delta = receipt.state_deltas[0]
    assert delta.target_kind == LLM_INFERENCE_TARGET_KIND
    assert delta.target_id == "llm_node:circe"
    assert delta.after["node_id"] == "circe"
    # One upstream failure is below the 2-failure minimum: 0.0, not 0.25.
    assert delta.after["pressure_hints"] == {"inference_failure_pressure": 0.0}
    assert delta.after["failure_window"]["attempted"] == 4
    assert delta.after["failure_window"]["failed"] == 1
    assert delta.after["served_by_labels"] == ["circe-worker-2", "circe-worker-fast-1"]
    assert delta.after["outcome_classes"] == {"served": 3, "upstream_timeout": 1}
    assert delta.after["window_sec"] == 60.0
    assert delta.after["completion_tokens"] == 30
    assert set(receipt.accepted_event_ids) == {e.event_id for e in events}

    state = projection.nodes["llm_node:circe"]
    assert state.calls == 4 and state.served == 3 and state.upstream_failed == 1
    assert projection.projection_id == LLM_INFERENCE_PROJECTION_ID


def test_all_served_is_a_real_zero_not_absent():
    _, receipt = _reduce(_window([(OK, "circe-worker-2")] * 5))
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.0}


def test_refusals_alone_are_not_measured_as_calm():
    """Only the gateway said no; the backend was never asked. That is not a
    reading of backend health either way, so no hint is emitted at all."""
    _, receipt = _reduce(_window([(REFUSED, "circe-worker-2")] * 3))
    delta = receipt.state_deltas[0]
    assert delta.after["refused"] == 3
    assert delta.after["pressure_hints"] == {}
    assert delta.after["inference_failure_pressure"] is None


def test_refusals_do_not_dilute_or_inflate_the_failure_share():
    calls = [(OK, "circe-worker-2")] * 2 + [(TIMEOUT, "circe-worker-2")] * 2 + [(REFUSED, "circe-worker-2")] * 8
    _, receipt = _reduce(_window(calls))
    # 2 failures of 4 upstream attempts, floored denominator 10 -> 0.2 (refusals ignored).
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.2}


def test_upstream_4xx_is_a_bad_request_not_a_node_failure():
    bad = {"text": "[Error: llamacpp failed: Client error '400 Bad Request' for url 'u'", "raw": {}}
    _, receipt = _reduce(_window([(bad, "circe-worker-2"), (OK, "circe-worker-2")]))
    after = receipt.state_deltas[0].after
    assert after["request_invalid"] == 1
    assert after["pressure_hints"] == {"inference_failure_pressure": 0.0}


def test_unknown_and_unrouted_labels_never_mint_a_field_node():
    events = _window([(TIMEOUT, "atlas-worker-1"), (TIMEOUT, None), (TIMEOUT, "mystery"), (OK, "circe-worker-2")])
    projection, receipt = _reduce(events)
    assert [d.target_id for d in receipt.state_deltas] == ["llm_node:circe"]
    assert projection.last_unattributed_calls == 3


def test_empty_window_is_accepted_with_no_deltas():
    projection, receipt = _reduce(_window([]))
    assert receipt.state_deltas == []
    assert len(receipt.accepted_event_ids) == 1
    assert projection.last_window_id == "20250925T120000Z"


def test_second_window_replaces_first_and_records_before():
    projection, _ = _reduce(_window([(TIMEOUT, "circe-worker-2")] * 2))
    projection, receipt = _reduce(_window([(OK, "circe-worker-2")], start=T0 + 60), projection)
    delta = receipt.state_deltas[0]
    assert delta.operation == "update"
    assert delta.before["inference_failure_pressure"] == 0.2
    assert delta.before["upstream_failed"] == 2
    assert delta.after["upstream_failed"] == 0
    # Window counts are replaced; the failure reading still spans both windows.
    assert delta.after["pressure_hints"] == {"inference_failure_pressure": 0.2}


def test_single_timeout_on_a_quiet_minute_is_not_full_failure():
    """The live incident (2026-09-25..29): one agent-lane timeout, one call in the
    window, read 1.0 and held for hundreds of field ticks."""
    _, receipt = _reduce(_window([(TIMEOUT, "circe-worker-agent")]))
    after = receipt.state_deltas[0].after
    assert after["pressure_hints"] == {"inference_failure_pressure": 0.0}
    assert after["failure_window"]["failed"] == 1


def test_failures_in_separate_windows_add_up_then_age_out():
    projection, _ = _reduce(_window([(TIMEOUT, "circe-worker-agent")]))
    projection, receipt = _reduce(_window([(TIMEOUT, "circe-worker-agent")], start=T0 + 180), projection)
    fw = receipt.state_deltas[0].after["failure_window"]
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.2}
    assert (fw["failed"], fw["attempted"], fw["windows"]) == (2, 2, 2)
    # 11 minutes after the second failure both have left the 600 s span.
    projection, receipt = _reduce(
        _window([(OK, "circe-worker-agent")], start=T0 + 180 + FAILURE_WINDOW_SEC + 60), projection
    )
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.0}
    assert len(projection.recent_windows["llm_node:circe"]) == 1


def test_worst_worker_is_not_diluted_by_a_busy_healthy_one():
    calls = [(OK, "circe-worker-chat")] * 30 + [(TIMEOUT, "circe-worker-agent")] * 2
    _, receipt = _reduce(_window(calls))
    fw = receipt.state_deltas[0].after["failure_window"]
    # pooled: 2 / 32 = 0.0625; the agent worker alone: 2 / max(2, 10) = 0.2
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.2}
    assert fw["scope"] == "circe-worker-agent"
    assert (fw["scope_failed"], fw["scope_attempted"]) == (2, 2)


def test_failures_spread_across_workers_still_count_pooled():
    calls = [(TIMEOUT, "circe-worker-agent"), (TIMEOUT, "circe-worker-chat"), (OK, "circe-worker-chat")]
    _, receipt = _reduce(_window(calls))
    fw = receipt.state_deltas[0].after["failure_window"]
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.2}
    assert fw["scope"] == "node"


def test_replayed_window_is_not_counted_twice():
    events = _window([(TIMEOUT, "circe-worker-agent")])
    projection, _ = _reduce(events)
    projection, receipt = _reduce(events, projection)
    assert receipt.state_deltas[0].after["failure_window"]["failed"] == 1
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.0}


def test_summary_from_an_older_gateway_without_worker_counts_reads_pooled():
    trace = "llm_gateway.inference:athena:w1"
    ev = GrammarEventV1(
        event_id="e1",
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id="e1",
            trace_id=trace,
            atom_type="observation",
            semantic_role=ROLE_NODE_WINDOW,
            layer="inference",
            summary="node=circe calls=4 served=1 upstream_failed=3 workers=circe-worker-agent classes=served:1|upstream_timeout:3",
        ),
        provenance=GrammarProvenanceV1(source_service=LLM_INFERENCE_SOURCE_SERVICE),
    )
    _, receipt = _reduce([ev])
    after = receipt.state_deltas[0].after
    assert after["pressure_hints"] == {"inference_failure_pressure": 0.3}
    assert after["failure_window"]["scope"] == "node"


def test_gateway_summary_carries_per_worker_counts():
    events = _window([(OK, "circe-worker-chat"), (TIMEOUT, "circe-worker-agent"), (REFUSED, "circe-worker-chat")])
    summary = events[0].atom.summary
    assert "worker_attempted=circe-worker-agent:1|circe-worker-chat:1" in summary
    assert "worker_failed=circe-worker-agent:1" in summary


def test_duplicate_node_atom_is_not_double_counted():
    events = _window([(TIMEOUT, "circe-worker-2")])
    node_event = events[0]
    dup = node_event.model_copy(update={"event_id": node_event.event_id + ":dup"})
    _, receipt = _reduce([node_event, dup, events[-1]])
    assert receipt.state_deltas[0].after["upstream_failed"] == 1


def test_foreign_source_is_noop():
    events = _window([(TIMEOUT, "circe-worker-2")])
    forged = [e.model_copy(update={"provenance": GrammarProvenanceV1(source_service="orion-bus")}) for e in events]
    projection, receipt = _reduce(forged)
    assert receipt.state_deltas == []
    assert set(receipt.noop_event_ids) == {e.event_id for e in forged}
    assert projection.nodes == {}


def test_bad_trace_is_noop():
    events = _window([(TIMEOUT, "circe-worker-2")])
    bad = [e.model_copy(update={"trace_id": "bus.transport:x:y"}) for e in events]
    _, receipt = _reduce(bad)
    assert receipt.state_deltas == [] and receipt.noop_event_ids


def test_pipeline_groups_traces_and_saves_once():
    events = _window([(TIMEOUT, "circe-worker-2")], gateway="athena") + _window([(OK, "circe-worker-2")], gateway="other")
    saved: list[LlmInferenceProjectionV1] = []
    receipts = []
    stats = process_llm_inference_grammar_events(
        events=events,
        load_projection=lambda: empty_llm_inference_projection(now=NOW),
        save_projection=saved.append,
        save_receipt=receipts.append,
        now=NOW,
    )
    assert stats == {"events": len(events), "receipts": 2, "traces": 2}
    assert len(saved) == 1 and len(receipts) == 2


@pytest.mark.parametrize(
    "trace, expected",
    [
        ("llm_gateway.inference:athena:20260925T120000Z", ("athena", "20260925T120000Z")),
        ("llm_gateway.inference::w", None),
        ("llm_gateway.inference:athena:", None),
        ("bus.transport:athena:w", None),
        ("", None),
    ],
)
def test_parse_trace(trace, expected):
    assert parse_llm_inference_trace_id(trace) == expected


def test_failure_share_edges():
    def w(served: int, failed: int, wid: str = "w") -> LlmInferenceWindowCountV1:
        return LlmInferenceWindowCountV1(window_id=wid, window_end=NOW, served=served, upstream_failed=failed)

    assert failure_reading([]).pressure is None
    assert failure_reading([w(0, 0)]).pressure is None  # nothing sent upstream
    assert failure_reading([w(0, FAILURE_MIN_COUNT - 1)]).pressure == 0.0
    assert failure_reading([w(0, 4)]).pressure == 4 / FAILURE_MIN_DENOMINATOR
    assert failure_reading([w(0, 40)]).pressure == 1.0
    assert (FAILURE_WINDOW_SEC, FAILURE_MIN_DENOMINATOR, FAILURE_MIN_COUNT) == (600.0, 10, 2)
    assert known_field_node("CIRCE") == "circe"
    assert known_field_node("atlas") is None  # decommissioned 2026-08-21
    assert known_field_node(None) is None


def test_hand_built_atom_without_node_is_unattributed():
    trace = "llm_gateway.inference:athena:w1"
    ev = GrammarEventV1(
        event_id="e1",
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id="e1",
            trace_id=trace,
            atom_type="observation",
            semantic_role=ROLE_NODE_WINDOW,
            layer="inference",
            summary="calls=3 served=0 upstream_failed=3",
        ),
        provenance=GrammarProvenanceV1(source_service=LLM_INFERENCE_SOURCE_SERVICE),
    )
    projection, receipt = _reduce([ev])
    assert receipt.state_deltas == [] and projection.last_unattributed_calls == 3


# ── gpu-pool stage 6.2: per-role wait/model clocks (consumer side) ─────────────────────────


def _hand_atom(summary: str, trace: str = "llm_gateway.inference:athena:w62") -> GrammarEventV1:
    return GrammarEventV1(
        event_id=f"{trace}:00",
        event_kind="atom_emitted",
        trace_id=trace,
        emitted_at=NOW,
        atom=GrammarAtomV1(
            atom_id=f"{trace}:00",
            trace_id=trace,
            atom_type="observation",
            semantic_role=ROLE_NODE_WINDOW,
            layer="inference",
            summary=summary,
        ),
        provenance=GrammarProvenanceV1(source_service=LLM_INFERENCE_SOURCE_SERVICE),
    )


def _clocked(wait_ms, model_ms, role):
    clock = EMIT.CallClock()
    clock.wait_ms, clock.model_ms, clock.role = wait_ms, model_ms, role
    return clock


def test_round_trip_carries_per_role_clocks_into_the_projection_and_delta():
    rec = EMIT.InferenceWindowRecorder(clock=lambda: T0)
    fast = {"text": "ok", "raw": {"timings": {"predicted_per_second": 55.0}}}
    rec.record(fast, served_by="circe-worker-fast", timing=_clocked(10, 300, "fast"))
    rec.record(fast, served_by="circe-worker-chat", timing=_clocked(128000, 9000, "chat"))
    rec.record(TIMEOUT, served_by="circe-worker-chat", timing=_clocked(2000, 60000, "chat"))
    start, _end, buckets = rec.drain()
    events = EMIT.build_window_events(gateway_node="athena", window_start=start, window_end=start + 60, buckets=buckets)

    projection, receipt = _reduce(events)
    state = projection.nodes["llm_node:circe"]
    assert set(state.by_role) == {"chat", "fast"}
    chat = state.by_role["chat"]
    assert (chat.calls, chat.served, chat.upstream_failed) == (2, 1, 1)
    # both calls waited (the timeout too); nearest-rank over [2000, 128000]
    assert (chat.wait_p50_ms, chat.wait_p95_ms) == (2000, 128000)
    assert chat.model_p50_ms == 9000  # served only: the timeout's 60 s budget is not a speed
    assert chat.decode_tps_p50 == 55.0 and chat.decode_tps_samples == 1
    assert state.by_role["fast"].model_p50_ms == 300
    # the delta the field digester sees carries it too (debug only, no pressure hint from it)
    after = receipt.state_deltas[0].after
    assert after["by_role"]["chat"]["wait_p95_ms"] == 128000
    assert set(after["pressure_hints"]) <= {"inference_failure_pressure"}
    assert "latency_p50_ms" not in after


def test_summary_from_a_gateway_before_stage_6_2_still_reduces_with_empty_roles():
    ev = _hand_atom("node=circe calls=2 served=2 upstream_failed=0 p50_ms=900 p95_ms=1200 "
                    "workers=circe-worker-chat classes=served:2")
    projection, receipt = _reduce([ev])
    state = projection.nodes["llm_node:circe"]
    assert state.by_role == {} and state.served == 2
    assert "latency_p50_ms" not in receipt.state_deltas[0].after


def test_malformed_role_entries_are_dropped_not_guessed():
    ev = _hand_atom("node=circe calls=3 served=3 upstream_failed=0 classes=served:3 "
                    "roles=chat[calls:2|served:2|wait_p50_ms:40|model_p50_ms:x|decode_tps_p50:nan]"
                    "fast[calls:1|served:1|decode_tps_p50:-4]junk")
    projection, _ = _reduce([ev])
    roles = projection.nodes["llm_node:circe"].by_role
    assert set(roles) == {"chat", "fast"}
    assert roles["chat"].wait_p50_ms == 40 and roles["chat"].model_p50_ms is None
    assert roles["chat"].decode_tps_p50 is None and roles["fast"].decode_tps_p50 is None


def test_persisted_row_from_the_previous_reducer_still_loads():
    """The live row carries the retired latency_p*_ms fields. Under extra="forbid" a plain
    removal would crash-loop the reducer on its first load (2026-07-24 incident class)."""
    live_shape = {
        "projection_id": LLM_INFERENCE_PROJECTION_ID,
        "generated_at": "2026-09-30T09:44:00Z",
        "nodes": {"llm_node:circe": {
            "calls": 8, "served": 8, "node_id": "circe", "refused": 0, "target_id": "llm_node:circe",
            "window_sec": 60.0, "observed_at": "2026-09-30T09:44:00.687844Z", "gateway_node": "gateway",
            "prompt_tokens": 8369, "latency_p50_ms": 1999, "latency_p95_ms": 79240,
            "schema_version": "llm_inference.node_state.v1", "outcome_classes": {"served": 8},
            "request_invalid": 0, "source_trace_id": "llm_gateway.inference:gateway:20260930T094259Z",
            "upstream_failed": 0, "sample_window_id": "20260930T094259Z",
            "served_by_labels": ["circe-worker-metacog", "circe-worker-chat"], "completion_tokens": 536,
            "evidence_event_ids": ["x"], "inference_failure_pressure": 0.0,
        }},
    }
    loaded = LlmInferenceProjectionV1.model_validate(live_shape)
    dumped = loaded.nodes["llm_node:circe"].model_dump()
    assert "latency_p50_ms" not in dumped and dumped["by_role"] == {}


def test_node_state_still_forbids_unknown_fields():
    from pydantic import ValidationError

    from orion.schemas.llm_inference_projection import LlmInferenceNodeStateV1, LlmInferenceRoleStateV1

    base = dict(target_id="llm_node:circe", node_id="circe", gateway_node="g", sample_window_id="w",
                source_trace_id="t", observed_at=NOW)
    with pytest.raises(ValidationError):
        LlmInferenceNodeStateV1(**base, latency_p99_ms=1)
    with pytest.raises(ValidationError):
        LlmInferenceRoleStateV1(calls=1, latency_ms=3)


def test_passthrough_only_window_does_not_move_inference_failure_pressure():
    """HTTP passthrough calls reach by_role only (stage 6.2 is record-only): a window with
    nothing but passthrough failures writes no failure pressure and adds nothing to its span."""
    rec = EMIT.InferenceWindowRecorder(clock=lambda: T0)
    for _ in range(5):
        rec.record_outcome("upstream_http_5xx", served_by="circe-worker-agent",
                           timing=_clocked(10, 100, "agent"), http=True)
    start, _end, buckets = rec.drain()
    events = EMIT.build_window_events(gateway_node="athena", window_start=start, window_end=start + 60, buckets=buckets)
    projection, receipt = _reduce(events)
    after = receipt.state_deltas[0].after
    assert after["pressure_hints"] == {}  # not measured, never a fake calm 0.0 or a 1.0
    agent = projection.nodes["llm_node:circe"].by_role["agent"]
    assert (agent.calls, agent.http_calls, agent.upstream_failed) == (5, 5, 5)
    assert "llm_node:circe" not in projection.recent_windows  # not folded at all


def test_a_role_only_window_does_not_slide_or_resend_the_live_failure_reading():
    """Before 6.2 a minute with no bus traffic produced no node atom, so the field held its last
    inference_failure_pressure. A role-only window (HTTP passthroughs) must keep that cadence:
    no fold, no hint, the previous reading and its span untouched (review finding)."""
    failing = _window([(TIMEOUT, "circe-worker-agent")] * 3)
    projection, first = _reduce(failing)
    before_pressure = projection.nodes["llm_node:circe"].inference_failure_pressure
    before_span = list(projection.recent_windows["llm_node:circe"])
    assert first.state_deltas[0].after["pressure_hints"]["inference_failure_pressure"] > 0

    rec = EMIT.InferenceWindowRecorder(clock=lambda: T0 + 60)
    rec.record_outcome("served", served_by="circe-worker-agent", timing=_clocked(5, 900, "agent"), http=True)
    start, _end, buckets = rec.drain()
    role_only = EMIT.build_window_events(gateway_node="athena", window_start=start, window_end=start + 60,
                                         buckets=buckets)
    projection, receipt = _reduce(role_only, projection)
    after = receipt.state_deltas[0].after
    assert after["pressure_hints"] == {}
    assert projection.recent_windows["llm_node:circe"] == before_span
    assert projection.nodes["llm_node:circe"].inference_failure_pressure == before_pressure
    assert projection.nodes["llm_node:circe"].by_role["agent"].http_calls == 1


def test_an_ungranted_wait_reaches_the_projection_under_the_pool_node():
    """A call the pool never granted has no worker; its wait -- the 'line is long' signal --
    is filed under the pool's host node so the reducer keeps it (review finding)."""
    rec = EMIT.InferenceWindowRecorder(clock=lambda: T0)
    clock = EMIT.CallClock(pool_node="circe")
    clock.wait_ms = 90000
    rec.record({"text": "", "raw": {"error": "gpu_pool_unavailable"}}, served_by=None, timing=clock)
    start, _end, buckets = rec.drain()
    events = EMIT.build_window_events(gateway_node="athena", window_start=start, window_end=start + 60, buckets=buckets)
    projection, _ = _reduce(events)
    assert projection.last_unattributed_calls == 1  # node counts unchanged
    ungranted = projection.nodes["llm_node:circe"].by_role["ungranted"]
    assert (ungranted.calls, ungranted.refused, ungranted.wait_p50_ms) == (1, 1, 90000)
    assert ungranted.model_p50_ms is None


def test_round_trip_carries_occupancy_slots_and_cache_counts():
    rec = EMIT.InferenceWindowRecorder(clock=lambda: T0)
    for busy, tps in [(1, 60.0), (2, 35.0), (2, 33.0)]:
        clock = _clocked(5, 800, "metacog")
        clock.busy_at_grant = busy
        rec.record({"text": "ok", "raw": {"timings": {"predicted_per_second": tps, "prompt_n": 20, "cache_n": 180}}},
                   served_by="circe-worker-metacog", timing=clock)
    start, _end, buckets = rec.drain()
    events = EMIT.build_window_events(gateway_node="athena", window_start=start, window_end=start + 60,
                                      buckets=buckets, role_slots={"metacog": 4})
    projection, _ = _reduce(events)
    m = projection.nodes["llm_node:circe"].by_role["metacog"]
    assert (m.decode_tps_solo_p50, m.decode_tps_solo_samples) == (60.0, 1)
    assert (m.decode_tps_shared_p50, m.decode_tps_shared_samples) == (33.0, 2)
    assert (m.busy_p50, m.busy_max, m.slots) == (2, 2, 4)
    assert (m.prompt_n, m.cache_n, m.cache_reports) == (60, 540, 3)
