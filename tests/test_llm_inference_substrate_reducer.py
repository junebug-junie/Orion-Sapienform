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
    inference_failure_pressure,
    known_field_node,
    parse_llm_inference_trace_id,
)
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


def _window(calls: list[tuple[dict, str | None]], *, gateway: str = "athena") -> list[GrammarEventV1]:
    rec = EMIT.InferenceWindowRecorder(clock=lambda: 1_758_801_600.0)
    for result, served_by in calls:
        rec.record(result, served_by=served_by, elapsed_s=0.4)
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
    assert delta.after["pressure_hints"] == {"inference_failure_pressure": 0.25}
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
    _, receipt = _reduce(_window([(OK, "circe-worker-2"), (TIMEOUT, "circe-worker-2")] + [(REFUSED, "circe-worker-2")] * 8))
    assert receipt.state_deltas[0].after["pressure_hints"] == {"inference_failure_pressure": 0.5}


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
    projection, _ = _reduce(_window([(TIMEOUT, "circe-worker-2")]))
    projection, receipt = _reduce(_window([(OK, "circe-worker-2")]), projection)
    delta = receipt.state_deltas[0]
    assert delta.operation == "update"
    assert delta.before["inference_failure_pressure"] == 1.0
    assert delta.after["pressure_hints"] == {"inference_failure_pressure": 0.0}


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
    assert inference_failure_pressure(served=0, upstream_failed=0) is None
    assert inference_failure_pressure(served=0, upstream_failed=4) == 1.0
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
