"""storage_write lane: sql-writer window events -> reducer -> StateDeltaV1.

Inputs are built with the writer's real emitter
(services/orion-sql-writer/app/write_health.py, loaded by path so its service
``app`` package never collides with another service's), so a wire-format drift
between producer and reducer fails here rather than silently producing no deltas.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.schemas.storage_write_projection import (
    ROLE_FAMILY_WINDOW,
    STORAGE_WRITE_NODE_ID,
    StorageWriteProjectionV1,
)
from orion.substrate.storage_write_loop.extract import parse_storage_write_trace_id
from orion.substrate.storage_write_loop.pipeline import (
    empty_storage_write_projection,
    process_storage_write_grammar_events,
)
from orion.substrate.storage_write_loop.reducer import reduce_storage_write_trace_events

REPO = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)


def _load_write_health():
    path = REPO / "services" / "orion-sql-writer" / "app" / "write_health.py"
    spec = importlib.util.spec_from_file_location("sqlw_write_health_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load_write_health()


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _window(start: datetime, outcomes: dict[str, dict[str, int]], *, sec: float = 60.0) -> list[GrammarEventV1]:
    clock = _Clock(start.timestamp())
    rec = wh.WriteHealthRecorder(clock=clock)
    for family, classes in outcomes.items():
        for cls, n in classes.items():
            rec.record(family, cls, count=n, latency_ms=5.0)
    clock.t = start.timestamp() + sec
    s, e, buckets, qmax = rec.drain()
    return wh.build_window_events(writer_node="athena", window_start=s, window_end=e, buckets=buckets,
                                  grammar_queue_max=qmax)


def _reduce(events, projection=None):
    projection = projection or empty_storage_write_projection(now=T0)
    return reduce_storage_write_trace_events(events=events, projection=projection, now=T0)


def _hint(receipt):
    return receipt.state_deltas[0].after["pressure_hints"].get("write_failure_pressure")


def test_trace_id_round_trip():
    events = _window(T0, {"grammar_events": {"committed": 3}})
    assert parse_storage_write_trace_id(events[0].trace_id) == ("athena", "20260926T030000Z")
    assert parse_storage_write_trace_id("sql_writer.storage:") is None
    assert parse_storage_write_trace_id("llm_gateway.inference:athena:x") is None


def test_calm_window_reads_measured_zero_not_absent():
    events = _window(T0, {"grammar_events": {"committed": 265}, "gpu_pool_events": {"committed": 200, "duplicate": 3}})
    proj, receipt = _reduce(events)
    assert _hint(receipt) == 0.0
    assert proj.write_failure_pressure == 0.0
    assert receipt.state_deltas[0].target_id == STORAGE_WRITE_NODE_ID
    assert proj.families["gpu_pool_events"].duplicate == 3
    assert proj.families["gpu_pool_events"].write_p50_ms == 5


def test_home_cooling_burst_reads_worst_family_not_pooled():
    """2026-09-26 03:00: every home_cooling_sample row failed to serialize while
    hundreds of other writes landed. Pooled that is ~2%; the family is 100%."""
    events = _window(T0, {
        "home_cooling_sample": {"serialization": 7},
        "grammar_events": {"committed": 265},
        "gpu_pool_events": {"committed": 207},
    })
    proj, receipt = _reduce(events)
    assert _hint(receipt) == pytest.approx(0.7)  # 7 / max(7, 10)
    reading = receipt.state_deltas[0].after["failure_window"]
    assert reading["scope"] == "home_cooling_sample"
    assert reading["failed"] == 7 and reading["attempted"] == 479
    # three minutes of the burst saturates the family
    for i in range(1, 3):
        proj, receipt = _reduce(_window(T0 + timedelta(minutes=i), {"home_cooling_sample": {"serialization": 7}}), proj)
    assert _hint(receipt) == 1.0


def test_single_failure_is_hysteresis_zero():
    events = _window(T0, {"cockpit_turn_sighting": {"validation": 1, "committed": 4}})
    _proj, receipt = _reduce(events)
    assert _hint(receipt) == 0.0


def test_idle_window_is_not_measured_and_carries_no_hint():
    events = _window(T0, {})
    assert [e.atom.semantic_role for e in events] == ["storage_writer_window_completed"]
    proj, receipt = _reduce(events)
    assert receipt.state_deltas[0].after["pressure_hints"] == {}
    assert proj.write_failure_pressure is None


def test_failures_age_out_of_the_span_on_event_time():
    proj, receipt = _reduce(_window(T0, {"journal_entries": {"constraint": 5}}))
    assert _hint(receipt) == pytest.approx(0.5)
    # 11 minutes later, healthy traffic only: the failures have left the 600 s span
    proj, receipt = _reduce(_window(T0 + timedelta(minutes=11), {"journal_entries": {"committed": 3}}), proj)
    assert _hint(receipt) == 0.0
    assert len(proj.recent_windows) == 1


def test_replayed_trace_counts_once_and_keeps_delta_id():
    events = _window(T0, {"journal_entries": {"constraint": 5}})
    proj1, r1 = _reduce(events)
    proj2, r2 = _reduce(events, proj1)
    assert _hint(r2) == _hint(r1) == pytest.approx(0.5)
    assert len(proj2.recent_windows) == 1


def test_split_trace_merges_by_family():
    events = _window(T0, {"a_table": {"validation": 3}, "b_table": {"validation": 3}})
    proj, _ = _reduce(events[:1])
    proj, receipt = _reduce(events[1:], proj)
    assert set(proj.families) == {"a_table", "b_table"}
    assert receipt.state_deltas[0].after["failure_window"]["failed"] == 6


def test_foreign_source_is_noop():
    events = _window(T0, {"x": {"validation": 5}})
    forged = [e.model_copy(update={"provenance": GrammarProvenanceV1(source_service="orion-hub")}) for e in events]
    _proj, receipt = _reduce(forged)
    assert receipt.state_deltas == []
    assert len(receipt.noop_event_ids) == len(forged)


def test_unknown_failure_class_is_dropped_not_guessed():
    trace = "sql_writer.storage:athena:20260926T030000Z"
    eid = f"{trace}:00:{ROLE_FAMILY_WINDOW}"
    event = GrammarEventV1(
        event_id=eid, event_kind="atom_emitted", trace_id=trace, emitted_at=T0,
        atom=GrammarAtomV1(atom_id=eid, trace_id=trace, atom_type="observation", semantic_role=ROLE_FAMILY_WINDOW,
                           layer="storage",
                           summary="family=x attempted=99 committed=1 duplicate=0 failed=9 classes=martian:9|timeout:2"),
        provenance=GrammarProvenanceV1(source_service="orion-sql-writer"),
    )
    proj, _ = _reduce([event])
    fam = proj.families["x"]
    assert fam.failure_classes == {"timeout": 2}
    assert fam.attempted == 3  # recomputed from parts, not the wire's 99


def test_projection_round_trips_under_forbid():
    proj, _ = _reduce(_window(T0, {"journal_entries": {"constraint": 5, "committed": 1}}))
    again = StorageWriteProjectionV1.model_validate(proj.model_dump(mode="json"))
    assert again == proj


def test_pipeline_reduces_traces_in_one_batch():
    batch = _window(T0, {"a": {"timeout": 2}}) + _window(T0 + timedelta(minutes=1), {"a": {"timeout": 2}})
    saved, receipts = [], []
    stats = process_storage_write_grammar_events(
        events=batch,
        load_projection=lambda: empty_storage_write_projection(now=T0),
        save_projection=saved.append,
        save_receipt=receipts.append,
        now=T0,
    )
    assert stats["traces"] == 2 and len(receipts) == 2
    assert _hint(receipts[-1]) == pytest.approx(0.4)  # 4 / max(4, 10)
    assert saved[-1].write_failure_pressure == pytest.approx(0.4)
