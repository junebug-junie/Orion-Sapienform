from datetime import datetime, timezone

import scripts.hub_surface_routes as hub_surface


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _Conn:
    def __init__(self, shared_queue):
        # `shared_queue` is the SAME list object the engine holds -- some
        # routes here open a second `.connect()` for a follow-up query
        # (durable_runs' "example" lookup), so the queue must be consumed
        # across connections, not reset fresh on each one.
        self._queued = shared_queue
        self.executed_sql = []

    def execute(self, clause, *a, **k):
        self.executed_sql.append(str(clause))
        rows = self._queued.pop(0) if self._queued else []
        return _Result(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Engine:
    def __init__(self, *queued_rows):
        self._queued = list(queued_rows)  # one entry per expected execute() call, in order
        self.last_conn = None

    def connect(self):
        self.last_conn = _Conn(self._queued)
        return self.last_conn


# --------------------------------------------------------------------------
# Pure aggregation functions -- no I/O.
# --------------------------------------------------------------------------


def test_summarize_bridge_computes_percentages_of_parseable_rows():
    rows = [
        {"self_model_json": {"voluntary_override_absent_reason": "goal_matched_no_loop", "attention_reason": "bottom_up_salience"}},
        {"self_model_json": {"voluntary_override_absent_reason": "goal_matched_no_loop", "attention_reason": "bottom_up_salience"}},
        {"self_model_json": {"voluntary_override_absent_reason": None, "attention_reason": "top_down_override"}},
        {"self_model_json": "not json"},  # malformed -- counted, not silently dropped
    ]
    out = hub_surface.summarize_bridge(rows)
    assert out["sample_count"] == 3
    assert out["malformed_row_count"] == 1
    # 2 of 3 parseable rows -> goal_matched_no_loop
    assert out["branches"]["goal_matched_no_loop"] == 66.7
    # 1 of 3 -> top_down_override
    assert out["branches"]["top_down_override"] == 33.3


def test_summarize_bridge_empty_window_reads_as_zero_samples_not_zero_percent():
    out = hub_surface.summarize_bridge([])
    assert out["sample_count"] == 0
    assert out["branches"]["top_down_override"] == 0.0


def test_summarize_bridge_string_encoded_json_is_parsed():
    rows = [{"self_model_json": '{"voluntary_override_absent_reason": "no_open_loops", "attention_reason": "no_data"}'}]
    out = hub_surface.summarize_bridge(rows)
    assert out["branches"]["no_open_loops"] == 100.0
    assert out["malformed_row_count"] == 0


def test_summarize_durable_lifecycle_counts_distinct_runs_and_raw_events():
    rows = [
        {"run_id": "a", "status": "failed"},
        {"run_id": "a", "status": "resumed"},
        {"run_id": "a", "status": "failed"},
        {"run_id": "a", "status": "resumed"},
        {"run_id": "a", "status": "completed"},
        {"run_id": "b", "status": "completed"},
        {"run_id": "c", "status": "running"},
    ]
    out = hub_surface.summarize_durable_lifecycle(rows)
    assert out["completed_runs"] == 2  # a and b, distinct
    assert out["resumed_events"] == 2  # raw transition count, not distinct-run count
    assert out["resumed_runs"] == 1
    assert out["failed_events"] == 2
    assert out["running_runs"] == 1
    assert out["abandoned_runs"] == 0


def test_pick_example_run_prefers_most_retried_over_most_recent():
    rows = [
        {"run_id": "only-once", "resumed_from_node": "harness_turn"},
        {"run_id": "retried-a-lot", "resumed_from_node": "harness_turn"},
        {"run_id": "retried-a-lot", "resumed_from_node": "harness_turn"},
        {"run_id": "retried-a-lot", "resumed_from_node": "harness_turn"},
    ]
    assert hub_surface.pick_example_run(rows) == "retried-a-lot"


def test_pick_example_run_none_when_nothing_ever_resumed():
    rows = [{"run_id": "a", "resumed_from_node": None}]
    assert hub_surface.pick_example_run(rows) is None


# --------------------------------------------------------------------------
# Route-level: aggregation wired to a fake engine (this repo's established
# convention, see test_attention_loops_reader.py).
# --------------------------------------------------------------------------


def test_bridge_route_normalizes_an_out_of_range_window(monkeypatch):
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine([]))
    out = hub_surface.bridge(minutes=999)
    assert out["window_minutes"] == hub_surface.DEFAULT_WINDOW_MINUTES
    assert out["baseline"] == hub_surface.BRIDGE_BASELINE


def test_durable_runs_route_includes_live_kickoff_flag_and_example(monkeypatch):
    now = datetime.now(timezone.utc)
    lifecycle_rows = [{"run_id": "r1", "status": "completed"}]
    resume_rows = [{"run_id": "r1", "resumed_from_node": "harness_turn"}]
    step_rows = [
        {"node": "harness_turn", "status": "resumed", "next_node": "read_turn_result",
         "resumed_from_node": "harness_turn", "generated_at": now, "detail": {}},
        {"node": "finish", "status": "completed", "next_node": None,
         "resumed_from_node": None, "generated_at": now, "detail": {"reach_out": False}},
    ]
    # One engine instance, reused across both `_engine()` calls the route
    # makes -- a lambda that builds a fresh `_Engine(...)` per call would
    # reset the shared queue back to `lifecycle_rows` on the second call.
    engine = _Engine(lifecycle_rows, resume_rows, step_rows)
    monkeypatch.setattr(hub_surface, "_engine", lambda: engine)
    monkeypatch.setattr(hub_surface.settings, "HUB_CURIOSITY_KICKOFF_VIA_CORTEX", True)

    out = hub_surface.durable_runs(minutes=1440)
    assert out["kickoff_via_cortex"] is True
    assert out["completed_runs"] == 1
    assert out["example"]["run_id"] == "r1"
    assert out["example"]["steps"][1]["detail"]["reach_out"] is False
    assert out["example_scope"] == "all_time"

    # The fake returns whatever's queued regardless of the SQL text or bind
    # params sent -- assert on them directly too, or deleting the `WHERE
    # run_id = :run_id` filter (or typoing the bind key) would still pass.
    third_call_sql = engine.last_conn.executed_sql[-1]
    assert "WHERE run_id = :run_id" in third_call_sql
    assert "ORDER BY generated_at ASC" in third_call_sql


def test_bridge_trend_buckets_by_day_and_carries_the_baseline(monkeypatch):
    day1 = datetime(2026, 9, 6, tzinfo=timezone.utc)
    day2 = datetime(2026, 9, 7, tzinfo=timezone.utc)
    rows = [
        {"day": day1, "self_model_json": {"voluntary_override_absent_reason": "goal_matched_no_loop", "attention_reason": "bottom_up_salience"}},
        {"day": day2, "self_model_json": {"voluntary_override_absent_reason": None, "attention_reason": "top_down_override"}},
        {"day": day2, "self_model_json": {"voluntary_override_absent_reason": None, "attention_reason": "top_down_override"}},
    ]
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine(rows))
    out = hub_surface.bridge_trend()
    assert out["baseline"] == hub_surface.BRIDGE_BASELINE
    assert [d["day"] for d in out["days"]] == ["2026-09-06", "2026-09-07"]
    assert out["days"][0]["branches"]["goal_matched_no_loop"] == 100.0
    assert out["days"][1]["branches"]["top_down_override"] == 100.0
    assert out["days"][1]["sample_count"] == 2


def test_durable_runs_route_has_no_example_when_nothing_resumed(monkeypatch):
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine([{"run_id": "r1", "status": "completed"}], []))
    out = hub_surface.durable_runs(minutes=1440)
    assert out["example"] is None


def test_activity_route_shapes_rows(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [{"generated_at": now, "process": "durable_run", "reason_narrative": "harness_turn:resumed", "attention_reason": "harness_turn:resumed"}]
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine(rows))
    out = hub_surface.activity(limit=10)
    assert out["rows"][0]["process"] == "durable_run"
    assert out["rows"][0]["generated_at"] == now.isoformat()


def test_durable_runs_trend_computes_cumulative_completions_per_day_deduped(monkeypatch):
    day1 = datetime(2026, 9, 6, tzinfo=timezone.utc)
    day2 = datetime(2026, 9, 7, tzinfo=timezone.utc)
    rows = [
        {"day": day1, "run_id": "a"},
        {"day": day2, "run_id": "b"},
        {"day": day2, "run_id": "b"},  # same run appearing twice must not double-count
    ]
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine(rows))
    out = hub_surface.durable_runs_trend()
    series = out["series"]
    assert series == [
        {"day": "2026-09-06", "cumulative_completed": 1},
        {"day": "2026-09-07", "cumulative_completed": 2},
    ]
    assert out["milestones"] == hub_surface.HUB_SURFACE_MILESTONES


def test_recent_attention_route_reflects_fresh_rows(monkeypatch):
    now = datetime.now(timezone.utc)
    rows = [
        {"process": "cortex_turn", "reason_narrative": "just talked to Juniper", "generated_at": now},
    ]
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine(rows))
    out = hub_surface.recent_attention()
    assert out["stale"] is False
    assert len(out["items"]) == 1
    assert out["items"][0]["process"] == "cortex_turn"
    assert out["items"][0]["narrative"] == "just talked to Juniper"
    assert out["items"][0]["age_label"] == "moments ago"


def test_recent_attention_route_reads_as_stale_when_nothing_recent(monkeypatch):
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine([]))
    out = hub_surface.recent_attention()
    assert out["stale"] is True
    assert out["items"] == []


def test_recent_attention_route_fails_open_on_db_error(monkeypatch):
    # Review finding (2026-09-07): hub-surface.js's loadAll() awaits every
    # panel in one Promise.all, so an unhandled exception here would blank
    # out the bridge/durable-runs/activity panels too, not just this one.
    class _ExplodingEngine:
        def connect(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(hub_surface, "_engine", lambda: _ExplodingEngine())
    out = hub_surface.recent_attention()
    assert out["stale"] is True
    assert out["items"] == []
    assert out["mirrors_cortex_exec_defaults"] is True


def test_recent_attention_route_discloses_mirrored_defaults(monkeypatch):
    monkeypatch.setattr(hub_surface, "_engine", lambda: _Engine([]))
    out = hub_surface.recent_attention()
    assert out["mirrors_cortex_exec_defaults"] is True


def test_recent_attention_route_uses_the_same_pure_builder_cortex_exec_uses():
    # Not a mock, not a reimplementation -- this is the actual shared function
    # (orion.substrate.recent_attention_cue.build_recent_attention_cue), the
    # same one services/orion-cortex-exec/app/recent_attention_reader.py
    # calls. Import identity, not just behavior, is the point: this route can
    # never silently drift from what a real chat turn's prompt sees.
    from orion.substrate.recent_attention_cue import build_recent_attention_cue

    assert hub_surface.build_recent_attention_cue is build_recent_attention_cue


def test_recent_attention_query_sql_is_the_shared_constant(monkeypatch):
    engine = _Engine([])
    monkeypatch.setattr(hub_surface, "_engine", lambda: engine)
    hub_surface.recent_attention()
    executed = engine.last_conn.executed_sql[0]
    assert executed == hub_surface.RECENT_ATTENTION_QUERY_SQL
