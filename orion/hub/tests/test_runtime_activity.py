"""RuntimeActivity folds real event shapes into one honest snapshot."""
from __future__ import annotations

import asyncio

import pytest

from orion.hub.runtime_activity import (
    RuntimeActivity,
    get_runtime_activity,
    lane_for_model_label,
    reset_runtime_activity,
    summarize_step,
)


class Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _activity(**kw) -> tuple[RuntimeActivity, Clock]:
    clock = Clock()
    return RuntimeActivity(now=clock, **kw), clock


def test_curiosity_run_lifecycle_dispatch_to_completed():
    act, clock = _activity()
    act.run_dispatched(run_id="r1", correlation_id="c1", line="self_inquiry")
    snap = act.snapshot()
    assert snap["busy"] is True
    run = snap["curiosity_runs"][0]
    assert run["status"] == "dispatched" and run["active"] is True and run["line"] == "self_inquiry"

    clock.t += 30
    act.run_state(
        {
            "run_id": "r1",
            "workflow": "curiosity.investigate",
            "node": "harness_turn",
            "next_node": "read_turn_result",
            "status": "running",
            "correlation_id": "c1",
            "generated_at": "2026-09-09T00:00:30Z",
        }
    )
    run = act.snapshot()["curiosity_runs"][0]
    assert run["node"] == "harness_turn" and run["status"] == "running" and run["active"]
    # started_at is the dispatch moment, so duration counts the whole wait.
    assert run["duration_sec"] == 30.0

    clock.t += 600
    act.run_state(
        {
            "run_id": "r1",
            "node": "finish",
            "status": "completed",
            "correlation_id": "c1",
            "detail": {"line": "self_inquiry", "reach_out": False, "finding_text": "x" * 500, "self_definition": "SECRET"},
        }
    )
    snap = act.snapshot()
    run = snap["curiosity_runs"][0]
    assert run["active"] is False and run["status"] == "completed"
    assert run["finish"]["reach_out"] is False
    assert len(run["finish"]["finding_text"]) == 240
    assert "self_definition" not in run["finish"]
    assert snap["busy"] is False
    assert [t["status"] for t in run["transitions"]] == ["running", "completed"]


def test_run_state_alone_creates_a_run_and_failed_keeps_the_error():
    act, _ = _activity()
    act.run_state({"run_id": "r9", "correlation_id": "c9", "node": "harness_turn", "status": "running"})
    act.run_state({"run_id": "r9", "node": "harness_turn", "status": "failed", "detail": {"error": "RuntimeError: boom"}})
    run = act.snapshot()["curiosity_runs"][0]
    assert run["status"] == "failed" and run["error"] == "RuntimeError: boom" and not run["active"]


def test_run_joins_its_harness_turn_by_correlation_id():
    act, clock = _activity()
    act.run_dispatched(run_id="r1", correlation_id="c1", line="investigate")
    act.turn_requested(correlation_id="c1", mode="orion", model_label="MODEL_AGENT", source="curiosity_investigation")
    lane = lane_for_model_label("MODEL_AGENT")
    run = act.snapshot()["curiosity_runs"][0]
    assert run["turn"]["lane"] == lane
    assert run["turn"]["phase"] == "queued"
    clock.t += 12
    act.harness_step({"correlation_id": "c1", "step_index": -1, "step": {"_cockpit": "boot", "prompt": "never shown"}})
    run = act.snapshot()["curiosity_runs"][0]
    assert run["turn"]["phase"] == "running"
    assert run["turn"]["queued_sec"] == 12.0
    assert run["turn"]["recent_steps"] == ["motor boot"]
    assert "never shown" not in str(run)


def test_lane_queue_running_is_evidence_based_not_request_based():
    act, clock = _activity()
    act.turn_requested(correlation_id="a", mode="orion", model_label=None, source="chat")
    act.turn_requested(correlation_id="b", mode="orion", model_label=None, source="chat")
    lanes = act.snapshot()["lanes"]
    assert lanes["chat"]["running"] == []
    assert [t["correlation_id"] for t in lanes["chat"]["queued"]] == ["a", "b"]
    assert lanes["agent"] == {"running": [], "queued": [], "recent": []}

    act.harness_step(
        {
            "correlation_id": "a",
            "step_index": 0,
            "step": {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}},
        }
    )
    lanes = act.snapshot()["lanes"]
    assert [t["correlation_id"] for t in lanes["chat"]["running"]] == ["a"]
    assert lanes["chat"]["running"][0]["recent_steps"] == ["tool Read"]
    assert lanes["chat"]["running"][0]["step_count"] == 1
    assert [t["correlation_id"] for t in lanes["chat"]["queued"]] == ["b"]

    class Run:
        compliance_verdict = "completed"
        fcc_elapsed_sec = 41.5
        fcc_served_model = "qwen"
        step_count = 7

    clock.t += 50
    act.turn_finished(correlation_id="a", run=Run())
    lanes = act.snapshot()["lanes"]
    assert lanes["chat"]["running"] == []
    recent = lanes["chat"]["recent"][0]
    assert recent["ok"] is True and recent["step_count"] == 7 and recent["served_model"] == "qwen"
    assert recent["elapsed_sec"] == 50.0

    act.turn_finished(correlation_id="b", run=None, error="rpc_timeout")
    recent = {t["correlation_id"]: t for t in act.snapshot()["lanes"]["chat"]["recent"]}
    assert recent["b"]["ok"] is False and recent["b"]["error"] == "rpc_timeout"
    assert act.snapshot()["busy"] is False


def test_step_for_an_unknown_turn_is_kept_on_the_unknown_lane():
    act, _ = _activity()
    act.harness_step({"correlation_id": "ghost", "step_index": 3, "step": {}})
    lanes = act.snapshot()["lanes"]
    assert [t["correlation_id"] for t in lanes["unknown"]["running"]] == ["ghost"]
    assert lanes["unknown"]["running"][0]["step_count"] == 4


def test_finished_turns_and_runs_expire():
    act, clock = _activity(finished_ttl_sec=60, max_finished=2)
    for i in range(4):
        act.turn_requested(correlation_id=f"t{i}", mode="orion", model_label=None, source="chat")
        act.turn_finished(correlation_id=f"t{i}", run=None, error="x")
        act.run_state({"run_id": f"r{i}", "correlation_id": f"t{i}", "status": "completed", "node": "finish"})
    snap = act.snapshot()
    assert len(snap["lanes"]["chat"]["recent"]) == 2
    assert len(snap["curiosity_runs"]) == 2
    clock.t += 61
    act.turn_requested(correlation_id="live", mode="orion", model_label=None, source="chat")
    snap = act.snapshot()
    assert snap["lanes"]["chat"]["recent"] == []
    assert snap["curiosity_runs"] == []


def test_dispatched_run_with_no_state_event_ages_out():
    act, clock = _activity(dispatched_stale_sec=100)
    act.run_dispatched(run_id="orphan", correlation_id="c", line="investigate")
    assert act.snapshot()["curiosity_runs"][0]["run_id"] == "orphan"
    clock.t += 50
    # Still within the staleness window -- not evicted just for sitting quietly.
    assert [r["run_id"] for r in act.snapshot()["curiosity_runs"]] == ["orphan"]
    clock.t += 51
    # Never got a first state event (runner never started, or died before
    # its first transition) -- presumed abandoned, dropped outright rather
    # than lingering as a permanent phantom "active" run.
    assert act.snapshot()["curiosity_runs"] == []
    assert act.snapshot()["busy"] is False


def test_a_state_event_resets_the_dispatched_staleness_clock():
    act, clock = _activity(dispatched_stale_sec=100)
    act.run_dispatched(run_id="r1", correlation_id="c", line="investigate")
    clock.t += 90
    act.run_state({"run_id": "r1", "correlation_id": "c", "status": "running", "node": "harness_turn"})
    clock.t += 90
    # 180s since dispatch, but only 90s since the real "running" status --
    # the staleness rule only applies while status is still bare "dispatched".
    assert [r["run_id"] for r in act.snapshot()["curiosity_runs"]] == ["r1"]


def test_backfill_adopts_only_still_active_runs():
    act, _ = _activity()
    adopted = act.backfill_runs(
        [
            {"run_id": "old", "status": "completed", "node": "finish", "correlation_id": "c"},
            {"run_id": "live", "status": "resumed", "node": "harness_turn", "correlation_id": "c2",
             "generated_at": "2026-09-09T01:00:00Z", "first_seen_at": "2026-09-09T00:30:00Z"},
        ]
    )
    assert adopted == 1
    runs = act.snapshot()["curiosity_runs"]
    assert [r["run_id"] for r in runs] == ["live"]
    assert runs[0]["started_at"] == "2026-09-09T00:30:00Z"
    assert runs[0]["transitions"][0]["backfilled"] is True
    # A second backfill never overwrites what live events already said.
    assert act.backfill_runs([{"run_id": "live", "status": "running", "node": "journal"}]) == 0


def test_backfill_recovers_line_from_the_row_detail_column():
    act, _ = _activity()
    # `detail` comes back from psycopg as a dict for JSONB, or a JSON string
    # depending on driver/registration -- both are handled.
    act.backfill_runs(
        [
            {"run_id": "r1", "status": "running", "node": "harness_turn", "correlation_id": "c1",
             "detail": {"line": "self_inquiry"}},
            {"run_id": "r2", "status": "running", "node": "harness_turn", "correlation_id": "c2",
             "detail": "{\"line\": \"investigate\"}"},
            {"run_id": "r3", "status": "running", "node": "harness_turn", "correlation_id": "c3", "detail": None},
        ]
    )
    by_id = {r["run_id"]: r for r in act.snapshot()["curiosity_runs"]}
    assert by_id["r1"]["line"] == "self_inquiry"
    assert by_id["r2"]["line"] == "investigate"
    assert by_id["r3"]["line"] is None


def test_gateway_snapshot_and_error_are_both_reported():
    act, _ = _activity()
    act.gateway_admission({"upstreams": {"http://a": {"inflight": 1, "waiting": 0, "max_inflight": 2}}})
    g = act.snapshot()["gateway"]
    assert g["snapshot"]["upstreams"]["http://a"]["inflight"] == 1 and g["error"] is None
    act.gateway_admission(None, error="connect_failed")
    g = act.snapshot()["gateway"]
    assert g["error"] == "connect_failed"
    # Last good snapshot stays visible next to the error, not blanked.
    assert g["snapshot"]["upstreams"]["http://a"]["inflight"] == 1


@pytest.mark.asyncio
async def test_every_fold_wakes_subscribers_and_a_full_queue_does_not_raise():
    act, _ = _activity()
    q = act.subscribe()
    act.turn_requested(correlation_id="a", mode="orion", model_label=None, source="chat")
    assert await asyncio.wait_for(q.get(), 1) == 1
    for _ in range(20):
        act.gateway_admission({})
    assert act.version == 21
    act.unsubscribe(q)


def test_summarize_step_never_leaks_prompt_text():
    assert summarize_step({"_cockpit": "boot", "prompt": "p" * 1000}, index=-1) == "motor boot"
    assert summarize_step({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]}}, index=0) == "tool Bash"
    assert summarize_step({}, index=0) != ""
    assert summarize_step("not a dict", index=0) == "step"  # type: ignore[arg-type]


def test_singleton_reset_gives_tests_a_clean_instance():
    a = get_runtime_activity()
    a.turn_requested(correlation_id="x", mode="orion", model_label=None, source="chat")
    b = reset_runtime_activity()
    assert b is get_runtime_activity() and b is not a
    assert b.snapshot()["lanes"]["chat"]["queued"] == []
