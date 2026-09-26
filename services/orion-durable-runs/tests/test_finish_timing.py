"""Timing and correlation the runner already knows, carried on the finish
detail (2026-09-22, curiosity-tab redesign patch 1b).

`harness_turn` records what it measured itself around the Hub RPC
(`harness_turn_meta`); `finish_detail` copies it out as
`turn_correlation_id` / `harness_elapsed_sec` / `harness_started_at` /
`harness_finished_at`; the runner's `failed` detail names the node and the
turn's correlation. Every field is additive and optional: a checkpoint
written before the key existed finishes with them absent, never crashes.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from orion.schemas.durable_run import (  # noqa: E402
    CURIOSITY_NODES,
    CuriosityRunBriefV1,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
    DurableRunRequestV1,
)
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS  # noqa: E402

from app.admission_runtime import AdmissionRuntime  # noqa: E402
from app.admitted_graph import GRANTED, WAITING, AdmissionDeps, build_admitted_graph  # noqa: E402
from app.graph import (  # noqa: E402
    Deps,
    HarnessTurnFailed,
    build_curiosity_graph,
    failed_turn_correlation_id,
    failed_turn_meta,
    finish_detail,
    make_nodes,
    recorded_turn_correlation_id,
    turn_correlation_id,
)
from app.runner import DEFAULT_WORKFLOW, DurableRunner, WorkflowSpec  # noqa: E402
from app.self_sense_graph import Deps as SelfSenseDeps  # noqa: E402
from app.self_sense_graph import build_self_sense_graph, finish_detail as self_sense_finish_detail  # noqa: E402

RUN_CORR = "7dcc3944-29bb-5d8f-915f-90f4e6968d47"
TIMING_KEYS = ("turn_correlation_id", "harness_elapsed_sec", "harness_started_at", "harness_finished_at")


# --- fakes ------------------------------------------------------------------


def _lease(generation: int = 1) -> dict[str, Any]:
    # A granted GPU pool hold's ref (stage 4.5).
    return {"lease_id": "hold-1", "generation": generation, "role": "agent", "holder": "durable-runs:run-timing-001"}


def _state(*, leased: bool) -> dict[str, Any]:
    state = {
        "run_id": "run-timing-001", "correlation_id": RUN_CORR, "attempt": 0,
        "brief": {"prompt": "Investigate.", "session_id": "orion_curiosity", "timeout_sec": 100},
    }
    if leased:
        state["lease"] = _lease()
    return state


def _deps(*, turn_ok: bool = True, sleep_sec: float = 0.02, fail_journal: bool = False) -> Deps:
    async def run_turn(req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        await asyncio.sleep(sleep_sec)
        if not turn_ok:
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, ok=False, error="rpc:TimeoutError")
        return CuriosityTurnResultV1(
            run_id=req.run_id, correlation_id=req.correlation_id, text="found it",
            debug={"harness_step_count": 14, "elapsed_sec": 900.0},
        )

    async def read_turn_result(run_id: str) -> dict:
        return {"outcome": {"run_id": run_id, "continue_line": True, "continue_note": "", "reach_out": False, "reach_out_why": ""},
                "footprint": {"Finding": 1}, "hops": [], "evidence_summary": None, "graph_readable": True}

    async def publish_attention_row(facts: dict) -> bool:
        return True

    async def publish_journal(entry) -> str | None:
        if fail_journal:
            raise RuntimeError("bus down")
        return entry.entry_id

    return Deps(run_turn=run_turn, read_turn_result=read_turn_result, publish_attention_row=publish_attention_row, publish_journal=publish_journal)


def _request(run_id: str = "run-timing-001") -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id=run_id, workflow=DEFAULT_WORKFLOW, correlation_id=RUN_CORR,
        brief=CuriosityRunBriefV1(prompt="Pick something.", session_id="orion_curiosity", timeout_sec=3500.0, graph_configured=True,
                                  material={"approved_total": 1, "approved_by_kind": {"semantic": 1}, "crystallization_count": 0, "relation_total": 0, "relation_count": 0}),
    )


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _runner(monkeypatch, deps: Deps) -> tuple[DurableRunner, list[dict[str, Any]]]:
    """A runner with fakes behind an in-memory saver and no bus; `_emit_state`
    is recorded so the detail dicts can be asserted on."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    import app.settings as settings_mod

    settings_mod._settings = None
    saver = InMemorySaver()
    runner = DurableRunner(settings_mod.get_settings(), bus=None, checkpointer=saver)
    runner._workflows[DEFAULT_WORKFLOW] = WorkflowSpec(
        workflow=DEFAULT_WORKFLOW, graph=build_curiosity_graph(deps, saver), nodes=list(CURIOSITY_NODES),
        finish_detail=finish_detail, failed_turn_correlation_id=failed_turn_correlation_id,
    )
    emitted: list[dict[str, Any]] = []

    async def record(state, *, spec, node, status, detail=None, resumed_from=None):
        emitted.append({"node": node, "status": status, "detail": dict(detail or {})})

    monkeypatch.setattr(runner, "_emit_state", record)
    return runner, emitted


def _iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), value
    return parsed


# --- completed detail -------------------------------------------------------


def test_completed_detail_carries_runner_timing_and_the_turn_correlation() -> None:
    graph = build_curiosity_graph(_deps(sleep_sec=0.02), InMemorySaver())
    req = _request()
    initial = {"run_id": req.run_id, "correlation_id": req.correlation_id, "brief": req.brief.model_dump(mode="json"), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg(req.run_id)))
    detail = finish_detail(final)
    for key in TIMING_KEYS:
        assert key in detail, key
    # Unleased: the turn ran under the run's own correlation.
    assert detail["turn_correlation_id"] == RUN_CORR
    # Measured by the runner around the RPC, not copied from Hub's debug (900.0).
    assert isinstance(detail["harness_elapsed_sec"], float)
    assert 0.01 <= detail["harness_elapsed_sec"] < 30.0  # slept 0.02s; loose bounds for a starved runner
    started, finished = _iso(detail["harness_started_at"]), _iso(detail["harness_finished_at"])
    assert finished >= started
    # The existing fields are untouched.
    assert detail["finding_text"] == "found it" and detail["attempts"] == 1 and detail["line"] == "investigate"
    # And the source of truth is checkpointed on the thread, so a resume after
    # the turn still finishes with the numbers the turn actually measured.
    snap = asyncio.run(graph.aget_state(_cfg(req.run_id)))
    assert snap.values["harness_turn_meta"]["turn_correlation_id"] == RUN_CORR


def test_completed_detail_via_the_runner_state_event_carries_the_fields(monkeypatch) -> None:
    runner, emitted = _runner(monkeypatch, _deps())

    async def go() -> None:
        await runner.start_run(_request())
        await asyncio.gather(*runner._active.values(), return_exceptions=True)

    asyncio.run(go())
    completed = [e for e in emitted if e["status"] == "completed"]
    assert len(completed) == 1 and completed[0]["node"] == "finish"
    assert set(TIMING_KEYS) <= set(completed[0]["detail"])
    assert completed[0]["detail"]["turn_correlation_id"] == RUN_CORR


# --- leased vs unleased -----------------------------------------------------


def test_leased_turn_records_the_derived_correlation_and_unleased_records_the_runs() -> None:
    seen: list[CuriosityTurnRequestV1] = []

    async def turn(request: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        seen.append(request)
        return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, text="finding", debug={})

    node = make_nodes(Deps(turn, AsyncMock(), AsyncMock(), AsyncMock()))["harness_turn"]
    leased, unleased = _state(leased=True), _state(leased=False)
    leased_out = asyncio.run(node(leased))
    unleased_out = asyncio.run(node(unleased))

    leased_meta, unleased_meta = leased_out["harness_turn_meta"], unleased_out["harness_turn_meta"]
    assert leased_meta["turn_correlation_id"] == seen[0].correlation_id == turn_correlation_id(leased)
    assert leased_meta["turn_correlation_id"] != RUN_CORR  # fenced per lease
    assert unleased_meta["turn_correlation_id"] == seen[1].correlation_id == RUN_CORR

    # finish_detail reads the recorded value, not a re-derivation: clearing
    # the lease afterwards (what admission does at release) must not flip
    # the reported correlation back to the run's lineage.
    leased_final = {**leased, **leased_out, "lease": None}
    assert finish_detail(leased_final)["turn_correlation_id"] == turn_correlation_id(leased)
    assert finish_detail({**unleased, **unleased_out})["turn_correlation_id"] == RUN_CORR


# --- failed detail ----------------------------------------------------------


def test_failed_turn_detail_names_the_node_and_the_turn_correlation(monkeypatch) -> None:
    runner, emitted = _runner(monkeypatch, _deps(turn_ok=False))

    async def go() -> None:
        await runner.start_run(_request())
        await asyncio.gather(*runner._active.values(), return_exceptions=True)

    asyncio.run(go())
    failed = [e for e in emitted if e["status"] == "failed"]
    assert len(failed) == 1
    detail = failed[0]["detail"]
    assert detail["node"] == "harness_turn"
    assert detail["turn_correlation_id"] == RUN_CORR
    assert detail["error"].startswith("HarnessTurnFailed")


def test_failure_after_the_turn_uses_the_recorded_correlation_and_names_that_node(monkeypatch) -> None:
    runner, emitted = _runner(monkeypatch, _deps(fail_journal=True))

    async def go() -> None:
        await runner.start_run(_request())
        await asyncio.gather(*runner._active.values(), return_exceptions=True)

    asyncio.run(go())
    failed = [e for e in emitted if e["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["detail"]["node"] == "journal"
    assert failed[0]["detail"]["turn_correlation_id"] == RUN_CORR


def test_failed_detail_for_a_leased_thread_is_the_derived_identity() -> None:
    # `_drive` never runs admitted (leased) threads in production -- the
    # runner skips any state with `admission` -- so live this branch always
    # yields the run's own correlation. Pinned anyway so the helper's derive
    # path is the same one `harness_turn` uses if that ever changes.
    spec = SimpleNamespace(failed_turn_correlation_id=failed_turn_correlation_id)
    leased = _state(leased=True)
    detail = DurableRunner._failed_detail(spec, "harness_turn", leased, HarnessTurnFailed("rpc:TimeoutError"))
    assert detail == {"error": "HarnessTurnFailed: rpc:TimeoutError", "node": "harness_turn", "turn_correlation_id": turn_correlation_id(leased)}


def test_failed_detail_never_raises_and_is_bare_for_workflows_that_cannot_name_a_turn() -> None:
    no_corr = SimpleNamespace(failed_turn_correlation_id=None)
    assert DurableRunner._failed_detail(no_corr, "llm_call", {}, RuntimeError("x")) == {"error": "RuntimeError: x", "node": "llm_call"}

    def boom(_state: dict[str, Any]) -> str | None:
        raise KeyError("run_id")

    exploding = SimpleNamespace(failed_turn_correlation_id=boom)
    assert DurableRunner._failed_detail(exploding, "harness_turn", {}, RuntimeError("x")) == {"error": "RuntimeError: x", "node": "harness_turn"}
    # The bare state the runner falls back to when no snapshot exists at all:
    # an empty lineage derives to "" and is dropped, never emitted as "".
    bare = {"run_id": "r", "correlation_id": ""}
    assert failed_turn_correlation_id(bare) is None
    spec = SimpleNamespace(failed_turn_correlation_id=failed_turn_correlation_id)
    assert "turn_correlation_id" not in DurableRunner._failed_detail(spec, "harness_turn", bare, RuntimeError("x"))
    assert failed_turn_correlation_id({}) is None


# --- admitted path ----------------------------------------------------------


class _AdmittedWorld:
    """Mirror of test_admitted_graph's fake pool, trimmed to what this needs."""

    def __init__(self, *, fail: bool, max_attempts: int = 1):
        self.now = datetime(2026, 9, 22, tzinfo=timezone.utc)
        self.fail = fail
        self.max_attempts = max_attempts
        self.current_lease = None

    def grant(self, generation: int = 1) -> None:
        self.current_lease = {**_lease(generation), "holder": "durable-runs:study-001"}

    async def turn(self, req):
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="" if self.fail else "finding", ok=not self.fail)

    async def read(self, run_id):
        return {"graph_readable": True, "hops": []}

    async def row(self, facts):
        return True

    async def journal(self, entry):
        return entry.entry_id

    async def register(self, state):
        return {"status": "waiting_resource", "hold": {"request_id": "study-001:1", "lease_id": "hold-1"}}

    async def lease(self, state):
        if self.current_lease:
            return GRANTED, {"status": "admitted", "lease": dict(self.current_lease)}
        return WAITING, {"status": "waiting_resource", "lease": None}

    async def execute(self, state, node):
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        self.current_lease = None
        return {"lease": None, "hold": None}

    async def event(self, *args):
        pass

    def graph(self, saver):
        return build_admitted_graph(
            Deps(self.turn, self.read, self.row, self.journal),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event, now=lambda: self.now, max_attempts=self.max_attempts),
            saver,
        )


def _admitted_initial() -> dict[str, Any]:
    return {"run_id": "study-001", "correlation_id": "trace-001", "attempt": 0, "admission": {"resource": "llm.route.agent"},
            "brief": {"prompt": "Study evidence.", "session_id": "curiosity", "timeout_sec": 0.05}}


def test_admitted_failure_keeps_the_fenced_correlation_after_the_lease_is_cleared() -> None:
    world = _AdmittedWorld(fail=True)
    world.grant()
    graph = world.graph(InMemorySaver())
    cfg = _cfg("study-001")
    final = asyncio.run(graph.ainvoke(_admitted_initial(), cfg))
    assert final["status"] == "failed" and final["lease"] is None
    expected = turn_correlation_id({**_admitted_initial(), "lease": {**_lease(), "holder": "durable-runs:study-001"}})
    assert final["harness_turn_meta"] == {"turn_correlation_id": expected}
    assert expected != "trace-001"
    # The runtime's terminal detail reads only the recorded value -- a fresh
    # derivation here would name the run's lineage, since the lease is gone.
    assert recorded_turn_correlation_id(final) == expected
    runtime = object.__new__(AdmissionRuntime)
    seen: list[tuple[str, str, dict[str, Any]]] = []

    async def finish_projection(run_id, status, detail):
        seen.append((run_id, status, detail))
        return status

    runtime.store = SimpleNamespace(finish_projection=finish_projection)
    runtime._wake = asyncio.Event()
    runtime.outreach, runtime._hints, runtime._checked = {}, set(), {}
    asyncio.run(runtime._terminal("study-001", "failed", final))
    assert seen == [("study-001", "failed", {"error": final["last_error"], "turn_correlation_id": expected})]


def _fenced(generation: int) -> str:
    return turn_correlation_id({**_admitted_initial(), "lease": {**_lease(generation), "holder": "durable-runs:study-001"}})


def test_retry_under_a_new_lease_generation_replaces_the_stale_generations_id() -> None:
    """Attempt 1 fails under generation 1 (the `retrying` return carries the
    gen-1 id); the retry is granted generation 2 and succeeds -- the recorded
    id must be gen-2's, never the fenced gen-1's, and the finish detail's
    full timing describes the attempt that actually completed."""
    async def scenario() -> None:
        world = _AdmittedWorld(fail=True, max_attempts=2)
        world.grant(1)
        graph = world.graph(InMemorySaver())
        cfg = _cfg("study-001")
        await graph.ainvoke(_admitted_initial(), cfg)
        snap = await graph.aget_state(cfg)
        assert snap.values["status"] == "retrying" and snap.next == ("retry_wait",)
        assert snap.values["lease"] is None
        assert snap.values["harness_turn_meta"] == {"turn_correlation_id": _fenced(1)}

        world.now += timedelta(seconds=31)
        world.fail = False
        world.grant(2)
        final = await graph.ainvoke(Command(resume=True), cfg)
        assert final["status"] == "completed"
        meta = final["harness_turn_meta"]
        assert meta["turn_correlation_id"] == _fenced(2) != _fenced(1)
        detail = finish_detail(final)
        assert detail["turn_correlation_id"] == _fenced(2)
        assert set(TIMING_KEYS) <= set(detail)

    asyncio.run(scenario())


def test_retry_that_fails_again_under_generation_two_reports_generation_two() -> None:
    async def scenario() -> None:
        world = _AdmittedWorld(fail=True, max_attempts=2)
        world.grant(1)
        graph = world.graph(InMemorySaver())
        cfg = _cfg("study-001")
        await graph.ainvoke(_admitted_initial(), cfg)
        world.now += timedelta(seconds=31)
        world.grant(2)
        final = await graph.ainvoke(Command(resume=True), cfg)
        assert final["status"] == "failed"
        assert recorded_turn_correlation_id(final) == _fenced(2)

    asyncio.run(scenario())


def test_failed_turn_meta_is_empty_for_a_malformed_lease_so_the_wrapper_still_takes_its_failure_path() -> None:
    assert failed_turn_meta({**_admitted_initial(), "lease": {"lane": "agent"}}) == {}
    assert failed_turn_meta({}) == {}
    assert failed_turn_meta({"run_id": "r", "correlation_id": ""}) == {}
    assert failed_turn_meta(_state(leased=True)) == {"harness_turn_meta": {"turn_correlation_id": turn_correlation_id(_state(leased=True))}}


def test_worker_recovery_fence_records_the_fenced_generation_before_clearing_the_lease() -> None:
    """The one path where a stale id could otherwise surface: a recovering
    driver fences an in-flight harness_turn (lease -> None) and the run later
    dies at the deadline. `_terminal` must then name the fenced generation."""
    import app.admission_runtime as ar

    fenced_state = {**_admitted_initial(), "lease": {**_lease(3), "holder": "durable-runs:study-001"}, "attempt": 1,
                    "harness_turn_meta": {"turn_correlation_id": "older-attempt"}}
    updates: list[dict[str, Any]] = []

    class _Graph:
        async def aupdate_state(self, cfg, values, as_node=None):
            updates.append(dict(values))

    # Exercise exactly the fence branch's payload, as the runtime builds it.
    payload = {"lease": None, "turn_fence": 1, "status": "retrying", "retry_node": None, **ar.failed_turn_meta(fenced_state)}
    asyncio.run(_Graph().aupdate_state(None, payload, as_node="retry_wait"))
    assert updates[0]["lease"] is None
    assert updates[0]["harness_turn_meta"] == {"turn_correlation_id": turn_correlation_id(fenced_state)}
    assert updates[0]["harness_turn_meta"]["turn_correlation_id"] != "older-attempt"
    # And the runtime source really uses it in that arm (not just importable).
    src = Path(ar.__file__).read_text()
    arm = src.split("async def _recover(")[1].split("if workflow == DEFAULT_WORKFLOW:")[1].split("else:")[0]
    assert "failed_turn_meta(state)" in arm and 'as_node="retry_wait"' in arm


def test_admitted_success_carries_full_timing_to_the_finish_detail() -> None:
    world = _AdmittedWorld(fail=False)
    world.grant()
    graph = world.graph(InMemorySaver())
    final = asyncio.run(graph.ainvoke(_admitted_initial(), _cfg("study-001")))
    assert final["status"] == "completed"
    detail = finish_detail(final)
    assert set(TIMING_KEYS) <= set(detail)
    assert detail["turn_correlation_id"] != "trace-001"
    assert detail["harness_elapsed_sec"] >= 0.0


# --- timing unavailable -----------------------------------------------------


def test_a_checkpoint_from_before_the_key_finishes_without_the_fields() -> None:
    """Old thread resumed after deploy: `harness_turn` already ran, so no
    `harness_turn_meta` will ever be written. The detail must be the
    pre-patch shape -- fields absent, not null, not a crash."""
    old = {**_state(leased=False), "text": "found it", "debug": {"harness_step_count": 3}, "outcome": {"reach_out": False},
           "journal_entry_id": "j1", "attempt": 1, "status": "completed"}
    detail = finish_detail(old)
    for key in TIMING_KEYS:
        assert key not in detail, key
    assert detail["finding_text"] == "found it"
    # A leased pre-key checkpoint still stamped `debug.turn_correlation_id`;
    # that older breadcrumb is honoured so the join works for it too.
    old_leased = {**old, "debug": {"turn_correlation_id": "derived-abc", "parent_correlation_id": RUN_CORR}}
    detail = finish_detail(old_leased)
    assert detail["turn_correlation_id"] == "derived-abc"
    assert "harness_elapsed_sec" not in detail


def test_finish_detail_tolerates_bare_and_malformed_meta() -> None:
    assert finish_detail({})["line"] == "investigate"
    assert "turn_correlation_id" not in finish_detail({})
    for junk in (None, "str", 3, [], {"turn_correlation_id": None, "harness_elapsed_sec": None}):
        detail = finish_detail({"harness_turn_meta": junk, "debug": junk})
        for key in TIMING_KEYS:
            assert key not in detail, (junk, key)


# --- self-sense-eval --------------------------------------------------------


def test_self_sense_finish_detail_names_each_questions_turn_and_its_timing() -> None:
    seen: list[CuriosityTurnRequestV1] = []

    async def run_turn(request: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        seen.append(request)
        return CuriosityTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id, text="an answer", ok=True)

    async def publish_rows(rows) -> tuple[int, int]:
        return len(rows), 0

    graph = build_self_sense_graph(SelfSenseDeps(run_turn=run_turn, publish_rows=publish_rows), InMemorySaver())
    initial = {"run_id": "sse-001", "correlation_id": RUN_CORR, "workflow": "self_sense_eval", "attempt": 0,
               "brief": {"questions": list(SELF_SENSE_QUESTIONS), "timeout_sec": 5.0}}
    final = asyncio.run(graph.ainvoke(initial, _cfg("sse-001")))
    detail = self_sense_finish_detail(final)
    turns = detail["turns"]
    assert set(turns) == {key for key, _ in SELF_SENSE_QUESTIONS}
    by_corr = {r.correlation_id: r for r in seen}
    for entry in turns.values():
        assert set(TIMING_KEYS) <= set(entry)
        assert entry["turn_correlation_id"] in by_corr  # joinable to the real turn
        assert entry["harness_elapsed_sec"] >= 0.0
        assert _iso(entry["harness_finished_at"]) >= _iso(entry["harness_started_at"])
    # Pre-existing counters unchanged.
    assert detail["published"] == len(SELF_SENSE_QUESTIONS) and detail["failed"] == 0


def test_self_sense_finish_detail_tolerates_answers_recorded_before_timing() -> None:
    old = {"answers": {"q1": {"text": "x", "debug": {}, "correlation_id": "c1"}, "q2": "junk", "q3": {"text": ""}},
           "published": 1, "failed": 0, "empty": 0, "attempt": 1}
    turns = self_sense_finish_detail(old)["turns"]
    assert turns == {"q1": {"turn_correlation_id": "c1"}}
    assert self_sense_finish_detail({})["turns"] == {}
