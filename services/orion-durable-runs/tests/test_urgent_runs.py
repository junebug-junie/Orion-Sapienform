"""Urgent curiosity runs through durable-runs (Plan 3 Task 3).

An urgent brief carries its seed onto the Hub turn request, retries at most
twice with a short backoff, and reads back the run's `:IncidentReport`; the
finish detail names the incident, the report, and a flag when there is no
structured verdict. Ordinary runs are unchanged -- every urgent key is absent.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from orion.curiosity.incident_report import LABEL_INCIDENT_REPORT, NO_STRUCTURED_VERDICT  # noqa: E402
from orion.schemas.curiosity_urgent import CuriosityUrgentSeedV1  # noqa: E402
from orion.schemas.durable_run import CuriosityRunBriefV1, CuriosityTurnRequestV1, CuriosityTurnResultV1  # noqa: E402

from app.admitted_graph import GRANTED, AdmissionDeps, build_admitted_graph  # noqa: E402
from app.graph import Deps, finish_detail, make_nodes  # noqa: E402
from app.runner import DurableRunner  # noqa: E402
from app.settings import Settings  # noqa: E402

RUN_ID = "abc123def456"
URGENT_KEYS = ("urgent", "incident_report", "report_flag")
NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _seed() -> CuriosityUrgentSeedV1:
    return CuriosityUrgentSeedV1(
        incident_id="0123456789abcdef",
        question="Is athena actually overheating?",
        trigger="manual",
        subject="athena",
        evidence={"cooling": {"cabinet_temp_c": 41.5}},
        requested_at=NOW,
    )


def _brief(*, urgent: bool) -> dict[str, Any]:
    brief = CuriosityRunBriefV1(prompt="Investigate.", session_id="orion_curiosity", timeout_sec=900.0,
                                urgent=_seed() if urgent else None)
    return brief.model_dump(mode="json")


def _state(*, urgent: bool) -> dict[str, Any]:
    return {"run_id": RUN_ID, "correlation_id": "trace-urgent", "attempt": 0, "brief": _brief(urgent=urgent)}


def _report_row(**overrides: Any) -> dict[str, Any]:
    row = {"node_id": 7, "run_id": RUN_ID, "incident_id": "0123456789abcdef", "is_real": "real",
           "likely_cause": "intake fan stalled", "evidence": "cabinet_temp_c rose 6C in 20 min",
           "severity": "high", "operator_action": "check the intake fan", "confidence": 0.8,
           "written_at": 1_790_000_000_000}
    row.update(overrides)
    return row


class FakeReader:
    """Answers every worldview read with no rows; IncidentReport queries get `report_rows`."""

    def __init__(self, report_rows: list[dict[str, Any]] | None = None, *, fail_report: bool = False):
        self.report_rows = report_rows or []
        self.fail_report = fail_report
        self.queries: list[str] = []

    def query(self, cypher: str) -> list[dict[str, Any]]:
        self.queries.append(cypher)
        if f":{LABEL_INCIDENT_REPORT}" in cypher:
            if self.fail_report:
                raise RuntimeError("graph went away")
            return list(self.report_rows)
        return []

    def saw_incident_query(self) -> bool:
        return any(f":{LABEL_INCIDENT_REPORT}" in q for q in self.queries)


def _runner(reader: FakeReader | None) -> DurableRunner:
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    runner = DurableRunner(settings, bus=None, checkpointer=None)
    runner._reader = reader  # type: ignore[assignment]
    return runner


# --- harness_turn carries the seed ------------------------------------------


def _turn_deps(sent: list[CuriosityTurnRequestV1], reads: list[tuple[str, dict]] | None = None) -> Deps:
    async def run_turn(req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        sent.append(req)
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="verdict: real")

    async def read_turn_result(run_id: str, **kwargs: Any) -> dict:
        if reads is not None:
            reads.append((run_id, kwargs))
        return {"graph_readable": True, "incident_report": None, "report_flag": NO_STRUCTURED_VERDICT}

    async def row(facts: dict) -> bool:
        return True

    async def journal(entry) -> str | None:
        return entry.entry_id

    return Deps(run_turn=run_turn, read_turn_result=read_turn_result, publish_attention_row=row, publish_journal=journal)


def test_harness_turn_copies_the_seed_onto_the_turn_request():
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_turn_deps(sent))["harness_turn"](_state(urgent=True)))
    [req] = sent
    assert req.urgent == _seed()
    wire = req.model_dump(mode="json", exclude_none=True)
    assert wire["urgent"]["incident_id"] == "0123456789abcdef"
    assert wire["urgent"]["question"] == "Is athena actually overheating?"


def test_ordinary_turn_request_has_no_urgent_key_on_the_wire():
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_turn_deps(sent))["harness_turn"](_state(urgent=False)))
    [req] = sent
    assert req.urgent is None
    assert "urgent" not in req.model_dump(mode="json", exclude_none=True)


def test_read_node_asks_for_the_report_only_on_urgent_runs():
    reads: list[tuple[str, dict]] = []
    nodes = make_nodes(_turn_deps([], reads))
    urgent_out = asyncio.run(nodes["read_turn_result"](_state(urgent=True)))
    ordinary_out = asyncio.run(nodes["read_turn_result"](_state(urgent=False)))
    assert reads == [(RUN_ID, {"urgent": True}), (RUN_ID, {})]
    assert urgent_out["report_flag"] == NO_STRUCTURED_VERDICT and urgent_out["incident_report"] is None
    assert not any(key in ordinary_out for key in ("incident_report", "report_flag"))


# --- _read_turn_result reads the IncidentReport ------------------------------


def test_read_turn_result_returns_the_report_for_an_urgent_run():
    reader = FakeReader([_report_row(confidence=1.7)])
    found = asyncio.run(_runner(reader)._read_turn_result(RUN_ID, urgent=True))
    assert reader.saw_incident_query()
    assert found["report_flag"] is None
    report = found["incident_report"]
    assert report == {
        "incident_id": "0123456789abcdef", "is_real": "real", "likely_cause": "intake fan stalled",
        "evidence": ["cabinet_temp_c rose 6C in 20 min"], "severity": "high",
        "operator_action": "check the intake fan", "confidence": 1.0,
    }


def test_read_turn_result_flags_a_missing_report():
    found = asyncio.run(_runner(FakeReader([]))._read_turn_result(RUN_ID, urgent=True))
    assert found["incident_report"] is None and found["report_flag"] == NO_STRUCTURED_VERDICT


def test_read_turn_result_never_raises_when_the_report_read_fails():
    found = asyncio.run(_runner(FakeReader(fail_report=True))._read_turn_result(RUN_ID, urgent=True))
    assert found["incident_report"] is None and found["report_flag"] == NO_STRUCTURED_VERDICT


def test_read_turn_result_never_raises_when_the_whole_read_fails(monkeypatch):
    import app.runner as runner_mod

    def boom(reader, run_id):
        raise RuntimeError("worldview exploded")

    monkeypatch.setattr(runner_mod, "read_turn_outcome", boom)
    found = asyncio.run(_runner(FakeReader([_report_row()]))._read_turn_result(RUN_ID, urgent=True))
    assert found["incident_report"] is None and found["report_flag"] == NO_STRUCTURED_VERDICT
    assert found["graph_readable"] is False


def test_read_turn_result_without_a_graph_flags_urgent_runs():
    found = asyncio.run(_runner(None)._read_turn_result(RUN_ID, urgent=True))
    assert found["incident_report"] is None and found["report_flag"] == NO_STRUCTURED_VERDICT


def test_ordinary_read_never_queries_the_incident_report():
    reader = FakeReader([_report_row()])
    found = asyncio.run(_runner(reader)._read_turn_result(RUN_ID))
    assert reader.queries, "the ordinary reads still ran"
    assert not reader.saw_incident_query()
    assert not any(key in found for key in ("incident_report", "report_flag"))


# --- finish_detail -------------------------------------------------------------


def test_finish_detail_carries_the_incident_and_the_report_for_urgent_runs():
    report = {"incident_id": "0123456789abcdef", "is_real": "real", "likely_cause": "fan",
              "evidence": ["reading"], "severity": "high", "operator_action": "check fan", "confidence": 0.8}
    state = {**_state(urgent=True), "text": "It is real.", "attempt": 1,
             "incident_report": report, "report_flag": None}
    detail = finish_detail(state)
    assert detail["urgent"] == {
        "incident_id": "0123456789abcdef", "trigger": "manual", "subject": "athena",
        "question": "Is athena actually overheating?", "requested_at": "2026-09-28T12:00:00Z",
    }
    assert detail["incident_report"] == report
    assert detail["report_flag"] is None
    assert "evidence" not in detail["urgent"], "the seed's evidence bundle stays in the brief"


def test_finish_detail_flags_an_urgent_run_without_a_report():
    detail = finish_detail({**_state(urgent=True), "text": "prose only"})
    assert detail["incident_report"] is None
    assert detail["report_flag"] == NO_STRUCTURED_VERDICT


def test_finish_detail_for_ordinary_runs_has_no_urgent_keys():
    detail = finish_detail({**_state(urgent=False), "text": "a finding"})
    assert not any(key in detail for key in URGENT_KEYS)


# --- retry budget ----------------------------------------------------------------


class RetryWorld:
    """A pool that always grants; the turn (or the journal) always fails."""

    def __init__(self, *, fail_turn: bool = True, fail_journal: bool = False):
        self.now = NOW
        self.fail_turn = fail_turn
        self.fail_journal = fail_journal
        self.turns = 0
        self.journals = 0

    async def turn(self, req):
        self.turns += 1
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id,
                                    text="" if self.fail_turn else "finding", ok=not self.fail_turn)

    async def read(self, run_id, **kwargs):
        return {"graph_readable": True}

    async def row(self, facts):
        return True

    async def journal(self, entry):
        self.journals += 1
        if self.fail_journal:
            raise RuntimeError("bus down")
        return entry.entry_id

    async def register(self, state):
        return {"status": "waiting_resource", "hold": {"request_id": f"{RUN_ID}:1", "lease_id": "hold-1"}, "hold_seq": 1}

    async def lease(self, state):
        return GRANTED, {"status": "admitted", "hold": state.get("hold"),
                         "lease": {"lease_id": "hold-1", "generation": 1, "role": "agent", "holder": "durable-runs:x"}}

    async def execute(self, state, node):
        return await node(state)

    async def release(self, state, reason, keep_requeued=False):
        return {"lease": None, "hold": None}

    async def event(self, state, name, detail):
        return None

    def graph(self, saver):
        return build_admitted_graph(
            Deps(self.turn, self.read, self.row, self.journal),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, max_attempts=3, retry_base_seconds=30.0, retry_max_seconds=300.0),
            saver,
        )


def _admitted_initial(*, urgent: bool) -> dict[str, Any]:
    return {**_state(urgent=urgent), "admission": {"resource": "llm.route.agent"}, "status": "queued"}


CFG = {"configurable": {"thread_id": RUN_ID}}


def _retry_delays(world: RetryWorld, *, urgent: bool) -> tuple[list[float], dict[str, Any]]:
    """Drive the admitted graph through every retry; the backoff each retry scheduled, and the end state."""

    async def scenario():
        graph = world.graph(InMemorySaver())
        await graph.ainvoke(_admitted_initial(urgent=urgent), CFG)
        delays: list[float] = []
        for _ in range(10):
            snap = await graph.aget_state(CFG)
            if not snap.next:
                return delays, dict(snap.values)
            retry_at = datetime.fromisoformat(snap.values["retry_at"])
            delays.append((retry_at - world.now).total_seconds())
            world.now = retry_at
            await graph.ainvoke(Command(resume=True), CFG)
        raise AssertionError("graph never terminated")

    return asyncio.run(scenario())


def test_urgent_turn_retries_once_after_ten_seconds_then_fails():
    world = RetryWorld()
    delays, end = _retry_delays(world, urgent=True)
    assert delays == [10.0]
    assert world.turns == 2
    assert end["status"] == "failed" and end["attempt"] == 2
    assert end["last_error"].startswith("HarnessTurnFailed")


def test_ordinary_turn_keeps_three_attempts_and_thirty_second_backoff():
    world = RetryWorld()
    delays, end = _retry_delays(world, urgent=False)
    assert delays == [30.0, 60.0]
    assert world.turns == 3
    assert end["status"] == "failed" and end["attempt"] == 3


def test_urgent_budget_never_exceeds_the_service_budget():
    world = RetryWorld()

    def graph(saver):
        return build_admitted_graph(
            Deps(world.turn, world.read, world.row, world.journal),
            AdmissionDeps(world.register, world.lease, world.execute, world.release, world.event,
                          now=lambda: world.now, max_attempts=1),
            saver,
        )

    world.graph = graph  # type: ignore[method-assign]
    delays, end = _retry_delays(world, urgent=True)
    assert delays == [] and world.turns == 1 and end["status"] == "failed"


def test_urgent_tail_node_uses_the_same_tight_budget():
    world = RetryWorld(fail_turn=False, fail_journal=True)
    delays, end = _retry_delays(world, urgent=True)
    assert delays == [10.0]
    assert world.journals == 2 and world.turns == 1
    assert end["status"] == "failed" and end["tail_attempts"] == {"journal": 2}


def test_ordinary_tail_node_budget_unchanged():
    world = RetryWorld(fail_turn=False, fail_journal=True)
    delays, end = _retry_delays(world, urgent=False)
    assert delays == [30.0, 60.0]
    assert world.journals == 3 and end["status"] == "failed"


class ReachOutWorld(RetryWorld):
    """The turn succeeds and Orion asks to share; records releases and Door-A keeps."""

    def __init__(self):
        super().__init__(fail_turn=False)
        self.releases: list[str] = []
        self.kept: list[dict] = []

    async def read(self, run_id, **kwargs):
        return {"graph_readable": True, "outcome": {"reach_out": True, "reach_out_why": "worth saying"}}

    async def release(self, state, reason, keep_requeued=False):
        if state.get("lease"):
            self.releases.append(reason)
        return {"lease": None, "hold": None}

    async def guard(self, state):
        return state.get("lease")

    async def keep(self, state):
        self.kept.append(state["lease"])

    def graph(self, saver):
        return build_admitted_graph(
            Deps(self.turn, self.read, self.row, self.journal),
            AdmissionDeps(self.register, self.lease, self.execute, self.release, self.event,
                          now=lambda: self.now, guard=self.guard, keep_for_outreach=self.keep),
            saver,
        )


def _finish(world: ReachOutWorld, *, urgent: bool) -> dict[str, Any]:
    async def scenario():
        graph = world.graph(InMemorySaver())
        return await graph.ainvoke(_admitted_initial(urgent=urgent), CFG)

    return asyncio.run(scenario())


def test_urgent_run_never_keeps_the_outreach_hold_even_when_orion_asks_to_share():
    """Hub never composes Door-A for an urgent run, so nothing would release a kept hold."""
    world = ReachOutWorld()
    end = _finish(world, urgent=True)
    assert end["status"] == "completed"
    assert world.releases == ["completed"] and world.kept == []
    assert "gpu_lease" not in finish_detail(end)


def test_ordinary_reach_out_still_keeps_the_hold_for_door_a():
    world = ReachOutWorld()
    end = _finish(world, urgent=False)
    assert world.releases == [] and len(world.kept) == 1
    assert finish_detail(end)["gpu_lease"]["lease_id"] == "hold-1"


# --- the urgent deadline (Hub sets admission.deadline_at = now + urgent timeout) ------------


class QueuedWorld(RetryWorld):
    """The pool never grants: the run sits at resource_wait."""

    async def lease(self, state):
        from app.admitted_graph import WAITING

        return WAITING, {"status": "waiting_resource", "lease": None}


class _DeadlineStore:
    def __init__(self, request: dict[str, Any]):
        self.row = {"run_id": RUN_ID, "request": request, "terminal": None, "control": None}
        self.projections: list[tuple[str, dict]] = []
        self.events: list[str] = []

    async def touch(self, run_id):
        return None

    async def get_run(self, run_id):
        return dict(self.row)

    async def record_event(self, run_id, name, detail, **kwargs):
        self.events.append(name)

    async def finish_projection(self, run_id, status, detail, **kwargs):
        self.projections.append((status, detail))
        self.row["terminal"] = status
        return status


def _deadline_runtime(world: RetryWorld, *, deadline: datetime):
    """The real AdmissionRuntime._drive / _terminal over an in-memory graph and store."""
    from contextlib import asynccontextmanager

    from app.admission_runtime import DEFAULT_WORKFLOW, AdmissionRuntime

    request = {"run_id": RUN_ID, "correlation_id": "trace-urgent", "workflow": DEFAULT_WORKFLOW,
               "brief": _brief(urgent=True), "requested_at": NOW.isoformat(),
               "admission": {"resource": "llm.route.agent", "priority": "urgent",
                             "deadline_at": deadline.isoformat()}}
    rt = object.__new__(AdmissionRuntime)
    rt.store = _DeadlineStore(request)
    rt.graphs = {DEFAULT_WORKFLOW: world.graph(InMemorySaver())}
    rt.now = lambda: world.now
    rt.outreach, rt._hints, rt._checked = {}, set(), {}
    rt._wake = asyncio.Event()
    rt.released = []

    async def release(state, reason, keep_requeued=False):
        rt.released.append(reason)
        return {"lease": None, "hold": None}

    async def event(state, name, detail):
        rt.store.events.append(name)

    async def hold_ready(run_id, state):
        return False

    @asynccontextmanager
    async def claim(run_id):
        yield True

    rt.release, rt.event, rt._hold_ready, rt.claim = release, event, hold_ready, claim
    return rt


def test_urgent_run_still_queued_at_its_deadline_fails_with_the_urgent_detail():
    """Never granted a GPU: the deadline still ends the run with a terminal `failed` that
    names the incident, which the outbox publishes as the run-state event Hub reports on."""
    world = QueuedWorld()
    rt = _deadline_runtime(world, deadline=NOW + timedelta(seconds=1200))

    async def scenario():
        await rt._drive(rt.store.row)
        snap = await rt.graphs["curiosity.investigate"].aget_state(CFG)
        assert snap.next == ("resource_wait",) and rt.store.projections == []
        world.now = NOW + timedelta(seconds=1201)
        await rt._drive(rt.store.row)

    asyncio.run(scenario())
    [(status, detail)] = rt.store.projections
    assert status == "failed"
    assert detail == {"error": "workflow_deadline", "urgent": URGENT_SUMMARY}
    assert rt.released == ["deadline"]
    from app.admission_runtime import TERMINAL_STATE_NODE

    assert TERMINAL_STATE_NODE["failed"] == "failed"  # the node Hub treats as terminal


def test_urgent_turn_running_at_its_deadline_fails_with_the_urgent_detail():
    """The deadline fires inside the turn (AdmissionRuntime.execute raises WorkflowDeadline)."""
    from app.admission_runtime import AdmissionRuntime
    from app.admitted_graph import WorkflowDeadline

    world = RetryWorld(fail_turn=False)

    async def execute(state, node):
        raise WorkflowDeadline("workflow_deadline")

    world.execute = execute  # type: ignore[method-assign]
    _delays, end = _retry_delays(world, urgent=True)
    assert end["status"] == "failed" and world.turns == 0
    detail = AdmissionRuntime._terminal_detail_for("curiosity.investigate", "failed", end)
    assert detail["error"] == "workflow_deadline" and detail["urgent"] == URGENT_SUMMARY


def test_failed_urgent_run_detail_carries_the_error():
    """What Hub's must-deliver report reads off a failed admitted run (Task 6)."""
    from app.admission_runtime import AdmissionRuntime

    world = RetryWorld()
    _delays, end = _retry_delays(world, urgent=True)
    detail = AdmissionRuntime._terminal_detail_for("curiosity.investigate", "failed", end)
    assert detail["error"].startswith("HarnessTurnFailed")


# --- terminal detail: failed and cancelled urgent runs say they were urgent --------


def _terminal_detail(status: str, *, urgent: bool, **extra: Any) -> dict[str, Any]:
    from app.admission_runtime import AdmissionRuntime

    state = {**_state(urgent=urgent), **extra}
    return AdmissionRuntime._terminal_detail_for("curiosity.investigate", status, state)


URGENT_SUMMARY = {
    "incident_id": "0123456789abcdef", "trigger": "manual", "subject": "athena",
    "question": "Is athena actually overheating?", "requested_at": "2026-09-28T12:00:00Z",
}


def test_urgent_failed_detail_names_the_incident_and_the_error():
    detail = _terminal_detail("failed", urgent=True, last_error="workflow_deadline")
    assert detail == {"error": "workflow_deadline", "urgent": URGENT_SUMMARY}


def test_urgent_cancelled_detail_names_the_incident_and_says_cancelled():
    detail = _terminal_detail("cancelled", urgent=True)
    assert detail == {"error": "cancelled", "urgent": URGENT_SUMMARY}


def test_ordinary_failed_and_cancelled_details_unchanged():
    assert _terminal_detail("failed", urgent=False, last_error="workflow_deadline") == {"error": "workflow_deadline"}
    assert _terminal_detail("cancelled", urgent=False) == {}


def _projection_runtime(seen: list):
    from app.admission_runtime import AdmissionRuntime

    runtime = object.__new__(AdmissionRuntime)

    async def finish_projection(run_id, status, detail, **kwargs):
        seen.append((status, detail, kwargs))
        return status

    runtime.store = SimpleNamespace(finish_projection=finish_projection)
    runtime._wake = asyncio.Event()
    runtime.outreach, runtime._hints, runtime._checked = {}, set(), {}
    return runtime


def test_urgent_terminal_hands_the_store_an_urgent_cancel_detail():
    """A cancel that wins the race against completion publishes `cancelled_detail`, so an urgent
    run's must carry the incident too."""
    seen: list = []
    state = {**_state(urgent=True), "text": "done", "status": "completed"}
    asyncio.run(_projection_runtime(seen)._terminal(RUN_ID, "completed", state))
    [(status, _detail, kwargs)] = seen
    assert status == "completed"
    assert kwargs == {"cancelled_detail": {"error": "cancelled", "urgent": URGENT_SUMMARY}}


def test_ordinary_terminal_projection_call_unchanged():
    seen: list = []
    state = {**_state(urgent=False), "text": "done", "status": "completed"}
    asyncio.run(_projection_runtime(seen)._terminal(RUN_ID, "completed", state))
    [(status, _detail, kwargs)] = seen
    assert status == "completed" and kwargs == {}


# --- a doomed retry is not run (run a153451fe423) ------------------------------


def _retry_with_deadline(seconds_left: float) -> tuple[RetryWorld, list[float], dict[str, Any]]:
    world = RetryWorld()
    deadline = (world.now + timedelta(seconds=seconds_left)).isoformat()

    async def scenario():
        graph = world.graph(InMemorySaver())
        initial = {**_admitted_initial(urgent=True),
                   "admission": {"resource": "llm.route.agent", "priority": "urgent", "deadline_at": deadline}}
        await graph.ainvoke(initial, CFG)
        delays: list[float] = []
        for _ in range(10):
            snap = await graph.aget_state(CFG)
            if not snap.next:
                return delays, dict(snap.values)
            retry_at = datetime.fromisoformat(snap.values["retry_at"])
            delays.append((retry_at - world.now).total_seconds())
            world.now = retry_at
            await graph.ainvoke(Command(resume=True), CFG)
        raise AssertionError("graph never terminated")

    delays, end = asyncio.run(scenario())
    return world, delays, end


def test_urgent_retry_skipped_when_less_than_one_attempt_is_left():
    # The incident: attempt 1 used its 900 s turn, ~300 s of the 1200 s deadline remained, and
    # attempt 2 restarted from zero only to be killed by workflow_deadline mid-motor.
    world, delays, end = _retry_with_deadline(300.0)
    assert delays == [] and world.turns == 1
    assert end["status"] == "failed" and end["attempt"] == 1
    assert end["last_error"].startswith("retry_skipped_insufficient_time: HarnessTurnFailed")


def test_urgent_retry_still_runs_with_a_whole_attempt_left():
    # 10 s backoff + one 900 s attempt fits.
    world, delays, end = _retry_with_deadline(910.0)
    assert delays == [10.0] and world.turns == 2


def test_urgent_retry_skipped_when_the_backoff_eats_the_margin():
    # 905 s left now, but the retry would start after the 10 s backoff with only 895 s.
    world, delays, end = _retry_with_deadline(905.0)
    assert delays == [] and world.turns == 1
    assert end["last_error"].startswith("retry_skipped_insufficient_time")


def test_ordinary_run_without_a_deadline_retries_as_before():
    world = RetryWorld()
    delays, end = _retry_delays(world, urgent=False)
    assert delays == [30.0, 60.0] and world.turns == 3


def test_finish_detail_marks_a_salvaged_urgent_draft():
    state = {**_state(urgent=True), "text": "Orion's partial verdict", "incident_report": None,
             "debug": {"draft_salvaged": True, "salvaged_from_error": "finalize_reply_deadline: cut"}}
    detail = finish_detail(state)
    assert detail["draft_salvaged"] is True
    assert detail["salvaged_from_error"] == "finalize_reply_deadline: cut"
    assert detail["finding_text"] == "Orion's partial verdict"
    plain = finish_detail({**_state(urgent=True), "text": "done", "debug": {}})
    assert "draft_salvaged" not in plain


def test_turn_limit_sent_to_hub_is_clamped_to_the_run_deadline():
    # durable-runs' attempt timer is min(brief.timeout_sec, deadline - now); Hub must be told the
    # same, or its finalize reserve lines up with a release that already happened.
    from app.graph import attempt_timeout_sec

    state = {**_state(urgent=True), "admission": {"deadline_at": (NOW + timedelta(seconds=600)).isoformat()}}
    assert attempt_timeout_sec(state, 900.0, now=NOW) == 600.0
    assert attempt_timeout_sec({**state, "admission": {}}, 900.0, now=NOW) == 900.0
    late = {**state, "admission": {"deadline_at": (NOW - timedelta(seconds=5)).isoformat()}}
    assert attempt_timeout_sec(late, 900.0, now=NOW) == 1.0

    sent: list[CuriosityTurnRequestV1] = []
    near = {**state, "admission": {"deadline_at": (datetime.now(timezone.utc) + timedelta(seconds=300)).isoformat()}}
    asyncio.run(make_nodes(_turn_deps(sent))["harness_turn"](near))
    assert 290.0 <= sent[0].timeout_sec <= 300.0


def test_salvaged_draft_is_not_journaled_as_a_finished_investigation():
    journaled: list[Any] = []
    deps = _turn_deps([])

    async def journal(entry):
        journaled.append(entry)
        return entry.entry_id

    deps.publish_journal = journal
    state = {**_state(urgent=True), "text": "half a verdict", "debug": {"draft_salvaged": True}}
    assert asyncio.run(make_nodes(deps)["journal"](state)) == {"journal_entry_id": None}
    assert journaled == []
