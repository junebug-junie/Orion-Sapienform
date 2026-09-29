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
