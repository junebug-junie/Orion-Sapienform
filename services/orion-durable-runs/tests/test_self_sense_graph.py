"""self_sense_eval's own durable-run graph: asks the four fixed questions
(one per Deps.run_turn call, reusing curiosity's own turn-execution RPC
shape), scores and publishes a row per question via Deps.publish_rows, then
finishes. A per-question turn failure never aborts the run -- same contract
Hub's own in-process `_run_self_sense_eval` already has.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1  # noqa: E402
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS  # noqa: E402

from app.self_sense_graph import Deps, build_self_sense_graph, finish_detail  # noqa: E402

_QUESTION_BY_TEXT = {text: key for key, text in SELF_SENSE_QUESTIONS}


class _World:
    """Fake deps that record every run_turn call and every published row."""

    def __init__(self, *, fail_keys: set[str] | None = None):
        self.turn_calls: list[CuriosityTurnRequestV1] = []
        self.published_rows: list = []
        self.fail_keys = fail_keys or set()

    def deps(self) -> Deps:
        return Deps(run_turn=self._run_turn, publish_rows=self._publish_rows)

    async def _run_turn(self, request: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turn_calls.append(request)
        # Identify the question by its PROMPT TEXT, not correlation_id --
        # correlation_id is now a real uuid4 per question (review finding,
        # 2026-09-21: a derived "<run_corr>:<question_key>" string both
        # collided in Hub's per-(run_id, correlation_id) turn cache and
        # broke build_row's is_uuid() traceability check), so it can no
        # longer be reverse-engineered to a question_key here either.
        question_key = _QUESTION_BY_TEXT[request.prompt]
        if question_key in self.fail_keys:
            return CuriosityTurnResultV1(
                run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error="empty_generation"
            )
        return CuriosityTurnResultV1(
            run_id=request.run_id, correlation_id=request.correlation_id, text=f"answer:{question_key}", ok=True
        )

    async def _publish_rows(self, rows: list) -> tuple[int, int]:
        self.published_rows.extend(rows)
        return len(rows), 0


def _brief() -> dict:
    return {
        "prompt": "self-sense eval",
        "session_id": "self-sense-eval",
        "timeout_sec": 600.0,
        "source_tag": "curiosity_self_sense_eval",
        "line": "self_sense_eval",
        "questions": [list(q) for q in SELF_SENSE_QUESTIONS],
        "self_definition_version": 5,
        "lived_answers": [],
    }


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def test_a_full_run_asks_every_question_once_and_publishes_one_row_each():
    world = _World()
    graph = build_self_sense_graph(world.deps(), InMemorySaver())
    initial = {"run_id": "sse-run-1", "correlation_id": "corr-1", "workflow": "self_sense_eval", "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("sse-run-1")))

    assert len(world.turn_calls) == len(SELF_SENSE_QUESTIONS)
    asked_prompts = {c.prompt for c in world.turn_calls}
    assert asked_prompts == {q for _, q in SELF_SENSE_QUESTIONS}
    # Every turn ran under the shared clean self-sense session, not curiosity's.
    assert all(c.session_id == "self-sense-eval" for c in world.turn_calls)
    # Every question gets its OWN real correlation_id -- distinct (the actual
    # bug: Hub's turn cache is keyed on (run_id, correlation_id), and a
    # shared/derived correlation_id across questions under one run_id made
    # questions 2-4 silently receive question 1's cached answer) and a real
    # uuid4, not a derived string (so build_row/is_uuid don't treat it as
    # synthetic). Review findings, 2026-09-21.
    turn_corr_ids = [c.correlation_id for c in world.turn_calls]
    assert len(set(turn_corr_ids)) == len(SELF_SENSE_QUESTIONS)
    for corr_id in turn_corr_ids:
        UUID(corr_id)  # raises ValueError if not a real uuid

    assert len(world.published_rows) == len(SELF_SENSE_QUESTIONS)
    # Published rows carry the SAME correlation_id the real turn used, not a
    # fresh/synthetic one -- turn-to-row traceability.
    assert {r.correlation_id for r in world.published_rows} == set(turn_corr_ids)
    assert final["status"] == "completed"
    assert final["published"] == len(SELF_SENSE_QUESTIONS) and final["failed"] == 0 and final["empty"] == 0
    detail = finish_detail(final)
    assert detail["line"] == "self_sense_eval" and detail["published"] == len(SELF_SENSE_QUESTIONS)


def test_a_failed_question_still_publishes_a_none_source_row_for_the_others():
    fail_key = SELF_SENSE_QUESTIONS[0][0]
    world = _World(fail_keys={fail_key})
    graph = build_self_sense_graph(world.deps(), InMemorySaver())
    initial = {"run_id": "sse-run-2", "correlation_id": "corr-2", "workflow": "self_sense_eval", "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("sse-run-2")))

    # All four questions still get asked and published -- one bad question
    # doesn't cost the other three, same contract as the in-process path.
    assert len(world.turn_calls) == len(SELF_SENSE_QUESTIONS)
    assert len(world.published_rows) == len(SELF_SENSE_QUESTIONS)
    assert final["empty"] == 1
    failed_row = next(r for r in world.published_rows if r.question_key == fail_key)
    assert failed_row.answer_source == "none"


def test_a_crash_between_ask_and_publish_resumes_without_re_asking_answered_questions():
    world = _World()
    saver = InMemorySaver()
    initial = {"run_id": "sse-run-3", "correlation_id": "corr-3", "workflow": "self_sense_eval", "brief": _brief(), "attempt": 0}
    config = _cfg("sse-run-3")

    # Drive only the first node directly (simulating a crash right after
    # ask_questions checkpoints, before publish runs).
    graph = build_self_sense_graph(world.deps(), saver)
    asyncio.run(graph.ainvoke(initial, config, interrupt_after=["ask_questions"]))
    assert len(world.turn_calls) == len(SELF_SENSE_QUESTIONS)

    # Resume: publish runs, ask_questions does NOT re-ask (every key already
    # in `answers`).
    final = asyncio.run(graph.ainvoke(None, config))
    assert len(world.turn_calls) == len(SELF_SENSE_QUESTIONS)  # unchanged
    assert final["status"] == "completed"
    assert len(world.published_rows) == len(SELF_SENSE_QUESTIONS)


# Live 2026-10-02..06: four self-sense runs on the gpu2 worker "completed" with 16/16 blank answers
# while every LLM call behind them 500'd. The per-turn error names the real cause.
UPSTREAM_500 = "upstream_http_5xx:http_500: No user query found in messages."


class _AllFailWorld(_World):
    async def _run_turn(self, request: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        self.turn_calls.append(request)
        return CuriosityTurnResultV1(
            run_id=request.run_id, correlation_id=request.correlation_id, ok=False, error=UPSTREAM_500
        )


def test_every_answer_empty_fails_the_run_and_publishes_nothing():
    world = _AllFailWorld()
    graph = build_self_sense_graph(world.deps(), InMemorySaver())
    initial = {"run_id": "sse-run-x", "correlation_id": "corr-x", "workflow": "self_sense_eval",
               "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("sse-run-x")))

    assert len(world.turn_calls) == len(SELF_SENSE_QUESTIONS)
    assert world.published_rows == []
    assert final["status"] == "failed"
    assert final["last_error"] == f"self_sense_no_answers: {UPSTREAM_500}"
    detail = finish_detail(final)
    assert detail["error"] == final["last_error"]
    assert detail["published"] == 0 and detail["empty"] == len(SELF_SENSE_QUESTIONS)
    # Each turn's own error is kept, not just a log line.
    assert all(t["error"] == UPSTREAM_500 for t in detail["turns"].values())


def test_one_empty_answer_keeps_its_error_in_the_turn_detail():
    fail_key = SELF_SENSE_QUESTIONS[0][0]
    world = _World(fail_keys={fail_key})
    graph = build_self_sense_graph(world.deps(), InMemorySaver())
    initial = {"run_id": "sse-run-y", "correlation_id": "corr-y", "workflow": "self_sense_eval",
               "brief": _brief(), "attempt": 0}
    final = asyncio.run(graph.ainvoke(initial, _cfg("sse-run-y")))

    assert final["status"] == "completed"
    turns = finish_detail(final)["turns"]
    assert turns[fail_key]["error"] == "empty_generation"
    assert all("error" not in t for k, t in turns.items() if k != fail_key)


def test_runner_reports_a_no_answer_run_as_failed_not_completed(monkeypatch):
    """The plain (non-admitted) runner emits the terminal state from the graph's own status, and a
    run that reached END failed has nothing to resume (next_node None) -- asserted on the real
    published DurableRunStateV1, not a stubbed _emit_state."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    import app.settings as settings_mod
    from app.runner import SELF_SENSE_EVAL_WORKFLOW, DurableRunner, WorkflowSpec
    from orion.schemas.durable_run import SELF_SENSE_EVAL_NODES, DurableRunStateV1

    settings_mod._settings = None
    saver = InMemorySaver()
    runner = DurableRunner(settings_mod.get_settings(), bus=object(), checkpointer=saver)
    world = _AllFailWorld()
    spec = WorkflowSpec(workflow=SELF_SENSE_EVAL_WORKFLOW, graph=build_self_sense_graph(world.deps(), saver),
                        nodes=list(SELF_SENSE_EVAL_NODES), finish_detail=finish_detail)
    events: list[DurableRunStateV1] = []

    async def publish(channel, kind, payload, corr=None):
        if isinstance(payload, DurableRunStateV1):
            events.append(payload)
        return True

    monkeypatch.setattr(runner, "_publish", publish)
    initial = {"run_id": "sse-run-r", "correlation_id": "corr-r", "workflow": "self_sense_eval",
               "brief": _brief(), "attempt": 0}
    try:
        asyncio.run(runner._drive(spec, "sse-run-r", initial))
    finally:
        settings_mod._settings = None
    last = events[-1]
    assert last.node == SELF_SENSE_EVAL_NODES[-1]
    assert last.status == "failed"
    assert last.next_node is None
    assert last.detail["error"].startswith("self_sense_no_answers:")
    # Intermediate nodes are still ordinary running transitions.
    assert all(e.status == "running" for e in events[:-1])
