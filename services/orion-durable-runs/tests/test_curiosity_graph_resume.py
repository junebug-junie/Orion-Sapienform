"""The durable curiosity graph: every node's result is checkpointed, a crash
between nodes resumes at the next node without re-running earlier ones, a
failed harness turn leaves the thread resumable at harness_turn, and the
runner's sweep finds exactly the unfinished threads.

Runs against LangGraph's in-memory saver; the Postgres saver has the same
interface (`aget_state`, `alist`, `astream`) and is exercised by the live
restart test in the PR report, not here.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from orion.schemas.durable_run import (  # noqa: E402
    CURIOSITY_NODES,
    CuriosityRunBriefV1,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
    DurableRunRequestV1,
)

from app.graph import Deps, HarnessTurnFailed, build_curiosity_graph, finish_detail  # noqa: E402


class _World:
    """Fake deps that record every call, so a test can assert which nodes ran."""

    def __init__(self, *, turn_ok: bool = True, fail_journal_once: bool = False):
        self.calls: list[str] = []
        self.turn_ok = turn_ok
        self.fail_journal_once = fail_journal_once
        self.journal_entries: list = []

    def deps(self) -> Deps:
        async def run_turn(req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
            self.calls.append(f"turn:{req.attempt}")
            if not self.turn_ok:
                return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, ok=False, error="rpc:TimeoutError")
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="found it", debug={"harness_step_count": 14, "elapsed_sec": 900.0})

        async def read_turn_result(run_id: str) -> dict:
            self.calls.append("read")
            return {"outcome": {"run_id": run_id, "continue_line": True, "continue_note": "keep pulling", "reach_out": True, "reach_out_why": "worth it"},
                    "footprint": {"Finding": 3, "Prior": 1}, "hops": [[1, "first stop"]], "evidence_summary": "3/3 joined", "graph_readable": True}

        async def publish_attention_row(facts: dict) -> bool:
            self.calls.append("row")
            return True

        async def publish_journal(entry) -> str | None:
            self.calls.append("journal")
            if self.fail_journal_once:
                self.fail_journal_once = False
                raise RuntimeError("bus down")
            self.journal_entries.append(entry)
            return entry.entry_id

        return Deps(run_turn=run_turn, read_turn_result=read_turn_result, publish_attention_row=publish_attention_row, publish_journal=publish_journal)


def _request(run_id: str = "abc123def456") -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id=run_id, workflow="curiosity.investigate", correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47",
        brief=CuriosityRunBriefV1(prompt="Pick something.", session_id="orion_curiosity", timeout_sec=3500.0, graph_configured=True,
                                  material={"approved_total": 40, "approved_by_kind": {"semantic": 40}, "crystallization_count": 12, "relation_total": 300, "relation_count": 6}),
    )


def _initial(req: DurableRunRequestV1) -> dict:
    return {"run_id": req.run_id, "correlation_id": req.correlation_id, "brief": req.brief.model_dump(mode="json"), "attempt": 0}


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def test_a_full_run_walks_every_node_in_order_and_journals_once() -> None:
    world = _World()
    graph = build_curiosity_graph(world.deps(), InMemorySaver())
    req = _request()
    final = asyncio.run(graph.ainvoke(_initial(req), _cfg(req.run_id)))
    assert world.calls == ["turn:1", "read", "row", "journal"]
    assert final["status"] == "completed" and final["text"] == "found it"
    assert len(world.journal_entries) == 1
    entry = world.journal_entries[0]
    assert entry.source_ref == f"curiosity:{req.run_id}" and entry.source_kind == "self_study"
    assert "Offered 12 of 40 approved concepts" in entry.body and "Wrote to its own graph: Finding 3, Prior 1" in entry.body
    detail = finish_detail(final)
    assert detail["reach_out"] is True and detail["finding_text"] == "found it" and detail["attempts"] == 1
    snap = asyncio.run(graph.aget_state(_cfg(req.run_id)))
    assert snap.next == ()  # terminal


def test_a_crash_after_the_turn_resumes_without_re_running_the_turn() -> None:
    """The expensive node's result survives: journal fails once (the process
    'dies' there), the thread is re-invoked with None, and the turn is NOT
    re-issued -- only the failed node and what follows run."""
    world = _World(fail_journal_once=True)
    graph = build_curiosity_graph(world.deps(), InMemorySaver())
    req = _request()
    with pytest.raises(RuntimeError):
        asyncio.run(graph.ainvoke(_initial(req), _cfg(req.run_id)))
    assert world.calls == ["turn:1", "read", "row", "journal"]
    snap = asyncio.run(graph.aget_state(_cfg(req.run_id)))
    assert snap.next == ("journal",)
    assert snap.values["text"] == "found it"  # checkpointed before the crash

    final = asyncio.run(graph.ainvoke(None, _cfg(req.run_id)))  # resume
    assert world.calls == ["turn:1", "read", "row", "journal", "journal"]
    assert final["status"] == "completed" and len(world.journal_entries) == 1


def test_a_failed_turn_leaves_the_thread_resumable_at_harness_turn(monkeypatch) -> None:
    """Through the runner, because the attempt counter is the runner's doing:
    a node that raised cannot record its own attempt, so the failure handler
    stamps it on the thread and the re-issued turn says attempt 2."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    import app.settings as settings_mod
    settings_mod._settings = None
    from app.runner import DurableRunner

    saver = InMemorySaver()
    world = _World(turn_ok=False)
    runner = DurableRunner(settings_mod.get_settings(), bus=None, checkpointer=saver)
    runner._graph = build_curiosity_graph(world.deps(), saver)
    req = _request()

    async def first_attempt() -> None:
        await runner.start_run(req)
        await asyncio.gather(*runner._active.values(), return_exceptions=True)

    asyncio.run(first_attempt())
    snap = asyncio.run(runner._graph.aget_state(_cfg(req.run_id)))
    assert snap.next == ("harness_turn",)
    assert "text" not in snap.values and snap.values["attempt"] == 1

    async def hub_comes_back() -> dict:
        world.turn_ok = True
        counts = await runner.resume_unfinished()
        await asyncio.gather(*runner._active.values(), return_exceptions=True)
        return counts

    counts = asyncio.run(hub_comes_back())
    assert counts["resumed"] == 1
    assert world.calls[:2] == ["turn:1", "turn:2"]
    final = asyncio.run(runner._graph.aget_state(_cfg(req.run_id)))
    assert final.next == () and final.values["attempt"] == 2 and final.values["status"] == "completed"


def test_node_order_matches_the_contract() -> None:
    assert CURIOSITY_NODES == ("harness_turn", "read_turn_result", "publish_attention_row", "journal", "finish")


def test_runner_sweep_finds_unfinished_threads_and_resumes_them(monkeypatch) -> None:
    """The runner-level rule: a thread with a next node is unfinished; the
    sweep re-invokes it; a finished one is left alone."""
    monkeypatch.setenv("POSTGRES_URI", "postgresql://unused/unused")
    monkeypatch.setenv("ORION_BUS_ENABLED", "false")
    import app.settings as settings_mod
    settings_mod._settings = None
    from app.runner import DurableRunner

    saver = InMemorySaver()
    world = _World(fail_journal_once=True)
    runner = DurableRunner(settings_mod.get_settings(), bus=None, checkpointer=saver)
    runner._graph = build_curiosity_graph(world.deps(), saver)  # fakes behind the same saver

    async def scenario() -> tuple[list, dict, list]:
        req = _request()
        await runner.start_run(req)
        await asyncio.gather(*runner._active.values(), return_exceptions=True)
        unfinished_before = await runner.unfinished_threads()
        counts = await runner.resume_unfinished()
        await asyncio.gather(*runner._active.values(), return_exceptions=True)
        unfinished_after = await runner.unfinished_threads()
        return unfinished_before, counts, unfinished_after

    before, counts, after = asyncio.run(scenario())
    assert [(t, n) for t, n, _ in before] == [("abc123def456", "journal")]
    assert counts["resumed"] == 1 and counts["abandoned"] == 0
    assert after == []
    assert world.calls == ["turn:1", "read", "row", "journal", "journal"]
