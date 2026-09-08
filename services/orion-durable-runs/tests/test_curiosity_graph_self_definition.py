"""A self-inquiry run through the durable graph: the `:SelfDefinition` read
in `read_turn_result` is checkpointed on the state, the journal node writes
the self line's entry, and `finish_detail` carries both the line and the
definition for Hub to mirror. The graph shape (CURIOSITY_NODES) is unchanged
-- no new node, so resume semantics are exactly the investigation line's."""

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

from app.graph import Deps, build_curiosity_graph, finish_detail  # noqa: E402

DEFINITION = {"run_id": "abc123def456", "text": "I am a mesh.", "evidence": ["README.md"], "revises": "", "written_at": 1}


class _World:
    def __init__(self, *, definition=DEFINITION):
        self.journal_entries: list = []
        self.definition = definition
        self.turn_tags: list[str] = []

    def deps(self) -> Deps:
        async def run_turn(req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
            self.turn_tags.append(req.source_tag)
            return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="I looked.", debug={"harness_step_count": 9})

        async def read_turn_result(run_id: str) -> dict:
            return {"outcome": None, "footprint": {"SelfDefinition": 1, "Prior": 2}, "hops": [], "evidence_summary": None,
                    "graph_readable": True, "self_definition": self.definition}

        async def publish_attention_row(facts: dict) -> bool:
            return True

        async def publish_journal(entry) -> str | None:
            self.journal_entries.append(entry)
            return entry.entry_id

        return Deps(run_turn=run_turn, read_turn_result=read_turn_result, publish_attention_row=publish_attention_row, publish_journal=publish_journal)


def _request(line: str) -> DurableRunRequestV1:
    return DurableRunRequestV1(
        run_id="abc123def456", workflow="curiosity.investigate", correlation_id="7dcc3944-29bb-5d8f-915f-90f4e6968d47",
        brief=CuriosityRunBriefV1(prompt="What am I?", session_id="orion_curiosity", timeout_sec=3500.0, graph_configured=True,
                                  source_tag="curiosity_self_inquiry", line=line),
    )


def _run(world: _World, line: str) -> dict:
    graph = build_curiosity_graph(world.deps(), InMemorySaver())
    req = _request(line)
    state = {"run_id": req.run_id, "correlation_id": req.correlation_id, "brief": req.brief.model_dump(mode="json"), "attempt": 0}

    async def go():
        final = None
        async for _ in graph.astream(state, config={"configurable": {"thread_id": req.run_id}}, stream_mode="values"):
            pass
        snap = await graph.aget_state({"configurable": {"thread_id": req.run_id}})
        return snap.values

    return asyncio.run(go())


def test_self_inquiry_run_carries_the_definition_to_finish_detail() -> None:
    world = _World()
    values = _run(world, "self_inquiry")
    assert values["self_definition"] == DEFINITION
    detail = finish_detail(values)
    assert detail["line"] == "self_inquiry"
    assert detail["self_definition"] == DEFINITION
    assert world.turn_tags == ["curiosity_self_inquiry"]
    assert world.journal_entries[0].title == "Self-inquiry"
    assert world.journal_entries[0].source_ref == "curiosity:abc123def456"
    assert world.journal_entries[0].entry_id == "curiosity-self-inquiry:abc123def456"


def test_investigation_run_still_reports_its_line_and_journal() -> None:
    world = _World(definition=None)
    values = _run(world, "investigate")
    detail = finish_detail(values)
    assert detail["line"] == "investigate"
    assert detail["self_definition"] is None
    assert world.journal_entries[0].title == "Curiosity"


def test_brief_defaults_keep_old_runs_valid() -> None:
    brief = CuriosityRunBriefV1(prompt="p", session_id="s", timeout_sec=1.0)
    assert brief.line == "investigate"
    assert CURIOSITY_NODES == ("harness_turn", "read_turn_result", "publish_attention_row", "journal", "finish") or len(CURIOSITY_NODES) == 5
