"""The curiosity run as a LangGraph state graph.

Nodes, in order (`orion.schemas.durable_run.CURIOSITY_NODES`):

    harness_turn -> read_turn_result -> publish_attention_row -> journal -> finish

Each node reads and writes named keys of `CuriosityRunState` and nothing
else; every node's result is checkpointed by the compiled graph's saver
before the next node runs, so a restart between any two nodes resumes at
the next one with the earlier results intact. `harness_turn` is the only
node whose *work* is not resumable (Hub runs an FCC subprocess; a restart
mid-turn re-issues the turn under the same run_id -- design doc MQ2).

The nodes talk to the world through `Deps`, a plain object of callables,
so the graph is testable with fakes and an in-memory saver; the real
implementations live in `runner.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from orion.curiosity.journal import MaterialCounts, build_investigation_journal_entry
from orion.schemas.durable_run import (
    CURIOSITY_NODES,
    CuriosityRunBriefV1,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
)

logger = logging.getLogger("orion-durable-runs.graph")

# Bounded copy of the finding carried on the `finish` state event for Hub's
# outreach composer. The journal holds the full text.
FINDING_TEXT_CAP = 8000


class CuriosityRunState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    brief: dict[str, Any]  # CuriosityRunBriefV1.model_dump()
    attempt: int
    # harness_turn
    text: str
    debug: dict[str, Any]
    # read_turn_result
    outcome: dict[str, Any] | None
    footprint: dict[str, int] | None
    hops: list[list[Any]]
    evidence_summary: str | None
    graph_readable: bool
    # publish_attention_row / journal
    attention_row_published: bool
    journal_entry_id: str | None
    # finish
    status: str


class HarnessTurnFailed(RuntimeError):
    """Raised by `harness_turn` when Hub did not return a usable turn. The
    graph stops here with `harness_turn` still the next node, so the resume
    sweep re-issues it; nothing after it runs on an empty turn."""


@dataclass
class Deps:
    run_turn: Callable[[CuriosityTurnRequestV1], Awaitable[CuriosityTurnResultV1]]
    read_turn_result: Callable[[str], Awaitable[dict[str, Any]]]
    publish_attention_row: Callable[[dict[str, Any]], Awaitable[bool]]
    publish_journal: Callable[[Any], Awaitable[str | None]]


def _brief(state: CuriosityRunState) -> CuriosityRunBriefV1:
    return CuriosityRunBriefV1.model_validate(state["brief"])


def make_nodes(deps: Deps) -> dict[str, Callable[[CuriosityRunState], Awaitable[dict[str, Any]]]]:
    async def harness_turn(state: CuriosityRunState) -> dict[str, Any]:
        brief = _brief(state)
        attempt = int(state.get("attempt") or 0) + 1
        request = CuriosityTurnRequestV1(
            run_id=state["run_id"],
            correlation_id=state["correlation_id"],
            prompt=brief.prompt,
            fcc_model_label=brief.fcc_model_label,
            timeout_sec=brief.timeout_sec,
            source_tag=brief.source_tag,
            attempt=attempt,
        )
        result = await deps.run_turn(request)
        if not result.ok or not result.text.strip():
            raise HarnessTurnFailed(result.error or "empty_generation")
        return {"text": result.text, "debug": dict(result.debug), "attempt": attempt}

    async def read_turn_result(state: CuriosityRunState) -> dict[str, Any]:
        found = await deps.read_turn_result(state["run_id"])
        return {
            "outcome": found.get("outcome"),
            "footprint": found.get("footprint"),
            "hops": list(found.get("hops") or []),
            "evidence_summary": found.get("evidence_summary"),
            "graph_readable": bool(found.get("graph_readable", False)),
        }

    async def publish_attention_row(state: CuriosityRunState) -> dict[str, Any]:
        ok = await deps.publish_attention_row(
            {
                "run_id": state["run_id"],
                "correlation_id": state["correlation_id"],
                "outcome": state.get("outcome"),
                "graph_readable": state.get("graph_readable", False),
            }
        )
        return {"attention_row_published": bool(ok)}

    async def journal(state: CuriosityRunState) -> dict[str, Any]:
        brief = _brief(state)
        debug = state.get("debug") or {}
        hops = [(int(n), str(note)) for n, note in (state.get("hops") or [])]
        entry = build_investigation_journal_entry(
            material=MaterialCounts(
                approved_total=brief.material.approved_total,
                approved_by_kind=brief.material.approved_by_kind,
                crystallization_count=brief.material.crystallization_count,
                relation_total=brief.material.relation_total,
                relation_count=brief.material.relation_count,
            ),
            body_text=state["text"],
            correlation_id=state["correlation_id"],
            run_id=state["run_id"],
            harness_step_count=debug.get("harness_step_count"),
            harness_grounding_status=debug.get("harness_grounding_status"),
            harness_elapsed_sec=debug.get("elapsed_sec"),
            harness_fcc_elapsed_sec=debug.get("fcc_elapsed_sec"),
            graph_footprint=state.get("footprint"),
            hop_notes=hops or None,
        )
        entry_id = await deps.publish_journal(entry)
        return {"journal_entry_id": entry_id}

    async def finish(state: CuriosityRunState) -> dict[str, Any]:
        return {"status": "completed"}

    return {
        "harness_turn": harness_turn,
        "read_turn_result": read_turn_result,
        "publish_attention_row": publish_attention_row,
        "journal": journal,
        "finish": finish,
    }


def finish_detail(state: CuriosityRunState) -> dict[str, Any]:
    """What Hub needs from a completed run to decide outreach, bounded."""
    outcome = state.get("outcome") or {}
    text = state.get("text") or ""
    return {
        "reach_out": bool(outcome.get("reach_out")),
        "reach_out_why": str(outcome.get("reach_out_why") or "")[:1000],
        "continue_line": bool(outcome.get("continue_line")),
        "finding_text": text[:FINDING_TEXT_CAP],
        "journal_entry_id": state.get("journal_entry_id"),
        "attempts": int(state.get("attempt") or 0),
    }


def build_curiosity_graph(deps: Deps, checkpointer: Any):
    """Compile the graph with the given saver. Imported lazily so the
    schema/contract half of this package stays importable without langgraph."""
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(deps)
    g: StateGraph = StateGraph(CuriosityRunState)
    for name in CURIOSITY_NODES:
        g.add_node(name, nodes[name])
    g.add_edge(START, CURIOSITY_NODES[0])
    for a, b in zip(CURIOSITY_NODES, CURIOSITY_NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(CURIOSITY_NODES[-1], END)
    return g.compile(checkpointer=checkpointer)
