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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypedDict
from uuid import NAMESPACE_URL, uuid5

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
    # Read by the runner's workflow registry to route a checkpointed thread
    # back to its own graph on resume, before any graph-specific state is
    # known (runner.py's `_peek_workflow`). Absent on any checkpoint written
    # before 2026-09-21 -- the runner treats a missing key as
    # "curiosity.investigate", this graph's own workflow name, so old
    # in-flight threads resume exactly as before.
    workflow: str
    brief: dict[str, Any]  # CuriosityRunBriefV1.model_dump()
    attempt: int
    # harness_turn
    text: str
    debug: dict[str, Any]
    # What the runner itself knows about the last harness turn it issued
    # (2026-09-22): the correlation the turn actually ran under (derived
    # per lease when admitted, else the run's own) plus the runner-side
    # wall clock around the Hub RPC. Written by `harness_turn` on success
    # (`timed_turn`) and by the admitted wrapper on failure (correlation
    # only, captured before the lease is cleared). Absent on any checkpoint
    # written before this key existed -- every reader treats missing as
    # "not known", never as zero. This is the structured key that makes
    # the `harness_turn_trace` Postgres row (PK correlation_id) joinable to
    # a run without text-searching `final_text`.
    harness_turn_meta: dict[str, Any]
    # read_turn_result
    outcome: dict[str, Any] | None
    footprint: dict[str, int] | None
    hops: list[list[Any]]
    evidence_summary: str | None
    graph_readable: bool
    # self_inquiry line only: the run's `:SelfDefinition` or `:LivedAnswer`,
    # bounded, carried on the finish event for Hub to mirror into
    # self_concept_history. Lived draws write LivedAnswer (not SelfDefinition).
    self_definition: dict[str, Any] | None
    lived_answer: dict[str, Any] | None
    # publish_attention_row / journal
    attention_row_published: bool
    journal_entry_id: str | None
    # finish
    status: str
    admission: dict[str, Any]
    lease: dict[str, Any] | None
    retry_at: str | None
    last_error: str | None
    requested_at: str
    retry_node: str | None
    tail_attempts: dict[str, int]


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


def turn_correlation_id(state: CuriosityRunState) -> str:
    """Fence the subprocess identity while keeping run-level lineage stable.

    The governor's cancellation registry is keyed by correlation, including
    cancellations arriving before process registration. A lost generation's
    delayed cancellation must never address its successor's subprocess.
    """
    lease = state.get("lease")
    if not lease:
        return state["correlation_id"]
    identity = f"orion:durable:turn:{state['run_id']}:{state['correlation_id']}:{lease['lease_id']}:{lease['generation']}"
    return str(uuid5(NAMESPACE_URL, identity))


async def timed_turn(
    run_turn: Callable[[CuriosityTurnRequestV1], Awaitable[CuriosityTurnResultV1]],
    request: CuriosityTurnRequestV1,
) -> tuple[CuriosityTurnResultV1, dict[str, Any]]:
    """Run one Hub turn and measure it from this process: wall seconds
    around the RPC (monotonic clock, so a host clock step cannot make it
    negative) plus ISO-UTC start/finish stamps. Hub reports its own
    `debug.elapsed_sec` too; this one is the runner's, so it also covers
    bus transit and reply decode. If `run_turn` raises there is no meta --
    the caller's failure path derives the correlation instead."""
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    result = await run_turn(request)
    elapsed = time.monotonic() - t0
    finished = datetime.now(timezone.utc)
    meta = {
        "turn_correlation_id": request.correlation_id,
        "harness_elapsed_sec": round(elapsed, 3),
        "harness_started_at": started.isoformat(),
        "harness_finished_at": finished.isoformat(),
    }
    return result, meta


def recorded_turn_correlation_id(state: dict[str, Any]) -> str | None:
    """The correlation the last issued harness turn ran under, as persisted
    by the node itself -- `harness_turn_meta` first, then the older
    `debug.turn_correlation_id` the leased path has stamped since before
    this key existed. None when nothing recorded it; never derived here."""
    meta = state.get("harness_turn_meta")
    if isinstance(meta, dict):
        value = meta.get("turn_correlation_id")
        if isinstance(value, str) and value:
            return value
    debug = state.get("debug")
    if isinstance(debug, dict):
        value = debug.get("turn_correlation_id")
        if isinstance(value, str) and value:
            return value
    return None


def failed_turn_correlation_id(state: dict[str, Any]) -> str | None:
    """For the runner's own failure path (`_drive`), where the state is the
    exact snapshot the raising node ran against: the recorded value if the
    turn completed earlier, else the identity `harness_turn` would have
    derived from this same state. None rather than a raise when the state
    is too bare to derive from."""
    recorded = recorded_turn_correlation_id(state)
    if recorded:
        return recorded
    try:
        return turn_correlation_id(state) or None  # type: ignore[arg-type]
    except (KeyError, TypeError):
        return None


def failed_turn_meta(state: dict[str, Any]) -> dict[str, Any]:
    """`{"harness_turn_meta": {"turn_correlation_id": ...}}` for a turn
    attempt that is being abandoned while its lease is still on the state --
    the admitted wrapper's failure returns and the recovery fence both clear
    the lease next, after which the fenced id can no longer be re-derived.
    Empty (not a raise) when the state cannot name one, so a malformed lease
    still reaches the caller's own failure handling."""
    try:
        corr = turn_correlation_id(state)  # type: ignore[arg-type]
    except (KeyError, TypeError):
        return {}
    return {"harness_turn_meta": {"turn_correlation_id": corr}} if corr else {}


def make_nodes(deps: Deps) -> dict[str, Callable[[CuriosityRunState], Awaitable[dict[str, Any]]]]:
    async def harness_turn(state: CuriosityRunState) -> dict[str, Any]:
        brief = _brief(state)
        attempt = int(state.get("attempt") or 0) + 1
        request = CuriosityTurnRequestV1(
            run_id=state["run_id"],
            correlation_id=turn_correlation_id(state),
            prompt=brief.prompt,
            fcc_model_label=brief.fcc_model_label,
            timeout_sec=brief.timeout_sec,
            source_tag=brief.source_tag,
            attempt=attempt,
            lease=state.get("lease"),
            assigned_lane=(state.get("lease") or {}).get("lane"),
        )
        result, meta = await timed_turn(deps.run_turn, request)
        if not result.ok or not result.text.strip():
            raise HarnessTurnFailed(result.error or "empty_generation")
        debug = dict(result.debug)
        if state.get("lease"):
            debug.update(turn_correlation_id=request.correlation_id, parent_correlation_id=state["correlation_id"])
        return {"text": result.text, "debug": debug, "attempt": attempt, "harness_turn_meta": meta}

    async def read_turn_result(state: CuriosityRunState) -> dict[str, Any]:
        found = await deps.read_turn_result(state["run_id"])
        return {
            "outcome": found.get("outcome"),
            "footprint": found.get("footprint"),
            "hops": list(found.get("hops") or []),
            "evidence_summary": found.get("evidence_summary"),
            "graph_readable": bool(found.get("graph_readable", False)),
            "self_definition": found.get("self_definition"),
            "lived_answer": found.get("lived_answer"),
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
            line=brief.line,
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


# finish_detail keys copied from `harness_turn_meta` when present. Additive
# and optional: a run whose turn predates the key finishes without them.
HARNESS_META_DETAIL_KEYS = ("harness_elapsed_sec", "harness_started_at", "harness_finished_at")


def harness_meta_detail(state: dict[str, Any]) -> dict[str, Any]:
    """The timing/correlation slice of a finish detail, only the keys the
    runner actually has. Never raises on a bare or pre-key state."""
    out: dict[str, Any] = {}
    corr = recorded_turn_correlation_id(state)
    if corr:
        out["turn_correlation_id"] = corr
    meta = state.get("harness_turn_meta")
    if isinstance(meta, dict):
        for key in HARNESS_META_DETAIL_KEYS:
            value = meta.get(key)
            if value is not None:
                out[key] = value
    return out


def finish_detail(state: CuriosityRunState) -> dict[str, Any]:
    """What Hub needs from a completed run to decide outreach, bounded.
    Timing/correlation keys (`turn_correlation_id`, `harness_elapsed_sec`,
    `harness_started_at`, `harness_finished_at`) are present only when the
    runner recorded them -- see `harness_turn_meta`."""
    outcome = state.get("outcome") or {}
    text = state.get("text") or ""
    brief = state.get("brief") or {}
    lived = state.get("lived_answer")
    family = ""
    if isinstance(lived, dict) and lived:
        family = str(lived.get("family") or "lived").strip() or "lived"
    elif str(brief.get("line") or "") == "self_inquiry":
        # Anatomy self-inquiry writes SelfDefinition; lived writes LivedAnswer.
        family = "anatomy" if state.get("self_definition") else ""
    return {
        "line": str(brief.get("line") or "investigate"),
        "self_definition": state.get("self_definition"),
        "lived_answer": lived,
        "self_question_family": family,
        "reach_out": bool(outcome.get("reach_out")),
        "reach_out_why": str(outcome.get("reach_out_why") or "")[:1000],
        "continue_line": bool(outcome.get("continue_line")),
        "finding_text": text[:FINDING_TEXT_CAP],
        "journal_entry_id": state.get("journal_entry_id"),
        "attempts": int(state.get("attempt") or 0),
        **harness_meta_detail(state),
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
