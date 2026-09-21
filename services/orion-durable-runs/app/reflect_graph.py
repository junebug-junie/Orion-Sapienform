"""The self_study.reflect run as a LangGraph state graph.

Nodes, in order (`orion.schemas.durable_run.SELF_STUDY_REFLECT_NODES`):

    llm_call -> finish

Its own graph, not a branch inside `graph.py` or `self_sense_graph.py` --
reflect's downstream work (validating the model's raw findings against real
evidence, publishing to the journal and `self_concept_history`) needs the
FULL snapshot and induced concepts for evidence-chain construction, not just
the small `self_study_reflect_input` summary this graph's brief carries.
That logic is cortex-exec's own (`services/orion-cortex-exec/app/self_study.py`:
`_finding_from_llm_item`, `publish_self_reflection_artifacts`,
`publish_self_concept_history_from_reflection`) and stays there, unmoved --
cortex-exec dispatches this run, waits synchronously for its completion
event (same external contract `_call_self_study_reflect_llm` always had:
`list[dict] | None`, a list of raw finding dicts on success), and does the
rest itself with the snapshot/concepts it already has in scope. This graph
does ONLY the one thing that benefits from durable admission: the actual LLM
call, which can now request GPU2 elastic-burst capacity the same way
investigation/self-inquiry/self-sense-eval do.

`llm_call` sends the exact same shape `CortexClientRequest`
(`verb="self_study.reflect"`, `options={"policy_dispatch_only": True, ...}`)
cortex-exec's `_call_self_study_reflect_llm` always built directly -- this
patch only moves WHERE that request is sent from, not its shape. The
fence/think-block stripping regexes are a small local copy of
`self_study.py`'s own (same "unrelated producers, no shared caller" judgment
that file's own docstring already makes for the same regexes relative to
`orion.journaler.worker`'s).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

logger = logging.getLogger("orion-durable-runs.reflect_graph")

SELF_STUDY_REFLECT_VERB = "self_study.reflect"

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>", flags=re.IGNORECASE | re.DOTALL)


def _strip_reflect_text(text: str) -> str:
    stripped = _THINK_BLOCK_RE.sub(" ", text).strip()
    match = _JSON_FENCE_RE.search(stripped)
    return match.group(1).strip() if match else stripped


def parse_reflect_findings(text: str) -> list[dict[str, Any]] | None:
    """The model's raw completion text -> a list of raw finding dicts, or
    None on any parse/shape failure (empty text, bad JSON, wrong top-level
    shape). Pure -- no IO -- so it's testable without a fake bus. Mirrors
    cortex-exec's own `_call_self_study_reflect_llm` tail exactly."""
    if not text:
        return None
    try:
        parsed = json.loads(_strip_reflect_text(text))
    except (json.JSONDecodeError, ValueError):
        return None
    findings = parsed.get("findings") if isinstance(parsed, dict) else None
    if not isinstance(findings, list):
        return None
    return [item for item in findings if isinstance(item, dict)]


class ReflectRunState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]  # CuriosityRunBriefV1.model_dump(), reflect shape
    attempt: int
    # llm_call
    findings: list[dict[str, Any]]
    llm_call_ok: bool
    llm_call_error: str | None
    # finish
    status: str
    admission: dict[str, Any]
    lease: dict[str, Any] | None
    retry_at: str | None
    last_error: str | None
    requested_at: str
    retry_node: str | None
    tail_attempts: dict[str, int]


class ReflectLlmCallFailed(RuntimeError):
    """Raised by `llm_call` only on a transport-level failure (RPC error,
    timeout, undecodable reply) -- never for a non-ok result, empty text, or
    unparseable JSON, all of which `build_row`-equivalent handling here
    turns into `llm_call_ok=False` state instead, so the run still reaches
    `finish` and cortex-exec's synchronous waiter gets a real (empty)
    completion rather than hanging until ITS OWN timeout. The thread stays
    resumable at `llm_call` on this."""


@dataclass
class Deps:
    call_reflect_llm: Callable[[dict[str, Any], str], Awaitable[list[dict[str, Any]] | None]]
    """(self_study_reflect_input, llm_route) -> raw findings list, or None on
    any non-transport failure (bad input, non-ok result, empty/unparseable
    text, wrong JSON shape) -- mirrors `_call_self_study_reflect_llm`'s own
    return contract exactly."""


def make_nodes(deps: Deps) -> dict[str, Callable[[ReflectRunState], Awaitable[dict[str, Any]]]]:
    async def llm_call(state: ReflectRunState) -> dict[str, Any]:
        brief = state["brief"]
        attempt = int(state.get("attempt") or 0) + 1
        reflect_input = brief.get("self_study_reflect_input") or {}
        llm_route = brief.get("llm_route") or ""
        try:
            findings = await deps.call_reflect_llm(reflect_input, llm_route)
        except Exception as exc:  # noqa: BLE001 -- transport failure, resumable
            raise ReflectLlmCallFailed(f"{type(exc).__name__}: {exc}") from exc
        if findings is None:
            logger.info("self_study_reflect_llm_call_no_findings run=%s", state["run_id"])
            return {"findings": [], "llm_call_ok": False, "llm_call_error": "llm_call_failed", "attempt": attempt}
        return {"findings": findings, "llm_call_ok": True, "llm_call_error": None, "attempt": attempt}

    async def finish(state: ReflectRunState) -> dict[str, Any]:
        return {"status": "completed"}

    return {"llm_call": llm_call, "finish": finish}


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    """What cortex-exec's synchronous waiter needs from a completed reflect
    run, bounded. `findings` is the raw model output -- cortex-exec still
    validates every item against its own snapshot/concepts before trusting
    any of it, same as it always has; this is not pre-validated evidence."""
    return {
        "line": "reflect",
        "llm_call_ok": bool(state.get("llm_call_ok")),
        "llm_call_error": state.get("llm_call_error"),
        "findings": list(state.get("findings") or []),
        "attempts": int(state.get("attempt") or 0),
    }


def build_reflect_graph(deps: Deps, checkpointer: Any):
    """Compile the graph with the given saver. Imported lazily so the
    schema/contract half of this package stays importable without langgraph."""
    from langgraph.graph import END, START, StateGraph

    from orion.schemas.durable_run import SELF_STUDY_REFLECT_NODES

    nodes = make_nodes(deps)
    g: StateGraph = StateGraph(ReflectRunState)
    for name in SELF_STUDY_REFLECT_NODES:
        g.add_node(name, nodes[name])
    g.add_edge(START, SELF_STUDY_REFLECT_NODES[0])
    for a, b in zip(SELF_STUDY_REFLECT_NODES, SELF_STUDY_REFLECT_NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(SELF_STUDY_REFLECT_NODES[-1], END)
    return g.compile(checkpointer=checkpointer)
