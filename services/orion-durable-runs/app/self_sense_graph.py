"""The self-sense-eval run as a LangGraph state graph.

Nodes, in order (`orion.schemas.durable_run.SELF_SENSE_EVAL_NODES`):

    ask_questions -> publish -> finish

Deliberately its OWN graph, not a branch inside `graph.py`'s curiosity
pipeline -- self-sense-eval has no material/worldview read, no journal, no
outreach, and asks four fixed questions instead of one open prompt. Sharing
`journal`/`read_turn_result` with investigation would have meant threading a
`line == "self_sense_eval"` conditional through nodes that write investigation-
shaped state into the same graph real production investigation runs already
depend on -- the workflow registry (this arc's step 1, PR #2273) exists
precisely so a second graph doesn't need to do that.

`ask_questions` reuses curiosity's own turn-execution RPC to Hub
(`CuriosityTurnRequestV1`/`Deps.run_turn`) -- Hub's `_handle_turn_request` is
already content-agnostic (it just runs `execute_unified_turn` with whatever
prompt/session_id/source_tag it's given), so no new Hub-side RPC channel was
needed, only the additive `CuriosityTurnRequestV1.session_id` field (this
patch) so each question runs under the same clean session
`orion.evals.self_sense_runner.SESSION_ID` names, not curiosity's shared
investigation session.

A per-question turn failure never aborts the run -- same contract Hub's own
in-process `_run_self_sense_eval` already has: `build_row` turns a missing
answer into a `"none"`-source row, not a crashed run, so one bad question
doesn't cost the other three.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

from orion.evals.self_sense_runner import SESSION_ID as SELF_SENSE_SESSION_ID
from orion.evals.self_sense_runner import build_row
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1

from app.graph import HARNESS_META_DETAIL_KEYS, timed_turn

logger = logging.getLogger("orion-durable-runs.self_sense_graph")

SELF_SENSE_EVAL_TAG = "curiosity_self_sense_eval"


class SelfSenseRunState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]  # CuriosityRunBriefV1.model_dump(), self_sense_eval shape
    attempt: int
    # ask_questions
    answers: dict[str, dict[str, Any]]  # question_key -> {"text": str, "debug": dict}
    # publish
    published: int
    failed: int
    empty: int
    # finish
    status: str
    admission: dict[str, Any]
    lease: dict[str, Any] | None
    retry_at: str | None
    last_error: str | None
    requested_at: str
    retry_node: str | None
    tail_attempts: dict[str, int]


class SelfSenseAskFailed(RuntimeError):
    """Raised by `ask_questions` only on a transport-level failure (no bus,
    RPC exception) -- never for an individual empty/failed answer, which
    `build_row` already turns into a valid "none"-source row per question.
    The thread stays resumable at `ask_questions` on this."""


@dataclass
class Deps:
    run_turn: Callable[[CuriosityTurnRequestV1], Awaitable[CuriosityTurnResultV1]]
    publish_rows: Callable[[list[Any]], Awaitable[tuple[int, int]]]  # -> (published, failed)


def _brief_questions(brief: dict[str, Any]) -> list[tuple[str, str]]:
    raw = brief.get("questions") or []
    return [(str(k), str(q)) for k, q in raw]


def make_nodes(deps: Deps) -> dict[str, Callable[[SelfSenseRunState], Awaitable[dict[str, Any]]]]:
    async def ask_questions(state: SelfSenseRunState) -> dict[str, Any]:
        brief = state["brief"]
        questions = _brief_questions(brief)
        attempt = int(state.get("attempt") or 0) + 1
        answers: dict[str, dict[str, Any]] = dict(state.get("answers") or {})
        for question_key, question in questions:
            if question_key in answers:
                continue  # already answered on a prior attempt (resume)
            # A fresh uuid4 per question, not a composite of the run's own
            # correlation_id -- two reasons, both review findings
            # (2026-09-21): (1) Hub's _turn_result_for cache/inflight key is
            # keyed on (run_id, correlation_id); reusing a derived, merely
            # question-scoped string still works for uniqueness, but a real
            # uuid4 is also what (2) orion.evals.self_sense_runner.is_uuid()
            # needs to recognise this as a genuine turn correlation id rather
            # than fall back to a synthetic derived one -- the same
            # "correlation_id IS the envelope's correlation_id" contract
            # every other self-sense-eval producer (Hub's in-process loop,
            # the host script) already gives it.
            correlation_id = str(uuid.uuid4())
            request = CuriosityTurnRequestV1(
                run_id=state["run_id"],
                correlation_id=correlation_id,
                prompt=question,
                fcc_model_label=brief.get("fcc_model_label"),
                timeout_sec=float(brief.get("timeout_sec") or 600.0),
                source_tag=SELF_SENSE_EVAL_TAG,
                attempt=attempt,
                lease=state.get("lease"),
                assigned_lane=(state.get("lease") or {}).get("lane"),
                session_id=SELF_SENSE_SESSION_ID,
            )
            try:
                result, meta = await timed_turn(deps.run_turn, request)
            except Exception as exc:  # noqa: BLE001 -- transport failure, resumable
                raise SelfSenseAskFailed(f"{type(exc).__name__}: {exc}") from exc
            if not result.ok:
                logger.info(
                    "self_sense_eval_question_failed run=%s question=%s error=%s",
                    state["run_id"], question_key, result.error,
                )
            # Runner-measured timing per question (same keys curiosity's
            # `harness_turn_meta` carries), so each answer's harness row is
            # joinable and timed without Hub's help.
            answers[question_key] = {
                "text": result.text or "", "debug": dict(result.debug or {}), "correlation_id": correlation_id,
                **{k: meta[k] for k in HARNESS_META_DETAIL_KEYS if meta.get(k) is not None},
            }
        return {"answers": answers, "attempt": attempt}

    async def publish(state: SelfSenseRunState) -> dict[str, Any]:
        brief = state["brief"]
        questions = _brief_questions(brief)
        answers = state.get("answers") or {}
        self_definition_version = brief.get("self_definition_version")
        lived_answers = brief.get("lived_answers") or []
        rows = []
        empty = 0
        for question_key, question in questions:
            answer = answers.get(question_key) or {}
            text = str(answer.get("text") or "")
            if not text:
                empty += 1
            # The SAME correlation_id ask_questions minted for this
            # question's real turn -- never a derived/composite string, so
            # build_row/is_uuid recognise it as genuine and skip the
            # synthetic fallback (review finding, 2026-09-21). Falls back to
            # a fresh uuid4 only if ask_questions somehow never populated
            # this key (shouldn't happen -- every question is always
            # written, even on failure -- kept defensive rather than a
            # KeyError over a publish-time edge case).
            correlation_id = str(answer.get("correlation_id") or uuid.uuid4())
            rows.append(
                build_row(
                    run_id=state["run_id"],
                    question_key=question_key,
                    question=question,
                    http_text=text,
                    trace_text=None,
                    correlation_id=correlation_id,
                    self_definition_version=self_definition_version,
                    lived_answers=lived_answers,
                )
            )
        published, failed = await deps.publish_rows(rows)
        return {"published": published, "failed": failed, "empty": empty}

    async def finish(state: SelfSenseRunState) -> dict[str, Any]:
        return {"status": "completed"}

    return {"ask_questions": ask_questions, "publish": publish, "finish": finish}


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    """What a listener needs from a completed self-sense-eval run, bounded.
    No `reach_out`/`continue_line` -- self-sense-eval never asks to reach
    out, unlike investigation/self-inquiry."""
    return {
        "line": "self_sense_eval",
        "published": int(state.get("published") or 0),
        "failed": int(state.get("failed") or 0),
        "empty": int(state.get("empty") or 0),
        "attempts": int(state.get("attempt") or 0),
        "turns": _turns_detail(state),
    }


def _turns_detail(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-question `{turn_correlation_id, harness_elapsed_sec, ...}` --
    the same field names as curiosity's finish detail, keyed by question,
    since one self-sense run is several harness turns. Timing keys are
    present only for answers recorded after the runner started measuring;
    the correlation is always there (ask_questions always writes it).
    Bounded by the brief's question list (single digits)."""
    answers = state.get("answers")
    if not isinstance(answers, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, answer in answers.items():
        if not isinstance(answer, dict):
            continue
        entry: dict[str, Any] = {}
        corr = answer.get("correlation_id")
        if isinstance(corr, str) and corr:
            entry["turn_correlation_id"] = corr
        for k in HARNESS_META_DETAIL_KEYS:
            if answer.get(k) is not None:
                entry[k] = answer[k]
        if entry:
            out[str(key)] = entry
    return out


def build_self_sense_graph(deps: Deps, checkpointer: Any):
    """Compile the graph with the given saver. Imported lazily so the
    schema/contract half of this package stays importable without langgraph."""
    from langgraph.graph import END, START, StateGraph

    nodes = make_nodes(deps)
    g: StateGraph = StateGraph(SelfSenseRunState)
    from orion.schemas.durable_run import SELF_SENSE_EVAL_NODES

    for name in SELF_SENSE_EVAL_NODES:
        g.add_node(name, nodes[name])
    g.add_edge(START, SELF_SENSE_EVAL_NODES[0])
    for a, b in zip(SELF_SENSE_EVAL_NODES, SELF_SENSE_EVAL_NODES[1:]):
        g.add_edge(a, b)
    g.add_edge(SELF_SENSE_EVAL_NODES[-1], END)
    return g.compile(checkpointer=checkpointer)
