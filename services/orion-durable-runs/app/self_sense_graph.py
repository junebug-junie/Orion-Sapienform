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
from orion.schemas.gpu_pool import GpuLeaseRefV1

from app.graph import HARNESS_META_DETAIL_KEYS, timed_turn

logger = logging.getLogger("orion-durable-runs.self_sense_graph")

SELF_SENSE_EVAL_TAG = "curiosity_self_sense_eval"


class SelfSenseRunState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]  # CuriosityRunBriefV1.model_dump(), self_sense_eval shape
    attempt: int
    hold_takebacks: int   # times the pool took the hold back mid-node (never an attempt)
    # ask_questions
    answers: dict[str, dict[str, Any]]  # question_key -> {"text": str, "debug": dict}
    # publish
    published: int
    failed: int
    empty: int
    # finish
    status: str
    admission: dict[str, Any]
    lease: dict[str, Any] | None  # granted GPU pool hold ref (admitted runs)
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    retry_at: str | None
    last_error: str | None
    requested_at: str
    retry_node: str | None
    tail_attempts: dict[str, int]
    # The last question turn issued (admitted graph): {"question_key", "turn_correlation_id",
    # "timeout_sec"} -- checkpointed with the outcome, so a hung or failed turn is traceable.
    last_turn: dict[str, Any] | None


def _gpu_lease(state: dict[str, Any]) -> GpuLeaseRefV1 | None:
    lease = state.get("lease")
    if not lease:
        return None
    return GpuLeaseRefV1.model_validate({key: lease[key] for key in ("lease_id", "generation", "role", "holder")})


def no_answers_error(answers: dict[str, Any] | None) -> str | None:
    """None when at least one answer has text; otherwise a bounded reason naming the turns'
    own errors (deduplicated, in question order).

    Every question empty means nothing was measured: the run is a failure, not a "completed"
    run of "none"-source rows. Live 2026-10-02..06: four self-sense runs on the gpu2 worker
    "completed" with 16/16 blank answers while every LLM call behind them 500'd."""
    answers = answers or {}
    if not answers or any(str((a or {}).get("text") or "").strip() for a in answers.values()):
        return None
    errors: list[str] = []
    for answer in answers.values():
        err = str((answer or {}).get("error") or "empty_generation")
        if err not in errors:
            errors.append(err)
    return f"self_sense_no_answers: {'; '.join(errors)}"[:500]


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


# Hub is told a turn limit this much shorter than the node's own timer, so a slow turn comes back
# as Hub's timeout result (an empty answer, recorded) before the durable timer cancels it.
TURN_REPLY_GRACE_SEC = 30.0


def next_unanswered(state: dict[str, Any]) -> tuple[str, str] | None:
    """The first brief question with no recorded answer, or None when every one has one."""
    answers = state.get("answers") or {}
    for question_key, question in _brief_questions(state["brief"]):
        if question_key not in answers:
            return question_key, question
    return None


def question_share_sec(brief: dict[str, Any]) -> float:
    """One question's slice of the brief's budget (``timeout_sec`` split across the questions).

    Live 2026-10-23 (run 20261023T180730Z-0842e5): the node timer AND every question's turn limit
    were the whole 8840 s budget, so one hung question held agent-gpu2 for 8840 s and the run
    failed with every answer lost."""
    total = float(brief.get("timeout_sec") or 600.0)
    return max(1.0, total / max(1, len(_brief_questions(brief))))


def question_turn_id(state: dict[str, Any], question_key: str) -> str:
    """The turn id the admitted graph issues for ``question_key`` from THIS checkpointed state: a
    uuid5 over the run, the question, and every counter that changes when the turn is replayed
    (failed attempt, restart fence, a fresh hold, a pool take-back). Derivable from the checkpoint
    alone, so a restarted worker can cancel the turn its predecessor left running in Hub
    (``AdmissionRuntime._cancel_harness``); still a real UUID for ``is_uuid``, still fresh per
    replay for Hub's ``(run_id, correlation_id)`` result cache."""
    parts = (state.get("run_id"), question_key, int(state.get("attempt") or 0), int(state.get("turn_fence") or 0),
             int(state.get("hold_seq") or 0), int(state.get("hold_takebacks") or 0))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "orion:self_sense_eval:" + ":".join(map(str, parts))))


async def ask_one(deps: Deps, state: dict[str, Any], question_key: str, question: str, *,
                  correlation_id: str, timeout_sec: float, attempt: int) -> dict[str, Any]:
    """Ask one question under the run's hold; the recorded answer (never raises on a failed or
    empty turn -- that is an answer with ``error``). Raises ``SelfSenseAskFailed`` on transport."""
    brief = state["brief"]
    request = CuriosityTurnRequestV1(
        run_id=state["run_id"],
        correlation_id=correlation_id,
        prompt=question,
        fcc_model_label=brief.get("fcc_model_label"),
        timeout_sec=timeout_sec,
        source_tag=SELF_SENSE_EVAL_TAG,
        attempt=attempt,
        # The run's pool hold (stage 4.5): each question's calls attach to it. Its role is
        # never a route, so no assigned_lane.
        gpu_lease=_gpu_lease(state),
        session_id=SELF_SENSE_SESSION_ID,
    )
    # Stamp the in-flight id on the same state dict execute() holds, so a timeout/cancel can
    # address this uuid4 (not the lease-derived curiosity id). Cleared when the turn returns.
    inflight = state.setdefault("_inflight_turn_correlation_ids", [])
    if isinstance(inflight, list):
        inflight.append(correlation_id)
    try:
        result, meta = await timed_turn(deps.run_turn, request)
    except Exception as exc:  # noqa: BLE001 -- transport failure, resumable
        raise SelfSenseAskFailed(f"{type(exc).__name__}: {exc}") from exc
    finally:
        if isinstance(inflight, list) and correlation_id in inflight:
            inflight.remove(correlation_id)
    if not result.ok:
        logger.info(
            "self_sense_eval_question_failed run=%s question=%s error=%s",
            state["run_id"], question_key, result.error,
        )
    # Runner-measured timing per question (same keys curiosity's `harness_turn_meta` carries),
    # so each answer's harness row is joinable and timed without Hub's help.
    answer = {
        "text": result.text or "", "debug": dict(result.debug or {}), "correlation_id": correlation_id,
        **{k: meta[k] for k in HARNESS_META_DETAIL_KEYS if meta.get(k) is not None},
    }
    if not result.ok or not (result.text or "").strip():
        # Why this answer is empty, kept on state (and in finish_detail's turns) instead of
        # only in a log line.
        answer["error"] = str(result.error or "empty_generation")[:300]
    return answer


def make_nodes(deps: Deps) -> dict[str, Callable[[SelfSenseRunState], Awaitable[dict[str, Any]]]]:
    async def ask_questions(state: SelfSenseRunState) -> dict[str, Any]:
        brief = state["brief"]
        attempt = int(state.get("attempt") or 0) + 1
        answers: dict[str, dict[str, Any]] = dict(state.get("answers") or {})
        for question_key, question in _brief_questions(brief):
            if question_key in answers:
                continue  # already answered on a prior attempt (resume)
            # A fresh uuid4 per question, not a composite of the run's own correlation_id --
            # Hub's _turn_result_for cache keys on (run_id, correlation_id), and
            # orion.evals.self_sense_runner.is_uuid() must recognise it as a genuine turn
            # correlation id (review findings, 2026-09-21).
            answers[question_key] = await ask_one(
                deps, state, question_key, question, correlation_id=str(uuid.uuid4()),
                timeout_sec=float(brief.get("timeout_sec") or 600.0), attempt=attempt)
        return {"answers": answers, "attempt": attempt}

    async def publish(state: SelfSenseRunState) -> dict[str, Any]:
        no_answers = no_answers_error(state.get("answers"))
        if no_answers is not None:
            # Never publish non-measurements as a completed eval; finish reports the run failed.
            return {"published": 0, "failed": 0, "empty": len(state.get("answers") or {}),
                    "last_error": no_answers}
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
                    motor_steps=(answer.get("debug") or {}).get("harness_step_count"),
                )
            )
        published, failed = await deps.publish_rows(rows)
        return {"published": published, "failed": failed, "empty": empty}

    async def finish(state: SelfSenseRunState) -> dict[str, Any]:
        if no_answers_error(state.get("answers")) is not None:
            return {"status": "failed"}
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
        "attempts": _attempts(state),
        **({"error": str(state["last_error"])[:500]} if state.get("last_error") else {}),
        "turns": _turns_detail(state),
    }


def _attempts(state: dict[str, Any]) -> int:
    """Most attempts any question took. The admitted graph counts attempts per question (``attempt``
    resets when an answer lands; each answer records its own ``attempts``); the plain graph's
    ``attempt`` is its pass count."""
    answers = state.get("answers") if isinstance(state.get("answers"), dict) else {}
    per_question = [int(a.get("attempts") or 0) for a in answers.values() if isinstance(a, dict)]
    return max([*per_question, int(state.get("attempt") or 0)])


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
        if answer.get("error"):
            entry["error"] = str(answer["error"])[:300]
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
