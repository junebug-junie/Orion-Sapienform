"""One durable-run step run as one harness turn: the seam multi-step harness workflows sit on.

Why this module exists: curiosity.investigate and self_sense_eval already run harness turns from a
durable run, over the existing Hub turn RPC (``Runner._run_turn`` -> ``orion:curiosity:turn:request``
-> Hub ``execute_unified_turn`` -> orion-harness-governor's FCC motor). But each graph handles the
turn's outcome its own way: curiosity reads the result back in later nodes, and self_sense_eval
records an empty answer as a valid (empty) measurement. A workflow whose steps are *work* (the
planned ``reminiscence.episode``: five hops, each a harness turn) needs the opposite contract,
written once:

* the step's final text is checkpointed as the step's result (``state["steps"][step_key]``), with
  the turn's correlation id, so the turn is findable in ``cockpit_turn_sighting`` /
  ``harness_turn_trace``;
* a refusal, a timeout, an empty reply or a reply for another turn raises ``HarnessStepFailed``
  (retryable), so the graph's ``failed_attempt`` policy retries it. A step never succeeds with
  ``final_len=0``.

No new channel and no Hub change: the request is the same ``CuriosityTurnRequestV1`` self_sense_eval
sends (explicit ``session_id``, its own ``source_tag``), carrying the run's GPU pool hold, so Hub
fences the hold and the governor dispatches the turn exactly as it does a self_sense_eval question.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable

from app.graph import HARNESS_META_DETAIL_KEYS, _gpu_lease, timed_turn
from orion.schemas.durable_run import CuriosityTurnRequestV1, CuriosityTurnResultV1

RunTurn = Callable[[CuriosityTurnRequestV1], Awaitable[CuriosityTurnResultV1]]

# Hub is told a turn limit this much shorter than the node's own timer, so a slow turn comes back as
# Hub's timeout result (a step failure, retried) before the durable timer cancels it. Same value
# and reason as self_sense_graph.TURN_REPLY_GRACE_SEC.
TURN_REPLY_GRACE_SEC = 30.0
# What a step result keeps of Hub's success debug dict (``CuriosityInvestigation._generate``).
DEBUG_KEYS = ("elapsed_sec", "fcc_elapsed_sec", "harness_step_count", "harness_grounding_status", "fcc_model_label")
# How much of the step text a state row's detail carries (the checkpoint keeps all of it).
DETAIL_TEXT_CAP = 2000


class HarnessStepFailed(RuntimeError):
    """The harness turn came back without usable text (refusal, timeout, empty, wrong turn).

    A RuntimeError on purpose: ``admitted_graph.is_deterministic`` treats it as retryable, so the
    graph's ``failed_attempt`` policy spends one attempt and retries after backoff."""


def harness_step_correlation_id(state: dict[str, Any], step_key: str) -> str:
    """The turn id for ``step_key`` from THIS checkpointed state: a uuid5 over the run, the step,
    and every counter that changes when the turn is replayed (failed attempt, restart fence, a
    fresh hold, a pool take-back). Derivable from the checkpoint alone, so a restarted worker can
    cancel the turn its predecessor left running (``AdmissionRuntime._cancel_harness``); fresh per
    replay, so Hub's ``(run_id, correlation_id)`` result cache never hands a retry a stale answer.
    Same construction as ``self_sense_graph.question_turn_id``."""
    parts = (state.get("run_id"), step_key, int(state.get("attempt") or 0), int(state.get("turn_fence") or 0),
             int(state.get("hold_seq") or 0), int(state.get("hold_takebacks") or 0))
    workflow = str(state.get("workflow") or "durable")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"orion:harness_step:{workflow}:" + ":".join(map(str, parts))))


async def run_harness_step(
    run_turn: RunTurn,
    state: dict[str, Any],
    *,
    step_key: str,
    prompt: str,
    correlation_id: str,
    timeout_sec: float,
    source_tag: str,
    session_id: str,
) -> dict[str, Any]:
    """Run one harness turn under the run's hold; the step's result record.

    Raises ``HarnessStepFailed`` when the turn has no usable text. Transport exceptions from
    ``run_turn`` propagate unchanged (also retryable)."""
    request = CuriosityTurnRequestV1(
        run_id=state["run_id"],
        correlation_id=correlation_id,
        prompt=prompt,
        timeout_sec=max(1.0, float(timeout_sec)),
        source_tag=source_tag,
        attempt=int(state.get("attempt") or 0) + 1,
        gpu_lease=_gpu_lease(state),
        session_id=session_id,
    )
    result, meta = await timed_turn(run_turn, request)
    if result.run_id != request.run_id or result.correlation_id != request.correlation_id:
        raise HarnessStepFailed(f"harness_step_identity_mismatch:{step_key}")
    text = result.text or ""
    if not result.ok or not text.strip():
        raise HarnessStepFailed(f"harness_step_failed:{step_key}:{str(result.error or 'empty_generation')[:300]}")
    debug = result.debug if isinstance(result.debug, dict) else {}
    return {
        "step_key": step_key,
        "text": text,
        "final_len": len(text),
        "turn_correlation_id": correlation_id,
        "debug": {k: debug[k] for k in DEBUG_KEYS if k in debug},
        **{k: meta[k] for k in HARNESS_META_DETAIL_KEYS if meta.get(k) is not None},
    }


def step_detail(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """A step record bounded for a state row's ``detail`` (the checkpoint keeps the whole text)."""
    if not isinstance(record, dict):
        return None
    text = str(record.get("text") or "")
    return {
        **{k: v for k, v in record.items() if k not in ("text", "debug")},
        "text": text[:DETAIL_TEXT_CAP],
        "text_truncated": len(text) > DETAIL_TEXT_CAP,
    }
