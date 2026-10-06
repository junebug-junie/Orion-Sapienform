"""memory.episode_distill: one closed conversation episode -> Orion-voiced memories (SHADOW).

    load_episode -> resource_request -> resource_wait -> distill -> persist -> finish
                                                         distill (failed attempt) -> retry_wait -> resource_request

Spec: docs/superpowers/specs/2026-09-30-memory-episode-redesign-design.md, section 1 (Stage 1).

* ``load_episode`` (no GPU): reads the episode's turns, FULL text, from chat_history_log and the
  candidate referent keys; both are checkpointed, so a resumed run distills exactly what it read.
* ``distill`` runs under ``AdmissionRuntime.execute`` (the hold is heartbeated while the model
  answers) and calls the LLM gateway DIRECTLY on route ``memory_distill`` with the hold's ref
  (``options.gpu_lease``): no cortex-orch hop, because orch's recall step would mix retrieved
  memories into the evidence. An unparseable answer is a failed attempt (bounded, backed off);
  waiting for the hold never is.
* ``persist`` lets the hold go first (no GPU needed), then validates deterministically
  (orion.memory.episode.validate) and writes the shadow tables in one idempotent transaction.
  A crash after the write resumes at ``persist`` and re-writes nothing (deterministic ids).

Nothing here changes live behavior: the tables are read by no consumer except the report.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, replay_if_requeued,
    resource_nodes, taken_back,
)
from orion.memory.episode.distill import (
    UNMARKED_TEMPLATE_VERSION,
    parse_distillation,
    render_prompt,
    template_prompt_version,
    turns_from_state,
)
from orion.memory.episode.validate import coverage, validate_distillation
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.memory_episode import EpisodeDistillBriefV1

LoadFn = Callable[[EpisodeDistillBriefV1], Awaitable[dict[str, Any]]]
"""brief -> {"turns": [turn state dicts], "candidate_referents": [keys]}."""
CallLlmFn = Callable[..., Awaitable[dict[str, Any]]]
"""(prompt, *, brief, run_id, correlation_id, gpu_lease) -> {"text", "usage", "model", "latency_ms"}.
Raises on transport / non-ok / empty text: each is a bounded attempt."""
PersistFn = Callable[..., Awaitable[dict[str, int]]]
"""(**kwargs of orion.memory.episode.store.persist_episode minus pool) -> counts."""

EPISODE_DISTILL_MIN_ATTEMPTS = 4


class EpisodeDistillState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int
    hold_takebacks: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    status: str
    last_error: str | None
    turns: list[dict[str, Any]]
    candidate_referents: list[str]
    loaded_at: str | None
    distill_started_at: str | None
    answer_text: str | None
    # The version of the template the answer was generated from (stamped at render time). Absent on
    # checkpoints written before this field existed: those were rendered from the unmarked v2 template.
    rendered_prompt_version: str | None
    usage: dict[str, Any]
    model: str | None
    llm_latency_ms: int | None
    persisted: dict[str, int] | None


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    brief = state.get("brief") or {}
    persisted = state.get("persisted") or {}
    return {
        "line": "memory",
        "episode_id": brief.get("episode_id"),
        "turns": len(state.get("turns") or []),
        "memories": persisted.get("memories"),
        "rejections": persisted.get("rejections"),
        "questions": persisted.get("questions"),
        "downgrades": persisted.get("downgrades"),
        "attempts": int(state.get("attempt") or 0),
    }


def _hold_wait_ms(state: dict[str, Any]) -> int | None:
    """Time from the episode loading (just before the hold request) to the distill call starting."""
    try:
        start = datetime.fromisoformat(str(state["loaded_at"]))
        end = datetime.fromisoformat(str(state["distill_started_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def rendered_prompt_version(state: dict[str, Any]) -> str:
    """The template version the stored answer came from. A checkpoint written before the graph
    stamped it was rendered by the unmarked (v2) template; the brief's version is not used, because
    the brief comes from memory-consolidation's image, not the one that rendered the prompt."""
    return str(state.get("rendered_prompt_version") or UNMARKED_TEMPLATE_VERSION)


def build_episode_distill_graph(load: LoadFn, call_llm: CallLlmFn, persist: PersistFn, admission: AdmissionDeps,
                                checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def load_episode(state: EpisodeDistillState) -> dict:
        brief = EpisodeDistillBriefV1.model_validate(state["brief"])
        loaded = await load(brief)
        turns = list(loaded.get("turns") or [])
        if not turns:
            return {"status": "failed", "last_error": "episode_turns_missing", "turns": []}
        return {"turns": turns, "candidate_referents": list(loaded.get("candidate_referents") or []),
                "loaded_at": admission.now().isoformat(), "status": "running"}

    async def operation(state: dict[str, Any]) -> dict[str, Any]:
        lease = state["lease"]
        ref = GpuLeaseRefV1.model_validate({k: lease[k] for k in ("lease_id", "generation", "role", "holder")})
        brief = EpisodeDistillBriefV1.model_validate(state["brief"])
        started = admission.now().isoformat()
        rendered_version = template_prompt_version()
        prompt = render_prompt(episode_id=brief.episode_id, turns=turns_from_state(state["turns"]),
                               candidate_referents=state.get("candidate_referents") or [])
        answer = await call_llm(prompt, brief=brief, run_id=state["run_id"],
                                correlation_id=state["correlation_id"], gpu_lease=ref)
        text = str(answer.get("text") or "")
        if not text.strip():
            raise ValueError("empty_generation")
        parse_distillation(text)  # raises -> a bounded attempt; the text is re-parsed in persist
        return {"answer_text": text, "rendered_prompt_version": rendered_version,
                "usage": dict(answer.get("usage") or {}), "model": answer.get("model"),
                "llm_latency_ms": answer.get("latency_ms"), "distill_started_at": started,
                "attempt": int(state.get("attempt") or 0) + 1}

    async def distill_node(state: EpisodeDistillState) -> dict:
        try:
            result = await admission.execute(dict(state), operation)
            return {**result, "status": "running", "last_error": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:
            return await taken_back(admission, dict(state), exc.release_reason, f"{type(exc).__name__}: {exc}",
                                    {"status": "waiting_resource"})
        except Exception as exc:  # noqa: BLE001 -- a real distill attempt failed: bounded re-try
            replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
            if replay is not None:
                return replay
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt >= max(admission.max_attempts, EPISODE_DISTILL_MIN_ATTEMPTS):
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error}
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempt - 1))
            return {**released, "status": "retrying", "attempt": attempt, "last_error": error,
                    "retry_node": None, "retry_at": (admission.now() + timedelta(seconds=delay)).isoformat()}

    async def retry_wait(state: EpisodeDistillState) -> dict:
        from langgraph.types import interrupt

        if admission.now() < datetime.fromisoformat(state["retry_at"]):
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        return {"status": "retrying"}

    async def persist_node(state: EpisodeDistillState) -> dict:
        released = await admission.release(dict(state), "completed")
        brief = EpisodeDistillBriefV1.model_validate(state["brief"])
        turns = turns_from_state(state["turns"])
        prompt_version = rendered_prompt_version(dict(state))
        result = validate_distillation(parse_distillation(state["answer_text"] or ""), turns,
                                       episode_id=brief.episode_id, prompt_version=prompt_version)
        counts = await persist(
            episode_id=brief.episode_id, run_id=state["run_id"], result=result, model_route=brief.llm_route,
            model=state.get("model"), prompt_version=prompt_version, usage=state.get("usage") or {},
            llm_latency_ms=state.get("llm_latency_ms"), hold_wait_ms=_hold_wait_ms(dict(state)),
            coverage=coverage(result, turns)["coverage"],
        )
        return {**released, "persisted": dict(counts), "status": "running"}

    async def finish(state: EpisodeDistillState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: EpisodeDistillState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(EpisodeDistillState)
    for name, node in {"load_episode": load_episode, "resource_request": resource_request,
                       "resource_wait": resource_wait, "distill": distill_node, "retry_wait": retry_wait,
                       "persist": persist_node, "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "load_episode")
    g.add_conditional_edges("load_episode", lambda s: "failed" if s.get("status") == "failed" else "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "distill", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("distill", lambda s: (
        "resource_request" if s.get("status") == "waiting_resource"
        else "retry_wait" if s.get("status") == "retrying"
        else "failed" if s.get("status") == "failed" else "persist"))
    g.add_edge("retry_wait", "resource_request")
    g.add_edge("persist", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)


def base_run_id(episode_id: str) -> str:
    return f"memdistill-{episode_id}"


def request_from_closed_event(event: dict[str, Any], *, settings: Any, now: datetime | None = None,
                              run_id: str | None = None):
    """``memory.episode.closed.v1`` -> the DurableRunRequestV1 that distills it, or None to skip.

    run_id is deterministic per episode, so a re-delivered close event is a duplicate submit the
    admission store refuses, never a second distillation. Skipped (command-only) episodes and
    episodes from a non-direct platform get no run.
    """
    from orion.schemas.durable_run import DurableRunRequestV1
    from orion.schemas.memory_episode import MEMORY_EPISODE_DISTILL_WORKFLOW, MemoryEpisodeClosedV1
    from orion.schemas.resource_admission import ResourceRequirementV1

    closed = MemoryEpisodeClosedV1.model_validate(event)
    if closed.episode_status != "closed" or closed.source_platform or not closed.turn_ids:
        return None
    now = now or datetime.now(timezone.utc)
    route = str(settings.memory_episode_distill_route)
    brief = EpisodeDistillBriefV1(
        episode_id=closed.episode_id,
        source_platform=closed.source_platform,
        turn_ids=list(closed.turn_ids),
        started_at=closed.started_at,
        ended_at=closed.ended_at,
        close_reason=closed.close_reason,
        llm_route=route,
        timeout_sec=float(settings.memory_episode_distill_timeout_sec),
        max_tokens=int(settings.memory_episode_distill_max_tokens),
    )
    return DurableRunRequestV1(
        run_id=run_id or base_run_id(closed.episode_id),
        workflow=MEMORY_EPISODE_DISTILL_WORKFLOW,
        correlation_id=closed.episode_id,
        brief=brief,
        admission=ResourceRequirementV1(
            resource=f"llm.route.{route}",
            preferred_lane=route,
            priority="system",
            # Under DURABLE_RUNS_MAX_AGE_HOURS (24 h): the spec's bound on a distill run.
            deadline_at=now + timedelta(hours=float(settings.memory_episode_distill_deadline_hours)),
        ),
    )
