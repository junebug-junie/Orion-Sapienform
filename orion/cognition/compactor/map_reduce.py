"""Compactor digest map-reduce, as a step machine a durable run can checkpoint.

Moved out of cortex-orch (``workflow_runtime._run_compactor_digest``) so the LLM half of
``github_compactor_pass`` / ``chat_history_compactor_pass`` can run as an admitted durable run
(workflow ``compactor.digest``, services/orion-durable-runs/app/compactor_digest_graph.py): every
chunk digest and the merge call is ONE graph node execution under a GPU pool hold, and its result
is checkpointed before the next call starts.

The machine is pure: :func:`next_call` says which call comes next from what is already
checkpointed, :func:`record_chunk` / :func:`record_merge` / :func:`merge_gave_up` fold one call's
outcome back in, and :func:`assemble` turns the finished progress into the day's digest. The
guarantees are the ones PR #2422 shipped:

* one call if the window fits, else one digest per chunk, then one merge call;
* a merge whose input is over :data:`DIGEST_INPUT_CHAR_BUDGET` is skipped (no call);
* a merge that drops any chunk's reference falls back to the deterministic join
  (``merge_mode=concatenated``), and so does a merge that failed every attempt;
* ``journal_body`` is never trimmed; over-budget card prose is fitted afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from orion.cognition.chat_history_compactor.digest import (
    build_chat_history_compactor_merge_input,
    concatenate_chat_partial_digests,
    fit_chat_compactor_digest_within_budget,
    parse_chat_history_compactor_digest_json,
)
from orion.cognition.compactor.chunking import json_char_len
from orion.cognition.compactor.constants import (
    DIGEST_INPUT_CHAR_BUDGET,
    DIGEST_LLM_ROUTE,
    DIGEST_MAX_TOKENS,
    DIGEST_ORCH_RPC_TIMEOUT_SEC,
)
from orion.cognition.compactor.digest import EMPTY_COMPLETION_ERROR
from orion.cognition.github_compactor.digest import (
    build_github_compactor_merge_input,
    concatenate_github_partial_digests,
    fit_digest_within_budget,
    parse_github_compactor_digest_json,
)
from orion.schemas.actions.chat_history_compactor import ChatHistoryCompactorDigestV1
from orion.cognition.compactor.constants import COMPACTOR_FINALIZE_RPC_TIMEOUT_SEC
from orion.schemas.actions.github_compactor import GithubCompactorDigestV1
from orion.schemas.compactor_digest_run import (
    DURABLE_DIGEST_KEY, CompactorDigestResultV1, CompactorDigestRunBriefV1,
)

CompactorKind = Literal["github", "chat"]


@dataclass(frozen=True)
class CompactorDigestSpec:
    kind: str
    verb: str
    prompt: str
    input_key: str
    metadata_key: str
    model_cls: Any
    parse_json: Callable[[str], Any]
    error_prefix: str
    fit_budget: Callable[[Any], tuple[Any, list[str]]]
    build_merge_input: Callable[..., dict[str, Any]]
    concatenate: Callable[..., Any]
    refs_field: str


SPECS: dict[str, CompactorDigestSpec] = {
    "github": CompactorDigestSpec(
        kind="github",
        verb="github_compactor_digest_v1",
        prompt="Compact merged PR activity into repo development digest.",
        input_key="github_compactor_input",
        metadata_key="github_compactor_digest",
        model_cls=GithubCompactorDigestV1,
        parse_json=parse_github_compactor_digest_json,
        error_prefix="github_compactor_digest_failed",
        fit_budget=fit_digest_within_budget,
        build_merge_input=build_github_compactor_merge_input,
        concatenate=concatenate_github_partial_digests,
        refs_field="pr_refs",
    ),
    "chat": CompactorDigestSpec(
        kind="chat",
        verb="chat_history_compactor_digest_v1",
        prompt="Compact recent Hub chat into a durable memory digest.",
        input_key="chat_history_compactor_input",
        metadata_key="chat_history_compactor_digest",
        model_cls=ChatHistoryCompactorDigestV1,
        parse_json=parse_chat_history_compactor_digest_json,
        error_prefix="chat_compactor_digest_failed",
        fit_budget=fit_chat_compactor_digest_within_budget,
        build_merge_input=build_chat_history_compactor_merge_input,
        concatenate=concatenate_chat_partial_digests,
        refs_field="turn_refs",
    ),
}


class CompactorDigestCallError(RuntimeError):
    """One digest call produced no usable digest (verb failure, rejected/empty/invalid JSON).
    The durable graph counts it as one bounded attempt."""


def build_digest_request_payload(
    spec: CompactorDigestSpec,
    input_payload: dict[str, Any],
    *,
    workflow_id: str,
    correlation_id: str,
    session_id: str,
    user_id: str | None = None,
    llm_route: str = DIGEST_LLM_ROUTE,
    timeout_sec: float = DIGEST_ORCH_RPC_TIMEOUT_SEC,
    gpu_lease: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``CortexClientRequest`` (as a dict) for one digest call through cortex-orch.

    Same verb/options the orch-local path sent (brain lane, JSON object, ``max_tokens``), with
    thinking off through ``chat_template_kwargs``, plus ``options.gpu_lease`` so cortex-exec forwards the run's hold and the
    gateway attaches the call to it instead of queueing behind it. No ``workflow_request`` and no
    ``durable_run`` key in metadata: cortex-orch must execute the verb, not re-enter a workflow.
    """
    options: dict[str, Any] = {
        "source": "orion-durable-runs:compactor.digest",
        "workflow_execution": True,
        "workflow_id": workflow_id,
        "response_format": {"type": "json_object"},
        "return_json": True,
        # Thinking OFF via the switch cortex-exec actually forwards to the gateway -> llama.cpp.
        # The old ``"reasoning": {"effort": "none"}`` here had no consumer anywhere: live
        # 2026-09-30 corr 2af9b6ea spent 16000 tokens (48k chars of reasoning_content) and was
        # cut off at finish_reason=length; 0db311e7 (09-25) capped at 8000 with content="".
        # With thinking off the same verb finishes in ~1200-1400 tokens (07f394ac, 383796de).
        "chat_template_kwargs": {"enable_thinking": False},
        "timeout_sec": float(timeout_sec),
        "max_tokens": int(DIGEST_MAX_TOKENS),
        "llm_route": llm_route,
    }
    if gpu_lease is not None:
        options["gpu_lease"] = dict(gpu_lease)
    return {
        "mode": "brain",
        "route_intent": "none",
        "verb": spec.verb,
        "packs": [],
        "options": options,
        "recall": {"enabled": False, "required": False, "max_items": 0},
        "context": {
            "messages": [],
            "raw_user_text": spec.prompt,
            "user_message": spec.prompt,
            "session_id": session_id,
            "user_id": user_id,
            "trace_id": correlation_id,
            "metadata": {
                "workflow_subverb": spec.verb,
                "workflow_id": workflow_id,
                spec.input_key: input_payload,
            },
        },
    }


def digest_from_payload(spec: CompactorDigestSpec, payload: dict[str, Any]) -> tuple[Any | None, str | None]:
    """Validate a digest verb result payload. ``(digest, None)`` on success, else
    ``(None, error_token)``."""
    if payload.get("ok") is False:
        error = payload.get("error")
        if isinstance(error, dict):
            error = error.get("message") or error
        return None, f"{spec.error_prefix}:{error or payload.get('status')}"
    result_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if result_metadata.get("structured_output_rejected"):
        return None, f"{spec.error_prefix}:structured_output_rejected"
    digest_raw = result_metadata.get(spec.metadata_key)
    try:
        if isinstance(digest_raw, dict):
            return spec.model_cls.model_validate(digest_raw), None
        return spec.parse_json(str(payload.get("final_text") or "")), None
    except (ValueError, TypeError) as exc:
        if str(exc) == EMPTY_COMPLETION_ERROR:
            return None, f"{spec.error_prefix}:empty_completion"
        return None, f"{spec.error_prefix}:invalid_json:{exc}"


# --- the step machine --------------------------------------------------------------------------

def step_label(index: int, total: int) -> str:
    return "single" if total == 1 else f"chunk_{index + 1}_of_{total}"


def _refs(spec: CompactorDigestSpec, digest: Any) -> list[str]:
    return list(getattr(digest, spec.refs_field) or [])


def partial_refs(spec: CompactorDigestSpec, partials: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for raw in partials:
        for ref in _refs(spec, spec.model_cls.model_validate(raw)):
            if ref not in seen:
                seen.append(ref)
    return seen


def merge_input_for(spec: CompactorDigestSpec, inputs: list[dict[str, Any]],
                    partials: list[dict[str, Any]]) -> dict[str, Any]:
    return spec.build_merge_input(base_input=inputs[0],
                                  partial_digests=[spec.model_cls.model_validate(p) for p in partials])


def next_call(spec: CompactorDigestSpec, inputs: list[dict[str, Any]], partials: list[dict[str, Any]],
              merge: dict[str, Any] | None) -> dict[str, Any] | None:
    """The next LLM call, or None when the digest can be assembled.

    ``{"kind": "chunk", "index", "label", "input"}`` or ``{"kind": "merge", "label", "input"}``.
    A merge whose input is over budget is never a call: :func:`resolve_merge_without_call`
    records it as skipped first.
    """
    total = len(inputs)
    if len(partials) < total:
        index = len(partials)
        return {"kind": "chunk", "index": index, "label": step_label(index, total), "input": inputs[index]}
    if total == 1 or merge is not None:
        return None
    return {"kind": "merge", "label": "merge", "input": merge_input_for(spec, inputs, partials)}


def resolve_merge_without_call(spec: CompactorDigestSpec, inputs: list[dict[str, Any]],
                               partials: list[dict[str, Any]], merge: dict[str, Any] | None) -> dict[str, Any] | None:
    """A merge record decided with no LLM call (input over budget), or None when a merge call
    should be made (or no merge applies)."""
    if len(inputs) <= 1 or len(partials) < len(inputs) or merge is not None:
        return None
    if json_char_len(merge_input_for(spec, inputs, partials)) > DIGEST_INPUT_CHAR_BUDGET:
        # The chunk bodies together do not fit one call: a merge would overrun the context (or be
        # silently truncated). Join the chunk digests instead.
        return {"status": "skipped", "digest": None, "reason": "merge_input_over_budget"}
    return None


def record_merge(spec: CompactorDigestSpec, partials: list[dict[str, Any]], merged: Any) -> tuple[dict[str, Any], str | None]:
    """Fold a successful merge call in: ``(merge_record, refs_missing_error_or_None)``. A merge
    that dropped any chunk's reference is not used -- the join keeps the coverage."""
    merged_refs = set(_refs(spec, merged))
    missing = [ref for ref in partial_refs(spec, partials) if ref not in merged_refs]
    if missing:
        return ({"status": "skipped", "digest": None, "reason": f"merge_dropped_refs:{len(missing)}"},
                f"refs_missing:{missing[:20]}")
    return {"status": "done", "digest": merged.model_dump(mode="json"), "reason": None}, None


def merge_gave_up(error: str) -> dict[str, Any]:
    """Every merge attempt failed: the chunk digests are still real model output over real input."""
    return {"status": "skipped", "digest": None, "reason": f"merge_failed:{str(error)[:200]}"}


def assemble(spec: CompactorDigestSpec, inputs: list[dict[str, Any]], partials: list[dict[str, Any]],
             merge: dict[str, Any] | None, *, window_label: str) -> dict[str, Any]:
    """The finished digest + how it was made. Pure; called on checkpointed progress only."""
    if not inputs or len(partials) != len(inputs):
        raise ValueError("compactor_digest_incomplete")
    merge_skipped_reason: str | None = None
    if len(inputs) == 1:
        digest = spec.model_cls.model_validate(partials[0])
        merge_mode = "single"
    elif merge is not None and merge.get("status") == "done" and isinstance(merge.get("digest"), dict):
        digest = spec.model_cls.model_validate(merge["digest"])
        merge_mode = "llm_merge"
    else:
        merge_skipped_reason = (merge or {}).get("reason") or "merge_missing"
        digest = spec.concatenate([spec.model_cls.model_validate(p) for p in partials], window_label=window_label)
        merge_mode = "concatenated"
    digest, trimmed_fields = spec.fit_budget(digest)
    return {
        "digest": digest.model_dump(mode="json"),
        "chunk_count": len(inputs),
        "merge_mode": merge_mode,
        "merge_skipped_reason": merge_skipped_reason,
        "trimmed_fields": list(trimmed_fields or []),
    }


def finalize_request_payload(brief: "CompactorDigestRunBriefV1", result: "CompactorDigestResultV1", *,
                             correlation_id: str) -> dict[str, Any]:
    """The cortex-orch workflow request that finishes the day (card + journal, no LLM): what the
    durable run's ``finalize`` node sends. ``durable_digest`` makes orch finalize instead of
    fetching; no ``durable_run`` key, so it can never re-submit."""
    workflow_request: dict[str, Any] = {
        "workflow_id": brief.workflow_id,
        DURABLE_DIGEST_KEY: result.model_dump(mode="json"),
    }
    policy = brief.finalize.get("execution_policy")
    if isinstance(policy, dict):
        workflow_request["execution_policy"] = {**policy, "invocation_mode": "immediate"}
    return {
        "mode": "brain",
        "route_intent": "none",
        "verb": None,
        "packs": [],
        "options": {"source": "orion-durable-runs", "policy_dispatch_only": True,
                    "timeout_sec": float(COMPACTOR_FINALIZE_RPC_TIMEOUT_SEC)},
        "recall": {"enabled": False, "required": False},
        "context": {
            "messages": [],
            "raw_user_text": f"{brief.workflow_id} finalize ({brief.window_label})",
            "user_message": f"{brief.workflow_id} finalize ({brief.window_label})",
            "session_id": brief.session_id,
            "user_id": brief.user_id,
            "trace_id": correlation_id,
            "metadata": {"workflow_request": workflow_request,
                         "workflow_dispatch_source": "orion-durable-runs"},
        },
    }
