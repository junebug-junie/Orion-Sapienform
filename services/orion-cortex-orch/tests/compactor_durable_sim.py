"""An in-test stand-in for the ``compactor.digest`` durable run, for orch-level compactor tests.

Production: ``_execute_*_compactor_pass`` fetches + chunks, then ``_submit_compactor_digest_run``
hands the digest calls to orion-durable-runs and returns ``accepted``; the durable run later calls
back with ``workflow_request.durable_digest`` and the same pass function finalizes (card + journal).

Here ``_submit_compactor_digest_run`` is replaced by :class:`InlineDurableRun`, which drives the
SAME step machine the durable graph drives (``orion.cognition.compactor.map_reduce``) with the
test's own ``call_verb_runtime`` fake answering the digest verb, then re-enters the pass with the
finished digest exactly as the durable run's finalize call would. So one orch test still covers
fetch -> chunk -> digest -> merge/fallback -> card + journal + metadata + notify.

Deliberately NOT simulated: attempts, pool waits, holds, restarts -- one attempt per call, and a
failed chunk call raises (as the durable run would after its bounded attempts). Those semantics are
the durable graph's and are tested there (services/orion-durable-runs/tests/test_compactor_digest_graph.py).
"""
from __future__ import annotations

import contextvars
from typing import Any

from orion.cognition.compactor.map_reduce import (
    SPECS, assemble, build_digest_request_payload, digest_from_payload, finalize_request_payload,
    merge_gave_up, next_call, record_merge, resolve_merge_without_call,
)
from orion.schemas.compactor_digest_run import CompactorDigestResultV1
from orion.schemas.cortex.contracts import CortexClientRequest

SIM_LEASE = {"lease_id": "hold-sim", "generation": 1, "role": "agent", "holder": "durable-runs:sim"}
_PASS_KWARGS: contextvars.ContextVar[dict | None] = contextvars.ContextVar("compactor_pass_kwargs", default=None)


def _payload_from_verb_result(result: Any) -> dict[str, Any]:
    """What cortex-orch's front door returns for a verb call, from a test's fake verb result."""
    output = getattr(result, "output", None)
    if output is None:
        output = getattr(result, "payload", None)
    output = output if isinstance(output, dict) else {}
    inner = output.get("result") if isinstance(output.get("result"), dict) else output
    ok = bool(getattr(result, "ok", True)) and (inner.get("status") in (None, "success"))
    payload = {"ok": ok, "status": inner.get("status") or ("success" if ok else "fail"),
               "final_text": inner.get("final_text"), "metadata": inner.get("metadata") or {}}
    if not getattr(result, "ok", True):
        payload["error"] = {"message": getattr(result, "error", None) or "verb_failed"}
    return payload


class InlineDurableRun:
    def __init__(self, namespaces: list[dict]) -> None:
        # Every live copy of app.workflow_runtime's globals: this suite purges and re-imports the
        # `app` package (tests/_orch_import_guard.py), so a test module can hold functions whose
        # globals are an older module object than the one in sys.modules.
        self.namespaces = namespaces
        self.briefs: list[Any] = []
        self.deadlines: list[Any] = []
        self.digest_requests: list[CortexClientRequest] = []
        self.results: list[CompactorDigestResultV1] = []
        self.finalized: Any = None

    def install(self, monkeypatch) -> "InlineDurableRun":
        for ns in self.namespaces:
            for name in ("_execute_github_compactor_pass", "_execute_chat_history_compactor_pass"):
                monkeypatch.setitem(ns, name, self._wrap(ns[name]))
            monkeypatch.setitem(ns, "_submit_compactor_digest_run", self.submit)
        return self

    def _wrap(self, original):
        async def wrapped(**kwargs):
            token = _PASS_KWARGS.set({**kwargs, "_original": original})
            try:
                result = await original(**kwargs)
            finally:
                _PASS_KWARGS.reset(token)
            if result.status == "accepted" and self.finalized is not None:
                # The durable run finished and finalized in-line: what the scheduler/test sees is
                # the finalize result, as it would be once the real run completed.
                finalized, self.finalized = self.finalized, None
                return finalized
            return result
        return wrapped

    async def submit(self, *, bus, source, correlation_id, brief, deadline_at):
        kwargs = _PASS_KWARGS.get()
        assert kwargs is not None, "submit outside a compactor pass"
        self.briefs.append(brief)
        self.deadlines.append(deadline_at)
        spec = SPECS[brief.kind]
        inputs = list(brief.inputs)
        partials: list[dict] = []
        merge = None
        attempts: list[dict] = []
        while True:
            decided = resolve_merge_without_call(spec, inputs, partials, merge)
            if decided is not None:
                merge = decided
                continue
            call = next_call(spec, inputs, partials, merge)
            if call is None:
                break
            request = CortexClientRequest.model_validate(build_digest_request_payload(
                spec, call["input"], workflow_id=brief.workflow_id, correlation_id=correlation_id,
                session_id=brief.session_id, user_id=brief.user_id, llm_route=brief.llm_route,
                timeout_sec=brief.timeout_sec, gpu_lease=SIM_LEASE))
            self.digest_requests.append(request)
            verb_result = await kwargs["call_verb_runtime"](
                bus, source=source, client_request=request, correlation_id=correlation_id,
                causality_chain=[], trace={}, timeout_sec=brief.timeout_sec)
            digest, error = digest_from_payload(spec, _payload_from_verb_result(verb_result))
            attempts.append({"step": call["label"], "ok": error is None, **({"error": error} if error else {})})
            if call["kind"] == "chunk":
                if error is not None:
                    raise kwargs["_original"].__globals__["WorkflowExecutionError"](error)
                partials.append(digest.model_dump(mode="json"))
            elif error is not None:
                merge = merge_gave_up(error)
            else:
                merge, refs_error = record_merge(spec, partials, digest)
                if refs_error:
                    attempts[-1] = {**attempts[-1], "ok": False, "error": refs_error}
        result = CompactorDigestResultV1(
            run_id="compactor:sim", kind=brief.kind, workflow_id=brief.workflow_id,
            window_label=brief.window_label, llm_route=brief.llm_route, attempts=attempts,
            gpu_roles=[SIM_LEASE["role"]], finalize=dict(brief.finalize),
            **assemble(spec, inputs, partials, merge, window_label=brief.window_label))
        self.results.append(result)
        # The exact request the durable run's finalize node sends (shared builder, no copy here).
        finalize_req = CortexClientRequest.model_validate(
            finalize_request_payload(brief, result, correlation_id=correlation_id))
        original = kwargs["_original"]
        self.finalized = await original(**{**{k: v for k, v in kwargs.items() if k != "_original"},
                                           "req": finalize_req})
        return {"run_id": result.run_id, "status": "accepted", "receipt_status": "waiting_resource",
                "generation": 1, "deadline_at": deadline_at.isoformat(), "chunk_count": len(inputs)}
