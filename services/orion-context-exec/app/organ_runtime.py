from __future__ import annotations

import logging
from typing import Any

from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.context_exec import ContextExecRequestV1

from . import llm_tools, recall_tools, repo_tools, trace_tools
from .organ_status import record_recall, record_repo, record_trace
from .settings import settings

logger = logging.getLogger("orion-context-exec.organ_runtime")


class OrganRuntime:
    """Read-only organ broker for a single context-exec run."""

    def __init__(
        self,
        *,
        bus: OrionBusAsync | None,
        request: ContextExecRequestV1,
        run_id: str,
        llm_route: str | None = None,
    ) -> None:
        self.bus = bus
        self.request = request
        self.run_id = run_id
        self.llm_route = (llm_route or request.llm_profile or "chat").strip().lower()
        self._trace_reads: dict[str, dict[str, Any]] = {}
        self.pending_llm_subcalls: list[dict[str, Any]] = []
        self.llm_rpc_calls: list[dict[str, Any]] = []
        self.organ_status: dict[str, dict[str, Any]] | None = None

    def record_llm_subcall(
        self,
        *,
        route: str,
        prompt: str,
        context: Any = None,
        schema: str | None = None,
    ) -> None:
        entry = {
            "route": str(route).strip().lower(),
            "prompt": prompt,
            "context": context,
            "schema": schema,
            "result": None,
        }
        self.pending_llm_subcalls.append(entry)
        self.llm_rpc_calls.append({"route": entry["route"], "prompt": prompt})

    async def flush_llm_subcalls(self) -> None:
        for entry in self.pending_llm_subcalls:
            if entry.get("result") is not None:
                continue
            entry["result"] = await llm_tools.llm_chat_route(
                self.bus,
                prompt=str(entry["prompt"]),
                route=str(entry["route"]),
                correlation_id=self.request.correlation_id,
                session_id=self.request.session_id,
                user_id=self.request.user_id,
                context=entry.get("context"),
                schema=entry.get("schema"),
            )

    async def llm_chat(
        self,
        prompt: str,
        *,
        route: str | None = None,
        context: Any = None,
        schema: str | None = None,
        messages: Any = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        route_key = (route or self.llm_route).strip().lower()
        result = await llm_tools.llm_chat_route(
            self.bus,
            prompt=prompt,
            route=route_key,
            correlation_id=self.request.correlation_id,
            session_id=self.request.session_id,
            user_id=self.request.user_id,
            context=context,
            schema=schema,
            messages=messages,
            stop=stop,
        )
        self.llm_rpc_calls.append({"route": route_key, "prompt": prompt, "result": result})
        return result

    async def traces_search(
        self,
        *,
        query: str | None = None,
        corr_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not settings.context_exec_real_trace_enabled:
            return []
        if not self.request.permissions.read_redis_traces:
            return []
        effective_corr = corr_id or self.request.correlation_id
        hits = trace_tools.traces_search(
            query=query,
            corr_id=effective_corr,
            run_id=run_id,
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for hit in hits:
            dumped = hit.model_dump(mode="json")
            out.append(dumped)
            if hit.handle:
                self._trace_reads[hit.handle] = trace_tools.traces_read(hit.handle)
        if self.organ_status is not None:
            record_trace(self.organ_status, out)
        return out

    async def traces_read(self, handle: str) -> dict[str, Any]:
        if handle in self._trace_reads:
            return self._trace_reads[handle]
        body = trace_tools.traces_read(handle)
        self._trace_reads[handle] = body
        return body

    async def recall_query(
        self,
        query: str,
        *,
        profile: str = "assist.light.v1",
        limit: int | None = None,
    ) -> dict[str, Any]:
        if not settings.context_exec_real_recall_enabled:
            return {"hits": []}
        if not self.request.permissions.read_recall:
            return {"hits": []}
        result = await recall_tools.recall_query(
            self.bus,
            query=query,
            profile=profile,
            limit=limit,
            correlation_id=self.request.correlation_id,
            session_id=self.request.session_id,
        )
        payload = result.model_dump(mode="json")
        if self.organ_status is not None:
            record_recall(self.organ_status, payload)
        return payload

    def repo_grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not settings.context_exec_real_repo_enabled:
            return []
        if not self.request.permissions.read_repo:
            return []
        hits = [
            h.model_dump(mode="json")
            for h in repo_tools.repo_grep(pattern, path=path, limit=limit)
        ]
        if self.organ_status is not None:
            record_repo(self.organ_status, hits)
        return hits

    def repo_read(self, path: str, *, max_chars: int | None = None) -> dict[str, Any] | None:
        if not settings.context_exec_real_repo_enabled:
            return None
        if not self.request.permissions.read_repo:
            return None
        cap = max_chars if max_chars is not None else settings.context_exec_repo_max_file_chars
        rf = repo_tools.repo_read(path, max_chars=cap)
        return rf.model_dump(mode="json") if rf else None
