from __future__ import annotations

import logging
from typing import Any, Callable

from orion.schemas.context_exec import ContextExecPermissionV1

from . import repo_tools
from .security import PolicyBlockedError, check_callable

logger = logging.getLogger("orion-context-exec.sandbox")


class ContextNamespace:
    """Proxy namespace exposed to RLM; outer service enforces permissions."""

    def __init__(
        self,
        *,
        permissions: ContextExecPermissionV1,
        subcall_fn: Callable[..., Any] | None = None,
        memory_fn: dict[str, Callable[..., Any]] | None = None,
        recall_fn: Callable[..., Any] | None = None,
        traces_fn: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.permissions = permissions
        self._subcall = subcall_fn
        self._memory = memory_fn or {}
        self._recall = recall_fn
        self._traces = traces_fn or {}
        self._final: Any = None
        self._final_var: str | None = None
        self._locals: dict[str, Any] = {}

    def _guard(self, name: str) -> None:
        check_callable(self.permissions, name)

    @property
    def memory(self) -> _MemoryProxy:
        return _MemoryProxy(self)

    @property
    def recall(self) -> _RecallProxy:
        return _RecallProxy(self)

    @property
    def repo(self) -> _RepoProxy:
        return _RepoProxy(self)

    @property
    def traces(self) -> _TracesProxy:
        return _TracesProxy(self)

    @property
    def llm(self) -> _LlmProxy:
        return _LlmProxy(self)

    def FINAL(self, obj: Any) -> None:
        self._final = obj

    def FINAL_VAR(self, name: str) -> None:
        self._final_var = name

    def get_final(self) -> Any:
        if self._final_var:
            return self._locals.get(self._final_var)
        return self._final

    def set_local(self, name: str, value: Any) -> None:
        self._locals[name] = value


class _MemoryProxy:
    def __init__(self, ns: ContextNamespace) -> None:
        self._ns = ns

    def search_claims(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        self._ns._guard("memory.search_claims")
        fn = self._ns._memory.get("search_claims")
        return fn(query, limit=limit) if fn else []

    def read(self, handle: str) -> dict[str, Any]:
        self._ns._guard("memory.read")
        fn = self._ns._memory.get("read")
        return fn(handle) if fn else {}

    def write(self, *_a: object, **_k: object) -> None:
        self._ns._guard("memory.write")
        raise PolicyBlockedError("memory.write blocked")


class _RecallProxy:
    def __init__(self, ns: ContextNamespace) -> None:
        self._ns = ns

    def query(self, query: str, profile: str = "assist.light.v1", limit: int = 12) -> dict[str, Any]:
        self._ns._guard("recall.query")
        if self._ns._recall:
            return self._ns._recall(query, profile=profile, limit=limit)
        return {"hits": []}


class _RepoProxy:
    def __init__(self, ns: ContextNamespace) -> None:
        self._ns = ns

    def grep(self, pattern: str, path: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        self._ns._guard("repo.grep")
        if not self._ns.permissions.read_repo:
            return []
        return [h.model_dump(mode="json") for h in repo_tools.repo_grep(pattern, path=path, limit=limit)]

    def read(self, path: str, max_chars: int = 12000) -> dict[str, Any] | None:
        self._ns._guard("repo.read")
        if not self._ns.permissions.read_repo:
            return None
        rf = repo_tools.repo_read(path, max_chars=max_chars)
        return rf.model_dump(mode="json") if rf else None

    def write(self, *_a: object, **_k: object) -> None:
        self._ns._guard("repo.write")
        repo_tools.repo_write()


class _TracesProxy:
    def __init__(self, ns: ContextNamespace) -> None:
        self._ns = ns

    def search(
        self,
        query: str | None = None,
        corr_id: str | None = None,
        run_id: str | None = None,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        self._ns._guard("traces.search")
        if not self._ns.permissions.read_redis_traces:
            return []
        fn = self._ns._traces.get("search")
        return fn(query=query, corr_id=corr_id, run_id=run_id, limit=limit) if fn else []

    def read(self, handle: str) -> dict[str, Any]:
        self._ns._guard("traces.read")
        fn = self._ns._traces.get("read")
        return fn(handle) if fn else {}


class _LlmProxy:
    def __init__(self, ns: ContextNamespace) -> None:
        self._ns = ns

    def subcall(self, prompt: str, context: Any = None, schema: str | None = None) -> dict[str, Any]:
        self._ns._guard("llm.subcall")
        if self._ns._subcall:
            return self._ns._subcall(prompt=prompt, context=context, schema=schema)
        return {"ok": True, "result": {}, "summary": "no subcall broker"}
