"""SmolagentsCodeEngine — REPL-based reasoning loop for context-exec."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from smolagents.models import Model, ChatMessage

from orion.core.bus.bus_schemas import LLMMessage
from orion.schemas.context_exec import ContextExecRequestV1

from .callable_namespace import ContextNamespace
from .organ_runtime import OrganRuntime
from .rlm_engine import RLMEngine
from .settings import settings

logger = logging.getLogger("orion-context-exec.smolcode_engine")


def _to_llm_messages(messages: list) -> tuple[list[LLMMessage], str]:
    """Convert smolagents messages to Orion LLMMessage list, preserving roles.

    Returns (messages, last_user_text) — last_user_text feeds raw_user_text/telemetry.
    """
    out: list[LLMMessage] = []
    last_user = ""
    for msg in messages:
        if hasattr(msg, "role"):
            role = str(msg.role.value if hasattr(msg.role, "value") else msg.role)
            raw = msg.content or ""
        elif isinstance(msg, dict):
            role = str(msg.get("role", "user"))
            raw = msg.get("content", "")
        else:
            continue
        content = raw if isinstance(raw, str) else " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        )
        role = role if role in {"system", "user", "assistant", "tool"} else "user"
        out.append(LLMMessage(role=role, content=content))
        if role == "user":
            last_user = content
    if not out:
        out = [LLMMessage(role="user", content="")]
    return out, last_user


class OrionSmolagentsModel(Model):
    """smolagents Model wrapper that calls organ_runtime.llm_chat via agent lane."""

    def __init__(
        self,
        runtime: OrganRuntime,
        loop: asyncio.AbstractEventLoop,
        *,
        per_step_timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._runtime = runtime
        self._loop = loop
        self._per_step_timeout = float(
            per_step_timeout if per_step_timeout is not None else settings.context_exec_llm_timeout_sec
        )

    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        response_format: object = None,
        tools_to_call_from: object = None,
        **kwargs: object,
    ) -> ChatMessage:
        llm_messages, last_user = _to_llm_messages(messages)
        future = asyncio.run_coroutine_threadsafe(
            self._runtime.llm_chat(
                last_user,
                route="agent",
                messages=llm_messages,
                stop=list(stop_sequences) if stop_sequences else None,
            ),
            self._loop,
        )
        result = future.result(timeout=self._per_step_timeout)
        content = result.get("content") or ""
        return ChatMessage(role="assistant", content=content)


def _make_tools(runtime: OrganRuntime, loop: asyncio.AbstractEventLoop) -> list:
    """Build the four read-only smolagents tools backed by OrganRuntime."""
    from smolagents import tool

    @tool
    def repo_grep(pattern: str) -> str:
        """Search the repository for lines matching a regex pattern.

        Args:
            pattern: Regular expression to search for in file contents.

        Returns:
            Newline-separated matches as 'path:line: content'. 'No matches found.' if empty.
        """
        hits = runtime.repo_grep(pattern, limit=30)
        if not hits:
            return "No matches found."
        return "\n".join(
            f"{h.get('path')}:{h.get('line_start')}: {h.get('snippet', '')}"
            for h in hits
        )

    @tool
    def repo_read(path: str) -> str:
        """Read the full content of a file from the repository.

        Args:
            path: Relative path from repo root, e.g. 'services/orion-context-exec/app/runner.py'.

        Returns:
            File content as a string, or an error message if not found/allowed.
        """
        result = runtime.repo_read(path)
        if result is None:
            return f"File not found or not permitted: {path}"
        content = result.get("content", "")
        if result.get("truncated"):
            content += "\n[TRUNCATED]"
        return content

    @tool
    def repo_list(path: str = "") -> str:
        """List files and directories under a repository path.

        Args:
            path: Relative path from repo root. Empty string lists the top level.

        Returns:
            Newline-separated entries. Directories end with '/'.
        """
        if not settings.context_exec_real_repo_enabled:
            return "repo reads disabled by settings"
        if not runtime.request.permissions.read_repo:
            return "repo read permission denied"
        from .repo_tools import repo_list as _repo_list
        entries = _repo_list(path)
        if not entries:
            return f"No allowed entries under: {path!r}"
        return "\n".join(entries)

    @tool
    def recall_query(query: str) -> str:
        """Search persisted user context and conversation history.

        Args:
            query: Natural language query to search recalled knowledge.

        Returns:
            Newline-separated recall snippets, or 'No recall results.' if empty.
        """
        future = asyncio.run_coroutine_threadsafe(
            runtime.recall_query(query, limit=10),
            loop,
        )
        try:
            result = future.result(timeout=30)
        except Exception as exc:
            return f"recall unavailable: {exc}"
        hits = result.get("hits") or []
        if not hits:
            return "No recall results."
        return "\n".join(
            f"- {h.get('snippet') or h.get('title') or 'hit'}" for h in hits[:10]
        )

    return [repo_grep, repo_read, repo_list, recall_query]


class SmolagentsCodeEngine(RLMEngine):
    """REPL-based reasoning engine using smolagents CodeAgent + local coder model."""

    engine_name = "smolcode"

    async def run(
        self,
        request: ContextExecRequestV1,
        namespace: ContextNamespace,
        *,
        organ_runtime: OrganRuntime | None = None,
        step_callbacks: list | None = None,
        max_steps: int | None = None,
        per_step_timeout: float | None = None,
    ) -> Any:
        if organ_runtime is None:
            return {
                "error": "organ_runtime required for smolcode engine",
                "engine": "smolcode",
                "mode": request.mode,
            }

        from smolagents import CodeAgent  # lazy import — only loaded when engine is selected

        loop = asyncio.get_running_loop()
        tools = _make_tools(organ_runtime, loop)
        model = OrionSmolagentsModel(organ_runtime, loop, per_step_timeout=per_step_timeout)
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=int(max_steps or settings.context_exec_agent_repl_max_steps),
            step_callbacks=list(step_callbacks) if step_callbacks else None,
        )

        try:
            result = await loop.run_in_executor(None, agent.run, request.text)
            return {
                "summary": str(result),
                "mode": request.mode,
                "engine": "smolcode",
            }
        except Exception as exc:
            logger.error("smolcode engine failed: %s", exc, exc_info=True)
            return {
                "error": str(exc),
                "mode": request.mode,
                "engine": "smolcode",
            }
