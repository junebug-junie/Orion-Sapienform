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
from .workspace import ContextExecWorkspace

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
            raw = msg.get("content") or ""
        else:
            continue
        content = raw if isinstance(raw, str) else " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        )
        role = {"tool-call": "assistant", "tool-response": "tool"}.get(role, role)
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


def _repo_reads_enabled(runtime: OrganRuntime) -> bool:
    return bool(settings.context_exec_real_repo_enabled and runtime.request.permissions.read_repo)


def _make_tools(
    runtime: OrganRuntime,
    loop: asyncio.AbstractEventLoop,
    *,
    workspace_info: dict[str, Any] | None = None,
    workspace: ContextExecWorkspace | None = None,
) -> list:
    """Build smolagents tools for agent_repl: read-only repo, workspace artifacts, recall."""
    from smolagents import tool

    from . import repo_tools, workspace_tools

    @tool
    def repo_grep(pattern: str, path: str = "", limit: int = 100, literal: bool = False) -> str:
        """Search the repository for lines matching a regex or literal pattern.

        Args:
            pattern: Regular expression (or literal string when literal=True).
            path: Optional repo-relative directory to search under.
            limit: Maximum number of matches to return.
            literal: When True, treat pattern as a literal string.

        Returns:
            Newline-separated matches as 'path:line: content', or a clear status message.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        hits = runtime.repo_grep(pattern, path=path or None, limit=limit, literal=literal)
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
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        result = runtime.repo_read(path)
        if result is None:
            return f"File not found or not permitted: {path}"
        content = result.get("content", "")
        if result.get("truncated"):
            content += "\n[TRUNCATED]"
        return content

    @tool
    def repo_read_range(path: str, start_line: int, end_line: int) -> str:
        """Read a numbered line range from a repository file.

        Args:
            path: Repo-relative file path.
            start_line: First line number (1-based, inclusive).
            end_line: Last line number (inclusive).

        Returns:
            Line-numbered content or a clear blocked/absent/range error message.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        return repo_tools.repo_read_range(path, start_line, end_line)

    @tool
    def repo_find_files(pattern: str, path: str = "", limit: int = 200) -> str:
        """Find repository files matching a glob-style pattern.

        Args:
            pattern: Glob pattern matched against file names or repo-relative paths.
            path: Optional repo-relative directory to search under.
            limit: Maximum number of paths to return.

        Returns:
            Newline-separated repo-relative paths or a clear status message.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        return repo_tools.repo_find_files(pattern, path=path, limit=limit)

    @tool
    def repo_tree(path: str = "", depth: int = 2, limit: int = 300) -> str:
        """Show a bounded directory tree under a repository path.

        Args:
            path: Repo-relative directory. Empty string means repo root.
            depth: Maximum directory depth to display.
            limit: Maximum number of tree lines to return.

        Returns:
            Deterministic tree text or a clear blocked/absent error message.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        return repo_tools.repo_tree(path=path, depth=depth, limit=limit)

    @tool
    def repo_outline(path: str) -> str:
        """Return a Python AST outline (imports, classes, functions) with line numbers.

        Args:
            path: Repo-relative Python file path.

        Returns:
            Outline text, or a clear unsupported/blocked/absent message.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        return repo_tools.repo_outline(path)

    @tool
    def repo_list(path: str = "") -> str:
        """List files and directories under a repository path.

        Args:
            path: Relative path from repo root. Empty string lists the top level.

        Returns:
            Newline-separated entries. Directories end with '/'.
        """
        if not _repo_reads_enabled(runtime):
            if not settings.context_exec_real_repo_enabled:
                return "repo reads disabled by settings"
            return "repo read permission denied"
        entries = repo_tools.repo_list(path)
        if not entries:
            return f"No allowed entries under: {path!r}"
        return "\n".join(entries)

    @tool
    def patch_validate(unified_diff: str) -> str:
        """Validate unified diff syntax and repo path policy without applying changes.

        Args:
            unified_diff: Unified diff text to validate.

        Returns:
            Concise valid/invalid summary with files touched and reason if invalid.
        """
        return repo_tools.patch_validate(unified_diff)

    @tool
    def workspace_write(path: str, content: str) -> str:
        """Write content to a path inside the current run workspace (not canonical repo).

        Args:
            path: Workspace-relative path (scratch/, outputs/, patches/, repo/).
            content: Text content to write.

        Returns:
            Confirmation with workspace-relative path or unavailable/blocked message.
        """
        return workspace_tools.workspace_write(workspace_info, workspace, path, content)

    @tool
    def workspace_read(path: str) -> str:
        """Read a file from the current run workspace.

        Args:
            path: Workspace-relative path.

        Returns:
            File content or unavailable/blocked/absent message.
        """
        return workspace_tools.workspace_read(workspace_info, workspace, path)

    @tool
    def workspace_list(path: str = "") -> str:
        """List entries under a path in the current run workspace.

        Args:
            path: Workspace-relative directory. Empty string lists workspace root.

        Returns:
            Newline-separated entries or unavailable/blocked message.
        """
        return workspace_tools.workspace_list(workspace_info, workspace, path)

    @tool
    def workspace_write_patch(name: str, unified_diff: str) -> str:
        """Write a unified diff under workspace patches/ (not canonical repo).

        Args:
            name: Patch filename (`.patch` appended if missing).
            unified_diff: Unified diff text.

        Returns:
            Confirmation with patches/ path or unavailable message.
        """
        return workspace_tools.workspace_write_patch(
            workspace_info, workspace, name, unified_diff
        )

    @tool
    def workspace_write_report(name: str, markdown: str) -> str:
        """Write a markdown report under workspace outputs/.

        Args:
            name: Report filename (`.md` appended if missing).
            markdown: Markdown content.

        Returns:
            Confirmation with outputs/ path or unavailable message.
        """
        return workspace_tools.workspace_write_report(
            workspace_info, workspace, name, markdown
        )

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

    return [
        repo_grep,
        repo_read,
        repo_read_range,
        repo_find_files,
        repo_tree,
        repo_outline,
        repo_list,
        patch_validate,
        workspace_write,
        workspace_read,
        workspace_list,
        workspace_write_patch,
        workspace_write_report,
        recall_query,
    ]


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
        workspace_info: dict[str, Any] | None = None,
        workspace: ContextExecWorkspace | None = None,
    ) -> Any:
        if organ_runtime is None:
            return {
                "error": "organ_runtime required for smolcode engine",
                "engine": "smolcode",
                "mode": request.mode,
            }

        from smolagents import CodeAgent  # lazy import — only loaded when engine is selected

        loop = asyncio.get_running_loop()
        tools = _make_tools(
            organ_runtime,
            loop,
            workspace_info=workspace_info,
            workspace=workspace,
        )
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
