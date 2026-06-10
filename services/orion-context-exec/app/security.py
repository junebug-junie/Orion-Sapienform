from __future__ import annotations

import logging
from typing import Any

from orion.schemas.context_exec import ContextExecPermissionV1

logger = logging.getLogger("orion-context-exec.security")

BLOCKED_CALLABLES = frozenset(
    {
        "repo.write",
        "memory.write",
        "graph.update",
        "shell",
        "network",
    }
)


class PolicyBlockedError(PermissionError):
    pass


def check_callable(permissions: ContextExecPermissionV1, callable_name: str) -> None:
    if callable_name in BLOCKED_CALLABLES:
        raise PolicyBlockedError(f"callable blocked by policy: {callable_name}")
    if callable_name.startswith("repo.") and callable_name != "repo.grep" and callable_name != "repo.read":
        if not permissions.read_repo and "write" in callable_name:
            raise PolicyBlockedError(f"repo write blocked: {callable_name}")
    if callable_name.startswith("memory.write") and not permissions.write_memory:
        raise PolicyBlockedError("memory write blocked")
    if callable_name.startswith("graph.") and "update" in callable_name and not permissions.write_graph:
        raise PolicyBlockedError("graph write blocked")
    if callable_name == "shell" and not permissions.shell_enabled:
        raise PolicyBlockedError("shell blocked")
    if callable_name == "network" and not permissions.network_enabled:
        raise PolicyBlockedError("network blocked")


def enforce_no_write_settings(write_enabled: bool, permissions: ContextExecPermissionV1) -> None:
    if write_enabled:
        return
    if any(
        [
            permissions.write_memory,
            permissions.write_graph,
            permissions.write_repo,
            permissions.mutate_runtime,
            permissions.network_enabled,
            permissions.shell_enabled,
        ]
    ):
        logger.warning("write flags requested but CONTEXT_EXEC_WRITE_ENABLED=false; callables remain read-only")
