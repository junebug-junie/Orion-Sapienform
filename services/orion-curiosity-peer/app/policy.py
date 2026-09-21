"""Re-exports the shared Cursor read-only argv policy.

Moved to `orion/curiosity/cursor_policy.py` so `orion/curiosity/supervisor.py`
(a second, grading-side Cursor caller) can use the exact same safety check
instead of forking it. This module stays so every existing `from app.policy
import ...` in this service keeps working unchanged.
"""

from __future__ import annotations

from orion.curiosity.cursor_policy import (
    FORBIDDEN_CLI_FLAGS,
    assert_read_only_cli_argv,
    build_cursor_agent_argv,
)

__all__ = [
    "FORBIDDEN_CLI_FLAGS",
    "assert_read_only_cli_argv",
    "build_cursor_agent_argv",
]
