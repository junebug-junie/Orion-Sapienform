"""Deterministic FCC operator briefs for harness motor turns."""

from __future__ import annotations

# Motor FCC sessions share one context window across all tool steps; full-file reads
# can blow the budget before the turn finishes (see unified-turn live runs).
HARNESS_MOTOR_MAX_READ_LINES = 200

_READ_DISCIPLINE = (
    f"Before Read on any file: prefer rg/Grep with a path or pattern. "
    f"Never Read an entire file longer than {HARNESS_MOTOR_MAX_READ_LINES} lines — "
    f"use Read offset/limit in chunks or rg for symbols and log lines."
)

HARNESS_REPO_OPERATOR_BRIEF = f"""\
Orion harness motor — repo/technical turn.
Use tools from the start when the answer depends on code, config, or runtime facts.
{_READ_DISCIPLINE}
Cite concrete file paths, symbols, and commands in the draft. Do not guess repo structure from memory.
"""

HARNESS_RUNTIME_OPERATOR_BRIEF = """\
Orion harness motor — runtime/debug turn.
Verify live state with tools (logs, docker, bus traces) before diagnosing. Name exact services, channels, and commands.
"""

HARNESS_UNIFIED_OPERATOR_BRIEF = f"""\
Orion harness motor.
Tools are available from the start. Your imperative states what this turn requires.
When the imperative calls for facts from the codebase or live runtime, use tools before
answering. Record each meaningful step. Do not guess repo structure or service state from memory.
{_READ_DISCIPLINE} For live failures, inspect logs, docker, and bus traces before diagnosing.
"""
