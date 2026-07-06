"""Deterministic FCC operator briefs for harness motor turns."""

from __future__ import annotations

HARNESS_REPO_OPERATOR_BRIEF = """\
Orion harness motor — repo/technical turn.
Use tools from the start when the answer depends on code, config, or runtime facts.
Before Read on any file: prefer rg/Grep with a path or pattern. For large files, use Read offset/limit in chunks.
Cite concrete file paths, symbols, and commands in the draft. Do not guess repo structure from memory.
"""

HARNESS_RUNTIME_OPERATOR_BRIEF = """\
Orion harness motor — runtime/debug turn.
Verify live state with tools (logs, docker, bus traces) before diagnosing. Name exact services, channels, and commands.
"""
