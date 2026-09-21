"""Read-only Cursor Agent CLI argv policy -- shared by every caller that
shells out to Cursor read-only, not just `orion-curiosity-peer`.

`--mode ask` is the read-only gate (CLI: ask = Q&A / read-only). Never pass
`--force`, `--yolo`, `--approve-mcps`, or omit mode. Plan mode is rejected —
it may propose edits; every caller of this policy stays ask-only.

Moved here from `services/orion-curiosity-peer/app/policy.py` (which now
re-exports from this module) when `orion/curiosity/supervisor.py` grew a
second Cursor caller (grading, not investigation) -- a safety-critical
argv check like this must have exactly one implementation, not a fork per
caller. `services/orion-curiosity-peer/app/` may not be imported from
`orion/` (CLAUDE.md §5 service-boundary rule: cross-service seams are
`orion/bus/`, `orion/schemas/`, shared models, documented APIs -- not
another service's `app/`), so the shared copy had to live here, not there.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Forbidden mutation / expansion flags (exact argv tokens).
FORBIDDEN_CLI_FLAGS: frozenset[str] = frozenset(
    {
        "--force",
        "-f",
        "--yolo",
        "--approve-mcps",
        "--plan",  # shorthand for --mode=plan
    }
)


def build_cursor_agent_argv(
    *,
    agent_bin: str,
    prompt: str,
    workspace: str,
    model: Optional[str] = None,
) -> list[str]:
    """Build print-mode ask argv for a sealed contractor prompt.

    Shape mirrors FCC / ``claude -p``: subprocess argv, host/desktop auth,
    no Python SDK.
    """
    argv: list[str] = [
        agent_bin,
        "-p",
        "--mode",
        "ask",
        "--output-format",
        "text",
        "--workspace",
        workspace,
        "--trust",
    ]
    if model and str(model).strip():
        argv.extend(["--model", str(model).strip()])
    argv.append(prompt)
    return argv


def _has_print_flag(argv: Sequence[str]) -> bool:
    return "-p" in argv or "--print" in argv


def _mode_is_ask(argv: Sequence[str]) -> bool:
    """True when argv selects ask mode via ``--mode ask`` or ``--mode=ask``."""
    for i, tok in enumerate(argv):
        if tok == "--mode" and i + 1 < len(argv):
            return argv[i + 1] == "ask"
        if tok.startswith("--mode="):
            return tok.split("=", 1)[1] == "ask"
    return False


def _has_flag_with_value(argv: Sequence[str], flag: str) -> bool:
    """True when ``flag`` appears followed by a non-option value, or ``flag=``."""
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return True
        if tok.startswith(f"{flag}="):
            return True
    return False


def assert_read_only_cli_argv(argv: Sequence[str]) -> None:
    """Raise ValueError unless argv is a read-only print-mode ask invocation."""
    if not argv:
        raise ValueError("cursor agent argv is empty")

    tokens = list(argv)
    for tok in tokens:
        if tok in FORBIDDEN_CLI_FLAGS:
            raise ValueError(f"forbidden cursor agent flag: {tok}")
        # Combined forms like --force=true
        bare = tok.split("=", 1)[0]
        if bare in FORBIDDEN_CLI_FLAGS:
            raise ValueError(f"forbidden cursor agent flag: {tok}")

    if not _has_print_flag(tokens):
        raise ValueError("cursor agent argv must include -p / --print")

    if not _mode_is_ask(tokens):
        raise ValueError(
            "cursor agent argv must use --mode ask "
            "(plan mode is not allowed for this peer)"
        )

    if not _has_flag_with_value(tokens, "--workspace"):
        raise ValueError("cursor agent argv must include --workspace <path>")

    if "--trust" not in tokens:
        raise ValueError("cursor agent argv must include --trust")
