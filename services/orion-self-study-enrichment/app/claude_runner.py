"""One-shot, no-tool-use `claude -p` subprocess invocation.

Deliberately NOT `orion/harness/fcc_motor.py`'s full interactive-turn
machinery (streaming stream-json, MCP config, tool permissions) -- this is a
single evidence-in/prose-out call with tools disabled entirely
(`--tools ""`), so none of that machinery is needed. Reuses
`orion/fcc/claude_spawn.py`'s `setting_sources_argv`/`claude_permission_argv`
helpers where they genuinely fit (both do -- this call still benefits from
skipping the repo's own project-level CLAUDE.md/hooks, and from
non-interactive permission auto-approval, even with tools off, since
`--dangerously-skip-permissions`/`--permission-mode bypassPermissions` also
silences any other prompt Claude Code might otherwise emit).

Model id: `claude-sonnet-5` (see settings.py) -- the real Sonnet 5 string
used elsewhere in this repo, e.g. `orion/dev_economics/pricing.py`.

Effort: `claude -p --help` on this host DOES expose a real `--effort
<level>` flag (low/medium/high/xhigh/max) for headless use -- confirmed by
running the binary directly, not guessed. This module passes it through
(default "medium" -- a cheap classification/summarization-shaped call, not
agentic/coding work, so a lower effort level is the documented fit; see
`shared/agent-design.md`'s Model Parameters table in the claude-api skill).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion.fcc.claude_spawn import claude_permission_argv, setting_sources_argv


@dataclass(frozen=True)
class ClaudeRunResult:
    ok: bool
    text: str
    raw_stdout: str
    exit_code: int
    error: str | None = None


def build_argv(
    prompt: str,
    *,
    claude_bin: str,
    model: str,
    effort: str,
    setting_sources_env_key: str,
) -> list[str]:
    argv: list[str] = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--tools",
        "",  # no tool use -- evidence-in/prose-out only
    ]
    if effort:
        argv.extend(["--effort", effort])
    argv.extend(setting_sources_argv(setting_sources_env_key))
    perm = claude_permission_argv(auto_approve=True)
    if perm:
        argv.extend(perm)
    return argv


def _extract_text(parsed: dict[str, Any]) -> str:
    # `--output-format json` (non-streaming, single result) shape: a top-level
    # "result" string field carries the assistant's final text. Fall back to
    # a couple of other plausible shapes defensively rather than raising --
    # this is dev-tooling-adjacent infra, not a live path, so degrade to an
    # empty string (caller treats that as a non-usable result) rather than
    # crashing the consumer loop over an unexpected CLI output shape.
    if isinstance(parsed.get("result"), str):
        return parsed["result"]
    content = parsed.get("content")
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return ""


def run_claude_once(
    prompt: str,
    *,
    claude_bin: str,
    model: str,
    effort: str,
    setting_sources_env_key: str,
    timeout_sec: float,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ClaudeRunResult:
    argv = build_argv(
        prompt,
        claude_bin=claude_bin,
        model=model,
        effort=effort,
        setting_sources_env_key=setting_sources_env_key,
    )
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ClaudeRunResult(ok=False, text="", raw_stdout="", exit_code=-1, error=f"timeout after {exc.timeout}s")
    except FileNotFoundError as exc:
        return ClaudeRunResult(ok=False, text="", raw_stdout="", exit_code=-1, error=str(exc))

    if proc.returncode != 0:
        return ClaudeRunResult(
            ok=False,
            text="",
            raw_stdout=proc.stdout or "",
            exit_code=proc.returncode,
            error=(proc.stderr or "").strip()[:2000],
        )

    try:
        parsed = json.loads(proc.stdout)
    except Exception as exc:
        return ClaudeRunResult(
            ok=False,
            text="",
            raw_stdout=proc.stdout or "",
            exit_code=proc.returncode,
            error=f"unparseable claude -p json output: {exc}",
        )

    text = _extract_text(parsed).strip()
    if not text:
        return ClaudeRunResult(
            ok=False,
            text="",
            raw_stdout=proc.stdout or "",
            exit_code=proc.returncode,
            error="claude -p returned no usable text (raw_len=0)",
        )
    return ClaudeRunResult(ok=True, text=text, raw_stdout=proc.stdout or "", exit_code=proc.returncode)
