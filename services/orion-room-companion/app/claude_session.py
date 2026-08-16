"""Session-continuous `claude -p` invocation for one social room.

Derived from `services/orion-self-study-enrichment/app/claude_runner.py`,
which is a *one-shot* evidence-in/prose-out call. The difference that
justifies a separate module rather than a parameter on that one: a room
participant has to remember the conversation.

Claude Code gives that for free via `--session-id <uuid>` on the first turn
and `--resume <uuid>` on every turn after. Verified live 2026-08-14 across
two separate subprocess invocations: turn 2 recalled a name and a number
introduced in turn 1, and `--resume` returned the same `session_id`. So the
room does NOT re-stuff the whole transcript on every turn -- the transcript
in `RoomClaudeRequestV1` is context for the *first* turn of a session and a
recovery path if the CLI session is ever lost, not the memory mechanism.

Tools stay off (`--tools ""`). v1 Claude is a companion to think with, not
an agent: no filesystem, no repo, no network fetch.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orion.fcc.claude_spawn import claude_permission_argv, setting_sources_argv


@dataclass(frozen=True)
class ClaudeTurnResult:
    ok: bool
    text: str
    claude_session_id: str | None
    model: str | None
    cost_usd: float
    duration_ms: int
    exit_code: int
    error: str | None = None


def build_argv(
    prompt: str,
    *,
    claude_bin: str,
    model: str,
    effort: str,
    setting_sources_env_key: str,
    session_id: str,
    resume: bool,
    append_system_prompt: str | None = None,
) -> list[str]:
    argv: list[str] = [claude_bin, "-p", prompt, "--output-format", "json", "--model", model]
    # --session-id mints a session; --resume continues one. Passing both is a
    # CLI error, so this is an either/or, never a belt-and-suspenders pair.
    if resume:
        argv.extend(["--resume", session_id])
    else:
        argv.extend(["--session-id", session_id])
    argv.extend(["--tools", ""])  # companion, not agent -- see module docstring
    if effort:
        argv.extend(["--effort", effort])
    if append_system_prompt:
        argv.extend(["--append-system-prompt", append_system_prompt])
    argv.extend(setting_sources_argv(setting_sources_env_key))
    perm = claude_permission_argv(auto_approve=True)
    if perm:
        argv.extend(perm)
    return argv


def _extract_text(parsed: dict[str, Any]) -> str:
    if isinstance(parsed.get("result"), str):
        return parsed["result"]
    content = parsed.get("content")
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return ""


def _extract_model(parsed: dict[str, Any]) -> str | None:
    """Which model actually served the turn, per the CLI's own accounting.

    Not cosmetic. If `ANTHROPIC_BASE_URL` ever leaks into this subprocess
    (orion-hub's FCC lane sets it to a local gateway), the call still returns
    plausible text -- and this field plus `total_cost_usd` are how a local
    model masquerading as Claude gets caught. `modelUsage` is keyed by the
    real model id the request was billed under.

    `modelUsage` routinely holds MORE than one model: Claude Code bills small
    background work (title generation and similar) to a cheaper model on the
    same turn. Observed live on the first smoke run, where `modelUsage` led
    with `claude-haiku-4-5` even though Sonnet answered -- so taking the first
    key reported the wrong model, and a guard that names the wrong model is
    worse than none. Pick the entry that did the most output work instead.
    """
    usage = parsed.get("modelUsage")
    if isinstance(usage, dict) and usage:

        def _output_tokens(item: Any) -> int:
            if not isinstance(item, dict):
                return 0
            for key in ("outputTokens", "output_tokens"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
            return 0

        return max(usage.items(), key=lambda kv: _output_tokens(kv[1]))[0]
    model = parsed.get("model")
    return model if isinstance(model, str) else None


def run_room_turn(
    prompt: str,
    *,
    claude_bin: str,
    model: str,
    effort: str,
    setting_sources_env_key: str,
    session_id: str,
    resume: bool,
    timeout_sec: float,
    append_system_prompt: str | None = None,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> ClaudeTurnResult:
    argv = build_argv(
        prompt,
        claude_bin=claude_bin,
        model=model,
        effort=effort,
        setting_sources_env_key=setting_sources_env_key,
        session_id=session_id,
        resume=resume,
        append_system_prompt=append_system_prompt,
    )
    started = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - started) * 1000)

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
        return ClaudeTurnResult(
            ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
            duration_ms=_elapsed_ms(), exit_code=-1, error=f"timeout after {exc.timeout}s",
        )
    except FileNotFoundError as exc:
        return ClaudeTurnResult(
            ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
            duration_ms=_elapsed_ms(), exit_code=-1, error=str(exc),
        )

    duration_ms = _elapsed_ms()

    # A non-zero exit is not the only failure shape: an auth failure exits 0
    # and returns {"is_error": true, ...} with a human-readable `result`.
    # Surfacing that text matters -- a room whose token expired must say so
    # out loud rather than degrade into silence.
    try:
        parsed = json.loads(proc.stdout) if proc.stdout else {}
    except Exception as exc:
        return ClaudeTurnResult(
            ok=False, text="", claude_session_id=None, model=None, cost_usd=0.0,
            duration_ms=duration_ms, exit_code=proc.returncode,
            error=f"unparseable claude -p json output: {exc}",
        )

    returned_session = parsed.get("session_id") if isinstance(parsed.get("session_id"), str) else None
    cost = parsed.get("total_cost_usd")
    cost_usd = float(cost) if isinstance(cost, (int, float)) else 0.0
    served_model = _extract_model(parsed)

    if proc.returncode != 0 or parsed.get("is_error"):
        detail = _extract_text(parsed).strip() or (proc.stderr or "").strip()
        return ClaudeTurnResult(
            ok=False, text="", claude_session_id=returned_session, model=served_model,
            cost_usd=cost_usd, duration_ms=duration_ms, exit_code=proc.returncode,
            error=(detail or "claude -p failed with no diagnostic")[:2000],
        )

    text = _extract_text(parsed).strip()
    if not text:
        # AGENTS.md 0A: raw_len=0 is never a success. A room utterance with no
        # words is an empty shell, not a quiet participant.
        return ClaudeTurnResult(
            ok=False, text="", claude_session_id=returned_session, model=served_model,
            cost_usd=cost_usd, duration_ms=duration_ms, exit_code=proc.returncode,
            error="claude -p returned no usable text (raw_len=0)",
        )

    return ClaudeTurnResult(
        ok=True, text=text, claude_session_id=returned_session, model=served_model,
        cost_usd=cost_usd, duration_ms=duration_ms, exit_code=proc.returncode,
    )
