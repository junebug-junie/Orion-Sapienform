"""Spawn Claude Code harness turns for Hub agent-claude mode."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from scripts.fcc_env_catalog import load_fcc_env, resolve_auth_token
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL, label_to_claude_model_id
from orion.fcc.claude_spawn import auto_approve_from_env, claude_permission_argv, extend_mcp_argv

logger = logging.getLogger("orion-hub.fcc_claude_bridge")

# asyncio subprocess stdout defaults to 64KiB; claude stream-json lines can exceed that
# when tool results embed large file reads.
DEFAULT_STREAM_READ_LIMIT = 8 * 1024 * 1024


def parse_stream_json_line(line: str) -> Optional[Dict[str, Any]]:
    stripped = str(line or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"type": "raw", "content": stripped}
    if not isinstance(parsed, dict):
        return {"type": "raw", "content": stripped}
    return parsed


def parse_stream_json_lines(lines: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        parsed = parse_stream_json_line(line)
        if parsed is not None:
            out.append(parsed)
    return out


def build_step_frame(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": str(raw.get("type") or "unknown"), "raw": raw}


def summarize_harness_step(step: Dict[str, Any], *, index: int) -> str:
    """One-line summary of a harness step for chat history / thought_process."""
    if not isinstance(step, dict):
        return f"[{index}] step"
    stype = str(step.get("type") or "event")
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return f"[{index}] {stype}"

    if stype == "assistant" or raw.get("type") == "assistant":
        text = _text_blocks_from_assistant(raw)
        if text.strip():
            return f"[{index}] assistant: {text.strip()[:2000]}"
    if stype == "result" or raw.get("type") == "result":
        result = raw.get("result")
        if isinstance(result, str) and result.strip():
            return f"[{index}] result: {result.strip()[:500]}"
    return f"[{index}] {stype}"


def summarize_harness_steps_for_history(steps: List[Dict[str, Any]]) -> str:
    """Deterministic harness transcript for reasoning_trace.content → thought_process."""
    lines = [
        summarize_harness_step(step, index=i)
        for i, step in enumerate(steps or [])
        if isinstance(step, dict)
    ]
    return "\n".join(lines).strip()


def build_harness_reasoning_trace(
    *,
    steps: List[Dict[str, Any]],
    correlation_id: str,
    session_id: Optional[str] = None,
    model_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    content = summarize_harness_steps_for_history(steps)
    if not content:
        return None
    return {
        "trace_role": "reasoning",
        "trace_stage": "mid_answer",
        "content": content,
        "correlation_id": str(correlation_id),
        "session_id": session_id,
        "model": str(model_label or "").strip() or None,
        "metadata": {
            "source": "agent_claude_harness",
            "step_count": len(steps or []),
        },
    }


def _text_blocks_from_assistant(event: Dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def extract_final_from_stream_event(
    event: Dict[str, Any],
    *,
    accumulated: str,
) -> Tuple[str, Optional[str], Optional[int]]:
    etype = str(event.get("type") or "")
    session_id = event.get("session_id")
    duration_ms = event.get("duration_ms")
    dur = int(duration_ms) if isinstance(duration_ms, (int, float)) else None
    sid = str(session_id) if session_id else None

    if etype == "result":
        result = event.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip(), sid, dur
        if isinstance(result, dict):
            text = str(result.get("result") or result.get("text") or "").strip()
            if text:
                return text, sid, dur

    assistant_text = _text_blocks_from_assistant(event)
    if assistant_text.strip():
        return assistant_text.strip(), sid, dur

    return accumulated, sid, dur


_ACTIVE: Dict[str, asyncio.subprocess.Process] = {}


def _register_process(correlation_id: str, proc: asyncio.subprocess.Process) -> None:
    _ACTIVE[str(correlation_id)] = proc


def _unregister_process(correlation_id: str) -> None:
    _ACTIVE.pop(str(correlation_id), None)


def active_turns() -> list[dict]:
    return [{"correlation_id": cid} for cid in sorted(_ACTIVE.keys())]


def _maybe_render_mcp_config(*, correlation_id: str) -> Optional[Path]:
    from scripts.fcc_env_catalog import expand_env_path, load_fcc_env
    from scripts.fcc_mcp_config import render_mcp_config
    from scripts.settings import settings

    if not settings.HUB_AGENT_CLAUDE_MCP_ENABLED:
        return None
    env = load_fcc_env(expand_env_path(settings.HUB_FCC_ENV_PATH))
    return render_mcp_config(
        correlation_id=correlation_id,
        fcc_env=env,
        include_aitown=bool(settings.HUB_AITOWN_ENABLED),
    )


def _preflight_fcc_server(url: str, *, timeout_sec: float = 3.0) -> None:
    health_url = str(url or "").rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"fcc-server health returned {resp.status}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f"fcc-server unreachable at {url}: {exc}") from exc


def _build_subprocess_env(*, fcc_server_url: str, auth_token: str) -> Dict[str, str]:
    from scripts.settings import settings

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["TERM"] = "dumb"
    max_ctx = int(getattr(settings, "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS", 65536))
    read_max = int(getattr(settings, "HUB_AGENT_CLAUDE_FILE_READ_MAX_TOKENS", 8192))
    autocompact_pct = float(
        getattr(settings, "HUB_AGENT_CLAUDE_AUTOCOMPACT_PCT_OVERRIDE", 70.0)
    )
    if max_ctx > 0:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(max_ctx)
        # Align compact threshold math with llamacpp ceiling, not model catalog default.
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(max_ctx)
    if read_max > 0:
        env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = str(read_max)
    if 0 < autocompact_pct <= 100:
        pct = int(autocompact_pct) if autocompact_pct == int(autocompact_pct) else autocompact_pct
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(pct)
    # Never inherit operator flags that disable Claude Code auto-compact.
    env.pop("DISABLE_COMPACT", None)
    env.pop("DISABLE_AUTO_COMPACT", None)
    return env


def is_context_overflow_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "exceed_context_size_error" in lowered
        or "exceeds the available context size" in lowered
    )


def context_overflow_operator_hint(*, n_ctx: int | None = None) -> str:
    ctx = int(n_ctx or 65536)
    return (
        f"\n\n---\nHub: context window full (~{ctx} tokens on llamacpp). "
        "Prefer rg/Grep before Read; use Read offset/limit on large files "
        "(orion/bus/channels.yaml is ~65KB). Raise ctx_size in config/llm_profiles.yaml "
        "or route ~/.fcc/.env MODEL_* to a backend with more headroom."
    )


async def run_turn(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
    workspace: str,
    fcc_server_url: str,
    auth_token: str,
    claude_bin: str,
    timeout_sec: float,
) -> AsyncIterator[Dict[str, object]]:
    from scripts.settings import settings

    if len(_ACTIVE) >= int(settings.HUB_AGENT_CLAUDE_MAX_CONCURRENT):
        yield {
            "type": "error",
            "error": "Another agent-claude turn is already running",
            "error_code": "fcc_claude_max_concurrent",
        }
        return

    label = str(fcc_model_label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    try:
        model_id = label_to_claude_model_id(label)
    except ValueError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_claude_bad_model_label"}
        return

    try:
        _preflight_fcc_server(fcc_server_url)
    except RuntimeError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_claude_spawn_failed"}
        return

    mcp_config_path: Optional[Path] = None
    try:
        from scripts.fcc_mcp_config import McpPreflightError

        mcp_config_path = _maybe_render_mcp_config(correlation_id=correlation_id)
    except McpPreflightError as exc:
        yield {"type": "error", "error": str(exc), "error_code": exc.error_code}
        return

    argv = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model_id,
    ]
    if mcp_config_path is not None:
        extend_mcp_argv(argv, mcp_config_path)
    perm = claude_permission_argv(
        auto_approve=auto_approve_from_env("HUB_AGENT_CLAUDE_SKIP_PERMISSIONS")
    )
    if perm:
        model_idx = argv.index("--model")
        for offset, token in enumerate(perm):
            argv.insert(model_idx + offset, token)
    started = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    exit_code = 1
    read_limit = int(getattr(settings, "HUB_AGENT_CLAUDE_STREAM_READ_LIMIT", DEFAULT_STREAM_READ_LIMIT))
    if read_limit < 65536:
        read_limit = 65536

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            env=_build_subprocess_env(fcc_server_url=fcc_server_url, auth_token=auth_token),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=read_limit,
        )
        _register_process(correlation_id, proc)
        assert proc.stdout is not None

        while True:
            try:
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                yield {
                    "type": "error",
                    "error": f"agent-claude turn timed out after {timeout_sec}s",
                    "error_code": "fcc_claude_timeout",
                }
                return
            except asyncio.LimitOverrunError as exc:
                proc.kill()
                yield {
                    "type": "error",
                    "error": (
                        "claude stream-json line exceeded read limit "
                        f"({read_limit} bytes): {exc}. "
                        "Try a narrower prompt or raise HUB_AGENT_CLAUDE_STREAM_READ_LIMIT."
                    ),
                    "error_code": "fcc_claude_stream_line_limit",
                    "llm_response": accumulated or None,
                    "metadata": {
                        "fcc_model_label": label,
                        "claude_session_id": claude_session_id,
                        "stream_read_limit": read_limit,
                    },
                }
                return

            if not line_bytes:
                break

            parsed = parse_stream_json_line(line_bytes.decode("utf-8", errors="replace"))
            if parsed is None:
                continue

            step = build_step_frame(parsed)
            yield {"type": "step", "step": step}

            text, sid, _dur = extract_final_from_stream_event(parsed, accumulated=accumulated)
            if text:
                accumulated = text
            if sid:
                claude_session_id = sid

        exit_code = await proc.wait()
    except FileNotFoundError:
        yield {
            "type": "error",
            "error": f"claude binary not found: {claude_bin!r} (check PATH or HUB_AGENT_CLAUDE_BIN)",
            "error_code": "fcc_claude_spawn_failed",
        }
        return
    finally:
        _unregister_process(correlation_id)
        if mcp_config_path is not None:
            from scripts.fcc_mcp_config import cleanup_mcp_config

            cleanup_mcp_config(mcp_config_path)

    stderr_snippet = ""
    stderr_stream = getattr(proc, "stderr", None) if proc is not None else None
    if stderr_stream is not None:
        try:
            stderr_bytes = await stderr_stream.read()
            stderr_snippet = stderr_bytes.decode("utf-8", errors="replace").strip()[:500]
        except Exception:
            stderr_snippet = ""

    duration_ms = int((time.monotonic() - started) * 1000)
    metadata = {
        "fcc_model_label": label,
        "claude_session_id": claude_session_id,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
    }

    if exit_code != 0:
        err_msg = f"claude exited with code {exit_code}"
        if stderr_snippet:
            err_msg = f"{err_msg}: {stderr_snippet}"
        yield {
            "type": "error",
            "error": err_msg,
            "error_code": "fcc_claude_nonzero_exit",
            "metadata": metadata,
            "llm_response": accumulated,
        }
        return

    yield {
        "type": "final",
        "llm_response": accumulated,
        "metadata": metadata,
    }


async def cancel_turn(correlation_id: str) -> bool:
    proc = _ACTIVE.get(str(correlation_id))
    if proc is None:
        return False
    proc.kill()
    _unregister_process(correlation_id)
    return True


async def run_turn_from_settings(
    *,
    prompt: str,
    fcc_model_label: str,
    correlation_id: str,
) -> AsyncIterator[Dict[str, object]]:
    from scripts.fcc_env_catalog import expand_env_path, load_fcc_env, resolve_auth_token
    from scripts.settings import settings

    env = load_fcc_env(expand_env_path(settings.HUB_FCC_ENV_PATH))
    token = resolve_auth_token(env, override=settings.HUB_FCC_AUTH_TOKEN)
    async for event in run_turn(
        prompt=prompt,
        fcc_model_label=fcc_model_label,
        correlation_id=correlation_id,
        workspace=settings.HUB_AGENT_CLAUDE_WORKSPACE,
        fcc_server_url=settings.HUB_FCC_SERVER_URL,
        auth_token=token,
        claude_bin=settings.HUB_AGENT_CLAUDE_BIN,
        timeout_sec=float(settings.HUB_AGENT_CLAUDE_TIMEOUT_SEC),
    ):
        yield event
