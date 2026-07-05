"""Spawn Claude Code harness turns for Hub agent-claude mode."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from scripts.fcc_env_catalog import load_fcc_env, resolve_auth_token
from scripts.fcc_model_mapping import DEFAULT_FCC_MODEL_LABEL, label_to_claude_model_id

logger = logging.getLogger("orion-hub.fcc_claude_bridge")


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


def _preflight_fcc_server(url: str, *, timeout_sec: float = 3.0) -> None:
    health_url = str(url or "").rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"fcc-server health returned {resp.status}")
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise RuntimeError(f"fcc-server unreachable at {url}: {exc}") from exc


def _build_subprocess_env(*, fcc_server_url: str, auth_token: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["TERM"] = "dumb"
    return env


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

    argv = [
        claude_bin,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--verbose",
        "--model",
        model_id,
    ]
    started = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    exit_code = 1

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            env=_build_subprocess_env(fcc_server_url=fcc_server_url, auth_token=auth_token),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
