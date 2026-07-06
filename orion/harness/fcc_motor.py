"""FCC subprocess motor — patterns adapted from services/orion-hub/scripts/fcc_claude_bridge.py."""

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

logger = logging.getLogger("orion.harness.fcc_motor")

DEFAULT_STREAM_READ_LIMIT = 8 * 1024 * 1024
DEFAULT_FCC_MODEL_LABEL = "MODEL_SONNET"


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


def summarize_harness_step(step: Dict[str, Any], *, index: int) -> str:
    if not isinstance(step, dict):
        return f"[{index}] step"
    stype = str(step.get("type") or "event")
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return f"[{index}] {stype}"

    if stype == "assistant" or raw.get("type") == "assistant":
        text = _text_blocks_from_assistant(raw)
        if text.strip():
            return f"[{index}] assistant: {text.strip()[:500]}"
    if stype == "result" or raw.get("type") == "result":
        result = raw.get("result")
        if isinstance(result, str) and result.strip():
            return f"[{index}] result: {result.strip()[:500]}"
    return f"[{index}] {stype}"


def _extract_tool_name(step: Dict[str, Any]) -> str | None:
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return None
    message = raw.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use" and isinstance(block.get("name"), str):
            return block["name"]
    return None


def expand_env_path(raw: str) -> Path:
    return Path(os.path.expanduser(str(raw or "").strip() or "~/.fcc/.env"))


def load_fcc_env(path: Path | str) -> Dict[str, str]:
    p = Path(path)
    if not p.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_auth_token(env: Dict[str, str], *, override: str = "") -> str:
    token = str(override or "").strip()
    if token:
        return token
    return str(env.get("ANTHROPIC_AUTH_TOKEN") or "").strip()


def label_to_claude_model_id(label: str, env: Dict[str, str]) -> str:
    key = str(label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    model_id = str(env.get(key) or env.get("MODEL") or "").strip()
    if not model_id:
        raise ValueError(f"FCC env missing model for label {key!r}")
    return model_id


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
    env.pop("DISABLE_COMPACT", None)
    env.pop("DISABLE_AUTO_COMPACT", None)
    return env


async def run_fcc_turn(
    *,
    prompt: str,
    correlation_id: str,
    fcc_model_label: str | None = None,
    workspace: str,
    fcc_server_url: str,
    auth_token: str,
    claude_bin: str,
    timeout_sec: float,
    stream_read_limit: int = DEFAULT_STREAM_READ_LIMIT,
) -> AsyncIterator[Dict[str, object]]:
    """Yield step/final/error frames from an fcc claude subprocess turn."""
    label = str(fcc_model_label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
    try:
        model_id = label_to_claude_model_id(label, env)
    except ValueError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_bad_model_label"}
        return

    try:
        _preflight_fcc_server(fcc_server_url)
    except RuntimeError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_spawn_failed"}
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
    if os.geteuid() != 0:
        argv.insert(-2, "--dangerously-skip-permissions")

    started = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    exit_code = 1
    if stream_read_limit < 65536:
        stream_read_limit = 65536

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            env=_build_subprocess_env(fcc_server_url=fcc_server_url, auth_token=auth_token),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=stream_read_limit,
        )
        assert proc.stdout is not None

        while True:
            try:
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                proc.kill()
                yield {
                    "type": "error",
                    "error": f"fcc turn timed out after {timeout_sec}s",
                    "error_code": "fcc_timeout",
                }
                return
            except asyncio.LimitOverrunError as exc:
                proc.kill()
                yield {
                    "type": "error",
                    "error": f"fcc stream line exceeded read limit: {exc}",
                    "error_code": "fcc_stream_line_limit",
                    "llm_response": accumulated or None,
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
            "error": f"claude binary not found: {claude_bin!r}",
            "error_code": "fcc_spawn_failed",
        }
        return

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
            "error_code": "fcc_nonzero_exit",
            "metadata": metadata,
            "llm_response": accumulated,
        }
        return

    yield {
        "type": "final",
        "llm_response": accumulated,
        "metadata": metadata,
    }
