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

from orion.fcc.claude_spawn import claude_permission_argv, extend_mcp_argv
from orion.fcc.context_budget import (
    annotate_harness_step,
    apply_context_overflow_hint,
    build_context_pressure_step,
    context_fill_pct,
    context_pressure_threshold_chars,
    is_context_overflow_text,
    max_context_chars,
    measure_step_payload_chars,
    summarize_context_risk_suffix,
)

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


def _tool_result_body_text(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        parts = [
            str(b.get("text"))
            for b in body
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _summarize_content_blocks(
    content: Any,
    *,
    text_cap: int = 500,
    tool_result_cap: int = 600,
) -> str:
    """Compact one-line summary of a claude message's content blocks.

    Covers text, tool_use, and tool_result so downstream finalize/reflect
    passes can see that (and what) tools returned — not just an empty role tag.
    """
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        if btype == "text" and isinstance(block.get("text"), str) and block["text"].strip():
            parts.append(block["text"].strip()[:text_cap])
        elif btype == "tool_use":
            name = str(block.get("name") or "tool")
            args = block.get("input")
            arg_str = ""
            if isinstance(args, dict) and args:
                bits = [f"{k}={str(v)[:60]}" for k, v in list(args.items())[:4]]
                arg_str = "(" + ", ".join(bits) + ")"
            parts.append(f"tool_use {name}{arg_str}")
        elif btype == "tool_result":
            text = _tool_result_body_text(block.get("content"))
            err = " [error]" if block.get("is_error") else ""
            size = f" ({len(text)} chars)" if text else ""
            snippet = f": {text.strip()[:tool_result_cap]}" if text.strip() else ""
            parts.append(f"tool_result{err}{size}{snippet}")
    return " | ".join(p for p in parts if p)


def summarize_harness_step(step: Dict[str, Any], *, index: int) -> str:
    if not isinstance(step, dict):
        return f"[{index}] step"
    stype = str(step.get("type") or "event")
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return f"[{index}] {stype}"
    rtype = str(raw.get("type") or stype)

    if rtype in ("assistant", "user"):
        message = raw.get("message") if isinstance(raw.get("message"), dict) else raw
        summary = _summarize_content_blocks(message.get("content"))
        if summary:
            return f"[{index}] {rtype}: {summary}" + summarize_context_risk_suffix(step)
        return f"[{index}] {rtype}" + summarize_context_risk_suffix(step)
    if rtype == "result":
        result = raw.get("result")
        if isinstance(result, str) and result.strip():
            return f"[{index}] result: {result.strip()[:500]}"
    if rtype == "system":
        subtype = raw.get("subtype") or raw.get("system_subtype")
        base = f"[{index}] system {subtype}" if subtype else f"[{index}] system"
        return base + summarize_context_risk_suffix(step)
    base = f"[{index}] {rtype}"
    return base + summarize_context_risk_suffix(step)


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


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_skip_claude_permissions() -> bool:
    """Whether to pass --dangerously-skip-permissions to claude -p.

    Docker harness runs as root; HARNESS_FCC_SKIP_PERMISSIONS=true (default in
    governor compose) avoids blocking Bash/MCP on approval prompts with no operator.
    When unset, preserve legacy host-dev behavior: skip only for non-root euid.
    """
    raw = os.environ.get("HARNESS_FCC_SKIP_PERMISSIONS", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if _env_truthy("HARNESS_FCC_SKIP_PERMISSIONS"):
        return True
    return os.geteuid() != 0


def _harness_aitown_env(fcc_env: Dict[str, str]) -> Dict[str, str]:
    """Merge harness service overrides into FCC env for AI Town MCP probes."""
    ae = dict(fcc_env)
    override = str(os.environ.get("HARNESS_AITOWN_CONVEX_URL") or "").strip()
    if override:
        ae["AITOWN_CONVEX_URL"] = override
    return ae


def _maybe_render_mcp_config(*, correlation_id: str) -> Optional[Path]:
    from orion.fcc.mcp_config import render_mcp_config

    if not _env_truthy("HARNESS_FCC_MCP_ENABLED"):
        return None
    env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
    include_aitown = _env_truthy("HARNESS_AITOWN_ENABLED")
    return render_mcp_config(
        correlation_id=correlation_id,
        fcc_env=env,
        include_aitown=include_aitown,
        aitown_env=_harness_aitown_env(env) if include_aitown else None,
    )


def _fcc_context_env(env: dict[str, str]) -> None:
    """Align llamacpp context ceiling + auto-compact with hub agent-claude."""
    max_ctx = int(os.environ.get("HARNESS_FCC_MAX_CONTEXT_TOKENS", "65536") or "65536")
    read_max = int(os.environ.get("HARNESS_FCC_FILE_READ_MAX_TOKENS", "8192") or "8192")
    autocompact_pct = float(os.environ.get("HARNESS_FCC_AUTOCOMPACT_PCT_OVERRIDE", "70") or "70")
    if max_ctx > 0:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(max_ctx)
        env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(max_ctx)
    if read_max > 0:
        env["CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS"] = str(read_max)
    if 0 < autocompact_pct <= 100:
        pct = int(autocompact_pct) if autocompact_pct == int(autocompact_pct) else autocompact_pct
        env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(pct)
    from orion.fcc.context_budget import extend_fcc_subprocess_env

    extend_fcc_subprocess_env(
        env,
        workspace=os.environ.get("HARNESS_FCC_WORKSPACE"),
    )


def _build_subprocess_env(*, fcc_server_url: str, auth_token: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["TERM"] = "dumb"
    _fcc_context_env(env)
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

    mcp_config_path: Optional[Path] = None
    try:
        from orion.fcc.mcp_config import McpPreflightError

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
    if _should_skip_claude_permissions():
        perm = claude_permission_argv(auto_approve=True)
        if perm:
            model_idx = argv.index("--model")
            for offset, token in enumerate(perm):
                argv.insert(model_idx + offset, token)

    started = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    exit_code = 1
    budget_chars = len(prompt)
    ceiling_chars = max_context_chars()
    context_nudge_sent = False
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
            step = annotate_harness_step(step, accumulated_chars=budget_chars, max_chars=ceiling_chars)
            budget_chars += measure_step_payload_chars(step)
            yield {"type": "step", "step": step}
            if (
                not context_nudge_sent
                and budget_chars >= context_pressure_threshold_chars()
            ):
                fill = context_fill_pct(accumulated_chars=budget_chars, max_chars=ceiling_chars)
                yield {
                    "type": "step",
                    "step": annotate_harness_step(
                        build_context_pressure_step(fill_pct=fill),
                        accumulated_chars=budget_chars,
                        max_chars=ceiling_chars,
                    ),
                }
                context_nudge_sent = True

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
    finally:
        if mcp_config_path is not None:
            from orion.fcc.mcp_config import cleanup_mcp_config

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
        if is_context_overflow_text(accumulated) or is_context_overflow_text(err_msg):
            accumulated = apply_context_overflow_hint(accumulated)
            err_msg = apply_context_overflow_hint(err_msg)
        yield {
            "type": "error",
            "error": err_msg,
            "error_code": "fcc_nonzero_exit",
            "metadata": metadata,
            "llm_response": accumulated,
        }
        return

    if is_context_overflow_text(accumulated):
        accumulated = apply_context_overflow_hint(accumulated)

    yield {
        "type": "final",
        "llm_response": accumulated,
        "metadata": metadata,
    }
