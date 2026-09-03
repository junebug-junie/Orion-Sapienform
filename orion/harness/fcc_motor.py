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
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

import httpx

from orion.fcc.claude_spawn import claude_permission_argv, extend_mcp_argv, setting_sources_argv
from orion.fcc.turn_lock import turn_in_progress
from orion.fcc.context_budget import (
    annotate_harness_step,
    apply_context_overflow_hint,
    build_context_pressure_step,
    context_fill_pct,
    context_pressure_threshold_chars,
    is_context_overflow_text,
    is_provider_error_envelope,
    max_context_chars,
    max_context_tokens,
    measure_step_payload_chars,
    summarize_context_risk_suffix,
)

logger = logging.getLogger("orion.harness.fcc_motor")

DEFAULT_STREAM_READ_LIMIT = 8 * 1024 * 1024
DEFAULT_FCC_MODEL_LABEL = "MODEL_SONNET"
DEFAULT_STREAM_STALL_TIMEOUT_SEC = 180.0

# Live FCC claude subprocesses keyed by correlation_id (harness cancel path).
_ACTIVE: Dict[str, asyncio.subprocess.Process] = {}
# Cancel arrived before spawn finished — kill immediately on register.
_PENDING_CANCEL: set[str] = set()


def _register_process(correlation_id: str, proc: asyncio.subprocess.Process) -> None:
    cid = str(correlation_id)
    if cid in _PENDING_CANCEL:
        _PENDING_CANCEL.discard(cid)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        logger.info("fcc_motor_cancelled_on_register corr=%s", cid)
        return
    _ACTIVE[cid] = proc


def _unregister_process(correlation_id: str) -> None:
    cid = str(correlation_id)
    _ACTIVE.pop(cid, None)
    _PENDING_CANCEL.discard(cid)


def active_fcc_turns() -> list[dict[str, str]]:
    return [{"correlation_id": cid} for cid in sorted(_ACTIVE.keys())]


def cancel_fcc_turn(correlation_id: str) -> bool:
    """SIGKILL a live FCC claude subprocess for this correlation_id, if any.

    If the process is not registered yet, arm a pending cancel so spawn registration
    kills immediately (covers Hub disconnect during preflight/spawn).
    """
    cid = str(correlation_id)
    proc = _ACTIVE.get(cid)
    if proc is None:
        _PENDING_CANCEL.add(cid)
        logger.info("fcc_motor_cancel_pending corr=%s", cid)
        return True
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    _unregister_process(cid)
    logger.info("fcc_motor_cancelled corr=%s", cid)
    return True


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


def _weights_file_basename(raw: str) -> str:
    """Reduce a served-model string to a basename with any weights-file
    extension stripped. Shared by both served-model paths (post-hoc
    discovery from a completed turn, and the pre-turn /routes probe below)
    so a raw server-side filesystem path (e.g.
    "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf") never reaches a user-facing
    field or a prompt verbatim.
    """
    basename = raw.rsplit("/", 1)[-1]
    for ext in (".gguf", ".bin", ".safetensors"):
        if basename.lower().endswith(ext):
            basename = basename[: -len(ext)]
            break
    return basename or raw


def _served_model_from_assistant(event: Dict[str, Any]) -> Optional[str]:
    """Pull the real backend model out of a stream-json "assistant" event.

    `fcc_model_label` (e.g. "MODEL_SONNET") only names the ~/.fcc/.env route
    alias the turn requested -- confirmed live 2026-08-19 that MODEL_SONNET
    and MODEL_OPUS both point at the identical `llamacpp/chat` route, so the
    label alone cannot tell two backends apart. `CLAUDE_CODE_ENABLE_GATEWAY_
    MODEL_DISCOVERY=1` (already set in `_build_subprocess_env`) is the
    documented Claude Code CLI flag for trusting a custom gateway's reported
    model over the one requested, and orion-llm-gateway's anthropic_passthrough
    is a raw byte passthrough that never rewrites the upstream response body --
    confirmed live via direct curl to /v1/messages that llama.cpp's own
    Anthropic-compat endpoint echoes the real served weights file (e.g.
    "/models/gguf/Qwen_Qwen3-8B-Q4_K_M.gguf") in the response's top-level
    "model" key regardless of what alias was requested, the same fact
    `_served_model()` in orion-llm-gateway/llm_backend.py uses for the direct
    chat_general path. This is the harness-motor equivalent: the CLI's own
    "assistant" stream-json event nests that same Messages response under
    "message", so "message.model" should carry the real value through to
    here. Returns None on any missing/malformed field so a discovery miss
    never breaks the turn -- callers fall back to `fcc_model_label`.

    Reduced to a basename with any weights-file extension stripped: the raw
    value echoed above is a full server-side filesystem path, and
    response_identity (where this ultimately lands, via
    chat_history_log.response_identity) is a user-facing "who answered"
    field in Hub chat history, not an infra debug surface -- it should never
    show an internal path like "/models/gguf/...".
    """
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    model = message.get("model")
    if not isinstance(model, str):
        return None
    model = model.strip()
    if not model:
        return None
    return _weights_file_basename(model)


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


def _extract_tool_result_errors(step: Dict[str, Any]) -> List[str]:
    """Error text for each tool_result content block where is_error is true.

    This -- not the once-per-turn fcc-subprocess `error` event branch in runner.py --
    is the real substrate for repeated-tool-failure detection: is_error is set per
    tool_result block on every step, so an in-turn tool_use -> tool_result round-trip
    (e.g. a denied permission) surfaces here even when the overall turn otherwise
    succeeds. Mirrors _extract_tool_name's shape/traversal.
    """
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return []
    message = raw.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []
    errors: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result" and block.get("is_error"):
            errors.append(_tool_result_body_text(block.get("content")))
    return errors


# Fixed allowlist for context_gathering_ratio (2026-07-24) -- a deterministic
# tool-name match, NOT a taxonomy service. Any tool name not listed here is
# uncounted (neither bucket), rather than guessed into one. MCP prefixes are
# limited to servers this codebase has confirmed are read-only-by-construction
# (gitnexus: graph queries only; firecrawl: search/scrape only) -- an unlisted MCP
# server defaults to uncounted since there's no way to verify it can't mutate state.
_CONTEXT_GATHERING_TOOLS = frozenset({"Read", "Grep", "Glob", "WebSearch", "WebFetch", "ToolSearch"})
_CONTEXT_GATHERING_MCP_PREFIXES = ("mcp__gitnexus__", "mcp__firecrawl__")
_EXECUTION_TOOLS = frozenset({"Bash", "Edit", "Write", "MultiEdit", "NotebookEdit"})


def classify_step_tool_kind(tool_name: str | None) -> str | None:
    """"context_gathering", "execution", or None (uncounted) for one tool_use call."""
    name = str(tool_name or "")
    if not name:
        return None
    if name in _CONTEXT_GATHERING_TOOLS:
        return "context_gathering"
    if name in _EXECUTION_TOOLS:
        return "execution"
    if name.startswith(_CONTEXT_GATHERING_MCP_PREFIXES):
        return "context_gathering"
    return None


def extract_result_output_tokens(step: Dict[str, Any]) -> Optional[int]:
    """Real output_tokens from the harness CLI's own end-of-turn result event.

    Confirmed live 2026-07-24: Claude Code's stream-json `result` message carries a
    top-level `usage` object with real provider-computed token counts (input_tokens,
    output_tokens, cache_read_input_tokens, ...) -- exactly one per FCC-motor
    invocation (the CLI's own cumulative-run summary), not one per step.
    """
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict) or str(raw.get("type") or "") != "result":
        return None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("output_tokens")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        return None
    return max(0, tokens)


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
    """Resolve a turn's model selector to the string `claude --model` receives.

    Two accepted shapes, checked in this order:

    1. An already-resolved ``"<backend>/<route>"`` spec (``"llamacpp/agent"``) is
       returned verbatim. This is what the FCC server itself speaks -- the
       ``MODEL*`` values in ``~/.fcc/.env`` are exactly these strings -- so a
       caller that already knows which gateway route it wants does not have to
       round-trip through an env key that would have to be hand-added to a file
       whose own header says "Managed by Free Claude Code /admin". Hub's COMPUTE
       lane arrives this way, via `orion.llm.routes.fcc_model_for_route`.
    2. Otherwise the label is an env KEY (``"MODEL_SONNET"``) and is looked up,
       falling back to ``MODEL``. Unchanged: this is the path every existing
       caller takes.

    The order matters. Shape 1 must be checked FIRST, because ``env.get()`` on a
    ``"llamacpp/agent"`` key misses and the ``or env.get("MODEL")`` fallback
    would then silently serve `harness` -- the wrong model, with no error, which
    is the exact failure this function is being asked to fix.
    """
    key = str(label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    if _route_key_from_fcc_env_value(key) is not None:
        return key
    model_id = str(env.get(key) or env.get("MODEL") or "").strip()
    if not model_id:
        raise ValueError(f"FCC env missing model for label {key!r}")
    return model_id


# Backends orion-llm-gateway's anthropic_passthrough actually routes (see
# services/orion-llm-gateway/app/anthropic_passthrough.py's
# _ANTHROPIC_COMPAT_BACKENDS) -- the only ones GET /routes can answer for.
_ROUTE_PROBE_BACKENDS = frozenset({"llamacpp", "llama-cpp"})


def _route_key_from_fcc_env_value(raw_value: str) -> Optional[Tuple[str, str]]:
    """Split a ~/.fcc/.env MODEL_* value like "llamacpp/chat" into
    (backend, route_key). None on anything that doesn't have exactly this
    "<backend>/<route>" shape (e.g. a bare model id with no "/") so callers
    fail open instead of guessing.
    """
    raw_value = str(raw_value or "").strip()
    if "/" not in raw_value:
        return None
    backend, _, route_key = raw_value.partition("/")
    backend = backend.strip().lower().replace("_", "-")
    route_key = route_key.strip()
    if not backend or not route_key:
        return None
    return backend, route_key


async def probe_route_runtime(
    fcc_model_label: str | None,
    *,
    env: Dict[str, str] | None = None,
    gateway_url: str | None = None,
    timeout_sec: float = 2.0,
) -> Tuple[Optional[str], Optional[int]]:
    """Orion capability: best-effort "what backend am I about to run on"
    read, for injecting into the harness system prompt *before* a turn
    starts.

    `_served_model_from_assistant` above only learns the truth from a
    turn's own stream-json output -- after the prompt was already sent, too
    late for self-context. This is the pre-turn equivalent: resolve
    fcc_model_label (e.g. "MODEL_SONNET") through ~/.fcc/.env to an
    orion-llm-gateway route key, then read that route's live-probed model
    off GET /routes. Not a new probe -- route_catalog.py's `_probe_model`
    already live-probes and caches this exact fact (15s TTL) for the Hub
    route picker; this just reads it.

    Fails open to None on: no label, missing/malformed env entry, a
    non-llamacpp backend (MODEL_HAIKU's nvidia_nim route isn't in this
    route table -- see _ROUTE_PROBE_BACKENDS), an unreachable gateway, a
    non-2xx response, or a route id with no cached model yet (worker down).
    A self-context probe must never block or fail a turn over a missing
    fact about itself.
    """
    label = str(fcc_model_label or "").strip()
    if not label:
        return None, None
    resolved_env = (
        env
        if env is not None
        else load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
    )
    # Same two label shapes `label_to_claude_model_id` accepts, in the same
    # order: an already-resolved "<backend>/<route>" spec is itself the answer,
    # otherwise the label is an env key to look up. Without the first branch a
    # COMPUTE-lane turn ("llamacpp/agent") misses the env lookup and this probe
    # fails open to None -- costing the harness prompt its "what model am I
    # about to run on" self-context on exactly the lane where it differs most
    # from the default.
    parsed = _route_key_from_fcc_env_value(label) or _route_key_from_fcc_env_value(
        resolved_env.get(label, "")
    )
    if parsed is None:
        return None, None
    backend, route_key = parsed
    if backend not in _ROUTE_PROBE_BACKENDS:
        return None, None

    url = str(
        gateway_url or os.environ.get("HARNESS_LLM_GATEWAY_URL", "http://llm-gateway:8210")
    ).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(f"{url}/routes")
            if response.status_code >= 400:
                return None, None
            payload = response.json()
    except Exception:
        logger.warning("probe_route_runtime failed label=%s route=%s", label, route_key, exc_info=True)
        return None, None

    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(routes, list):
        return None, None
    for route in routes:
        if not isinstance(route, dict) or route.get("id") != route_key:
            continue
        raw_ctx = route.get("n_ctx")
        # Absent on an older gateway that predates this field, and null whenever the
        # worker is down or answered an unexpected shape. Both mean "no ceiling known",
        # which callers must treat as "fall back to the configured default" -- never as
        # unlimited.
        n_ctx = raw_ctx if isinstance(raw_ctx, int) and not isinstance(raw_ctx, bool) and raw_ctx > 0 else None
        model = route.get("model")
        if isinstance(model, str) and model.strip():
            return _weights_file_basename(model.strip()), n_ctx
        return None, n_ctx
    return None, None


async def probe_current_served_model(
    fcc_model_label: str | None,
    *,
    env: Dict[str, str] | None = None,
    gateway_url: str | None = None,
    timeout_sec: float = 2.0,
) -> Optional[str]:
    """Just the served-model half of `probe_route_runtime`, for the prompt.

    Kept as its own name because that is what the harness prefix asks for and
    what every existing caller and test already says; the context window is a
    separate concern with a separate consumer (the motor's own budget), and
    neither should have to know it is sharing one HTTP call with the other.
    """
    model, _ = await probe_route_runtime(
        fcc_model_label, env=env, gateway_url=gateway_url, timeout_sec=timeout_sec
    )
    return model


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


def _stream_stall_timeout_sec(turn_timeout_sec: float) -> float:
    """Max wait for one stream-json line, separate from the whole-turn budget.

    Claude Code only writes a stream-json line once a step fully completes (no
    --include-partial-messages), so a single message that never reaches a stop
    condition blocks readline() with no output at all. Before this cap, that
    wait was `timeout_sec` itself (up to HARNESS_FCC_TIMEOUT_SEC, 900s default)
    per *line* rather than per turn, so one stuck message could hang a turn for
    the full 15 minutes. Live-reproduced 2026-07-12: a two-word greeting turn
    streamed 540KB/4333 upstream chunks with zero stream-json output before
    being killed manually at ~11 minutes.

    Deliberately not adding `--include-partial-messages` here: an earlier,
    separate session prototyped that (streamed text deltas, preserved partial
    `llm_response` on cutoff) alongside this same stall cap and a model-id
    "no-thinking" normalization, then reverted all of it with no recorded
    rationale (branch fix/fcc-motor-idle-watchdog, commits a8e8221c/f4c3d5dc,
    reverted by e995278a/7bd8be83). Checked against this incident: thinking
    was already off for this model via ENABLE_MODEL_THINKING=false in
    ~/.fcc/.env, so that piece wasn't the cause here. Partial-message
    streaming would give real visibility into a runaway generation instead of
    just killing it blind, but it's a bigger behavioral change with an
    unknown-but-presumably-real reason it was backed out; revisit only with a
    fresh understanding of why, not a blind re-apply.
    """
    raw = str(os.environ.get("HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC") or "").strip()
    configured = DEFAULT_STREAM_STALL_TIMEOUT_SEC
    if raw:
        try:
            configured = float(raw)
        except ValueError:
            logger.warning(
                "invalid HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=%r; using default %.1fs",
                raw,
                DEFAULT_STREAM_STALL_TIMEOUT_SEC,
            )
    if configured <= 0:
        return max(1.0, float(turn_timeout_sec))
    effective = max(1.0, min(float(turn_timeout_sec), configured))
    if effective != configured:
        logger.warning(
            "HARNESS_FCC_STREAM_STALL_TIMEOUT_SEC=%s clamped to %.1fs (turn budget %.1fs)",
            raw or configured,
            effective,
            turn_timeout_sec,
        )
    return effective


def _should_skip_claude_permissions() -> bool:
    """Whether claude -p should get full-auto-approve permission argv.

    The actual flag differs by caller (see claude_permission_argv() in
    orion/fcc/claude_spawn.py): --dangerously-skip-permissions on the host,
    --permission-mode bypassPermissions as root (this function only decides
    whether to auto-approve at all, not which flag). Docker harness runs as
    root; HARNESS_FCC_SKIP_PERMISSIONS=true (default in governor compose)
    avoids blocking Bash/MCP on approval prompts with no operator. When
    unset, preserve legacy host-dev behavior: skip only for non-root euid.
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
    include_context_mode = _env_truthy("HARNESS_FCC_CONTEXT_MODE_ENABLED")
    if _env_truthy("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED"):
        # Hook mode: the context-mode Claude Code plugin owns its MCP server;
        # a standalone entry would double-register the ctx_* tools.
        if include_context_mode:
            logger.warning(
                "context_mode_hooks_mode_wins corr=%s: both "
                "HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED and "
                "HARNESS_FCC_CONTEXT_MODE_ENABLED are set; hook mode wins, "
                "skipping the standalone context-mode MCP server",
                correlation_id,
            )
        include_context_mode = False
    return render_mcp_config(
        correlation_id=correlation_id,
        fcc_env=env,
        include_aitown=include_aitown,
        aitown_env=_harness_aitown_env(env) if include_aitown else None,
        include_gitnexus=_env_truthy("HARNESS_FCC_GITNEXUS_ENABLED"),
        include_context_mode=include_context_mode,
        context_mode_dir=os.environ.get("HARNESS_FCC_CONTEXT_MODE_DIR"),
        context_mode_project_dir=os.environ.get("HARNESS_FCC_WORKSPACE"),
    )


def _fcc_context_env(env: dict[str, str], *, n_ctx: Optional[int] = None) -> None:
    """Align llamacpp context ceiling + auto-compact with hub agent-claude.

    `n_ctx`, when known, is the window the route's worker is actually serving
    and overrides the container-wide env default -- see
    `orion.fcc.context_budget.max_context_tokens`. It matters most right here:
    CLAUDE_CODE_AUTO_COMPACT_WINDOW is what the claude subprocess compacts
    against, so on the 32768-token `agent` lane an unadjusted 131072 means the
    subprocess never compacts before the worker rejects the request.
    """
    max_ctx = max_context_tokens(n_ctx)
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
        n_ctx=n_ctx,
    )


def _build_subprocess_env(
    *,
    fcc_server_url: str,
    auth_token: str,
    fcc_env: Optional[Mapping[str, str]] = None,
    turn_budget_sec: Optional[float] = None,
    turn_deadline_epoch: Optional[float] = None,
    turn_step_stall_sec: Optional[float] = None,
    n_ctx: Optional[int] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = str(fcc_server_url).rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = auth_token
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env["TERM"] = "dumb"
    # cwd=workspace below is the repo checkout, so this claude -p subprocess
    # picks up .claude/settings.json's SessionStart/Stop hooks same as any
    # interactive session. Those hooks (agent board checkin/checkout) are
    # scoped to human/agent coding sessions in a worktree, not per-turn FCC
    # chat calls -- without this marker they fire on every Orion chat turn,
    # inflating step counts and writing spurious board presence rows for the
    # shared checkout. See scripts/hooks/session_start_agent_board.py and
    # session_stop_agent_board.py, which no-op when this is set.
    env["ORION_FCC_SUBPROCESS"] = "1"
    _fcc_context_env(env, n_ctx=n_ctx)
    if _env_truthy("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED"):
        # Point the context-mode plugin's hooks + MCP server at the same
        # storage root the standalone stage would use.
        env["CONTEXT_MODE_PLATFORM"] = "claude-code"
        env["CONTEXT_MODE_PROJECT_DIR"] = (
            os.environ.get("HARNESS_FCC_WORKSPACE") or os.getcwd()
        )
        env["CONTEXT_MODE_DIR"] = (
            os.environ.get("HARNESS_FCC_CONTEXT_MODE_DIR") or "/var/lib/orion/context-mode"
        )
    env.pop("DISABLE_COMPACT", None)
    env.pop("DISABLE_AUTO_COMPACT", None)
    # Orion's own read-only Postgres DSN and its FalkorDB graph credentials,
    # allowlisted out of ~/.fcc/.env. `os.environ.copy()` above is the harness
    # CONTAINER's environment and has never carried them (measured live
    # 2026-08-26: 0 matches for ORION_CURIOSITY inside the container), while
    # the file itself is already mounted and already readable from inside a
    # turn -- so this makes reaching them ergonomic, not newly possible. The
    # boundary is enforced by the role and the ACL, not here. See
    # orion/curiosity/sandbox_env.py for the full account, including why the
    # kill switch is the absence of the keys rather than a new flag.
    if fcc_env:
        from orion.curiosity.sandbox_env import inject_curiosity_credentials

        inject_curiosity_credentials(env, fcc_env)
    # The turn's own deadline, in the sandbox, sourced from the code that
    # enforces it. HARNESS_FCC_TIMEOUT_SEC lives in the harness-governor's env
    # and no other service can read it, so any budget a *prompt builder* stated
    # would be a second copy free to drift from the real wall. Stamping it here
    # means the only number Orion ever sees is the one the killer is using.
    #
    # Wall-clock (`time.time()`), not the monotonic clock the timeout loop runs
    # on, because the consumer is `date +%s` inside a Bash tool call. The two
    # can disagree if the host clock steps mid-turn; that is a rounding-level
    # risk on a ~26-minute budget and the alternative is a number the sandbox
    # cannot compare against anything.
    #
    # THE WHOLE-TURN DEADLINE IS NOT THE ONLY WALL, which is why the per-step
    # stall cap is stamped alongside it. `_stream_stall_timeout_sec` bounds a
    # SINGLE readline, and the CLI emits no stream-json line until a step
    # completes -- so one query that runs past it dies with
    # `fcc_stream_stalled` while the whole-turn clock still reads generous. A
    # prompt that showed only the outer number would be encouraging exactly the
    # unbounded single step that trips the inner one.
    #
    # Cleared rather than merely left unset when the caller does not know these:
    # `os.environ.copy()` above would otherwise pass through a value inherited
    # from an enclosing process. Clearing does NOT make the absent state
    # self-evident to a shell -- `$(( $UNSET - $(date +%s) ))` prints a
    # confident negative and exits 0 -- it only makes the wrong number absurd
    # (~-1.8e9) instead of plausible (~-3000). The prompt carries the actual
    # guard, testing for emptiness before doing the arithmetic.
    if turn_budget_sec is not None:
        env["ORION_TURN_BUDGET_SEC"] = str(int(turn_budget_sec))
    else:
        env.pop("ORION_TURN_BUDGET_SEC", None)
    if turn_deadline_epoch is not None:
        env["ORION_TURN_DEADLINE_EPOCH"] = str(int(turn_deadline_epoch))
    else:
        env.pop("ORION_TURN_DEADLINE_EPOCH", None)
    if turn_step_stall_sec is not None:
        env["ORION_TURN_STEP_STALL_SEC"] = str(int(turn_step_stall_sec))
    else:
        env.pop("ORION_TURN_STEP_STALL_SEC", None)
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
    """Orion capability: the actual FCC-Claude process.

    Spawns `claude -p` against the FCC server and yields step/final/error
    frames from its stream-json output, annotated with context pressure, with
    a per-turn ephemeral MCP config (GitHub/Firecrawl and, when flagged,
    AI Town, GitNexus, Context Mode). This is the lowest seam that still
    speaks Orion frames; below it is a subprocess.

    Runtime evidence: step frames carrying raw stream-json events, fcc_*
    error codes, and final metadata with claude_session_id and exit code.
    Start here when the motor died before producing any steps (spawn, MCP
    preflight, model label, or timeout failures).
    """
    label = str(fcc_model_label or DEFAULT_FCC_MODEL_LABEL).strip() or DEFAULT_FCC_MODEL_LABEL
    env = load_fcc_env(expand_env_path(os.environ.get("HARNESS_FCC_ENV_PATH", "~/.fcc/.env")))
    try:
        model_id = label_to_claude_model_id(label, env)
    except ValueError as exc:
        yield {"type": "error", "error": str(exc), "error_code": "fcc_bad_model_label"}
        return

    # The window the lane's worker is actually serving, so every budget below is the
    # ceiling THIS turn will really hit rather than the container's one-size default.
    # Best-effort by construction: None (older gateway, worker down, non-llamacpp
    # backend) falls the whole chain back to the env ceiling, exactly as before.
    lane_n_ctx = await probe_route_runtime(label, env=env)
    lane_n_ctx = lane_n_ctx[1]

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
    argv.extend(setting_sources_argv("HARNESS_FCC_SETTING_SOURCES"))
    if mcp_config_path is not None:
        extra_allowed_tools: Optional[List[str]] = None
        if _env_truthy("HARNESS_FCC_CONTEXT_MODE_HOOKS_ENABLED"):
            # Claude Code 2.1 pre-approval pattern is the bare server name,
            # mirroring mcp_allowed_tool_patterns; the plugin-owned server is
            # not in the rendered config, so pre-approve it explicitly.
            extra_allowed_tools = ["mcp__plugin_context-mode_context-mode"]
        extend_mcp_argv(argv, mcp_config_path, extra_allowed_tools=extra_allowed_tools)
    if _should_skip_claude_permissions():
        perm = claude_permission_argv(auto_approve=True)
        if perm:
            model_idx = argv.index("--model")
            for offset, token in enumerate(perm):
                argv.insert(model_idx + offset, token)

    started = time.monotonic()
    deadline = started + float(timeout_sec)
    # Same instant, wall clock -- for the subprocess env only. The loop below
    # still enforces against `deadline` (monotonic).
    deadline_epoch = time.time() + float(timeout_sec)
    stall_timeout_sec = _stream_stall_timeout_sec(timeout_sec)
    steps_seen = 0
    proc: Optional[asyncio.subprocess.Process] = None
    accumulated = ""
    claude_session_id: Optional[str] = None
    served_model: Optional[str] = None
    exit_code = 1
    budget_chars = len(prompt)
    ceiling_chars = max_context_chars(lane_n_ctx)
    pressure_chars = context_pressure_threshold_chars(lane_n_ctx)
    context_nudge_sent = False
    if stream_read_limit < 65536:
        stream_read_limit = 65536

    # Hold the shared sandbox lock for the whole turn. Hub refreshes the sandbox to
    # origin/main on every browser refresh (hard reset + clean), and its only prior
    # interlock was fcc_claude_bridge.active_turns() -- a dict local to the *hub*
    # process, blind to this one. Since live chat turns dispatch here via the
    # governor's bus RPC path, that dict is empty during essentially every real turn,
    # so a refresh mid-turn could reset the tree out from under this subprocess.
    # See orion/fcc/turn_lock.py.
    _turn_lock = turn_in_progress(workspace)
    _turn_lock.__enter__()
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace,
            env=_build_subprocess_env(
                n_ctx=lane_n_ctx,
                fcc_server_url=fcc_server_url,
                auth_token=auth_token,
                # Already loaded above for the model label; passed on so the
                # curiosity credentials reach the subprocess too.
                fcc_env=env,
                turn_budget_sec=float(timeout_sec),
                turn_deadline_epoch=deadline_epoch,
                turn_step_stall_sec=stall_timeout_sec,
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=stream_read_limit,
        )
        assert proc.stdout is not None
        _register_process(correlation_id, proc)

        while True:
            remaining = deadline - time.monotonic()
            read_wait = max(0.0, min(stall_timeout_sec, remaining))
            try:
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=read_wait)
            except asyncio.TimeoutError:
                proc.kill()
                stalled = stall_timeout_sec < remaining
                if stalled:
                    error_code = "fcc_stream_stalled"
                    error_msg = (
                        f"fcc stream stalled for {stall_timeout_sec:.1f}s without "
                        f"completing a step (turn_timeout={timeout_sec:.1f}s, "
                        f"steps_seen={steps_seen})"
                    )
                else:
                    error_code = "fcc_timeout"
                    error_msg = f"fcc turn timed out after {timeout_sec}s"
                yield {
                    "type": "error",
                    "error": error_msg,
                    "error_code": error_code,
                    "steps_seen": steps_seen,
                    "llm_response": accumulated or None,
                    # Carry forward any served_model already discovered from an
                    # earlier assistant event this turn -- a stall/timeout after
                    # partial progress shouldn't lose identity that was already
                    # known (see the identical carry-forward on the ceiling-
                    # exceeded and line-limit error yields below, and the
                    # authoritative one on the "final"/fcc_nonzero_exit yields).
                    "metadata": {"fcc_served_model": served_model},
                }
                return
            except asyncio.LimitOverrunError as exc:
                proc.kill()
                yield {
                    "type": "error",
                    "error": f"fcc stream line exceeded read limit: {exc}",
                    "error_code": "fcc_stream_line_limit",
                    "llm_response": accumulated or None,
                    "metadata": {"fcc_served_model": served_model},
                }
                return

            if not line_bytes:
                break

            parsed = parse_stream_json_line(line_bytes.decode("utf-8", errors="replace"))
            if parsed is None:
                continue
            steps_seen += 1

            step = build_step_frame(parsed)
            step = annotate_harness_step(step, accumulated_chars=budget_chars, max_chars=ceiling_chars)
            budget_chars += measure_step_payload_chars(step)
            yield {"type": "step", "step": step}
            if (
                not context_nudge_sent
                and budget_chars >= pressure_chars
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

            # A "result" event is the CLI's own authoritative signal that the
            # turn already completed -- never kill on it. Without this guard,
            # a legitimately long-but-successful answer can get its own
            # already-generated result payload counted against the ceiling
            # (measure_step_payload_chars has no result-specific branch, so
            # it falls back to json.dumps(raw), re-measuring text already
            # counted via the preceding assistant deltas) and lose the
            # completed answer to a false-positive runaway-draft error.
            if str(parsed.get("type") or "") != "result" and budget_chars >= ceiling_chars:
                proc.kill()
                yield {
                    "type": "error",
                    "error": (
                        f"fcc draft exceeded context ceiling ({budget_chars} >= "
                        f"{ceiling_chars} chars) without completing the turn"
                    ),
                    "error_code": "fcc_draft_length_ceiling_exceeded",
                    "steps_seen": steps_seen,
                    "llm_response": accumulated or None,
                    "metadata": {"fcc_served_model": served_model},
                }
                return

            text, sid, _dur = extract_final_from_stream_event(parsed, accumulated=accumulated)
            if text:
                accumulated = text
            if sid:
                claude_session_id = sid
            if str(parsed.get("type") or "") == "assistant":
                seen_model = _served_model_from_assistant(parsed)
                if seen_model:
                    served_model = seen_model

        exit_code = await proc.wait()
    except FileNotFoundError:
        yield {
            "type": "error",
            "error": f"claude binary not found: {claude_bin!r}",
            "error_code": "fcc_spawn_failed",
        }
        return
    finally:
        _unregister_process(correlation_id)
        _turn_lock.__exit__(None, None, None)
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
        # Real served model, when the CLI's stream-json "assistant" events
        # echoed one back (see _served_model_from_assistant). None when the
        # backend never revealed it (e.g. a fast-fail before any assistant
        # turn, or a non-discovery-aware backend) -- callers fall back to
        # fcc_model_label.
        "fcc_served_model": served_model,
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
        accumulated = apply_context_overflow_hint(accumulated, n_ctx=lane_n_ctx)

    # A zero exit code is not evidence the turn produced cognition. FCC returns an
    # upstream failure as a 200 whose assistant text IS the error (see
    # `is_provider_error_envelope`), so without this the motor's happy path hands a
    # provider error report onward as Orion's answer and every consumer -- finalize,
    # chat history, the WS final frame -- records a success. Emitting `error` here is
    # what makes CLAUDE.md's "no empty-shell cognition" hold on this path: the reply is
    # preserved on the error frame for diagnosis rather than spoken as an answer.
    if is_provider_error_envelope(accumulated):
        yield {
            "type": "error",
            "error": accumulated.strip().splitlines()[0] if accumulated.strip() else "provider error",
            "error_code": (
                "fcc_context_overflow"
                if is_context_overflow_text(accumulated)
                else "fcc_provider_error"
            ),
            "metadata": metadata,
            "llm_response": accumulated,
        }
        return

    yield {
        "type": "final",
        "llm_response": accumulated,
        "metadata": metadata,
    }
