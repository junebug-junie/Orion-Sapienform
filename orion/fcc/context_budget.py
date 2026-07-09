"""Shared FCC / harness context budget helpers (llamacpp parity)."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

ContextRisk = Literal["ok", "warn", "critical"]

DEFAULT_MAX_CONTEXT_TOKENS = 65536
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_PRESSURE_THRESHOLD_PCT = 70.0
DEFAULT_MCP_TOOL_RESULT_MAX_CHARS = 12_000

CONTEXT_PRESSURE_NUDGE = (
    "Context nearly full — answer from what you have; no more tools."
)


def _env_float(key: str, default: float) -> float:
    raw = str(os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = str(os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def max_context_tokens() -> int:
    """Motor context ceiling; prefer harness env, then hub agent-claude env."""
    for key in (
        "HARNESS_FCC_MAX_CONTEXT_TOKENS",
        "HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS",
    ):
        val = _env_int(key, 0)
        if val > 0:
            return val
    return DEFAULT_MAX_CONTEXT_TOKENS


def chars_per_token_estimate() -> int:
    return max(1, _env_int("ORION_FCC_CHARS_PER_TOKEN", DEFAULT_CHARS_PER_TOKEN))


def max_context_chars() -> int:
    return max_context_tokens() * chars_per_token_estimate()


def context_pressure_threshold_pct() -> float:
    for key in (
        "HARNESS_FCC_CONTEXT_PRESSURE_PCT",
        "HUB_AGENT_CLAUDE_CONTEXT_PRESSURE_PCT",
    ):
        val = _env_float(key, 0.0)
        if 0 < val <= 100:
            return val
    return DEFAULT_PRESSURE_THRESHOLD_PCT


def mcp_tool_result_max_chars() -> int:
    return max(
        1024,
        _env_int("ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS", DEFAULT_MCP_TOOL_RESULT_MAX_CHARS),
    )


def extend_fcc_subprocess_env(env: dict[str, str], *, workspace: str | None = None) -> None:
    """Ensure MCP stdio proxy and orion package resolve in claude subprocess."""
    roots: list[str] = []
    for candidate in (
        workspace,
        os.environ.get("HARNESS_FCC_WORKSPACE"),
        os.environ.get("HUB_AGENT_CLAUDE_WORKSPACE"),
        "/app",
        os.getcwd(),
    ):
        text = str(candidate or "").strip()
        if text and text not in roots:
            roots.append(text)
    existing = [p for p in str(env.get("PYTHONPATH") or "").split(":") if p]
    merged = list(dict.fromkeys([*roots, *existing]))
    if merged:
        env["PYTHONPATH"] = ":".join(merged)
    env.setdefault("ORION_FCC_MCP_TOOL_RESULT_MAX_CHARS", str(mcp_tool_result_max_chars()))
    env.setdefault("HARNESS_FCC_CONTEXT_PRESSURE_PCT", str(context_pressure_threshold_pct()))
    env.setdefault("HARNESS_FCC_MAX_CONTEXT_TOKENS", str(max_context_tokens()))


def context_pressure_threshold_chars() -> int:
    return int(max_context_chars() * (context_pressure_threshold_pct() / 100.0))


def tool_result_body_text(body: Any) -> str:
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


def measure_step_payload_chars(step: dict[str, Any]) -> int:
    """Rough byte budget for one stream-json harness step."""
    if not isinstance(step, dict):
        return 0
    raw = step.get("raw") if isinstance(step.get("raw"), dict) else step
    if not isinstance(raw, dict):
        return len(json.dumps(step, default=str))
    message = raw.get("message") if isinstance(raw.get("message"), dict) else raw
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                total += len(block["text"])
            elif block.get("type") == "tool_result":
                total += len(tool_result_body_text(block.get("content")))
            elif block.get("type") == "tool_use":
                total += len(json.dumps(block.get("input") or {}, default=str))
        return total
    return len(json.dumps(raw, default=str))


def context_fill_pct(*, accumulated_chars: int, max_chars: int | None = None) -> int:
    ceiling = max_chars or max_context_chars()
    if ceiling <= 0:
        return 0
    return min(100, int((accumulated_chars * 100) / ceiling))


def context_risk_level(
    *,
    accumulated_chars: int,
    step_chars: int = 0,
    max_chars: int | None = None,
) -> ContextRisk:
    ceiling = max_chars or max_context_chars()
    warn_at = context_pressure_threshold_chars()
    total = accumulated_chars + step_chars
    tool_max = mcp_tool_result_max_chars()
    if step_chars >= tool_max or total >= ceiling:
        return "critical"
    if total >= warn_at or step_chars >= int(tool_max * 0.6):
        return "warn"
    return "ok"


def is_context_overflow_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return (
        "exceed_context_size_error" in lowered
        or "exceeds the available context size" in lowered
        or "prompt is too long" in lowered
    )


def context_overflow_operator_hint(*, n_ctx: int | None = None) -> str:
    ctx = int(n_ctx or max_context_tokens())
    return (
        f"\n\n---\nHub: context window full (~{ctx} tokens on llamacpp). "
        "Prefer rg/Grep before Read; use Read offset/limit on large files "
        "(orion/bus/channels.yaml is ~65KB). For GitHub: get_pull_request when a PR "
        "number is known; list_pull_requests with perPage=1 only. Raise ctx_size in "
        "config/llm_profiles.yaml or route ~/.fcc/.env MODEL_* to a backend with more headroom."
    )


def apply_context_overflow_hint(text: str, *, n_ctx: int | None = None) -> str:
    body = str(text or "")
    if not is_context_overflow_text(body):
        return body
    hint = context_overflow_operator_hint(n_ctx=n_ctx)
    if hint.strip() in body:
        return body
    return f"{body.rstrip()}{hint}"


def annotate_harness_step(
    step: dict[str, Any],
    *,
    accumulated_chars: int,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Attach observability fields without mutating stream-json raw payload."""
    if not isinstance(step, dict):
        return step
    step_chars = measure_step_payload_chars(step)
    fill = context_fill_pct(accumulated_chars=accumulated_chars + step_chars, max_chars=max_chars)
    risk = context_risk_level(
        accumulated_chars=accumulated_chars,
        step_chars=step_chars,
        max_chars=max_chars,
    )
    out = dict(step)
    out["context_obs"] = {
        "step_chars": step_chars,
        "accumulated_chars": accumulated_chars + step_chars,
        "fill_pct": fill,
        "risk": risk,
    }
    return out


def build_context_pressure_step(*, fill_pct: int) -> dict[str, Any]:
    """Synthetic harness step surfaced to operator UI when budget is tight."""
    return {
        "type": "context_pressure",
        "raw": {
            "type": "system",
            "subtype": "context_pressure",
            "context_fill_pct": fill_pct,
            "message": CONTEXT_PRESSURE_NUDGE,
        },
    }


def summarize_context_risk_suffix(step: dict[str, Any]) -> str:
    obs = step.get("context_obs") if isinstance(step.get("context_obs"), dict) else {}
    risk = str(obs.get("risk") or "")
    step_chars = int(obs.get("step_chars") or 0)
    fill = int(obs.get("fill_pct") or 0)
    if risk == "critical":
        return f" ⚠ context critical ({step_chars} chars, {fill}% fill)"
    if risk == "warn":
        return f" ⚠ context risk ({step_chars} chars, {fill}% fill)"
    if step_chars >= 8000:
        return f" ({step_chars} chars)"
    return ""
