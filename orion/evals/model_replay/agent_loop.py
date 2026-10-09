"""A minimal Claude-Code-shaped tool loop over a llama.cpp worker's Anthropic endpoint.

Production sends `claude -p` -> llm-gateway /v1/messages -> the worker's own /v1/messages
(services/orion-llm-gateway/app/anthropic_passthrough.py: "every pool route speaks /v1/messages").
The replay speaks that same wire format straight to the worker the pool granted, with the same
tool names and argument shapes Claude Code uses, so the model sees a familiar surface. What it
does NOT reproduce: Claude Code's own system prompt and tool descriptions, WebFetch's summarizer
model, MCP servers (read_recall/read_memory/read_graph, firecrawl, github). Both models get the
identical substitute, so the comparison is fair even where fidelity to production is not exact.

Sampling is pinned per request (identical for both models): the two profiles' own defaults in
config/llm_profiles.yaml -- temperature 1.0, top_k 20, top_p 0.95, min_p 0, n_predict 16384,
reasoning_effort xhigh with preserve_thinking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "max_tokens": 16384,
    "chat_template_kwargs": {"reasoning_effort": "xhigh", "preserve_thinking": True},
}

HARNESS_SYSTEM = (
    "You are running headless inside Orion's harness (a Claude Code session started with `claude -p`). "
    "Your working directory is /repo, a read-only checkout of Orion's own repository; /scratch and /tmp are "
    "writable scratch space. Your environment carries the variables the task mentions (ORION_CURIOSITY_*, "
    "ORION_TURN_*). Use the tools when the task needs facts; when you are done, reply with your final answer "
    "as plain text (no tool call)."
)

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "Bash": {"description": "Run a bash command and return its output.",
             "input_schema": {"type": "object", "properties": {
                 "command": {"type": "string"}, "description": {"type": "string"},
                 "timeout": {"type": "number"}}, "required": ["command"]}},
    "Read": {"description": "Read a file (with line numbers). Use offset/limit for long files.",
             "input_schema": {"type": "object", "properties": {
                 "file_path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
                 "required": ["file_path"]}},
    "Grep": {"description": "Search file contents with a regular expression.",
             "input_schema": {"type": "object", "properties": {
                 "pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"},
                 "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
                 "-i": {"type": "boolean"}, "head_limit": {"type": "integer"}}, "required": ["pattern"]}},
    "Glob": {"description": "Find files by glob pattern.",
             "input_schema": {"type": "object", "properties": {
                 "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}},
    "WebFetch": {"description": "Fetch a URL and return its text content.",
                 "input_schema": {"type": "object", "properties": {
                     "url": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["url"]}},
    "WebSearch": {"description": "Search the web.",
                  "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}},
}


INFRA_ERRORS = frozenset({"LeaseUnavailable", "LeaseBacklogged", "PoolRpcTimeout", "ConnectError",
                          "ConnectTimeout"})


class MessagesClient(Protocol):
    def create(self, body: dict[str, Any], timeout_sec: float) -> dict[str, Any]: ...


class HttpMessagesClient:
    """POST {base}/v1/messages. Nothing else is ever sent to the worker."""

    def __init__(self, base_url: str, transport: Any = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        # One client per rig, so close() from another thread cuts an in-flight call (run cancelled).
        self._client = httpx.Client(headers={"anthropic-version": "2023-06-01", "x-api-key": "orion-model-replay"},
                                    transport=transport)

    def create(self, body: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
        r = self._client.post(f"{self.base_url}/v1/messages", json=body, timeout=max(5.0, timeout_sec))
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


class AttachedClient:
    """Run each call inside ``attach(timeout)`` (an async context manager on ``loop``) -- see
    pool_hold.attach_factory. The worker call itself runs in a thread; the lease heartbeat on the loop."""

    def __init__(self, inner: MessagesClient, *, loop: Any, attach: Callable[[float], Any]) -> None:
        self.inner, self.loop, self.attach = inner, loop, attach

    def create(self, body: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
        import asyncio

        async def call() -> dict[str, Any]:
            async with self.attach(timeout_sec):
                return await asyncio.to_thread(self.inner.create, body, timeout_sec)

        return asyncio.run_coroutine_threadsafe(call(), self.loop).result()


@dataclass
class Step:
    t_start: float
    elapsed_sec: float
    stop_reason: Optional[str]
    input_tokens: int
    output_tokens: int
    text_chars: int
    thinking_chars: int
    tool_calls: list[str]
    error: Optional[str] = None


@dataclass
class LoopResult:
    final_text: str = ""
    stop_reason: Optional[str] = None
    end: str = ""                   # finished | length_cut | timeout | step_cap | transport_error | cancelled
    steps: list[Step] = field(default_factory=list)
    elapsed_sec: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)


def run_loop(
    client: MessagesClient,
    *,
    user_message: str,
    tools: list[str],
    call_tool: Callable[[str, dict[str, Any]], tuple[str, bool]],
    timeout_sec: float,
    system: Optional[str] = HARNESS_SYSTEM,
    max_steps: int = 200,
    model_name: str = "replay",
    clock: Callable[[], float] = time.monotonic,
    should_stop: Callable[[], bool] = lambda: False,
) -> LoopResult:
    t0 = clock()
    deadline = t0 + timeout_sec
    res = LoopResult(messages=[{"role": "user", "content": user_message}])
    tool_defs = [{"name": n, **TOOL_SCHEMAS[n]} for n in tools]
    for _ in range(max_steps):
        if should_stop():
            res.end = "cancelled"  # the run is unwinding; its holds are being released
            break
        remaining = deadline - clock()
        if remaining <= 0:
            res.end = "timeout"
            break
        body: dict[str, Any] = {"model": model_name, "messages": res.messages, **SAMPLING}
        if system:
            body["system"] = system
        if tool_defs:
            body["tools"] = tool_defs
        s0 = clock()
        try:
            reply = client.create(body, remaining)
        except Exception as exc:  # noqa: BLE001
            took = clock() - s0
            name = type(exc).__name__
            res.steps.append(Step(s0 - t0, took, None, 0, 0, 0, 0, [], f"{name}: {exc}"[:500]))
            if should_stop():
                res.end = "cancelled"
            elif name in INFRA_ERRORS:
                res.end = "infra_error"      # the hold/attach or the route to the worker failed, not the model
            elif "timeout" in name.lower() or clock() >= deadline:
                res.end = "timeout"
            else:
                res.end = "transport_error"
            break
        blocks = reply.get("content") or []
        usage = reply.get("usage") or {}
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        thinking = sum(len(b.get("thinking") or "") for b in blocks if b.get("type") == "thinking")
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        res.steps.append(Step(s0 - t0, clock() - s0, reply.get("stop_reason"), int(usage.get("input_tokens") or 0),
                              int(usage.get("output_tokens") or 0), sum(map(len, texts)), thinking,
                              [u.get("name", "") for u in uses]))
        res.messages.append({"role": "assistant", "content": blocks})
        res.stop_reason = reply.get("stop_reason")
        if uses and res.stop_reason != "max_tokens":
            results = []
            for u in uses:
                out, is_err = call_tool(str(u.get("name")), dict(u.get("input") or {}))
                results.append({"type": "tool_result", "tool_use_id": u.get("id"), "content": out,
                                **({"is_error": True} if is_err else {})})
            res.messages.append({"role": "user", "content": results})
            continue
        res.final_text = "".join(texts).strip()
        res.end = "length_cut" if res.stop_reason == "max_tokens" else "finished"
        break
    else:
        res.end = "step_cap"
    res.elapsed_sec = clock() - t0
    return res
