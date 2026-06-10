"""Assertions for Hub golden probes — context-exec routing must be proven, not assumed."""

from __future__ import annotations

import json
from typing import Any


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def _blob(response: dict[str, Any]) -> str:
    return json.dumps(response, default=str).lower()


def _has_context_exec_signal(response: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    raw = response.get("raw") if isinstance(response.get("raw"), dict) else {}
    routing = response.get("routing_debug") if isinstance(response.get("routing_debug"), dict) else {}
    route_opts = routing.get("options") if isinstance(routing.get("options"), dict) else {}
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    auto_route = meta.get("auto_route") if isinstance(meta.get("auto_route"), dict) else {}

    hard_evidence = False

    if route_opts.get("agent_runtime_engine") == "context_exec":
        reasons.append("routing_debug.options.agent_runtime_engine=context_exec")
        hard_evidence = True
    if route_opts.get("context_exec_mode"):
        reasons.append(f"routing_debug.options.context_exec_mode={route_opts.get('context_exec_mode')}")

    for node in _walk(response):
        if not isinstance(node, dict):
            continue
        if node.get("context_exec_attempted") is True:
            reasons.append("runtime_debug.context_exec_attempted")
            hard_evidence = True
        if isinstance(node.get("context_exec"), dict):
            reasons.append("structured.context_exec")
            hard_evidence = True
        if node.get("step_name") == "context_exec" or node.get("verb_name") == "context_exec":
            reasons.append("step.context_exec")
            hard_evidence = True
        if "ContextExecService" in node:
            reasons.append("result.ContextExecService")
            hard_evidence = True
        if node.get("engine") == "context_exec":
            reasons.append("runtime_debug.engine=context_exec")
            hard_evidence = True

    if raw.get("verb") == "agent_runtime":
        reasons.append("raw.verb=agent_runtime")
    if "context_exec_investigation" in str(auto_route.get("reason") or ""):
        reasons.append(f"auto_route.reason={auto_route.get('reason')}")

    return (hard_evidence, reasons)


def assert_hub_context_exec_routing(
    response: dict[str, Any],
    *,
    probe_name: str,
    expected_mode: str | None = None,
) -> None:
    if response.get("error"):
        raise AssertionError(f"{probe_name}: hub error={response.get('error')}")

    raw = response.get("raw") if isinstance(response.get("raw"), dict) else {}
    verb = str(raw.get("verb") or "")
    final_text = str(raw.get("final_text") or response.get("text") or "")
    blob = _blob(response)

    if "llamacpp timed out" in final_text.lower():
        raise AssertionError(
            f"{probe_name}: LLM timeout — request likely stayed on chat_general, not context-exec"
        )

    if verb == "chat_general":
        raise AssertionError(
            f"{probe_name}: raw.verb=chat_general — cortex-orch did not route to agent_runtime/context-exec"
        )

    ok, reasons = _has_context_exec_signal(response)
    if not ok:
        extra = ""
        if verb == "agent_runtime":
            extra = " (orch reached agent_runtime but ContextExecService did not run — check cortex-exec flags)"
        raise AssertionError(
            f"{probe_name}: no context-exec execution evidence in Hub response{extra} "
            f"(verb={verb!r}, session_id={response.get('session_id')!r}, signals={reasons})"
        )

    routing = response.get("routing_debug") if isinstance(response.get("routing_debug"), dict) else {}
    route_opts = routing.get("options") if isinstance(routing.get("options"), dict) else {}

    if expected_mode:
        routed_mode = route_opts.get("context_exec_mode")
        if not routed_mode:
            for node in _walk(response):
                if isinstance(node, dict) and isinstance(node.get("context_exec"), dict):
                    routed_mode = node["context_exec"].get("mode")
                    if routed_mode:
                        break
        if routed_mode and str(routed_mode) != expected_mode:
            raise AssertionError(
                f"{probe_name}: expected context_exec_mode={expected_mode!r}, got {routed_mode!r}; signals={reasons}"
            )

    print(f"{probe_name} ok: {', '.join(reasons)}")
