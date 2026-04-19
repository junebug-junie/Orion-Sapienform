from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.cortex.contracts import (
    AgentTraceStepV1,
    AgentTraceSummaryV1,
    AgentTraceToolStatV1,
)
from orion.schemas.cortex.types import StepExecutionResult

ToolFamily = Literal[
    "reasoning",
    "planning",
    "recall",
    "communication",
    "runtime",
    "external",
    "device",
    "memory",
    "orchestration",
    "unknown",
]
ActionKind = Literal[
    "inspect",
    "analyze",
    "retrieve",
    "decide",
    "delegate",
    "notify",
    "write",
    "execute",
    "summarize",
    "unknown",
]
EffectKind = Literal[
    "read_only",
    "state_change",
    "side_effect",
    "external_io",
    "unknown",
]

TOOL_FAMILY_ORDER: List[str] = [
    "planning",
    "recall",
    "reasoning",
    "communication",
    "runtime",
    "external",
    "memory",
    "orchestration",
    "device",
    "unknown",
]
ACTION_ORDER: List[str] = [
    "inspect",
    "analyze",
    "retrieve",
    "decide",
    "delegate",
    "notify",
    "write",
    "execute",
    "summarize",
    "unknown",
]
EFFECT_ORDER: List[str] = ["read_only", "state_change", "side_effect", "external_io", "unknown"]


class _ToolTaxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_family: ToolFamily
    action_kind: ActionKind
    effect_kind: EffectKind


_EXPLICIT_TOOL_TAXONOMY: Dict[str, _ToolTaxonomy] = {
    "planner_react": _ToolTaxonomy(tool_family="planning", action_kind="decide", effect_kind="read_only"),
    "recall": _ToolTaxonomy(tool_family="recall", action_kind="retrieve", effect_kind="read_only"),
    "RecallService": _ToolTaxonomy(tool_family="recall", action_kind="retrieve", effect_kind="read_only"),
    "agent_chain": _ToolTaxonomy(tool_family="orchestration", action_kind="delegate", effect_kind="read_only"),
    "AgentChainService": _ToolTaxonomy(tool_family="orchestration", action_kind="delegate", effect_kind="read_only"),
    "council_checkpoint": _ToolTaxonomy(tool_family="communication", action_kind="delegate", effect_kind="read_only"),
    "CouncilService": _ToolTaxonomy(tool_family="communication", action_kind="delegate", effect_kind="read_only"),
    "triage": _ToolTaxonomy(tool_family="reasoning", action_kind="inspect", effect_kind="read_only"),
    "analyze_text": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "evaluate": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "assess_risk": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "extract_facts": _ToolTaxonomy(tool_family="reasoning", action_kind="retrieve", effect_kind="read_only"),
    "compare_options": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "counterfactual": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "self_critique": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "pattern_detect": _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only"),
    "introspect": _ToolTaxonomy(tool_family="reasoning", action_kind="inspect", effect_kind="read_only"),
    "introspect_spark": _ToolTaxonomy(tool_family="reasoning", action_kind="inspect", effect_kind="read_only"),
    "goal_formulate": _ToolTaxonomy(tool_family="planning", action_kind="decide", effect_kind="read_only"),
    "plan_action": _ToolTaxonomy(tool_family="planning", action_kind="decide", effect_kind="read_only"),
    "auto_route": _ToolTaxonomy(tool_family="planning", action_kind="decide", effect_kind="read_only"),
    "summarize_context": _ToolTaxonomy(tool_family="communication", action_kind="summarize", effect_kind="read_only"),
    "story_weave": _ToolTaxonomy(tool_family="communication", action_kind="summarize", effect_kind="read_only"),
    "answer_direct": _ToolTaxonomy(tool_family="communication", action_kind="summarize", effect_kind="read_only"),
    "finalize_response": _ToolTaxonomy(tool_family="communication", action_kind="summarize", effect_kind="read_only"),
    "write_guide": _ToolTaxonomy(tool_family="communication", action_kind="write", effect_kind="read_only"),
    "write_tutorial": _ToolTaxonomy(tool_family="communication", action_kind="write", effect_kind="read_only"),
    "write_runbook": _ToolTaxonomy(tool_family="communication", action_kind="write", effect_kind="read_only"),
    "write_recommendation": _ToolTaxonomy(tool_family="communication", action_kind="write", effect_kind="read_only"),
    "search_web": _ToolTaxonomy(tool_family="external", action_kind="retrieve", effect_kind="external_io"),
    "skills.system.notify_chat_message.v1": _ToolTaxonomy(tool_family="communication", action_kind="notify", effect_kind="side_effect"),
}


def _coerce_text(value: Any, *, limit: int = 180) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _taxonomy_for(tool_id: str | None, *, step_name: str | None = None) -> _ToolTaxonomy:
    for candidate in (tool_id, step_name):
        if candidate and candidate in _EXPLICIT_TOOL_TAXONOMY:
            return _EXPLICIT_TOOL_TAXONOMY[candidate]
    key = str(tool_id or step_name or "").lower()
    if not key:
        return _ToolTaxonomy(tool_family="unknown", action_kind="unknown", effect_kind="unknown")
    if key.startswith("write_"):
        return _ToolTaxonomy(tool_family="communication", action_kind="write", effect_kind="read_only")
    if key.startswith("search_"):
        return _ToolTaxonomy(tool_family="external", action_kind="retrieve", effect_kind="external_io")
    if key.startswith("notify") or ".notify" in key:
        return _ToolTaxonomy(tool_family="communication", action_kind="notify", effect_kind="side_effect")
    if key.startswith("perceive_"):
        return _ToolTaxonomy(tool_family="device", action_kind="inspect", effect_kind="external_io")
    if "recall" in key or "memory" in key:
        return _ToolTaxonomy(tool_family="recall", action_kind="retrieve", effect_kind="read_only")
    if key in {"plan", "plan_action", "goal_formulate"} or key.startswith("plan"):
        return _ToolTaxonomy(tool_family="planning", action_kind="decide", effect_kind="read_only")
    if key.startswith("analyze") or key.startswith("evaluate") or key.startswith("assess") or key.startswith("triage"):
        return _ToolTaxonomy(tool_family="reasoning", action_kind="analyze", effect_kind="read_only")
    return _ToolTaxonomy(tool_family="unknown", action_kind="unknown", effect_kind="unknown")


# Exec steps sometimes carry extra keys (e.g. metadata) before the service payload; never use
# ``next(iter(step.result.keys()))`` — that mis-labels the row and skips AgentChain failure overrides.
_SERVICE_RESULT_KEYS: tuple[str, ...] = (
    "AgentChainService",
    "PlannerReactService",
    "RecallService",
    "CouncilService",
)


def _step_service_key(step: StepExecutionResult) -> str | None:
    if not isinstance(step.result, dict) or not step.result:
        return None
    sn = str(step.step_name or "").strip()
    if sn == "agent_chain" and "AgentChainService" in step.result:
        return "AgentChainService"
    if sn == "planner_react" and "PlannerReactService" in step.result:
        return "PlannerReactService"
    if sn in {"recall", "recall_step"} and "RecallService" in step.result:
        return "RecallService"
    if "council" in sn and "CouncilService" in step.result:
        return "CouncilService"
    for key in _SERVICE_RESULT_KEYS:
        if key in step.result:
            return key
    return next(iter(step.result.keys()))


def _planner_action(step: StepExecutionResult) -> Dict[str, Any] | None:
    if not isinstance(step.result, dict):
        return None
    payload = step.result.get("PlannerReactService")
    if not isinstance(payload, dict):
        return None
    trace = payload.get("trace")
    if not isinstance(trace, list) or not trace:
        return None
    last = trace[-1]
    if isinstance(last, dict) and isinstance(last.get("action"), dict):
        return last["action"]
    return None


def _planner_summary(step: StepExecutionResult) -> str:
    if not isinstance(step.result, dict):
        return "Planner evaluated the request."
    payload = step.result.get("PlannerReactService")
    if not isinstance(payload, dict):
        return "Planner evaluated the request."
    final_answer = payload.get("final_answer") if isinstance(payload.get("final_answer"), dict) else {}
    content = _coerce_text(final_answer.get("content"), limit=120)
    action = _planner_action(step) or {}
    tool_id = action.get("tool_id")
    if tool_id:
        return f"Planner selected {tool_id}."
    if content:
        return f"Planner produced a final answer draft: {content}"
    stop_reason = payload.get("stop_reason")
    if stop_reason:
        return f"Planner completed with stop reason {stop_reason}."
    return "Planner evaluated the request."


def _extract_observation_text(observation: Any) -> str:
    if isinstance(observation, dict):
        for key in ("llm_output", "text", "content", "answer"):
            if observation.get(key):
                return _coerce_text(observation.get(key))
        if isinstance(observation.get("structured"), dict) and observation["structured"]:
            return _coerce_text(observation["structured"])
    return _coerce_text(observation)


def _bound_capability_payload(agent_payload: Dict[str, Any]) -> Dict[str, Any]:
    direct_bound = agent_payload.get("bound_capability")
    if isinstance(direct_bound, dict):
        return direct_bound
    structured = agent_payload.get("structured")
    if isinstance(structured, dict) and isinstance(structured.get("bound_capability"), dict):
        payload = structured.get("bound_capability") or {}
        if isinstance(payload, dict):
            capability_decision = payload.get("capability_decision") if isinstance(payload.get("capability_decision"), dict) else {}
            if not payload.get("selected_skill") and capability_decision.get("selected_skill"):
                payload = {**payload, "selected_skill": capability_decision.get("selected_skill")}
            return payload
    return {}


def _agent_chain_failure_signals(agent_payload: Dict[str, Any]) -> bool:
    """
    True when AgentChainService delivered a bound-capability failure even if the outer
    StepExecutionResult.status is still success (RPC completed).
    """
    if not isinstance(agent_payload, dict):
        return False
    bound = _bound_capability_payload(agent_payload)
    if isinstance(bound, dict):
        if str(bound.get("status") or "").strip().lower() == "fail":
            return True
        obs = bound.get("observation")
        if isinstance(obs, dict):
            rpr = obs.get("raw_payload_ref")
            if isinstance(rpr, dict):
                if rpr.get("ok") is False:
                    return True
                if rpr.get("domain_negative") is True:
                    return True
    structured = agent_payload.get("structured")
    if isinstance(structured, dict):
        fr = str(structured.get("finalization_reason") or "")
        if fr == "bound_capability_domain_negative":
            return True
        if fr.startswith("bound_capability_") and any(
            tok in fr for tok in ("fail", "negative", "closed", "blocked", "timeout")
        ):
            return True
    rd = agent_payload.get("runtime_debug")
    if isinstance(rd, dict):
        path = str(rd.get("bound_capability_terminal_path") or "").lower()
        if path and any(
            tok in path
            for tok in (
                "_fail",
                "_negative",
                "fail_closed",
                "domain_negative",
                "timeout",
                "blocked",
                "empty_terminal",
                "no_compatible",
            )
        ):
            # Avoid treating a successful terminal path as failure.
            if "bound_direct_success" in path or path.endswith("success"):
                return False
            return True
    return False


def _agent_chain_failure_detail(agent_payload: Dict[str, Any]) -> str:
    bound = _bound_capability_payload(agent_payload)
    if isinstance(bound, dict):
        d = str(bound.get("detail") or "").strip()
        if d:
            return d
    text = str(agent_payload.get("text") or "").strip()
    if text:
        return text
    return "bound capability reported failure"


def _agent_chain_delegate_status(step: StepExecutionResult, agent_payload: Dict[str, Any]) -> str:
    if _agent_chain_failure_signals(agent_payload):
        return "fail"
    return str(step.status or "unknown")


def _agent_chain_delegate_summary(step: StepExecutionResult, agent_payload: Dict[str, Any], tool_id: str | None) -> str:
    if isinstance(agent_payload, dict) and _agent_chain_failure_signals(agent_payload):
        detail = _agent_chain_failure_detail(agent_payload)
        return f"Bound capability failed: {_coerce_text(detail, limit=220)}."
    return _step_summary(step, tool_id)


def _bound_effect_kind(bound_payload: Dict[str, Any]) -> EffectKind:
    policy = bound_payload.get("policy_metadata") if isinstance(bound_payload.get("policy_metadata"), dict) else {}
    if not policy:
        capability_decision = bound_payload.get("capability_decision") if isinstance(bound_payload.get("capability_decision"), dict) else {}
        policy = capability_decision.get("policy") if isinstance(capability_decision.get("policy"), dict) else {}
    execute_opt_in = bool(policy.get("execute_opt_in"))
    risk_class = str(policy.get("risk_class") or "").strip().lower()
    if execute_opt_in or risk_class in {"high_impact", "state_change"}:
        return "side_effect"
    if bound_payload.get("selected_skill"):
        return "external_io"
    return "read_only"


def _step_summary(step: StepExecutionResult, tool_id: str | None) -> str:
    service_key = _step_service_key(step)
    if step.step_name == "planner_react":
        return _planner_summary(step)
    if service_key == "RecallService":
        recall_payload = step.result.get("RecallService") if isinstance(step.result, dict) else {}
        if isinstance(recall_payload, dict):
            count = int(recall_payload.get("count") or 0)
            profile = recall_payload.get("profile")
            if profile:
                return f"Recall retrieved {count} memory item(s) with profile {profile}."
            return f"Recall retrieved {count} memory item(s)."
        return "Recall retrieved memory context."
    if service_key == "AgentChainService":
        agent_payload = step.result.get("AgentChainService") if isinstance(step.result, dict) else {}
        bound_payload = _bound_capability_payload(agent_payload) if isinstance(agent_payload, dict) else {}
        if isinstance(bound_payload, dict):
            bound_status = str(bound_payload.get("status") or "").strip().lower()
            bound_reason = str(bound_payload.get("reason") or "").strip()
            bound_detail = str(bound_payload.get("detail") or "").strip()
            if bound_status == "fail" or bound_reason:
                message = bound_detail or bound_reason or "bound capability failure"
                return f"Bound capability execution failed: {_coerce_text(message, limit=140)}."
            selected_skill = str(bound_payload.get("selected_skill") or "").strip()
            selected_verb = str(bound_payload.get("selected_verb") or "").strip()
            if selected_skill:
                return f"Bound capability executed {selected_verb or 'semantic_verb'} via {selected_skill}."
        nested = []
        if isinstance(agent_payload, dict):
            planner_raw = agent_payload.get("planner_raw") if isinstance(agent_payload.get("planner_raw"), dict) else {}
            nested = planner_raw.get("trace") if isinstance(planner_raw.get("trace"), list) else []
        return f"Delegated to Agent Chain with {len(nested)} nested trace step(s)."
    if service_key == "CouncilService":
        return "Delegated to Council checkpoint for review."
    if isinstance(step.result, dict):
        payload = step.result.get(service_key or "")
        text = _extract_observation_text(payload)
        if text:
            return text
    if tool_id:
        return f"Executed {tool_id}."
    return f"Completed {step.step_name}."


def _nested_agent_trace_steps(step: StepExecutionResult, *, start_index: int) -> List[AgentTraceStepV1]:
    if not isinstance(step.result, dict):
        return []
    agent_payload = step.result.get("AgentChainService")
    if not isinstance(agent_payload, dict):
        return []
    failure = _agent_chain_failure_signals(agent_payload)
    planner_raw = agent_payload.get("planner_raw") if isinstance(agent_payload.get("planner_raw"), dict) else {}
    trace = planner_raw.get("trace") if isinstance(planner_raw.get("trace"), list) else []
    nested_steps: List[AgentTraceStepV1] = []
    next_index = start_index
    for trace_item in trace:
        if not isinstance(trace_item, dict):
            continue
        action = trace_item.get("action") if isinstance(trace_item.get("action"), dict) else {}
        tool_id = action.get("tool_id")
        taxonomy = _taxonomy_for(tool_id)
        summary = _extract_observation_text(trace_item.get("observation"))
        if not summary:
            summary = f"Agent Chain selected {tool_id or 'an internal tool'}."
        if failure:
            row_status: str = "fail"
        else:
            row_status = "success" if trace_item.get("observation") is not None else "partial"
        nested_steps.append(
            AgentTraceStepV1(
                index=next_index,
                event_type="agent_delegate_tool",
                tool_id=str(tool_id) if tool_id else None,
                tool_family=taxonomy.tool_family,
                action_kind=taxonomy.action_kind,
                effect_kind=taxonomy.effect_kind,
                status=row_status,
                duration_ms=None,
                summary=summary,
                detail={
                    "source": "AgentChainService.planner_raw.trace",
                    "step_index": trace_item.get("step_index"),
                    "action": action,
                },
            )
        )
        next_index += 1
    if not nested_steps:
        bound_payload = _bound_capability_payload(agent_payload)
        selected_skill = str(bound_payload.get("selected_skill") or "").strip()
        selected_verb = str(bound_payload.get("selected_verb") or "").strip()
        if selected_skill or str(bound_payload.get("selected_verb") or "").strip():
            target = selected_skill or "no_concrete_skill"
            nested_fail = failure or str(bound_payload.get("status") or "").strip().lower() == "fail"
            nested_steps.append(
                AgentTraceStepV1(
                    index=next_index,
                    event_type="agent_delegate_tool",
                    tool_id=selected_skill or str(bound_payload.get("selected_verb") or "").strip() or "bound_capability",
                    tool_family="runtime",
                    action_kind="execute",
                    effect_kind=_bound_effect_kind(bound_payload),
                    status="fail" if nested_fail else "success",
                    duration_ms=None,
                    summary=f"Bound capability selected {selected_verb or 'semantic_verb'} -> {target}.",
                    detail={
                        "source": "AgentChainService.bound_capability",
                        "selected_verb": selected_verb,
                        "selected_skill": selected_skill,
                        "capability_decision": bound_payload.get("capability_decision"),
                    },
                )
            )
    return nested_steps


def _normalized_steps(steps: Iterable[StepExecutionResult]) -> List[AgentTraceStepV1]:
    normalized: List[AgentTraceStepV1] = []
    next_index = 0
    for step in steps:
        service_key = _step_service_key(step)
        raw_tool_id = None
        event_type = "tool_call"
        if step.step_name == "planner_react":
            event_type = "planner_decision"
            raw_tool_id = "planner_react"
        elif service_key == "RecallService":
            raw_tool_id = "recall"
        elif service_key == "AgentChainService":
            event_type = "delegate"
            raw_tool_id = "agent_chain"
        elif service_key == "CouncilService":
            event_type = "delegate"
            raw_tool_id = "council_checkpoint"
        else:
            raw_tool_id = step.verb_name or service_key or step.step_name
        taxonomy = _taxonomy_for(raw_tool_id, step_name=step.step_name)
        summary_tool_id = str(raw_tool_id) if raw_tool_id else None
        step_status = step.status or "unknown"
        step_summary = _step_summary(step, summary_tool_id)
        if service_key == "AgentChainService" and isinstance(step.result, dict):
            agent_payload = step.result.get("AgentChainService")
            if isinstance(agent_payload, dict):
                bound_payload = _bound_capability_payload(agent_payload)
                if bound_payload:
                    taxonomy = _ToolTaxonomy(
                        tool_family="orchestration",
                        action_kind="execute" if bound_payload.get("selected_skill") else "delegate",
                        effect_kind=_bound_effect_kind(bound_payload),
                    )
                step_status = _agent_chain_delegate_status(step, agent_payload)
                step_summary = _agent_chain_delegate_summary(step, agent_payload, summary_tool_id)
        normalized.append(
            AgentTraceStepV1(
                index=next_index,
                event_type=event_type,
                tool_id=str(raw_tool_id) if raw_tool_id else None,
                tool_family=taxonomy.tool_family,
                action_kind=taxonomy.action_kind,
                effect_kind=taxonomy.effect_kind,
                status=step_status,
                duration_ms=step.latency_ms,
                summary=step_summary,
                detail={
                    "step_name": step.step_name,
                    "verb_name": step.verb_name,
                    "service": service_key,
                    "node": step.node,
                    "error": step.error,
                },
            )
        )
        next_index += 1
        if service_key == "AgentChainService":
            nested = _nested_agent_trace_steps(step, start_index=next_index)
            normalized.extend(nested)
            next_index += len(nested)
    return normalized


def _counts_with_order(counter: Counter[str], order: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in order:
        out[key] = int(counter.get(key, 0))
    extras = [key for key in counter.keys() if key not in out]
    for key in sorted(extras):
        out[key] = int(counter[key])
    return out


def _tool_stats(steps: List[AgentTraceStepV1]) -> List[AgentTraceToolStatV1]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        if not step.tool_id:
            continue
        bucket = grouped.setdefault(
            step.tool_id,
            {
                "tool_family": step.tool_family,
                "action_kind": step.action_kind,
                "effect_kind": step.effect_kind,
                "count": 0,
                "duration_ms": 0,
            },
        )
        bucket["count"] += 1
        if step.duration_ms:
            bucket["duration_ms"] += int(step.duration_ms)
    stats = [
        AgentTraceToolStatV1(
            tool_id=tool_id,
            tool_family=payload["tool_family"],
            action_kind=payload["action_kind"],
            effect_kind=payload["effect_kind"],
            count=int(payload["count"]),
            duration_ms=int(payload["duration_ms"] or 0),
        )
        for tool_id, payload in grouped.items()
    ]
    stats.sort(key=lambda item: (TOOL_FAMILY_ORDER.index(item.tool_family) if item.tool_family in TOOL_FAMILY_ORDER else 999, item.tool_id))
    return stats


def _pluralize(word: str, count: int) -> str:
    return word if count == 1 else f"{word}s"


def _join_phrases(items: List[str]) -> str:
    filtered = [item for item in items if item]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    if len(filtered) == 2:
        return f"{filtered[0]} and {filtered[1]}"
    return ", ".join(filtered[:-1]) + f", and {filtered[-1]}"


def _deterministic_summary(
    *,
    tools: List[AgentTraceToolStatV1],
    action_counts: Dict[str, int],
    effect_counts: Dict[str, int],
    final_text: str | None,
    failed_step_count: int,
) -> str:
    families = []
    seen = set()
    for tool in tools:
        if tool.tool_family and tool.tool_family not in seen and tool.tool_family != "unknown":
            seen.add(tool.tool_family)
            families.append(tool.tool_family)
    action_fragments: List[str] = []
    if action_counts.get("inspect"):
        action_fragments.append("inspected prior context")
    if action_counts.get("analyze"):
        action_fragments.append("analyzed the request")
    if action_counts.get("retrieve"):
        action_fragments.append("retrieved supporting context")
    if action_counts.get("decide"):
        action_fragments.append("made planning decisions")
    if action_counts.get("delegate"):
        action_fragments.append("delegated work across runtime services")
    if action_counts.get("summarize"):
        action_fragments.append("synthesized the final response")
    if action_counts.get("write"):
        action_fragments.append("used write-oriented response tools")
    if action_counts.get("notify"):
        action_fragments.append("issued notification-style actions")
    if action_counts.get("execute"):
        action_fragments.append("executed runtime actions")
    if not action_fragments:
        action_fragments.append("processed the request")
    sentence = "Agent " + _join_phrases(action_fragments)
    if families:
        top_families = families[:3]
        sentence += f", using {_join_phrases(top_families)} {_pluralize('tool family', len(top_families))}"
    if final_text:
        sentence += ", and returned a final response"
    if effect_counts.get("state_change") or effect_counts.get("side_effect") or effect_counts.get("external_io"):
        effect_notes = []
        if effect_counts.get("external_io"):
            effect_notes.append("external I/O")
        if effect_counts.get("side_effect"):
            effect_notes.append("side effects")
        if effect_counts.get("state_change"):
            effect_notes.append("state changes")
        sentence += f" with {_join_phrases(effect_notes)}"
    else:
        sentence += " without write or side-effect actions"
    if failed_step_count > 0:
        sentence += f", and encountered {failed_step_count} failed {_pluralize('step', failed_step_count)}"
    return sentence + "."


def build_agent_trace_summary(
    *,
    correlation_id: str | None,
    message_id: str | None,
    mode: str | None,
    status: str | None,
    final_text: str | None,
    steps: List[StepExecutionResult],
    metadata: Dict[str, Any] | None = None,
) -> AgentTraceSummaryV1 | None:
    mode_value = str(mode or "")
    if mode_value != "agent":
        return None
    normalized_steps = _normalized_steps(steps or [])
    tool_stats = _tool_stats(normalized_steps)
    action_counter: Counter[str] = Counter(step.action_kind for step in normalized_steps if step.action_kind)
    effect_counter: Counter[str] = Counter(step.effect_kind for step in normalized_steps if step.effect_kind)
    duration_ms = sum(int(step.duration_ms or 0) for step in normalized_steps)
    unique_families = []
    for item in tool_stats:
        if item.tool_family not in unique_families:
            unique_families.append(item.tool_family)
    unique_families.sort(key=lambda value: TOOL_FAMILY_ORDER.index(value) if value in TOOL_FAMILY_ORDER else 999)
    action_counts = _counts_with_order(action_counter, ACTION_ORDER)
    effect_counts = _counts_with_order(effect_counter, EFFECT_ORDER)
    failed_step_count = sum(1 for step in normalized_steps if str(step.status or "").lower() not in {"success"})
    if (
        normalized_steps
        and failed_step_count == 0
        and str(status or "").lower() not in {"success", "unknown"}
    ):
        failed_step_count = 1
    summary_text = _deterministic_summary(
        tools=tool_stats,
        action_counts=action_counts,
        effect_counts=effect_counts,
        final_text=final_text,
        failed_step_count=failed_step_count,
    )
    raw_steps = [step.model_dump(mode="json") if hasattr(step, "model_dump") else step for step in (steps or [])]
    raw_metadata = metadata if isinstance(metadata, dict) else {}
    return AgentTraceSummaryV1(
        corr_id=str(correlation_id) if correlation_id else None,
        message_id=message_id,
        mode="agent",
        status=str(status or "unknown"),
        duration_ms=duration_ms,
        step_count=len(normalized_steps),
        tool_call_count=sum(tool.count for tool in tool_stats),
        unique_tool_count=len(tool_stats),
        unique_tool_families=unique_families,
        action_counts=action_counts,
        effect_counts=effect_counts,
        summary_text=summary_text,
        tools=tool_stats,
        steps=normalized_steps,
        raw={
            "source": "cortex_steps_v1",
            "step_results": raw_steps,
            "metadata": raw_metadata,
        },
    )
