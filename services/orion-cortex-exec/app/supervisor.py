from __future__ import annotations

import logging
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orion

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import LLMMessage, ServiceRef
from orion.cognition.packs_loader import PackManager
from orion.cognition.planner.loader import VerbRegistry
from orion.schemas.agents.schemas import (
    ContextBlock,
    DeliberationRequest,
    FinalAnswer,
    Goal,
    Limits,
    PlannerRequest,
    Preferences,
    ToolDef,
    TraceStep,
    AgentChainRequest,
)
from orion.schemas.cortex.schemas import ExecutionPlan, ExecutionStep, PlanExecutionResult, StepExecutionResult
from orion.cognition.verb_activation import is_active
from orion.cognition.delivery_grounding import build_delivery_grounding_context
from orion.cognition.quality_evaluator import should_rewrite_for_instructional
from orion.schemas.agents.bound_capability import (
    BoundCapabilityExecutionRequestV1,
    CapabilityRecoveryDecisionV1,
    CapabilityRecoveryReasonV1,
)

from .clients import AgentChainClient, CouncilClient, LLMGatewayClient, PlannerReactClient
from .executor import _last_user_message, call_step_services, run_recall_step
from .recall_utils import delivery_safe_recall_decision, has_inline_recall
from .settings import settings

logger = logging.getLogger("orion.cortex.exec.supervisor")

ORION_PKG_DIR = Path(orion.__file__).resolve().parent
PROMPTS_DIR = ORION_PKG_DIR / "cognition" / "prompts"
VERBS_DIR = ORION_PKG_DIR / "cognition" / "verbs"


def _load_prompt_content(template_ref: Optional[str]) -> Optional[str]:
    if not template_ref:
        return None
    ref = template_ref.strip()
    if not ref.endswith(".j2"):
        return ref
    path = PROMPTS_DIR / ref
    if path.exists():
        return path.read_text(encoding="utf-8")
    logger.warning("Prompt template not found: %s", path)
    return ref


def _verb_to_step(cfg) -> ExecutionStep:
    prompt_text = _load_prompt_content(cfg.prompt_template)
    if _is_capability_backed_cfg(cfg):
        raise RuntimeError(
            f"capability_backed_llm_fallback_blocked:{cfg.name}"
        )
    services = cfg.services or ["LLMGatewayService"]
    return ExecutionStep(
        verb_name=cfg.name,
        step_name=cfg.name,
        description=cfg.description or "",
        order=0,
        services=services,
        prompt_template=prompt_text,
        requires_gpu=bool(cfg.requires_gpu),
        requires_memory=bool(cfg.requires_memory),
        timeout_ms=int(cfg.timeout_ms or settings.step_timeout_ms),
    )

def _is_capability_backed_cfg(cfg: Any) -> bool:
    execution_mode = str(getattr(cfg, "execution_mode", "") or "").strip().lower()
    return execution_mode == "capability_backed" or bool(getattr(cfg, "requires_capability_selector", False))


def _extract_observation(step_res: StepExecutionResult) -> Dict[str, Any]:
    for payload in (step_res.result or {}).values():
        if not isinstance(payload, dict):
            continue
        if payload.get("llm_output"):
            return {"llm_output": payload.get("llm_output"), "raw": payload}
        if payload.get("content"):
            return {"llm_output": payload.get("content"), "raw": payload}
        if payload.get("text"):
            return {"llm_output": payload.get("text"), "raw": payload}
    return {"raw": step_res.result}


def _extract_latest_planner_thought(planner_result: Any) -> Optional[str]:
    if not isinstance(planner_result, dict):
        return None
    planner_payload = planner_result.get("PlannerReactService", {})
    if not isinstance(planner_payload, dict):
        return None
    trace = planner_payload.get("trace")
    if not isinstance(trace, list) or not trace:
        return None
    last = trace[-1]
    if not isinstance(last, dict):
        return None
    thought = last.get("thought")
    return str(thought) if thought is not None else None


def _truncate_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _compact_json(value: Any, limit: int = 220) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(value)
    return _truncate_text(rendered, limit=limit)


def _detect_scaffolding_markers(text: Any) -> List[str]:
    lowered = str(text or "").lower()
    markers: List[str] = []
    if any(k in lowered for k in ("you are ", "system prompt", "developer instruction")):
        markers.append("system_directive")
    if any(k in lowered for k in ("return your normal council output contract", "toolset summary", "recent trace:")):
        markers.append("tool_scaffold")
    if any(k in lowered for k in ("delegate_tool_execution", "plan_only", "non-goals", "important")):
        markers.append("meta_instruction")
    return markers


def _sanitize_text_block(text: Any) -> str:
    lines = []
    for line in str(text or "").splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("you are "):
            continue
        if "return your normal council output contract" in lowered:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_key_terms(text: str) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "have", "what", "your", "about", "into",
        "when", "where", "which", "there", "would", "could", "should", "please", "help",
    }
    terms: List[str] = []
    for raw in str(text or "").lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if len(token) < 4 or token in stop:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:12]


_OPERATIONAL_INTENT_TERMS = {
    "run", "execute", "cleanup", "clean", "prune", "restart", "remove", "delete", "stop",
    "start", "deploy", "rollback", "docker", "container", "kubernetes", "cluster", "runtime",
}
_SHELL_GUIDANCE_PATTERN = re.compile(r"(^|\n)\s*(```\w*\n)?\s*(docker|kubectl|systemctl|rm\s+-rf|sudo|killall|service)\b", re.IGNORECASE)


def _detect_operational_intent(user_text: str) -> bool:
    lowered = str(user_text or "").lower()
    return any(term in lowered for term in _OPERATIONAL_INTENT_TERMS)


def _available_operational_semantic_tools(toolset: List[ToolDef]) -> List[ToolDef]:
    operational: List[ToolDef] = []
    for tool in toolset:
        if str(tool.tool_id or "") in {"agent_chain", "tool_chain", "finalize_response", "answer_direct"}:
            continue
        mode = str(getattr(tool, "execution_mode", "") or "").lower()
        side_effect = str(getattr(tool, "side_effect_level", "") or "").lower()
        if mode == "capability_backed" or side_effect in {"low", "moderate", "high"}:
            operational.append(tool)
    return operational



_RUNTIME_CLEANUP_TERMS = {"docker", "container", "cleanup", "stopped", "prune", "runtime", "housekeep", "housekeeping", "dry", "run"}
_CHANGE_SUMMARY_TERMS = {"summarize", "summary", "recent", "changes", "change", "changelog", "intel"}
_UNSAFE_ASSISTANT_COMMAND_PATTERN = re.compile(r"(docker\s+(rm|rmi|system\s+prune|container\s+prune|volume\s+rm|network\s+rm)|rm\s+-rf|kubectl\s+delete)", re.IGNORECASE)
_OPERATIONAL_VERB_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("assess_mesh_presence", ("node", "nodes", "mesh", "up", "online", "alive", "status")),
    ("assess_storage_health", ("disk", "storage", "health", "filesystem", "volume")),
    ("summarize_recent_changes", ("summarize", "summary", "recent", "changes", "prs", "pull request", "commits")),
    ("housekeep_runtime", ("cleanup", "clean up", "stopped", "container", "containers", "runtime", "prune", "dry-run", "dry run")),
]


def _tool_semantic_haystack(tool: ToolDef) -> str:
    families = " ".join(str(item) for item in (getattr(tool, "preferred_skill_families", []) or []))
    return " ".join(
        [
            str(tool.tool_id or ""),
            str(tool.description or ""),
            families,
            str(getattr(tool, "execution_mode", "") or ""),
            str(getattr(tool, "side_effect_level", "") or ""),
        ]
    ).lower()


def _semantic_override_scores(*, user_text: str, normalized_action_input: Dict[str, Any] | None, tools: List[ToolDef]) -> List[Dict[str, Any]]:
    action_blob = json.dumps(normalized_action_input or {}, ensure_ascii=False, default=str).lower()
    ask = f"{str(user_text or '').lower()} {action_blob}".strip()
    ask_has_runtime_cleanup = all(term in ask for term in ("cleanup", "container")) or ("docker" in ask and ("prune" in ask or "stopped" in ask))
    ask_terms = set(_extract_key_terms(ask))
    scored: List[Dict[str, Any]] = []
    for tool in tools:
        hay = _tool_semantic_haystack(tool)
        score = 0
        reasons: List[str] = []

        lexical_hits = sorted(term for term in ask_terms if term and term in hay)
        if lexical_hits:
            score += len(lexical_hits) * 3
            reasons.append(f"lexical_overlap:{','.join(lexical_hits[:6])}")

        runtime_hits = sorted(term for term in _RUNTIME_CLEANUP_TERMS if term in ask and term in hay)
        if runtime_hits:
            score += len(runtime_hits) * 5
            reasons.append(f"runtime_hits:{','.join(runtime_hits[:6])}")

        if ask_has_runtime_cleanup and "runtime_housekeeping" in hay:
            score += 12
            reasons.append("preferred_family:runtime_housekeeping")
        if ask_has_runtime_cleanup and "housekeep" in hay:
            score += 8
            reasons.append("verb_housekeep_match")

        if ask_has_runtime_cleanup and any(term in hay for term in _CHANGE_SUMMARY_TERMS):
            score -= 18
            reasons.append("mismatch:change_summary_for_runtime_cleanup")
        if ask_has_runtime_cleanup and "summarize_recent_changes" in hay:
            score -= 30
            reasons.append("hard_mismatch:summarize_recent_changes")

        if str(getattr(tool, "execution_mode", "") or "").lower() == "capability_backed":
            score += 2
            reasons.append("capability_backed")

        scored.append({"tool_id": tool.tool_id, "score": score, "reasons": reasons, "tool": tool})

    scored.sort(key=lambda item: (int(item.get("score") or 0), str(item.get("tool_id") or "")), reverse=True)
    return scored


def _select_preferred_operational_tool(
    user_text: str,
    tools: List[ToolDef],
    *,
    normalized_action_input: Dict[str, Any] | None = None,
) -> Tuple[Optional[ToolDef], List[Dict[str, Any]]]:
    if not tools:
        return None, []
    scored = _semantic_override_scores(user_text=user_text, normalized_action_input=normalized_action_input, tools=tools)
    best = scored[0]["tool"] if scored else None
    return best, scored


def _select_canonical_operational_tool(user_text: str, tools: List[ToolDef]) -> Optional[ToolDef]:
    if not tools:
        return None
    hay = str(user_text or "").lower()
    tool_map = {str(t.tool_id or ""): t for t in tools}
    priority_rules = [
        ("assess_storage_health", ("disk", "storage")),
        ("summarize_recent_changes", ("recent pr", "recent pull request", "recent changes", "summarize recent")),
        ("housekeep_runtime", ("cleanup", "dry-run", "dry run", "stopped containers", "prune")),
        ("assess_mesh_presence", ("nodes are up", "which nodes", "mesh presence", "nodes up")),
    ]
    for tool_id, phrases in priority_rules:
        if tool_id not in tool_map:
            continue
        if any(phrase in hay for phrase in phrases):
            return tool_map[tool_id]
    best_tool: Optional[ToolDef] = None
    best_hits = 0
    for tool_id, terms in _OPERATIONAL_VERB_RULES:
        if tool_id not in tool_map:
            continue
        hits = sum(1 for term in terms if term in hay)
        if hits > best_hits:
            best_hits = hits
            best_tool = tool_map[tool_id]
    return best_tool if best_hits >= 2 else None


def _should_override_planner_operational_action(
    *,
    planner_tool_id: str | None,
    preferred_tool_id: str | None,
    override_scores: List[Dict[str, Any]],
    minimum_score_gap: int = 10,
) -> tuple[bool, Dict[str, Any]]:
    planner_id = str(planner_tool_id or "").strip()
    preferred_id = str(preferred_tool_id or "").strip()
    score_map = {
        str(item.get("tool_id") or ""): int(item.get("score") or 0)
        for item in (override_scores or [])
        if str(item.get("tool_id") or "").strip()
    }
    planner_score = score_map.get(planner_id)
    preferred_score = score_map.get(preferred_id)
    diagnostics = {
        "planner_tool_id": planner_id or None,
        "preferred_tool_id": preferred_id or None,
        "planner_score": planner_score,
        "preferred_score": preferred_score,
        "minimum_score_gap": minimum_score_gap,
        "score_gap": (preferred_score - planner_score) if isinstance(preferred_score, int) and isinstance(planner_score, int) else None,
    }
    if not planner_id or not preferred_id or planner_id == preferred_id:
        return False, diagnostics
    if planner_score is None or preferred_score is None:
        return False, diagnostics
    should_override = (preferred_score - planner_score) >= int(minimum_score_gap)
    return should_override, diagnostics


def _extract_bound_failure_signal(step_results: List[StepExecutionResult]) -> Dict[str, Any] | None:
    def _bound_payload(agent_payload: Dict[str, Any]) -> Dict[str, Any]:
        direct = agent_payload.get("bound_capability")
        if isinstance(direct, dict):
            return direct
        structured = agent_payload.get("structured")
        if isinstance(structured, dict) and isinstance(structured.get("bound_capability"), dict):
            return structured.get("bound_capability") or {}
        return {}

    for step in reversed(step_results or []):
        payload = (step.result or {}).get("AgentChainService") if isinstance(step.result, dict) else None
        if not isinstance(payload, dict):
            continue
        bound = _bound_payload(payload)
        if not isinstance(bound, dict):
            continue
        status_value = str(bound.get("status") or "").strip().lower()
        reason = str(bound.get("reason") or "").strip()
        path = str(bound.get("path") or "").strip()
        if status_value == "fail" or reason or path.startswith("bound_direct_"):
            return {
                "step_name": step.step_name,
                "verb_name": step.verb_name,
                "reason": reason or None,
                "path": path or None,
                "status": status_value or None,
                "detail": str(bound.get("detail") or "").strip() or None,
            }
    return None


def _sanitize_operational_history(messages: List[LLMMessage], *, user_text: str) -> Tuple[List[LLMMessage], Dict[str, Any]]:
    lowered = str(user_text or "").lower()
    safety_sensitive = any(token in lowered for token in ("dry-run", "dry run", "preview", "safe"))
    operational = _detect_operational_intent(lowered)
    if not operational:
        return messages, {"operational_history_filtering": False, "dropped": 0}

    filtered: List[LLMMessage] = []
    dropped = 0
    for msg in messages:
        if msg.role != "assistant":
            filtered.append(msg)
            continue
        content = str(msg.content or "")
        has_unsafe_command = bool(_UNSAFE_ASSISTANT_COMMAND_PATTERN.search(content))
        if safety_sensitive and has_unsafe_command:
            dropped += 1
            continue
        filtered.append(msg)
    return filtered, {
        "operational_history_filtering": True,
        "safety_sensitive": safety_sensitive,
        "dropped": dropped,
        "kept": len(filtered),
    }


def _build_blocked_execution_explanation(*, selected_tool: Optional[str], no_write_active: bool) -> str:
    tool_part = f" '{selected_tool}'" if selected_tool else ""
    if no_write_active:
        return (
            f"Execution is disabled in this session (NO_WRITE active), so I’m not running{tool_part}. "
            "I can provide a safe preview of what would run, but this is not execution."
        )
    return (
        f"Execution is blocked by runtime policy, so I’m not running{tool_part}. "
        "I can provide a safe preview-only explanation, but no commands were executed."
    )




def _execution_policy_state_from_ctx(ctx: Dict[str, Any]) -> Dict[str, Any]:
    options = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
    tool_execution_policy = str(options.get("tool_execution_policy") or "").strip().lower()
    action_execution_policy = str(options.get("action_execution_policy") or "").strip().lower()
    no_write_active = bool(ctx.get("no_write_active")) or bool(options.get("no_write_active"))
    forbidden_by_tool_policy = tool_execution_policy == "none"
    forbidden_by_action_policy = action_execution_policy == "none"
    execution_forbidden = bool(no_write_active or forbidden_by_tool_policy or forbidden_by_action_policy)
    blocked_reasons: List[str] = []
    if no_write_active:
        blocked_reasons.append("no_write_active=true")
    if forbidden_by_tool_policy:
        blocked_reasons.append("tool_execution_policy=none")
    if forbidden_by_action_policy:
        blocked_reasons.append("action_execution_policy=none")
    return {
        "no_write_active": no_write_active,
        "tool_execution_policy": tool_execution_policy or None,
        "action_execution_policy": action_execution_policy or None,
        "execution_forbidden": execution_forbidden,
        "execution_blocked_reason": ";".join(blocked_reasons) if blocked_reasons else None,
    }


def _blocked_bound_step_result(*, tool_id: str, tool_input: Dict[str, Any], correlation_id: str, policy_state: Dict[str, Any]) -> StepExecutionResult:
    blocked_reason = str(policy_state.get("execution_blocked_reason") or "execution_forbidden_policy")
    no_write_active = bool(policy_state.get("no_write_active"))
    text = _build_blocked_execution_explanation(selected_tool=tool_id, no_write_active=no_write_active)
    metadata = {
        "selected_verb": tool_id,
        "no_write_active": no_write_active,
        "tool_execution_policy": policy_state.get("tool_execution_policy"),
        "action_execution_policy": policy_state.get("action_execution_policy"),
        "execution_blocked_reason": blocked_reason,
        "selected_tool_would_have_been": tool_id,
        "path": "blocked_pre_dispatch",
        "dispatch_skipped": True,
        "reply_emitted": True,
    }
    logger.warning(
        "bound_capability_pre_dispatch_blocked corr_id=%s selected_verb=%s no_write_active=%s tool_execution_policy=%s action_execution_policy=%s blocked_reason=%s dispatch_skipped=true",
        correlation_id,
        tool_id,
        no_write_active,
        policy_state.get("tool_execution_policy"),
        policy_state.get("action_execution_policy"),
        blocked_reason,
    )
    return StepExecutionResult(
        status="success",
        verb_name=str(tool_id),
        step_name="bound_capability_pre_dispatch_blocked",
        order=0,
        result={
            "AgentChainService": {
                "text": text,
                "bound_capability": {
                    "status": "fail",
                    "reason": CapabilityRecoveryReasonV1.policy_blocked.value,
                    **metadata,
                },
                "runtime_debug": {
                    "bound_capability_pre_dispatch_blocked": True,
                    "bound_capability_policy_state": metadata,
                },
            }
        },
        latency_ms=0,
        node=settings.node_name,
        logs=["blocked <- bound capability pre-dispatch policy gate"],
        error=None,
    )

def _is_operational_guidance_text(answer: str) -> bool:
    return bool(_SHELL_GUIDANCE_PATTERN.search(str(answer or "")))


def _grounding_verdict(user_text: str, answer_text: str) -> Dict[str, Any]:
    ask_terms = _extract_key_terms(user_text)
    answer_lower = str(answer_text or "").lower()
    overlap = [term for term in ask_terms if term in answer_lower]
    unrelated_markers = [m for m in ("orion service setup", "content workflow", "internal instructions") if m in answer_lower]
    scaffolding_markers = _detect_scaffolding_markers(answer_text)
    if not ask_terms:
        anchored = not unrelated_markers and not scaffolding_markers
    else:
        anchored = bool(overlap) and not unrelated_markers and not scaffolding_markers
    return {
        "anchored": anchored,
        "ask_terms": ask_terms,
        "overlap_terms": overlap[:8],
        "unrelated_markers": unrelated_markers,
        "scaffolding_markers": scaffolding_markers,
    }


def _has_recent_followup_continuity(messages: Any, answer_text: str) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    last_user = None
    last_assistant = None
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if last_user is None and role == "user":
            last_user = content
            continue
        if last_user is not None and role == "assistant":
            last_assistant = content
            break
    if not last_user or not last_assistant:
        return False
    user_tokens = _extract_key_terms(last_user)
    if len(user_tokens) > 2:
        return False
    assistant_tokens = _extract_key_terms(last_assistant)
    if not assistant_tokens:
        return False
    answer_lower = str(answer_text or "").lower()
    return any(token in answer_lower for token in assistant_tokens[:8])


def _coerce_pack_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    return []


def _effective_packs(ctx: Dict[str, Any], req: ExecutionPlan) -> List[str]:
    packs: List[str] = []
    for source in (
        ctx.get("packs"),
        (ctx.get("metadata") or {}).get("packs") if isinstance(ctx.get("metadata"), dict) else None,
        req.metadata.get("packs") if isinstance(req.metadata, dict) else None,
    ):
        for pack in _coerce_pack_list(source):
            if pack not in packs:
                packs.append(pack)
    return packs


def _build_council_prompt(
    *,
    goal: str,
    mode: str,
    node_name: str,
    correlation_id: str,
    verb_name: str | None,
    allowed_verbs: List[str],
    toolset: List[ToolDef],
    trace: List[TraceStep],
    final_text: str | None,
) -> str:
    lines: List[str] = []
    lines.append("Council Supervision Context")
    lines.append(f"- goal: {_truncate_text(goal, 2000)}")
    lines.append(f"- mode: {mode}")
    lines.append(f"- node: {node_name}")
    lines.append(f"- correlation_id: {correlation_id}")
    if verb_name:
        lines.append(f"- entry_verb: {verb_name}")
    if allowed_verbs:
        lines.append(f"- allowed_verbs: {', '.join(allowed_verbs[:50])}")

    lines.append("\nToolset summary:")
    if not toolset:
        lines.append("- (none)")
    else:
        for tool in toolset[:25]:
            lines.append(f"- {tool.tool_id}: {_truncate_text(tool.description or '', 120)}")
        if len(toolset) > 25:
            lines.append(f"- ... ({len(toolset) - 25} more tools omitted)")

    lines.append("\nRecent trace:")
    if not trace:
        lines.append("- (no trace yet)")
    else:
        for step in trace[-8:]:
            action = step.action if isinstance(step.action, dict) else {}
            tool_id = action.get("tool_id") or "none"
            args = _compact_json(action.get("input") or {}, 220)
            obs = step.observation if isinstance(step.observation, dict) else {"raw": step.observation}
            obs_text = obs.get("llm_output") or obs.get("text") or obs.get("content") or obs.get("error") or _compact_json(obs, 280)
            lines.append(
                f"- step {step.step_index}: tool={tool_id} args={args} obs={_truncate_text(obs_text, 280)}"
            )

    if final_text:
        lines.append("\nProvisional final_text:")
        lines.append(_truncate_text(final_text, 1500))

    lines.append("\nReturn your normal council output contract.")
    return "\n".join(lines)


def _extract_council_debug(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    opinions_raw = result_payload.get("opinions") if isinstance(result_payload, dict) else []
    opinions: List[Dict[str, Any]] = []
    if isinstance(opinions_raw, list):
        for item in opinions_raw[:12]:
            if not isinstance(item, dict):
                continue
            opinions.append(
                {
                    "agent_name": _truncate_text(item.get("agent_name") or item.get("name") or "unknown", 80),
                    "confidence": item.get("confidence"),
                    "text": _truncate_text(item.get("text") or "", 800),
                }
            )

    verdict_raw = result_payload.get("verdict") if isinstance(result_payload, dict) else {}
    verdict = {}
    if isinstance(verdict_raw, dict):
        verdict = {
            "action": verdict_raw.get("action"),
            "reason": _truncate_text(verdict_raw.get("reason") or "", 500),
            "constraints": verdict_raw.get("constraints") if isinstance(verdict_raw.get("constraints"), dict) else {},
        }

    blink_raw = result_payload.get("blink") if isinstance(result_payload, dict) else {}
    blink: Dict[str, Any] = {}
    if isinstance(blink_raw, dict):
        scores = blink_raw.get("scores") if isinstance(blink_raw.get("scores"), dict) else {}
        blink = {
            "proposed_answer": _truncate_text(blink_raw.get("proposed_answer") or "", 500),
            "scores": scores,
        }

    return {
        "opinions": opinions,
        "verdict": verdict,
        "blink": blink,
    }


def _normalize_planner_decision(
    *,
    planner_step: StepExecutionResult,
    planner_final: FinalAnswer | None,
    action: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload = planner_step.result.get("PlannerReactService", {}) if isinstance(planner_step.result, dict) else {}
    raw_stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
    continue_reason = payload.get("continue_reason") if isinstance(payload, dict) else None

    if raw_stop_reason in {"final_answer", "delegate", "continue", "error"}:
        stop_reason = raw_stop_reason
    elif planner_step.status != "success":
        stop_reason = "error"
    elif action:
        stop_reason = "delegate"
    elif planner_final and planner_final.content:
        stop_reason = "final_answer"
    else:
        stop_reason = "continue"

    if not continue_reason and action:
        continue_reason = "action_present"

    final_present = bool(planner_final and planner_final.content)
    action_present = bool(action)
    will_continue = planner_step.status == "success" and action_present

    return {
        "stop_reason": stop_reason,
        "continue_reason": continue_reason,
        "final_present": final_present,
        "action_present": action_present,
        "will_continue": will_continue,
    }



def _ensure_agent_chain_tool(toolset: List[ToolDef]) -> List[ToolDef]:
    if any((t.tool_id or "") == "agent_chain" for t in toolset):
        return toolset
    return [
        *toolset,
        ToolDef(
            tool_id="agent_chain",
            description="Escalate to AgentChainService for multi-step runtime execution.",
            input_schema={},
            output_schema={},
        ),
    ]




class Supervisor:
    """
    Hierarchical controller that can pick reasoning paths, run ReAct with verbs, checkpoint with Council,
    and escalate to Agent Chain.
    """

    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.registry = VerbRegistry(VERBS_DIR)
        self.pack_manager = PackManager(ORION_PKG_DIR / "cognition")
        self.llm_client = LLMGatewayClient(bus)
        self.planner_client = PlannerReactClient(bus)
        self.agent_client = AgentChainClient(bus)
        self.council_client = CouncilClient(bus)

    def _toolset(self, packs: List[str] | None = None, tags: List[str] | None = None) -> List[ToolDef]:
        try:
            names: List[str] = []
            if packs:
                self.pack_manager.load_packs()
                names = self.pack_manager.load_verb_set(packs)
            verbs = self.registry.filter(names=names or None, tags=tags)
        except Exception as exc:
            logger.warning("Verb registry load failed, falling back to all verbs: %s", exc)
            verbs = self.registry.list()

        tools: List[ToolDef] = []
        for v in verbs:
            if not is_active(v.name, node_name=settings.node_name):
                continue
            tools.append(
                ToolDef(
                    tool_id=v.name,
                    description=v.description or f"Orion verb: {v.name}",
                    input_schema=v.input_schema or {},
                    output_schema=v.output_schema or {},
                    execution_mode=v.execution_mode or None,
                    requires_capability_selector=bool(v.requires_capability_selector),
                    preferred_skill_families=list(v.preferred_skill_families or []),
                    side_effect_level=v.side_effect_level or None,
                )
            )
        return tools

    def _should_use_react(self, mode: str, tools: List[ToolDef]) -> bool:
        if mode in {"agent", "council"}:
            return True
        if mode == "direct_chat":
            return False
        return bool(tools)



    async def _planner_step(
        self,
        *,
        source: ServiceRef,
        goal_text: str,
        toolset: List[ToolDef],
        trace: List[TraceStep],
        ctx: Dict[str, Any],
        correlation_id: str,
        diagnostic: bool = False,
    ) -> Tuple[PlannerRequest, StepExecutionResult, FinalAnswer | None, Dict[str, Any] | None]:
        logger.info(
            "grounding_snapshot component=planner_react corr_id=%s trace_id=%s session_id=%s text_source=goal text_head=%r prior_step_results_count=%s recall_included=%s scaffolding_markers=%s",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            _truncate_text(goal_text, 220),
            len(ctx.get("prior_step_results") or []) if isinstance(ctx.get("prior_step_results"), list) else len(ctx.get("prior_step_results") or {}) if isinstance(ctx.get("prior_step_results"), dict) else 0,
            bool(ctx.get("memory_digest")),
            _detect_scaffolding_markers(goal_text),
        )
        ext_facts = {"text": ctx.get("memory_digest", "")}
        if ctx.get("output_mode"):
            ext_facts["output_mode"] = ctx["output_mode"]
        if ctx.get("response_profile"):
            ext_facts["response_profile"] = ctx["response_profile"]
        for key in ("delivery_grounding_mode", "grounding_context", "anti_generic_drift"):
            if ctx.get(key):
                ext_facts[key] = ctx[key]
        ext_facts["no_write_active"] = bool(ctx.get("no_write_active"))
        ext_facts["execution_blocked_reason"] = ctx.get("execution_blocked_reason")
        ext_facts["operational_intent_detected"] = bool(ctx.get("operational_intent_detected"))
        ext_facts["available_operational_tools"] = list(ctx.get("available_operational_tools") or [])
        history_msgs = [LLMMessage(**m) if not isinstance(m, LLMMessage) else m for m in (ctx.get("messages") or [])]
        sanitized_history, history_filter_meta = _sanitize_operational_history(history_msgs, user_text=goal_text)
        ext_facts["operational_history_filter"] = history_filter_meta
        planner_req = PlannerRequest(
            request_id=correlation_id,
            caller="cortex-exec",
            goal=Goal(description=goal_text, metadata={"verb": ctx.get("verb")}),
            context=ContextBlock(
                conversation_history=sanitized_history,
                external_facts=ext_facts,
            ),
            toolset=toolset,
            trace=trace,
            limits=Limits(max_steps=1, timeout_seconds=int(settings.step_timeout_ms / 1000)),
            preferences=Preferences(plan_only=True, return_trace=True, delegate_tool_execution=True),
        )
        reply_channel = f"{settings.exec_result_prefix}:PlannerReactService:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> PlannerReactService (plan_only) reply={reply_channel}"]
        planner_timeout = float(
            ctx.get("planner_timeout_sec")
            or (settings.step_timeout_ms / 1000.0)
        )
        if diagnostic:
            planner_timeout = min(planner_timeout, float(settings.diagnostic_agent_timeout_sec))
        planner_res = await self.planner_client.plan(
            source=source,
            req=planner_req,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=planner_timeout,
        )
        logs.append(f"ok <- PlannerReactService status={planner_res.status}")
        logger.info(
            "grounding_snapshot component=planner_react_result corr_id=%s trace_id=%s session_id=%s text_source=planner_final text_head=%r prior_step_results_count=%s recall_included=%s scaffolding_markers=%s",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            _truncate_text((planner_res.final_answer.content if planner_res.final_answer else ""), 220),
            len(trace),
            bool(ctx.get("memory_digest")),
            _detect_scaffolding_markers(planner_res.final_answer.content if planner_res.final_answer else ""),
        )

        step_res = StepExecutionResult(
            status="success" if planner_res.status == "ok" else "fail",
            verb_name=ctx.get("verb") or "unknown",
            step_name="planner_react",
            order=-1,
            result={"PlannerReactService": planner_res.model_dump(mode="json")},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
            error=None if planner_res.status == "ok" else (planner_res.error or {}).get("message"),
        )
        final = planner_res.final_answer
        action: Dict[str, Any] | None = None
        if planner_res.trace:
            last = planner_res.trace[-1]
            action = last.action
        return planner_req, step_res, final, action

    async def _execute_action(
        self,
        *,
        source: ServiceRef,
        action: Dict[str, Any],
        ctx: Dict[str, Any],
        correlation_id: str,
        mode: str,
        packs: List[str],
    ) -> StepExecutionResult:
        tool_id = action.get("tool_id")
        tool_input = action.get("input") or {}
        logger.info(
            "grounding_snapshot component=%s corr_id=%s trace_id=%s session_id=%s text_source=action_input text_head=%r prior_step_results_count=%s recall_included=%s scaffolding_markers=%s",
            f"action:{tool_id}",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            _compact_json(tool_input, 220),
            len(ctx.get("prior_step_results") or []) if isinstance(ctx.get("prior_step_results"), list) else len(ctx.get("prior_step_results") or {}) if isinstance(ctx.get("prior_step_results"), dict) else 0,
            bool(ctx.get("memory_digest")),
            _detect_scaffolding_markers(tool_input),
        )
        if str(tool_id) == "agent_chain":
            logger.info("dispatch_action corr_id=%s mode=%s step=agent_chain route=AgentChainService", correlation_id, mode)
            chain_ctx = {**ctx, **tool_input}
            return await self._agent_chain_escalation(
                source=source,
                correlation_id=correlation_id,
                ctx=chain_ctx,
                packs=packs,
            )
        if not tool_id or not is_active(str(tool_id), node_name=settings.node_name):
            logger.warning("Inactive verb selected by supervisor corr_id=%s verb=%s", correlation_id, tool_id)
            return StepExecutionResult(
                status="fail",
                verb_name=str(tool_id or "unknown"),
                step_name="inactive_verb_guard",
                order=0,
                result={"error": "inactive_verb", "verb": tool_id, "node": settings.node_name},
                latency_ms=0,
                node=settings.node_name,
                logs=[f"reject <- inactive verb {tool_id}"],
                error=f"inactive_verb:{tool_id}",
            )

        verb_cfg = self.registry.get(tool_id)
        if _is_capability_backed_cfg(verb_cfg):
            policy_state = _execution_policy_state_from_ctx(ctx)
            logger.info(
                "bound_capability_policy_state corr_id=%s selected_verb=%s no_write_active=%s tool_execution_policy=%s action_execution_policy=%s execution_forbidden=%s",
                correlation_id,
                tool_id,
                policy_state.get("no_write_active"),
                policy_state.get("tool_execution_policy"),
                policy_state.get("action_execution_policy"),
                policy_state.get("execution_forbidden"),
            )
            if policy_state.get("execution_forbidden"):
                return _blocked_bound_step_result(
                    tool_id=str(tool_id),
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                    correlation_id=correlation_id,
                    policy_state=policy_state,
                )
            logger.info(
                "dispatch_action corr_id=%s mode=%s step=%s route=AgentChainService(bound_capability)",
                correlation_id,
                mode,
                tool_id,
            )
            bound_execution = BoundCapabilityExecutionRequestV1(
                selected_verb=str(tool_id),
                normalized_action_input=tool_input if isinstance(tool_input, dict) else {},
                execution_mode="capability_backed",
                requires_capability_selector=bool(getattr(verb_cfg, "requires_capability_selector", False)),
                preferred_skill_families=list(getattr(verb_cfg, "preferred_skill_families", []) or []),
                side_effect_level=getattr(verb_cfg, "side_effect_level", None),
                planner_correlation_id=correlation_id,
                planner_metadata={"planner_mode": mode, "source": "supervisor_planner_delegate"},
                selected_tool_metadata={"tool_id": str(tool_id), "description": getattr(verb_cfg, "description", "")},
                policy_metadata={
                    "no_write_active": bool(policy_state.get("no_write_active")),
                    "tool_execution_policy": policy_state.get("tool_execution_policy"),
                    "action_execution_policy": policy_state.get("action_execution_policy"),
                    "execution_blocked_reason": policy_state.get("execution_blocked_reason"),
                },
                recovery=CapabilityRecoveryDecisionV1(
                    reason=CapabilityRecoveryReasonV1.internal_contract_error,
                    allow_replan=False,
                    replanned=False,
                    detail="default_fail_closed",
                ),
            )
            chain_ctx = {**ctx, "__bound_execution": bound_execution.model_dump(mode="json")}
            try:
                return await self._agent_chain_escalation(
                    source=source,
                    correlation_id=correlation_id,
                    ctx=chain_ctx,
                    packs=packs,
                )
            except Exception as exc:
                logger.error(
                    "bound_capability_fail_closed corr_id=%s selected_verb=%s reason=capability_executor_unavailable error=%s",
                    correlation_id,
                    tool_id,
                    exc,
                )
                return StepExecutionResult(
                    status="fail",
                    verb_name=str(tool_id),
                    step_name="bound_capability_execution_failed",
                    order=0,
                    result={
                        "error": "capability_executor_unavailable",
                        "selected_verb": str(tool_id),
                        "bound_capability_execution": bound_execution.model_dump(mode="json"),
                    },
                    latency_ms=0,
                    node=settings.node_name,
                    logs=["fail_closed <- bound capability escalation unavailable"],
                    error=f"capability_executor_unavailable:{tool_id}",
                )
        step = _verb_to_step(verb_cfg)

        exec_ctx = {**ctx, **tool_input}
        exec_ctx.setdefault("verb", tool_id)

        step_res = await call_step_services(
            self.bus,
            source=source,
            step=step,
            ctx=exec_ctx,
            correlation_id=correlation_id,
            diagnostic=bool(ctx.get("diagnostic")),
        )
        return step_res

    async def _council_checkpoint(
        self,
        *,
        source: ServiceRef,
        correlation_id: str,
        history: List[Dict[str, Any]],
        goal_text: str,
        mode: str,
        verb_name: str | None,
        allowed_verbs: List[str],
        toolset: List[ToolDef],
        trace: List[TraceStep],
        final_text: str | None,
        node_name: str,
    ) -> StepExecutionResult:
        prompt = _build_council_prompt(
            goal=goal_text,
            mode=mode,
            node_name=node_name,
            correlation_id=correlation_id,
            verb_name=verb_name,
            allowed_verbs=allowed_verbs,
            toolset=toolset,
            trace=trace,
            final_text=final_text,
        )
        deliberation = DeliberationRequest(
            prompt=prompt,
            history=history,
            event="council_deliberation",
            trace_id=correlation_id,
        )
        reply_channel = f"{settings.channel_council_reply_prefix}:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> CouncilService reply={reply_channel}"]
        try:
            council_res = await self.council_client.deliberate(
                source=source,
                req=deliberation,
                correlation_id=correlation_id,
                reply_to=reply_channel,
                timeout_sec=float(settings.diagnostic_agent_timeout_sec),
            )
            logs.append("ok <- CouncilService")
            status = "success"
            result_payload = council_res.model_dump(mode="json")
            result_payload["debug_compact"] = _extract_council_debug(result_payload)
            error_msg = None
        except Exception as exc:
            logs.append(f"timeout/exception <- CouncilService: {exc}")
            status = "fail"
            result_payload = {"error": str(exc)}
            error_msg = str(exc)
        return StepExecutionResult(
            status=status,
            verb_name="council_checkpoint",
            step_name="council_checkpoint",
            order=99,
            result={"CouncilService": result_payload},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
        )

    async def _agent_chain_escalation(
        self,
        *,
        source: ServiceRef,
        correlation_id: str,
        ctx: Dict[str, Any],
        packs: List[str],
    ) -> StepExecutionResult:
        logger.info(
            "grounding_snapshot component=agent_chain_entry corr_id=%s trace_id=%s session_id=%s text_source=last_user_message text_head=%r prior_step_results_count=%s recall_included=%s scaffolding_markers=%s",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            _truncate_text(_last_user_message(ctx), 220),
            len(ctx.get("prior_step_results") or []) if isinstance(ctx.get("prior_step_results"), list) else len(ctx.get("prior_step_results") or {}) if isinstance(ctx.get("prior_step_results"), dict) else 0,
            bool(ctx.get("memory_digest")),
            _detect_scaffolding_markers(_last_user_message(ctx)),
        )
        agent_req = {
            "text": _last_user_message(ctx),
            "mode": ctx.get("mode") or "agent",
            "session_id": ctx.get("session_id"),
            "user_id": ctx.get("user_id"),
            "messages": ctx.get("messages") or [],
            "packs": packs,
            "output_mode": ctx.get("output_mode"),
            "response_profile": ctx.get("response_profile"),
        }
        bound_execution = ctx.get("__bound_execution")
        if isinstance(bound_execution, dict):
            contract = BoundCapabilityExecutionRequestV1.model_validate(bound_execution)
            logger.info(
                "bound_capability_request_received corr_id=%s selected_verb=%s selected_verb_preserved=1",
                correlation_id,
                contract.selected_verb,
            )
            agent_req["goal_description"] = "execute_selected_capability"
            agent_req["bound_capability_execution"] = contract.model_dump(mode="json")
        reply_channel = f"{settings.exec_result_prefix}:AgentChainService:{correlation_id}"
        t0 = time.time()
        logs = [f"rpc -> AgentChainService reply={reply_channel}"]
        agent_res = await self.agent_client.run_chain(
            source=source,
            req=AgentChainRequest(**agent_req),
            correlation_id=correlation_id,
            reply_to=reply_channel,
            timeout_sec=float(settings.step_timeout_ms) / 1000.0,
        )
        logs.append("ok <- AgentChainService")
        logger.info(
            "grounding_snapshot component=agent_chain_exit corr_id=%s trace_id=%s session_id=%s text_source=agent_result text_head=%r prior_step_results_count=%s recall_included=%s scaffolding_markers=%s",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            _truncate_text(agent_res.text, 220),
            len(ctx.get("prior_step_results") or []) if isinstance(ctx.get("prior_step_results"), list) else len(ctx.get("prior_step_results") or {}) if isinstance(ctx.get("prior_step_results"), dict) else 0,
            bool(ctx.get("memory_digest")),
            _detect_scaffolding_markers(agent_res.text),
        )
        return StepExecutionResult(
            status="success",
            verb_name="agent_chain",
            step_name="agent_chain",
            order=100,
            result={"AgentChainService": agent_res.model_dump(mode="json")},
            latency_ms=int((time.time() - t0) * 1000),
            node=settings.node_name,
            logs=logs,
        )

    async def execute(
        self,
        *,
        source: ServiceRef,
        req: ExecutionPlan,
        correlation_id: str,
        ctx: Dict[str, Any],
        recall_cfg: Dict[str, Any],
    ) -> PlanExecutionResult:
        step_results: List[StepExecutionResult] = []
        memory_used = False
        recall_debug: Dict[str, Any] = {}
        ctx.setdefault("debug", {})

        verb_recall_profile = None
        if isinstance(req.metadata, dict):
            verb_recall_profile = req.metadata.get("recall_profile") or None
        ctx.setdefault("plan_recall_profile", verb_recall_profile)
        recall_required = bool(recall_cfg.get("required", False))
        recall_policy = delivery_safe_recall_decision(
            recall_cfg,
            req.steps,
            output_mode=ctx.get("output_mode"),
            verb_profile=verb_recall_profile,
            user_text=_last_user_message(ctx),
            runtime_mode=ctx.get("mode") or req.metadata.get("mode") or "agent",
        )
        selected_profile = recall_policy["profile"]
        profile_source = recall_policy["profile_source"]
        ctx.setdefault("debug", {})["recall_gating_reason"] = recall_policy["recall_gating_reason"]
        ctx.setdefault("debug", {})["recall_profile_source"] = profile_source
        ctx.setdefault("debug", {})["recall_profile_override_source"] = recall_policy.get("profile_override_source")
        inline_recall = has_inline_recall(req.steps)
        should_recall = bool(recall_policy["run_recall"])
        recall_reason = str(recall_policy["reason"])

        if should_recall and not inline_recall:
            logger.info(
                "Supervisor recall resolved profile=%s source=%s override_source=%s gating=%s recall_gating_reason=%s",
                selected_profile,
                profile_source,
                recall_policy.get("profile_override_source"),
                recall_reason,
                recall_policy["recall_gating_reason"],
            )
            recall_step, recall_debug, _ = await run_recall_step(
                self.bus,
                source=source,
                ctx=ctx,
                correlation_id=correlation_id,
                recall_cfg=recall_cfg,
                recall_profile=selected_profile,
                diagnostic=bool(ctx.get("diagnostic")),
            )
            step_results.append(recall_step)
            memory_used = recall_step.status == "success"
            if recall_required and recall_step.status != "success":
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=ctx.get("mode") or req.metadata.get("mode") or "agent",
                    final_text=None,
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=recall_step.error,
                )
        else:
            if inline_recall:
                recall_debug = {
                    "skipped": "inline_recall_step_present",
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                logger.info(
                    "Supervisor recall skipped; inline RecallService step present",
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg},
                )
            else:
                recall_debug = {
                    "skipped": recall_reason,
                    "recall_gating_reason": recall_policy["recall_gating_reason"],
                }
                logger.info(
                    "Supervisor recall skipped by gating (%s) recall_gating_reason=%s",
                    recall_reason,
                    recall_policy["recall_gating_reason"],
                    extra={"correlation_id": correlation_id, "recall_cfg": recall_cfg},
                )

        packs = _effective_packs(ctx, req)
        ctx["packs"] = packs
        tags = ctx.get("verb_tags") or []
        tools = self._toolset(packs=packs, tags=tags)
        logger.info(
            "supervisor_wiring corr=%s output_mode=%s profile=%s packs=%s tool_ids=%s",
            correlation_id,
            ctx.get("output_mode"),
            ctx.get("response_profile"),
            packs,
            [t.tool_id for t in tools[:30]],
        )
        mode = ctx.get("mode") or req.metadata.get("mode") or "agent"
        if mode in {"agent", "council"}:
            tools = _ensure_agent_chain_tool(tools)
        if mode == "auto":
            logger.warning(
                "Supervisor received unexpected mode=auto corr_id=%s; preserving deterministic execution path",
                correlation_id,
            )
        options = ctx.get("options") if isinstance(ctx.get("options"), dict) else {}
        policy_state = _execution_policy_state_from_ctx(ctx)
        no_write_active = bool(policy_state.get("no_write_active"))
        execution_blocked_reason = policy_state.get("execution_blocked_reason")
        user_text = _last_user_message(ctx)
        operational_intent_detected = _detect_operational_intent(user_text)
        operational_tools = _available_operational_semantic_tools(tools)
        offered_operational_tool_ids = [t.tool_id for t in operational_tools]
        preferred_operational_tool, override_scores = _select_preferred_operational_tool(
            user_text,
            operational_tools,
            normalized_action_input=ctx.get("normalized_action_input") if isinstance(ctx.get("normalized_action_input"), dict) else None,
        )
        canonical_operational_tool = _select_canonical_operational_tool(user_text, operational_tools)
        if canonical_operational_tool is not None:
            operational_intent_detected = True
            preferred_operational_tool = canonical_operational_tool
        ctx["no_write_active"] = no_write_active
        ctx["execution_blocked_reason"] = execution_blocked_reason
        ctx["operational_intent_detected"] = operational_intent_detected
        ctx["available_operational_tools"] = offered_operational_tool_ids
        if canonical_operational_tool is not None:
            ctx.setdefault("debug", {})["canonical_operational_tool"] = canonical_operational_tool.tool_id
        allowed_verbs = [str(v).strip() for v in (options.get("allowed_verbs") or []) if str(v).strip()]
        if operational_intent_detected:
            logger.info(
                "operational_intent_detected corr_id=%s user_intent=%s offered_semantic_tools=%s no_write_active=%s",
                correlation_id,
                "operational_action",
                offered_operational_tool_ids,
                no_write_active,
            )
            logger.info(
                "semantic_tool_override_candidates corr_id=%s candidates=%s",
                correlation_id,
                [
                    {
                        "tool_id": item.get("tool_id"),
                        "score": item.get("score"),
                        "reasons": item.get("reasons"),
                    }
                    for item in override_scores
                ],
            )
            if preferred_operational_tool:
                logger.info(
                    "semantic_tool_override_selected corr_id=%s selected=%s score=%s reasons=%s",
                    correlation_id,
                    preferred_operational_tool.tool_id,
                    next((item.get("score") for item in override_scores if item.get("tool_id") == preferred_operational_tool.tool_id), None),
                    next((item.get("reasons") for item in override_scores if item.get("tool_id") == preferred_operational_tool.tool_id), []),
                )
            logger.info(
                "semantic_tool_override_rejected corr_id=%s rejected=%s",
                correlation_id,
                [
                    {
                        "tool_id": item.get("tool_id"),
                        "score": item.get("score"),
                        "reasons": item.get("reasons"),
                    }
                    for item in override_scores[1:]
                ],
            )

        if not self._should_use_react(mode, tools):
            logger.info("Supervisor: using direct LLM path")
            if not is_active(req.verb_name, node_name=settings.node_name):
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=f"Verb '{req.verb_name}' is inactive on node {settings.node_name}.",
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=f"inactive_verb:{req.verb_name}",
                )
            direct_cfg = self.registry.get(req.verb_name)
            step = _verb_to_step(direct_cfg)
            direct_step = await call_step_services(
                self.bus,
                source=source,
                step=step,
                ctx=ctx,
                correlation_id=correlation_id,
                diagnostic=bool(ctx.get("diagnostic")),
            )
            step_results.append(direct_step)
            final_obs = _extract_observation(direct_step)
            return PlanExecutionResult(
                verb_name=req.verb_name,
                request_id=correlation_id,
                status="success" if direct_step.status == "success" else "fail",
                blocked=False,
                blocked_reason=None,
                steps=step_results,
                mode=mode,
                final_text=final_obs.get("llm_output"),
                memory_used=memory_used,
                recall_debug=recall_debug,
                error=direct_step.error,
            )

        trace: List[TraceStep] = []
        max_steps = int(ctx.get("max_steps") or 3)
        final_text: Optional[str] = None
        planner_thought: Optional[str] = None
        executed_steps: List[str] = [s.step_name for s in step_results]
        semantic_execution_validated = False
        selected_tool_would_have_been = preferred_operational_tool.tool_id if preferred_operational_tool else None

        for loop_index in range(max_steps):
            goal_text = _last_user_message(ctx)
            planner_req, planner_step, planner_final, action = await self._planner_step(
                source=source,
                goal_text=goal_text,
                toolset=tools,
                trace=trace,
                ctx=ctx,
                correlation_id=correlation_id,
            )
            step_results.append(planner_step)
            executed_steps.append(planner_step.step_name)

            planner_thought = _extract_latest_planner_thought(planner_step.result)
            decision = _normalize_planner_decision(
                planner_step=planner_step,
                planner_final=planner_final,
                action=action,
            )

            if planner_final and planner_final.content:
                final_text = planner_final.content

            logger.info(
                "planner_decision corr_id=%s mode=%s verb=%s step=%s stop_reason=%s continue_reason=%s action_present=%s final_present=%s will_continue=%s remaining_steps=%s",
                correlation_id,
                mode,
                req.verb_name,
                planner_step.step_name,
                decision["stop_reason"],
                decision["continue_reason"],
                decision["action_present"],
                decision["final_present"],
                decision["will_continue"],
                max(0, max_steps - (loop_index + 1)),
            )

            tool_id = (action or {}).get("tool_id") if isinstance(action, dict) else None
            logger.info("planner_action corr_id=%s tool_id=%s lane=%s", correlation_id, tool_id, "depth2")
            if operational_intent_detected and preferred_operational_tool and mode in {"agent", "council"}:
                should_override, override_diag = _should_override_planner_operational_action(
                    planner_tool_id=str(tool_id or ""),
                    preferred_tool_id=preferred_operational_tool.tool_id,
                    override_scores=override_scores,
                )
                if canonical_operational_tool is not None and str(tool_id or "") != canonical_operational_tool.tool_id:
                    should_override = True
                    override_diag = {
                        **override_diag,
                        "forced_by": "canonical_operational_rail",
                        "preferred_tool_id": canonical_operational_tool.tool_id,
                    }
                ctx.setdefault("debug", {})["semantic_override_decision"] = override_diag
                if should_override and action:
                    logger.info(
                        "semantic_tool_override_applied corr_id=%s planner_tool=%s planner_score=%s preferred_tool=%s preferred_score=%s score_gap=%s",
                        correlation_id,
                        override_diag.get("planner_tool_id"),
                        override_diag.get("planner_score"),
                        override_diag.get("preferred_tool_id"),
                        override_diag.get("preferred_score"),
                        override_diag.get("score_gap"),
                    )
                    action = {
                        **action,
                        "tool_id": preferred_operational_tool.tool_id,
                    }
                    tool_id = preferred_operational_tool.tool_id

            if decision["stop_reason"] == "final_answer" and not decision["action_present"]:
                forced_delegate = False
                if operational_intent_detected and preferred_operational_tool and mode in {"agent", "council"}:
                    logger.info(
                        "semantic_tool_preferred corr_id=%s chosen_path=%s selected_tool=%s no_write_active=%s reason=%s",
                        correlation_id,
                        "blocked_explanation" if no_write_active else "delegate",
                        preferred_operational_tool.tool_id,
                        no_write_active,
                        "operational_intent_with_semantic_tool_available",
                    )
                    ctx.setdefault("debug", {})["selected_tool_would_have_been"] = preferred_operational_tool.tool_id
                    if no_write_active:
                        ctx.setdefault("debug", {})["no_write_downgrade"] = True
                        ctx.setdefault("debug", {})["downgraded_to_explanation"] = True
                        final_text = _build_blocked_execution_explanation(
                            selected_tool=preferred_operational_tool.tool_id,
                            no_write_active=True,
                        )
                        logger.info(
                            "no_write_downgrade corr_id=%s selected_tool_would_have_been=%s blocked_execution_explanation_emitted=true",
                            correlation_id,
                            preferred_operational_tool.tool_id,
                        )
                        break
                    action = {
                        "tool_id": preferred_operational_tool.tool_id,
                        "input": {"text": goal_text},
                    }
                    decision["action_present"] = True
                    decision["stop_reason"] = "delegate"
                    decision["continue_reason"] = "operational_semantic_tool_preferred"
                    forced_delegate = True
                if decision["action_present"]:
                    logger.info(
                        "planner_final_answer_overridden corr_id=%s chosen_path=delegate selected_tool=%s",
                        correlation_id,
                        (action or {}).get("tool_id"),
                    )
                    final_text = None
                if forced_delegate:
                    pass
                elif not trace:
                    trace.append(
                        TraceStep(
                            step_index=0,
                            thought=planner_thought,
                            action=None,
                            observation={"llm_output": final_text},
                        )
                    )
                if not forced_delegate:
                    logger.info(
                        "agent_runtime_stop corr_id=%s mode=%s verb=%s reason=%s final_len=%s action_present=%s executed_steps=%s skipped_steps=%s",
                        correlation_id,
                        mode,
                        req.verb_name,
                        decision["stop_reason"],
                        len(final_text or ""),
                        decision["action_present"],
                        executed_steps,
                        ["agent_chain"],
                    )
                    break

            if planner_step.status != "success":
                logger.info(
                    "agent_runtime_stop corr_id=%s mode=%s verb=%s reason=planner_step_%s final_len=%s action_present=%s executed_steps=%s skipped_steps=%s",
                    correlation_id,
                    mode,
                    req.verb_name,
                    planner_step.status,
                    len(final_text or ""),
                    decision["action_present"],
                    executed_steps,
                    ["agent_chain"],
                )
                break

            if not action:
                logger.info("planner_action_ambiguous corr_id=%s default=agent_chain", correlation_id)
                action = {"tool_id": "agent_chain", "input": {}}

            action_step = await self._execute_action(
                source=source,
                action=action,
                ctx=ctx,
                correlation_id=correlation_id,
                mode=mode,
                packs=packs or [],
            )
            step_results.append(action_step)
            executed_steps.append(action_step.step_name)
            if action_step.status != "success" and str(action_step.error or "").startswith("inactive_verb:"):
                return PlanExecutionResult(
                    verb_name=req.verb_name,
                    request_id=correlation_id,
                    status="fail",
                    blocked=False,
                    blocked_reason=None,
                    steps=step_results,
                    mode=mode,
                    final_text=f"Verb '{action.get('tool_id')}' is inactive on node {settings.node_name}.",
                    memory_used=memory_used,
                    recall_debug=recall_debug,
                    error=action_step.error,
                )
            obs = _extract_observation(action_step)
            trace.append(
                TraceStep(
                    step_index=len(trace),
                    thought=None,
                    action=action,
                    observation=obs,
                )
            )
            if obs.get("llm_output"):
                final_text = obs.get("llm_output")

            try:
                agent_payload = (action_step.result or {}).get("AgentChainService", {})
                if isinstance(agent_payload, dict):
                    bound_payload = agent_payload.get("bound_capability")
                    if not isinstance(bound_payload, dict):
                        structured_payload = agent_payload.get("structured")
                        if isinstance(structured_payload, dict):
                            bound_payload = structured_payload.get("bound_capability")
                    if isinstance(bound_payload, dict) and str(bound_payload.get("path") or "").startswith("blocked_"):
                        ctx.setdefault("debug", {})["downgraded_to_explanation"] = True
                        ctx.setdefault("debug", {})["no_write_downgrade"] = True
            except Exception:
                pass

            selected_tool = str((action or {}).get("tool_id") or "")
            selected_cfg = None
            if selected_tool and selected_tool not in {"agent_chain", "tool_chain"}:
                try:
                    selected_cfg = self.registry.get(selected_tool)
                except Exception:
                    selected_cfg = None
            selected_is_capability = _is_capability_backed_cfg(selected_cfg) if selected_cfg else False
            if selected_tool and selected_tool not in {"agent_chain", "tool_chain"}:
                if selected_is_capability:
                    semantic_execution_validated = True
                if selected_is_capability:
                    logger.info(
                        "agent_runtime_stop corr_id=%s mode=%s verb=%s reason=capability_bound_executed tool_id=%s",
                        correlation_id,
                        mode,
                        req.verb_name,
                        selected_tool,
                    )
                    break
                logger.info(
                    "agent_runtime_continue corr_id=%s mode=%s verb=%s reason=delegate_then_agent_chain tool_id=%s",
                    correlation_id,
                    mode,
                    req.verb_name,
                    selected_tool,
                )
                final_text = None
                break

        # Council checkpoint (opt-in unless explicit mode=council)
        council_step: Optional[StepExecutionResult] = None

        require_council = bool(ctx.get("require_council")) or (ctx.get("mode") == "council")
        if require_council:
            council_step = await self._council_checkpoint(
                source=source,
                correlation_id=correlation_id,
                history=ctx.get("messages") or [],
                goal_text=_last_user_message(ctx),
                mode=mode,
                verb_name=req.verb_name,
                allowed_verbs=allowed_verbs,
                toolset=tools,
                trace=trace,
                final_text=final_text,
                node_name=settings.node_name,
            )
            step_results.append(council_step)
            try:
                council_payload = council_step.result.get("CouncilService", {})
                if isinstance(council_payload, dict):
                    final_text = council_payload.get("final_text") or final_text
                    compact_debug = council_payload.get("debug_compact")
                    if isinstance(compact_debug, dict):
                        ctx.setdefault("debug", {})["council"] = compact_debug
                        recall_debug["council_debug"] = compact_debug
            except Exception:
                pass

        if ctx.get("force_agent_chain"):
            logger.info(
                "agent_runtime_continue corr_id=%s mode=%s verb=%s reason=force_agent_chain",
                correlation_id,
                mode,
                req.verb_name,
            )
            final_text = None

        # Escalate to agent chain if still no final text
        if not final_text:
            logger.info(
                "agent_runtime_continue corr_id=%s mode=%s verb=%s reason=needs_chain step=agent_chain",
                correlation_id,
                mode,
                req.verb_name,
            )
            agent_step = await self._agent_chain_escalation(
                source=source,
                correlation_id=correlation_id,
                ctx=ctx,
                packs=packs or [],
            )
            step_results.append(agent_step)
            try:
                agent_payload = agent_step.result.get("AgentChainService", {})
                if isinstance(agent_payload, dict):
                    final_text = agent_payload.get("text") or final_text
            except Exception:
                pass

        # Shallow planner-only final (no agent_chain): re-synthesize if meta-plan for delivery-oriented mode
        if final_text and mode in {"agent", "council"}:
            grounding = build_delivery_grounding_context(
                user_text=_last_user_message(ctx),
                output_mode=ctx.get("output_mode"),
            )
            shallow, rewrite_reason = should_rewrite_for_instructional(
                final_text,
                ctx.get("output_mode"),
                request_text=_last_user_message(ctx),
                grounding_mode=grounding.get("delivery_grounding_mode"),
            )
            if shallow:
                logger.info(
                    "supervisor_finalize_response_invoked=1 corr=%s output_mode=%s reason=%s grounding=%s",
                    correlation_id,
                    ctx.get("output_mode"),
                    rewrite_reason,
                    grounding.get("delivery_grounding_mode"),
                )
                ctx.setdefault("debug", {})["supervisor_meta_plan_finalize"] = True
                try:
                    ser_trace = json.dumps(
                        [
                            t.model_dump(mode="json") if hasattr(t, "model_dump") else dict(t)
                            for t in trace
                        ],
                        default=str,
                    )[:12000]
                except Exception:
                    ser_trace = "[]"
                goal_txt = _last_user_message(ctx)
                ser_trace = _sanitize_text_block(ser_trace)
                finalize_input_blob = {
                    "original_request": goal_txt,
                    "request": goal_txt,
                    "text": goal_txt,
                    "trace": ser_trace,
                    "output_mode": ctx.get("output_mode") or "direct_answer",
                    "response_profile": ctx.get("response_profile") or "direct_answer",
                    **grounding,
                }
                scaffold_markers = _detect_scaffolding_markers(finalize_input_blob)
                if scaffold_markers:
                    logger.warning(
                        "finalizer_sanitize corr_id=%s markers=%s",
                        correlation_id,
                        scaffold_markers,
                    )
                    finalize_input_blob["trace"] = _sanitize_text_block(str(finalize_input_blob.get("trace") or ""))
                fin_step = await self._execute_action(
                    source=source,
                    action={
                        "tool_id": "finalize_response",
                        "input": finalize_input_blob,
                    },
                    ctx=ctx,
                    correlation_id=correlation_id,
                    mode=mode,
                    packs=packs or [],
                )
                step_results.append(fin_step)
                if fin_step.status == "success":
                    obs = _extract_observation(fin_step)
                    final_text = obs.get("llm_output") or final_text

        if final_text and operational_intent_detected and _is_operational_guidance_text(final_text):
            if no_write_active:
                final_text = _build_blocked_execution_explanation(
                    selected_tool=selected_tool_would_have_been,
                    no_write_active=True,
                )
                ctx.setdefault("debug", {})["no_write_downgrade"] = True
                ctx.setdefault("debug", {})["downgraded_to_explanation"] = True
                logger.info(
                    "final_answer_operational_guidance_blocked corr_id=%s chosen_path=blocked_explanation no_write_active=true reason=direct_command_guidance_blocked",
                    correlation_id,
                )
            elif not semantic_execution_validated:
                final_text = (
                    "I can provide a safe preview, but I’m not executing operations in this answer. "
                    "Use the semantic runtime housekeeping path for execution; this response is explanation-only."
                )
                ctx.setdefault("debug", {})["downgraded_to_explanation"] = True
                logger.info(
                    "final_answer_operational_guidance_blocked corr_id=%s chosen_path=safe_preview no_write_active=false reason=unvalidated_direct_command_guidance",
                    correlation_id,
                )
            else:
                logger.info(
                    "final_answer_operational_guidance_allowed corr_id=%s chosen_path=delegate validated_semantic_execution=true",
                    correlation_id,
                )

        logger.info(
            "agent_runtime_stop corr_id=%s mode=%s verb=%s reason=loop_complete final_len=%s output_mode=%s packs=%s steps=%s",
            correlation_id,
            mode,
            req.verb_name,
            len(final_text or ""),
            ctx.get("output_mode"),
            packs,
            [s.step_name for s in step_results],
        )
        user_text = _last_user_message(ctx)
        verdict = _grounding_verdict(user_text, final_text or "")
        recent_followup_continuity = _has_recent_followup_continuity(ctx.get("messages"), final_text or "")
        logger.info(
            "grounding_verdict corr_id=%s trace_id=%s session_id=%s anchored=%s recent_followup_continuity=%s overlap_terms=%s unrelated_markers=%s scaffolding_markers=%s",
            correlation_id,
            ctx.get("trace_id"),
            ctx.get("session_id"),
            verdict["anchored"],
            recent_followup_continuity,
            verdict["overlap_terms"],
            verdict["unrelated_markers"],
            verdict["scaffolding_markers"],
        )
        explanation_downgrade = bool((ctx.get("debug") or {}).get("downgraded_to_explanation"))
        bound_failure_signal = _extract_bound_failure_signal(step_results)
        preserve_operational_failure_text = bound_failure_signal is not None
        if not verdict["anchored"] and not recent_followup_continuity and not explanation_downgrade and not preserve_operational_failure_text:
            logger.warning(
                "grounding_guardrail_triggered corr_id=%s ask_head=%r answer_head=%r",
                correlation_id,
                _truncate_text(user_text, 220),
                _truncate_text(final_text, 220),
            )
            final_text = (
                f"I may have drifted from your request. You asked: {user_text}\n\n"
                "Could you restate the key constraint and I will answer directly from that prompt?"
            )
        elif preserve_operational_failure_text:
            logger.info(
                "grounding_guardrail_bypassed_for_operational_failure corr_id=%s reason=%s path=%s",
                correlation_id,
                bound_failure_signal.get("reason"),
                bound_failure_signal.get("path"),
            )

        runtime_policy_meta = {
            "no_write_active": no_write_active,
            "execution_blocked_reason": execution_blocked_reason,
            "operational_intent_detected": operational_intent_detected,
            "semantic_tool_preferred": bool(operational_intent_detected and preferred_operational_tool),
            "downgraded_to_explanation": bool((ctx.get("debug") or {}).get("downgraded_to_explanation")),
            "selected_tool_would_have_been": selected_tool_would_have_been,
            "bound_failure_signal": bound_failure_signal,
        }
        logger.info(
            "blocked_execution_explanation_emitted corr_id=%s emitted=%s selected_tool=%s",
            correlation_id,
            runtime_policy_meta["downgraded_to_explanation"],
            selected_tool_would_have_been,
        )

        overall_status = "success" if step_results and all(s.status == "success" for s in step_results if s) else "partial"
        overall_error: str | None = None
        if bound_failure_signal is not None:
            overall_status = "fail"
            overall_error = (
                bound_failure_signal.get("detail")
                or bound_failure_signal.get("reason")
                or "bound_capability_failed"
            )
        elif overall_status != "success":
            overall_error = step_results[-1].error if step_results else None
        return PlanExecutionResult(
            verb_name=req.verb_name,
            request_id=correlation_id,
            status=overall_status,
            blocked=False,
            blocked_reason=None,
            steps=step_results,
            mode=mode,
            final_text=final_text,
            memory_used=memory_used,
            recall_debug=recall_debug,
            metadata={"runtime_policy": runtime_policy_meta},
            error=overall_error,
        )
