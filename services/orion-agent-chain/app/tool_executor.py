from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatResultPayload, ServiceRef
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, CortexClientResult, LLMMessage, RecallDirective

from .actions_skill_registry import ActionsSkillRegistry
from .capability_bridge import normalize_capability_observation, resolve_capability_decision
from .settings import settings

logger = logging.getLogger("agent-chain.tool-exec")


class ToolExecutor:
    """Executes LLM-backed cognition verbs locally from agent-chain."""

    def __init__(self, bus: OrionBusAsync, *, base_dir: str | None = None) -> None:
        self.bus = bus
        self.base_dir = Path(base_dir or settings.cognition_base_dir)
        self.verbs_dir = self.base_dir / "verbs"
        self.prompts_dir = self.base_dir / "prompts"
        self.renderer = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._actions_registry = ActionsSkillRegistry(verbs_dir=self.verbs_dir)

    def _load_verb(self, tool_id: str) -> Dict[str, Any]:
        path = self.verbs_dir / f"{tool_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Verb config not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid verb config for {tool_id}")
        return data

    def _resolve_prompt_template(self, verb_cfg: Dict[str, Any]) -> str | None:
        direct = verb_cfg.get("prompt_template")
        if isinstance(direct, str) and direct:
            return direct

        for key in ("steps", "plan"):
            items = verb_cfg.get(key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and isinstance(item.get("prompt_template"), str):
                    return item["prompt_template"]
        return None


    def _normalize_inputs(self, tool_id: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        ctx = dict(tool_input or {})

        raw_text = ctx.get("text")
        if isinstance(raw_text, str):
            text_value = raw_text
        elif raw_text is None:
            text_value = json.dumps(ctx, ensure_ascii=False) if ctx else ""
        else:
            text_value = str(raw_text)
        ctx["text"] = text_value

        fallback_map = {
            "triage": "request",
            "plan_action": "goal",
            "goal_formulate": "intention",
            "summarize_context": "context_raw",
            "tag_enrich": "fragment",
            "pattern_detect": "fragments",
            "evaluate": "output",
            "assess_risk": "scenario",
            # Delivery verbs
            "answer_direct": "request",
            "finalize_response": "original_request",
            "write_guide": "request",
            "write_tutorial": "request",
            "write_runbook": "request",
            "write_recommendation": "request",
            "compare_options": "request",
            "synthesize_patterns": "request",
            "generate_code_scaffold": "request",
        }
        target_field = fallback_map.get(tool_id)
        if target_field and target_field not in ctx:
            ctx[target_field] = text_value

        return ctx

    def _render_prompt(self, template_name: str | None, tool_input: Dict[str, Any]) -> str:
        ctx = self._normalize_inputs("", tool_input)
        if not template_name:
            return ctx["text"]
        try:
            return self.renderer.get_template(template_name).render(**ctx)
        except Exception:
            logger.warning("Template render failed for %s; falling back to json context", template_name)
            return ctx["text"]

    async def execute_llm_verb(self, tool_id: str, tool_input: Dict[str, Any], *, parent_correlation_id: str | None = None) -> Dict[str, Any]:
        verb_cfg = self._load_verb(tool_id)
        execution_mode = str(verb_cfg.get("execution_mode") or "").strip().lower()
        if execution_mode == "capability_backed" or bool(verb_cfg.get("requires_capability_selector")):
            return await self._execute_capability_backed_verb(
                verb_cfg=verb_cfg,
                tool_id=tool_id,
                tool_input=tool_input,
                parent_correlation_id=parent_correlation_id,
            )
        services = verb_cfg.get("services") or []
        if isinstance(services, list) and services and "LLMGatewayService" not in services:
            raise RuntimeError(f"Tool {tool_id} is not LLM-only and cannot be executed in agent-chain delegate mode")

        normalized_input = self._normalize_inputs(tool_id, tool_input)
        template_name = self._resolve_prompt_template(verb_cfg)
        prompt = self._render_prompt(template_name, normalized_input)

        corr = str(uuid4())
        reply_channel = f"{settings.llm_reply_prefix}:{corr}"
        env = BaseEnvelope(
            kind="llm.chat.request",
            source=ServiceRef(name=settings.service_name, version=settings.service_version),
            correlation_id=corr,
            reply_to=reply_channel,
            payload={
                "messages": [{"role": "user", "content": prompt}],
                "raw_user_text": normalized_input.get("text") or "",
                "route": "agent",
                "options": {"temperature": 0.2},
            },
        )

        logger.info(
            "[agent-chain] tool_exec tool=%s parent=%s llm_reply=%s",
            tool_id,
            parent_correlation_id,
            reply_channel,
        )
        msg = await self.bus.rpc_request(
            settings.llm_request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.default_timeout_seconds),
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"LLM decode failed: {decoded.error}")

        payload = decoded.envelope.payload or {}
        chat = ChatResultPayload.model_validate(payload)
        return {
            "llm_output": chat.text,
            "spark_meta": chat.spark_meta,
            "model_used": chat.model_used,
        }

    async def _execute_capability_backed_verb(
        self,
        *,
        verb_cfg: Dict[str, Any],
        tool_id: str,
        tool_input: Dict[str, Any],
        parent_correlation_id: str | None,
    ) -> Dict[str, Any]:
        decision = resolve_capability_decision(
            verb=tool_id,
            preferred_skill_families=list(verb_cfg.get("preferred_skill_families") or []),
            registry=self._actions_registry,
        )
        if not decision.selected_skill:
            return normalize_capability_observation(
                decision=decision,
                execution_summary="No compatible orion-actions skill available.",
                raw_payload={"status": "unavailable"},
            )

        corr = str(uuid4())
        reply_channel = f"orion:agent-chain:capability:reply:{corr}"
        normalized_input = self._normalize_inputs(tool_id, tool_input)
        req = CortexClientRequest(
            mode="brain",
            route_intent="none",
            verb=decision.selected_skill,
            packs=["executive_pack"],
            options={"policy_dispatch_only": True, "source": "agent_chain_capability_bridge"},
            recall=RecallDirective(enabled=False, required=False, mode="hybrid"),
            context=CortexClientContext(
                messages=[LLMMessage(role="user", content=str(normalized_input.get("text") or ""))],
                raw_user_text=str(normalized_input.get("text") or ""),
                user_message=str(normalized_input.get("text") or ""),
                session_id="agent-chain-capability",
                user_id="agent-chain",
                trace_id=parent_correlation_id,
                metadata={
                    "capability_decision": decision.model_dump(mode="json"),
                    "capability_bridge": True,
                },
            ),
        )
        env = BaseEnvelope(
            kind="cortex.orch.request",
            source=ServiceRef(name=settings.service_name, version=settings.service_version),
            correlation_id=corr,
            reply_to=reply_channel,
            payload=req.model_dump(mode="json"),
        )
        msg = await self.bus.rpc_request(
            "orion:cortex:request",
            env,
            reply_channel=reply_channel,
            timeout_sec=float(settings.default_timeout_seconds),
        )
        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            raise RuntimeError(f"Capability bridge decode failed: {decoded.error}")
        payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        result = CortexClientResult.model_validate(payload)
        summary = f"Executed {decision.selected_skill}: status={result.status} ok={result.ok}"
        return normalize_capability_observation(
            decision=decision,
            execution_summary=summary,
            raw_payload={"status": result.status, "ok": result.ok, "final_text": result.final_text},
        )
