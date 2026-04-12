from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from jinja2 import Template

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, ServiceRef
from orion.cognition.verb_catalog import (
    VerbInfo,
    filter_allowed,
    load_verb_catalog,
    rank_verbs_for_query,
    serialize_shortlist,
)
from orion.schemas.cortex.contracts import (
    AutoDepthDecisionV1,
    CortexClientRequest,
    OutputModeDecisionV1,
)

from .output_mode_classifier import classify_output_mode
from .settings import get_settings

logger = logging.getLogger("orion.cortex.orch.router")

ORCH_INTERNAL_DENY = {
    "introspect_spark",
    "log_orion_metacognition",
    "log_collapse_mirror",
    "auto_route",
    "auto_depth_select",
}
ENGINEERING_TERMS = {
    "fix",
    "debug",
    "refactor",
    "stack trace",
    "traceback",
    "docker",
    "compose",
    "error",
    "logs",
    "log triage",
    "exception",
    "pytest",
}
ANALYSIS_TERMS = {"analyze", "analysis", "summarize", "summary", "extract", "classify", "intent"}
COUNCIL_TERMS = {"argue both sides", "deliberate", "debate", "multi-perspective", "deep deliberation", "council"}
INSTRUCTION_TERMS = {"how do", "how to", "instructions", "deploy", "guide", "tutorial", "setup", "walkthrough"}
MENU_HINT_TERMS = {"options", "choose", "proceed", "deep dive", "axes", "first one", "second one", "third one"}
TOPIC_FILLER_PREFIXES = ("hm ", "hmm ", "uh ", "um ", "okay ", "ok ", "sure ", "let's do ", "lets do ", "deep dive on ")
ORDINAL_SELECTION_TERMS = {
    "the first one",
    "first one",
    "that first one",
    "the second one",
    "second one",
    "that second one",
    "the third one",
    "third one",
    "that third one",
}

# Substrings that indicate self-coaching / wellness free-form, not operational "how do I …" runbooks.
# Keeps auto-routed prompts on the stable brain/chat_general + verb-runtime path instead of agent_runtime
# (planner/agent-chain can stall or exceed Hub waits on gateway:result).
PERSONAL_COACHING_HINTS = (
    "motivat",
    "better version",
    "myself",
    "procrastin",
    "burnout",
    "feel stuck",
    " i feel ",
    "i'm stuck",
    "im stuck",
    "anxiety",
    "depress",
    "self-care",
    "self care",
    "mental health",
    "can't focus",
    "cant focus",
    "lazy",
    "habit",
    " self-discipline",
    " self discipline",
    "personal discipline",
)


def _looks_like_personal_coaching(text: str) -> bool:
    t = " " + " ".join(str(text or "").lower().split()) + " "
    return any(hint in t for hint in PERSONAL_COACHING_HINTS)


@dataclass(frozen=True)
class RoutedRequest:
    request: CortexClientRequest
    decision: AutoDepthDecisionV1
    output_mode_decision: OutputModeDecisionV1


class DecisionRouter:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.settings = get_settings()

    def _user_text(self, req: CortexClientRequest) -> str:
        return " ".join(
            part for part in [req.context.raw_user_text or "", req.context.user_message or ""] if part
        ).strip()

    def build_shortlist(self, req: CortexClientRequest, *, k: int = 10) -> list[VerbInfo]:
        catalog = load_verb_catalog()
        catalog = filter_allowed(catalog, allow_categories={"cognition"}, denylist_names=ORCH_INTERNAL_DENY)
        shortlist = rank_verbs_for_query(catalog, self._user_text(req), k=k)
        return shortlist

    def _last_assistant_message(self, req: CortexClientRequest) -> str:
        for msg in reversed(req.context.messages or []):
            if str(getattr(msg, "role", "") or "").strip().lower() != "assistant":
                continue
            content = str(getattr(msg, "content", "") or "").strip()
            if content:
                return content
        return ""

    def _extract_menu_options(self, assistant_text: str) -> list[str]:
        text = str(assistant_text or "")
        options: set[str] = set()
        # Prefer explicit parenthetical lists like "(Adaptive Learning, Action System, or Mesh Continuity)".
        for segment in re.findall(r"\(([^)]{8,240})\)", text):
            parts = re.split(r",|\bor\b|\band\b|/|\|", segment, flags=re.IGNORECASE)
            for part in parts:
                candidate = " ".join(part.strip().lower().split())
                if len(candidate) >= 4 and len(candidate.split()) <= 5 and re.search(r"[a-z]", candidate):
                    options.add(candidate)
        # Also pick title-like bullet items.
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith(("-", "*", "•")):
                continue
            candidate = re.sub(r"^[\-\*\•]\s*", "", line)
            candidate = re.split(r"[:(]", candidate, maxsplit=1)[0].strip().lower()
            if len(candidate) >= 4 and len(candidate.split()) <= 6:
                options.add(candidate)
        return sorted(options)

    def _looks_like_menu_turn(self, assistant_text: str) -> bool:
        text = str(assistant_text or "").lower()
        if not text:
            return False
        has_bullets = any(mark in text for mark in ("\n- ", "\n* ", "\n• "))
        has_hint = any(term in text for term in MENU_HINT_TERMS)
        return has_bullets and has_hint

    def _looks_like_topic_selection_reply(self, user_text: str, menu_options: list[str]) -> bool:
        text = " ".join(str(user_text or "").lower().split())
        if not text or len(text) > 80:
            return False
        normalized = re.sub(r"^[^a-z0-9]+", "", text)
        normalized = re.sub(r"^(hm+|uh+|um+)\b[\s,.:;-]*", "", normalized).strip()
        for prefix in TOPIC_FILLER_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
        if any(term in normalized for term in ORDINAL_SELECTION_TERMS):
            return True
        for option in menu_options:
            if option and option in normalized:
                return True
        return False

    def _normalize_menu_topic_selection(self, user_text: str, menu_options: list[str]) -> str | None:
        text = " ".join(str(user_text or "").lower().split())
        if not text:
            return None
        normalized = re.sub(r"^[^a-z0-9]+", "", text)
        normalized = re.sub(r"^(hm+|uh+|um+)\b[\s,.:;-]*", "", normalized).strip()
        for prefix in TOPIC_FILLER_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                break
        for option in menu_options:
            if option and option in normalized:
                return option.title()
        if "first one" in normalized and menu_options:
            return menu_options[0].title()
        if "second one" in normalized and len(menu_options) >= 2:
            return menu_options[1].title()
        if "third one" in normalized and len(menu_options) >= 3:
            return menu_options[2].title()
        return None

    def heuristic_router(self, req: CortexClientRequest, *, shortlist: list[VerbInfo]) -> AutoDepthDecisionV1:
        text = self._user_text(req).lower()
        if any(term in text for term in COUNCIL_TERMS):
            return AutoDepthDecisionV1(execution_depth=3, primary_verb=None, confidence=0.82, reason="heuristic:council", source="heuristic")
        if "```" in text or any(term in text for term in ENGINEERING_TERMS):
            return AutoDepthDecisionV1(execution_depth=2, primary_verb=None, confidence=0.85, reason="heuristic:engineering", source="heuristic")
        if "planner" in text and ("agent chain" in text or "agent_chain" in text):
            return AutoDepthDecisionV1(
                execution_depth=2,
                primary_verb=None,
                confidence=0.86,
                reason="heuristic:planner_agent_chain_design",
                source="heuristic",
            )
        if any(term in text for term in INSTRUCTION_TERMS):
            return AutoDepthDecisionV1(execution_depth=2, primary_verb=None, confidence=0.84, reason="heuristic:instruction", source="heuristic")
        if any(term in text for term in ANALYSIS_TERMS):
            primary = shortlist[0].name if shortlist else "analyze_text"
            return AutoDepthDecisionV1(execution_depth=1, primary_verb=primary, confidence=0.79, reason="heuristic:single_verb", source="heuristic")
        if len(text) <= 80 and "?" in text:
            return AutoDepthDecisionV1(execution_depth=0, primary_verb=None, confidence=0.75, reason="heuristic:simple_question", source="heuristic")
        return AutoDepthDecisionV1(execution_depth=0, primary_verb=None, confidence=0.61, reason="heuristic:default", source="heuristic")

    async def llm_router(self, req: CortexClientRequest, *, correlation_id: str, source: ServiceRef, shortlist: list[VerbInfo]) -> AutoDepthDecisionV1:
        prompt = self._build_prompt(req, shortlist=shortlist)
        payload = ChatRequestPayload(
            route="chat",
            messages=[{"role": "user", "content": prompt}],
            raw_user_text=req.context.raw_user_text or req.context.user_message,
            options={
                "temperature": 0.0,
                "max_tokens": 220,
                "stream": False,
                "response_format": {"type": "json_object"},
            },
            user_id=req.context.user_id,
            session_id=req.context.session_id,
        )
        return await asyncio.wait_for(
            self._rpc_llm(payload=payload, correlation_id=correlation_id, source=source),
            timeout=5.0,
        )

    async def route(self, req: CortexClientRequest, *, correlation_id: str, source: ServiceRef) -> RoutedRequest:
        prior_assistant = self._last_assistant_message(req)
        menu_options = self._extract_menu_options(prior_assistant)
        if self._looks_like_menu_turn(prior_assistant) and self._looks_like_topic_selection_reply(self._user_text(req), menu_options):
            selected_topic = self._normalize_menu_topic_selection(self._user_text(req), menu_options)
            decision = AutoDepthDecisionV1(
                execution_depth=0,
                primary_verb=None,
                confidence=0.99,
                reason="heuristic:menu_topic_selection_followup",
                source="heuristic",
            )
            rewritten = req.model_copy(deep=True)
            rewritten.mode = "brain"
            rewritten.verb = "chat_general"
            rewritten.options["execution_depth"] = 0
            rewritten.options["route_intent"] = "none"
            output_mode_decision = classify_output_mode(self._user_text(req))
            rewritten.options["output_mode"] = output_mode_decision.output_mode
            rewritten.options["response_profile"] = output_mode_decision.response_profile
            rewritten.options["output_mode_decision"] = output_mode_decision.model_dump()
            rewritten.options["menu_topic_selection"] = {
                "enabled": True,
                "matched_options": menu_options[:6],
                "selected_topic": selected_topic,
            }
            return RoutedRequest(request=rewritten, decision=decision, output_mode_decision=output_mode_decision)

        shortlist = self.build_shortlist(req)
        if self.settings.auto_router_llm_enabled:
            try:
                decision = await self.llm_router(req, correlation_id=correlation_id, source=source, shortlist=shortlist)
            except Exception as exc:
                logger.warning("auto_depth llm failed corr=%s err=%s", correlation_id, exc)
                fallback = self.heuristic_router(req, shortlist=shortlist)
                decision = fallback.model_copy(update={"source": "fallback", "reason": "fallback:llm_failure"})
        else:
            decision = self.heuristic_router(req, shortlist=shortlist)

        clamped = self._clamp_decision(decision, shortlist=shortlist)
        user_text = self._user_text(req)
        if clamped.execution_depth >= 1 and _looks_like_personal_coaching(user_text):
            clamped = AutoDepthDecisionV1(
                execution_depth=0,
                primary_verb=None,
                confidence=min(0.9, float(clamped.confidence)),
                reason="stabilize:personal_coaching_chat_general",
                source="heuristic",
            )
        rewritten = req.model_copy(deep=True)
        rewritten.options["execution_depth"] = clamped.execution_depth

        # Output mode classification for delivery-oriented routing
        output_mode_decision = classify_output_mode(self._user_text(req))
        rewritten.options["output_mode"] = output_mode_decision.output_mode
        rewritten.options["response_profile"] = output_mode_decision.response_profile
        rewritten.options["output_mode_decision"] = output_mode_decision.model_dump()

        if clamped.execution_depth == 1:
            rewritten.mode = "brain"
            rewritten.verb = clamped.primary_verb or "analyze_text"
        elif clamped.execution_depth == 2:
            rewritten.mode = "agent"
            rewritten.verb = "agent_runtime"
        elif clamped.execution_depth == 3:
            rewritten.mode = "council"
            rewritten.verb = "council_runtime"
        else:
            rewritten.mode = "brain"
            rewritten.verb = "chat_general"
        return RoutedRequest(request=rewritten, decision=clamped, output_mode_decision=output_mode_decision)

    def _clamp_decision(self, decision: AutoDepthDecisionV1, *, shortlist: list[VerbInfo]) -> AutoDepthDecisionV1:
        depth = max(0, min(3, int(decision.execution_depth)))
        allowed = {v.name for v in shortlist}
        primary_verb = decision.primary_verb if depth == 1 and decision.primary_verb in allowed else None
        if depth == 1 and not primary_verb:
            primary_verb = shortlist[0].name if shortlist else "analyze_text"
        if depth != 1:
            primary_verb = None
        return AutoDepthDecisionV1(
            execution_depth=depth,
            primary_verb=primary_verb,
            confidence=max(0.0, min(1.0, float(decision.confidence))),
            reason=decision.reason or "clamped",
            source=decision.source,
        )

    async def _rpc_llm(self, *, payload: ChatRequestPayload, correlation_id: str, source: ServiceRef) -> AutoDepthDecisionV1:
        reply_channel = f"{self.settings.auto_router_llm_reply_prefix}:{uuid4()}"
        env = BaseEnvelope(
            kind="llm.chat.request",
            source=source,
            correlation_id=correlation_id,
            reply_to=reply_channel,
            payload=payload.model_dump(mode="json"),
        )
        message = await self.bus.rpc_request(
            self.settings.auto_router_llm_request_channel,
            env,
            reply_channel=reply_channel,
            timeout_sec=5.0,
        )
        decoded = self.bus.codec.decode(message.get("data"))
        if not decoded.ok:
            raise RuntimeError(decoded.error or "llm_decode_failed")
        response_payload = decoded.envelope.payload if isinstance(decoded.envelope.payload, dict) else {}
        text = str(response_payload.get("content") or response_payload.get("text") or "").strip()
        if not text:
            raw = response_payload.get("raw") or {}
            text = str(raw.get("content") or raw.get("text") or "").strip()
        data = json.loads(text)
        return AutoDepthDecisionV1.model_validate(data)

    def _build_prompt(self, req: CortexClientRequest, *, shortlist: list[VerbInfo]) -> str:
        prompt_path = Path(__file__).resolve().parents[3] / "orion" / "cognition" / "prompts" / "auto_depth_select_prompt.j2"
        template = Template(prompt_path.read_text(encoding="utf-8"))
        return template.render(
            verb_shortlist=serialize_shortlist(shortlist, max_chars=1800),
            user_text=self._user_text(req),
        )
