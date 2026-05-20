"""Mind stance handoff step — ChatStanceBrief-compatible payload."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1
from orion.mind.validation import validate_merged_stance_brief_optional

from .evidence import MindEvidencePackV1
from .llm_client import MindLLMClientProtocol
from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry, utc_now_iso

_VALID_CONVERSATION_FRAMES: frozenset[str] = frozenset(
    {
        "technical",
        "planning",
        "reflective",
        "playful_relational",
        "identity_emergence",
        "mixed",
    }
)
_VALID_TASK_MODES: frozenset[str] = frozenset(
    {
        "direct_response",
        "triage",
        "technical_collaboration",
        "identity_dialogue",
        "reflective_dialogue",
        "playful_exchange",
        "mixed",
    }
)
_VALID_IDENTITY_SALIENCE: frozenset[str] = frozenset({"low", "medium", "high"})
_CONVERSATION_FRAME_ALIASES: dict[str, str] = {
    "smoketest": "mixed",
    "casual": "playful_relational",
    "relational": "playful_relational",
    "general": "mixed",
}
_TASK_MODE_ALIASES: dict[str, str] = {
    "direct_answer": "direct_response",
    "answer_directly": "direct_response",
}
_IDENTITY_SALIENCE_ALIASES: dict[str, str] = {
    "neutral": "low",
    "minimal": "low",
    "none": "low",
}

_STANCE_SYSTEM = """You convert the active cognitive frontier into a compact ChatStanceBrief-compatible stance_payload.
You are NOT writing the final user reply.
Respect the current user message first.
Return strict JSON only (no prose) with these exact enum values:
- conversation_frame: technical | planning | reflective | playful_relational | identity_emergence | mixed
- task_mode: direct_response | triage | technical_collaboration | identity_dialogue | reflective_dialogue | playful_exchange | mixed
- identity_salience: low | medium | high
Also include: user_intent, self_relevance, juniper_relevance, response_priorities, response_hazards, answer_strategy, stance_summary.
Example for a simple operational turn:
{"conversation_frame":"mixed","task_mode":"direct_response","identity_salience":"low","user_intent":"User is running a smoketest.","self_relevance":"Confirm receipt without over-interpreting.","juniper_relevance":"Stay concise and operational.","response_priorities":["confirm receipt"],"response_hazards":["do not invent context"],"answer_strategy":"DirectAnswer","stance_summary":"Operational smoketest turn."}"""


def try_coerce_stance_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Coerce common invalid LLM enum strings before ChatStanceBrief validation."""
    out = dict(payload)
    changed = False

    frame = str(out.get("conversation_frame") or "").strip().lower()
    if frame not in _VALID_CONVERSATION_FRAMES:
        coerced = _CONVERSATION_FRAME_ALIASES.get(frame, "mixed")
        out["conversation_frame"] = coerced
        changed = True

    task_mode = str(out.get("task_mode") or "").strip().lower()
    if task_mode not in _VALID_TASK_MODES:
        coerced = _TASK_MODE_ALIASES.get(task_mode, "direct_response")
        out["task_mode"] = coerced
        changed = True

    salience = str(out.get("identity_salience") or "").strip().lower()
    if salience not in _VALID_IDENTITY_SALIENCE:
        coerced = _IDENTITY_SALIENCE_ALIASES.get(salience, "low")
        out["identity_salience"] = coerced
        changed = True

    return out, changed


def _minimal_stance_from_pack(pack: MindEvidencePackV1) -> dict[str, Any]:
    ut = (pack.current_user_text or "").strip() or "(empty turn)"
    return {
        "conversation_frame": "mixed",
        "task_mode": "direct_response",
        "identity_salience": "medium",
        "user_intent": ut[:500],
        "self_relevance": "Mind synthesis handoff seed.",
        "juniper_relevance": "Honor the latest user turn without inventing context.",
        "answer_strategy": "DirectAnswer",
        "stance_summary": "Mind stance handoff from active frontier.",
        "response_priorities": ["answer the user turn first"],
        "response_hazards": [],
    }


def run_stance_handoff(
    frontier: ActiveCognitiveFrontierV1,
    synthesis: SemanticSynthesisV1,
    pack: MindEvidencePackV1,
    *,
    client: MindLLMClientProtocol,
    route: str,
    model_id: str,
    max_tokens: int,
    context: MindLLMRequestContext | None = None,
    timeout_sec: float | None = None,
) -> tuple[dict[str, Any], str | None, MindPhaseTelemetry]:
    user_prompt = json.dumps(
        {
            "current_user_text": pack.current_user_text,
            "selected_frontier": [m.model_dump(mode="json") for m in frontier.selected],
            "hazards": frontier.hazards,
            "response_directives": frontier.response_directives,
            "claims": [c.model_dump(mode="json") for c in synthesis.claims[:12]],
        },
        ensure_ascii=False,
    )
    started = utc_now_iso()
    t0 = time.perf_counter()
    raw, err, meta = client.request_json(
        system_prompt=_STANCE_SYSTEM,
        user_prompt=user_prompt,
        route=route,
        max_tokens=max_tokens,
        temperature=0.2,
        context=context,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = MindPhaseTelemetry(
        phase_name="stance_handoff",
        route=route,
        model=str(meta.get("model_used") or model_id),
        started_at=started,
        elapsed_ms=elapsed_ms,
        ok=raw is not None and err is None,
        parse_ok=raw is not None,
        error=err,
        token_usage=dict(meta.get("usage") or {}),
    )
    if raw is None:
        base = _minimal_stance_from_pack(pack)
        telemetry.validation_ok = False
        return base, err or "stance_handoff_failed", telemetry
    payload = raw.get("stance_payload") if isinstance(raw.get("stance_payload"), dict) else raw
    if not isinstance(payload, dict):
        telemetry.validation_ok = False
        return _minimal_stance_from_pack(pack), "stance_payload_not_object", telemetry
    merged = _minimal_stance_from_pack(pack)
    merged.update(payload)
    merged, _ = try_coerce_stance_payload(merged)
    valid, validation_err = validate_merged_stance_brief_optional(merged)
    if valid is None:
        telemetry.validation_ok = False
        return merged, f"stance_brief_invalid:{validation_err}", telemetry
    telemetry.validation_ok = True
    return valid.model_dump(mode="json"), None, telemetry
