"""Mind stance handoff step — ChatStanceBrief-compatible payload."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1
from orion.mind.validation import validate_merged_stance_brief_optional

from .evidence import MindEvidencePackV1
from .llm_client import MindLLMClientProtocol

_STANCE_SYSTEM = """You convert the active cognitive frontier into a compact ChatStanceBrief-compatible stance_payload.
You are NOT writing the final user reply.
Respect the current user message first.
Produce JSON with fields: conversation_frame, task_mode, identity_salience, user_intent, self_relevance,
juniper_relevance, active_identity_facets, active_relationship_facets, reflective_themes, active_tensions,
response_priorities, response_hazards, answer_strategy, stance_summary.
Use only supported conversation_frame and task_mode enum values from ChatStanceBrief."""


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
) -> tuple[dict[str, Any], str | None]:
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
    raw, err = client.request_json(
        system_prompt=_STANCE_SYSTEM,
        user_prompt=user_prompt,
        route=route,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    if raw is None:
        base = _minimal_stance_from_pack(pack)
        return base, err or "stance_handoff_failed"
    payload = raw.get("stance_payload") if isinstance(raw.get("stance_payload"), dict) else raw
    if not isinstance(payload, dict):
        return _minimal_stance_from_pack(pack), "stance_payload_not_object"
    merged = _minimal_stance_from_pack(pack)
    merged.update(payload)
    valid, validation_err = validate_merged_stance_brief_optional(merged)
    if valid is None:
        return merged, f"stance_brief_invalid:{validation_err}"
    return valid.model_dump(mode="json"), None
