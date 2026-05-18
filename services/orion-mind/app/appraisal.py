"""Mind active cognitive frontier appraisal step."""

from __future__ import annotations

import json

from pydantic import ValidationError

from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1

from .evidence import MindEvidencePackV1
from .guardrails import filter_active_frontier
from .llm_client import MindLLMClientProtocol

_APPRAISAL_SYSTEM = """You judge which semantic claims should shape the next assistant response.
Select a small active frontier (at most 6 matters). Defer or suppress the rest with reasons.
Preserve hazards. Do not let background identity dominate ordinary turns.
Include feature scores for selected matters.
Return strict JSON matching ActiveCognitiveFrontierV1 with schema_version mind.active_cognitive_frontier.v1."""


def run_active_frontier_judge(
    synthesis: SemanticSynthesisV1,
    pack: MindEvidencePackV1,
    *,
    client: MindLLMClientProtocol,
    route: str,
    model_id: str,
    max_tokens: int,
    thinking: bool = False,
) -> tuple[ActiveCognitiveFrontierV1 | None, str | None]:
    user_prompt = json.dumps(
        {
            "current_user_text": pack.current_user_text,
            "semantic_claims": [c.model_dump(mode="json") for c in synthesis.claims],
            "evidence_refs_available": [it.evidence_ref for it in pack.items],
        },
        ensure_ascii=False,
    )
    raw, err = client.request_json(
        system_prompt=_APPRAISAL_SYSTEM,
        user_prompt=user_prompt,
        route=route,
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=thinking,
    )
    if raw is None:
        return None, err or "appraisal_failed"
    try:
        frontier = ActiveCognitiveFrontierV1.model_validate(raw)
    except ValidationError as exc:
        return None, f"frontier_schema_invalid:{exc}"
    claim_ids = {c.claim_id for c in synthesis.claims}
    frontier = frontier.model_copy(
        update={
            "model_id": model_id,
            "diagnostics": frontier.diagnostics.model_copy(
                update={
                    "selected_count": len(frontier.selected),
                    "deferred_count": len(frontier.deferred),
                    "suppressed_count": len(frontier.suppressed),
                    "llm_ok": True,
                    "llm_error": None,
                }
            ),
        }
    )
    return filter_active_frontier(frontier, pack, claim_ids=claim_ids), None
