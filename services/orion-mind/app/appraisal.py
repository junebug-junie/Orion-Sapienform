"""Mind active cognitive frontier appraisal step."""

from __future__ import annotations

import json
import time

from pydantic import ValidationError

from orion.mind.synthesis_v1 import ActiveCognitiveFrontierV1, SemanticSynthesisV1

from .evidence import MindEvidencePackV1
from .guardrails import filter_active_frontier
from .llm_client import MindLLMClientProtocol
from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry, utc_now_iso

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
    context: MindLLMRequestContext | None = None,
    timeout_sec: float | None = None,
) -> tuple[ActiveCognitiveFrontierV1 | None, str | None, MindPhaseTelemetry]:
    user_prompt = json.dumps(
        {
            "current_user_text": pack.current_user_text,
            "semantic_claims": [c.model_dump(mode="json") for c in synthesis.claims],
            "evidence_refs_available": [it.evidence_ref for it in pack.items],
        },
        ensure_ascii=False,
    )
    started = utc_now_iso()
    t0 = time.perf_counter()
    raw, err, meta = client.request_json(
        system_prompt=_APPRAISAL_SYSTEM,
        user_prompt=user_prompt,
        route=route,
        max_tokens=max_tokens,
        temperature=0.1,
        thinking=thinking,
        context=context,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = MindPhaseTelemetry(
        phase_name="active_frontier_judge",
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
        return None, err or "appraisal_failed", telemetry
    try:
        frontier = ActiveCognitiveFrontierV1.model_validate(raw)
    except ValidationError as exc:
        telemetry.validation_ok = False
        return None, f"frontier_schema_invalid:{exc}", telemetry
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
    filtered = filter_active_frontier(frontier, pack, claim_ids=claim_ids)
    telemetry.validation_ok = bool(filtered.selected)
    return filtered, None, telemetry
