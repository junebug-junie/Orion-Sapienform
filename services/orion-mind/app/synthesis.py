"""Mind semantic synthesis step."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import MindEvidencePackV1, SemanticSynthesisV1
from .guardrails import filter_semantic_claims
from .llm_client import MindLLMClientProtocol
from .llm_context import MindLLMRequestContext
from .phase_telemetry import MindPhaseTelemetry

_SEMANTIC_SYSTEM = """You extract grounded semantic claims from evidence for an internal cognition organ.
You are NOT writing the user-facing reply.
Every claim MUST cite evidence_refs from the provided evidence pack only.
Labels must be semantic meanings, never source tags like identity_yaml, projection, autonomy, or social_bridge.
Identity background is not primary evidence unless the current turn explicitly involves identity, selfhood, role boundaries, or Orion/Juniper relationship framing.
Prefer uncertainty over invention.
Return strict JSON matching SemanticSynthesisV1 shape with schema_version mind.semantic_synthesis.v1."""


def _pack_prompt(pack: MindEvidencePackV1) -> str:
    items = [
        {
            "evidence_ref": it.evidence_ref,
            "source_kind": it.source_kind,
            "label": it.label,
            "text": it.text,
            "metadata": it.metadata,
        }
        for it in pack.items
    ]
    return json.dumps(
        {"current_user_text": pack.current_user_text, "evidence_items": items},
        ensure_ascii=False,
    )


def run_semantic_synthesis(
    pack: MindEvidencePackV1,
    *,
    client: MindLLMClientProtocol,
    route: str,
    model_id: str,
    max_tokens: int,
    context: MindLLMRequestContext | None = None,
    timeout_sec: float | None = None,
) -> tuple[SemanticSynthesisV1 | None, str | None, MindPhaseTelemetry]:
    from .phase_telemetry import utc_now_iso

    started = utc_now_iso()
    t0 = time.perf_counter()
    raw, err, meta = client.request_json(
        system_prompt=_SEMANTIC_SYSTEM,
        user_prompt=_pack_prompt(pack),
        route=route,
        max_tokens=max_tokens,
        temperature=0.15,
        context=context,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = MindPhaseTelemetry(
        phase_name="semantic_synthesis",
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
        return None, err or "semantic_synthesis_failed", telemetry
    try:
        synthesis = SemanticSynthesisV1.model_validate(raw)
    except ValidationError as exc:
        telemetry.validation_ok = False
        return None, f"semantic_schema_invalid:{exc}", telemetry
    synthesis = synthesis.model_copy(
        update={
            "model_id": model_id,
            "extraction_mode": "llm",
            "diagnostics": synthesis.diagnostics.model_copy(
                update={
                    "evidence_item_count": len(pack.items),
                    "projection_item_count_seen": int(pack.diagnostics.get("projection_item_count") or 0),
                    "recall_fragments_seen": int(pack.diagnostics.get("recall_fragment_count") or 0),
                    "autonomy_fields_seen": int(pack.diagnostics.get("autonomy_fields_seen") or 0),
                    "social_fields_seen": int(pack.diagnostics.get("social_fields_seen") or 0),
                    "llm_ok": True,
                    "llm_error": None,
                }
            ),
        }
    )
    filtered = filter_semantic_claims(synthesis, pack)
    telemetry.validation_ok = bool(filtered.claims)
    return filtered, None, telemetry
