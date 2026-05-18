"""Mind semantic synthesis step."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from orion.mind.synthesis_v1 import MindEvidencePackV1, SemanticSynthesisV1
from .guardrails import filter_semantic_claims
from .llm_client import MindLLMClientProtocol

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
) -> tuple[SemanticSynthesisV1 | None, str | None]:
    raw, err = client.request_json(
        system_prompt=_SEMANTIC_SYSTEM,
        user_prompt=_pack_prompt(pack),
        route=route,
        max_tokens=max_tokens,
        temperature=0.15,
    )
    if raw is None:
        return None, err or "semantic_synthesis_failed"
    try:
        synthesis = SemanticSynthesisV1.model_validate(raw)
    except ValidationError as exc:
        return None, f"semantic_schema_invalid:{exc}"
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
    return filter_semantic_claims(synthesis, pack), None
