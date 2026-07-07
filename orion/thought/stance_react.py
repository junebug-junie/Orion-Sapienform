from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import uuid4

from orion.thought.json_extract import extract_first_json_object_text
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import StanceHarnessSliceV1, StanceReactRequestV1, ThoughtEventV1, HubAssociationBundleV1
from orion.thought.coalition import (
    align_evidence_refs_to_coalition,
    coalition_ids_from_association,
)
from orion.thought.policy_refusal import evaluate_thought_disposition
from orion.thought.stance_quality import enforce_thought_stance_quality


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _coerce_required_str(value: Any, *, default: str) -> str:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or default
    text = str(value).strip()
    return text or default


def normalize_stance_react_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM JSON mistakes before ThoughtEventV1 validation."""
    data = dict(raw)
    data["event_id"] = _coerce_required_str(data.get("event_id"), default=str(uuid4()))
    data["created_at"] = _coerce_required_str(
        data.get("created_at"),
        default=datetime.now(timezone.utc).isoformat(),
    )
    data["profile"] = _coerce_required_str(data.get("profile"), default="stance_react")
    data["producer"] = _coerce_required_str(data.get("producer"), default="stance_react_v1")
    data["llm_profile"] = _coerce_required_str(data.get("llm_profile"), default="brain")
    data["imperative"] = _coerce_required_str(data.get("imperative"), default="")
    data["tone"] = _coerce_required_str(data.get("tone"), default="neutral")
    data["strain_refs"] = _coerce_str_list(data.get("strain_refs"))
    data["evidence_refs"] = _coerce_str_list(data.get("evidence_refs"))
    data["disposition_reasons"] = _coerce_str_list(data.get("disposition_reasons"))

    slice_raw = data.get("stance_harness_slice")
    if not isinstance(slice_raw, dict):
        slice_raw = {}
    sl = dict(slice_raw)
    sl["task_mode"] = _coerce_required_str(sl.get("task_mode"), default="mixed")
    sl["conversation_frame"] = _coerce_required_str(sl.get("conversation_frame"), default="mixed")
    sl["answer_strategy"] = _coerce_required_str(sl.get("answer_strategy"), default="direct")
    sl["interaction_regime"] = _coerce_optional_str(sl.get("interaction_regime"))
    sl["companion_closing_move"] = _coerce_optional_str(sl.get("companion_closing_move"))
    sl["response_priorities"] = _coerce_str_list(sl.get("response_priorities"))
    sl["response_hazards"] = _coerce_str_list(sl.get("response_hazards"))
    data["stance_harness_slice"] = sl
    return data


def build_stance_react_failure_thought(
    *,
    correlation_id: str,
    session_id: str | None,
    reason: str,
) -> ThoughtEventV1:
    """Valid defer ThoughtEventV1 for RPC reply channels that require ThoughtEventV1."""
    return ThoughtEventV1(
        event_id=str(uuid4()),
        correlation_id=correlation_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        imperative="",
        tone="neutral",
        strain_refs=[],
        evidence_refs=[],
        disposition="defer",
        disposition_reasons=[reason[:240]],
        stance_harness_slice=StanceHarnessSliceV1(
            task_mode="mixed",
            conversation_frame="mixed",
            answer_strategy="defer",
        ),
    )


def decode_stance_react_json(raw: str) -> dict[str, Any]:
    """Parse ThoughtEventV1 JSON from model text, tolerating markdown wrappers."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("stance_react payload is empty")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    blob = extract_first_json_object_text(text)
    if blob:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"stance_react payload is not a JSON object: {text[:160]!r}")


def slim_association_for_prompt(association: HubAssociationBundleV1) -> dict[str, Any]:
    broadcast = association.broadcast
    return {
        "correlation_id": association.correlation_id,
        "broadcast_stale": association.broadcast_stale,
        "read_source": association.read_source,
        "attended_node_ids": list(broadcast.attended_node_ids) if broadcast else [],
        "open_loop_ids": [loop.id for loop in broadcast.frame.open_loops] if broadcast else [],
    }


def slim_repair_bundle_for_prompt(bundle: TurnAppraisalBundleV1 | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    paradigms = {
        name: {
            "level": slice_.level,
            "confidence": slice_.confidence,
            "contract_delta": slice_.contract_delta,
        }
        for name, slice_ in bundle.paradigms.items()
    }
    slim: dict[str, Any] = {
        "correlation_id": bundle.correlation_id,
        "paradigms": paradigms,
    }
    contract = (bundle.metadata_attachments or {}).get("repair_pressure_contract")
    if isinstance(contract, dict) and contract:
        slim["repair_pressure_contract"] = contract
    return slim


def parse_stance_react_payload(
    raw: dict[str, Any] | str,
    *,
    correlation_id: str | None = None,
    session_id: str | None = None,
) -> ThoughtEventV1:
    """Validate JSON from cortex stance_react step into ThoughtEventV1."""
    if isinstance(raw, str):
        raw = decode_stance_react_json(raw)
    if not isinstance(raw, dict):
        raise TypeError("stance_react payload must be a JSON object")
    # The grounding capsule is assembled deterministically in cortex-exec and mapped
    # on from result metadata — never authored by the stance LLM. Drop any model-supplied
    # value so a hallucinated capsule cannot masquerade as grounded self-context.
    raw.pop("grounding_capsule", None)
    if correlation_id:
        raw.setdefault("correlation_id", correlation_id)
    if session_id is not None:
        raw.setdefault("session_id", session_id)
    return ThoughtEventV1.model_validate(normalize_stance_react_raw(raw))


def apply_stance_react_pipeline(
    thought: ThoughtEventV1,
    request: StanceReactRequestV1,
) -> ThoughtEventV1:
    coalition_ids = coalition_ids_from_association(request.association)
    aligned = align_evidence_refs_to_coalition(thought, coalition_ids)
    enriched, _ = enforce_thought_stance_quality(aligned, request.stance_inputs)
    decision = evaluate_thought_disposition(
        enriched,
        association_stale=request.association.broadcast_stale,
        coalition_ids=coalition_ids,
    )
    enriched.disposition = cast(Literal["proceed", "defer", "refuse"], decision.disposition)
    enriched.disposition_reasons = list(decision.reasons)
    enriched.boundary_register = decision.boundary_register
    return enriched
