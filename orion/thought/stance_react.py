from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import uuid4

from orion.thought.json_extract import extract_first_json_object_text
from orion.schemas.pre_turn_appraisal import TurnAppraisalBundleV1
from orion.schemas.thought import StanceHarnessSliceV1, StanceReactRequestV1, ThoughtEventV1, HubAssociationBundleV1
from orion.thought.coalition import (
    align_evidence_refs_to_coalition,
    coalition_ids_from_association,
    prompt_turn_refs,
)
from orion.thought.policy_refusal import evaluate_thought_disposition
from orion.thought.stance_quality import enforce_thought_stance_quality
from orion.schemas.temporal_self import UNCITED_SENTENCES_SHOWN, DayCitationCheckV1, TemporalSelfStanceCueV1
from orion.temporal_self.citations import ARC_EVIDENCE_RE, arc_ref, check_arc_citations, gate_arc_refs

logger = logging.getLogger(__name__)

IMPERATIVE_MAX_LEN = 400
TONE_MAX_LEN = 200


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


def _coerce_and_clamp_str(
    value: Any,
    *,
    default: str,
    max_len: int,
    field: str,
    correlation_id: str | None,
) -> str:
    """Coerce like _coerce_required_str, then clamp overlong LLM output instead of
    letting ThoughtEventV1's Field(max_length=...) reject the whole turn."""
    text = _coerce_required_str(value, default=default)
    if len(text) > max_len:
        logger.warning(
            "stance_react_field_overflow field=%s len=%d max=%d corr=%s — clamping",
            field,
            len(text),
            max_len,
            correlation_id,
        )
        text = text[:max_len]
    return text


def normalize_stance_react_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM JSON mistakes before ThoughtEventV1 validation."""
    data = dict(raw)
    correlation_id = _coerce_optional_str(data.get("correlation_id"))
    data["event_id"] = _coerce_required_str(data.get("event_id"), default=str(uuid4()))
    data["created_at"] = _coerce_required_str(
        data.get("created_at"),
        default=datetime.now(timezone.utc).isoformat(),
    )
    data["profile"] = _coerce_required_str(data.get("profile"), default="stance_react")
    data["producer"] = _coerce_required_str(data.get("producer"), default="stance_react_v1")
    data["llm_profile"] = _coerce_required_str(data.get("llm_profile"), default="brain")
    data["imperative"] = _coerce_and_clamp_str(
        data.get("imperative"),
        default="",
        max_len=IMPERATIVE_MAX_LEN,
        field="imperative",
        correlation_id=correlation_id,
    )
    data["tone"] = _coerce_and_clamp_str(
        data.get("tone"),
        default="neutral",
        max_len=TONE_MAX_LEN,
        field="tone",
        correlation_id=correlation_id,
    )
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
    # No correlation_id and no full `hub:turn:<id>`: the model copying that UUID
    # is what burned max_tokens live 2026-10-02 (see coalition.HUB_TURN_REF_TOKEN).
    broadcast = association.broadcast
    attended = list(broadcast.attended_node_ids) if broadcast else []
    return {
        "broadcast_stale": association.broadcast_stale,
        "read_source": association.read_source,
        "attended_node_ids": prompt_turn_refs(attended, association.correlation_id),
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
        "paradigms": paradigms,
    }
    # A failed paradigm has no slice. Say so explicitly, so the stance does not read an empty
    # `paradigms` as "no repair pressure": the level is unknown (e.g. the probe was shed by heat).
    if bundle.failed_paradigms:
        reasons = dict(bundle.failure_reasons or {})
        slim["unmeasured"] = {
            name: f"level unknown: {reasons.get(name) or 'failed'}" for name in bundle.failed_paradigms
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
    # Same treatment for the autonomy slice: it is the V2 reducer's own persisted
    # drive/tension state, assembled deterministically in cortex-exec and mapped on
    # from result metadata — never authored by the stance LLM.
    raw.pop("autonomy_slice", None)
    # And for the day-citation check: the gate's result, never the model's claim about it.
    raw.pop("day_citations", None)
    # Record identity is owned by the transport, never the stance LLM: it has no
    # clock and no id source, and invents plausible values when asked.
    raw.pop("event_id", None)
    raw.pop("created_at", None)
    if correlation_id:
        raw["correlation_id"] = correlation_id
    raw["session_id"] = session_id
    return ThoughtEventV1.model_validate(normalize_stance_react_raw(raw))


def _merge_day_checks(checks: list[DayCitationCheckV1], refs_rejected: list[str],
                      refs_valid: list[str], shown: int) -> DayCitationCheckV1:
    valid: list[str] = []
    for ref in refs_valid + [r for c in checks for r in c.cited_valid]:
        if ref not in valid:
            valid.append(ref)
    rejected = list(refs_rejected) + [r for c in checks for r in c.rejected_unknown]
    uncited = sum(c.uncited_day_sentences for c in checks)
    examples = [e for c in checks for e in c.uncited_examples]
    label = "UNVERIFIED" if uncited else ("verified" if valid else "no_day_claims")
    return DayCitationCheckV1(
        shown=shown, cited_valid=valid, rejected_unknown=rejected, uncited_day_sentences=uncited,
        uncited_examples=examples[:UNCITED_SENTENCES_SHOWN], label=label, action="stripped" if rejected else "kept",
    )


def gate_day_citations(
    thought: ThoughtEventV1,
    day_cue: TemporalSelfStanceCueV1 | None,
) -> tuple[ThoughtEventV1, set[str]]:
    """Temporal Self patch 4: check this stance's references to Orion's own day.

    Arc ids the cue showed join the allowed evidence set; any other ``arc:`` ref is removed from
    evidence_refs, strain_refs (which widen the allowed set, so an unknown id there would launder
    itself) and the imperative/tone text, and counted. A sentence naming a shown label without
    citing its arc is counted and the check labelled UNVERIFIED -- never blocked or retried.
    Returns (thought with ``day_citations`` set, the ``arc:<id>`` refs to add to the coalition)."""
    shown = list(day_cue.arc_ids) if day_cue is not None and day_cue.text else []
    labels = dict(day_cue.labels) if day_cue is not None and day_cue.text else {}
    evidence, rejected_e = gate_arc_refs(thought.evidence_refs, shown)
    strain, rejected_s = gate_arc_refs(thought.strain_refs, shown)
    cited_refs = [r for r in evidence + strain if ARC_EVIDENCE_RE.match(r)]
    cited_ids = [r[4:] for r in cited_refs]
    imperative, check_i = check_arc_citations(thought.imperative, shown, labels=labels, cited_elsewhere=cited_ids)
    tone, check_t = check_arc_citations(thought.tone, shown, labels=labels, cited_elsewhere=cited_ids)
    touched = bool(shown or rejected_e or rejected_s or check_i.rejected_unknown or check_t.rejected_unknown)
    if not touched:
        return thought, set()
    check = _merge_day_checks([check_i, check_t], rejected_e + rejected_s, cited_refs, len(shown))
    gated = thought.model_copy(update={
        "evidence_refs": evidence, "strain_refs": strain, "imperative": imperative, "tone": tone,
        "day_citations": check,
    })
    return gated, {arc_ref(a) for a in shown}


def apply_stance_react_pipeline(
    thought: ThoughtEventV1,
    request: StanceReactRequestV1,
    *,
    day_cue: TemporalSelfStanceCueV1 | None = None,
) -> ThoughtEventV1:
    coalition_ids = coalition_ids_from_association(request.association)
    thought, day_arc_refs = gate_day_citations(thought, day_cue)
    coalition_ids = coalition_ids | day_arc_refs
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
