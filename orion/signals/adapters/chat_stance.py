"""Chat stance brief -> chat_stance OrionSignalV1 (Milestone B3).

2026-07-22: dropped the ``_TASK_MODE_COHERENCE``/``_SALIENCE_VALENCE`` lookup
tables. They mapped ``task_mode``/``identity_salience`` enum labels to
hardcoded constants -- a relabeling, not a measurement. Replaced with:

  - ``confidence``: scored from cortex-exec's own repair/enforcement telemetry
    (``chat_stance_debug.enforcement`` -- fallback/semantic-fallback/quality-
    modification/normalization/parse-error flags it already computes per
    turn). See ``chat_stance_scoring.score_stance_confidence``.
  - ``coherence``: cosine similarity between this turn's and the previous
    turn's (same session) stance text embedding, via orion-vector-host's
    ``/embedding`` endpoint directly (``orion.signals.vector_host_client`` --
    not concept-induction's embedder, a different subsystem). No prior-turn
    embedding (first turn in a session) or a failed embed call -> coherence is
    omitted from the signal, not guessed.
  - ``valence``: dropped outright. Bucketing ``identity_salience`` into 3
    constants measured nothing real and no honest replacement was found; see
    docs/superpowers/specs/2026-07-22-self-state-phi-burn-brainstorm.md.

Real wire shape (confirmed against a live ``orion:cognition:trace`` capture,
not assumed): cortex-exec's ``executor.py`` writes the real
``ChatStanceBrief``/``ChatStanceDebug`` (PascalCase) into a *step's*
``result`` dict (``merged_result["ChatStanceBrief"] = ...``), which lands at
``payload["steps"][i]["result"]["ChatStanceBrief"]`` on the published
envelope -- never at the payload's top level, and
``build_cognition_trace_metadata()`` (services/orion-cortex-exec/app/main.py)
only ever forwards a boolean ``metadata.chat_stance_debug_present`` flag, not
the object itself. ``session_id`` similarly lives at
``payload["metadata"]["session_id"]``, not the payload top level. A prior
version of this adapter (and, before this rewrite, the original hardcoded-
lookup-table version) only checked the top level and so never found real data
in production -- confirmed by tracing the producer and by a live capture.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from orion.schemas.chat_stance import ChatStanceBrief
from orion.signals.adapters.base import OrionSignalAdapter
from orion.signals.adapters.chat_stance_scoring import (
    cosine_similarity_01,
    score_stance_confidence,
)
from orion.signals.models import OrionOrganRegistryEntry, OrionSignalV1
from orion.signals.normalization import NormalizationContext, clamp01
from orion.signals.registry import ORGAN_REGISTRY
from orion.signals.signal_ids import make_signal_id
from orion.signals.vector_host_client import embed_text

_FORBIDDEN_TEXT_KEYS = frozenset(
    {
        "stance_summary",
        "user_intent",
        "juniper_relevance",
        "self_relevance",
        "answer_strategy",
        "temporal_context",
        "audience_context",
        "environmental_context",
        "operational_context",
    }
)

# No repair telemetry available at all (bare brief, or a debug-presence flag
# with no actual debug object) -- an honest "we don't know" placeholder, not a
# fabricated per-task-mode number. Always paired with a note explaining why.
NO_TELEMETRY_CONFIDENCE = 0.5

# Bound on distinct sessions tracked for turn-to-turn coherence. A long-running
# gateway process would otherwise accumulate one embedding vector per session
# forever; oldest-inserted sessions are evicted once this cap is exceeded
# (mirrors cortex-exec's own _PRIOR_STANCE_MAX_ENTRIES bound for the same kind
# of per-session cache).
_MAX_TRACKED_SESSIONS = 512
_SESSION_ORDER_KEY = "_session_order"


def _find_step_result(payload: dict, *keys: str) -> Optional[dict]:
    """Return the first step's ``result`` dict that contains any of ``keys``.

    Real cortex-exec envelopes carry ChatStanceBrief/ChatStanceDebug nested at
    payload["steps"][i]["result"][key], never at the payload top level.
    """
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        result = step.get("result")
        if isinstance(result, dict) and any(k in result for k in keys):
            return result
    return None


def _extract_brief_dict(payload: dict) -> dict[str, Any] | None:
    for key in ("chat_stance_brief", "ChatStanceBrief"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return raw
    meta = payload.get("metadata")
    if isinstance(meta, dict):
        raw = meta.get("chat_stance_brief")
        if isinstance(raw, dict):
            return raw
    debug = payload.get("chat_stance_debug")
    if isinstance(debug, dict):
        contract = debug.get("final_prompt_contract") or {}
        if isinstance(contract, dict):
            raw = contract.get("chat_stance_brief")
            if isinstance(raw, dict):
                return raw
    step_result = _find_step_result(payload, "ChatStanceBrief")
    if step_result is not None:
        raw = step_result.get("ChatStanceBrief")
        if isinstance(raw, dict):
            return raw
    return None


def _extract_debug_dict(payload: dict) -> dict[str, Any] | None:
    debug = payload.get("chat_stance_debug")
    if isinstance(debug, dict):
        return debug
    step_result = _find_step_result(payload, "ChatStanceDebug")
    if step_result is not None:
        debug = step_result.get("ChatStanceDebug")
        if isinstance(debug, dict):
            return debug
    return None


def _extract_session_id(payload: dict) -> str | None:
    meta = payload.get("metadata")
    if isinstance(meta, dict) and meta.get("session_id"):
        return str(meta["session_id"])
    if payload.get("session_id"):
        return str(payload["session_id"])
    return None


def _stance_text(brief_raw: dict[str, Any]) -> str:
    summary = str(brief_raw.get("stance_summary") or "").strip()
    strategy = str(brief_raw.get("answer_strategy") or "").strip()
    return f"{summary}\n{strategy}".strip()


def _remember_session_embedding(
    norm_ctx: NormalizationContext, organ_id: str, session_id: str, vector: list[float]
) -> None:
    """Store `vector` under a bounded, oldest-evicted set of session keys."""
    order = norm_ctx.get_value(organ_id, _SESSION_ORDER_KEY)
    if not isinstance(order, list):
        order = []
    if session_id in order:
        order.remove(session_id)
    order.append(session_id)
    while len(order) > _MAX_TRACKED_SESSIONS:
        stale_session = order.pop(0)
        norm_ctx.delete_value(organ_id, f"embedding:{stale_session}")
    norm_ctx.set_value(organ_id, _SESSION_ORDER_KEY, order)
    norm_ctx.set_value(organ_id, f"embedding:{session_id}", vector)


class ChatStanceAdapter(OrionSignalAdapter):
    organ_id = "chat_stance"

    def can_handle(self, channel: str, payload: dict) -> bool:
        if _extract_brief_dict(payload) is not None:
            return True
        if "chat_stance" in channel:
            return True
        if "cognition:trace" in channel and payload.get("metadata", {}).get("chat_stance_debug_present"):
            return True
        if _extract_debug_dict(payload) is not None:
            return True
        return False

    def adapt(
        self,
        channel: str,
        payload: dict,
        registry: Dict[str, OrionOrganRegistryEntry],
        prior_signals: Dict[str, OrionSignalV1],
        norm_ctx: NormalizationContext,
    ) -> Optional[OrionSignalV1]:
        entry = registry.get(self.organ_id) or ORGAN_REGISTRY.get(self.organ_id)
        if entry is None:
            return None

        now = datetime.now(timezone.utc)
        brief_raw = _extract_brief_dict(payload)
        notes: list[str] = []
        debug = _extract_debug_dict(payload)
        dimensions: dict[str, float] = {}

        if brief_raw is None:
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if debug is not None:
                confidence, reasons = score_stance_confidence(debug)
                notes.extend(reasons)
            elif meta.get("chat_stance_debug_present"):
                confidence = NO_TELEMETRY_CONFIDENCE
                notes.append("stance_debug_flag_only_no_object")
            else:
                return None
            dimensions["confidence"] = confidence
            notes.append("stance_debug_only_no_brief")
        else:
            try:
                ChatStanceBrief.model_validate(brief_raw)
            except Exception:
                notes.append("partial_chat_stance_payload_schema_validation_skipped")

            if debug is not None:
                confidence, reasons = score_stance_confidence(debug)
                notes.extend(reasons)
            else:
                confidence = NO_TELEMETRY_CONFIDENCE
                notes.append("no_repair_telemetry_available")
            dimensions["confidence"] = confidence

            text = _stance_text(brief_raw)
            session_id = _extract_session_id(payload)
            if text and session_id:
                current_vec = embed_text(text)
                if current_vec is not None:
                    prev_vec = norm_ctx.get_value(self.organ_id, f"embedding:{session_id}")
                    if prev_vec is not None:
                        coherence = cosine_similarity_01(prev_vec, current_vec)
                        if coherence is not None:
                            dimensions["coherence"] = coherence
                        else:
                            notes.append("coherence_unavailable_vector_mismatch")
                    else:
                        notes.append("coherence_unavailable_first_turn_in_session")
                    _remember_session_embedding(norm_ctx, self.organ_id, session_id, current_vec)
                else:
                    notes.append("coherence_unavailable_embed_failed")
            elif not session_id:
                notes.append("coherence_unavailable_no_session_id")
            else:
                notes.append("coherence_unavailable_no_stance_text")

        corr = (
            payload.get("correlation_id")
            or payload.get("_envelope_correlation_id")
            or (payload.get("metadata") or {}).get("root_correlation_id")
        )
        src_id = str(corr) if corr else None
        if src_id is None:
            verb = str(payload.get("verb") or payload.get("verb_name") or "chat")
            src_id = f"{verb}:{int(now.timestamp())}"
            notes.append("synthetic_correlation_id")

        causal_parents = [
            prior_signals[p].signal_id
            for p in (entry.causal_parent_organs or [])
            if p in prior_signals
        ]

        task_mode = ""
        if brief_raw and isinstance(brief_raw, dict):
            task_mode = str(brief_raw.get("task_mode") or "")
        summary = f"chat_stance task_mode={task_mode or 'unknown'}" if task_mode else "chat_stance"

        return OrionSignalV1(
            signal_id=make_signal_id(self.organ_id, src_id),
            organ_id=self.organ_id,
            organ_class=entry.organ_class,
            signal_kind="chat_stance",
            dimensions={k: clamp01(float(v)) for k, v in dimensions.items()},
            causal_parents=causal_parents,
            source_event_id=src_id,
            observed_at=now,
            emitted_at=now,
            summary=summary,
            notes=notes[:5],
        )
