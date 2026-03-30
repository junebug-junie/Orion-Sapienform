from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, CollapseMirrorStoredV1
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest, RecallDirective
from orion.schemas.notify import NotificationRecord
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1

from .schemas import JournalEntryDraftV1, JournalEntryWriteV1, JournalMode, JournalTriggerV1

JOURNAL_COMPOSE_VERB = "journal.compose"
JOURNAL_WRITE_KIND = "journal.entry.write.v1"
JOURNAL_CREATED_KIND = "journal.entry.created.v1"
logger = logging.getLogger("orion.journaler.worker")


_TRIGGER_TO_MODE: dict[str, JournalMode] = {
    "daily_summary": "daily",
    "collapse_response": "response",
    "metacog_digest": "digest",
    "manual": "manual",
    "notify_summary": "daily",
}


def journal_mode_for_trigger(trigger: JournalTriggerV1) -> JournalMode:
    return _TRIGGER_TO_MODE[trigger.trigger_kind]


def cooldown_key_for_trigger(trigger: JournalTriggerV1) -> str:
    if trigger.trigger_kind == "collapse_response":
        ref = (trigger.source_ref or "").strip()
        if not ref:
            ref = hashlib.sha256((trigger.summary or "").strip().encode("utf-8")).hexdigest()[:20]
        return f"actions:journal:{trigger.trigger_kind}:{trigger.source_kind}:{ref}"
    material = json.dumps(
        {
            "trigger_kind": trigger.trigger_kind,
            "source_kind": trigger.source_kind,
            "source_ref": trigger.source_ref,
            "summary": trigger.summary.strip(),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
    return f"actions:journal:{trigger.trigger_kind}:{trigger.source_kind}:{digest}"


def build_manual_trigger(*, summary: str, prompt_seed: str | None = None, source_ref: str | None = None) -> JournalTriggerV1:
    return JournalTriggerV1(
        trigger_kind="manual",
        source_kind="manual",
        source_ref=source_ref,
        summary=summary,
        prompt_seed=prompt_seed,
    )


def build_scheduler_trigger(*, summary: str, prompt_seed: str | None = None, source_ref: str | None = None) -> JournalTriggerV1:
    return JournalTriggerV1(
        trigger_kind="daily_summary",
        source_kind="scheduler",
        source_ref=source_ref,
        summary=summary,
        prompt_seed=prompt_seed,
    )


def build_notify_summary_trigger(record: NotificationRecord) -> JournalTriggerV1:
    prompt_seed = record.body_md or record.body_text
    summary = (record.body_text or record.title or "").strip() or f"Notify summary: {record.event_kind}"
    return JournalTriggerV1(
        trigger_kind="notify_summary",
        source_kind="notify",
        source_ref=str(record.notification_id),
        summary=summary,
        prompt_seed=prompt_seed,
    )


def build_collapse_trigger(entry: CollapseMirrorEntryV2) -> JournalTriggerV1:
    prompt_parts = [
        f"Trigger: {entry.trigger}",
        f"Summary: {entry.summary}",
    ]
    if entry.what_changed_summary:
        prompt_parts.append(f"What changed: {entry.what_changed_summary}")
    if entry.mantra:
        prompt_parts.append(f"Mantra: {entry.mantra}")
    return JournalTriggerV1(
        trigger_kind="collapse_response",
        source_kind="collapse_mirror",
        source_ref=str(entry.event_id or entry.id or "") or None,
        summary=entry.summary,
        prompt_seed="\n".join(prompt_parts),
    )


def build_collapse_stored_trigger(event: CollapseMirrorStoredV1) -> JournalTriggerV1:
    prompt_parts = [
        f"Trigger: {event.trigger}",
        f"Summary: {event.summary}",
    ]
    if event.what_changed_summary:
        prompt_parts.append(f"What changed: {event.what_changed_summary}")
    if event.mantra:
        prompt_parts.append(f"Mantra: {event.mantra}")
    return JournalTriggerV1(
        trigger_kind="collapse_response",
        source_kind="collapse_mirror",
        source_ref=event.mirror_id,
        summary=event.summary,
        prompt_seed="\n".join(prompt_parts),
    )


def build_metacog_trigger(trigger: MetacogTriggerV1) -> JournalTriggerV1:
    prompt_seed = json.dumps(
        {
            "trigger_kind": trigger.trigger_kind,
            "zen_state": trigger.zen_state,
            "pressure": trigger.pressure,
            "window_sec": trigger.window_sec,
            "upstream": trigger.upstream,
        },
        sort_keys=True,
        default=str,
    )
    return JournalTriggerV1(
        trigger_kind="metacog_digest",
        source_kind="metacog",
        source_ref=trigger.timestamp,
        summary=trigger.reason,
        prompt_seed=prompt_seed,
    )


def build_compose_request(
    trigger: JournalTriggerV1,
    *,
    session_id: str,
    trace_id: str,
    user_id: str | None = None,
    recall_profile: str | None = "reflect.v1",
    options: dict[str, Any] | None = None,
) -> CortexClientRequest:
    metadata = {
        "journal_trigger": trigger.model_dump(mode="json"),
        "journal_mode": journal_mode_for_trigger(trigger),
    }
    context = CortexClientContext(
        messages=[],
        raw_user_text=trigger.summary,
        user_message=trigger.summary,
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        metadata=metadata,
    )
    return CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb=JOURNAL_COMPOSE_VERB,
        packs=[],
        options={"source": "orion-actions", "policy_dispatch_only": True, **(options or {})},
        recall=RecallDirective(enabled=True, required=False, profile=recall_profile),
        context=context,
    )


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>", flags=re.IGNORECASE | re.DOTALL)

@dataclass(frozen=True)
class _DraftParseDiagnostics:
    raw_length: int
    response_text_source: str
    reasoning_fields_present: bool
    fences_stripped: bool
    think_tags_detected: bool
    think_blocks_stripped: bool
    leading_non_json_stripped: bool
    object_extraction_attempted: bool


def _extract_first_json_object_text(text: str) -> str | None:
    candidate = (text or "").strip()
    if not candidate:
        return None

    start = candidate.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : idx + 1]
            continue

    return None


def _validate_journal_draft_payload(parsed: dict[str, Any]) -> None:
    required = ("mode", "title", "body")
    for key in required:
        if key not in parsed:
            raise ValueError(f"journal_draft_missing_required_key:{key}")

    if not isinstance(parsed.get("mode"), str):
        raise ValueError("journal_draft_invalid_type:mode")
    title = parsed.get("title")
    if title is not None and not isinstance(title, str):
        raise ValueError("journal_draft_invalid_type:title")
    if not isinstance(parsed.get("body"), str):
        raise ValueError("journal_draft_invalid_type:body")


def _try_parse_object(text: str) -> dict[str, Any] | None:
    candidate = (text or "").strip()
    if not candidate:
        return None

    try:
        direct = json.loads(candidate)
    except Exception:
        return None
    return direct if isinstance(direct, dict) else None


def _missing_required_key_from_error(message: str) -> str | None:
    prefix = "journal_draft_missing_required_key:"
    if not isinstance(message, str) or not message.startswith(prefix):
        return None
    key = message[len(prefix) :].strip()
    return key or None


def _parse_journal_draft_json(final_text: str) -> tuple[dict[str, Any], _DraftParseDiagnostics]:
    attempts: list[str] = []
    base = (final_text or "").strip()
    attempts.append(base)
    fences_stripped = False
    think_tags_detected = "<think>" in base.lower()
    think_blocks_stripped = False
    leading_non_json_stripped = False
    for match in _FENCED_JSON_RE.finditer(base):
        fenced = (match.group(1) or "").strip()
        if fenced:
            attempts.append(fenced)
            fences_stripped = True

    cleaned_attempts: list[str] = []
    for text in attempts:
        cleaned_attempts.append(text)
        if think_tags_detected:
            stripped = _THINK_BLOCK_RE.sub(" ", text).strip()
            if stripped != text:
                think_blocks_stripped = True
                cleaned_attempts.append(stripped)
            start = stripped.find("{")
            if start > 0:
                leading_non_json_stripped = True
                cleaned_attempts.append(stripped[start:].strip())
    attempts = cleaned_attempts

    extraction_attempted = False
    for text in attempts:
        parsed = _try_parse_object(text)
        if isinstance(parsed, dict):
            _validate_journal_draft_payload(parsed)
            return parsed, _DraftParseDiagnostics(
                raw_length=len(final_text or ""),
                response_text_source="final_text",
                reasoning_fields_present=False,
                fences_stripped=fences_stripped,
                think_tags_detected=think_tags_detected,
                think_blocks_stripped=think_blocks_stripped,
                leading_non_json_stripped=leading_non_json_stripped,
                object_extraction_attempted=extraction_attempted,
            )

        extracted = _extract_first_json_object_text(text)
        if extracted:
            extraction_attempted = True
            parsed = _try_parse_object(extracted)
            if isinstance(parsed, dict):
                _validate_journal_draft_payload(parsed)
                return parsed, _DraftParseDiagnostics(
                    raw_length=len(final_text or ""),
                    response_text_source="final_text",
                    reasoning_fields_present=False,
                    fences_stripped=fences_stripped,
                    think_tags_detected=think_tags_detected,
                    think_blocks_stripped=think_blocks_stripped,
                    leading_non_json_stripped=leading_non_json_stripped,
                    object_extraction_attempted=extraction_attempted,
                )

    raise ValueError("journal_draft_parse_failed")


def _select_draft_text(payload: dict[str, Any]) -> tuple[str | None, str]:
    final_text = payload.get("final_text")
    if isinstance(final_text, str) and final_text.strip():
        return final_text, "final_text"
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content, "content"
    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        return text, "text"
    return None, "missing"


def draft_from_cortex_result(payload: dict[str, Any]) -> JournalEntryDraftV1:
    final_text, text_source = _select_draft_text(payload)
    if not isinstance(final_text, str) or not final_text.strip():
        raise ValueError("cortex_orch_missing_final_text")
    reasoning_fields_present = bool(
        (isinstance(payload.get("reasoning_content"), str) and payload.get("reasoning_content").strip())
        or isinstance(payload.get("reasoning_trace"), dict)
    )
    diagnostics = _DraftParseDiagnostics(
        raw_length=len(final_text),
        response_text_source=text_source,
        reasoning_fields_present=reasoning_fields_present,
        fences_stripped=False,
        think_tags_detected="<think>" in final_text.lower(),
        think_blocks_stripped=False,
        leading_non_json_stripped=False,
        object_extraction_attempted=False,
    )
    try:
        parsed, diagnostics = _parse_journal_draft_json(final_text)
        diagnostics = _DraftParseDiagnostics(
            raw_length=diagnostics.raw_length,
            response_text_source=text_source,
            reasoning_fields_present=reasoning_fields_present,
            fences_stripped=diagnostics.fences_stripped,
            think_tags_detected=diagnostics.think_tags_detected,
            think_blocks_stripped=diagnostics.think_blocks_stripped,
            leading_non_json_stripped=diagnostics.leading_non_json_stripped,
            object_extraction_attempted=diagnostics.object_extraction_attempted,
        )
        return JournalEntryDraftV1.model_validate(parsed)
    except Exception as exc:
        parse_context = {
            "error": type(exc).__name__,
            "message": str(exc),
            "correlation_id": payload.get("correlation_id"),
            "verb": payload.get("verb"),
            "status": payload.get("status"),
            "raw_output_length": diagnostics.raw_length,
            "response_text_source": diagnostics.response_text_source,
            "reasoning_fields_present": diagnostics.reasoning_fields_present,
            "fences_stripped": diagnostics.fences_stripped,
            "think_tags_detected": diagnostics.think_tags_detected,
            "think_blocks_stripped": diagnostics.think_blocks_stripped,
            "leading_non_json_stripped": diagnostics.leading_non_json_stripped,
            "json_object_extraction_attempted": diagnostics.object_extraction_attempted,
            "missing_required_key": _missing_required_key_from_error(str(exc)),
        }
        logger.error(
            "journal_draft_parse_failed context=%s final_text_preview=%r",
            parse_context,
            final_text[:256],
        )
        raise ValueError(f"journal_draft_parse_failed:{json.dumps(parse_context, default=str, sort_keys=True)}") from exc


def build_write_payload(
    draft: JournalEntryDraftV1,
    *,
    trigger: JournalTriggerV1,
    correlation_id: str | None,
    author: str,
    entry_id: str | None = None,
    created_at: datetime | None = None,
) -> JournalEntryWriteV1:
    ts = created_at or datetime.now(timezone.utc)
    return JournalEntryWriteV1(
        entry_id=entry_id or str(uuid4()),
        created_at=ts,
        author=author,
        mode=draft.mode,
        title=draft.title,
        body=draft.body,
        source_kind=trigger.source_kind,
        source_ref=trigger.source_ref,
        correlation_id=correlation_id,
    )


def build_created_event_payload(write: JournalEntryWriteV1) -> dict[str, Any]:
    return write.model_dump(mode="json")
