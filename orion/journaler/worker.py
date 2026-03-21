from __future__ import annotations

import hashlib
import json
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


def draft_from_cortex_result(payload: dict[str, Any]) -> JournalEntryDraftV1:
    final_text = payload.get("final_text")
    if not isinstance(final_text, str) or not final_text.strip():
        raise ValueError("cortex_orch_missing_final_text")
    parsed = json.loads(final_text)
    if not isinstance(parsed, dict):
        raise ValueError("journal_draft_not_object")
    return JournalEntryDraftV1.model_validate(parsed)


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
