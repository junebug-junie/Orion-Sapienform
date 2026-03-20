from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ChatRequestPayload, LLMMessage, ServiceRef
from orion.core.contracts.recall import RecallQueryV1, RecallReplyV1
from orion.core.bus.bus_schemas import ChatResultPayload
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.notify import NotificationRequest


ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1 = "respond_to_juniper_collapse_mirror.v1"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str


ACTION_CATALOG: Dict[str, ActionSpec] = {
    ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1: ActionSpec(
        name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
        description="Recall + synthesize + message Juniper when Juniper writes a Collapse Mirror.",
    )
}


class ActionDedupe:
    """Tiny in-memory dedupe with inflight protection.

    - Prevents parallel double-processing (inflight)
    - Prevents repeat processing for TTL after successful completion (done)

    This is intentionally simple and process-local.
    """

    def __init__(self, ttl_seconds: int = 86400):
        self.ttl_seconds = int(ttl_seconds)
        self._done_expiry: Dict[str, float] = {}
        self._inflight: set[str] = set()

    def _now(self) -> float:
        return time.time()

    def _prune(self, now: Optional[float] = None) -> None:
        if now is None:
            now = self._now()
        expired = [k for k, exp in self._done_expiry.items() if exp <= now]
        for k in expired:
            self._done_expiry.pop(k, None)

    def try_acquire(self, key: str, *, now: Optional[float] = None) -> bool:
        if not key:
            return True
        if now is None:
            now = self._now()
        self._prune(now)
        exp = self._done_expiry.get(key)
        if exp is not None and exp > now:
            return False
        if key in self._inflight:
            return False
        self._inflight.add(key)
        return True

    def release(self, key: str) -> None:
        if not key:
            return
        self._inflight.discard(key)

    def mark_done(self, key: str, *, now: Optional[float] = None) -> None:
        if not key:
            return
        if now is None:
            now = self._now()
        self._inflight.discard(key)
        self._done_expiry[key] = now + float(self.ttl_seconds)
        self._prune(now)


def should_trigger(entry: CollapseMirrorEntryV2) -> bool:
    return str(entry.observer or "").strip().lower() == "juniper"


def dedupe_key_for(entry: CollapseMirrorEntryV2, env: BaseEnvelope) -> str:
    return (
        str(entry.event_id or "").strip()
        or str(entry.id or "").strip()
        or str(env.correlation_id)
    )


def collapse_to_fragment(entry: CollapseMirrorEntryV2) -> str:
    parts: list[str] = []
    parts.append(f"Trigger: {entry.trigger}")
    parts.append(f"Summary: {entry.summary}")
    if entry.what_changed_summary:
        parts.append(f"What changed: {entry.what_changed_summary}")
    if entry.observer_state:
        parts.append("Observer state: " + "; ".join(entry.observer_state))
    if entry.emergent_entity:
        parts.append(f"Emergent entity: {entry.emergent_entity}")
    if entry.mantra:
        parts.append(f"Mantra: {entry.mantra}")
    return "\n".join([p for p in parts if p and str(p).strip()])


def collapse_to_markdown(entry: CollapseMirrorEntryV2) -> str:
    lines: list[str] = []
    lines.append("### Collapse Mirror")
    lines.append(f"- **observer**: {entry.observer}")
    lines.append(f"- **type**: {entry.type}")
    lines.append(f"- **emergent_entity**: {entry.emergent_entity}")
    if entry.tags:
        lines.append(f"- **tags**: {', '.join(entry.tags)}")
    lines.append("")
    lines.append(f"**Trigger:** {entry.trigger}")
    lines.append("")
    lines.append(f"**Summary:** {entry.summary}")
    if entry.what_changed_summary:
        lines.append("")
        lines.append(f"**What changed:** {entry.what_changed_summary}")
    if entry.observer_state:
        lines.append("")
        lines.append("**Observer state:**")
        for s in entry.observer_state:
            lines.append(f"- {s}")
    lines.append("")
    lines.append(f"**Mantra:** {entry.mantra}")
    return "\n".join(lines).strip() + "\n"


def build_recall_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    entry: CollapseMirrorEntryV2,
    reply_to: str,
    profile: str,
    session_id: Optional[str] = None,
    node_id: Optional[str] = None,
) -> BaseEnvelope:
    q = RecallQueryV1(
        fragment=collapse_to_fragment(entry),
        profile=profile,
        session_id=session_id,
        node_id=node_id,
        verb="collapse_mirror",
        intent="respond_to_juniper",
    )
    return parent.derive_child(
        kind="recall.query.v1",
        source=source,
        payload=q,
        reply_to=reply_to,
    )


def _system_prompt() -> str:
    return (
        "You are Orion. A Collapse Mirror entry was authored by Juniper. "
        "Do two things and use the exact delimiters below.\n\n"
        "[INTROSPECT]\n"
        "Write a brief introspect+synthesize view (private, not addressed to Juniper).\n"
        "[/INTROSPECT]\n\n"
        "[MESSAGE]\n"
        "Write a supportive, specific message addressed to Juniper. "
        "Be concise, grounded in the mirror and relevant memory.\n"
        "[/MESSAGE]\n"
    )


def build_llm_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    entry: CollapseMirrorEntryV2,
    memory_rendered: str,
    reply_to: str,
    route: Optional[str] = None,
) -> BaseEnvelope:
    user_text = collapse_to_markdown(entry)
    user_text += "\nRELEVANT MEMORY\n"
    user_text += (memory_rendered or "").strip() + "\n"

    req = ChatRequestPayload(
        messages=[
            LLMMessage(role="system", content=_system_prompt()),
            LLMMessage(role="user", content=user_text),
        ],
        raw_user_text=entry.summary,
        route=route,
        options={"max_tokens": 512, "temperature": 0.3},
        session_id=None,
        user_id=None,
    )

    return parent.derive_child(
        kind="llm.chat.request",
        source=source,
        payload=req,
        reply_to=reply_to,
    )


_SECTION_RE = re.compile(
    r"\[(INTROSPECT|MESSAGE)\]\s*(.*?)\s*\[/\1\]",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_message_sections(text: str) -> Tuple[str, str]:
    introspect = ""
    message = ""
    if not text:
        return introspect, message

    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        # fallback: whole text is message
        return "", text.strip()

    for m in matches:
        label = (m.group(1) or "").strip().lower()
        content = (m.group(2) or "").strip()
        if label == "introspect":
            introspect = content
        elif label == "message":
            message = content

    if not message:
        message = text.strip()
    return introspect, message


def preview_text(message: str, *, max_len: int = 280) -> str:
    msg = (message or "").strip()
    if len(msg) <= max_len:
        return msg
    return msg[: max_len - 1].rstrip() + "…"


def build_notify_request(
    *,
    source_service: str,
    recipient_group: str,
    session_id: str,
    correlation_id: str,
    dedupe_key: str,
    dedupe_window_seconds: int,
    entry: CollapseMirrorEntryV2,
    action_name: str,
    introspect_text: str,
    message_text: str,
) -> NotificationRequest:
    body_md = "## Orion — Collapse Mirror\n\n"
    body_md += message_text.strip() + "\n"
    if introspect_text.strip():
        body_md += "\n---\n\n<details><summary>Introspect</summary>\n\n"
        body_md += introspect_text.strip() + "\n\n</details>\n"

    context: Dict[str, Any] = {
        "action_name": action_name,
        "collapse_event_id": entry.event_id,
        "collapse_id": entry.id,
        "collapse_type": entry.type,
        "collapse_tags": list(entry.tags or []),
        "collapse_emergent_entity": entry.emergent_entity,
        "preview_text": preview_text(message_text),
    }

    return NotificationRequest(
        source_service=source_service,
        event_kind="orion.chat.message",
        severity="info",
        title="Orion — Collapse Mirror",
        body_text=message_text.strip(),
        body_md=body_md,
        recipient_group=recipient_group,
        session_id=session_id,
        correlation_id=correlation_id,
        dedupe_key=f"actions:collapse_reply:{dedupe_key}",
        dedupe_window_seconds=int(dedupe_window_seconds),
        tags=["chat", "message", "actions", "collapse"],
        context=context,
    )


def build_audit_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    status: str,
    action_name: str,
    event_id: str,
    reason: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> BaseEnvelope:
    payload: Dict[str, Any] = {
        "status": status,
        "action_name": action_name,
        "event_id": event_id,
    }
    if reason:
        payload["reason"] = reason
    if extra:
        payload.update(extra)

    return parent.derive_child(
        kind="actions.audit.v1",
        source=source,
        payload=payload,
        reply_to=None,
    )


def new_reply_channel(prefix: str) -> str:
    return f"{prefix}:{uuid4()}"


def decode_recall_reply(payload: Dict[str, Any]) -> str:
    """Return bundle.rendered with safe fallback."""
    try:
        rr = RecallReplyV1.model_validate(payload)
        return rr.bundle.rendered
    except Exception:
        # fallback: best-effort
        bundle = payload.get("bundle") or {}
        return str(bundle.get("rendered") or "")


def decode_llm_result(payload: Dict[str, Any]) -> str:
    try:
        result = ChatResultPayload.model_validate(payload)
        return result.text
    except Exception:
        return str(payload.get("content") or payload.get("text") or "")
