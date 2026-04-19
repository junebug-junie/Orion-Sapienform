from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, LLMMessage, ServiceRef
from orion.journaler import JournalTriggerV1, build_compose_request
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2
from orion.schemas.cortex.contracts import CortexClientContext, CortexClientRequest


ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1 = "respond_to_juniper_collapse_mirror.v1"
ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB = "actions.respond_to_juniper_collapse_mirror.v1"
SKILL_BIOMETRICS_SNAPSHOT_V1 = "skills.biometrics.snapshot.v1"
SKILL_GPU_NVIDIA_SMI_SNAPSHOT_V1 = "skills.gpu.nvidia_smi_snapshot.v1"
SKILL_NOTIFY_CHAT_MESSAGE_V1 = "skills.system.notify_chat_message.v1"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str


ACTION_CATALOG: Dict[str, ActionSpec] = {
    ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1: ActionSpec(
        name=ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
        description="Dispatch a Cortex action when Juniper writes a Collapse Mirror.",
    )
}


class ActionDedupe:
    """Tiny in-memory dedupe with inflight protection."""

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
        if key:
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
    return str(entry.event_id or "").strip() or str(entry.id or "").strip() or str(env.correlation_id)


def collapse_to_fragment(entry: CollapseMirrorEntryV2) -> str:
    parts: list[str] = [
        f"Trigger: {entry.trigger}",
        f"Summary: {entry.summary}",
    ]
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
    lines: list[str] = [
        "### Collapse Mirror",
        f"- **observer**: {entry.observer}",
        f"- **type**: {entry.type}",
        f"- **emergent_entity**: {entry.emergent_entity}",
    ]
    if entry.tags:
        lines.append(f"- **tags**: {', '.join(entry.tags)}")
    lines.extend(["", f"**Trigger:** {entry.trigger}", "", f"**Summary:** {entry.summary}"])
    if entry.what_changed_summary:
        lines.extend(["", f"**What changed:** {entry.what_changed_summary}"])
    if entry.observer_state:
        lines.extend(["", "**Observer state:**", *[f"- {state}" for state in entry.observer_state]])
    lines.extend(["", f"**Mantra:** {entry.mantra}"])
    return "\n".join(lines).strip() + "\n"


def build_skill_cortex_orch_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    verb: str,
    session_id: str,
    user_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    options: Dict[str, Any] | None = None,
    recall_enabled: bool = False,
) -> BaseEnvelope:
    context = CortexClientContext(
        messages=[],
        raw_user_text=f"scheduled skill dispatch: {verb}",
        user_message=f"scheduled skill dispatch: {verb}",
        session_id=session_id,
        user_id=user_id,
        trace_id=str(parent.correlation_id),
        metadata=metadata or {},
    )
    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb=verb,
        packs=[],
        options={"source": "orion-actions", "policy_dispatch_only": True, **(options or {})},
        recall={"enabled": recall_enabled, "required": False, "profile": None},
        context=context,
    )
    return parent.derive_child(kind="cortex.orch.request", source=source, payload=req, reply_to=None)


def build_cortex_orch_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    entry: CollapseMirrorEntryV2,
    session_id: str,
    recipient_group: str,
    dedupe_key: str,
    dedupe_window_seconds: int,
    recall_profile: str,
    verb: str = ACTIONS_RESPOND_TO_JUNIPER_CORTEX_VERB,
) -> BaseEnvelope:
    collapse_md = collapse_to_markdown(entry)
    context = CortexClientContext(
        messages=[LLMMessage(role="user", content=collapse_md)],
        raw_user_text=entry.summary,
        user_message=collapse_md,
        session_id=session_id,
        trace_id=str(parent.correlation_id),
        metadata={
            "action_name": ACTION_RESPOND_TO_JUNIPER_COLLAPSE_V1,
            "action_verb": verb,
            "collapse_entry": entry.model_dump(mode="json"),
            "collapse_event_id": entry.event_id,
            "collapse_trigger": entry.trigger,
            "collapse_summary": entry.summary,
            "collapse_mantra": entry.mantra,
            "collapse_tags": list(entry.tags or []),
            "recipient_group": recipient_group,
            "notify_dedupe_key": f"actions:collapse_reply:{dedupe_key}",
            "notify_dedupe_window_seconds": int(dedupe_window_seconds),
            "session_id": session_id,
            "recall_profile": recall_profile,
            "recall_fragment": collapse_to_fragment(entry),
        },
    )
    req = CortexClientRequest(
        mode="brain",
        route_intent="none",
        verb=verb,
        packs=[],
        options={"source": "orion-actions", "policy_dispatch_only": True},
        recall={"enabled": True, "required": False, "profile": recall_profile},
        context=context,
    )
    return parent.derive_child(kind="cortex.orch.request", source=source, payload=req, reply_to=None)


def build_journal_cortex_orch_envelope(
    parent: BaseEnvelope,
    *,
    source: ServiceRef,
    trigger: JournalTriggerV1,
    session_id: str,
    user_id: str | None = None,
    recall_profile: str | None = "collapse_mirror.v1",
    options: Dict[str, Any] | None = None,
) -> BaseEnvelope:
    req = build_compose_request(
        trigger,
        session_id=session_id,
        user_id=user_id,
        trace_id=str(parent.correlation_id),
        recall_profile=recall_profile,
        options=options,
    )
    return parent.derive_child(kind="cortex.orch.request", source=source, payload=req, reply_to=None)


async def dispatch_cortex_request(*, bus: Any, channel: str, envelope: BaseEnvelope) -> None:
    await bus.publish(channel, envelope)


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
    return parent.derive_child(kind="actions.audit.v1", source=source, payload=payload, reply_to=None)


def new_reply_channel(prefix: str) -> str:
    return f"{prefix}:{uuid4()}"


_SECTION_RE = re.compile(r"\[(INTROSPECT|MESSAGE)\]\s*(.*?)\s*\[/\1\]", flags=re.DOTALL | re.IGNORECASE)


def extract_message_sections(text: str) -> Tuple[str, str]:
    introspect = ""
    message = ""
    if not text:
        return introspect, message
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return "", text.strip()
    for match in matches:
        label = (match.group(1) or "").strip().lower()
        content = (match.group(2) or "").strip()
        if label == "introspect":
            introspect = content
        elif label == "message":
            message = content
    return introspect, message or text.strip()
