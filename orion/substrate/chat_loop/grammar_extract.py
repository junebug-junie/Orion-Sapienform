from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.chat_projection import ChatTurnStateV1
from orion.schemas.grammar import GrammarEventV1

from .constants import CHAT_SOURCE_SERVICE, CHAT_TRACE_PREFIX

_WORD_COUNT_RE = re.compile(r"\((\d+) words?\)")
_SESSION_RE = re.compile(r"User message in session (\S+)")

_EVIDENCE_CAP = 50


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _parse_trace_id(trace_id: str) -> tuple[str, str]:
    """Parse hub.chat:{node_id}:{turn_id} → (node_id, turn_id)."""
    if not trace_id.startswith(CHAT_TRACE_PREFIX):
        raise ValueError(f"trace_id must start with '{CHAT_TRACE_PREFIX}', got: {trace_id!r}")
    remainder = trace_id[len(CHAT_TRACE_PREFIX):]
    parts = remainder.split(":", 1)
    node_id = parts[0] if len(parts) > 0 else "unknown"
    turn_id = parts[1] if len(parts) > 1 else "unknown"
    return node_id, turn_id


def extract_chat_turn_state(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
) -> ChatTurnStateV1:
    clock = _utc_now(now)

    if not events:
        raise ValueError("events must not be empty")

    trace_id = events[0].trace_id or ""
    node_id, turn_id = _parse_trace_id(trace_id)

    session_id = ""
    word_count = 0
    repair_pressure_level = 0.0
    repair_pressure_confidence = 0.0
    has_repair_signal = False
    evidence_event_ids: list[str] = []

    for event in events:
        if event.provenance.source_service != CHAT_SOURCE_SERVICE:
            continue
        atom = event.atom
        if atom is None:
            continue

        role = atom.semantic_role or ""
        summary = atom.summary or ""

        if role == "user_utterance":
            m = _WORD_COUNT_RE.search(summary)
            if m:
                word_count = int(m.group(1))
            m2 = _SESSION_RE.search(summary)
            if m2:
                session_id = m2.group(1)
        elif role == "repair_signal":
            repair_pressure_level = atom.salience or 0.0
            repair_pressure_confidence = atom.confidence or 0.0
            has_repair_signal = True
        elif role == "session_context":
            if atom.text_value is not None:
                session_id = atom.text_value

        if event.event_kind == "atom_emitted":
            if len(evidence_event_ids) < _EVIDENCE_CAP:
                evidence_event_ids.append(event.event_id)

    return ChatTurnStateV1(
        trace_id=trace_id,
        turn_id=turn_id,
        session_id=session_id,
        node_id=node_id,
        observed_at=clock,
        word_count=word_count,
        repair_pressure_level=repair_pressure_level,
        repair_pressure_confidence=repair_pressure_confidence,
        has_repair_signal=has_repair_signal,
        evidence_event_ids=evidence_event_ids,
        last_updated_at=clock,
    )


def compute_chat_pressure_hints(turn: ChatTurnStateV1) -> dict[str, float]:
    conversation_load = min(1.0, turn.word_count / 150.0)
    repair_pressure = turn.repair_pressure_level
    topic_coherence = max(0.0, 1.0 - turn.repair_pressure_level)
    return {
        "conversation_load": conversation_load,
        "repair_pressure": repair_pressure,
        "topic_coherence": topic_coherence,
    }
