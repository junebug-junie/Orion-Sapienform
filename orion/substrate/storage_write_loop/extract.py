"""Parse one sql-writer window trace (``sql_writer.storage:<writer>:<window_id>``)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orion.schemas.grammar import GrammarEventV1
from orion.schemas.storage_write_projection import (
    MAX_FAMILIES_PER_WINDOW,
    ROLE_FAMILY_WINDOW,
    ROLE_WINDOW_COMPLETED,
    STORAGE_WRITE_SOURCE_SERVICE,
    STORAGE_WRITE_TRACE_PREFIX,
    WRITE_FAILURE_CLASSES,
    StorageWriteFamilyStateV1,
)

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")
_FAMILY_RE = re.compile(r"^[a-z0-9_.-]{1,96}$")


def _utc(ts: datetime | None, fallback: datetime) -> datetime:
    value = ts or fallback
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def parse_storage_write_trace_id(trace_id: str) -> tuple[str, str] | None:
    """``sql_writer.storage:<writer>:<window_id>`` -> (writer, window_id)."""
    if not trace_id or not trace_id.startswith(STORAGE_WRITE_TRACE_PREFIX):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) != 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2].strip()


def _parse_kv(summary: str) -> dict[str, str]:
    return {k.lower(): v.strip() for k, v in _KV_RE.findall(summary or "")}


def _int(kv: dict[str, str], key: str) -> int:
    try:
        return max(0, int(kv.get(key, "0") or 0))
    except ValueError:
        return 0


def _opt_int(kv: dict[str, str], key: str) -> int | None:
    raw = kv.get(key)
    if raw in (None, "", "none", "None"):
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _parse_classes(raw: str | None) -> dict[str, int]:
    """Only known failure classes are kept: an unknown name from a newer producer
    is dropped rather than guessed into the reading."""
    out: dict[str, int] = {}
    for part in (raw or "").split("|"):
        name, _, count = part.partition(":")
        name = name.strip()
        if not name or name == "none" or name not in WRITE_FAILURE_CLASSES:
            continue
        try:
            out[name] = out.get(name, 0) + max(0, int(count or 0))
        except ValueError:
            continue
    return out


@dataclass
class StorageWriteWindow:
    writer: str
    window_id: str
    window_end: datetime | None = None
    window_sec: float | None = None
    completed: bool = False
    grammar_queue_max: int | None = None
    families: dict[str, StorageWriteFamilyStateV1] = field(default_factory=dict)
    evidence_event_ids: list[str] = field(default_factory=list)


def extract_storage_write_window(
    events: list[GrammarEventV1], *, now: datetime | None = None
) -> StorageWriteWindow:
    """One trace (or a part of it, when a reducer batch splits a trace) -> window.

    Family atoms carry their own counts, so a split trace yields disjoint
    partial windows that the reducer merges by family key. ``window_end`` is the
    atoms' ``emitted_at`` (the writer's window end), so a backlog replays the same
    readings it would have produced live."""
    clock = now or datetime.now(timezone.utc)
    if not events:
        raise ValueError("events must not be empty")
    trace_id = events[0].trace_id or ""
    parsed = parse_storage_write_trace_id(trace_id)
    if not parsed:
        raise ValueError(f"invalid storage_write trace_id: {trace_id}")
    writer, window_id = parsed
    window = StorageWriteWindow(writer=writer, window_id=window_id)
    for event in events:
        if event.provenance.source_service != STORAGE_WRITE_SOURCE_SERVICE or not event.atom:
            continue
        role = (event.atom.semantic_role or "").strip()
        kv = _parse_kv(event.atom.summary or "")
        emitted = _utc(event.emitted_at, clock)
        if role == ROLE_WINDOW_COMPLETED:
            window.completed = True
            window.window_end = emitted
            try:
                window.window_sec = max(0.0, float(kv.get("window_sec", "0") or 0.0))
            except ValueError:
                window.window_sec = None
            window.grammar_queue_max = _opt_int(kv, "grammar_queue_max")
            window.evidence_event_ids.append(event.event_id)
        elif role == ROLE_FAMILY_WINDOW:
            family = (kv.get("family") or "").strip().lower()
            if not _FAMILY_RE.match(family) or family in window.families:
                # malformed, or a duplicate atom for one family (a producer bug;
                # summing would double-count a replayed event)
                continue
            if len(window.families) >= MAX_FAMILIES_PER_WINDOW + 1:
                continue
            classes = _parse_classes(kv.get("classes"))
            committed = _int(kv, "committed")
            duplicate = _int(kv, "duplicate")
            failed = sum(classes.values())
            window.families[family] = StorageWriteFamilyStateV1(
                family=family,
                # Recomputed from the parts, not trusted from the wire, so the
                # reading's denominator always equals what it names.
                attempted=committed + duplicate + failed,
                committed=committed,
                duplicate=duplicate,
                failed=failed,
                skipped=_int(kv, "skipped"),
                unrouted=_int(kv, "unrouted"),
                failure_classes=classes,
                commit_p50_ms=_opt_int(kv, "p50_ms"),
                commit_p95_ms=_opt_int(kv, "p95_ms"),
            )
            window.window_end = window.window_end or emitted
            window.evidence_event_ids.append(event.event_id)
    return window
