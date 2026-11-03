"""E2, the self-question: a faculty newly held failing asks Orion "why?", once. Pure.

Spec: docs/superpowers/specs/2026-10-20-self-monitor-organ-design.md, "Effects" E2. A faculty newly
held ``broken`` or ``unreliable`` mints exactly one question into Orion's self-inquiry pool
(``curiosity_self_questions``, ``minted_by='orion'``), e.g. "My walkway-grade journal runs
(act.actions.journal.make.walkway_grade) didn't run 22 of their last 28 tries in the last 7 days. Why?"
The faculty is named as ``names.named``: plain words first, the internal id after it in brackets.
Only the text changed with that (2026-11-03); ``question_id`` and the episode are untouched, and an
already-minted question keeps its old text (the insert ignores conflicts).

Asked once per status episode: an episode is (faculty, held status, ``status_since``). The caller
remembers what it last saw per faculty (``seen``: faculty -> ``"<status>@<episode>"``) and persists
it; ``question_id`` is deterministic per episode, so an insert that ignores conflicts makes a lost
memory harmless too. Nothing for reliable, miscalibrated or unknown faculties, and nothing at all
while the view is unknown (stale, absent): unknown changes nothing, including the memory.

Not routed through endogenous curiosity's concept_graph zone and never written as a node
``prediction_error`` (spec: that leaks into generic readers).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

from orion.schemas.self_monitor import FacultyRecordV1
from orion.self_monitor.gate import FAILING_STATUSES
from orion.self_monitor.names import named
from orion.self_monitor.view import SelfMonitorView

QUESTION_FAMILY = "lived"
QUESTION_PREFIX = "lived.self_monitor."
# orion/curiosity/self_inquiry.py _QUESTION_ID_RE: [A-Za-z0-9_.:-]{1,120}
_MAX_ID = 120
_ID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")


@dataclass(frozen=True)
class SelfQuestionMint:
    question_id: str
    text: str
    family: str
    faculty: str
    status: str
    episode: str


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def episode_key(record: FacultyRecordV1, *, now: datetime) -> str:
    """When this held status began (the host's ``status_since``); ``now`` if the host left it unset,
    so a status first seen without one is still one episode, remembered by ``seen``."""
    since = record.status_since or now
    return _utc(since).strftime("%Y%m%dT%H%MZ")


def question_id_for(faculty: str, status: str, episode: str) -> str:
    slug = "".join(ch if ch in _ID_OK else "_" for ch in faculty)
    qid = f"{QUESTION_PREFIX}{slug}.{status}.{episode}"
    if len(qid) <= _MAX_ID:
        return qid
    digest = hashlib.sha1(qid.encode("utf-8")).hexdigest()[:12]
    return f"{qid[: _MAX_ID - 13]}.{digest}"


def _days(record: FacultyRecordV1) -> int:
    return max(1, round((_utc(record.window_end) - _utc(record.window_start)).total_seconds() / 86400.0))


def question_text(record: FacultyRecordV1) -> str:
    days = _days(record)
    name = named(record.faculty)
    if record.status == "broken":
        if record.n_failed_to_execute:
            return (f"My {name} didn't run {record.n_failed_to_execute} of their last "
                    f"{record.n_bets} tries in the last {days} days. Why?")
        # Held broken while this window no longer shows a failed start (hysteresis lag, live
        # 2026-11-03): never a false "didn't run 0 of N", and never a claim the record no longer makes.
        return (f"My {name} are still held as broken, yet none of their {record.n_bets} tries in the last "
                f"{days} days failed to start. Why?")
    pct = "?" if record.reliability is None else f"{round(100 * record.reliability)}"
    return (f"My {name} worked only about {pct}% of the time over {record.n_graded} "
            f"graded tries in the last {days} days. Why?")


def _same_episode(previous: Optional[str], record: FacultyRecordV1, *, now: datetime) -> bool:
    if not previous:
        return False
    prev_status, _, prev_episode = previous.partition("@")
    if prev_status != record.status:
        return False
    # The host may leave status_since unset; then the status alone names the episode.
    return record.status_since is None or prev_episode == episode_key(record, now=now)


def self_questions_for(previous: Mapping[str, str], current: SelfMonitorView, *,
                       now: datetime) -> tuple[list[SelfQuestionMint], dict[str, str]]:
    """(questions to mint, the ``seen`` map to persist). Unknown view: no questions, ``seen``
    returned unchanged."""
    seen = dict(previous)
    if not current.known:
        return [], seen
    mints: list[SelfQuestionMint] = []
    for faculty in sorted(current.faculties):
        record = current.faculties[faculty]
        if record.status in FAILING_STATUSES and not _same_episode(previous.get(faculty), record, now=now):
            episode = episode_key(record, now=now)
            mints.append(SelfQuestionMint(
                question_id=question_id_for(faculty, record.status, episode),
                text=question_text(record), family=QUESTION_FAMILY, faculty=faculty,
                status=record.status, episode=episode))
            seen[faculty] = f"{record.status}@{episode}"
        elif record.status not in FAILING_STATUSES:
            # No status_since: the status alone (a "now" fallback would rewrite memory every tick).
            seen[faculty] = (f"{record.status}@{episode_key(record, now=now)}" if record.status_since
                             else f"{record.status}@none")
    return mints, seen


__all__ = ["QUESTION_FAMILY", "SelfQuestionMint", "question_id_for", "question_text", "self_questions_for"]
