"""One curiosity run, told as a story: what happened, in what order, and what
came of it -- joined across the three stores that each hold a piece.

WHY THIS EXISTS. Juniper, 2026-09-22: "I can't really understand what happens
in these runs, what happened between hops, if there were reach outs to me and
then subsequent replies I may have offered." Every piece she asked for already
existed -- a role choice and numbered hop notes with a millisecond clock in
Orion's own graph, lifecycle transitions and the journal write-up in Postgres,
the outreach decision in a third table keyed by a derived uuid -- but nothing
joined them, so the page showed a bare numbered list with no times, no start,
no end, no failures, and "wanted to reach out" with no word on whether it
happened. This module is the join.

PURE ASSEMBLY. Nothing here talks to a store. Every input is a list of plain
row dicts as the stores hand them back, so the join is testable with fixtures
from each store and the one place a cross-store bug can live is inspectable
without a database. The store readers live next to the endpoints
(`services/orion-hub/scripts/curiosity_run_store.py`) and the graph Cypher in
`orion/curiosity/atlas.py`.

DEGRADES BY NAMING WHAT IS MISSING, NOT BY GUESSING. Since 2026-09-14 only
`completed` lifecycle rows have been landing (cause under separate
investigation), so a run's start falls back to its earliest graph node and
says so; a reach-out with no decision row reads `not_recorded`, never
"blocked" or "sent"; a hop written before the graph clock shipped keeps `n` as
its only order and gets no attempt number, because two attempts' worth of
1,1,2,2 cannot be untangled after the fact -- that is the collision
`worldview.hop_order_key` exists to stop recurring, not something to undo.

LIFECYCLE HAS TWO SOURCES, AND THE NEWER ONE WINS. Since 2026-09-14 curiosity
runs go through the resource-admission path, which writes every transition to
`durable_resource_events` (accepted, waiting for a lane, admitted, running per
node, retrying, completed/failed) with the run's acceptance in
`durable_admission_runs`, and copies ONLY the terminal `completed` row into
`substrate_durable_run_state` -- mislabelling its workflow as
`curiosity.investigate` even for a self-sense check (root cause: PR #2288).
So when a run has admission-path rows they are its lifecycle and the bridge
row is ignored except as a fallback for the finish `detail`; a run with only
bridge rows (pre 09-14) is read from those. Neither is required.

Two sibling patches land fields this module READS when present and reports as
not recorded when absent -- it never requires them:
  - outreach decisions for pre-check blocks, with `result_json.source =
    "curiosity_outreach"` and `result_json.run_id`; and a reply stamp
    `client_meta.in_reply_to = <outreach correlation id>` on Juniper's next
    message in that session;
  - `harness_elapsed_sec` and `turn_correlation_id` on the finish row's
    `detail`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import NAMESPACE_URL, uuid5

from orion.curiosity.journal import OUTREACH_TAG
from orion.curiosity.self_inquiry import LINE_INVESTIGATE, LINE_SELF_INQUIRY
from orion.curiosity.worldview import _as_bool, _as_float, _as_int, hop_order_key
from orion.schemas.self_sense import SELF_SENSE_QUESTIONS

logger = logging.getLogger("orion.curiosity.run_story")

# The third line. Defined by name here because the loop that owns the constant
# is a Hub script this package cannot import; `services/orion-hub/tests/
# test_curiosity_routes_runs.py` pins the two equal.
LINE_SELF_SENSE_EVAL = "self_sense_eval"

# Plain words for the three lines -- the only vocabulary the page shows.
LINE_LABELS: dict[str, str] = {
    LINE_INVESTIGATE: "World question",
    LINE_SELF_INQUIRY: "Self question",
    LINE_SELF_SENSE_EVAL: "Self-sense check",
}
LINES = tuple(LINE_LABELS)

# Lifecycle workflows that ARE curiosity. `self_study.reflect` lands in the
# same table and is not; a naive `SELECT *` would leak it into the strip.
CURIOSITY_WORKFLOWS = ("curiosity.investigate", "self_sense_eval")
WORKFLOW_SELF_SENSE = "self_sense_eval"

# The journal title the self-inquiry line writes under (self_panel.py reads the
# same string). Used only as a line fallback when the finish row is missing.
_SELF_INQUIRY_JOURNAL_TITLE = "Self-inquiry"

# Decision vocabulary for the reach-out line. `not_recorded` is the honest
# default until patch 2 records pre-check blocks.
DECISION_SENT = "sent"
DECISION_NOT_RECORDED = "not_recorded"
DECISION_COMPOSED_EMPTY = "composed_empty"
DECISION_PASSED = "passed"
_REASON_TO_DECISION = {
    "sent": DECISION_SENT,
    "empty_generation": DECISION_COMPOSED_EMPTY,
    "orion_passed": DECISION_PASSED,
}

# Timeline tie-break within one millisecond: a role choice precedes a hop
# written at the same instant, an outcome precedes the journal it prompted.
_KIND_RANK = {
    "starting_prior": -1,
    "role_choice": 0,
    "lifecycle": 1,
    "attempt": 2,
    "hop": 3,
    "self_sense_answer": 3,
    "self_write": 4,
    "finding": 4,
    "revision": 5,
    "outcome": 6,
    "journal": 7,
    "outreach": 8,
    "reply": 9,
}

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_RUNNING = "running"
STATUS_UNKNOWN = "unknown"
_TERMINAL = (STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED)

# Admission-path events and what the timeline makes of each. Anything not
# listed is ignored (`run.lane_swap_suppressed`, `run.resource_granted`,
# `run.resource_eligibility_expanded`, `resource.elastic_*`).
EVENT_ACCEPTED = "run.accepted"
EVENT_WAITING = "run.waiting_resource"
EVENT_ADMITTED = "run.admitted"
EVENT_LANE_ASSIGNED = "run.lane_assigned"
EVENT_STARTED = "run.started"
EVENT_RUNNING = "run.running"
EVENT_RESUMED = "run.resumed"
EVENT_RETRYING = "run.retrying"
EVENT_COMPLETED = "run.completed"
EVENT_FAILED = "run.failed"
EVENT_CANCELLED = "run.cancelled"
EVENT_LEASE_RELEASED = "resource.lease_released"
EVENT_LEASE_EXPIRED = "resource.lease_expired"
EVENT_CHECKPOINT_RESUME_FAILED = "run.checkpoint_resume_failed"
ANOMALY_EVENTS = (EVENT_CHECKPOINT_RESUME_FAILED,)
WORKFLOW_REFLECT = "self_study.reflect"

# What the strip's glyph says about a run, decided here so the page and the
# tests read one rule.
OUTCOME_SENT = "reached_out_sent"
OUTCOME_REACH_BLOCKED = "reached_out_blocked"
OUTCOME_DIED = "died"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_EMPTY = "wrote_nothing"
OUTCOME_RUNNING = "running"
OUTCOME_FINISHED = "finished"

_TEXT_LIMIT = 4000
_NOTE_LIMIT = 2000


def outreach_key(run_id: str) -> str:
    """The one run -> message key. Mirrors `curiosity_investigation.py`'s
    `uuid5(NAMESPACE_URL, f"{OUTREACH_TAG}:{run_id}")` exactly; the Hub test
    next to the endpoints pins the two derivations equal."""
    return str(uuid5(NAMESPACE_URL, f"{OUTREACH_TAG}:{run_id}"))


def _ms(value: Any) -> Optional[int]:
    """Epoch ms from whatever a store handed back: a datetime (naive is UTC --
    `chat_history_log.created_at` is `timestamp without time zone` and writes
    UTC), an int/float epoch ms, a numeric string, or ISO text."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def _text(value: Any, limit: int = _TEXT_LIMIT) -> str:
    return str(value or "").strip()[:limit]


def _obj(value: Any) -> dict[str, Any]:
    """A JSON column as a dict, whether the driver decoded it or not."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# --- inputs ---------------------------------------------------------------


@dataclass
class RunStoryRows:
    """Every row list the join reads, as the stores return them. All optional:
    a missing store is an empty list, and the story says what it lacks."""

    lifecycle: list[dict[str, Any]] = field(default_factory=list)
    roles: list[dict[str, Any]] = field(default_factory=list)
    hops: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[dict[str, Any]] = field(default_factory=list)
    priors: list[dict[str, Any]] = field(default_factory=list)
    journals: list[dict[str, Any]] = field(default_factory=list)
    outreach: list[dict[str, Any]] = field(default_factory=list)
    chat: list[dict[str, Any]] = field(default_factory=list)
    readings: list[dict[str, Any]] = field(default_factory=list)
    self_sense: list[dict[str, Any]] = field(default_factory=list)
    self_writes: list[dict[str, Any]] = field(default_factory=list)
    help_requests: list[dict[str, Any]] = field(default_factory=list)
    peer_briefs: list[dict[str, Any]] = field(default_factory=list)
    # Admission path (since 2026-09-14): one `durable_admission_runs` row per
    # run, its `durable_resource_events`, and per-run counts of the noisy
    # anomaly events the store does not fetch row by row.
    admission: list[dict[str, Any]] = field(default_factory=list)
    resource_events: list[dict[str, Any]] = field(default_factory=list)
    event_counts: list[dict[str, Any]] = field(default_factory=list)


# --- outputs --------------------------------------------------------------


@dataclass(frozen=True)
class TimelineItem:
    at: Optional[int]
    kind: str
    data: dict[str, Any]
    attempt: Optional[int] = None

    def sort_key(self, anchor: Optional[int] = None) -> tuple:
        """Where an item sits when its clock is missing, by kind:

        - a hop or role choice: FIRST. Undated hops are legacy (pre
          2026-09-19) and predate every dated node in the run
          (`hop_order_key`); a role choice is written at the start.
        - a finding, revision or self-write: right after the last dated hop
          (`anchor`). 95 of 117 live Findings carry no clock, and the prompt
          has Orion write them after the hops; floating them to the top
          told the story backwards. Printed with no offset, so the
          placement is visibly a placement and not a measurement.
        - anything else (a `not_recorded` outreach, a reply): LAST.
        """
        rank = _KIND_RANK.get(self.kind, 50)
        n = _as_int(self.data.get("n"), 0)
        if self.at is not None:
            return (1, self.at, rank, n)
        if self.kind in ("hop", "role_choice", "attempt", "starting_prior"):
            return (0, 0, rank, n)
        if self.kind in ("finding", "revision", "self_write") and anchor is not None:
            return (1, anchor, rank + 0.5, n)
        return (2, 0, rank, n)


@dataclass(frozen=True)
class ReachOut:
    wanted: bool
    why: str
    decision: Optional[str]  # sent | blocked:<gate> | composed_empty | passed | not_recorded | None
    gate: Optional[str] = None
    decided_at: Optional[int] = None
    sent_at: Optional[int] = None
    composed_text: str = ""
    reply_at: Optional[int] = None
    reply_text: str = ""

    @property
    def sent(self) -> bool:
        return self.decision == DECISION_SENT


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    line: str
    line_known: bool
    plain_line_label: str
    started_at: Optional[int]
    started_from: str  # admission | lifecycle | graph | self_sense | none
    finished_at: Optional[int]
    accepted_at: Optional[int]
    admitted_at: Optional[int]
    lane: str
    retries: int
    anomalies: dict[str, int]
    status: str
    attempts: Optional[int]
    error: str
    hops: int
    findings: int
    revisions: int
    prior_touched: Optional[dict[str, Any]]
    reach_out: ReachOut
    journal_entry_id: str
    finding_text: str
    self_written: Optional[dict[str, Any]]
    self_sense: Optional[dict[str, Any]]
    harness: Optional[dict[str, Any]]
    outcome_kind: str

    @property
    def duration_sec(self) -> Optional[float]:
        """Accepted (or first clock) to finished -- the whole sitting,
        lane wait included."""
        if self.started_at is None or self.finished_at is None:
            return None
        return round((self.finished_at - self.started_at) / 1000, 1)

    @property
    def lane_wait_sec(self) -> Optional[float]:
        if self.accepted_at is None or self.admitted_at is None:
            return None
        return round(max(0, self.admitted_at - self.accepted_at) / 1000, 1)

    @property
    def active_sec(self) -> Optional[float]:
        """Admitted to finished: the part of the sitting Orion was actually
        running. None when either end is unknown."""
        if self.admitted_at is None or self.finished_at is None:
            return None
        return round(max(0, self.finished_at - self.admitted_at) / 1000, 1)


@dataclass(frozen=True)
class RunStory:
    run: RunSummary
    timeline: list[TimelineItem]
    journal_body: str
    readings_available: bool
    starting_prior: Optional[dict[str, Any]] = None
    summary: Optional[dict[str, Any]] = None
    prior_outcome: Optional[dict[str, Any]] = None


# --- the join -------------------------------------------------------------


def _slot_factory() -> dict[str, Any]:
    return {
        "lifecycle": [],
        "roles": [],
        "hops": [],
        "findings": [],
        "revisions": [],
        "outcomes": [],
        "journals": [],
        "self_sense": [],
        "self_writes": [],
        "help_requests": [],
        "peer_briefs": [],
        "admission": None,
        "events": [],
        "event_counts": {},
    }


def _group(rows: RunStoryRows) -> dict[str, dict[str, Any]]:
    slots: dict[str, dict[str, Any]] = {}

    def slot(run_id: Any) -> Optional[dict[str, Any]]:
        rid = _text(run_id, 64)
        if not rid:
            return None
        return slots.setdefault(rid, _slot_factory())

    for row in rows.lifecycle:
        if _text(row.get("workflow"), 60) not in CURIOSITY_WORKFLOWS:
            continue
        s = slot(row.get("run_id"))
        if s is not None:
            s["lifecycle"].append(row)
    for key in ("roles", "hops", "findings", "revisions", "outcomes", "self_writes",
                "help_requests", "peer_briefs"):
        for row in getattr(rows, key):
            s = slot(row.get("run_id"))
            if s is not None:
                s[key].append(row)
    for row in rows.journals:
        ref = _text(row.get("source_ref"), 120)
        rid = ref.split(":", 1)[1] if ref.startswith("curiosity:") else ""
        s = slot(rid)
        if s is not None:
            s["journals"].append(row)
    for row in rows.self_sense:
        s = slot(row.get("run_id"))
        if s is not None:
            s["self_sense"].append(row)
    reflect_ids: set[str] = set()
    for row in rows.admission:
        request = _obj(row.get("request"))
        if _text(request.get("workflow"), 60) == WORKFLOW_REFLECT:
            reflect_ids.add(_text(row.get("run_id"), 64))
            continue
        s = slot(row.get("run_id"))
        if s is not None:
            s["admission"] = row
    for row in rows.resource_events:
        rid = _text(row.get("run_id"), 64)
        if rid in reflect_ids:
            continue
        s = slot(rid)
        if s is not None:
            s["events"].append(row)
    for row in rows.event_counts:
        rid = _text(row.get("run_id"), 64)
        if rid in reflect_ids:
            continue
        s = slot(rid)
        if s is not None:
            s["event_counts"][_text(row.get("event"), 60)] = _as_int(row.get("n"), 0)
    for rid in reflect_ids:
        slots.pop(rid, None)
    return slots


def _lifecycle_sorted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: _ms(r.get("created_at")) or 0)


def _terminal_row(lifecycle: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The LAST completed/failed row by clock. A failed attempt followed by a
    resume and a completion is a completed run whose story shows the death."""
    terminal = [r for r in lifecycle if _text(r.get("status"), 20) in (STATUS_COMPLETED, STATUS_FAILED)]
    return terminal[-1] if terminal else None


def _completed_detail(lifecycle: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(lifecycle):
        if _text(row.get("status"), 20) == STATUS_COMPLETED:
            return _obj(row.get("detail"))
    return {}


@dataclass
class _Lifecycle:
    """What the run's lifecycle rows -- from whichever source -- say."""

    items: list[TimelineItem] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None
    finished_at: Optional[int] = None
    accepted_at: Optional[int] = None
    admitted_at: Optional[int] = None
    lane: str = ""
    retries: int = 0
    error: str = ""
    non_terminal_stamps: list[int] = field(default_factory=list)
    source: str = "none"  # admission | lifecycle | none


def _event_lane(detail: dict[str, Any]) -> str:
    lease = _obj(detail.get("lease")) if isinstance(detail.get("lease"), (dict, str)) else {}
    return _text(lease.get("lane") or detail.get("lane") or detail.get("requested_lane"), 40)


def _lifecycle_from_events(slot: dict[str, Any]) -> _Lifecycle:
    """The admission path, in generated_at order. Consecutive waits collapse
    to one item (a run re-announces its wait at each checkpoint)."""
    out = _Lifecycle(source="admission")
    events = sorted(slot["events"], key=lambda e: (_ms(e.get("generated_at")) or 0, _text(e.get("entry_id"), 200)))
    admission = slot["admission"] or {}
    out.accepted_at = _ms(admission.get("created_at"))
    waiting_open = False
    for e in events:
        kind = _text(e.get("event"), 60)
        payload = _obj(e.get("payload"))
        detail = _obj(payload.get("detail"))
        at = _ms(e.get("generated_at"))
        if at is not None and kind not in (EVENT_COMPLETED, EVENT_FAILED, EVENT_CANCELLED):
            out.non_terminal_stamps.append(at)
        if kind == EVENT_ACCEPTED:
            out.accepted_at = out.accepted_at or at
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={"node": "", "status": "accepted", "next_node": "", "resumed_from": "", "error": ""}))
        elif kind == EVENT_WAITING:
            if waiting_open:
                continue
            waiting_open = True
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": _text(detail.get("node"), 60), "status": "waiting",
                "lane": _event_lane(detail), "next_node": "", "resumed_from": "", "error": ""}))
        elif kind in (EVENT_ADMITTED, EVENT_LANE_ASSIGNED, EVENT_STARTED):
            lane = _event_lane(detail)
            out.lane = out.lane or lane
            if out.admitted_at is None:
                out.admitted_at = at
                wait = None
                if at is not None and out.accepted_at is not None:
                    wait = round(max(0, at - out.accepted_at) / 1000, 1)
                out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                    "node": _text(detail.get("node"), 60), "status": "admitted", "lane": lane,
                    "wait_sec": wait, "next_node": "", "resumed_from": "", "error": ""}))
            waiting_open = False
        elif kind == EVENT_RUNNING:
            waiting_open = False
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": _text(detail.get("node"), 60), "status": "running", "next_node": "", "resumed_from": "", "error": ""}))
        elif kind == EVENT_RESUMED:
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": _text(detail.get("node"), 60), "status": "resumed", "next_node": "",
                "resumed_from": _text(detail.get("node"), 60), "error": ""}))
        elif kind == EVENT_RETRYING:
            out.retries += 1
            out.error = _text(detail.get("error") or detail.get("reason"), 500) or out.error
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": _text(detail.get("node"), 60), "status": "retrying", "next_node": "", "resumed_from": "",
                "error": _text(detail.get("error") or detail.get("reason"), 500)}))
        elif kind == EVENT_LEASE_RELEASED:
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": "", "status": "lease_released", "lane": _event_lane(detail),
                "reason": _text(detail.get("reason"), 60), "next_node": "", "resumed_from": "", "error": ""}))
        elif kind == EVENT_LEASE_EXPIRED:
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": "", "status": "lease_expired", "lane": _event_lane(detail), "next_node": "", "resumed_from": "", "error": ""}))
        elif kind in (EVENT_COMPLETED, EVENT_FAILED, EVENT_CANCELLED):
            status = {EVENT_COMPLETED: STATUS_COMPLETED, EVENT_FAILED: STATUS_FAILED, EVENT_CANCELLED: STATUS_CANCELLED}[kind]
            out.status, out.finished_at = status, at
            if status == STATUS_COMPLETED:
                out.detail = detail
            else:
                out.error = _text(detail.get("error"), 500) or out.error
            out.items.append(TimelineItem(at=at, kind="lifecycle", data={
                "node": _text(detail.get("node"), 60) or "finish", "status": status, "next_node": "",
                "resumed_from": "", "error": _text(detail.get("error"), 500)}))
    terminal = _text(admission.get("terminal"), 20)
    if terminal in _TERMINAL:
        out.status = terminal
        out.finished_at = out.finished_at or _ms(admission.get("updated_at"))
    elif out.status is None:
        out.status = STATUS_RUNNING
    return out


def _lifecycle_from_bridge(lifecycle: list[dict[str, Any]]) -> _Lifecycle:
    """`substrate_durable_run_state` rows: the only source before 2026-09-14."""
    out = _Lifecycle(source="lifecycle")
    for row in lifecycle:
        at = _ms(row.get("created_at"))
        status = _text(row.get("status"), 20)
        d = _obj(row.get("detail"))
        if at is not None and status not in _TERMINAL:
            out.non_terminal_stamps.append(at)
        if status == STATUS_FAILED:
            out.error = _text(d.get("error"), 500)
        out.items.append(TimelineItem(at=at, kind="lifecycle", data={
            "node": _text(row.get("node"), 60), "next_node": _text(row.get("next_node"), 60),
            "status": status, "resumed_from": _text(row.get("resumed_from_node"), 60),
            "error": _text(d.get("error"), 500)}))
    terminal = _terminal_row(lifecycle)
    if terminal is not None:
        out.status = _text(terminal.get("status"), 20)
        out.finished_at = _ms(terminal.get("created_at"))
    elif lifecycle:
        out.status = STATUS_RUNNING
    out.detail = _completed_detail(lifecycle)
    return out


def _line_for(slot: dict[str, Any], detail: dict[str, Any], prior_lines: dict[str, str]) -> tuple[str, bool]:
    """(line, known). The finish row is the only place the line is recorded;
    everything after it is a fallback that says so via `known=False`."""
    request = _obj((slot.get("admission") or {}).get("request"))
    if _text(request.get("workflow"), 60) == WORKFLOW_SELF_SENSE:
        return LINE_SELF_SENSE_EVAL, True
    brief_line = _text(_obj(request.get("brief")).get("line"), 40)
    if brief_line in LINE_LABELS:
        return brief_line, True
    line = _text(detail.get("line"), 40)
    if line in LINE_LABELS:
        return line, True
    if slot["self_sense"] or any(
        _text(r.get("workflow"), 60) == WORKFLOW_SELF_SENSE for r in slot["lifecycle"]
    ):
        return LINE_SELF_SENSE_EVAL, True
    for j in slot["journals"]:
        if _text(j.get("title"), 80) == _SELF_INQUIRY_JOURNAL_TITLE:
            return LINE_SELF_INQUIRY, True
    for rev in slot["revisions"]:
        if prior_lines.get(_text(rev.get("prior_id"), 200)) == "self":
            return LINE_SELF_INQUIRY, False
    return LINE_INVESTIGATE, False


def _hop_items(hops: list[dict[str, Any]], readings: list[dict[str, Any]]) -> tuple[list[TimelineItem], Optional[int]]:
    """Hops in clock order with attempt numbers. A retried turn restarts `n`
    at 1 under the same run_id, so `n <= previous n` in clock order is a new
    attempt. Legacy hops (no clock) get no attempt: within that group `n` is
    all there is and 1,1,2,2 stays interleaved -- undoable."""
    ordered = sorted(
        hops,
        key=lambda h: hop_order_key((_as_int(h.get("n"), 0), _ms(h.get("written_at")))),
    )
    by_key: dict[tuple[int, Optional[int]], list[dict[str, Any]]] = {}
    for r in readings:
        by_key.setdefault((_as_int(r.get("hop_n"), 0), _ms(r.get("hop_written_at"))), []).append(r)

    items: list[TimelineItem] = []
    attempt = 0
    prev_n: Optional[int] = None
    for h in ordered:
        n = _as_int(h.get("n"), 0)
        at = _ms(h.get("written_at"))
        this_attempt: Optional[int] = None
        if at is not None:
            if prev_n is None or n <= prev_n:
                attempt += 1
                if attempt > 1:
                    items.append(TimelineItem(at=at, kind="attempt", data={"attempt": attempt, "n": n}, attempt=attempt))
            prev_n = n
            this_attempt = attempt
        hop_readings = [
            {
                "kind": _text(r.get("kind"), 60),
                "about_prior_id": _text(r.get("about_prior_id"), 200),
                "moved_the_claim": _as_bool(r.get("moved_the_claim")),
                "confidence": _as_float(r.get("reading_confidence")),
                "reasoning": _text(r.get("reasoning"), _NOTE_LIMIT),
            }
            for r in by_key.get((n, at), [])
        ]
        items.append(
            TimelineItem(
                at=at,
                kind="hop",
                data={"n": n, "note": _text(h.get("note"), _NOTE_LIMIT), "readings": hop_readings},
                attempt=this_attempt,
            )
        )
    return items, (attempt or None)


def _reach_out(
    run_id: str,
    *,
    outcome: Optional[dict[str, Any]],
    detail: dict[str, Any],
    outreach_by_key: dict[str, dict[str, Any]],
    sent_by_key: dict[str, dict[str, Any]],
    replies_by_key: dict[str, list[dict[str, Any]]],
) -> ReachOut:
    wanted = _as_bool(outcome.get("reach_out")) if outcome else False
    why = _text(outcome.get("reach_out_why"), _NOTE_LIMIT) if outcome else ""
    if not wanted and detail:
        wanted = _as_bool(detail.get("reach_out"))
        why = why or _text(detail.get("reach_out_why"), _NOTE_LIMIT)
    key = outreach_key(run_id)
    decision_row = outreach_by_key.get(key)
    sent_row = sent_by_key.get(key)
    replies = sorted(replies_by_key.get(key, []), key=lambda r: _ms(r.get("created_at")) or 0)

    decision: Optional[str] = None
    gate: Optional[str] = None
    decided_at: Optional[int] = None
    sent_at: Optional[int] = None
    composed = ""
    if decision_row is not None:
        reason = _text(decision_row.get("reason"), 60)
        decided_at = _ms(decision_row.get("decided_at"))
        decision = _REASON_TO_DECISION.get(reason)
        if decision is None:
            decision = f"blocked:{reason or 'unknown'}"
            gate = reason or "unknown"
        result = _obj(decision_row.get("result_json"))
        composed = _text(result.get("composed_text") or result.get("text"), _NOTE_LIMIT)
        if decision == DECISION_SENT:
            sent_at = decided_at
        # A decision row proves the run asked, whatever the graph says.
        wanted = True
    elif sent_row is not None:
        decision = DECISION_SENT
        wanted = True
    elif wanted:
        decision = DECISION_NOT_RECORDED
    if sent_row is not None:
        sent_at = _ms(sent_row.get("created_at")) or sent_at
        composed = _text(sent_row.get("response"), _NOTE_LIMIT) or composed
    reply = replies[0] if replies else None
    return ReachOut(
        wanted=wanted,
        why=why,
        decision=decision,
        gate=gate,
        decided_at=decided_at,
        sent_at=sent_at,
        composed_text=composed,
        reply_at=_ms(reply.get("created_at")) if reply else None,
        reply_text=_text(reply.get("prompt"), _NOTE_LIMIT) if reply else "",
    )


def _outcome_kind(*, status: str, reach: ReachOut, wrote: int) -> str:
    if reach.sent:
        return OUTCOME_SENT
    if reach.wanted:
        return OUTCOME_REACH_BLOCKED
    if status == STATUS_FAILED:
        return OUTCOME_DIED
    if status == STATUS_CANCELLED:
        return OUTCOME_CANCELLED
    if status == STATUS_RUNNING:
        return OUTCOME_RUNNING
    if wrote == 0:
        return OUTCOME_EMPTY
    return OUTCOME_FINISHED


def _starting_prior_for_slot(
    *,
    help_rows: list[dict[str, Any]],
    revisions: list[dict[str, Any]],
    prior_claims: dict[str, str],
    prior_meta: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Subject prior for the sitting: HelpRequest ABOUT first, else first revision.

    Status/confidence prefer the *start-of-sitting* values from the first
    PriorRevision of that prior (`from_status` / `from_confidence`). The live
    `:Prior` node is already post-write-back after a finished sitting.
    """
    def _start_state(pid: str) -> tuple[Optional[str], Optional[float]]:
        for rev in revisions:
            if _text(rev.get("prior_id"), 200) != pid:
                continue
            status = _text(rev.get("from_status"), 60) or None
            conf = _as_float(rev.get("from_confidence"))
            return status, conf
        meta = prior_meta.get(pid) or {}
        return _text(meta.get("status"), 60) or None, _as_float(meta.get("confidence"))

    for h in sorted(help_rows, key=lambda r: _ms(r.get("written_at")) or 0):
        pid = _text(h.get("prior_id"), 200)
        if not pid:
            continue
        meta = prior_meta.get(pid) or {}
        claim = _text(h.get("prior_claim")) or prior_claims.get(pid, "") or _text(meta.get("claim"))
        status, conf = _start_state(pid)
        # HelpRequest may still carry a hire-time snapshot; prefer revision
        # from_* when present, else the help row, else live meta (via _start_state).
        if status is None:
            status = _text(h.get("prior_status"), 60) or None
        if conf is None and h.get("prior_confidence") is not None:
            conf = _as_float(h.get("prior_confidence"))
        return {
            "prior_id": pid,
            "claim": claim,
            "status": status,
            "confidence": conf,
            "source": "help_request_about",
            "help_id": _text(h.get("help_id"), 200) or None,
        }
    if revisions:
        rev = revisions[0]
        pid = _text(rev.get("prior_id"), 200)
        if pid:
            meta = prior_meta.get(pid) or {}
            status, conf = _start_state(pid)
            return {
                "prior_id": pid,
                "claim": prior_claims.get(pid, "") or _text(meta.get("claim")),
                "status": status or _text(rev.get("from_status"), 60) or None,
                "confidence": conf if conf is not None else _as_float(rev.get("from_confidence")),
                "source": "prior_revision",
                "help_id": None,
            }
    return None


def _subject_prior_revision(
    revisions: list[dict[str, Any]],
    starting_prior: Optional[dict[str, Any]],
    prior_claims: dict[str, str],
) -> Optional[dict[str, Any]]:
    """Last PriorRevision that touches the sitting's subject prior.

    Side revisions of other priors must not drive the prior→outcome verdict.
    """
    if not revisions:
        return None
    subject_id = _text((starting_prior or {}).get("prior_id"), 200) if starting_prior else ""
    chosen = None
    for rev in revisions:
        pid = _text(rev.get("prior_id"), 200)
        if not pid:
            continue
        if subject_id and pid != subject_id:
            continue
        chosen = rev
    if chosen is None and not subject_id and revisions:
        chosen = revisions[-1]
    if chosen is None:
        return None
    pid = _text(chosen.get("prior_id"), 200)
    return {
        "prior_id": pid,
        "claim": prior_claims.get(pid, ""),
        "from": _as_float(chosen.get("from_confidence")),
        "to": _as_float(chosen.get("to_confidence")),
        "from_status": _text(chosen.get("from_status"), 60),
        "to_status": _text(chosen.get("to_status"), 60),
    }


def _prior_outcome_block(
    *,
    starting_prior: Optional[dict[str, Any]],
    prior_touched: Optional[dict[str, Any]],
    self_written: Optional[dict[str, Any]],
    peer_rows: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Prior → what was found, with a measured verdict label."""
    if starting_prior is None and self_written is None and not peer_rows and prior_touched is None:
        return None

    outcome_text = ""
    outcome_kind = "none"
    if self_written and _text(self_written.get("text")):
        outcome_text = _text(self_written.get("text"))
        outcome_kind = _text(self_written.get("kind"), 40) or "lived_answer"
    peer = None
    if peer_rows:
        latest = sorted(peer_rows, key=lambda r: _ms(r.get("written_at")) or 0)[-1]
        peer = {
            "status": _text(latest.get("status"), 60),
            "peer": _text(latest.get("peer"), 80),
            "summary": _text(latest.get("summary"), 1200),
            "refusal_reason": _text(latest.get("refusal_reason"), 240) or None,
            "help_id": _text(latest.get("help_id"), 200) or None,
        }
        if not outcome_text and peer["summary"]:
            outcome_text = peer["summary"]
            outcome_kind = "peer_brief"

    to_status = _text((prior_touched or {}).get("to_status"), 60).lower()
    from_conf = (prior_touched or {}).get("from")
    to_conf = (prior_touched or {}).get("to")
    if to_status in ("refuted", "retired_unresolvable", "retired"):
        verdict, basis = "refuted", f"Prior status moved to {to_status}"
    elif to_status in ("supported", "settled", "confirmed"):
        verdict, basis = "supported", f"Prior status moved to {to_status}"
    elif to_status == "revised":
        if (
            isinstance(from_conf, (int, float))
            and isinstance(to_conf, (int, float))
            and to_conf != from_conf
        ):
            direction = "up" if to_conf > from_conf else "down"
            verdict, basis = f"revised_{direction}", f"Confidence {from_conf} → {to_conf}"
        else:
            verdict, basis = "revised", "Prior marked revised"
    elif (
        isinstance(from_conf, (int, float))
        and isinstance(to_conf, (int, float))
        and to_conf > from_conf
    ):
        verdict, basis = "revised_up", f"Confidence {from_conf} → {to_conf}"
    elif (
        isinstance(from_conf, (int, float))
        and isinstance(to_conf, (int, float))
        and to_conf < from_conf
    ):
        verdict, basis = "revised_down", f"Confidence {from_conf} → {to_conf}"
    elif outcome_kind in ("lived_answer", "self_definition") and outcome_text:
        verdict, basis = "answered", f"Wrote {outcome_kind.replace('_', ' ')}"
        if peer and peer["status"] == "ok":
            basis += "; peer brief ok"
        elif peer and peer["status"]:
            basis += f"; peer {peer['status']}"
    elif peer and peer["status"] == "ok":
        verdict, basis = "peer_ok", "Peer brief returned ok"
    elif peer and peer["status"] in ("failed", "refused_budget"):
        verdict, basis = "peer_failed", f"Peer brief {peer['status']}"
    elif starting_prior:
        verdict, basis = "open", "Prior still open; no measured write-back"
    else:
        verdict, basis = "unknown", "No prior or outcome recorded"

    return {
        "prior": starting_prior,
        "outcome_text": outcome_text,
        "outcome_kind": outcome_kind,
        "verdict": verdict,
        "verdict_basis": basis,
        "revision": prior_touched,
        "peer": peer,
    }


def _summary_card(
    *,
    run: RunSummary,
    role: Optional[dict[str, Any]],
    help_rows: list[dict[str, Any]],
    peer_rows: list[dict[str, Any]],
    starting_prior: Optional[dict[str, Any]],
    prior_outcome: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "line": run.plain_line_label,
        "status": run.status,
        "hops": run.hops,
        "findings": run.findings,
        "revisions": run.revisions,
        "helps": len(help_rows),
        "peer_briefs": [
            {
                "status": _text(b.get("status"), 60),
                "peer": _text(b.get("peer"), 80),
            }
            for b in peer_rows
        ],
        "role": role,
        "duration_sec": run.duration_sec,
        "active_sec": run.active_sec,
        "lane_wait_sec": run.lane_wait_sec,
        "outcome_kind": run.outcome_kind,
        "has_starting_prior": starting_prior is not None,
        "verdict": (prior_outcome or {}).get("verdict"),
    }


def build_stories(rows: RunStoryRows) -> dict[str, RunStory]:
    """Every run the rows describe, keyed by run_id. One pass; the summary
    list and the single-run story are both projections of this."""
    slots = _group(rows)
    prior_claims = {_text(p.get("prior_id"), 200): _text(p.get("claim")) for p in rows.priors}
    prior_meta = {
        _text(p.get("prior_id"), 200): p
        for p in rows.priors
        if _text(p.get("prior_id"), 200)
    }
    prior_lines = {_text(p.get("prior_id"), 200): _text(p.get("line"), 40) for p in rows.priors}
    outreach_by_key = {_text(r.get("correlation_id"), 64): r for r in rows.outreach}
    sent_by_key: dict[str, dict[str, Any]] = {}
    replies_by_key: dict[str, list[dict[str, Any]]] = {}
    for c in rows.chat:
        meta = _obj(c.get("client_meta"))
        reply_to = _text(meta.get("in_reply_to"), 64)
        if reply_to:
            replies_by_key.setdefault(reply_to, []).append(c)
        elif _as_bool(meta.get("unsolicited")):
            sent_by_key.setdefault(_text(c.get("correlation_id"), 64), c)
    readings_by_run: dict[str, list[dict[str, Any]]] = {}
    for r in rows.readings:
        readings_by_run.setdefault(_text(r.get("hop_run_id"), 64), []).append(r)

    out: dict[str, RunStory] = {}
    for run_id, slot in slots.items():
        lifecycle = _lifecycle_sorted(slot["lifecycle"])
        bridge = _lifecycle_from_bridge(lifecycle)
        if slot["events"] or slot["admission"]:
            life = _lifecycle_from_events(slot)
            # The bridge row is a copy of the terminal event; its detail is
            # the fallback when the event carried none.
            if not life.detail:
                life.detail = bridge.detail
            if life.status in (None, STATUS_RUNNING) and bridge.status in _TERMINAL:
                life.status, life.finished_at = bridge.status, bridge.finished_at
        else:
            life = bridge
        detail = life.detail
        line, line_known = _line_for(slot, detail, prior_lines)
        sense_rows = sorted(slot["self_sense"], key=lambda r: _ms(r.get("created_at")) or 0)

        items: list[TimelineItem] = list(life.items)
        graph_stamps: list[int] = []

        roles = sorted(slot["roles"], key=lambda r: _ms(r.get("written_at")) or 0)
        help_rows = sorted(slot["help_requests"], key=lambda r: _ms(r.get("written_at")) or 0)
        peer_rows = sorted(slot["peer_briefs"], key=lambda r: _ms(r.get("written_at")) or 0)
        revisions_early = sorted(slot["revisions"], key=lambda r: _ms(r.get("written_at")) or 0)
        starting_prior = _starting_prior_for_slot(
            help_rows=help_rows,
            revisions=revisions_early,
            prior_claims=prior_claims,
            prior_meta=prior_meta,
        )
        if starting_prior is not None:
            items.append(TimelineItem(at=None, kind="starting_prior", data={
                "prior_id": starting_prior["prior_id"],
                "claim": starting_prior.get("claim") or "",
                "status": starting_prior.get("status"),
                "confidence": starting_prior.get("confidence"),
                "source": starting_prior.get("source"),
                "help_id": starting_prior.get("help_id"),
            }))

        for r in roles:
            at = _ms(r.get("written_at"))
            if at is not None:
                graph_stamps.append(at)
            items.append(TimelineItem(at=at, kind="role_choice", data={
                "choice": _text(r.get("choice"), 60), "why": _text(r.get("why"), _NOTE_LIMIT)}))

        hop_items, hop_attempts = _hop_items(slot["hops"], readings_by_run.get(run_id, []))
        for it in hop_items:
            if it.at is not None:
                graph_stamps.append(it.at)
        items.extend(hop_items)

        for f in slot["findings"]:
            at = _ms(f.get("written_at"))
            if at is not None:
                graph_stamps.append(at)
            items.append(TimelineItem(at=at, kind="finding", data={
                "finding_id": _text(f.get("finding_id"), 200),
                "text": _text(f.get("text")), "evidence": _text(f.get("evidence"))}))

        self_writes = sorted(slot["self_writes"], key=lambda r: _ms(r.get("written_at")) or 0)
        for w in self_writes:
            at = _ms(w.get("written_at"))
            if at is not None:
                graph_stamps.append(at)
            items.append(TimelineItem(at=at, kind="self_write", data={
                "kind": _text(w.get("kind"), 40) or "self_definition",
                "question_id": _text(w.get("question_id"), 120),
                "family": _text(w.get("family"), 40),
                "text": _text(w.get("text")), "evidence": _text(w.get("evidence")),
                "revises": _text(w.get("revises"), 200)}))

        revisions = sorted(slot["revisions"], key=lambda r: _ms(r.get("written_at")) or 0)
        for rev in revisions:
            at = _ms(rev.get("written_at"))
            if at is not None:
                graph_stamps.append(at)
            pid = _text(rev.get("prior_id"), 200)
            items.append(TimelineItem(at=at, kind="revision", data={
                "prior_id": pid, "claim": prior_claims.get(pid, ""),
                "from": _as_float(rev.get("from_confidence")),
                "to": _as_float(rev.get("to_confidence")),
                "from_status": _text(rev.get("from_status"), 60),
                "to_status": _text(rev.get("to_status"), 60)}))

        outcomes = sorted(slot["outcomes"], key=lambda r: _ms(r.get("written_at")) or 0)
        outcome = outcomes[-1] if outcomes else None
        for o in outcomes:
            at = _ms(o.get("written_at"))
            if at is not None:
                graph_stamps.append(at)
            items.append(TimelineItem(at=at, kind="outcome", data={
                "continue_line": _as_bool(o.get("continue_line")),
                "continue_note": _text(o.get("continue_note"), _NOTE_LIMIT),
                "reach_out": _as_bool(o.get("reach_out")),
                "reach_out_why": _text(o.get("reach_out_why"), _NOTE_LIMIT)}))

        for s in sense_rows:
            items.append(TimelineItem(at=_ms(s.get("created_at")), kind="self_sense_answer", data={
                "question_key": _text(s.get("question_key"), 60),
                "question": _text(s.get("question"), 500),
                "answer_text": _text(s.get("answer_text"), _NOTE_LIMIT),
                "answer_source": _text(s.get("answer_source"), 60),
                "self_label_score": _as_int(s["self_label_score"], 0) if s.get("self_label_score") is not None else None,
                "grounded_record_score": _as_int(s["grounded_record_score"], 0) if s.get("grounded_record_score") is not None else None,
            }))

        journals = sorted(slot["journals"], key=lambda r: _ms(r.get("created_at")) or 0)
        journal_body = ""
        journal_entry_id = _text(detail.get("journal_entry_id"), 200)
        for j in journals:
            journal_body = _text(j.get("body"), 20000)
            journal_entry_id = journal_entry_id or _text(j.get("entry_id"), 200)
            items.append(TimelineItem(at=_ms(j.get("created_at")), kind="journal", data={
                "entry_id": _text(j.get("entry_id"), 200), "title": _text(j.get("title"), 80)}))

        reach = _reach_out(
            run_id, outcome=outcome, detail=detail, outreach_by_key=outreach_by_key,
            sent_by_key=sent_by_key, replies_by_key=replies_by_key,
        )
        if reach.decision is not None:
            items.append(TimelineItem(
                at=reach.decided_at or reach.sent_at, kind="outreach",
                data={"decision": reach.decision, "gate": reach.gate,
                      "composed_text": reach.composed_text, "sent_at": reach.sent_at}))
        if reach.reply_at is not None:
            items.append(TimelineItem(at=reach.reply_at, kind="reply",
                                      data={"text": reach.reply_text}))

        # --- clocks and status -----------------------------------------
        # Start = acceptance when the admission path recorded it (the true
        # start of the sitting, lane wait included); else the earliest
        # NON-terminal lifecycle row -- since 2026-09-14 the bridge table only
        # receives `completed`, and that is the end, not the start; else the
        # first graph node; else the first self-sense score.
        sense_stamps = [x for x in (_ms(r.get("created_at")) for r in sense_rows) if x is not None]
        if life.accepted_at is not None:
            started_at, started_from = life.accepted_at, "admission"
        elif life.non_terminal_stamps:
            started_at, started_from = min(life.non_terminal_stamps), life.source
        elif graph_stamps:
            started_at, started_from = min(graph_stamps), "graph"
        elif sense_stamps:
            started_at, started_from = min(sense_stamps), "self_sense"
        else:
            started_at, started_from = None, "none"

        if life.status in _TERMINAL:
            status, finished_at = life.status, life.finished_at
        elif life.status == STATUS_RUNNING:
            status, finished_at = STATUS_RUNNING, None
        elif outcome is not None:
            # The graph says Orion wrote its end-of-turn note; nothing in
            # Postgres says Hub finished. Not "completed" -- the finish row
            # is the authority and it is missing.
            status, finished_at = STATUS_UNKNOWN, _ms(outcome.get("written_at"))
        elif sense_rows:
            status = STATUS_COMPLETED if len(sense_rows) >= len(SELF_SENSE_QUESTIONS) else STATUS_UNKNOWN
            finished_at = max(sense_stamps) if sense_stamps else None
        else:
            status, finished_at = STATUS_UNKNOWN, None

        attempts: Optional[int] = None
        if detail.get("attempts") is not None:
            attempts = _as_int(detail.get("attempts"), 0) or None
        if attempts is None and life.retries:
            attempts = life.retries + 1
        if attempts is None and lifecycle and life.source == "lifecycle":
            attempts = 1 + sum(1 for r in lifecycle if _text(r.get("status"), 20) == "resumed")
        if attempts is None:
            attempts = hop_attempts

        error = life.error

        prior_touched: Optional[dict[str, Any]] = None
        if revisions:
            rev = revisions[-1]
            pid = _text(rev.get("prior_id"), 200)
            prior_touched = {
                "prior_id": pid, "claim": prior_claims.get(pid, ""),
                "from": _as_float(rev.get("from_confidence")),
                "to": _as_float(rev.get("to_confidence")),
                "from_status": _text(rev.get("from_status"), 60),
                "to_status": _text(rev.get("to_status"), 60),
            }

        self_written: Optional[dict[str, Any]] = None
        if _text(detail.get("self_definition")):
            self_written = {"kind": "self_definition", "text": _text(detail.get("self_definition"))}
        elif _text(detail.get("lived_answer")):
            self_written = {"kind": "lived_answer", "text": _text(detail.get("lived_answer")),
                            "family": _text(detail.get("self_question_family"), 60)}
        elif self_writes:
            w = self_writes[-1]
            self_written = {"kind": _text(w.get("kind"), 40) or "self_definition",
                            "text": _text(w.get("text")), "family": _text(w.get("family"), 60)}

        self_sense: Optional[dict[str, Any]] = None
        if sense_rows:
            self_sense = {
                "questions_answered": len(sense_rows),
                "questions_expected": len(SELF_SENSE_QUESTIONS),
                "self_definition_version": next(
                    (_as_int(r.get("self_definition_version"), 0) for r in sense_rows
                     if r.get("self_definition_version") is not None), None),
                "scores": [
                    {"question_key": _text(r.get("question_key"), 60),
                     "self_label_score": _as_int(r["self_label_score"], 0) if r.get("self_label_score") is not None else None,
                     "grounded_record_score": _as_int(r["grounded_record_score"], 0) if r.get("grounded_record_score") is not None else None}
                    for r in sense_rows
                ],
            }

        harness: Optional[dict[str, Any]] = None
        if detail.get("harness_elapsed_sec") is not None or detail.get("turn_correlation_id"):
            harness = {
                "elapsed_sec": _as_float(detail.get("harness_elapsed_sec")),
                "turn_correlation_id": _text(detail.get("turn_correlation_id"), 64) or None,
            }

        wrote = (len(slot["hops"]) + len(slot["findings"]) + len(revisions)
                 + len(sense_rows) + len(self_writes))
        summary = RunSummary(
            run_id=run_id,
            line=line,
            line_known=line_known,
            plain_line_label=LINE_LABELS[line],
            started_at=started_at,
            started_from=started_from,
            finished_at=finished_at,
            accepted_at=life.accepted_at,
            admitted_at=life.admitted_at,
            lane=life.lane,
            retries=life.retries,
            anomalies={k: v for k, v in slot["event_counts"].items() if k in ANOMALY_EVENTS and v},
            status=status,
            attempts=attempts,
            error=error,
            hops=len(slot["hops"]),
            findings=len(slot["findings"]),
            revisions=len(revisions),
            prior_touched=prior_touched,
            reach_out=reach,
            journal_entry_id=journal_entry_id,
            finding_text=_text(detail.get("finding_text"), 600),
            self_written=self_written,
            self_sense=self_sense,
            harness=harness,
            outcome_kind=_outcome_kind(status=status, reach=reach, wrote=wrote),
        )
        dated_hops = [it.at for it in hop_items if it.at is not None]
        anchor = max(dated_hops) if dated_hops else None
        items.sort(key=lambda it: it.sort_key(anchor))

        role_summary = None
        if roles:
            r0 = roles[0]
            role_summary = {
                "choice": _text(r0.get("choice"), 60),
                "why": _text(r0.get("why"), _NOTE_LIMIT),
            }
        prior_outcome = _prior_outcome_block(
            starting_prior=starting_prior,
            prior_touched=_subject_prior_revision(revisions, starting_prior, prior_claims),
            self_written=self_written,
            peer_rows=peer_rows,
        )
        summary_card = _summary_card(
            run=summary,
            role=role_summary,
            help_rows=help_rows,
            peer_rows=peer_rows,
            starting_prior=starting_prior,
            prior_outcome=prior_outcome,
        )
        out[run_id] = RunStory(
            run=summary,
            timeline=items,
            journal_body=journal_body,
            readings_available=bool(readings_by_run.get(run_id)),
            starting_prior=starting_prior,
            summary=summary_card,
            prior_outcome=prior_outcome,
        )
    return out


def summaries(stories: dict[str, RunStory], *, line: str = "all") -> list[RunSummary]:
    """Newest first. A run with no clock at all sorts last, not first."""
    runs = [s.run for s in stories.values() if line == "all" or s.run.line == line]
    runs.sort(
        key=lambda r: (
            (r.started_at or r.finished_at) is not None,
            r.started_at or r.finished_at or 0,
        ),
        reverse=True,
    )
    return runs


def reach_out_totals(runs: list[RunSummary]) -> dict[str, Any]:
    """The tile: how many runs wanted to reach Juniper, how many did, and
    what stopped the rest. `not_recorded` is counted separately so a pre-check
    block that left no row is not passed off as a named gate."""
    wanted = [r for r in runs if r.reach_out.wanted]
    sent = sum(1 for r in wanted if r.reach_out.sent)
    gates: dict[str, int] = {}
    not_recorded = 0
    for r in wanted:
        d = r.reach_out.decision
        if d == DECISION_SENT:
            continue
        if d == DECISION_NOT_RECORDED or d is None:
            not_recorded += 1
            continue
        gates[d] = gates.get(d, 0) + 1
    top = max(gates.items(), key=lambda kv: kv[1])[0] if gates else None
    return {
        "wanted": len(wanted),
        "sent": sent,
        "blocked_by": gates,
        "top_block_reason": top,
        "not_recorded": not_recorded,
    }


# --- payloads -------------------------------------------------------------


def _reach_payload(r: ReachOut) -> dict[str, Any]:
    return {
        "wanted": r.wanted,
        "why": r.why,
        "decision": r.decision,
        "gate": r.gate,
        "decided_at": r.decided_at,
        "sent_at": r.sent_at,
        "composed_text": r.composed_text,
        "reply": {"at": r.reply_at, "text": r.reply_text} if r.reply_at is not None else None,
    }


def run_to_payload(r: RunSummary) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "line": r.line,
        "line_known": r.line_known,
        "plain_line_label": r.plain_line_label,
        "started_at": r.started_at,
        "started_at_iso": _iso(r.started_at),
        "started_from": r.started_from,
        "finished_at": r.finished_at,
        "finished_at_iso": _iso(r.finished_at),
        "duration_sec": r.duration_sec,
        "accepted_at": r.accepted_at,
        "admitted_at": r.admitted_at,
        "lane": r.lane,
        "lane_wait_sec": r.lane_wait_sec,
        "active_sec": r.active_sec,
        "retries": r.retries,
        "anomalies": r.anomalies,
        "status": r.status,
        "attempts": r.attempts,
        "error": r.error,
        "hops": r.hops,
        "findings": r.findings,
        "revisions": r.revisions,
        "prior_touched": r.prior_touched,
        "reach_out": _reach_payload(r.reach_out),
        "journal_entry_id": r.journal_entry_id,
        "finding_text": r.finding_text,
        "self_written": r.self_written,
        "self_sense": r.self_sense,
        "harness": r.harness,
        "outcome_kind": r.outcome_kind,
    }


def story_to_payload(s: RunStory) -> dict[str, Any]:
    start = s.run.started_at
    timeline = []
    for it in s.timeline:
        data = dict(it.data)
        # LivedAnswer / SelfDefinition rows carry nested `kind`; do not let
        # that overwrite the timeline event kind (self_write) or the UI
        # falls through to a bare label with no text.
        write_kind = None
        if it.kind == "self_write":
            write_kind = _text(data.pop("kind", None), 40) or "self_definition"
        entry = {
            "at": it.at,
            "at_iso": _iso(it.at),
            # Relative clock the page prints as +m:ss; None when either
            # end is unknown rather than a fake 0.
            "offset_sec": (
                round((it.at - start) / 1000, 1)
                if it.at is not None and start is not None
                else None
            ),
            "kind": it.kind,
            "attempt": it.attempt,
            **data,
        }
        if write_kind is not None:
            entry["write_kind"] = write_kind
        timeline.append(entry)
    return {
        "run": run_to_payload(s.run),
        "timeline": timeline,
        "journal_body": s.journal_body,
        "readings_available": s.readings_available,
        "harness": s.run.harness,
        "starting_prior": s.starting_prior,
        "summary": s.summary,
        "prior_outcome": s.prior_outcome,
    }
