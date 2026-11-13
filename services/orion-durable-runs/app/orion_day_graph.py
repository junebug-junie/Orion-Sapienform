"""The `orion_day.letter` admitted durable workflow: Orion's letter about yesterday, as ONE unified turn.

Design: orion/schemas/orion_day.py (contract; living Prime #3224 Part 3), orion/orion_day/letter_turn.py
(the turn's prompt and how its reply splits), orion/orion_day/ (gather + budget, run by Hub on Prime's order).

    resource_request -> resource_wait -> write_letter -> persist -> finish
                                         write_letter (failed attempt) -> retry_wait -> resource_request
                                         persist (retry) -> retry_wait -> persist
                                         (failed on a terminal error)

* ``write_letter`` runs ONE unified turn on the durable turn seam (``app/harness_step.py``) under the
  run's GPU pool hold: session ``ORION_DAY_LETTER_SESSION_ID``, source tag ``ORION_DAY_LETTER_SOURCE_TAG``,
  ``require_lookup=False`` (Orion may dig and recall, but a pure writing turn must not fail Hub's lookup
  gate). The reply is split by ``split_letter`` into the letter (``note_md``) and its closing threads
  (``carry_forward_md`` + ``threads``). Same exception policy as ``harness_smoke_graph``: a lost hold
  replays the node (never an attempt); a refusal, a timeout, an empty reply, a transport error or a
  letter under ``MIN_NOTE_CHARS`` is one failed attempt (``failed_attempt``: backoff in ``retry_wait``,
  then a fresh hold); past the budget the run fails with no row. The step record (full text + the turn's
  correlation id) and the gated texts are checkpointed together, and a checkpointed letter is never
  regenerated. The hold is released as soon as the letter is checkpointed: ``persist`` needs no GPU.
* The checkpoint carries a SLIM brief (``slim_brief``: no material, no digest). The full brief
  (~1 MB on a heavy day) lives once in the accepted request row (durable_admission_runs.request)
  and each node that needs it reads it from there (``OrionDayDeps.load_brief``).
* ``persist`` writes ``orion_day_letter`` with ``INSERT ... ON CONFLICT (letter_date) DO NOTHING``
  (one letter per day, first writer wins), then publishes the STORED row's note -- never the
  threads -- as a journal entry with a stable id (uuid5 of the date) and the row's own
  ``created_at``, so every replay (and a later run for the same day) republishes the identical
  entry. An insert failure retries and eventually fails the run (no letter exists). Once the
  row exists the run never fails over the journal: publish retries on its own budget, and if it
  is exhausted the run still completes with ``journal_published=False``.
  ``carry_forward_expires_at`` is written NULL: the curiosity carry-forward offer is retired; the
  threads reach curiosity through Prime, which reads them from ``finish_detail`` (``GET /runs/{run_id}``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt, replay_if_requeued,
    resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, HarnessStepFailed, RunTurn, harness_step_correlation_id, run_harness_step,
)
from orion.journaler.schemas import JournalEntryWriteV1
from orion.orion_day.budget import chronology_citations, extract_refs, gate_event_refs
from orion.orion_day.letter_turn import MAX_THREADS, NO_THREADS_MD, build_letter_prompt, split_letter
from orion.schemas.temporal_self import DayCitationCheckV1
from orion.temporal_self.citations import check_arc_citations
from orion.schemas.orion_day import (
    ORION_DAY_JOURNAL_SOURCE_KIND,
    ORION_DAY_JOURNAL_TRIGGER_KIND,
    ORION_DAY_LETTER_SESSION_ID,
    ORION_DAY_LETTER_SOURCE_TAG,
    OrionDayLetterSourcesV1,
    OrionDayRunBriefV1,
    orion_day_journal_entry_id,
)

logger = logging.getLogger(__name__)

# A letter shorter than this is not a letter (a refusal, a stub): one failed attempt.
MIN_NOTE_CHARS = 400
LETTER_STEP_KEY = "letter"
# Journal publish attempts once the letter row exists (never fails the run).
JOURNAL_MAX_ATTEMPTS = 8

# Brief keys kept in the checkpoint; everything else is read from the stored request row.
SLIM_BRIEF_KEYS = ("letter_date", "timezone", "window_start", "window_end", "llm_route", "timeout_sec",
                   "prime_act_id", "missed_letters")


class OrionDayState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]   # slim_brief(...)
    admission: dict[str, Any]
    requested_at: str
    attempt: int            # failed attempts of write_letter; reset when the letter lands
    hold_takebacks: int
    listener_waits: int   # RPCs published to nobody this run waited out (listener_wait; never an attempt)
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    steps: dict[str, dict[str, Any]]    # "letter" -> the turn's step record (full text, turn_correlation_id)
    harness_turn_meta: dict[str, Any]   # the last issued turn's identity (failed-run detail, cancel)
    note_md: str | None
    carry_forward_md: str | None
    threads: list[str]
    carry_forward_refs: dict[str, int]
    # Temporal Self patch 4: the arc citation gate's result per text (DayCitationCheckV1 json).
    note_arc_citations: dict[str, Any]
    carry_forward_arc_citations: dict[str, Any]
    # [event:] refs removed because the temporal core did not show them.
    note_rejected_events: list[str]
    carry_forward_rejected_events: list[str]
    letter_written: bool
    persisted: bool
    persist_outcome: str | None
    journal_entry_id: str | None
    journal_published: bool
    existing_run_id: str | None


class LetterTooShort(HarnessStepFailed):
    """The turn answered, but the letter part is under ``MIN_NOTE_CHARS``. One failed attempt
    (a RuntimeError via HarnessStepFailed: retryable), never a success."""


def slim_brief(brief: dict[str, Any]) -> dict[str, Any]:
    return {k: brief[k] for k in SLIM_BRIEF_KEYS if k in brief}


@dataclass
class OrionDayDeps:
    # The Hub turn RPC (Runner._run_turn): CuriosityTurnRequestV1 -> CuriosityTurnResultV1.
    run_turn: RunTurn
    # (row: dict) -> the STORED row for that letter_date: {"run_id", "note_md", "created_at"}
    # (this run's, or an earlier run's).
    persist_letter: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    # (JournalEntryWriteV1) -> entry_id on success, None on failure
    publish_journal: Callable[[JournalEntryWriteV1], Awaitable[str | None]]
    # (state) -> the FULL brief from the accepted request row (the checkpoint holds slim_brief)
    load_brief: Callable[[dict[str, Any]], Awaitable[OrionDayRunBriefV1]]


def gate_arc_citations(text: str, brief: OrionDayRunBriefV1) -> tuple[str, DayCitationCheckV1 | None]:
    """Temporal Self patch 4: arc ids the closed-day block and the temporal core did not show are
    removed from the text and counted; a sentence naming their labels or clock times without an
    [arc:] is counted and labelled UNVERIFIED (never retried). None when the letter had neither
    block and cites no arc."""
    shown, labels = chronology_citations(brief.material)
    has_block = brief.material.temporal_day is not None or brief.material.temporal_core is not None
    cleaned, check = check_arc_citations(text, shown, labels=labels if has_block else None)
    if not has_block and not check.rejected_unknown:
        return text, None
    return cleaned, check


def check_note(text: str, brief: OrionDayRunBriefV1) -> dict[str, Any]:
    """The letter part: the floor (a failed attempt below it), then the event and arc gates."""
    if len(text) < MIN_NOTE_CHARS:
        raise LetterTooShort(f"orion_day_letter: letter too short ({len(text)} chars < {MIN_NOTE_CHARS})")
    text, bad_events = gate_event_refs(text, brief.material)
    out: dict[str, Any] = {"note_md": text}
    if bad_events:
        out["note_rejected_events"] = bad_events
    cleaned, check = gate_arc_citations(text, brief)
    if check is not None:
        out.update(note_md=cleaned, note_arc_citations=check.model_dump(mode="json"))
    return out


def _bullets(md: str) -> list[str]:
    return [line[2:].strip() for line in md.splitlines() if line.startswith("- ") and line[2:].strip()]


def check_threads(carry_forward_md: str, brief: OrionDayRunBriefV1) -> dict[str, Any]:
    """The threads part: the same event and arc gates the carry-forward had. No floor -- a letter that
    names no thread keeps ``NO_THREADS_MD`` and Prime opens no concern from it. Refs are counted against
    the digest's refs (reported, not enforced: a thread without a bracket is still a thread)."""
    if carry_forward_md == NO_THREADS_MD:
        return {"carry_forward_md": carry_forward_md, "threads": [], "carry_forward_refs": {"valid": 0, "unknown": 0}}
    text, bad_events = gate_event_refs(carry_forward_md, brief.material)
    out: dict[str, Any] = {}
    if bad_events:
        out["carry_forward_rejected_events"] = bad_events
    text, check = gate_arc_citations(text, brief)
    if check is not None:
        out["carry_forward_arc_citations"] = check.model_dump(mode="json")
    threads = _bullets(text)
    if not threads:   # every bullet was only refs the gates removed
        text, threads = NO_THREADS_MD, []
    refs = extract_refs(text)
    known = set(brief.llm_view.included_refs)
    return {**out, "carry_forward_md": text, "threads": threads,
            "carry_forward_refs": {"valid": sum(1 for r in refs if r in known),
                                   "unknown": sum(1 for r in refs if r not in known)}}


def check_letter(text: str, brief: OrionDayRunBriefV1) -> dict[str, Any]:
    """Split the turn's reply and gate both parts. Raises (a failed attempt) on a too-short letter."""
    parts = split_letter(text)
    note = check_note(parts.note_md, brief)
    threads = check_threads(parts.carry_forward_md, brief)
    if note["note_md"].strip() == threads["carry_forward_md"].strip():
        # orion_day_letter requires two distinct texts (OrionDayLetterV1._separate).
        raise HarnessStepFailed("orion_day_letter: letter and threads are the same text")
    return {**note, **threads}


def letter_row(state: dict, brief: OrionDayRunBriefV1, now: datetime) -> dict[str, Any]:
    step = (state.get("steps") or {}).get(LETTER_STEP_KEY) or {}
    sources = OrionDayLetterSourcesV1(
        by_source=brief.material.sources,
        condensed=brief.llm_view.condensed,
        approx_tokens=brief.llm_view.approx_tokens,
        budget_tokens=brief.llm_view.budget_tokens,
        input_chars=dict(brief.llm_view.source_chars),
        input_shares=dict(brief.llm_view.source_shares),
        rejected_event_refs={name: list(state[key]) for name, key in (("note", "note_rejected_events"),
                                                                       ("carry_forward", "carry_forward_rejected_events"))
                             if state.get(key)},
        arc_citations={name: DayCitationCheckV1.model_validate(state[key])
                       for name, key in (("note", "note_arc_citations"),
                                         ("carry_forward", "carry_forward_arc_citations"))
                       if state.get(key)},
        turn_correlation_id=step.get("turn_correlation_id"),
        prime_act_id=brief.prime_act_id,
        missed_letters=list(brief.missed_letters),
    )
    return {
        "letter_date": brief.letter_date,
        "run_id": state["run_id"],
        "window_start": brief.window_start,
        "window_end": brief.window_end,
        "note_md": state["note_md"],
        "carry_forward_md": state["carry_forward_md"],
        "material": brief.material.model_dump(mode="json"),
        "sources": sources.model_dump(mode="json"),
        "journal_entry_id": orion_day_journal_entry_id(brief.letter_date),
        "created_at": now,
        # The curiosity carry-forward offer is retired (living Prime Part 3): never offered again.
        "carry_forward_expires_at": None,
    }


def journal_entry(stored: dict[str, Any], brief: OrionDayRunBriefV1) -> JournalEntryWriteV1:
    """The journal row for the day: the STORED note, the stored row's created_at, and the run that
    wrote it as correlation id, so every publish of a given day is identical whichever run sends it."""
    return JournalEntryWriteV1(
        entry_id=orion_day_journal_entry_id(brief.letter_date),
        created_at=stored["created_at"],
        author="orion",
        mode="daily",
        title=f"Orion's day -- {brief.letter_date.isoformat()}",
        body=stored["note_md"],
        source_kind=ORION_DAY_JOURNAL_SOURCE_KIND,
        source_ref=f"orion_day:{brief.letter_date.isoformat()}",
        correlation_id=str(stored["run_id"]),
        trigger_kind=ORION_DAY_JOURNAL_TRIGGER_KIND,
    )


def letter_turn_correlation_id(state: dict) -> str | None:
    """The unified turn that wrote this run's checkpointed letter, or None before it lands."""
    step = (state.get("steps") or {}).get(LETTER_STEP_KEY) or {}
    return step.get("turn_correlation_id")


def finish_detail(state: dict) -> dict[str, Any]:
    """Small, bounded facts for the terminal state event -- never the letter or the material.
    Hub reads the letter itself from orion_day_letter; ``persist_outcome`` (not ``completed``)
    says whether THIS run wrote it. Prime reads ``threads`` (at most ``MAX_THREADS``, each already
    cut by split_letter), ``turn_correlation_id`` and ``prime_act_id`` from ``GET /runs/{run_id}``."""
    brief = state.get("brief") or {}
    step = (state.get("steps") or {}).get(LETTER_STEP_KEY) or {}
    return {
        "line": "orion_day",
        "letter_date": brief.get("letter_date"),
        "prime_act_id": brief.get("prime_act_id"),
        "turn_correlation_id": step.get("turn_correlation_id"),
        "persisted": bool(state.get("persisted")),
        "persist_outcome": state.get("persist_outcome"),
        "existing_run_id": state.get("existing_run_id"),
        "journal_entry_id": state.get("journal_entry_id"),
        "journal_published": bool(state.get("journal_published")),
        "note_chars": len(state.get("note_md") or ""),
        "carry_forward_chars": len(state.get("carry_forward_md") or ""),
        "threads": list(state.get("threads") or [])[:MAX_THREADS],
        "threads_total": len(state.get("threads") or []),
        "carry_forward_refs": dict(state.get("carry_forward_refs") or {}),
        "arc_citations": {name: {"label": c.get("label"), "cited": len(c.get("cited_valid") or []),
                                 "rejected": len(c.get("rejected_unknown") or []),
                                 "uncited": c.get("uncited_day_sentences", 0)}
                          for name, c in (("note", state.get("note_arc_citations")),
                                          ("carry_forward", state.get("carry_forward_arc_citations"))) if c},
        "letter_attempts": step.get("attempts"),
        "harness_step_count": (step.get("debug") or {}).get("harness_step_count"),
        "hold_takebacks": int(state.get("hold_takebacks") or 0),
    }


def build_orion_day_graph(deps: OrionDayDeps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)

    def backoff(attempts: int) -> str:
        delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempts - 1))
        return (admission.now() + timedelta(seconds=delay)).isoformat()

    async def write_letter(state: OrionDayState) -> dict:
        state = dict(state)
        if LETTER_STEP_KEY in (state.get("steps") or {}) and state.get("note_md"):
            return {"status": "running"}    # resumed after the letter's checkpoint: never re-run the turn
        timeout_sec = float((state.get("brief") or {}).get("timeout_sec") or 1800.0)
        window = attempt_timeout_sec(state, timeout_sec, admission.now())   # the node timer, deadline-clamped
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC)
        correlation_id = harness_step_correlation_id(state, LETTER_STEP_KEY)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": LETTER_STEP_KEY}

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            brief = await deps.load_brief(held)
            record = await run_harness_step(
                deps.run_turn, held, step_key=LETTER_STEP_KEY, prompt=build_letter_prompt(brief),
                correlation_id=correlation_id, timeout_sec=turn_timeout,
                source_tag=ORION_DAY_LETTER_SOURCE_TAG, session_id=ORION_DAY_LETTER_SESSION_ID,
                retrieval_query=f"my day {brief.letter_date.isoformat()}", require_lookup=False)
            # Inside the attempt: a too-short letter is a failed attempt, never a checkpointed one.
            return {"record": record, "gated": check_letter(record["text"], brief)}

        try:
            result = await admission.execute({**state, "step_correlation_id": correlation_id}, operation)
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline", "harness_turn_meta": turn_meta}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:   # HoldPreempted too: the pool keeps its place, no attempt spent
            update = await taken_back(admission, state, exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "waiting_resource"})
            return {**update, "harness_turn_meta": turn_meta}
        except Exception as exc:  # noqa: BLE001 -- refusal, timeout, empty/short letter, transport: the shared policy
            # A failed turn may be the pool's (it took the hold back mid-turn): replay, no attempt.
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "harness_turn_meta": turn_meta}
            return await failed_attempt(admission, state, exc, waiting={"status": "retrying", "retry_node": None},
                                        extra={"harness_turn_meta": turn_meta}, attempt_sec=timeout_sec)
        record = result["record"]
        record["attempts"] = int(state.get("attempt") or 0) + 1
        update = {**result["gated"], "steps": {**(state.get("steps") or {}), LETTER_STEP_KEY: record},
                  "harness_turn_meta": turn_meta, "status": "running", "attempt": 0, "retry_at": None,
                  "retry_node": None, "last_error": None}
        # The letter is checkpointed with this update: the rest needs no GPU. Hand the hold back --
        # outside the try, so a release hiccup can never discard a finished letter.
        try:
            update.update(await admission.release({**state, **update}, "completed"))
        except Exception:  # noqa: BLE001 -- finish releases again; the pool's TTL is the backstop
            logger.warning("orion_day_hold_release_failed run=%s", state.get("run_id"), exc_info=True)
        return update

    async def persist(state: OrionDayState) -> dict:
        now = admission.now()
        attempts = dict(state.get("tail_attempts") or {})
        try:
            brief = await deps.load_brief(dict(state))
            stored = await deps.persist_letter(letter_row(dict(state), brief, now))
        except RunControlPending:
            raise
        except Exception as exc:  # noqa: BLE001 -- no row yet: back off and retry, then fail
            attempts["persist"] = attempts.get("persist", 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempts["persist"] >= admission.max_attempts:
                return {"status": "failed", "tail_attempts": attempts, "last_error": error}
            return {"status": "retrying", "tail_attempts": attempts, "last_error": error,
                    "retry_node": "persist", "retry_at": backoff(attempts["persist"])}
        ours = str(stored["run_id"]) == state["run_id"]
        if not ours:
            logger.warning("orion_day_letter_already_written run=%s existing=%s", state["run_id"], stored["run_id"])
        outcome = {"letter_written": True, "persisted": ours,
                   "persist_outcome": "written" if ours else "already_written",
                   "existing_run_id": None if ours else str(stored["run_id"])}
        # The row exists from here on: the journal never fails the run.
        entry = journal_entry(stored, brief)
        try:
            published = await deps.publish_journal(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("orion_day_journal_publish_failed run=%s err=%s", state["run_id"], exc)
            published = None
        if published:
            return {**outcome, "status": "running", "journal_entry_id": entry.entry_id, "journal_published": True,
                    "retry_node": None, "last_error": None}
        attempts["journal"] = attempts.get("journal", 0) + 1
        if attempts["journal"] >= JOURNAL_MAX_ATTEMPTS:
            logger.error("orion_day_journal_gave_up run=%s letter_date=%s", state["run_id"], brief.letter_date)
            return {**outcome, "status": "running", "journal_entry_id": None, "journal_published": False,
                    "tail_attempts": attempts, "retry_node": None, "last_error": "journal_publish_failed"}
        return {**outcome, "status": "retrying", "tail_attempts": attempts, "last_error": "journal_publish_failed",
                "retry_node": "persist", "retry_at": backoff(attempts["journal"])}

    async def finish(state: OrionDayState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: OrionDayState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_letter(state) -> str:
        return {"waiting_resource": "resource_request", "retrying": "retry_wait",
                "failed": "failed"}.get(state.get("status"), "persist")

    def after_grant(state) -> str:
        decision = after_wait(state)
        if decision != "granted":
            return decision
        # A run resumed with its letter already checkpointed goes straight on to persist.
        return "persist" if state.get("note_md") else "write_letter"

    g = StateGraph(OrionDayState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "write_letter": write_letter, "persist": persist, "retry_wait": retry_wait,
                       "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_grant,
                            {"write_letter": "write_letter", "persist": "persist",
                             "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("write_letter", after_letter)
    g.add_conditional_edges("persist", lambda s: {"retrying": "retry_wait", "failed": "failed"}.get(s.get("status"), "finish"))
    # A letter retry goes back for the hold (resource_request); a persist retry needs none.
    g.add_conditional_edges("retry_wait", lambda s: "persist" if s.get("retry_node") == "persist" else "resource_request",
                            {"persist": "persist", "resource_request": "resource_request"})
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
