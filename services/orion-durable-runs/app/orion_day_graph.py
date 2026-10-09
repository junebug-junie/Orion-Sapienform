"""The `orion_day.letter` admitted durable workflow: Orion's note about yesterday.

Design: orion/schemas/orion_day.py (contract), orion/orion_day/ (gather + budget, run by Hub).

    resource_request -> resource_wait -> write_note -> write_carry_forward
      -> persist -> finish            (failed on a terminal error; retry_wait between attempts)

* ``write_note`` / ``write_carry_forward`` each make ONE cortex verb call under the run's GPU
  pool hold (``options.gpu_lease``) and checkpoint their text. Two calls, two fields: the note
  prompt carries no instruction about future curiosity; the carry-forward prompt gets the
  finished note plus the digest. A lost/recalled hold replays the node (never an attempt). A
  transport failure, an error/empty/truncated completion (``runner.strict_final_text``) or a
  completion below the node's floor is one attempt; the next attempt waits out a backoff in
  ``retry_wait`` (a cortex restart must not burn the budget in seconds); past the budget the
  run fails. A restart resumes at the first node without a checkpointed text -- a finished
  note is never regenerated.
* The hold is released as soon as the carry-forward text is checkpointed: ``persist`` needs no GPU.
* The checkpoint carries a SLIM brief (``slim_brief``: no material, no digest). The full brief
  (~1 MB on a heavy day) lives once in the accepted request row (durable_admission_runs.request)
  and each node that needs it reads it from there (``OrionDayDeps.load_brief``) -- otherwise
  every checkpoint of the run would copy it, and the runner's resume sweep loads every
  checkpoint's blobs.
* ``persist`` writes ``orion_day_letter`` with ``INSERT ... ON CONFLICT (letter_date) DO NOTHING``
  (one letter per day, first writer wins), then publishes the STORED row's note -- never the
  carry-forward -- as a journal entry with a stable id (uuid5 of the date) and the row's own
  ``created_at``, so every replay (and a later run for the same day) republishes the identical
  entry. An insert failure retries and eventually fails the run (no letter exists). Once the
  row exists the run never fails over the journal: publish retries on its own budget, and if it
  is exhausted the run still completes with ``journal_published=False``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, resource_nodes, taken_back,
)
from orion.journaler.schemas import JournalEntryWriteV1
from orion.orion_day.budget import extract_refs
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.orion_day import (
    ORION_DAY_CARRY_FORWARD_VERB,
    ORION_DAY_JOURNAL_SOURCE_KIND,
    ORION_DAY_JOURNAL_TRIGGER_KIND,
    ORION_DAY_NOTE_VERB,
    OrionDayLetterSourcesV1,
    OrionDayRunBriefV1,
    orion_day_journal_entry_id,
)

logger = logging.getLogger(__name__)

# A completion shorter than this is not a note (a refusal, a stub).
MIN_NOTE_CHARS = 400
MIN_CARRY_FORWARD_CHARS = 40
LLM_WORK_NODES = frozenset({"write_note", "write_carry_forward"})
# The verb RPC gives up this long before AdmissionRuntime.execute's per-node timeout (both derive
# from brief.timeout_sec), so a slow call ends as the RPC's own clean error, not the node's.
RPC_TIMEOUT_MARGIN_SEC = 30.0
# Journal publish attempts once the letter row exists (never fails the run).
JOURNAL_MAX_ATTEMPTS = 8
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", flags=re.MULTILINE)

# Brief keys kept in the checkpoint; everything else is read from the stored request row.
SLIM_BRIEF_KEYS = ("letter_date", "timezone", "window_start", "window_end", "llm_route", "timeout_sec",
                   "carry_forward_ttl_hours")


class OrionDayState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]   # slim_brief(...)
    admission: dict[str, Any]
    requested_at: str
    attempt: int
    llm_attempts: dict[str, int]
    hold_takebacks: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    note_md: str | None
    carry_forward_md: str | None
    carry_forward_refs: dict[str, int]
    letter_written: bool
    persisted: bool
    persist_outcome: str | None
    journal_entry_id: str | None
    journal_published: bool
    existing_run_id: str | None


class EmptyGeneration(RuntimeError):
    """The verb answered, but with nothing usable. Counted as an attempt, never a success."""


def slim_brief(brief: dict[str, Any]) -> dict[str, Any]:
    return {k: brief[k] for k in SLIM_BRIEF_KEYS if k in brief}


@dataclass
class OrionDayDeps:
    # (verb, metadata, llm_route, *, gpu_lease, timeout_sec, user_text) -> final answer text; raises
    # on transport failure / non-ok result / empty, error-framed or truncated text.
    call_verb_text: Callable[..., Awaitable[str]]
    # (row: dict) -> the STORED row for that letter_date: {"run_id", "note_md", "created_at"}
    # (this run's, or an earlier run's).
    persist_letter: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    # (JournalEntryWriteV1) -> entry_id on success, None on failure
    publish_journal: Callable[[JournalEntryWriteV1], Awaitable[str | None]]
    # (state) -> the FULL brief from the accepted request row (the checkpoint holds slim_brief)
    load_brief: Callable[[dict[str, Any]], Awaitable[OrionDayRunBriefV1]]


def _lease_ref(state: dict) -> GpuLeaseRefV1 | None:
    lease = state.get("lease")
    if not lease:
        return None
    return GpuLeaseRefV1.model_validate({k: lease[k] for k in ("lease_id", "generation", "role", "holder")})


def note_metadata(brief: OrionDayRunBriefV1) -> dict[str, Any]:
    """The note verb's whole input. Deliberately nothing about future curiosity."""
    return {"orion_day_input": {
        "letter_date": brief.letter_date.isoformat(),
        "timezone": brief.timezone,
        "digest_md": brief.llm_view.digest_md,
    }}


def carry_forward_metadata(brief: OrionDayRunBriefV1, note_md: str) -> dict[str, Any]:
    return {"orion_day_input": {
        "letter_date": brief.letter_date.isoformat(),
        "timezone": brief.timezone,
        "digest_md": brief.llm_view.digest_md,
        "note_md": note_md,
    }}


def check_note(text: str, brief: OrionDayRunBriefV1) -> dict[str, Any]:
    if len(text) < MIN_NOTE_CHARS:
        raise EmptyGeneration(f"{ORION_DAY_NOTE_VERB}: completion too short ({len(text)} chars < {MIN_NOTE_CHARS})")
    return {}


def check_carry_forward(text: str, brief: OrionDayRunBriefV1) -> dict[str, Any]:
    """Structural floor: at least one list item. Its refs are counted against the digest's refs
    (reported, not enforced: a thread without a bracket is still a thread)."""
    if len(text) < MIN_CARRY_FORWARD_CHARS:
        raise EmptyGeneration(f"{ORION_DAY_CARRY_FORWARD_VERB}: completion too short ({len(text)} chars)")
    if not _LIST_ITEM_RE.search(text):
        raise EmptyGeneration(f"{ORION_DAY_CARRY_FORWARD_VERB}: no list item in the completion")
    refs = extract_refs(text)
    known = set(brief.llm_view.included_refs)
    return {"carry_forward_refs": {"valid": sum(1 for r in refs if r in known),
                                   "unknown": sum(1 for r in refs if r not in known)}}


def letter_row(state: dict, brief: OrionDayRunBriefV1, now: datetime) -> dict[str, Any]:
    sources = OrionDayLetterSourcesV1(
        by_source=brief.material.sources,
        condensed=brief.llm_view.condensed,
        approx_tokens=brief.llm_view.approx_tokens,
        budget_tokens=brief.llm_view.budget_tokens,
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
        "carry_forward_expires_at": now + timedelta(hours=brief.carry_forward_ttl_hours),
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


def finish_detail(state: dict) -> dict[str, Any]:
    """Small, bounded facts for the terminal state event -- never the texts or the material.
    Hub reads the letter itself from orion_day_letter; ``persist_outcome`` (not ``completed``)
    says whether THIS run wrote it."""
    brief = state.get("brief") or {}
    return {
        "line": "orion_day",
        "letter_date": brief.get("letter_date"),
        "persisted": bool(state.get("persisted")),
        "persist_outcome": state.get("persist_outcome"),
        "existing_run_id": state.get("existing_run_id"),
        "journal_entry_id": state.get("journal_entry_id"),
        "journal_published": bool(state.get("journal_published")),
        "note_chars": len(state.get("note_md") or ""),
        "carry_forward_chars": len(state.get("carry_forward_md") or ""),
        "carry_forward_refs": dict(state.get("carry_forward_refs") or {}),
        "llm_attempts": dict(state.get("llm_attempts") or {}),
        "hold_takebacks": int(state.get("hold_takebacks") or 0),
    }


def build_orion_day_graph(deps: OrionDayDeps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    def backoff(attempts: int) -> str:
        delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempts - 1))
        return (admission.now() + timedelta(seconds=delay)).isoformat()

    def llm_node(name: str, verb: str, field: str, metadata_for, check, user_text: str):
        async def operation(state: dict) -> dict:
            brief = await deps.load_brief(state)
            text = await deps.call_verb_text(
                verb, metadata_for(brief, state), brief.llm_route, gpu_lease=_lease_ref(state),
                timeout_sec=max(60.0, brief.timeout_sec - RPC_TIMEOUT_MARGIN_SEC), user_text=user_text,
            )
            text = (text or "").strip()
            return {field: text, **check(text, brief)}

        async def node(state: OrionDayState) -> dict:
            if state.get(field):
                return {"status": "running"}  # already checkpointed: never regenerate
            attempts = dict(state.get("llm_attempts") or {})
            try:
                result = await admission.execute(dict(state), operation)
            except WorkflowDeadline:
                released = await admission.release(dict(state), "workflow_deadline")
                return {**released, "status": "failed", "last_error": "workflow_deadline"}
            except RunControlPending:
                raise
            except HoldRecalled:
                return {"status": "waiting_resource", "lease": None, "hold": None}
            except HoldLost as exc:
                # The pool took the hold back: wait for it again and replay this node. Not an attempt.
                return await taken_back(admission, dict(state), exc.release_reason, f"{type(exc).__name__}: {exc}",
                                        {"status": "waiting_resource"})
            except Exception as exc:  # noqa: BLE001 -- transport failure or unusable completion: one attempt
                attempts[name] = attempts.get(name, 0) + 1
                error = f"{type(exc).__name__}: {exc}"[:500]
                if attempts[name] >= admission.max_attempts:
                    released = await admission.release(dict(state), "attempt_failed")
                    return {**released, "status": "failed", "llm_attempts": attempts, "last_error": error}
                released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
                return {**released, "status": "retrying", "llm_attempts": attempts, "last_error": error,
                        "retry_node": name, "retry_at": backoff(attempts[name])}
            attempts[name] = attempts.get(name, 0) + 1
            update = {**result, "llm_attempts": attempts, "status": "running", "last_error": None,
                      "retry_node": None}
            if name == "write_carry_forward":
                # Both texts are checkpointed with this update: the rest needs no GPU. Hand the hold
                # back -- outside the try, so a release hiccup can never discard a finished text.
                try:
                    update.update(await admission.release({**dict(state), **update}, "completed"))
                except Exception:  # noqa: BLE001 -- finish releases again; the pool's TTL is the backstop
                    logger.warning("orion_day_hold_release_failed run=%s", state.get("run_id"), exc_info=True)
            return update
        return node

    write_note = llm_node(
        "write_note", ORION_DAY_NOTE_VERB, "note_md",
        lambda brief, state: note_metadata(brief), check_note, "Write your note about the day.")
    write_carry_forward = llm_node(
        "write_carry_forward", ORION_DAY_CARRY_FORWARD_VERB, "carry_forward_md",
        lambda brief, state: carry_forward_metadata(brief, state["note_md"]), check_carry_forward,
        "List the threads from this day.")

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

    async def retry_wait(state: OrionDayState) -> dict:
        until = datetime.fromisoformat(state["retry_at"])
        if admission.now() < until:
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        return {"status": "retrying"}

    async def finish(state: OrionDayState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: OrionDayState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_llm(successor: str):
        def route(state) -> str:
            return {"waiting_resource": "resource_request", "retrying": "retry_wait",
                    "failed": "failed"}.get(state.get("status"), successor)
        return route

    def after_grant(state) -> str:
        decision = after_wait(state)
        if decision != "granted":
            return decision
        return "write_carry_forward" if state.get("note_md") else "write_note"

    g = StateGraph(OrionDayState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "write_note": write_note, "write_carry_forward": write_carry_forward,
                       "persist": persist, "retry_wait": retry_wait, "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_grant,
                            {"write_note": "write_note", "write_carry_forward": "write_carry_forward",
                             "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("write_note", after_llm("write_carry_forward"))
    g.add_conditional_edges("write_carry_forward", after_llm("persist"))
    g.add_conditional_edges("persist", lambda s: {"retrying": "retry_wait", "failed": "failed"}.get(s.get("status"), "finish"))
    # An LLM retry goes back for the hold (resource_request); a persist retry needs none.
    g.add_conditional_edges("retry_wait", lambda s: "persist" if s.get("retry_node") == "persist" else "resource_request",
                            {"persist": "persist", "resource_request": "resource_request"})
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
