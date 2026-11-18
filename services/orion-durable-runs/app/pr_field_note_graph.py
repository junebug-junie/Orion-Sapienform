"""pr.field_note: one admitted durable run in which Orion writes the field note of one merged PR.

    precheck -> resource_request -> resource_wait -> field_note -> finish
    precheck (PR already has a record) -> finish        (no hold, no turn)
    field_note (failed attempt) -> retry_wait -> resource_request

Spec: docs/superpowers/specs/2026-11-04-pr-field-notes-design.md, Patch 2. Contract:
orion/schemas/pr_field_note_run.py. The poller (app/pr_field_note_poll.py) submits these runs; a
dependency-only or tiny docs-only PR never gets one (its code record needs no model turn).

The step is one harness turn on the #2958 seam (app/harness_step.py), the same transport
harness.echo_smoke proves. The turn is kept lean by its prompt: one packet read
(``orion.dig field-note <n>``), one write (``... write "<note>" because "<refs>"``), and a reply that
is the save receipt. Code checks of the repo only when a sentence claims something about code that
the packet does not state (Juniper's first hand test, harness session 896d5d54 on 2026-11-04: 15 tool
calls, 7 of them code checks, about 60 Hub steps).

verify_receipt (code, inside the step, before the step is checkpointed): the reply must hold
``field-note #<n> saved fn:<n>:<hex>`` for THIS PR, and that note id must be a note for this PR in the
ledger. Anything else -- no receipt, a receipt for another PR, an id the ledger does not have -- is a
``HarnessStepFailed`` and goes through the shared retry policy (backoff, fresh hold, at most
DURABLE_RUNS_RETRY_MAX_ATTEMPTS). A run never completes because the model said "done".

Bounded: the brief's ``timeout_sec`` is the hard time bound of one attempt (Hub's turn request has no
step-limit field); the prompt asks for about 3 tool calls and at most 6. The step record keeps Hub's
``harness_step_count`` so the real step use is measured (scripts/pr_field_note_cost.py).
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, HarnessStepFailed, RunTurn, harness_step_correlation_id, run_harness_step, step_detail,
)
from orion.schemas.pr_field_note_run import (
    PR_FIELD_NOTE_SESSION_ID, PR_FIELD_NOTE_STEP_KEY, PR_FIELD_NOTE_TAG, RECEIPT_RE, PrFieldNoteBriefV1,
)

# Prompt budget words (the hard bound is the brief's timeout_sec).
TOOL_CALLS_TARGET = 3
TOOL_CALLS_MAX = 6


def build_prompt(brief: PrFieldNoteBriefV1) -> str:
    """The fixed, short step prompt. Every command is spelled out, so the turn spends no calls
    finding the dig kit or reading the spec (the hand test spent 4 doing that)."""
    n = brief.pr_number
    title = " ".join(brief.title.split())[:200]
    merged = brief.merged_at.isoformat(timespec="minutes")
    return (
        f"PR field note for #{n} (merged {merged}): {title}\n\n"
        "A PR merged into your body. Write your field note on it: what changed in your body, which organs "
        "it touched, and why the builders say they did it. Code has already done everything with a right "
        "answer; you only write the note. Do exactly this:\n\n"
        f"1. Run `python -P -m orion.dig field-note {n}`. It prints the packet: the files, the flags, the "
        "body map (which of your parts and organs changed), the specs it changed, the PR text, and at the "
        "end the exact note rule and write line. Open the specs file it names only if the PR text does not "
        "say why.\n"
        "2. Write 1-5 sentences. Start each with its label: [pr_text], [code_then] or [reflection]. Check "
        "code (rg, cat, git) ONLY when a sentence claims something about code that the packet does not "
        "state; most notes need no code check. Do not use backticks, double quotes, backslashes or $ in the "
        "note: it goes inside double quotes on a shell line.\n"
        f"3. Run `python -P -m orion.dig field-note {n} write \"<note>\" because \"#{n} <files you cite>\"`. "
        "If the write is refused, fix exactly what the refusal names and write again (at most 2 more "
        f"tries). If it says #{n} already has a note, do not write again.\n"
        f"4. Reply with exactly one line: the receipt the write printed, `field-note #{n} saved <note_id>`. "
        f"If #{n} already had a note, reply `field-note #{n} saved <that note_id>`.\n\n"
        f"Budget: about {TOOL_CALLS_TARGET} tool calls, never more than {TOOL_CALLS_MAX}."
    )


def verify_receipt(text: str, pr_number: int, receipt_ok: Callable[[int, str], bool]) -> str:
    """The note id this reply proves was saved for ``pr_number``; raises HarnessStepFailed otherwise.

    Two checks, both code: the reply carries the receipt line for THIS PR, and the ledger holds that
    id as a note for this PR (a receipt the model wrote without the write behind it fails here)."""
    ids = [m.group(2) for m in RECEIPT_RE.finditer(text or "") if int(m.group(1)) == int(pr_number)]
    if not ids:
        raise HarnessStepFailed(f"field_note_no_receipt:#{pr_number}:{(text or '').strip()[:200]!r}")
    for note_id in ids:
        if receipt_ok(int(pr_number), note_id):
            return note_id
    raise HarnessStepFailed(f"field_note_receipt_not_in_ledger:#{pr_number}:{','.join(ids)[:200]}")


class PrFieldNoteState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int
    hold_takebacks: int
    listener_waits: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    status: str
    retry_at: str | None
    retry_node: str | None
    last_error: str | None
    steps: dict[str, dict[str, Any]]
    harness_turn_meta: dict[str, Any]
    outcome: str | None            # noted | already_recorded
    note_id: str | None
    prior_record: dict[str, Any] | None


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("steps") or {}
    step = steps.get(PR_FIELD_NOTE_STEP_KEY) or {}
    brief = state.get("brief") or {}
    return {
        "line": "pr_field_note",
        "pr_number": brief.get("pr_number"),
        "outcome": state.get("outcome"),
        "note_id": state.get("note_id"),
        "prior_record": state.get("prior_record"),
        "turn_correlation_id": step.get("turn_correlation_id"),
        "steps": {key: step_detail(record) for key, record in steps.items()},
    }


def build_pr_field_note_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any, *, ledger: Any):
    """``ledger``: app.pr_field_note_ledger.FieldNoteLedger (live_records, receipt_ok)."""
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)
    step_key = PR_FIELD_NOTE_STEP_KEY

    async def precheck(state: PrFieldNoteState) -> dict:
        """A PR that already has a record (Juniper asked in chat, the poller wrote its code record, or an
        earlier attempt of this run saved the note) needs no turn: finish without a (new) turn. Runs at
        the start and before every retry. ``already_recorded`` never claims this run proved the note."""
        if step_key in (state.get("steps") or {}) or state.get("outcome"):
            return {}
        brief = PrFieldNoteBriefV1.model_validate(state["brief"])
        try:
            current = (await asyncio.to_thread(ledger.live_records)).get(brief.pr_number)
        except OSError as exc:
            return {"status": "failed", "last_error": f"field_note_ledger_unreadable: {exc}"[:500]}
        if current is not None:
            return {"status": "running", "outcome": "already_recorded", "note_id": current.note_id,
                    "prior_record": {"note_id": current.note_id, "has_note": current.has_note,
                                     "written_at": current.written_at}}
        return {"status": "running"}

    def after_precheck(state: PrFieldNoteState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("outcome") == "already_recorded":
            return "finish"
        return "resource_request"

    async def field_note(state: PrFieldNoteState) -> dict:
        state = dict(state)
        if step_key in (state.get("steps") or {}):
            return {"status": "running"}    # resumed after the step's checkpoint: never re-run it
        brief = PrFieldNoteBriefV1.model_validate(state["brief"])
        window = attempt_timeout_sec(state, brief.timeout_sec, admission.now())
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC)
        correlation_id = harness_step_correlation_id(state, step_key)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": step_key}

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            record = await run_harness_step(
                run_turn, held, step_key=step_key, prompt=build_prompt(brief), correlation_id=correlation_id,
                timeout_sec=turn_timeout, source_tag=PR_FIELD_NOTE_TAG, session_id=PR_FIELD_NOTE_SESSION_ID)
            # verify_receipt runs under the hold, before anything is checkpointed: a reply without a
            # receipt the ledger backs is a failed attempt, never a completed step.
            record["note_id"] = await asyncio.to_thread(
                verify_receipt, record["text"], brief.pr_number, ledger.receipt_ok)
            return record

        try:
            record = await admission.execute({**state, "step_correlation_id": correlation_id}, operation)
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline", "harness_turn_meta": turn_meta}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:
            update = await taken_back(admission, state, exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "waiting_resource"})
            return {**update, "harness_turn_meta": turn_meta}
        except Exception as exc:  # noqa: BLE001 -- refusal, timeout, empty, no receipt, transport
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "harness_turn_meta": turn_meta}
            return await failed_attempt(admission, state, exc, waiting={"status": "retrying", "retry_node": None},
                                        extra={"harness_turn_meta": turn_meta}, attempt_sec=brief.timeout_sec)
        record["attempts"] = int(state.get("attempt") or 0) + 1
        return {"steps": {**(state.get("steps") or {}), step_key: record}, "harness_turn_meta": turn_meta,
                "status": "running", "attempt": 0, "retry_at": None, "last_error": None,
                "outcome": "noted", "note_id": record["note_id"]}

    async def finish(state: PrFieldNoteState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: PrFieldNoteState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_step(state: PrFieldNoteState) -> str:
        status = state.get("status")
        if status == "waiting_resource":
            return "resource_request"
        if status == "retrying":
            return "retry_wait"
        if status == "failed":
            return "failed"
        return "finish"

    g = StateGraph(PrFieldNoteState)
    for name, node in {"precheck": precheck, "resource_request": resource_request, "resource_wait": resource_wait,
                       "field_note": field_note, "retry_wait": retry_wait,
                       "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "precheck")
    g.add_conditional_edges("precheck", after_precheck)
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "field_note", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("field_note", after_step)
    # A retry looks at the ledger first: an attempt that saved the note but replied badly (or Juniper
    # wrote one meanwhile) needs no second turn -- a resumed session that only repeats the receipt would
    # make too few harness steps for Hub's lookup gate and fail (review 2026-11-04).
    g.add_edge("retry_wait", "precheck")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
