"""reminiscence.episode: one admitted durable run a day in which Orion writes the five hops of one past
week, then reads them together once.

    precheck -> resource_request -> resource_wait -> step -> step -> ... -> finish
    precheck (today's five hops are already live) -> finish        (no hold, no turn)
    step (failed attempt) -> retry_wait -> precheck -> resource_request

Design: docs/superpowers/specs/2026-11-07-daily-reminiscence-design.md, section 6. Contract:
orion/schemas/reminiscence_run.py. The tick (app/reminiscence_poll.py) submits these runs.

``step`` runs the next step of REMINISCENCE_STEPS under the run's one hold and loops to itself until
all six are done: five hop steps (``hop:what_it_was`` ... ``hop:otherwise``), then ``cross_read``. Each
is one harness turn on the #2958 seam (app/harness_step.py) with its own source tag, so each hop is a
fresh claude session and only a retry of the same step resumes one.

The ledger is the checkpoint: before a hop's turn, code looks in the ledger, and a hop already live
today (written in chat, or saved by a failed attempt that replied badly) is recorded as
``already_written`` with no turn. verify_receipt (code, inside the step, under the hold, before the step
is checkpointed): the reply must hold ``reminisce <hop> saved rm:<episode>:<hop>:<hex>`` for THIS hop
and today's episode, and the ledger must hold that id. The cross-read reply must end with
``reminisce cross-read done``, and every rewrite receipt in it must be a rewrite in the ledger. finish
checks that all five hops are live. A run never completes because the model said "done".

Cleanup: the cross-read turn runs ``dig close <period end>`` (the export ``dig open`` made). A run that
ends early leaves that export to the dig kit's own bound (at most MAX_OPEN exports; 24 h old ones go),
so no seventh turn is spent on cleanup.

Bounded: each step attempt is ``brief.timeout_sec`` (the attempt timer and Hub's turn limit), each step
gets DURABLE_RUNS_RETRY_MAX_ATTEMPTS attempts (``attempt`` resets after a step succeeds), and the whole
run ends at its admission deadline (the tick sets REMINISCENCE_END_HOUR_LOCAL). Each step record keeps
Hub's ``harness_step_count`` and elapsed time, so the real cost is measured (scripts/pr_field_note_cost.py
--workflow reminiscence.episode).
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Callable, TypedDict
from zoneinfo import ZoneInfo

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, HarnessStepFailed, RunTurn, harness_step_correlation_id, run_harness_step, step_detail,
)
from orion.schemas.reminiscence_run import (
    CROSS_READ_DONE, CROSS_READ_STEP, RECEIPT_RE, REMINISCENCE_FIRST_DAY, REMINISCENCE_HOPS,
    REMINISCENCE_LIVED_RECORDS_FROM as LIVED_RECORDS_FROM,
    REMINISCENCE_SESSION_ID, REMINISCENCE_STEPS, ReminiscenceRunBriefV1, step_tag,
)

# The dig's day (orion/dig/__main__.py ZONE): an episode is the Denver day its hops are written on.
ZONE = ZoneInfo("America/Denver")
# Prompt budget words (the hard bound is the brief's timeout_sec per attempt).
TOOL_CALLS_TARGET = 4
TOOL_CALLS_MAX = 8
CROSS_READ_MAX_REWRITES = 2
DIG = "python -P -m orion.dig"

LABEL_RULE = (
    "Write 1-6 sentences, 40-1,000 characters in all. Start EVERY sentence with its own label tag: "
    "[code_then <ref>] seen in the code of that week, [pr_text <ref>] the builders' words in a PR or commit, "
    "[lived_record <ref>] your own journals and days, [now <ref>] true of you today, [reflection] your own "
    "thought. Refs: a repo path, a commit sha, #<pr>, fn:<pr>:<id>, part:/organ:/cap: ids. Code checks every "
    "ref under its label before it saves anything. Do not use backticks, double quotes, backslashes or $ in "
    "the note: it goes inside double quotes on a shell line."
)


def _what_to_read(hop: str, brief: ReminiscenceRunBriefV1) -> str:
    start, end = brief.period_start.isoformat(), brief.period_end.isoformat()
    if hop == "what_it_was":
        return (f"what was in your code that week: `{DIG} open {end}` (leave it open: the last turn closes it), "
                f"`{DIG} compare {end}`, `{DIG} history {start} {end}`. Read only what you need from the files they "
                "name. Needs at least one [code_then] or [pr_text] sentence.")
    if hop == "what_it_was_like":
        if brief.period_end < LIVED_RECORDS_FROM:
            return ("what that week was like. You have no lived record of it: your records start "
                    f"{LIVED_RECORDS_FROM}. Say so plainly, and use [pr_text] for what the builders wrote "
                    f"(`{DIG} history {start} {end}`); [lived_record] is refused for this week. Needs at least one "
                    "[pr_text] sentence.")
        return ("what that week was like: the read's lived_record line names what can cover it (`dig arc <id>`). "
                "Needs at least one [lived_record] or [pr_text] sentence.")
    if hop == "where_i_am_now":
        return (f"where you are now: the read's 'since' block (what changed in you since your last reminiscence) "
                f"and `{DIG} portrait`. Needs at least one [now] sentence with a live ref.")
    if hop == "where_im_going":
        return (f"where you are going: `{DIG} capacities gaps` and `{DIG} portrait` (what you cannot yet do). Needs "
                "at least one [now] or [pr_text] sentence.")
    return ("what you would do differently: the four hops the read shows. Needs at least one [reflection] "
            "sentence.")


def build_hop_prompt(brief: ReminiscenceRunBriefV1, hop: str) -> str:
    """The fixed, short prompt of one hop step. Every command is spelled out, so the turn spends no calls
    finding the dig kit or reading the design."""
    i = REMINISCENCE_HOPS.index(hop) + 1
    start, end = brief.period_start.isoformat(), brief.period_end.isoformat()
    n = (brief.period_start - REMINISCENCE_FIRST_DAY).days // 7 + 1
    week = (f"the week {start}..{end} (week {n} of the {brief.weeks_total} weeks of your past so far; you have "
            f"visited {brief.weeks_complete} of them, oldest first)")
    return (
        f"Reminiscence {brief.episode.isoformat()}, hop {i} of 5: {hop}. Today you visit {week}.\n\n"
        "Once a day you look at one week of your own past and write five short hops about it: what it was, what "
        "it was like, where you are now, where you are going, and what you would do differently. Code picked the "
        "week and checks every ref; you only write this one hop. Do exactly this:\n\n"
        f"1. Run `{DIG} reminisce`. It prints the week, its last commit, whether a lived record can exist, the "
        "hops already written today (read them: the hops read each other), what changed in you since your last "
        "reminiscence, and at the end the rule and a ready write line.\n"
        f"2. Read {_what_to_read(hop, brief)}\n"
        f"3. {LABEL_RULE}\n"
        f"4. Run `{DIG} reminisce write {hop} \"<note>\" because \"<refs>\"`. If the write is refused, fix exactly "
        f"what the refusal names and write again (at most 2 more tries). If it says {hop} is already written "
        "today, do not write again.\n"
        f"5. Reply with exactly one line: the receipt the write printed, `reminisce {hop} saved <hop_id>`. If "
        f"{hop} was already written today, reply `reminisce {hop} saved <the hop_id the refusal names>`.\n\n"
        f"Budget: about {TOOL_CALLS_TARGET} tool calls, never more than {TOOL_CALLS_MAX}."
    )


def build_cross_read_prompt(brief: ReminiscenceRunBriefV1) -> str:
    end = brief.period_end.isoformat()
    return (
        f"Reminiscence {brief.episode.isoformat()}, last turn: read your five hops together.\n\n"
        f"Today you wrote five hops about the week {brief.period_start.isoformat()}..{end}, each in its own "
        "turn. Now read them as one. Do exactly this:\n\n"
        f"1. Run `{DIG} reminisce`. It shows all five hops and whether each one holds now.\n"
        "2. If a hop reads STALE, or two hops disagree, or one says something another shows is wrong, rewrite "
        f"it: `{DIG} reminisce rewrite <hop> \"<note>\" because \"<refs>\"` (same rule as the write; the old "
        f"line is kept). Rewrite at most {CROSS_READ_MAX_REWRITES} hops. If they hold and agree, rewrite none.\n"
        f"3. Run `{DIG} close {end}` (the export of that week; it is fine if it says it was not open).\n"
        "4. Reply with one receipt line per rewrite, exactly as the rewrite printed it "
        f"(`reminisce <hop> saved <hop_id> ...`), then the line `{CROSS_READ_DONE}`. With no rewrite, reply only "
        f"`{CROSS_READ_DONE}`.\n\n"
        f"Budget: about 3 tool calls, never more than {TOOL_CALLS_MAX}."
    )


def build_prompt(brief: ReminiscenceRunBriefV1, step_key: str) -> str:
    if step_key == CROSS_READ_STEP:
        return build_cross_read_prompt(brief)
    return build_hop_prompt(brief, step_key.removeprefix("hop:"))


def _receipts(text: str, episode: date) -> list[tuple[str, str]]:
    """(hop, hop_id) of every receipt in ``text`` for ``episode`` whose id names the same hop."""
    out = []
    for m in RECEIPT_RE.finditer(text or ""):
        hop, hop_id, ep, id_hop = m.group(1), m.group(2), m.group(3), m.group(4)
        if ep == episode.isoformat() and hop == id_hop:
            out.append((hop, hop_id))
    return out


def verify_hop_receipt(text: str, episode: date, hop: str, receipt_ok: Callable[[date, str, str], bool]) -> str:
    """The hop id this reply proves was saved for ``hop`` today; raises HarnessStepFailed otherwise.

    Two checks, both code: the reply carries the receipt line for THIS hop and today's episode, and the
    ledger holds that id for this episode and hop (a receipt the model wrote without the write behind it
    fails here)."""
    ids = [hop_id for h, hop_id in _receipts(text, episode) if h == hop]
    if not ids:
        raise HarnessStepFailed(f"reminiscence_no_receipt:{hop}:{(text or '').strip()[:200]!r}")
    for hop_id in ids:
        if receipt_ok(episode, hop, hop_id):
            return hop_id
    raise HarnessStepFailed(f"reminiscence_receipt_not_in_ledger:{hop}:{','.join(ids)[:200]}")


def verify_cross_read(text: str, episode: date, rewrite_ok: Callable[[date, str], bool]) -> list[str]:
    """The rewrite ids the cross-read reply proves; raises HarnessStepFailed when the closing line is
    missing or a receipt it carries is not a rewrite in the ledger."""
    if CROSS_READ_DONE not in (text or ""):
        raise HarnessStepFailed(f"reminiscence_cross_read_not_done:{(text or '').strip()[:200]!r}")
    ids = [hop_id for _hop, hop_id in _receipts(text, episode)]
    bad = [i for i in ids if not rewrite_ok(episode, i)]
    if bad:
        raise HarnessStepFailed(f"reminiscence_rewrite_not_in_ledger:{','.join(bad)[:200]}")
    return list(dict.fromkeys(ids))


def pending_step(state: dict[str, Any]) -> str | None:
    """The first step of REMINISCENCE_STEPS with no checkpointed result, or None when all are done. The
    restart cancel (AdmissionRuntime._cancel_harness) derives the in-flight turn id from it."""
    done = state.get("steps") or {}
    return next((k for k in REMINISCENCE_STEPS if k not in done), None)


class ReminiscenceState(TypedDict, total=False):
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
    outcome: str | None            # complete | already_complete
    hop_ids: dict[str, str] | None


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("steps") or {}
    brief = state.get("brief") or {}
    return {
        "line": "reminiscence",
        "episode": brief.get("episode"),
        "period_start": brief.get("period_start"),
        "period_end": brief.get("period_end"),
        "outcome": state.get("outcome"),
        "hop_ids": state.get("hop_ids"),
        "turns": sum(1 for r in steps.values() if isinstance(r, dict) and r.get("turn_correlation_id")),
        "steps": {key: step_detail(record) for key, record in steps.items()},
    }


def build_reminiscence_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any, *, ledger: Any):
    """``ledger``: app.reminiscence_ledger.ReminiscenceLedger (hop_ids, receipt_ok, rewrite_ok)."""
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)

    def today() -> date:
        return admission.now().astimezone(ZONE).date()

    async def precheck(state: ReminiscenceState) -> dict:
        """The day passed (hops would land in another episode): fail. All five hops of today are already
        live and no step ran yet: finish with no hold. Runs at the start and before every retry."""
        brief = ReminiscenceRunBriefV1.model_validate(state["brief"])
        if today() != brief.episode:
            return {"status": "failed", "last_error": f"reminiscence_day_passed:{brief.episode}"}
        if state.get("steps"):
            return {"status": "running"}
        try:
            ids = await asyncio.to_thread(ledger.hop_ids, brief.episode)
        except OSError as exc:
            return {"status": "failed", "last_error": f"reminiscence_ledger_unreadable: {exc}"[:500]}
        if set(ids) == set(REMINISCENCE_HOPS):
            return {"status": "running", "outcome": "already_complete", "hop_ids": ids}
        return {"status": "running"}

    def after_precheck(state: ReminiscenceState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("outcome") == "already_complete":
            return "finish"
        return "resource_request"

    async def step(state: ReminiscenceState) -> dict:
        state = dict(state)
        step_key = pending_step(state)
        if step_key is None:
            return {"status": "running"}
        brief = ReminiscenceRunBriefV1.model_validate(state["brief"])
        steps = dict(state.get("steps") or {})
        if today() != brief.episode:
            released = await admission.release(state, "reminiscence_day_passed")
            return {**released, "status": "failed", "last_error": f"reminiscence_day_passed:{brief.episode}"}
        hop = step_key.removeprefix("hop:") if step_key != CROSS_READ_STEP else None
        if hop is not None:
            try:
                written = (await asyncio.to_thread(ledger.hop_ids, brief.episode)).get(hop)
            except OSError as exc:
                released = await admission.release(state, "ledger_unreadable")
                return {**released, "status": "failed", "last_error": f"reminiscence_ledger_unreadable: {exc}"[:500]}
            if written:
                # Written in chat, or by this step's own earlier attempt: the ledger is the checkpoint.
                steps[step_key] = {"step_key": step_key, "outcome": "already_written", "hop_id": written}
                return {"steps": steps, "status": "running", "attempt": 0, "retry_at": None, "last_error": None}
        window = attempt_timeout_sec(state, brief.timeout_sec, admission.now())
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC)
        correlation_id = harness_step_correlation_id(state, step_key)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": step_key}

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            record = await run_harness_step(
                run_turn, held, step_key=step_key, prompt=build_prompt(brief, step_key),
                correlation_id=correlation_id, timeout_sec=turn_timeout, source_tag=step_tag(step_key),
                session_id=REMINISCENCE_SESSION_ID)
            # Verified under the hold, before anything is checkpointed: a reply without a receipt the
            # ledger backs is a failed attempt, never a completed step.
            if hop is not None:
                record["hop_id"] = await asyncio.to_thread(
                    verify_hop_receipt, record["text"], brief.episode, hop, ledger.receipt_ok)
                record["outcome"] = "written"
            else:
                record["rewrites"] = await asyncio.to_thread(
                    verify_cross_read, record["text"], brief.episode, ledger.rewrite_ok)
                record["outcome"] = "cross_read"
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
        steps[step_key] = record
        return {"steps": steps, "harness_turn_meta": turn_meta, "status": "running", "attempt": 0,
                "retry_at": None, "last_error": None}

    def after_step(state: ReminiscenceState) -> str:
        status = state.get("status")
        if status == "waiting_resource":
            return "resource_request"
        if status == "retrying":
            return "retry_wait"
        if status == "failed":
            return "failed"
        return "step" if pending_step(state) is not None else "finish"

    async def finish(state: ReminiscenceState) -> dict:
        state = dict(state)
        if state.get("outcome") == "already_complete":
            released = await admission.release(state, "completed")
            return {**released, "status": "completed"}
        brief = ReminiscenceRunBriefV1.model_validate(state["brief"])
        try:
            ids = await asyncio.to_thread(ledger.hop_ids, brief.episode)
        except OSError as exc:
            released = await admission.release(state, "failed")
            return {**released, "status": "failed", "last_error": f"reminiscence_ledger_unreadable: {exc}"[:500]}
        if set(ids) != set(REMINISCENCE_HOPS):
            # Every step proved its receipt, yet a hop is not live: never call that a visited week.
            missing = [h for h in REMINISCENCE_HOPS if h not in ids]
            released = await admission.release(state, "failed")
            return {**released, "status": "failed", "hop_ids": ids,
                    "last_error": f"reminiscence_incomplete:{','.join(missing)}"}
        released = await admission.release(state, "completed")
        return {**released, "status": "completed", "outcome": "complete", "hop_ids": ids}

    async def failed(state: ReminiscenceState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(ReminiscenceState)
    for name, node in {"precheck": precheck, "resource_request": resource_request, "resource_wait": resource_wait,
                       "step": step, "retry_wait": retry_wait, "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "precheck")
    g.add_conditional_edges("precheck", after_precheck)
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "step", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("step", after_step)
    # A retry looks at the day first; the step itself looks at the ledger, so a hop that an attempt saved
    # but replied badly about takes no second turn.
    g.add_edge("retry_wait", "precheck")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
