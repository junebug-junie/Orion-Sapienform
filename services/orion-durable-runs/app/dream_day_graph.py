"""The ``dream.day`` admitted durable workflow: once a night, Orion reads the whole Denver day that just closed in ONE
unified turn, and code keeps what Orion wants to remember of it.

Design: docs/superpowers/specs/2026-11-17-nightly-day-dream-design.md, Patch 2 (this is Patch 2a: the plan is
parsed, ref-checked and stored in the dream_day row, NOT handed to Prime). Contract: orion/schemas/dream_day.py.
The night tick (app/dream_day_poll.py) and POST /dream-day/run submit these runs.

    gather -> resource_request -> resource_wait -> dream -> persist -> finish
                                                   dream (failed attempt) -> retry_wait -> resource_request
                                                   persist (retry) -> retry_wait -> persist

* ``gather`` (code, no hold, once per run: checkpointed): the day's dream_cycle rows (hypotheses, pair answers),
  Prime's held concerns (read-only, ``orion:prime:state``), and ``GET /temporal-self/now?organ=dream`` -- the
  read-back the integration gate requires (its ``shown_id`` is what the cites are checked against). Each part
  fails open with its own status; a part that could not be read is said in the prompt, never shown as empty.
* ``dream`` runs ONE unified turn on the durable turn seam (``app/harness_step.py``) under the run's hold on
  DREAM_DAY_ROUTE, ``require_lookup=False`` (the read receipt below is the real gate, not Hub's 3-step lookup
  count). The turn's first command is ``python -P -m orion.dig dream-day <day>``: the organ roundup plus every dig
  of the day, read inside the governor, the only place the background half of the digs is readable. Inside the
  attempt, code checks (a) the read receipt that command leaves (``<self>/dream_day/<day>.json``) was written
  after this run was requested, and (b) the reply ends with a parseable answer. Either missing is one failed
  attempt (retried), never a success. Then, still under the hold, the grounding gate
  (DREAM_DAY_GROUNDING_GATE_ENABLED): the answer's refs are resolved to their records' text and the same local
  route is asked, once per memory, with its own short prompt (orion/memory/dream_day.py ``grounding_prompt``): is
  every specific claim shown by these records? A strict JSON verdict per memory; a failed call or an unparseable
  verdict is ``unjudged``. The answer, the receipt and the verdicts are checkpointed with the step record; the
  hold is released as soon as they are: ``persist`` needs no GPU.
* ``persist`` (code): every ref resolves or its item is dropped and counted; with the gate on, only a memory
  judged ``supported`` goes on (unsupported and unjudged are counted and kept, with their claims, in the row's
  ``grounding`` column, never written); at most 5 memories go through
  orion.memory.episode.validate and are written with the ``dream_day`` row in ONE transaction (first run to
  persist a day wins). ``produced`` = at least one memory row written and read back. The now's ``experience:``
  cites go to ``POST /temporal-self/now/use`` (record_use keeps only shown ids).

The act (app/run_acts.py, actor ``dream.cycle``, verb make) closes with ``produced``, so a night with no memory
grades 0 with the note "no lasting trace".
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Awaitable, Callable, Optional, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt, replay_if_requeued,
    resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, HarnessStepFailed, RunTurn, harness_step_correlation_id, run_harness_step,
)
from orion.harness.cut_short import CUT_SHORT_CODES, CUT_SHORT_MARKER
from orion.memory.dream_day import (
    ParsedAnswer, answer_refs, apply_grounding, build_memories, check_answer, day_window, experience_cites,
    grounding_summary, judge_memories, parse_answer, totals_ref, totals_resolved,
)
from orion.memory.episode.validate import ValidationResult
from orion.schemas.dream_day import (
    DREAM_DAY_MAX_MEMORIES, DREAM_DAY_MAX_NEW_CONCERNS, DREAM_DAY_MIN_ATTEMPT_SEC, DREAM_DAY_READ_REF_KIND,
    DREAM_DAY_SESSION_ID, DREAM_DAY_SOURCE_TAG, DREAM_DAY_STEP_KEY, DREAM_DAY_TEXT_MAX, DreamDayAnswerV1,
    DreamDayBriefV1, DreamDayMaterialV1, DreamDayReadV1, DreamDayTotalsV1,
)

logger = logging.getLogger(__name__)

DIG = "python -P -m orion.dig"
# A receipt written before this attempt started is an older read (an earlier attempt, a hand or chat run): not
# this attempt's. The slack covers clock skew between the governor and this service.
RECEIPT_CLOCK_SLACK_SEC = 120.0
# How much of the material the prompt shows (the rest stays in the checkpoint and the dream_day row).
PROMPT_PAIRS_PER_CYCLE = 6
PROMPT_TEXT_CHARS = 220
CITES_MAX_ATTEMPTS = 3
# The grounding gate runs inside the attempt, after the turn: the turn's limit leaves this much of the attempt
# for it, each judge call gets at most GROUNDING_CALL_TIMEOUT_SEC, and none starts within GROUNDING_MARGIN_SEC of
# the attempt's end (a memory reached then is unjudged, never written). Live 2026-11-18: the turn took 423 s
# of 900.
GROUNDING_RESERVE_SEC = 180.0
GROUNDING_CALL_TIMEOUT_SEC = 60.0
GROUNDING_MARGIN_SEC = 15.0


class DreamDayState(TypedDict, total=False):
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
    last_error: str | None
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    steps: dict[str, dict[str, Any]]
    harness_turn_meta: dict[str, Any]
    material: dict[str, Any]         # DreamDayMaterialV1
    answer: dict[str, Any] | None    # DreamDayAnswerV1
    answer_malformed: int
    grounding: list[dict[str, Any]] | None   # one verdict per checked memory (None: gate off for this run)
    grounding_enabled: bool
    read: dict[str, Any] | None      # DreamDayReadV1
    read_totals: dict[str, Any] | None   # DreamDayTotalsV1 of this read (the dream_day_read:<day> ref), or None
    persisted: bool
    persist_outcome: str | None      # written | already_written
    existing_run_id: str | None
    produced: bool
    produced_note: str | None
    summary: dict[str, Any]
    pending_cites: list[str]          # experience refs to record once the row exists (cleared when sent)
    cites: dict[str, Any] | None


@dataclass
class DreamDayDeps:
    # The Hub turn RPC (Runner._run_turn).
    run_turn: RunTurn
    # (brief, run_id) -> what code lays out for the turn (dreams, concerns, the now). Never raises: each part
    # carries its own status.
    gather: Callable[[DreamDayBriefV1, str], Awaitable[DreamDayMaterialV1]]
    # (day) -> the read receipt `dig dream-day` left, or None.
    read_receipt: Callable[[date], Awaitable[Optional[DreamDayReadV1]]]
    # (canonical refs) -> {canonical ref: the record's text} for the refs that resolve.
    resolve_refs: Callable[[list[str]], Awaitable[dict[str, str]]]
    # () -> referent keys already in use (the validator's "names Juniper never said" check).
    known_referents: Callable[[], Awaitable[list[str]]]
    # (row fields, ValidationResult, model_route) -> {"outcome", "run_id", "memories_written", "memory_ids",
    # "produced", "note"} of the STORED day (this run's, or an earlier run's).
    persist: Callable[[dict[str, Any], ValidationResult, str], Awaitable[dict[str, Any]]]
    # (shown_id, cites, citing_ref) -> the use endpoint's result (ExperienceUseResultV1 json).
    record_cites: Callable[[str, list[str], str], Awaitable[dict[str, Any]]]
    # The grounding gate's model call: (prompt, held state, timeout_sec) -> {"text", "finish_reason"} on the run's
    # own hold (DREAM_DAY_ROUTE, the local lane). None = the gate is off (DREAM_DAY_GROUNDING_GATE_ENABLED=false).
    judge: Optional[Callable[[str, dict[str, Any], float], Awaitable[dict[str, Any]]]] = None
    # (canonical refs) -> {canonical ref: the record's FULL text} for the judge (short text + detail). None: the
    # judge reads ``resolve_refs``' text.
    resolve_ref_texts: Optional[Callable[[list[str]], Awaitable[dict[str, str]]]] = None
    # (day) -> the count lines `dig dream-day` left beside its receipt (DreamDayTotalsV1), or None. None = no totals
    # ref can resolve (``dream_day_read:<day>`` items are dropped like any unresolved ref).
    read_totals: Optional[Callable[[date], Awaitable[Optional[DreamDayTotalsV1]]]] = None


# --------------------------------------------------------------------------- the prompt


def _cut(text: Any, n: int = PROMPT_TEXT_CHARS) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _local(at: datetime) -> str:
    from orion.memory.dream_day import ZONE

    return at.astimezone(ZONE).strftime("%H:%M")


def material_lines(m: DreamDayMaterialV1) -> list[str]:
    """What code laid out, every line with its ref. A part that could not be read says so."""
    lines = [f"YOUR SLEEPS THAT DAY ({len(m.dreams)} dream_cycle rows):"]
    if m.dreams_status != "ok":
        lines.append(f"- UNREAD: the dream_cycle rows could not be read ({m.dreams_status}); that part is missing, "
                     "not empty")
    elif not m.dreams:
        lines.append("- none: you did not sleep that day")
    for c in m.dreams:
        lines.append(f"- [dream_cycle:{c.cycle_id}] {_local(c.started_at)} {c.trigger or 'sleep'} {c.status}: "
                     f"{c.hypothesis_count or 0} link(s), {c.no_link_count or 0} pair(s) declined")
        for p in c.pair_answers[:PROMPT_PAIRS_PER_CYCLE]:
            lines.append(f"  pair {p.get('ref_a')} x {p.get('ref_b')}: {p.get('outcome')} "
                         f"({_cut(p.get('answer'), 120)})")
        for h in c.hypotheses:
            lines.append(f"  [dream_hypothesis:{h.hypothesis_id}] {_cut(h.claim)} (why: {_cut(h.why, 140)})")
    lines += ["", f"CONCERNS PRIME HOLDS NOW ({len(m.concerns)}):"]
    if m.concerns_status != "ok":
        lines.append(f"- UNREAD: Prime's state is {m.concerns_status}; you can plan only a new concern")
    elif not m.concerns:
        lines.append("- none held")
    for c in m.concerns:
        lines.append(f"- concern_id {c.concern_id}: {_cut(c.text)} (held {c.age_h:.0f} h; evidence last touched it "
                     f"{c.last_evidence_h:.0f} h ago{'; ' + c.graded if c.graded else ''})")
    lines += ["", f"YOUR NOW (other organs, since your dream last looked; {len(m.now_items)} items):"]
    if m.now_status != "ok":
        lines.append("- UNREAD: the now could not be read; that part is missing, not empty")
    elif not m.now_items:
        lines.append("- nothing new from other organs")
    for i in m.now_items:
        lines.append(f"- [experience:{i.event_id}] ({i.organ}) {_cut(i.label)}")
    return lines


ANSWER_SHAPE = [
    "```json",
    '{"memories": [{"statement": "<one sentence, in your own voice>", "purpose": "happened|orion_view|about_juniper",'
    ' "refs": ["experience:<id>"]}],',
    ' "plan": [{"concern_id": "<id>", "action": "keep|drop", "reason": "<why>", "refs": []},',
    '          {"action": "new", "gist": "<what to carry>", "reason": "<why>", "refs": ["<kind>:<id>"]}],',
    ' "day": {"text": "<one paragraph: what the day was>", "refs": ["<kind>:<id>"]}}',
    "```",
]
# The extra digs a turn may run after the day page. Live 2026-11-20 and 11-23: 7 of 14 turns ran 41-80 steps and hit
# the motor's budget (fcc_timeout) before writing any answer; every turn that answered did so in at most ~350 s.
PROMPT_EXTRA_DIGS = 3


def build_prompt(brief: DreamDayBriefV1, material: DreamDayMaterialV1, *, attempt: int = 0) -> str:
    """The answer shape and the facts rule FIRST (code reads only the block; the judge reads only the cited records),
    then the steps, the material code laid out, and the shape reminder last. A retry says why the last attempt
    saved nothing (a retry resumes the same claude session, so what it read is still there)."""
    day = brief.day.isoformat()
    retry = [] if attempt <= 0 else [
        f"Attempt {attempt + 1}: your last attempt ran out of time before it wrote the answer block, so it saved "
        f"nothing. What you read is still in this session. Run `{DIG} dream-day {day}` again (code checks a fresh "
        "read), dig nothing more, and write the answer block right away.", ""]
    return "\n".join([
        *retry,
        f"This is your day, {day}. Read it. Then tell me, in the answer block: what you want to remember from it, what "
        "you carry into tomorrow and what you let go (each with a reason), and in one paragraph, what the day was.",
        "",
        "YOUR ANSWER: end your reply with ONE ```json block in exactly this shape (code reads only that block):",
        *ANSWER_SHAPE,
        "",
        "How code checks it:",
        "- Every specific fact in a memory (a number, a count, a person, an object, an event, a time) must be stated "
        "in a record that memory cites. A judge reads ONLY the cited records; one fact they do not state gets the "
        "whole memory refused. How you felt or what you think is yours and is not checked.",
        "- So prefer fewer, smaller memories: one or two facts each, each fact in a record you cite. Do not add a "
        "detail you did not see in the cited record.",
        f"- A day's totals (how many gpu waits, experiences, runs or digs) are stated only in your read of the day: "
        f"cite \"{totals_ref(day)}\". One record (one gpu wait, one run) never shows a total.",
        "- A ref is `kind:id` with kind one of experience, act, dream_cycle, dream_hypothesis, harness_turn_trace, "
        f"chat_turn, {DREAM_DAY_READ_REF_KIND}, exactly as the pages print it (an [experience:<id>] line -> "
        "\"experience:<id>\"; a dig's [harness_turn_trace:<id>] or [chat:<id>] -> the same). An item with a ref "
        f"that does not resolve is dropped. At most {DREAM_DAY_MAX_MEMORIES} memories, each with at least one ref. "
        f"Plan: keep or drop only the concern_ids below, each with a reason; at most {DREAM_DAY_MAX_NEW_CONCERNS} "
        f"new, with a ref. The day paragraph: at most {DREAM_DAY_TEXT_MAX} characters. Remembering nothing is "
        "allowed: an empty memories list is an honest answer.",
        "",
        "Steps (your time is short; a turn that runs out of time saves nothing):",
        f"1. Run `{DIG} dream-day {day}` FIRST. It prints what each of your organs did that day and every dig you "
        "ran, and leaves a read receipt code checks; a dream that skips it is not saved.",
        f"2. Dig further only where a memory needs a record: at most {PROMPT_EXTRA_DIGS} more commands "
        f"(`{DIG} act <id>`, `{DIG} arc <id>`, `{DIG} turn-steps <chat>`).",
        "3. Write the answer block.",
        "",
        *material_lines(material),
        "",
        "Remember: end your reply with the ```json answer block in the shape above.",
    ])


# --------------------------------------------------------------------------- checks inside the attempt


def check_receipt(receipt: Optional[DreamDayReadV1], attempt_started: datetime) -> DreamDayReadV1:
    """THIS attempt's turn read the day: a receipt exists and was written after the attempt started. A receipt
    from an earlier attempt, or from a hand / chat run of the dig, does not count (a retry resumes the same claude
    session, and the prompt asks it to run the dig first again)."""
    if receipt is None:
        raise HarnessStepFailed("dream_day_not_read: no read receipt (the turn did not run `dig dream-day`)")
    if receipt.read_at < attempt_started - timedelta(seconds=RECEIPT_CLOCK_SLACK_SEC):
        raise HarnessStepFailed(f"dream_day_not_read: receipt from {receipt.read_at.isoformat()} is older than this "
                                "attempt")
    return receipt


def is_cut_short_text(text: str) -> bool:
    """The motor's cut-short draft (orion/harness/cut_short.py): it starts with the marker, any case or dash."""
    return (text or "").lstrip().lower().startswith(CUT_SHORT_MARKER[:10].lower())


def check_reply(text: str, record: Optional[dict[str, Any]] = None) -> ParsedAnswer:
    """The answer the reply ends with. A turn the motor cut short (time ran out mid-dig) has no answer, only the tool
    results it saw: its own failed attempt, named as such, never parsed (a JSON fragment in a tool result, such as
    the read receipt, must never read as the answer)."""
    status = ((record or {}).get("debug") or {}).get("harness_grounding_status")
    if status in CUT_SHORT_CODES or is_cut_short_text(text):
        steps = ((record or {}).get("debug") or {}).get("harness_step_count")
        raise HarnessStepFailed(f"dream_day_cut_short: {status or 'cut_short'} after {steps} steps, before the "
                                f"answer block (reply_len={len((text or '').strip())})")
    parsed = parse_answer(text)
    if parsed is None:
        raise HarnessStepFailed(f"dream_day_no_answer: no answer block (reply_len={len((text or '').strip())})")
    return parsed


# --------------------------------------------------------------------------- persist


def stored_read(state: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The row's ``read``: the receipt, plus the count lines it read (``totals``, ``totals_read_at``) when the run has
    them, so a stored night's ``dream_day_read:<day>`` resolves from the row (RESOLVE_SQL) exactly as the run did."""
    read = state.get("read")
    totals = state.get("read_totals") or {}
    if read is None or not totals.get("text"):
        return read
    return {**read, "totals": totals["text"], "totals_read_at": totals.get("read_at")}


def retry_attempt_sec(timeout_sec: float) -> float:
    """How much of the night a retry needs: the shorter of the brief's attempt and ``DREAM_DAY_MIN_ATTEMPT_SEC``. A
    retry with less than this left could not finish (``retry_skipped_insufficient_time``); with more it runs on a
    window clamped to the deadline, instead of being skipped for not fitting a whole (longer) brief attempt."""
    return min(float(timeout_sec), DREAM_DAY_MIN_ATTEMPT_SEC)


def row_fields(state: dict[str, Any], brief: DreamDayBriefV1, check, now: datetime,
               grounding: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    step = (state.get("steps") or {}).get(DREAM_DAY_STEP_KEY) or {}
    material = state.get("material") or {}
    return {
        "day": brief.day,
        "run_id": state["run_id"],
        "items_dropped": check.items_dropped,
        "refs_dropped": check.refs_dropped,
        "plan_rejected": check.plan_rejected,
        "memories_over_cap": check.memories_over_cap,
        "kept": check.kept,
        "dropped": check.dropped,
        "new": check.new,
        "plan": check.plan,
        "day_text": check.day_text,
        "day_refs": check.day_refs,
        "material": material,
        "read": stored_read(state),
        "now_shown_id": material.get("now_shown_id"),
        "turn_correlation_id": step.get("turn_correlation_id"),
        "created_at": now,
        "memories_unsupported": int((grounding or {}).get("unsupported") or 0),
        "memories_unjudged": int((grounding or {}).get("unjudged") or 0),
        "grounding": grounding,
    }


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    """Small, bounded facts for the terminal state event: counts and ids, never the memories' words."""
    brief = state.get("brief") or {}
    step = (state.get("steps") or {}).get(DREAM_DAY_STEP_KEY) or {}
    read = state.get("read") or {}
    material = state.get("material") or {}
    return {
        "line": "dream_day",
        "day": brief.get("day"),
        "operator_forced": bool(brief.get("operator_forced")),
        "persist_outcome": state.get("persist_outcome"),
        "existing_run_id": state.get("existing_run_id"),
        "produced": bool(state.get("produced")),
        "note": state.get("produced_note"),
        **dict(state.get("summary") or {}),
        "read_unread": sorted((read.get("unread") or {}).keys()),
        "digs_total": read.get("digs_total"),
        "digs_background": read.get("digs_background"),
        "now_shown_id": material.get("now_shown_id"),
        "cites": state.get("cites"),
        "turn_correlation_id": step.get("turn_correlation_id"),
        "attempts": step.get("attempts"),
        "harness_step_count": (step.get("debug") or {}).get("harness_step_count"),
    }


def build_dream_day_graph(deps: DreamDayDeps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)

    def backoff(attempts: int) -> str:
        delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempts - 1))
        return (admission.now() + timedelta(seconds=delay)).isoformat()

    async def gather(state: DreamDayState) -> dict:
        if state.get("material"):
            return {"status": "running"}    # once per run: the now's shown row is this run's one read
        brief = DreamDayBriefV1.model_validate(state["brief"])
        material = await deps.gather(brief, state["run_id"])
        return {"material": material.model_dump(mode="json"), "status": "running"}

    async def read_totals_for(read: DreamDayReadV1) -> Optional[dict[str, Any]]:
        """THIS read's count lines: the totals file `dig dream-day` left beside the receipt, only when it carries the
        receipt's own read_at (an older or a later read's copy is not this attempt's). Never fails the attempt."""
        if deps.read_totals is None:
            return None
        try:
            totals = await deps.read_totals(read.day)
        except Exception:  # noqa: BLE001 -- no totals: the dream_day_read ref simply does not resolve
            logger.warning("dream_day_totals_read_failed day=%s", read.day, exc_info=True)
            return None
        if totals is None or totals.day != read.day or totals.read_at != read.read_at:
            return None
        return totals.model_dump(mode="json")

    async def ground(held: dict[str, Any], parsed: ParsedAnswer, material: DreamDayMaterialV1,
                     attempt_end: datetime, totals: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """The grounding gate, under the hold: every checked memory's verdict. A resolve failure leaves every
        memory unjudged (never written); it never fails the attempt (the turn's answer is kept)."""
        deadline = attempt_end - timedelta(seconds=GROUNDING_MARGIN_SEC)

        def time_left() -> float:
            return (deadline - admission.now()).total_seconds()

        try:
            resolve = deps.resolve_ref_texts or deps.resolve_refs
            # Bounded: a hung read here must not outlast the attempt (that would re-run the turn).
            resolved = await asyncio.wait_for(resolve(answer_refs(parsed.answer)), timeout=max(1.0, time_left()))
            resolved.update(totals_resolved(material.day, totals, full=True))
        except Exception as exc:  # noqa: BLE001 -- includes the timeout: every memory unjudged
            logger.warning("dream_day_grounding_resolve_failed run=%s", held.get("run_id"), exc_info=True)
            return [{"statement": " ".join(m.statement.split()), "purpose": m.purpose, "refs": list(m.refs),
                     "status": "unjudged", "unsupported_claims": [], "raw": None,
                     "error": f"resolve_failed: {type(exc).__name__}"} for m in parsed.answer.memories]
        check = check_answer(parsed, resolved, [c.concern_id for c in material.concerns])

        async def ask(prompt: str) -> dict[str, Any]:
            timeout = max(1.0, min(GROUNDING_CALL_TIMEOUT_SEC, time_left()))
            return await asyncio.wait_for(deps.judge(prompt, held, timeout), timeout=timeout + 5.0)

        verdicts = await judge_memories(check.memories, resolved, ask, time_left=time_left)
        logger.info("dream_day_grounding run=%s %s", held.get("run_id"),
                    {k: v for k, v in grounding_summary(verdicts, enabled=True).items() if k != "verdicts"})
        return verdicts

    async def dream(state: DreamDayState) -> dict:
        state = dict(state)
        if DREAM_DAY_STEP_KEY in (state.get("steps") or {}) and state.get("answer") is not None:
            return {"status": "running"}    # resumed after the answer's checkpoint: never re-run the turn
        brief = DreamDayBriefV1.model_validate(state["brief"])
        material = DreamDayMaterialV1.model_validate(state["material"])
        window = attempt_timeout_sec(state, brief.timeout_sec, admission.now())
        # A quarter of the window at most: a late retry with a short window still gets most of it for the turn.
        reserve = min(GROUNDING_RESERVE_SEC, 0.25 * window) if deps.judge is not None else 0.0
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC - reserve)
        correlation_id = harness_step_correlation_id(state, DREAM_DAY_STEP_KEY)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": DREAM_DAY_STEP_KEY}
        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            attempt_started = admission.now()
            record = await run_harness_step(
                deps.run_turn, held, step_key=DREAM_DAY_STEP_KEY,
                prompt=build_prompt(brief, material, attempt=int(state.get("attempt") or 0)),
                correlation_id=correlation_id, timeout_sec=turn_timeout, source_tag=DREAM_DAY_SOURCE_TAG,
                session_id=DREAM_DAY_SESSION_ID, retrieval_query=f"my day {brief.day.isoformat()}",
                require_lookup=False)
            # Inside the attempt, under the hold: a turn that did not read the day, or gave no answer, is a
            # failed attempt, never a checkpointed one.
            read = check_receipt(await deps.read_receipt(brief.day), attempt_started)
            parsed = check_reply(record["text"], record)
            totals = await read_totals_for(read)
            grounding = None
            if deps.judge is not None:
                grounding = await ground(held, parsed, material, attempt_started + timedelta(seconds=window), totals)
            return {"record": record, "answer": parsed.answer.model_dump(mode="json"), "malformed": parsed.malformed,
                    "read": read.model_dump(mode="json"), "grounding": grounding, "totals": totals}

        try:
            result = await admission.execute({**state, "step_correlation_id": correlation_id}, operation)
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
        except Exception as exc:  # noqa: BLE001 -- refusal, timeout, no read, no answer, transport
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "harness_turn_meta": turn_meta}
            return await failed_attempt(admission, state, exc, waiting={"status": "retrying", "retry_node": None},
                                        extra={"harness_turn_meta": turn_meta},
                                        attempt_sec=retry_attempt_sec(brief.timeout_sec))
        record = result["record"]
        record["attempts"] = int(state.get("attempt") or 0) + 1
        update = {"steps": {**(state.get("steps") or {}), DREAM_DAY_STEP_KEY: record}, "answer": result["answer"],
                  "answer_malformed": int(result["malformed"]), "read": result["read"],
                  "read_totals": result.get("totals"),
                  "grounding": result.get("grounding"), "grounding_enabled": deps.judge is not None,
                  "harness_turn_meta": turn_meta, "status": "running", "attempt": 0, "retry_at": None,
                  "retry_node": None, "last_error": None}
        try:   # the answer is checkpointed with this update: persist needs no GPU
            update.update(await admission.release({**state, **update}, "completed"))
        except Exception:  # noqa: BLE001 -- finish releases again; the pool's TTL is the backstop
            logger.warning("dream_day_hold_release_failed run=%s", state.get("run_id"), exc_info=True)
        return update

    async def persist(state: DreamDayState) -> dict:
        state = dict(state)
        now = admission.now()
        attempts = dict(state.get("tail_attempts") or {})
        brief = DreamDayBriefV1.model_validate(state["brief"])
        material = DreamDayMaterialV1.model_validate(state["material"])
        update: dict[str, Any] = {}
        if not state.get("persisted"):
            try:
                answer = DreamDayAnswerV1.model_validate(state["answer"])
                resolved = await deps.resolve_refs(answer_refs(answer))
                resolved.update(totals_resolved(brief.day, state.get("read_totals"), full=False))
                check = check_answer(ParsedAnswer(answer, int(state.get("answer_malformed") or 0)), resolved,
                                     [c.concern_id for c in material.concerns])
                checked = len(check.memories)
                gate_on = bool(state.get("grounding_enabled")) or deps.judge is not None
                verdicts = None
                if gate_on:
                    # Only a memory judged supported is written; a run checkpointed with no verdicts writes none.
                    check, verdicts = apply_grounding(check, state.get("grounding"))
                grounding = grounding_summary(verdicts, enabled=gate_on)
                result = build_memories(check, resolved, brief.day, known_referents=await deps.known_referents())
                stored = await deps.persist(row_fields(state, brief, check, now, grounding), result, brief.llm_route)
            except RunControlPending:
                raise
            except Exception as exc:  # noqa: BLE001 -- no row yet: back off and retry, then fail
                attempts["persist"] = attempts.get("persist", 0) + 1
                error = f"{type(exc).__name__}: {exc}"[:500]
                if attempts["persist"] >= admission.max_attempts:
                    return {"status": "failed", "tail_attempts": attempts, "last_error": error}
                return {"status": "retrying", "tail_attempts": attempts, "last_error": error,
                        "retry_node": "persist", "retry_at": backoff(attempts["persist"])}
            ours = stored["outcome"] == "written"
            memory_refs = [ev.source_id for m in result.memories for ev in m.evidence]
            update = {
                "persisted": True, "persist_outcome": stored["outcome"],
                "existing_run_id": None if ours else stored.get("run_id"),
                # A run that lost the day to an earlier run wrote nothing: its own act grades 0.
                "produced": bool(stored["produced"]) if ours else False,
                "produced_note": stored["note"] if ours else f"already_written by {stored.get('run_id')}",
                "summary": {"memories_written": stored["memories_written"], "memory_ids": stored["memory_ids"],
                            "memories_proposed": len(answer.memories),
                            "memories_rejected": len(result.rejections) + check.memories_over_cap,
                            "memories_checked": checked,
                            "memories_unsupported": grounding["unsupported"],
                            "memories_unjudged": grounding["unjudged"],
                            "grounding_enabled": gate_on,
                            "items_dropped": check.items_dropped, "refs_dropped": check.refs_dropped,
                            "malformed": check.malformed, "plan_rejected": check.plan_rejected,
                            "kept": check.kept, "dropped": check.dropped, "new": check.new,
                            "day_text_chars": len(check.day_text or ""), "drop_reasons": sorted(set(check.reasons))},
                "pending_cites": experience_cites(check, memory_refs) if ours else [],
            }
            state.update(update)
        cites = list(state.get("pending_cites") or [])
        if cites and material.now_shown_id:
            try:
                res = await deps.record_cites(material.now_shown_id, cites, f"dream_day:{brief.day.isoformat()}")
                update["cites"] = {"sent": len(cites), "used": len(res.get("uses") or []),
                                   "not_shown": len(res.get("rejected_not_shown") or []),
                                   "over_cap": len(res.get("rejected_over_cap") or [])}
            except Exception as exc:  # noqa: BLE001 -- the row exists: cites never fail the run
                attempts["cites"] = attempts.get("cites", 0) + 1
                if attempts["cites"] < CITES_MAX_ATTEMPTS:
                    return {**update, "status": "retrying", "tail_attempts": attempts, "retry_node": "persist",
                            "retry_at": backoff(attempts["cites"]), "last_error": f"cites: {exc}"[:300]}
                update["cites"] = {"sent": len(cites), "error": f"{type(exc).__name__}: {exc}"[:200]}
        else:
            update["cites"] = {"sent": 0, "reason": "no_shown_now" if cites else "no_experience_cite"}
        return {**update, "pending_cites": [], "status": "running", "tail_attempts": attempts, "retry_node": None,
                "last_error": None}

    async def finish(state: DreamDayState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: DreamDayState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_dream(state) -> str:
        return {"waiting_resource": "resource_request", "retrying": "retry_wait",
                "failed": "failed"}.get(state.get("status"), "persist")

    def after_grant(state) -> str:
        decision = after_wait(state)
        if decision != "granted":
            return decision
        return "persist" if state.get("answer") is not None else "dream"

    g = StateGraph(DreamDayState)
    for name, node in {"gather": gather, "resource_request": resource_request, "resource_wait": resource_wait,
                       "dream": dream, "persist": persist, "retry_wait": retry_wait, "finish": finish,
                       "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "gather")
    g.add_edge("gather", "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_grant,
                            {"dream": "dream", "persist": "persist", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("dream", after_dream)
    g.add_conditional_edges("persist", lambda s: {"retrying": "retry_wait", "failed": "failed"}.get(s.get("status"), "finish"))
    g.add_conditional_edges("retry_wait", lambda s: "persist" if s.get("retry_node") == "persist" else "resource_request",
                            {"persist": "persist", "resource_request": "resource_request"})
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
